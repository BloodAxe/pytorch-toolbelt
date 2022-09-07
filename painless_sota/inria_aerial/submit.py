import argparse
import os
import subprocess
from typing import List, Dict, Union, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader

from painless_sota.inria_aerial.data.image_io import read_tiff
from pytorch_toolbelt.datasets import OUTPUT_MASK_KEY
from pytorch_toolbelt.inference import tta, PickModelOutput, ImageSlicer, TileMerger
from pytorch_toolbelt.utils import to_numpy, image_to_tensor, transfer_weights
from pytorch_toolbelt.utils.catalyst import report_checkpoint
from torch import nn

from tqdm import tqdm
from pytorch_toolbelt.utils.fs import auto_file, find_in_dir


class InMemoryDataset(Dataset):
    def __init__(self, data: List[Dict], transform: A.Compose):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.transform(**self.data[item])
        data["image"] = image_to_tensor(data["image"]).float()
        return data


@torch.no_grad()
def predict(model: nn.Module, image: np.ndarray, image_size, normalize=A.Normalize(), batch_size=1) -> np.ndarray:

    tile_step = (3 * image_size[0] // 4, 3 * image_size[1] // 4)

    tile_slicer = ImageSlicer(image.shape, image_size, tile_step, weight="pyramid")
    tile_merger = TileMerger(tile_slicer.target_shape, 1, tile_slicer.weight, device="cuda")
    patches = tile_slicer.split(image)

    transform = A.Compose([normalize])

    data = list(
        {"image": patch, "coords": np.array(coords, dtype=int)} for (patch, coords) in zip(patches, tile_slicer.crops)
    )

    for batch in DataLoader(
        InMemoryDataset(data, transform),
        pin_memory=False,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    ):
        image = batch["image"].cuda()
        coords = batch["coords"]
        with torch.cuda.amp.autocast(True):
            output = model(image)
        tile_merger.integrate_batch(output, coords)

    mask = tile_merger.merge()

    mask = np.moveaxis(to_numpy(mask), 0, -1)
    mask = tile_slicer.crop_to_orignal_size(mask)

    return mask


def model_from_checkpoint(checkpoint_config: Union[str, Dict], strict) -> Tuple[nn.Module, Dict]:
    checkpoint_name = checkpoint_config

    checkpoint = torch.load(checkpoint_name, map_location="cpu")
    model_config = checkpoint["checkpoint_data"]["config"]["model"]

    model_state_dict = checkpoint["model_state_dict"]

    model = instantiate(model_config["architecture"], _recursive_=False)
    try:
        model.load_state_dict(model_state_dict, strict=strict)
    except RuntimeError as e:
        if not strict:
            print("BIG WARNING TO BRING YOUR ATTENTION")
            print("Failed to transfer weights using load_state_dict, re-trying with transfer_weights")
            print("This may end up with part of the model weights uninitialized.")
            print("It's highly recommended to re-train the model")
            print("BIG WARNING TO BRING YOUR ATTENTION")

            transfer_weights(model, model_state_dict)
        else:
            raise e
    return model.eval(), checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=None, required=True, help="Data dir")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        required=True,
        help="Checkpoint filename to use as initial model weights",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("-tta", "--tta", default=None, type=str, help="Type of TTA to use [fliplr, d4]")
    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_file = auto_file(args.checkpoint)
    run_dir = os.path.dirname(checkpoint_file)
    out_dir = os.path.join(run_dir, "submit")
    os.makedirs(out_dir, exist_ok=True)

    model, checkpoint = model_from_checkpoint(checkpoint_file, strict=True)
    validation_iou = checkpoint["valid_metrics"]["metric/jaccard"]
    threshold = checkpoint["valid_metrics"].get("metric/jaccard/threshold", 0.5)
    print(report_checkpoint(checkpoint))
    print("Validation IOU", validation_iou)
    print("Using threshold", threshold)

    model = nn.Sequential(PickModelOutput(model, OUTPUT_MASK_KEY), nn.Sigmoid())

    if args.tta == "fliplr":
        model = tta.GeneralizedTTA(model, tta.fliplr_image_augment, tta.fliplr_image_deaugment)
    elif args.tta == "flips":
        model = tta.GeneralizedTTA(model, tta.flips_image_augment, tta.flips_image_deaugment)
    elif args.tta == "d2":
        model = tta.GeneralizedTTA(model, tta.d2_image_augment, tta.d2_image_deaugment)
    elif args.tta == "d4":
        model = tta.GeneralizedTTA(model, tta.d4_image_augment, tta.d4_image_deaugment)
    elif args.tta == "ms-d2":
        model = tta.GeneralizedTTA(model, tta.d2_image_augment, tta.d2_image_deaugment)
        model = tta.MultiscaleTTA(model, size_offsets=[-128, -64, 64, 128])
    elif args.tta == "ms-d4":
        model = tta.GeneralizedTTA(model, tta.d4_image_augment, tta.d4_image_deaugment)
        model = tta.MultiscaleTTA(model, size_offsets=[-128, -64, 64, 128])
    elif args.tta == "ms":
        model = tta.MultiscaleTTA(model, size_offsets=[-128, -64, 64, 128])
    else:
        pass

    model = model.cuda().eval()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    test_predictions_dir = os.path.join(out_dir, "test_predictions")
    if args.tta is not None:
        test_predictions_dir += f"_{args.tta}"

    test_predictions_dir_raw = os.path.join(test_predictions_dir, "raw")
    test_predictions_dir_compressed = os.path.join(test_predictions_dir, "compressed")

    os.makedirs(test_predictions_dir_raw, exist_ok=True)
    os.makedirs(test_predictions_dir_compressed, exist_ok=True)

    test_images = find_in_dir(os.path.join(data_dir, "test", "images"))
    for fname in tqdm(test_images, total=len(test_images)):
        predicted_mask_fname = os.path.join(test_predictions_dir_raw, os.path.basename(fname))

        image = read_tiff(fname)
        mask = predict(model, image, image_size=(1024, 1024), batch_size=args.batch_size * torch.cuda.device_count())
        mask = ((mask > threshold) * 255).astype(np.uint8)
        cv2.imwrite(predicted_mask_fname, mask)

        name_compressed = os.path.join(test_predictions_dir_compressed, os.path.basename(fname))
        command = (
            "gdal_translate --config GDAL_PAM_ENABLED NO -co COMPRESS=CCITTFAX4 -co NBITS=1 "
            + predicted_mask_fname
            + " "
            + name_compressed
        )
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
