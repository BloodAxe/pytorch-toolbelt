import os
import subprocess
import warnings
from pathlib import Path
from typing import Union, Optional, Tuple
import hashlib

import numpy as np
import pandas as pd
import torch
import zipfile

from sklearn.model_selection import GroupKFold

from pytorch_toolbelt.utils import fs


__all__ = ["InriaAerialImageDataset"]


class InriaAerialImageDataset:
    """
    python -m pytorch_toolbelt.datasets.providers.inria_aerial inria_dataset
    """

    TASK = "binary_segmentation"
    METRIC = ""
    ORIGIN = "https://project.inria.fr/aerialimagelabeling"
    TRAIN_LOCATIONS = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
    TEST_LOCATIONS = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]

    urls = {
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001": "17a7d95c78e484328fd8fe5d5afa2b505e04b8db8fceb617819f3c935d1f39ec",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002": "b505cb223964b157823e88fbd5b0bd041afcbf39427af3ca1ce981ff9f61aff4",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003": "752916faa67be6fc6693f8559531598fa2798dc01b7d197263e911718038252e",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004": "b3893e78f92572455fc2c811af560a558d2a57f9b92eff62fa41399b607a6f44",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005": "a92eb20fdc9911c5ffe3afc514490b8f1e1e5b22301a6fc55d3b4e1624d8033f",
    }

    @classmethod
    def download_and_extract(cls, data_dir: Union[str, Path]) -> bool:
        try:
            from py7zr import py7zr
        except ImportError:
            print("You need to install py7zr to extract 7z-archive: `pip install py7zr`.")
            return False

        filenames = []
        for file_url, file_hash in cls.urls.items():
            file_path = os.path.join(data_dir, os.path.basename(file_url))
            if not os.path.isfile(file_path) or cls.sha256digest(file_path) != file_hash:
                os.makedirs(data_dir, exist_ok=True)
                torch.hub.download_url_to_file(file_url, file_path)

            filenames.append(file_path)

        main_archive = os.path.join(data_dir, "aerialimagelabeling.7z")
        with open(main_archive, "ab") as outfile:  # append in binary mode
            for fname in filenames:
                with open(fname, "rb") as infile:  # open in binary mode also
                    outfile.write(infile.read())

        with py7zr.SevenZipFile(main_archive, "r") as archive:
            archive.extractall(data_dir)
        os.unlink(main_archive)

        zip_archive = os.path.join(data_dir, "NEW2-AerialImageDataset.zip")
        with zipfile.ZipFile(zip_archive, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.unlink(zip_archive)
        return True

    @classmethod
    def init_from_folder(cls, data_dir: Union[str, Path], download: bool = False):
        data_dir = os.path.expanduser(data_dir)

        if download:
            if not cls.download_and_extract(data_dir):
                raise RuntimeError("Download and extract failed")

        return cls(os.path.join(data_dir, "AerialImageDataset"))

    @classmethod
    def sha256digest(cls, filename: str) -> str:
        blocksize = 4096
        sha = hashlib.sha256()
        with open(filename, "rb") as f:
            file_buffer = f.read(blocksize)
            while len(file_buffer) > 0:
                sha.update(file_buffer)
                file_buffer = f.read(blocksize)
        readable_hash = sha.hexdigest()
        return readable_hash

    @classmethod
    def read_tiff(
        cls, image_fname: str, crop_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ) -> np.ndarray:
        import rasterio
        from rasterio.windows import Window

        window = None
        if crop_coords is not None:
            (row_start, row_stop), (col_start, col_stop) = crop_coords
            window = Window.from_slices((row_start, row_stop), (col_start, col_stop))

        if not os.path.isfile(image_fname):
            raise FileNotFoundError(image_fname)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

            with rasterio.open(image_fname) as f:
                image = f.read(window=window)
                image = np.moveaxis(image, 0, -1)  # CHW->HWC
                if image.shape[2] == 1:
                    image = np.squeeze(image, axis=2)
                return image

    @classmethod
    def compress_prediction_mask(cls, predicted_mask_fname, compressed_mask_fname):
        command = (
            "gdal_translate --config GDAL_PAM_ENABLED NO -co COMPRESS=CCITTFAX4 -co NBITS=1 "
            + predicted_mask_fname
            + " "
            + compressed_mask_fname
        )
        subprocess.call(command, shell=True)

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, "train")
        self.test_dir = os.path.join(root_dir, "test")

        if not os.path.isdir(self.train_dir):
            raise FileNotFoundError(f"Train directory {self.train_dir} does not exist")
        if not os.path.isdir(self.test_dir):
            raise FileNotFoundError(f"Test directory {self.train_dir} does not exist")

        self.train_images = fs.find_images_in_dir(os.path.join(self.train_dir, "images"))
        self.train_masks = fs.find_images_in_dir(os.path.join(self.train_dir, "gt"))

        if len(self.train_images) != 180 or len(self.train_masks) != 180:
            raise RuntimeError("Number of train images and ground-truth masks must be 180")

    def get_test_df(self) -> pd.DataFrame:
        test_images = fs.find_images_in_dir(os.path.join(self.test_dir, "images"))
        df = pd.DataFrame.from_dict({"images": test_images})
        df["rows"] = 5000
        df["cols"] = 5000
        df["location"] = df["images"].apply(lambda x: fs.id_from_fname(x).rstrip("0123456789"))
        return df

    def get_train_val_split_train_df(self) -> pd.DataFrame:
        # For validation, we remove the first five images of every location
        # (e.g., austin{1-5}.tif, chicago{1-5}.tif) from the training set.
        # That is suggested validation strategy by competition host
        valid_locations = []
        for loc in self.TRAIN_LOCATIONS:
            for i in range(1, 6):
                valid_locations.append(f"{loc}{i}")

        df = pd.DataFrame.from_dict({"images": self.train_images, "masks": self.train_masks})
        df["location_with_index"] = df["images"].apply(lambda x: fs.id_from_fname(x))
        df["location"] = df["location_with_index"].apply(lambda x: x.rstrip("0123456789"))
        df["split"] = df["location_with_index"].apply(lambda l: "valid" if l in valid_locations else "train")
        df["rows"] = 5000
        df["cols"] = 5000
        return df

    def get_kfold_split_train_df(self, num_folds: int = 5) -> pd.DataFrame:
        df = pd.DataFrame.from_dict({"images": self.train_images, "masks": self.train_masks})
        df["location_with_index"] = df["images"].apply(lambda x: fs.id_from_fname(x))
        df["location"] = df["location_with_index"].apply(lambda x: x.rstrip("0123456789"))
        df["rows"] = 5000
        df["cols"] = 5000
        df["fold"] = -1
        kfold = GroupKFold(n_splits=num_folds)
        for fold, (train_index, test_index) in enumerate(kfold.split(df, df, groups=df["location"])):
            df.loc[test_index, "fold"] = fold
        return df


def download_and_extract(data_dir):
    ds = InriaAerialImageDataset.init_from_folder(data_dir, download=True)
    print(ds.get_test_df())
    print(ds.get_train_val_split_train_df())
    print(ds.get_kfold_split_train_df())


if __name__ == "__main__":
    from fire import Fire

    Fire(download_and_extract)
