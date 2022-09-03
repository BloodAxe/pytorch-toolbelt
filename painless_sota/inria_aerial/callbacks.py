from collections import defaultdict

import numpy as np
import torch
from catalyst.core import Callback, CallbackOrder, IRunner

from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger
from pytorch_toolbelt.utils.distributed import all_gather, is_main_process

__all__ = ["JaccardMetricPerImageWithOptimalThreshold"]


class JaccardMetricPerImageWithOptimalThreshold(Callback):
    """
    Callback that computes an optimal threshold for binarizing logits and theoretical IoU score at given threshold.
    """

    def __init__(
        self,
        targets_key: str,
        predictions_key: str,
        image_id_key: str,
        prefix: str,
        predictions_postprocess_fn=torch.sigmoid,
    ):
        super().__init__(CallbackOrder.Metric)
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.predictions_key = predictions_key
        self.targets_key = targets_key
        self.image_id_key = image_id_key
        self.thresholds = torch.arange(0.3, 0.6, 0.025).detach()
        self.scores_per_image = {}
        self.predictions_postprocess_fn = predictions_postprocess_fn

    def on_loader_start(self, runner: IRunner):
        self.scores_per_image = {}

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        image_id = runner.input[self.image_id_key]
        outputs = runner.output[self.predictions_key].detach()
        targets = runner.input[self.targets_key].detach()
        if self.predictions_postprocess_fn is not None:
            outputs = self.predictions_postprocess_fn(outputs)

        # Flatten images for easy computing IoU
        assert outputs.size(1) == 1
        assert targets.size(1) == 1
        outputs = outputs.view(outputs.size(0), -1, 1) > self.thresholds.to(outputs.dtype).to(outputs.device).view(
            1, 1, -1
        )
        targets = targets.view(targets.size(0), -1) == 1
        n = len(self.thresholds)

        for i, threshold in enumerate(self.thresholds):
            # Binarize outputs
            outputs_i = outputs[..., i]
            intersection = torch.sum(targets & outputs_i, dim=1)
            union = torch.sum(targets | outputs_i, dim=1)

            for img_id, img_intersection, img_union in zip(image_id, intersection, union):
                if img_id not in self.scores_per_image:
                    self.scores_per_image[img_id] = {"intersection": np.zeros(n), "union": np.zeros(n)}

                self.scores_per_image[img_id]["intersection"][i] += float(img_intersection)
                self.scores_per_image[img_id]["union"][i] += float(img_union)

    def on_loader_end(self, runner: IRunner):
        eps = 1e-7
        ious_per_image = []

        # Gather statistics from all nodes
        all_gathered_scores_per_image = all_gather(self.scores_per_image)

        n = len(self.thresholds)
        all_scores_per_image = defaultdict(lambda: {"intersection": np.zeros(n), "union": np.zeros(n)})
        for scores_per_image in all_gathered_scores_per_image:
            for image_id, values in scores_per_image.items():
                all_scores_per_image[image_id]["intersection"] += values["intersection"]
                all_scores_per_image[image_id]["union"] += values["union"]

        for image_id, values in all_scores_per_image.items():
            intersection: np.ndarray = values["intersection"]
            union: np.ndarray = values["union"]
            metric = intersection / (union + eps)
            ious_per_image.append(metric)

        thresholds = to_numpy(self.thresholds)
        iou = np.mean(ious_per_image, axis=0)
        assert len(iou) == len(thresholds)

        threshold_index = np.argmax(iou)
        iou_at_threshold = iou[threshold_index]
        threshold_value = thresholds[threshold_index]

        runner.loader_metrics[self.prefix + "/" + "threshold"] = float(threshold_value)
        runner.loader_metrics[self.prefix] = float(iou_at_threshold)

        if is_main_process():
            logger = get_tensorboard_logger(runner)
            logger.add_histogram(self.prefix, iou, global_step=runner.epoch)
