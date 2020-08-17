import numpy as np
from catalyst.dl import Callback, CallbackOrder, IRunner

from ..torch_utils import to_numpy

__all__ = ["MulticlassOnlinePseudolabelingCallback", "BCEOnlinePseudolabelingCallback", "PseudolabelDatasetMixin"]


class PseudolabelDatasetMixin:
    def set_target(self, index: int, value):
        raise NotImplementedError


class MulticlassOnlinePseudolabelingCallback(Callback):
    """
    Online pseudo-labeling callback for multi-class problem.

    >>> unlabeled_train = get_test_dataset(
    >>>     data_dir, image_size=image_size, augmentation=augmentations
    >>> )
    >>> unlabeled_eval = get_test_dataset(
    >>>     data_dir, image_size=image_size
    >>> )
    >>>
    >>> callbacks += [
    >>>     MulticlassOnlinePseudolabelingCallback(
    >>>         unlabeled_train.targets,
    >>>         pseudolabel_loader="label",
    >>>         prob_threshold=0.9)
    >>> ]
    >>> train_ds = train_ds + unlabeled_train
    >>>
    >>> loaders = collections.OrderedDict()
    >>> loaders["train"] = DataLoader(train_ds)
    >>> loaders["valid"] = DataLoader(valid_ds)
    >>> loaders["label"] = DataLoader(unlabeled_eval, shuffle=False) # ! shuffle=False is important !
    """

    def __init__(
        self,
        unlabeled_ds: PseudolabelDatasetMixin,
        pseudolabel_loader="label",
        prob_threshold=0.9,
        prob_ratio=None,
        output_key="logits",
        unlabeled_class=-100,
    ):
        super().__init__(CallbackOrder.External)
        self.unlabeled_ds = unlabeled_ds
        self.pseudolabel_loader = pseudolabel_loader
        self.prob_threshold = prob_threshold
        self.prob_ratio = prob_ratio
        self.predictions = []
        self.output_key = output_key
        self.unlabeled_class = unlabeled_class

    def on_epoch_start(self, runner: IRunner):
        pass

    def on_loader_start(self, runner: IRunner):
        if runner.loader_name == self.pseudolabel_loader:
            self.predictions = []

    def get_probabilities(self, runner: IRunner):
        probs = runner.output[self.output_key].detach().softmax(dim=1)
        return to_numpy(probs)

    def on_batch_end(self, runner: IRunner):
        if runner.loader_name == self.pseudolabel_loader:
            probs = self.get_probabilities(runner)
            self.predictions.extend(probs)

    def on_loader_end(self, runner: IRunner):
        if runner.loader_name == self.pseudolabel_loader:
            predictions = np.array(self.predictions)
            max_pred = np.argmax(predictions, axis=1)
            max_score = np.amax(predictions, axis=1)
            confident_mask = max_score > self.prob_threshold
            num_samples = len(predictions)

            for index, predicted_target, score in zip(range(num_samples, max_pred, max_score)):
                target = predicted_target if score > self.prob_threshold else self.unlabeled_class
                self.unlabeled_ds.set_target(index, target)

            num_confident_samples = confident_mask.sum()
            runner.loader_metrics["pseudolabeling/confident_samples"] = num_confident_samples
            runner.loader_metrics["pseudolabeling/confident_samples_mean_score"] = max_score[confident_mask].mean()

            runner.loader_metrics["pseudolabeling/unconfident_samples"] = len(predictions) - num_confident_samples
            runner.loader_metrics["pseudolabeling/unconfident_samples_mean_score"] = max_score[~confident_mask].mean()

    def on_epoch_end(self, runner: IRunner):
        pass


class BCEOnlinePseudolabelingCallback(MulticlassOnlinePseudolabelingCallback):
    def get_probabilities(self, runner: IRunner):
        probs = runner.output[self.output_key].detach().sigmoid()
        return to_numpy(probs)
