from typing import Optional

import torch
from torch import nn, Tensor

__all__ = ["BiTemperedLogisticLoss", "BinaryBiTemperedLogisticLoss"]


def log_t(u, t):
    """Compute log_t for `u'."""
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations: Tensor, t: float, num_iters: int) -> Tensor:
    """Return the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * logt_partition.pow(1.0 - t)

    logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = -log_t(1.0 / logt_partition, t) + mu

    return normalization_constants


def compute_normalization_binary_search(activations: Tensor, t: float, num_iters: int) -> Tensor:
    """Compute normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = torch.sum((normalized_activations > -1.0 / (1.0 - t)).to(torch.int32), dim=-1, keepdim=True).to(
        activations.dtype
    )

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(exp_t(normalized_activations - logt_partition, t), dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(lower * update + (1.0 - update) * logt_partition, shape_partition)
        upper = torch.reshape(upper * (1.0 - update) + update * logt_partition, shape_partition)

    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """

    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output

        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    """Compute normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, reduction="mean"):
    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot),
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """
    if len(labels.shape) < len(activations.shape):  # not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = (1 - label_smoothing * num_classes / (num_classes - 1)) * labels_onehot + label_smoothing / (
            num_classes - 1
        )

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = (
        labels_onehot * log_t(labels_onehot + 1e-10, t1)
        - labels_onehot * log_t(probabilities, t1)
        - labels_onehot.pow(2.0 - t1) / (2.0 - t1)
        + probabilities.pow(2.0 - t1) / (2.0 - t1)
    )
    loss_values = loss_values.sum(dim=-1)  # sum over classes

    if reduction == "none":
        return loss_values
    if reduction == "sum":
        return loss_values.sum()
    if reduction == "mean":
        return loss_values.mean()


class BiTemperedLogisticLoss(nn.Module):
    """

    https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
    https://arxiv.org/abs/1906.03361
    """

    def __init__(self, t1: float, t2: float, smoothing=0.0, ignore_index=None, reduction: str = "mean"):
        """

        Args:
            t1:
            t2:
            smoothing:
            ignore_index:
            reduction:
        """
        super(BiTemperedLogisticLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = bi_tempered_logistic_loss(
            predictions, targets, t1=self.t1, t2=self.t2, label_smoothing=self.smoothing, reduction="none"
        )

        if self.ignore_index is not None:
            mask = ~targets.eq(self.ignore_index)
            loss *= mask

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class BinaryBiTemperedLogisticLoss(nn.Module):
    """
    Modification of BiTemperedLogisticLoss for binary classification case.
    It's signature matches nn.BCEWithLogitsLoss: Predictions and target tensors must have shape [B,1,...]

    References:
        https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
        https://arxiv.org/abs/1906.03361
    """

    def __init__(
        self, t1: float, t2: float, smoothing: float = 0.0, ignore_index: Optional[int] = None, reduction: str = "mean"
    ):
        """

        Args:
            t1:
            t2:
            smoothing:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Forward method of the loss function

        Args:
            predictions: [B,1,...]
            targets: [B,1,...]

        Returns:
            Zero-sized tensor with reduced loss if self.reduction is `sum` or `mean`; Otherwise returns loss of the
            shape of `predictions` tensor.
        """
        if predictions.size(1) != 1 or targets.size(1) != 1:
            raise ValueError("Channel dimension for predictions and targets must be equal to 1")

        loss = bi_tempered_logistic_loss(
            torch.cat([-predictions, predictions], dim=1).moveaxis(1, -1),
            torch.cat([1 - targets, targets], dim=1).moveaxis(1, -1),
            t1=self.t1,
            t2=self.t2,
            label_smoothing=self.smoothing,
            reduction="none",
        ).unsqueeze(dim=1)

        if self.ignore_index is not None:
            mask = targets.eq(self.ignore_index)
            loss = torch.masked_fill(loss, mask, 0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
