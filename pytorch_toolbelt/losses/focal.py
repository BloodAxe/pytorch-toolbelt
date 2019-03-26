from torch.nn.modules.loss import _Loss

from .functional import sigmoid_focal_loss


class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _label_loss(self, label_input, label_target):
        batch_size, num_anchors, num_classes = label_input.size()
        loss = 0
        label_target = label_target.view(batch_size * num_anchors)
        label_input = label_input.view(batch_size * num_anchors, num_classes)

        # Filter anchors with -1 label from loss computation
        not_ignored = label_target >= 0
        num_pos = (label_target > 0).sum().item() + 1

        for cls in range(num_classes):
            cls_label_target = (label_target == (cls + 0)).long()
            cls_label_input = label_input[..., cls]

            cls_label_target = cls_label_target[not_ignored]
            cls_label_input = cls_label_input[not_ignored]

            loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha)
        return loss / num_pos

    def forward(self, pred_bboxes, pred_labels, true_bboxes, true_labels):
        """

        Ignores anchors having -1 target label
        """

        label_loss = self._label_loss(pred_labels, true_labels)
        return label_loss
