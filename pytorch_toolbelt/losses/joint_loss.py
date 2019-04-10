from torch.nn.modules.loss import _Loss


class JointLoss(_Loss):
    def __init__(self, first, second, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = first
        self.second = second
        self.first_weight = first_weight
        self.second_weight = second_weight

    def forward(self, *input):
        return self.first(*input) * self.first_weight + self.second(*input) * self.second_weight
