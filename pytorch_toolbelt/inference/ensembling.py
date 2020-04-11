from torch import nn
from typing import List, Union

__all__ = ["ApplySigmoidTo", "Ensembler"]


class ApplySigmoidTo(nn.Module):
    def __init__(self, model, output_key: Union[str, List[str]] = "logits"):
        super().__init__()
        # Prevents double-activation by passing output_key=["logits", "logits"]
        output_key = output_key if isinstance(output_key, (list, tuple)) else [output_key]
        self.output_keys = set(output_key)
        self.model = model

    def forward(self, input):
        output = self.model(input)
        for key in self.output_keys:
            output[key] = output[key].sigmoid()
        return output


class Ensembler(nn.Module):
    def __init__(self, models, average=True, outputs=None):
        """

        :param models:
        :param average:
        :param outputs: Name of model outputs to average and return from Ensembler.
            If None, all outputs from the first model will be used.
        """
        super().__init__()
        self.outputs = outputs
        self.models = nn.ModuleList(models)
        self.average = average

    def forward(self, input):

        output_0 = self.models[0](input)
        num_models = len(self.models)

        if self.outputs:
            keys = self.outputs
        else:
            keys = output_0.keys()

        for index in range(1, num_models):
            output_i = self.models[index](input)

            # Sum outputs
            for key in keys:
                output_0[key] += output_i[key]

        if self.average:
            for key in keys:
                output_0[key] /= num_models

        return output_0
