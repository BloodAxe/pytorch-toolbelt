from torch import nn

__all__ = ["Ensembler"]


class Ensembler(nn.Module):
    def __init__(self, models, average=True):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.average = average

    def forward(self, input):

        output_0 = self.models[0](input)
        num_models = len(self.models)

        for index in range(1, num_models):
            output_i = self.models[index](input)

            # Sum outputs
            for key in output_0.keys():
                output_0[key] += output_i[key]

        if self.average:
            for key in output_0.keys():
                output_0[key] /= num_models

        return output_0
