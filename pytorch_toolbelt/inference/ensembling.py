import collections

import torch
from torch import nn, Tensor
from typing import List, Union, Iterable, Optional, Dict, Tuple

__all__ = ["ApplySoftmaxTo", "ApplySigmoidTo", "Ensembler", "PickModelOutput", "SelectByIndex", "average_checkpoints"]

from pytorch_toolbelt.inference.tta import _deaugment_averaging


class ApplySoftmaxTo(nn.Module):
    output_keys: Tuple
    temperature: float
    dim: int

    def __init__(
        self,
        model: nn.Module,
        output_key: Union[str, int, Iterable[str]] = "logits",
        dim: int = 1,
        temperature: float = 1,
    ):
        """
        Apply softmax activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string, index or list of strings, indicating to what outputs softmax activation should be applied.
        :param dim: Tensor dimension for softmax activation
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        # By converting to set, we prevent double-activation by passing output_key=["logits", "logits"]
        output_key = (output_key,) if isinstance(output_key, (str, int)) else tuple(set(output_key))
        self.output_keys = output_key
        self.model = model
        self.dim = dim
        self.temperature = temperature

    def forward(self, *input, **kwargs):
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).softmax(dim=self.dim)
        return output


class ApplySigmoidTo(nn.Module):
    output_keys: Tuple
    temperature: float

    def __init__(self, model: nn.Module, output_key: Union[str, int, Iterable[str]] = "logits", temperature=1):
        """
        Apply sigmoid activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string index, or list of strings, indicating to what outputs sigmoid activation should be applied.
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        # By converting to set, we prevent double-activation by passing output_key=["logits", "logits"]
        output_key = (output_key,) if isinstance(output_key, (str, int)) else tuple(set(output_key))
        self.output_keys = output_key
        self.model = model
        self.temperature = temperature

    def forward(self, *input, **kwargs):  # skipcq: PYL-W0221
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).sigmoid_()
        return output


class Ensembler(nn.Module):
    __slots__ = ["outputs", "reduction", "return_some_outputs"]

    """
    Compute sum (or average) of outputs of several models.
    """

    def __init__(self, models: List[nn.Module], reduction: str = "mean", outputs: Optional[Iterable[str]] = None):
        """

        :param models:
        :param reduction: Reduction key ('mean', 'sum', 'gmean', 'hmean' or None)
        :param outputs: Name of model outputs to average and return from Ensembler.
            If None, all outputs from the first model will be used.
        """
        super().__init__()
        self.return_some_outputs = outputs is not None
        self.outputs = tuple(outputs) if outputs else tuple()
        self.models = nn.ModuleList(models)
        self.reduction = reduction

    def forward(self, *input, **kwargs):  # skipcq: PYL-W0221
        outputs = [model(*input, **kwargs) for model in self.models]
        output_is_dict = isinstance(outputs[0], dict)
        output_is_list = isinstance(outputs[0], (list, tuple))  # noqa

        if self.return_some_outputs:
            keys = self.outputs
        elif output_is_dict:
            keys = outputs[0].keys()
        elif output_is_list:
            keys = list(range(len(outputs[0])))
        elif torch.is_tensor(outputs[0]):
            keys = None
        else:
            raise RuntimeError()

        if keys is None:
            predictions = torch.stack(outputs)
            predictions = _deaugment_averaging(predictions, self.reduction)
            averaged_output = predictions
        else:
            averaged_output = {} if output_is_dict else []
            for key in keys:
                predictions = [output[key] for output in outputs]
                predictions = torch.stack(predictions)
                predictions = _deaugment_averaging(predictions, self.reduction)
                if output_is_dict:
                    averaged_output[key] = predictions
                else:
                    averaged_output.append(predictions)

        return averaged_output


class PickModelOutput(nn.Module):
    """
    Wraps a model that returns dict or list and returns only a specific element.

    Usage example:
        >>> model = MyAwesomeSegmentationModel() # Returns dict {"OUTPUT_MASK": Tensor, ...}
        >>> net  = nn.Sequential(PickModelOutput(model, "OUTPUT_MASK")), nn.Sigmoid())
    """

    __slots__ = ["target_key"]

    def __init__(self, model: nn.Module, key: Union[str, int]):
        super().__init__()
        self.model = model
        self.target_key = key

    def forward(self, *input, **kwargs) -> Tensor:
        output = self.model(*input, **kwargs)
        return output[self.target_key]


class SelectByIndex(nn.Module):
    """
    Select a single Tensor from the dict or list of output tensors.

    Usage example:
        >>> model = MyAwesomeSegmentationModel() # Returns dict {"OUTPUT_MASK": Tensor, ...}
        >>> net  = nn.Sequential(model, SelectByIndex("OUTPUT_MASK"), nn.Sigmoid())
    """

    __slots__ = ["target_key"]

    def __init__(self, key: Union[str, int]):
        super().__init__()
        self.target_key = key

    def forward(self, outputs: Dict[str, Tensor]) -> Tensor:
        return outputs[self.target_key]


def average_checkpoints(inputs: List[str]) -> collections.OrderedDict:
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16
    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location="cpu",
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model_state_dict"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model_state_dict"] = averaged_params
    return new_state
