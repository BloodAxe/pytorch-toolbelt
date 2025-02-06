import collections

import torch
from torch import nn, Tensor
from typing import List, Union, Iterable, Optional, Dict, Tuple

__all__ = [
    "ApplySoftmaxTo",
    "ApplySigmoidTo",
    "Ensembler",
    "PickModelOutput",
    "SelectByIndex",
    "average_checkpoints",
    "average_state_dicts",
]

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


def average_state_dicts(state_dicts: List[Mapping[str, Tensor]]) -> Mapping[str, Tensor]:
    """
    Averages multiple 'state_dict'

    """

    keys = state_dicts[0].keys()
    final_state_dict = collections.OrderedDict()

    for key in keys:
        # Collect the values (tensors) for this key from all checkpoints
        values = [sd[key] for sd in state_dicts]

        # Check the dtype of the first value (assuming all dtypes match)
        first_val = values[0]

        if not all(v.shape == first_val.shape for v in values):
            raise ValueError(f"Tensor shapes for key '{key}' are not consistent across checkpoints.")

        if first_val.dtype == torch.bool:
            # For bool, ensure all are identical
            for val in values[1:]:
                if not torch.equal(val, first_val):
                    raise ValueError(f"Boolean values for key '{key}' differ between checkpoints.")
            final_state_dict[key] = first_val  # Use the first if all identical

        elif torch.is_floating_point(first_val):
            # Average float values
            stacked = torch.stack(values, dim=0)
            target_dtype = stacked.dtype
            accum_dtype = torch.promote_types(target_dtype, torch.float32)  # Upcast to float32 if needed
            averaged = stacked.to(accum_dtype).mean(dim=0).to(target_dtype)
            final_state_dict[key] = averaged

        elif first_val.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }:
            # Average integer values (using integer division)
            stacked = torch.stack(values, dim=0)
            summed = stacked.sum(dim=0, dtype=torch.int64)
            averaged = summed // len(values)
            final_state_dict[key] = averaged.to(first_val.dtype)

        else:
            # If you have other special dtypes to handle, add logic here
            # or simply copy the first value if that is your intended behavior.
            raise TypeError(f"Unsupported dtype '{first_val.dtype}' encountered for key '{key}'.")

    return final_state_dict


def average_checkpoints(inputs: List[str], key=None, map_location="cpu", weights_only=True) -> collections.OrderedDict:
    """Loads checkpoints from inputs and returns a model with averaged weights.
    
    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
      key (str): An optional key to select a sub-dictionary from the checkpoint.
      map_location (str): A string describing how to remap storage locations when loading the model.
      weights_only (bool): If True, will only load the weights of the model.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    state_dicts = [torch.load(path, map_location="cpu", weights_only=weights_only) for path in inputs]
    if key is not None:
        state_dicts = [sd[key] for sd in state_dicts]

    avg_state_dict = average_state_dicts(state_dicts)
    if key is not None:
        avg_state_dict = {key: avg_state_dict}

    return avg_state_dict
