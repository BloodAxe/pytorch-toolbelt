from transformers import PerceiverForOpticalFlow

from pytorch_toolbelt.utils import count_parameters

model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")


print(model)
print(count_parameters(model))
