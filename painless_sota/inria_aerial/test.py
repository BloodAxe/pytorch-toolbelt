import torch

from pytorch_toolbelt.modules import GenericTimmEncoder
from pytorch_toolbelt.utils import count_parameters, describe_outputs

net = GenericTimmEncoder("convnext_large_in22k", pretrained=True).cuda().eval()

print(count_parameters(net))
print(net.strides)
print(net.channels)

input = torch.randn((1, 3, 512, 512)).cuda()
outputs = net(input)

print(describe_outputs(outputs))

import albumentations as A

A.ElasticTransform