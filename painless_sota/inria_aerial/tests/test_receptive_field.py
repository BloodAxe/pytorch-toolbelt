import cv2
import numpy as np
import torch

from painless_sota.inria_aerial.models import (
    PercieverIOForSegmentation,
    DecoderConfig,
    EncoderConfig,
    Space2DepthPreprocessorConfig,
    FourierPositionEncodingConfig, EncoderInputQueryConfig, Depth2SpacePostprocessorConfig,
)
from painless_sota.inria_aerial.models.perciever import (
    PerceiverConfig,
)
from pytorch_toolbelt.datasets import OUTPUT_MASK_KEY
from pytorch_toolbelt.modules import ACT_GELU, ACT_NONE
from pytorch_toolbelt.utils import to_numpy, fs, vstack_header


def normalize(activations):
    # transform activations so that all the values be in range [0, 1]
    # activations = activations - np.min(activations[:])
    activations = activations / np.max(activations[:])
    return activations


def visualize_activations(image, activations):
    activations = normalize(activations)

    # replicate the activations to go from 1 channel to 3
    # as we have colorful input image
    # we could use cvtColor with GRAY2BGR flag here, but it is not
    # safe - our values are floats, but cvtColor expects 8-bit or
    # 16-bit inttegers
    activations = np.stack([activations, activations, activations], axis=2)
    masked_image = (image * activations).astype(np.uint8)
    return masked_image


def test_receptive_field():
    averaged_acts = []

    config = PerceiverConfig(
        preprocessor=Space2DepthPreprocessorConfig(
            spatial_shape=(256, 256),
            num_input_channels=3,
            factor=4,
            num_output_channels=64,
            with_bn=True,
            activation=ACT_NONE,
            kernel_size=3,
        ),
        position_encoding=FourierPositionEncodingConfig(
            num_output_channels=None,
            num_frequency_bands=64,
            include_positions=True,
        ),
        encoder=EncoderConfig(
            num_latents=2048,
            num_latent_channels=512,
            num_cross_attention_heads=1,
            num_cross_attention_layers=1,
            num_self_attention_heads=16,
            num_self_attention_layers_per_block=16,
            num_self_attention_blocks=1,
            dropout=0.0,
            init_scale=0.05,
            attention_residual=True,
            activation=ACT_NONE,
            activation_checkpointing=False,
        ),
        decoder=DecoderConfig(
            num_cross_attention_heads=1,
            init_scale=0.05,
            dropout=0.0,
            attention_residual=False,
            activation=ACT_NONE,
            activation_checkpointing=False,
        ),
        output_query=EncoderInputQueryConfig(),
        postprocessor=Depth2SpacePostprocessorConfig(
            num_output_channels=1,
            factor=4,
            activation=ACT_NONE
        ),
    )

    for _ in range(1):
        net = PercieverIOForSegmentation(config).cuda().eval()

        input = torch.ones(((1, 3, 256, 256)), requires_grad=True).cuda()

        with torch.cuda.amp.autocast(True):
            outputs = net(input)

            if torch.is_tensor(outputs):
                mask = outputs
            else:
                mask = outputs[OUTPUT_MASK_KEY]

            grad = torch.zeros_like(mask)
            grad[:, :, 128, 128] = 1
            mask.backward(gradient=grad, inputs=[input])

        normalized_acts = normalize(to_numpy(input.grad.abs().sum(dim=1)[0]))
        averaged_acts.append(normalized_acts)

    averaged_acts = (np.mean(averaged_acts, axis=0) * 255).astype(np.uint8)
    averaged_acts = cv2.cvtColor(averaged_acts, cv2.COLOR_GRAY2BGR)

    image = averaged_acts.copy()

    image[128, 128, 1] = 255

    config_filename = "perciever_io"
    activation = "relu"
    img = cv2.addWeighted(image, 0.5, averaged_acts, 0.5, 0)
    img = vstack_header(img, f"{fs.id_from_fname(config_filename)}_{activation}")
    cv2.imwrite(filename=f"{fs.id_from_fname(config_filename)}_{activation}.png", img=img)
