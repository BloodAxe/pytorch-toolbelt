from functools import partial

import torch
from pytorch_toolbelt.inference.functional import pad_tensor, unpad_tensor
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.modules.abn import ACT_SELU
from pytorch_toolbelt.modules.fpn import FPNFuse, FPNBottleneckBlockBN
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn
from torch.nn import functional as F

from pytorch_toolbelt.modules.unet import UnetCentralBlock, UnetDecoderBlock, UnetEncoderBlock


class DoubleConvRelu(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, encoder: E.EncoderModule, decoder: D.DecoderModule, num_classes: int):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.fpn_fuse = FPNFuse()

        # Final Classifier
        output_features = sum(self.decoder.output_filters)

        self.logits = nn.Conv2d(output_features, num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_tensor(x, 32)

        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        features = self.fpn_fuse(dec_features)

        logits = self.logits(features)
        logits = F.interpolate(logits, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        logits = unpad_tensor(logits, pad)

        return logits

    def set_encoder_training_enabled(self, enabled):
        self.encoder.set_trainable(enabled)


class HiResSegmentationModel(nn.Module):
    def __init__(self, encoder: E.EncoderModule, num_classes: int, fpn_features: int):
        super().__init__()

        self.encoder = encoder
        self.layer0 = UnetEncoderBlock(3, 32, stride=1)
        self.layer1 = UnetEncoderBlock(32, 64, stride=2)

        # hard-coded assumption that encoder has first layer with stride of 4
        self.decoder = D.FPNDecoder(features=[32, 64] + encoder.output_filters,
                                    prediction_block=UnetEncoderBlock,
                                    bottleneck=FPNBottleneckBlockBN,
                                    fpn_features=fpn_features)
        self.fpn_fuse = FPNFuse()

        output_features = sum(self.decoder.output_filters)

        self.dropout = nn.Dropout2d(0.5, inplace=True)
        self.logits = nn.Conv2d(output_features, num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_tensor(x, 32)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        enc_features = self.encoder(x)
        dec_features = self.decoder([layer0, layer1] + enc_features)

        features = self.fpn_fuse(dec_features)

        features = self.dropout(features)
        features = self.logits(features)
        features = unpad_tensor(features, pad)
        return features

    def set_encoder_training_enabled(self, enabled):
        self.encoder.set_trainable(enabled)

    # def load_encoder_weights(self, snapshot_file: str):
    #     """ Loads weights of the encoder (only encoder) from checkpoint file"""
    #     checkpoint = torch.load(snapshot_file)
    #     model: OrderedDict = checkpoint['model']
    #     encoder_state = [(str.lstrip(key, 'encoder.'), value) for (key, value) in model.items() if
    #                      str.startswith(key, 'encoder.')]
    #     encoder_state = OrderedDict(encoder_state)
    #     self.encoder.load_state_dict(encoder_state, strict=True)


def fpn128_resnet34(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.Resnet34Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=128)

    return SegmentationModel(encoder, decoder, num_classes)


def fpn128_resnext50(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.SEResNeXt50Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=128)

    return SegmentationModel(encoder, decoder, num_classes)


def fpn256_resnext50(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.SEResNeXt50Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           prediction_block=DoubleConvRelu,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=256)

    return SegmentationModel(encoder, decoder, num_classes)


def hd_fpn_resnext50(num_classes=1, num_channels=3, fpn_features=128):
    assert num_channels == 3
    encoder = E.SEResNeXt50Encoder()
    return HiResSegmentationModel(encoder, num_classes, fpn_features)


def test_hd_fpn_resnext50():
    net = hd_fpn_resnext50(1, 3, 128).eval()
    image = torch.rand((1, 3, 512, 512))
    out = net(image)
    print(count_parameters(net))
    print(out.size())
#
# def fpn_senet154(num_classes=1, num_channels=3):
#     assert num_channels == 3
#     encoder = E.SENet154Encoder()
#     decoder = D.FPNDecoder(features=encoder.output_filters,
#                            bottleneck=FPNBottleneckBlockBN,
#                            prediction_block=UnetEncoderBlock,
#                            fpn_features=256,
#                            prediction_features=[128, 256, 512, 768])
#
#     return SegmentationModel(encoder, decoder, num_classes)
#
#
# def hdfpn_resnext50(num_classes=1, num_channels=3):
#     assert num_channels == 3
#     encoder = E.SEResNeXt50Encoder()
#     decoder = D.FPNDecoder(features=encoder.output_filters,
#                            prediction_block=UnetEncoderBlock,
#                            bottleneck=FPNBottleneckBlockBN,
#                            fpn_features=128)
#
#     return HiResSegmentationModel(encoder, decoder, num_classes)
#
#
# @torch.no_grad()
# def test_fpn_resnext50():
#     from pytorch_toolbelt.utils.torch_utils import count_parameters
#
#     net = fpn_resnext50().eval()
#     img = torch.rand((1, 3, 512, 512))
#     print(count_parameters(net))
#     print(count_parameters(net.encoder))
#     print(count_parameters(net.decoder))
#     print(count_parameters(net.logits))
#     out = net(img)
#     print(out.size())
#
#
# @torch.no_grad()
# def test_hires_fpn_resnext50():
#     from pytorch_toolbelt.utils.torch_utils import count_parameters
#
#     net = hdfpn_resnext50().eval()
#     img = torch.rand((1, 3, 512, 512))
#     print(count_parameters(net))
#     print(count_parameters(net.encoder))
#     print(count_parameters(net.decoder))
#     print(count_parameters(net.logits))
#     out = net(img)
#     print(out.size())
#
#
# @torch.no_grad()
# def test_fpn_senet154():
#     from pytorch_toolbelt.utils.torch_utils import count_parameters
#
#     net = fpn_senet154().eval()
#     img = torch.rand((1, 3, 512, 512))
#     print(count_parameters(net))
#     print(count_parameters(net.encoder))
#     print(count_parameters(net.decoder))
#     print(count_parameters(net.logits))
#     out = net(img)
#     print(out.size())
