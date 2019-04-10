from functools import partial

import torch
from pytorch_toolbelt.inference.functional import pad_tensor, unpad_tensor
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.modules.abn import ACT_SELU
from pytorch_toolbelt.modules.fpn import FPNFuse, FPNBottleneckBlockBN
from torch import nn
from torch.nn import functional as F

from pytorch_toolbelt.modules.unet import UnetCentralBlock, UnetDecoderBlock, UnetEncoderBlock


class SegmentationModel(nn.Module):
    def __init__(self, encoder: E.EncoderModule, decoder: D.DecoderModule, num_classes: int):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.logits = nn.Conv2d(self.decoder.output_filters[-1], num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_tensor(x, 32)

        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        features = dec_features[-1]
        logits = self.logits(features)

        logits = F.interpolate(logits, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        logits = unpad_tensor(logits, pad)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        return logits

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


class HiResSegmentationModel(nn.Module):
    def __init__(self, encoder: E.EncoderModule, decoder: D.DecoderModule, num_classes: int):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fpn_fuse = FPNFuse()
        self.bottleneck = nn.Conv2d(sum(self.decoder.output_filters), 64, kernel_size=1)

        self.edge_map1 = UnetEncoderBlock(3, 32, activation=ACT_SELU)
        self.edge_map2 = UnetEncoderBlock(32, 32, activation=ACT_SELU)

        self.smooth2 = UnetDecoderBlock(64, 32, 64, activation=ACT_SELU)
        self.smooth1 = UnetDecoderBlock(64, 32, 64, activation=ACT_SELU)

        self.logits = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_tensor(x, 32)

        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        features = self.fpn_fuse(dec_features)
        features = self.bottleneck(features)

        edge_map1 = self.edge_map1(x)
        edge_map2 = self.edge_map2(F.max_pool2d(edge_map1, kernel_size=3, padding=1, stride=2))

        # Upsample feature map
        features = self.smooth2(features, edge_map2)

        # Upsample feature map
        features = self.smooth1(features, edge_map1)

        logits = self.logits(features)

        logits = unpad_tensor(logits, pad)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        return logits

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


def fpn_resnext50(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.SEResNeXt50Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           strides=encoder.output_strides,
                           fpn_features=128, dropout=0.2)

    return SegmentationModel(encoder, decoder, num_classes)


def hdfpn_resnext50(num_classes=1, num_channels=3):
    assert num_channels == 3
    encoder = E.SEResNeXt50Encoder()
    decoder = D.FPNDecoder(features=encoder.output_filters,
                           bottleneck=FPNBottleneckBlockBN,
                           fpn_features=128, dropout=0.2)

    return HiResSegmentationModel(encoder, decoder, num_classes)


@torch.no_grad()
def test_fpn_resnext50():
    from pytorch_toolbelt.utils.torch_utils import count_parameters

    net = fpn_resnext50().eval()
    img = torch.rand((1, 3, 512, 512))
    print(count_parameters(net))
    print(count_parameters(net.encoder))
    print(count_parameters(net.decoder))
    print(count_parameters(net.logits))
    out = net(img)
    print(out.size())


@torch.no_grad()
def test_hires_fpn_resnext50():
    from pytorch_toolbelt.utils.torch_utils import count_parameters

    net = hdfpn_resnext50().eval()
    img = torch.rand((1, 3, 512, 512))
    print(count_parameters(net))
    print(count_parameters(net.encoder))
    print(count_parameters(net.decoder))
    print(count_parameters(net.logits))
    out = net(img)
    print(out.size())
