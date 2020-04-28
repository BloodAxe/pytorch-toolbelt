# Credit: https://github.com/PkuRainBow/OCNet.pytorch/blob/master/oc_module/asp_oc_block.py
import torch
import torch.nn.functional as F
from torch import nn

from ..modules.activations import ABN

__all__ = ["ObjectContextBlock", "ASPObjectContextBlock", "PyramidObjectContextBlock"]


class _SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, abn_block=ABN):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0
            ),
            abn_block(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0
        )
        self.W = nn.Conv2d(
            in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0
        )
        # Eugene Khvedchenya: Original implementation initialized weight of context convolution with zeros, which does not make sense to me
        # nn.init.constant(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):  # skipcq: PYL-W0221
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode="bilinear", align_corners=False)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels, key_channels, value_channels, out_channels, scale)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), abn_block=ABN):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes]
        )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            abn_block(out_channels),
            nn.Dropout2d(dropout),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class ObjectContextBlock(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), abn_block=ABN):
        super(ObjectContextBlock, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes]
        )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False), abn_block(out_channels)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class ASPObjectContextBlock(nn.Module):
    def __init__(self, features, out_features=256, dilations=(12, 24, 36), abn_block=ABN, dropout=0.1):
        super(ASPObjectContextBlock, self).__init__()
        self.context = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            abn_block(out_features),
            ObjectContextBlock(
                in_channels=out_features,
                out_channels=out_features,
                key_channels=out_features // 2,
                value_channels=out_features,
                dropout=dropout,
                sizes=([2]),
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            abn_block(out_features),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            abn_block(out_features),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            abn_block(out_features),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            abn_block(out_features),
        )

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features * 2, kernel_size=1, padding=0, dilation=1, bias=False),
            abn_block(out_features * 2),
            nn.Dropout2d(dropout),
        )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert len(feat1) == len(feat2)
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), dim=1))
        return z

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError("unknown input type")

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, torch.Tensor):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError("unknown input type")

        output = self.conv_bn_dropout(out)
        return output


class _PyramidSelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, abn_block=ABN):
        super(_PyramidSelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            abn_block(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0
        )
        self.W = nn.Conv2d(
            in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0
        )
        # nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, _, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        local_x = []
        local_y = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        local_list = []
        local_block_cnt = 2 * self.scale * self.scale
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
            query_local = query[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
            key_local = key[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]

            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.value_channels, -1)
            value_local = value_local.permute(0, 2, 1)

            query_local = query_local.contiguous().view(batch_size, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.key_channels, -1)

            sim_map = torch.matmul(query_local, key_local)
            sim_map = (self.key_channels ** -0.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.matmul(sim_map, value_local)
            context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size, self.value_channels, h_local, w_local)
            local_list.append(context_local)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = self.W(context)

        return context


class PyramidSelfAttentionBlock2D(_PyramidSelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(PyramidSelfAttentionBlock2D, self).__init__(
            in_channels, key_channels, value_channels, out_channels, scale
        )


class PyramidObjectContextBlock(nn.Module):
    """
    Output the combination of the context features and the original features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, dropout=0.05, sizes=([1, 2, 3, 6]), abn_block=ABN):
        super(PyramidObjectContextBlock, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, in_channels // 2, in_channels, size) for size in sizes]
        )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels * self.group, out_channels, kernel_size=1, padding=0, bias=False),
            abn_block(out_channels),
            nn.Dropout2d(dropout),
        )
        self.up_dr = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * self.group, kernel_size=1, padding=0, bias=False),
            abn_block(in_channels * self.group),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return PyramidSelfAttentionBlock2D(in_channels, key_channels, value_channels, output_channels, size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = [self.up_dr(feats)]
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn_dropout(torch.cat(context, 1))
        return output
