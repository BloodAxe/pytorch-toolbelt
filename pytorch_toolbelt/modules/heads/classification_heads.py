from typing import List

import torch
from torch import nn, Tensor

from pytorch_toolbelt.modules import instantiate_activation_block, GeneralizedMeanPooling2d
from pytorch_toolbelt.modules.interfaces import AbstractHead, FeatureMapsSpecification, HasOutputFeaturesSpecification

__all__ = [
    "GlobalAveragePoolingClassificationHead",
    "GlobalMaxPoolingClassificationHead",
    "GenericPoolingClassificationHead",
    "FullyConnectedClassificationHead",
    "GlobalMaxAvgPoolingClassificationHead",
    "GlobalMaxAvgSumPoolingClassificationHead",
    "GeneralizedMeanPoolingClassificationHead",
]


class GenericPoolingClassificationHead(AbstractHead):
    def __init__(
        self,
        *,
        input_spec: FeatureMapsSpecification,
        pooling: nn.Module,
        num_classes: int,
        dropout_rate: float = 0.0,
        feature_map_index: int = -1,
    ):
        super().__init__(input_spec)
        self.pooling = pooling
        self.feature_map_index = feature_map_index
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(input_spec.channels[self.feature_map_index], num_classes)

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        x = feature_maps[self.feature_map_index]
        x = self.pooling(x).flatten(start_dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class GlobalMaxPoolingClassificationHead(GenericPoolingClassificationHead, HasOutputFeaturesSpecification):
    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        dropout_rate: float = 0.0,
        feature_map_index: int = -1,
    ):
        pooling = nn.AdaptiveMaxPool2d((1, 1))
        super().__init__(
            input_spec=input_spec,
            pooling=pooling,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            feature_map_index=feature_map_index,
        )

    def get_output_spec(self) -> FeatureMapsSpecification:
        return FeatureMapsSpecification(channels=[self.num_classes], strides=[-1])


class GlobalAveragePoolingClassificationHead(GenericPoolingClassificationHead, HasOutputFeaturesSpecification):
    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        dropout_rate: float = 0.0,
        feature_map_index: int = -1,
    ):
        pooling = nn.AdaptiveAvgPool2d((1, 1))

        super().__init__(
            input_spec=input_spec,
            pooling=pooling,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            feature_map_index=feature_map_index,
        )

    def get_output_spec(self) -> FeatureMapsSpecification:
        return FeatureMapsSpecification(channels=[self.num_classes], strides=[-1])


class GlobalMaxAvgPoolingClassificationHead(AbstractHead, HasOutputFeaturesSpecification):
    def __init__(
        self,
        *,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        activation: str,
        dropout_rate: float = 0.0,
        feature_map_index: int = -1,
    ):
        super().__init__(input_spec)
        self.num_classes = num_classes
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_map_index = feature_map_index
        num_channels = input_spec.channels[self.feature_map_index]
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(num_channels * 2),
            nn.Linear(num_channels * 2, num_channels),
            instantiate_activation_block(activation, inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(num_channels),
            nn.Linear(num_channels, num_channels),
            instantiate_activation_block(activation, inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        x = feature_maps[self.feature_map_index]
        x_max = self.max_pooling(x).flatten(start_dim=1)
        x_avg = self.avg_pooling(x).flatten(start_dim=1)
        x = torch.cat([x_max, x_avg], dim=1)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x

    def get_output_spec(self) -> FeatureMapsSpecification:
        return FeatureMapsSpecification(channels=[self.num_classes], strides=[-1])


class GeneralizedMeanPoolingClassificationHead(AbstractHead, HasOutputFeaturesSpecification):
    def __init__(
        self,
        *,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        dropout_rate: float = 0.0,
        feature_map_index: int = -1,
    ):
        super().__init__(input_spec)
        self.num_classes = num_classes
        self.pooling = GeneralizedMeanPooling2d(l2_normalize=True, flatten=True)

        self.feature_map_index = feature_map_index
        num_channels = input_spec.channels[self.feature_map_index]
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        x = feature_maps[self.feature_map_index]
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def get_output_spec(self) -> FeatureMapsSpecification:
        return FeatureMapsSpecification(channels=[self.num_classes], strides=[-1])


class FullyConnectedClassificationHead(AbstractHead):
    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        dropout_rate: float = 0.0,
        feature_map_index: int = -1,
    ):
        super().__init__(input_spec)
        self.feature_map_index = feature_map_index
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        x = feature_maps[self.feature_map_index]
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class GlobalMaxAvgSumPoolingClassificationHead(AbstractHead, HasOutputFeaturesSpecification):
    def __init__(
        self,
        *,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        feature_map_index: int = -1,
        dropout_rate: float = 0.0,
    ):
        super().__init__(input_spec)
        self.num_classes = num_classes
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_map_index = feature_map_index
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate > 0 else nn.Identity()
        num_channels = input_spec.channels[self.feature_map_index]
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        x = feature_maps[self.feature_map_index]
        x_max = self.max_pooling(x).flatten(start_dim=1)
        x_avg = self.avg_pooling(x).flatten(start_dim=1)
        x = self.dropout((x_max + x_avg) * 0.5)
        x = self.classifier(x)
        return x

    def get_output_spec(self) -> FeatureMapsSpecification:
        return FeatureMapsSpecification(channels=[self.num_classes], strides=[-1])
