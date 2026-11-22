from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SpatialEncoder(nn.Module):
    """A compact CNN encoder shared conceptually by both modalities."""

    def __init__(self, in_channels: int, stem_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, stem_channels),
            ConvBlock(stem_channels, stem_channels),
            nn.MaxPool2d(2),
            ConvBlock(stem_channels, stem_channels * 2),
            ConvBlock(stem_channels * 2, stem_channels * 2),
            nn.MaxPool2d(2),
            ConvBlock(stem_channels * 2, stem_channels * 4),
            ConvBlock(stem_channels * 4, stem_channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = stem_channels * 4

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.stem(inputs)).flatten(1)
        return features


class DualStreamFusionCNN(nn.Module):
    """HSI and LiDAR streams with deep feature-level concatenation fusion."""

    def __init__(
        self,
        hsi_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        dropout: float = 0.3,
        fusion_mode: str = "fusion",
        hsi_weight: float = 1.0,
        lidar_weight: float = 1.0,
    ) -> None:
        super().__init__()
        valid_modes = {"fusion", "hsi", "lidar"}
        if fusion_mode not in valid_modes:
            raise ValueError(f"fusion_mode must be one of {sorted(valid_modes)}")
        if hsi_weight < 0.0 or lidar_weight < 0.0:
            raise ValueError("Fusion weights must be non-negative")
        if hsi_weight + lidar_weight <= 0.0:
            raise ValueError("At least one fusion weight must be positive")
        self.fusion_mode = fusion_mode
        self.hsi_weight = float(hsi_weight)
        self.lidar_weight = float(lidar_weight)
        self.hsi_encoder = SpatialEncoder(hsi_channels, stem_channels)
        self.lidar_encoder = SpatialEncoder(1, stem_channels)
        feature_dim = self.hsi_encoder.output_dim

        self.hsi_head = nn.Linear(feature_dim, num_classes)
        self.lidar_head = nn.Linear(feature_dim, num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward_features(
        self, hsi: torch.Tensor, lidar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hsi_features = self.hsi_encoder(hsi)
        lidar_features = self.lidar_encoder(lidar)
        return hsi_features, lidar_features

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        hsi_features, lidar_features = self.forward_features(hsi, lidar)
        if self.fusion_mode == "hsi":
            return self.hsi_head(hsi_features)
        if self.fusion_mode == "lidar":
            return self.lidar_head(lidar_features)
        weighted_features = torch.cat(
            (
                hsi_features * self.hsi_weight,
                lidar_features * self.lidar_weight,
            ),
            dim=1,
        )
        return self.fusion_head(weighted_features)

