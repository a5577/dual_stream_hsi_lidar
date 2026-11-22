from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import nn

from .model import SpatialEncoder


class CrossAttentionFusion(nn.Module):
    """Fuse aligned HSI and LiDAR CNN feature maps with mutual attention."""

    def __init__(
        self,
        encoder_channels: int,
        attention_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")

        self.hsi_projection = nn.Conv2d(
            encoder_channels, attention_dim, kernel_size=1
        )
        self.lidar_projection = nn.Conv2d(
            encoder_channels, attention_dim, kernel_size=1
        )
        self.hsi_norm = nn.LayerNorm(attention_dim)
        self.lidar_norm = nn.LayerNorm(attention_dim)

        self.hsi_to_lidar = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    attention_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.lidar_to_hsi = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    attention_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.hsi_ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(attention_dim, attention_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(attention_dim * 4, attention_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.lidar_ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(attention_dim, attention_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(attention_dim * 4, attention_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.hsi_ffn_norm = nn.ModuleList(
            [nn.LayerNorm(attention_dim) for _ in range(num_layers)]
        )
        self.lidar_ffn_norm = nn.ModuleList(
            [nn.LayerNorm(attention_dim) for _ in range(num_layers)]
        )

        self.hsi_pool_score = nn.Linear(attention_dim, 1)
        self.lidar_pool_score = nn.Linear(attention_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, 1),
        )
        self.output_norm = nn.LayerNorm(attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dim = attention_dim

    @staticmethod
    def _attention_pool(
        tokens: torch.Tensor,
        score_layer: nn.Linear,
    ) -> torch.Tensor:
        scores = score_layer(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)

    def forward(
        self,
        hsi_feature_map: torch.Tensor,
        lidar_feature_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hsi = self.hsi_projection(hsi_feature_map).flatten(2).transpose(1, 2)
        lidar = (
            self.lidar_projection(lidar_feature_map)
            .flatten(2)
            .transpose(1, 2)
        )
        last_hsi_attention = None
        last_lidar_attention = None

        for hsi_attention, lidar_attention, hsi_ffn, lidar_ffn, hsi_norm, lidar_norm in zip(
            self.hsi_to_lidar,
            self.lidar_to_hsi,
            self.hsi_ffn,
            self.lidar_ffn,
            self.hsi_ffn_norm,
            self.lidar_ffn_norm,
        ):
            hsi_query = self.hsi_norm(hsi)
            lidar_query = self.lidar_norm(lidar)
            hsi_context, last_hsi_attention = hsi_attention(
                hsi_query,
                lidar_query,
                lidar_query,
                need_weights=True,
                average_attn_weights=True,
            )
            lidar_context, last_lidar_attention = lidar_attention(
                lidar_query,
                hsi_query,
                hsi_query,
                need_weights=True,
                average_attn_weights=True,
            )
            hsi = hsi + self.dropout(hsi_context)
            lidar = lidar + self.dropout(lidar_context)
            hsi = hsi_norm(hsi + self.dropout(hsi_ffn(hsi)))
            lidar = lidar_norm(lidar + self.dropout(lidar_ffn(lidar)))

        pooled_hsi = self._attention_pool(hsi, self.hsi_pool_score)
        pooled_lidar = self._attention_pool(lidar, self.lidar_pool_score)
        gate = torch.sigmoid(
            self.gate(torch.cat((pooled_hsi, pooled_lidar), dim=1))
        )
        fused = gate * pooled_hsi + (1.0 - gate) * pooled_lidar
        return (
            self.output_norm(fused),
            gate.squeeze(1),
            last_hsi_attention,
            last_lidar_attention,
        )


class AttentionDualStreamCNN(nn.Module):
    """HSI/LiDAR CNN with cross-attention instead of vector concatenation."""

    def __init__(
        self,
        hsi_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        attention_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hsi_encoder = SpatialEncoder(hsi_channels, stem_channels)
        self.lidar_encoder = SpatialEncoder(1, stem_channels)
        encoder_channels = stem_channels * 4
        self.fusion = CrossAttentionFusion(
            encoder_channels=encoder_channels,
            attention_dim=attention_dim,
            num_heads=attention_heads,
            num_layers=attention_layers,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attention_dim, num_classes),
        )

    def forward_features(
        self,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hsi_feature_map = self.hsi_encoder.stem(hsi)
        lidar_feature_map = self.lidar_encoder.stem(lidar)
        if hsi_feature_map.shape[-2:] != lidar_feature_map.shape[-2:]:
            raise ValueError(
                "HSI and LiDAR branches must produce aligned feature-map sizes"
            )
        fused, gate, hsi_attention, lidar_attention = self.fusion(
            hsi_feature_map,
            lidar_feature_map,
        )
        self.last_fusion_gate = gate.detach()
        self.last_hsi_to_lidar_attention = (
            None if hsi_attention is None else hsi_attention.detach()
        )
        self.last_lidar_to_hsi_attention = (
            None if lidar_attention is None else lidar_attention.detach()
        )
        return fused, gate

    def forward(
        self,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        features, gate = self.forward_features(hsi, lidar)
        logits = self.classifier(features)
        if not return_attention:
            return logits
        return logits, {
            "fusion_gate": gate,
            "hsi_to_lidar": self.last_hsi_to_lidar_attention,
            "lidar_to_hsi": self.last_lidar_to_hsi_attention,
        }
