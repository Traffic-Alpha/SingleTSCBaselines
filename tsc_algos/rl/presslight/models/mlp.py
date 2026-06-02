'''
@Author: WANG Maonan
@Description: PressLight movement-matrix feature extractor
'''
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PressLightMovementModel(BaseFeaturesExtractor):
    """Encode movement time-series observations for PressLight."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        hidden_dims: list = None,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim)

        if hidden_dims is None:
            hidden_dims = [128, 128]

        if len(observation_space.shape) != 3:
            raise ValueError(
                f"PressLightMovementModel expects 3D observations, got {observation_space.shape}."
            )
        movement_feature_dim = observation_space.shape[2]

        layers = []
        prev_dim = movement_feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.movement_encoder = nn.Sequential(*layers)
        self.frame_encoder = nn.Sequential(
            nn.Linear(prev_dim * 2, features_dim),
            nn.ReLU(),
        )
        self.temporal_encoder = nn.GRU(
            input_size=features_dim,
            hidden_size=features_dim,
            batch_first=True,
        )

    def _encode_frame(self, observations: torch.Tensor) -> torch.Tensor:
        movement_mask = observations.abs().sum(dim=-1) > 0
        movement_features = self.movement_encoder(observations)

        mask = movement_mask.unsqueeze(-1)
        masked_features = movement_features * mask
        valid_counts = mask.sum(dim=1).clamp(min=1)
        mean_features = masked_features.sum(dim=1) / valid_counts

        max_features = movement_features.masked_fill(~mask, -1e9).max(dim=1).values
        max_features = torch.where(
            movement_mask.any(dim=1, keepdim=True),
            max_features,
            torch.zeros_like(max_features),
        )

        return self.frame_encoder(torch.cat([mean_features, max_features], dim=-1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, history_len, num_movements, feature_dim = observations.shape
        flat_observations = observations.reshape(
            batch_size * history_len,
            num_movements,
            feature_dim,
        )
        frame_features = self._encode_frame(flat_observations)
        sequence_features = frame_features.reshape(batch_size, history_len, -1)
        _, hidden = self.temporal_encoder(sequence_features)
        return hidden[-1]
