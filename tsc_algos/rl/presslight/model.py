'''
@Author: WANG Maonan
@Date: 2026-06-02 16:36:40
@Description: PressLight movement-matrix feature extractor
@LastEditTime: 2026-06-03 21:24:03
'''
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PressLightMovementModel(BaseFeaturesExtractor):
    """Encode each movement row, concatenate them, then encode the frame.

    Shape symbols:
    B = batch size, T = history length, M = number of movements,
    F = movement feature dimension, H = movement embedding dimension,
    D = output feature dimension.
    """

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
        num_movements = observation_space.shape[1] # M
        movement_feature_dim = observation_space.shape[2] # F

        layers = []
        prev_dim = movement_feature_dim # F
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim)) # (..., prev_dim) -> (..., hidden_dim)
            layers.append(nn.ReLU()) # (..., hidden_dim) -> (..., hidden_dim)
            if dropout > 0:
                layers.append(nn.Dropout(dropout)) # (..., hidden_dim) -> (..., hidden_dim)
            prev_dim = hidden_dim # update embedding dim; final value is H

        self.movement_encoder = nn.Sequential(*layers) # (B*T, M, F) -> (B*T, M, H)
        self.frame_encoder = nn.Sequential(
            nn.Linear(num_movements * prev_dim, features_dim), # (B*T, M*H) -> (B*T, D)
            nn.ReLU(), # (B*T, D) -> (B*T, D)
        )
        self.temporal_encoder = nn.GRU(
            input_size=features_dim, # input: (B, T, D)
            hidden_size=features_dim, # hidden: (1, B, D)
            batch_first=True,
        )

    def _encode_frame(self, observations: torch.Tensor) -> torch.Tensor:
        movement_mask = observations.abs().sum(dim=-1) > 0 # (B*T, M, F) -> (B*T, M)
        movement_features = self.movement_encoder(observations) # (B*T, M, F) -> (B*T, M, H)
        movement_features = movement_features * movement_mask.unsqueeze(-1) # (B*T, M, H) * (B*T, M, 1) -> (B*T, M, H)
        frame_features = movement_features.reshape(observations.shape[0], -1) # (B*T, M, H) -> (B*T, M*H)
        return self.frame_encoder(frame_features) # (B*T, M*H) -> (B*T, D)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, history_len, num_movements, feature_dim = observations.shape # observations: (B, T, M, F)
        flat_observations = observations.reshape(
            batch_size * history_len,
            num_movements,
            feature_dim,
        ) # (B, T, M, F) -> (B*T, M, F)
        frame_features = self._encode_frame(flat_observations) # (B*T, M, F) -> (B*T, D)
        sequence_features = frame_features.reshape(batch_size, history_len, -1) # (B*T, D) -> (B, T, D)
        _, hidden = self.temporal_encoder(sequence_features) # (B, T, D) -> hidden (1, B, D)
        return hidden[-1] # (1, B, D) -> (B, D)
