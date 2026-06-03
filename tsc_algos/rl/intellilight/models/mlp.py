'''
@Author: WANG Maonan
@Description: 简单的 MLP 基线模型（用于对比）
'''
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MLPJunctionModel(BaseFeaturesExtractor):
    """MLP 基线模型，将所有 lane 特征展平后通过 MLP 处理"""
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        hidden_dims: list = [256, 256],
    ):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0] * observation_space.shape[1]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, features_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        flattened = observations.reshape(batch_size, -1)
        features = self.mlp(flattened)
        return features
