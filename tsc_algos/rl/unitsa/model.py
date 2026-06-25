'''
@Author: WANG Maonan
@Date: 2026-06-03
@Description: UniTSA movement-token Transformer feature extractor
'''
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class UniTSAMovementTransformer(BaseFeaturesExtractor):
    """Encode UniTSA movement history with a Transformer.

    Shape symbols:
    B = batch size, T = history length, M = number of movements,
    F = movement feature dimension, E = token embedding dimension,
    N = number of movement-time tokens (T*M), L = total token length (N+1),
    D = output feature dimension.

    The observation contract is identical to PressLight:
    observations have shape (B, T, M, F). Each movement row is first projected
    into an embedding token, then a Transformer attends over all movement-time
    tokens. A CLS token provides the final feature vector for DQN.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 3:
            raise ValueError(
                "UniTSAMovementTransformer expects 3D observations "
                f"(T, M, F), got {observation_space.shape}."
            )
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.history_len = observation_space.shape[0] # T
        self.num_movements = observation_space.shape[1] # M
        self.movement_feature_dim = observation_space.shape[2] # F
        self.embedding_dim = embedding_dim # E

        self.movement_encoder = nn.Sequential(
            nn.Linear(self.movement_feature_dim, embedding_dim), # (..., F) -> (..., E)
            nn.LayerNorm(embedding_dim), # (..., E) -> (..., E)
            nn.GELU(), # (..., E) -> (..., E)
            nn.Dropout(dropout), # (..., E) -> (..., E)
            nn.Linear(embedding_dim, embedding_dim), # (..., E) -> (..., E)
            nn.LayerNorm(embedding_dim), # (..., E) -> (..., E)
            nn.GELU(), # (..., E) -> (..., E)
        ) # (B, T, M, F) -> (B, T, M, E)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim)) # (1, 1, E)
        self.time_embedding = nn.Parameter(torch.zeros(1, self.history_len, 1, embedding_dim)) # (1, T, 1, E)
        self.movement_embedding = nn.Parameter(torch.zeros(1, 1, self.num_movements, embedding_dim)) # (1, 1, M, E)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),
        ) # tokens: (B, L, E) -> (B, L, E)

        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), # (B, E) -> (B, E)
            nn.GELU(), # (B, E) -> (B, E)
            nn.Dropout(dropout), # (B, E) -> (B, E)
            nn.Linear(embedding_dim, features_dim), # (B, E) -> (B, D)
            nn.ReLU(), # (B, D) -> (B, D)
        ) # CLS feature: (B, E) -> (B, D)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.movement_embedding, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, history_len, num_movements, feature_dim = observations.shape # observations: (B, T, M, F)
        if history_len != self.history_len or num_movements != self.num_movements:
            raise ValueError(
                "UniTSAMovementTransformer got observation shape "
                f"{tuple(observations.shape[1:])}, expected "
                f"({self.history_len}, {self.num_movements}, {self.movement_feature_dim})."
            )
        if feature_dim != self.movement_feature_dim:
            raise ValueError(
                f"Unexpected movement feature dim {feature_dim}, expected {self.movement_feature_dim}."
            )

        movement_tokens = self.movement_encoder(observations) # (B, T, M, F) -> (B, T, M, E)
        movement_tokens = movement_tokens + self.time_embedding + self.movement_embedding # (B, T, M, E) + (1, T, 1, E) + (1, 1, M, E) -> (B, T, M, E)
        movement_tokens = movement_tokens.reshape(
            batch_size,
            history_len * num_movements,
            self.embedding_dim,
        ) # (B, T, M, E) -> (B, N, E), where N=T*M

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (1, 1, E) -> (B, 1, E)
        tokens = torch.cat([cls_tokens, movement_tokens], dim=1) # (B, 1, E) + (B, N, E) -> (B, L, E), where L=1+N

        zero_movement_mask = observations.abs().sum(dim=-1).reshape(
            batch_size,
            history_len * num_movements,
        ) == 0 # (B, T, M, F) -> (B, T, M) -> (B, N)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=observations.device) # (B, 1)
        key_padding_mask = torch.cat([cls_mask, zero_movement_mask], dim=1) # (B, 1) + (B, N) -> (B, L)

        encoded_tokens = self.transformer_encoder(
            tokens,
            src_key_padding_mask=key_padding_mask,
        ) # (B, L, E) -> (B, L, E)
        cls_output = encoded_tokens[:, 0, :] # (B, L, E) -> (B, E)
        return self.output_head(cls_output) # (B, E) -> (B, D)
