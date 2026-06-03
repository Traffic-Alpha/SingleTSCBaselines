'''
@Author: WANG Maonan
@Description: 基于 Transformer 的路口模型
'''
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerJunctionModel(BaseFeaturesExtractor):
    """基于 Transformer 的路口模型

    架构：
    1. 输入 lane 级别的特征 (num_lanes, feature_dim)
    2. 线性投影到 embedding 空间
    3. 添加可学习的 CLS token + 位置编码
    4. Transformer Encoder 处理
    5. CLS token 输出 -> MLP -> 最终特征
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        self.num_lanes = observation_space.shape[0]
        self.lane_feature_dim = observation_space.shape[1]
        self.d_model = features_dim

        # 输入投影
        self.input_projection = nn.Linear(self.lane_feature_dim, self.d_model)

        # CLS token + 位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_lanes + 1, self.d_model)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        # 输出 MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, features_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_encoding, std=0.02)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        lane_embeddings = self.input_projection(observations)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, lane_embeddings], dim=1)
        embeddings = embeddings + self.positional_encoding

        transformer_output = self.transformer_encoder(embeddings)
        cls_output = transformer_output[:, 0, :]
        features = self.output_mlp(cls_output)

        return features
