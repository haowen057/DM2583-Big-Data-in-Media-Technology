import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x.mean(dim=1, keepdim=True))
        return x * scale


class DiabetesTransformer(nn.Module):
    def __init__(self, num_features, config):
        super().__init__()
        d_model = config["D_MODEL"]
        nhead = config["NHEAD"]
        num_layers = config["NUM_LAYERS"]
        dropout = config["DROPOUT"]

        self.feature_embedding = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.feature_attn = FeatureAttention(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        x = self.feature_embedding(x).unsqueeze(1)
        x = x + self.pos_encoding[:, :1, :]
        x = self.transformer_encoder(x)
        x = self.feature_attn(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
