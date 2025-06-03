import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flash_attn import flash_attn_func

class MarketGuidedGating(nn.Module):
    def __init__(self, market_dim, feature_dim, beta=5):
        super().__init__()
        self.fc = nn.Linear(market_dim, feature_dim)
        self.beta = beta
        self.feature_dim = feature_dim

    def forward(self, x, m):
        alpha = self.feature_dim * F.softmax(self.fc(m) / self.beta, dim=-1)
        return x * alpha  # Hadamard product

class IntraStockEncoder(nn.Module):
    def __init__(self, feature_dim, embed_dim=256, nhead=4, max_len=60):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)#
        self.register_buffer('pos_encoder', self._get_sinusoid_encoding_table(max_len, embed_dim))#
        self.encoder_norm = nn.LayerNorm(embed_dim)#
        
        self.embed_dim = embed_dim#
        self.nhead = nhead#
        self.head_dim = embed_dim // nhead#
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"#
        
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim) #
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        

    def forward(self, x):
        batch, time, _ = x.shape#
        x = self.input_proj(x)#
        #print(f'x.shape : {x.shape}')
        x = x + self.pos_encoder[:time, :].unsqueeze(0)#
        x = self.encoder_norm(x)#

        # QKV projection
        qkv = self.qkv_proj(x)  # (batch, time, 3*embed_dim) #
        qkv = qkv.view(batch, time, 3, self.nhead, self.head_dim) #
        q, k, v = qkv.unbind(dim=2)  # (batch, time, nhead, head_dim) #

        # Flash Attention
        attn_out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)  # (batch, time, nhead, head_dim)#
        attn_out = attn_out.view(batch, time, self.embed_dim)

        # Residual + Norm
        x = self.norm1(attn_out + x)

        # Feed Forward
        ffn_out = self.ffn(x)
        out = self.norm2(ffn_out + x)  # 注意這裡是再次 residual+norm

        return out

    def _get_sinusoid_encoding_table(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class InterStockAggregator(nn.Module):
    def __init__(self, embed_dim=256, nhead=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch, stocks, time, embed_dim = x.shape#

        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * time, stocks, embed_dim)#

        qkv = self.qkv_proj(x_reshaped)
        qkv = qkv.view(batch * time, stocks, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        attn_out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        attn_out = attn_out.reshape(batch * time, stocks, embed_dim)

        x_attn = self.norm1(attn_out + x_reshaped)
        ffn_out = self.ffn(x_attn)
        out = self.norm2(ffn_out + x_attn)

        out = out.view(batch, time, stocks, embed_dim).permute(0, 2, 1, 3)

        return out

class TemporalAggregator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.w_lambda = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x):
        query = x[:, :, -1, :]
        scores = torch.einsum('bstf,fd,bsd->bst', x, self.w_lambda, query)
        weights = F.softmax(scores, dim=2)
        output = torch.einsum('bst,bstf->bsf', weights, x)
        return output

class FlashMASTER(nn.Module):
    def __init__(self, market_dim, feature_dim, embed_dim=256, nhead1=4, nhead2=2, beta=5):
        super().__init__()
        self.gating = MarketGuidedGating(market_dim, feature_dim, beta)
        self.intra_encoder = IntraStockEncoder(feature_dim, embed_dim, nhead1)
        self.inter_agg = InterStockAggregator(embed_dim, nhead2)
        self.temporal_agg = TemporalAggregator(embed_dim)
        self.predictor = nn.Linear(embed_dim, 1)

    def forward(self, x, market):
        batch, stocks, time, features = x.shape

        market_expanded = market[:, None, None, :].expand(-1, stocks, time, -1)
        market_scaled = self.gating(x, market_expanded)

        x_flat = market_scaled.view(batch * stocks, time, features)
        local_embed = self.intra_encoder(x_flat)
        local_embed = local_embed.view(batch, stocks, time, -1)

        inter_embed = self.inter_agg(local_embed)
        temporal_embed = self.temporal_agg(inter_embed)
        out = self.predictor(temporal_embed).squeeze(-1)

        return out
