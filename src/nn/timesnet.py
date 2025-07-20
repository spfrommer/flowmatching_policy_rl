from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.fft
from jaxtyping import Float
from nn.timesnet_layers import DataEmbedding, Inception_Block_V1


@dataclass
class TimesNetConfig:
    seq_len: int = 64
    pred_len: int = 0
    e_layers: int = 3
    d_model: int = 128
    top_k: int = 3
    d_ff: int = 256
    num_kernels: int = 4
    dropout: float = 0.2

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, config: TimesNetConfig):
        super(TimesBlock, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.k = config.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(config.d_model, config.d_ff,
                               num_kernels=config.num_kernels),
            nn.GELU(),
            Inception_Block_V1(config.d_ff, config.d_model,
                               num_kernels=config.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection

        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, in_channels: int, config: TimesNetConfig):
        super(TimesNet, self).__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.model = nn.ModuleList(
            [TimesBlock(config) for _ in range(config.e_layers)]
        )
        self.enc_embedding = DataEmbedding(
            in_channels, config.d_model, config.dropout
        )
        self.layer = config.e_layers
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(
            config.d_model * config.seq_len, 1
        )

    def forward(self, x_enc: Float[Tensor, 'b t']):
        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, 1)
        return output