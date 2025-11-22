import torch
import torch.nn as nn
from dataclasses import dataclass
from math import *


@dataclass
class cfg:
    num_patches = 4
    num_channel = 3
    img_dim = 8
    d_model = 512
    num_head = 6
    d_head = d_model // num_head
    num_encoder_layer = 12
    num_classes = 10

# TODO: change to use custom layernorm implementation
class LayerNorm:
    def __init__(self):
        pass

    def forward(self,x):
        pass

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layers = nn.ModuleList([Encoder()]*cfg.num_encoder_layer)
        self.head = nn.Linear(cfg.num_patches*cfg.d_model)

    def forward(self,x):
        b,_,_ = x.shape

        for layer in self.encoder_layers:
            x = layer(x) # [b,p,d_model]
        
        out = self.head(x.view(b,-1)) # [b,1]

        return out
        
    @staticmethod
    def get_patches(image, num_patches):
        unfolded_h = image.unfold(dimension=-2, size=num_patches, step=num_patches)
        unfolded_w = unfolded_h.unfold(dimension=-1, size=num_patches, step=num_patches)
        patches = unfolded_w.reshape(
            image.shape[0], num_patches, -1
        )  # [b, p, c * p_h * p_w]
        return patches


class MultiHeadAttention(nn.Module):
    def __init__(self):
        self.w_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model)

        self.proj_out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        b, p, _ = x.shape

        q = self.w_q(x)  # [b,p,d_model]
        k = self.w_k(x)  # [b,p,d_model]
        v = self.w_v(x)  # [b,p,d_model]

        q = q.view(b, p, cfg.num_head, cfg.d_head).transpose(1, 2)  # [b,h,p,d_head]
        k = k.view(b, p, cfg.num_head, cfg.d_head).transpose(1, 2)  # [b,h,p,d_head]
        v = v.view(b, p, cfg.num_head, cfg.d_head).transpose(1, 2)  # [b,h,p,d_head]

        scores = q @ k.T  # [b,h,p,p]
        attn_weights = torch.softmax(scores, axis=-1)  # [b,h,p,p]
        output = attn_weights @ v  # [b,h,p,p] @ [b,h,p,d_head] = [b,h,p,d_head]
        output = output.transpose(1, 2).contiguous().view(b, p, -1)

        output = self.proj_out(output)

        return output  # [b,p,d_model]

class MLP(nn.Module):
    def __init__(self):
        self.fc1_block = nn.Sequential(
            nn.Linear(cfg.d_model,4*cfg.d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.fc2_block = nn.Sequential(
            nn.Linear(4*cfg.d_model,cfg.d_model)
            nn.Dropout(0.1)
        )

    def forward(self,x):
        x = self.fc1_block(x) # [b,p,4*d_model]
        x = self.fc2_block(x) # [b,p,d_model]
        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)

        self.mha = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x):
        b, p, _ = x.shape
        x_org = x.copy()
        x = self.layer_norm1(x)  # [b,p,d_model] -> [b,p,d_model]
        mha_out = self.mha(x)  # [b,p,d_model]
        x = mha_out + x_org  # [b,p,d_model]
        
        x_org = x.copy()        # [b,p,d_model]
        x = self.layer_norm2(x) # [b,p,d_model]
        mlp_out = MLP(x) # [b,p,d_model]
        x = mlp_out + x_org #  [b,p,d_model]
        return x

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(cfg.num_channel * (cfg.num_patches) ** 2, cfg.d_model)
        pos = torch.arange(0, cfg.num_patches, dtype=torch.float32).unsqueeze(1)
        div_term = torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
        div_term = torch.exp(div_term * (-log(10000.0) / cfg.d_model))

        pe = torch.zeros(cfg.num_patches, cfg.d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pos_emb", pe)

    def forward(self, x):
        token_emb = self.proj(x)  # [b, p, c * p_h * p_w] -> [b,p,d_model]
        pos_emb = self.pos_emb[: x.shape[1], :].unsqueeze(0)  # [p,d_model]

        return token_emb + pos_emb  # [b, p, d_model]


if __name__ == "__main__":

    x = torch.randn(1, cfg.num_channel, cfg.img_dim, cfg.img_dim)  # [b,c,h,w]

    x = VisionTransformer.get_patches(
        x, cfg.num_patches
    )  # [b, p, c * p_h * p_w * p_size]

    emb = Embedding()

    emb_patches = emb(x)
