#Major\models\blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5


        # Self-Attention
        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-5)
        self.qkv = nn.Linear(self.d_model, self.d_model * 3, bias=True)
        self.attn_drop = nn.Dropout(cfg.get("attn_drop", 0.0))
        self.proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.proj_drop = nn.Dropout(cfg.get("proj_drop", 0.1))


        # MLP
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-5)
        mlp_hidden = int(self.d_model * cfg["mlp_ratio"])
        self.fc1 = nn.Linear(self.d_model, mlp_hidden, bias=True)
        self.fc2 = nn.Linear(mlp_hidden, self.d_model, bias=True)
        self.mlp_drop = nn.Dropout(cfg.get("mlp_drop", 0.1))
        self.resid_drop = nn.Dropout(cfg.get("resid_drop", 0.1))


    def forward(self, x, attn_mask=None):
        B, L, C = x.shape
        # --- MHSA ---
        y = self.ln1(x)
        qkv = self.qkv(y)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            mask = attn_mask[:, None, None, :].to(dtype=torch.bool, device=attn.device)
            attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)


        z = (attn @ v).transpose(1, 2).contiguous().view(B, L, C)
        x = x + self.proj_drop(self.proj(z))


        # --- MLP ---
        y = self.ln2(x)
        y = self.mlp_drop(F.gelu(self.fc1(y)))
        x = x + self.resid_drop(self.fc2(y))
        return x