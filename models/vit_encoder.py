#Major\models\vit_encoder.py
import torch
import torch.nn as nn
from .blocks import TransformerBlock


class ViTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg["patch_size"]
        image_size = cfg["image_size"]
        d_model = cfg["d_model"]
       
        self.n_patches = (image_size // self.patch_size) ** 2
        self.patch_dim = 3 * self.patch_size * self.patch_size


        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        self.pre_ln = nn.LayerNorm(d_model, eps=1e-5)
       
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ln_final = nn.LayerNorm(d_model, eps=1e-5)


        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.01)
        nn.init.normal_(self.pos_emb, std=0.01)


    def _to_patches(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().permute(0, 2, 3, 1, 4, 5).reshape(B, self.n_patches, self.patch_dim)
        return x


    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        patches = self._to_patches(pixel_values)
        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_emb
        x = self.pre_ln(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_final(x)[:, 0]