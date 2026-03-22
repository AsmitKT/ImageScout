#Major\models\text_encoder.py
import torch
import torch.nn as nn
from .blocks import TransformerBlock


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eos_token_id = cfg.get("eos_token_id", 49407)
        d_model = cfg["d_model"]
       
        self.token_emb = nn.Embedding(cfg["vocab_size"], d_model)
        self.pos_emb = nn.Parameter(torch.zeros(cfg["max_len"], d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ln_final = nn.LayerNorm(d_model, eps=1e-5)


        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.01)


    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:L]
        for blk in self.blocks:
            x = blk(x, attn_mask=attention_mask)
        x = self.ln_final(x)


        # EOS pooling
        with torch.no_grad():
            is_eos = (input_ids == self.eos_token_id)
            eos_idx = torch.where(
                is_eos.any(dim=1),
                is_eos.float().argmax(dim=1),
                (attention_mask.sum(dim=1) - 1).clamp(min=0)
            ).long()
        return x[torch.arange(B, device=x.device), eos_idx]