#Major\models\clip_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import TextEncoder
from .vit_encoder import ViTEncoder


class CLIPLike(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = TextEncoder(cfg)
        self.vision_encoder = ViTEncoder(cfg)
       
        d_model = cfg["d_model"]
        embed_dim = cfg["embed_dim"]
       
        self.text_proj = nn.Linear(d_model, embed_dim)
        self.image_proj = nn.Linear(d_model, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))


    def encode_text(self, input_ids, attention_mask):
        x = self.text_encoder(input_ids, attention_mask)
        return F.normalize(self.text_proj(x), dim=-1)


    def encode_image(self, pixel_values):
        v = self.vision_encoder(pixel_values)
        return F.normalize(self.image_proj(v), dim=-1)


    def forward(self, input_ids, attention_mask, pixel_values, return_loss=True):
        t = self.encode_text(input_ids, attention_mask)
        v = self.encode_image(pixel_values)
        ls = self.logit_scale.clamp(min=math.log(1/100.0), max=math.log(100.0)).exp()
        logits = ls * (t @ v.t())
       
        if not return_loss:
            return {"text_embeds": t, "image_embeds": v, "logits": logits}
           
        labels = torch.arange(t.size(0), device=t.device)
        loss_t = F.cross_entropy(logits, labels)
        loss_v = F.cross_entropy(logits.t(), labels)
        return {"loss": 0.5 * (loss_t + loss_v), "logits": logits}