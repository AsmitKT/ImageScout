#Major\data\collate.py
import torch
import torchvision.transforms as T
from transformers import CLIPTokenizerFast
from torch.utils.data import get_worker_info


_CLIP_MEAN=(0.48145466,0.4578275,0.40821073)
_CLIP_STD=(0.26862954,0.26130258,0.27577711)


class EpochRNG:
    def __init__(self,base_seed=1234):
        self.base=int(base_seed);self.epoch=0;self.counter=0
    def set_epoch(self,e):
        self.epoch=int(e);self.counter=0
    def next_seed(self,worker_id):
        s=self.base+self.epoch*100000+worker_id*9973+self.counter
        self.counter+=1
        return s


def build_tokenizer():
    tok=CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    return tok


def build_image_transform(aug_cfg:dict,train:bool=True):
    tfms=[]
    if train:
        rrc=aug_cfg.get("random_resized_crop",{"size":224,"scale":[0.9,1.0],"ratio":[0.9,1.1]})
        tfms.append(T.RandomResizedCrop(size=rrc["size"],scale=tuple(rrc["scale"]),ratio=tuple(rrc["ratio"])))
        flip_p=aug_cfg.get("horizontal_flip_p",0.5)
        tfms.append(T.RandomHorizontalFlip(p=flip_p))
        if aug_cfg.get("use_color_jitter",True):
            cj=aug_cfg.get("color_jitter",{"brightness":0.1,"contrast":0.1,"saturation":0.1,"hue":0.02})
            tfms.append(T.ColorJitter(**cj))
    else:
        tfms.append(T.Resize(224,antialias=True))
        tfms.append(T.CenterCrop(224))
    tfms.extend([T.ToTensor(),T.Normalize(mean=_CLIP_MEAN,std=_CLIP_STD)])
    return T.Compose(tfms)


def collate_fn(batch,tokenizer,img_transform,max_len=77,epoch_rng=None):
    wi=get_worker_info()
    wid=wi.id if wi is not None else 0
    imgs=[]
    for b in batch:
        if epoch_rng is not None:
            seed=epoch_rng.next_seed(wid)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                imgs.append(img_transform(b["image"]))
        else:
            imgs.append(img_transform(b["image"]))
    captions=[b["caption"] for b in batch]
    t=tokenizer(captions,padding="longest",truncation=True,max_length=max_len,add_special_tokens=True,return_tensors="pt",return_attention_mask=True)
    pixel_values=torch.stack(imgs,dim=0)
    meta={k:[b[k] for b in batch] for k in ["caption","filename","img_id","sentid"]}
    return{"input_ids":t.input_ids,"attention_mask":t.attention_mask,"pixel_values":pixel_values,"meta":meta}


class Collator:
    def __init__(self,tokenizer,img_transform,max_len=77,epoch_rng=None):
        self.tokenizer=tokenizer
        self.img_transform=obj=img_transform
        self.max_len=max_len
        self.epoch_rng=epoch_rng
    def set_epoch(self,e):
        if self.epoch_rng is not None:
            self.epoch_rng.set_epoch(e)
    def __call__(self,batch):
        return collate_fn(batch,self.tokenizer,self.img_transform,self.max_len,self.epoch_rng)