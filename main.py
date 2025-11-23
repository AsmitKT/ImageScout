#Major\main.py
import os, random, torch
from torch.utils.data import DataLoader
from config import CFG
from utils.devices import pick_device
from data.dataset import load_flickr30k, Flickr30kCLIPDataset
from data.collate import build_tokenizer, build_image_transform, Collator, EpochRNG
from models.clip_model import CLIPLike
from train import train_loop
from utils.samplers import GroupByFilenameBatchSampler, UniqueFilenameBatchSampler


def set_seeds(seed=42):
    torch.manual_seed(seed); random.seed(seed)
    torch.backends.cudnn.benchmark = True


def build_loaders(cfg_data, tokenizer, seed):
    df = load_flickr30k(cfg_data["base_dir"], cfg_data["csv_name"])
    img_dir = os.path.join(cfg_data["base_dir"], cfg_data["images_subdir"])
   
    trn = df[df["split"] == cfg_data["train_split_name"]].reset_index(drop=True)
    evl = df[df["split"] == cfg_data["eval_split_name"]].reset_index(drop=True)
   
    # Simplified transforms & collators
    tf_train = build_image_transform(cfg_data["aug"], train=True)
    tf_eval = build_image_transform(cfg_data["aug"], train=False)
    col_train = Collator(tokenizer, tf_train, cfg_data["max_text_len"], epoch_rng=EpochRNG(seed))
    col_eval = Collator(tokenizer, tf_eval, cfg_data["max_text_len"])


    # Sampler selection
    SamplerCls = GroupByFilenameBatchSampler if CFG["train"]["multi_positive"] else UniqueFilenameBatchSampler
    sampler = SamplerCls(trn, batch_size=cfg_data["batch_size"], seed=seed)
   
    loader_train = DataLoader(
        Flickr30kCLIPDataset(trn, img_dir),
        batch_sampler=sampler,
        num_workers=cfg_data["num_workers"],
        collate_fn=col_train,
        pin_memory=cfg_data["pin_memory"]
    )
    loader_eval = DataLoader(
        Flickr30kCLIPDataset(evl, img_dir),
        batch_size=cfg_data["batch_size"],
        collate_fn=col_eval,
        num_workers=cfg_data["num_workers"]
    )
    return loader_train, loader_eval


def main():
    set_seeds(CFG["train"]["seed"])
    device, backend = pick_device()
    print(f"Starting training on {device}")


    tokenizer = build_tokenizer()
    loader_trn, loader_val = build_loaders(CFG["data"], tokenizer, CFG["train"]["seed"])
   
    # Inject tokenizer specifics into model config before build
    CFG["model"]["eos_token_id"] = tokenizer.eos_token_id
    model = CLIPLike(CFG["model"]) # Simplified build


    train_loop(model, loader_trn, loader_val, device, backend, CFG["train"])


if __name__ == "__main__":
    main()