#Major\train.py
import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from utils.schedule import TwoPhaseCosineEpochsScaled
from utils.devices import AMPContext, get_grad_scaler


def _recall_at_k(logits, same_mask, ks=(1, 5, 10)):
    # Simply calculation for Recall@K
    out = {"t2i": {}, "i2t": {}}
    # Text-to-Image
    vals, idx = torch.topk(logits, k=max(ks), dim=1)
    rel = torch.gather(same_mask.float(), 1, idx).bool()
    for k in ks: out["t2i"][k] = rel[:, :k].any(dim=1).float().mean().item()
    # Image-to-Text
    valsT, idxT = torch.topk(logits.t(), k=max(ks), dim=1)
    relT = torch.gather(same_mask.t().float(), 1, idxT).bool()
    for k in ks: out["i2t"][k] = relT[:, :k].any(dim=1).float().mean().item()
    return out


def evaluate_epoch(model, loader, device):
    if not loader: return
    model.eval()
    Ts, Vs, Names = [], [], []
    with torch.no_grad():
        for batch in loader:
            T = model.encode_text(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            V = model.encode_image(batch["pixel_values"].to(device))
            Ts.append(T.cpu()); Vs.append(V.cpu()); Names.extend(batch["meta"]["filename"])
   
    T, V = torch.cat(Ts), torch.cat(Vs)
    logits = (model.logit_scale.exp().cpu() * (T @ V.t()))
    same = torch.tensor([[1 if Names[i]==Names[j] else 0 for j in range(len(Names))] for i in range(len(Names))], dtype=torch.bool)
    res = _recall_at_k(logits, same)
    print(f"[Eval] T2I R@1: {res['t2i'][1]:.4f} | I2T R@1: {res['i2t'][1]:.4f}")


def train_loop(model, loader_train, loader_eval, device, backend, cfg):
    ckpt_path = os.path.join(cfg["ckpt_dir"], "final.pt")
   
    # Simplified Resume: Check only for final.pt
    start_epoch = 0
    opt = AdamW(model.parameters(), lr=cfg["lr"], betas=cfg["betas"], weight_decay=cfg["weight_decay"])
    scaler = get_grad_scaler(backend, enabled=cfg["fp16"])


    if cfg.get("resume") and os.path.exists(ckpt_path):
        print(f"[Resume] Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        # We treat 'final.pt' as a finished state, but if you want to continue:
        # opt.load_state_dict(ckpt["optimizer_state"])
        # start_epoch = ckpt["epoch"]
        print("Weights loaded.")


    model.to(device).train()
    grad_accum = cfg["grad_accum"]
    sched = TwoPhaseCosineEpochsScaled(p1_epochs=150, p1_warmup=5, p2_epochs=12, p2_warmup=2)


    for epoch in range(start_epoch, cfg["epochs"]):
        if hasattr(loader_train, "batch_sampler"): loader_train.batch_sampler.set_epoch(epoch)
        pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
       
        for i, batch in enumerate(pbar):
            with AMPContext(backend, enabled=cfg["fp16"]):
                out = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["pixel_values"].to(device)
                )
                loss = out["loss"] / grad_accum
           
            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()


            if (i + 1) % grad_accum == 0:
                # LR Schedule Update
                mult = sched.lr_mult(epoch, i, len(loader_train))
                for pg in opt.param_groups: pg["lr"] = float(cfg["lr"]) * mult
               
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    opt.step()
                opt.zero_grad()
                pbar.set_postfix({"loss": f"{loss.item()*grad_accum:.4f}", "lr": f"{opt.param_groups[0]['lr']:.2e}"})


        if (epoch + 1) % cfg.get("eval_interval_epochs", 1) == 0:
            evaluate_epoch(model, loader_eval, device)
            model.train()


    # Save ONLY at the end
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_state": opt.state_dict(),
        "epoch": cfg["epochs"]
    }, ckpt_path)
    print(f"Training complete. Saved to {ckpt_path}")