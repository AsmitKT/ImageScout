#Major\tools\vmgq.py
import os, argparse, json, math, gc, torch
import torch.nn as nn
from config import CFG
from utils.devices import pick_device
from data.collate import build_tokenizer
from models.clip_model import CLIPLike
from tools.eval_model import eval_text_to_image


def get_kind(n, m):
    if not isinstance(m, (nn.Linear, nn.Conv2d)): return "other"
    if "text_encoder" in n: return "text_attn" if "qkv" in n or "proj" in n else "text_mlp"
    if "vision_encoder" in n:
        if "patch" in n: return "vision_patch"
        return "vision_attn" if "qkv" in n or "proj" in n else "vision_mlp"
    return "proj_head" if "proj" in n else "other"


def fake_quant(w, bits):
    if bits >= 32: return w
    flat = w.view(w.size(0), -1)
    eps = 1e-8
    if bits == 1:
        s = flat.abs().amax(1, keepdim=True).clamp(min=eps)
        q = (flat >= 0).float() * 2 - 1
    else:
        levels = (1 << (bits - 1)) - 1
        s = flat.abs().amax(1, keepdim=True).clamp(min=eps) / levels
        q = (flat / s).round().clamp(-levels, levels)
    return (q * s).view_as(w)


def get_metrics(path, dev, idx):
    jpath = os.path.join(os.path.splitext(path)[0], "eval_t2i_K10.json")
    if os.path.exists(jpath):
        print(f"[Cache] Found {jpath}"); return json.load(open(jpath))
    print(f"[Compute] Eval {path}"); return eval_text_to_image(10, dev, path, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--traj_ckpts", nargs="+", required=True)
    ap.add_argument("--index_dir", required=True); ap.add_argument("--metric_mode", default="recall")
    ap.add_argument("--bits_grid", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max_drop", type=float, default=0.02)
    ap.add_argument("--alpha_min", type=float, default=0.1); ap.add_argument("--alpha_max", type=float, default=0.4)
    args = ap.parse_args()


    # 1. Setup & Model
    dev, backend = pick_device()
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base = raw.get("model_state", raw)
   
    # UPDATED: Init Dummy Model with config injection
    tok = build_tokenizer(); mcfg = CFG["model"]
    mcfg["eos_token_id"] = tok.eos_token_id
    model = CLIPLike(mcfg)
   
    modules = {n: get_kind(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}
    del model; gc.collect()


    # 2. Metrics & Weights (Using Cache & Floor Logic)
    traj_paths = list(dict.fromkeys(args.traj_ckpts + [args.ckpt]))
    met = {p: get_metrics(p, dev, args.index_dir) for p in traj_paths}
   
    key = "R@5" if args.metric_mode == "recall" else "STS@K"
    get_s = lambda m: float(m.get(key, m.get("RWCSTS@K", 0.0))) # Handle STS fallback
    S_vals = [get_s(met[p]) for p in traj_paths]
    S_full = get_s(met[args.ckpt])
   
    S_floor = S_full * (1.0 - args.max_drop) if S_full > 0 else S_full
    w_denom = max(S_vals) - S_floor + 1e-8
    weights = [max(0.0, (s - S_floor) / w_denom) for s in S_vals]
    w_norm = torch.tensor(weights).float(); w_norm /= (w_norm.sum() + 1e-8)


    # 3. Trajectory Stats (Median Var, Sequential Mov)
    stats, all_M, states = {}, [], [torch.load(p, map_location="cpu", weights_only=False).get("model_state", torch.load(p, map_location="cpu", weights_only=False)) for p in traj_paths]
   
    for name in modules:
        key = name + ".weight"
        W = torch.stack([s[key].float() for s in states if key in s])
        if len(W) < 2: continue


        w_v = w_norm.to(W.device).view(-1, *([1]*(W.ndim-1)))
        mu = (w_v * W).sum(0)
        sigma2 = (w_v * (W - mu.unsqueeze(0))**2).sum(0).median().item()


        M = torch.diff(W, dim=0).abs().sum().item()
        R = W[-1].abs().max().item()
       
        stats[name] = {"sigma2": sigma2, "M": M, "R": R}; all_M.append(M)


    # 4. Bit Assignment
    M_med = float(torch.median(torch.tensor(all_M))) if all_M else 1.0
    bits_map, bits_sort = {}, sorted(set(args.bits_grid))
   
    for n, s in stats.items():
        alpha = args.alpha_min + (args.alpha_max - args.alpha_min) * min(1.0, max(0.0, s["M"] / (M_med + 1e-8)))
        b_req = math.ceil(math.log2((2 * s["R"]) / (math.sqrt(12 * alpha * max(s["sigma2"], 1e-12)))))
        bits_map[n] = next((b for b in bits_sort if b >= b_req), bits_sort[-1])


    # 5. Apply & Save
    new_state = {k: (fake_quant(v.float(), bits_map[k.replace(".weight","")]) if k.replace(".weight","") in bits_map else v) for k, v in base.items()}
    base_out = dict(raw); base_out["model_state"] = new_state
   
    out_base = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_pt = os.path.join(os.path.dirname(args.ckpt), f"VMGQ_{args.metric_mode}_{out_base}.pt")
    torch.save(base_out, out_pt)
   
    q_met = get_metrics(out_pt, dev, args.index_dir)
    meta = {
        "method": "VMGQ_Min", "source": args.ckpt, "metrics_quant": q_met, "conceptual_bits": sum(base[k].numel() * bits_map.get(k.replace(".weight",""), 32) for k in base if ".weight" in k),
        "module_bits": bits_map, "metric_floor": S_floor, "global_M_med": M_med
    }
    with open(out_pt.replace(".pt", ".json"), "w") as f: json.dump(meta, f, indent=2)
    print(f"Saved: {out_pt}")


if __name__ == "__main__": main()