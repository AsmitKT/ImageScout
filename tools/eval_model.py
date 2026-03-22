# Major\tools\eval_model.py
import os, json, argparse, math, datetime, numpy as np, torch
try: import faiss
except ImportError: raise SystemExit("FAISS required: pip install faiss-cpu")


from config import CFG
from utils.devices import pick_device
from data.dataset import load_flickr30k, Flickr30kCLIPDataset
from data.collate import build_tokenizer, build_image_transform, Collator
from models.clip_model import CLIPLike


# --- Shared Helpers (Internal) ---
def _l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
def _to_np(x): return x.detach().cpu().numpy().astype("float32")


def _compute_ranking_metrics(is_rel, K, total_rel):
    k_slice = is_rel[:K]
    hits = sum(k_slice)
   
    # AP
    ap, run_hits = 0.0, 0
    for i, r in enumerate(k_slice, 1):
        if r: run_hits += 1; ap += run_hits / i
    ap = ap / min(total_rel, K) if total_rel > 0 else 0.0
   
    # NDCG
    dcg = sum((r / math.log2(i + 2)) for i, r in enumerate(k_slice))
    idcg = sum((1 / math.log2(i + 2)) for i in range(min(total_rel, K)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
   
    return ap, ndcg, (K - hits) / float(K) if K > 0 else 0.0


def _compute_rw_centroid_sts(query_vec, retrieved_fns, centroids, K):
    """Computes Rank-Weighted Centroid STS using reciprocal rank weights (1/r)."""
    sims = []
    for fn in retrieved_fns[:K]:
        cen = centroids.get(fn)
        if cen is not None:
            c_n = cen / (cen.norm(dim=-1, keepdim=True) + 1e-10)
            q_n = query_vec / (query_vec.norm(dim=-1, keepdim=True) + 1e-10)
            sims.append(float(torch.mm(q_n, c_n.t()).item()))
        else: sims.append(0.0)
   
    if not sims: return 0.0
    # Weighting change: Use 1/(rank) instead of 1/log(rank)
    w = [1.0 / (i + 1) for i in range(len(sims))]
    return sum(wi * s for wi, s in zip(w, sims)) / sum(w)


def _load_resources(device, ckpt_path, index_dir):
    tok = build_tokenizer()
   
    # UPDATED: Inject EOS token and pass full config dict
    CFG["model"]["eos_token_id"] = tok.eos_token_id
    model = CLIPLike(CFG["model"])
   
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()


    idx_img = faiss.read_index(os.path.join(index_dir, "image.faiss"))
    with open(os.path.join(index_dir, "meta.json"), "r") as f: meta = json.load(f)


    d_cfg = CFG["data"]
    df = load_flickr30k(d_cfg["base_dir"], d_cfg["csv_name"])
    df = df[df["split"] == d_cfg["test_split_name"]].reset_index(drop=True)
    df = df[df["filename"].isin(set(meta["image"]["filenames"]))]
   
    ds = Flickr30kCLIPDataset(df, os.path.join(d_cfg["base_dir"], d_cfg["images_subdir"]))
    coll = Collator(tok, build_image_transform(CFG["data"]["aug"], train=False), d_cfg["max_text_len"])
    # FORCE num_workers to 0 to prevent Windows OOM crash
    loader = torch.utils.data.DataLoader(ds, batch_size=d_cfg["batch_size"], num_workers=0, collate_fn=coll)
   
    return model, tok, loader, idx_img, meta


@torch.inference_mode()
def _get_text_centroids(model, tokenizer, meta, device):
    captions = meta["text"]["captions"]
    all_embs = []
    for i in range(0, len(captions), 64):
        batch = captions[i:i+64]
        t = tokenizer(batch, padding=True, truncation=True, max_length=CFG["data"]["max_text_len"], return_tensors="pt")
        all_embs.append(model.encode_text(t.input_ids.to(device), t.attention_mask.to(device)).cpu())
   
    T_all = torch.cat(all_embs, 0)
    centroids, txt_rows_map = {}, {}
    for rid, fn in enumerate(meta["text"]["filenames"]):
        txt_rows_map.setdefault(fn, []).append(rid)
    for fn, rows in txt_rows_map.items():
        centroids[fn] = T_all[rows].mean(dim=0, keepdim=True)
    return centroids, txt_rows_map


# --- Main Exported Function ---


@torch.inference_mode()
def eval_text_to_image(K, device, ckpt_path, index_dir):
    model, tok, loader, idx_img, meta = _load_resources(device, ckpt_path, index_dir)
    comp_dev = torch.device("cpu") if device.type == "xpu" else device
    model.to(comp_dev)
   
    centroids, txt_rows_map = _get_text_centroids(model, tok, meta, comp_dev)
   
    hist = {k: [] for k in ["R@1", "R@5", "R@10", "MAP@K", "NDCG@K", "RWCSTS@K", "HardNegRate@K"]}
   
    print("[t2i] Running retrieval...")
    for b in loader:
        q_vecs = model.encode_text(b["input_ids"].to(comp_dev), b["attention_mask"].to(comp_dev)).cpu()
        gold_fns = b["meta"]["filename"]
       
        D, I = idx_img.search(_l2n(_to_np(q_vecs)), max(10, K))
       
        for i, (ret_ids, gold) in enumerate(zip(I, gold_fns)):
            ret_fns = [meta["image"]["filenames"][rid] for rid in ret_ids]
            hits_arr = [1 if fn == gold else 0 for fn in ret_fns]
           
            for rk in [1, 5, 10]: hist[f"R@{rk}"].append(1 if sum(hits_arr[:rk]) > 0 else 0)
           
            ap, ndcg, hard = _compute_ranking_metrics(hits_arr, K, 1)
            hist["MAP@K"].append(ap); hist["NDCG@K"].append(ndcg); hist["HardNegRate@K"].append(hard)
           
            hist["RWCSTS@K"].append(_compute_rw_centroid_sts(q_vecs[i:i+1], ret_fns, centroids, K))


    res = {"task": "text->image", "K": K, "queries": len(hist["MAP@K"])}
    for k in hist:
        res[k] = float(np.round(np.mean(hist[k]) * 100.0, 2)) if hist[k] else 0.0
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()


    K = 10
    index_dir = os.path.splitext(args.ckpt)[0]


    device, _ = pick_device()
    res = eval_text_to_image(K, device, args.ckpt, index_dir)


    out_name = f"eval_t2i_K{K}.json"
    out_path = os.path.join(index_dir, out_name)
    with open(out_path, "w") as f: json.dump(res, f, indent=2)
    print(f"[eval] Saved: {out_path}\n", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()