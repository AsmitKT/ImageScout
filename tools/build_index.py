import os,json,argparse,torch,numpy as np
try:
    import faiss
except ImportError:
    raise SystemExit("FAISS is required. Install with: pip install faiss-cpu")
from config import CFG
from utils.devices import pick_device
from data.dataset import load_flickr30k,Flickr30kCLIPDataset
from data.collate import build_tokenizer,build_image_transform,Collator
from models.clip_model import CLIPLike


def _load_test_df():
    d=CFG["data"]
    base=d["base_dir"]
    df=load_flickr30k(base,d["csv_name"])
    return df[df["split"]==d["test_split_name"]].reset_index(drop=True)


def _to_np(x:torch.Tensor)->np.ndarray:
    return x.detach().cpu().numpy().astype("float32",copy=False)


def _l2n(a:np.ndarray)->np.ndarray:
    n=np.linalg.norm(a,axis=1,keepdims=True)+1e-10
    return a/n


def _ip_index(mat:np.ndarray):
    d=mat.shape[1]
    idx=faiss.IndexFlatIP(d)
    idx.add(mat)
    return idx


def _unique_images(V_all:torch.Tensor,names:list[str]):
    first={}
    keep=[]
    for i,n in enumerate(names):
        if n not in first:
            first[n]=i
            keep.append(i)
    V_u=V_all[keep].contiguous()
    name_list=[names[i] for i in keep]
    name_to_idx={n:j for j,n in enumerate(name_list)}
    return V_u,name_list,name_to_idx


def _load_model(device):
    tok=build_tokenizer()
    # UPDATED: Inject EOS token and pass full config dict
    CFG["model"]["eos_token_id"] = tok.eos_token_id
    model=CLIPLike(CFG["model"]).to(device)
    return model,tok


def _load_ckpt(model,device,ckpt_path:str):
    if not (ckpt_path.endswith(".pt") and os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"Checkpoint not found or not a .pt file: {ckpt_path}")
    state=torch.load(ckpt_path,map_location=device,weights_only=False)
    model.load_state_dict(state["model_state"],strict=True)
    model.eval()
    print(f"[ckpt] Loaded {ckpt_path}")


@torch.no_grad()
def _encode_batches(model,loader,device):
    model.eval()
    T_all,V_all,names,caps=[],[],[],[]
    for b in loader:
        ids=b["input_ids"].to(device)
        attn=b["attention_mask"].to(device)
        pix=b["pixel_values"].to(device)
        T=model.encode_text(ids,attn)
        V=model.encode_image(pix)
        T_all.append(T.cpu());V_all.append(V.cpu())
        names.extend(b["meta"]["filename"])
        caps.extend(b["meta"]["caption"])
    return torch.cat(T_all,0),torch.cat(V_all,0),names,caps


def main():
    p=argparse.ArgumentParser(description="Build FAISS indexes from CSV test split.")
    p.add_argument("--ckpt",required=True)
    args=p.parse_args()
    
    # Delegate to the function below to avoid code duplication
    build_index_from_app(args.ckpt)


# --- NEW: Wrapper for App Integration (uses existing functions) ---
def build_index_from_app(ckpt_path, model=None, tok=None, device=None):
    """
    Calls the original helper functions. 
    If model/tok/device are provided (from Flask), it skips reloading them.
    """
    out_dir = os.path.splitext(ckpt_path)[0]
    
    # 1. Setup Device
    if device is None:
        device, backend = pick_device()
        print("Using device:", device, "| backend:", backend)
    else:
        # We assume backend is available or irrelevant for just encoding
        backend = "unknown_app" 

    # 2. Setup Model (If not provided by App)
    if model is None:
        model, tok = _load_model(device)
        _load_ckpt(model, device, ckpt_path)
    
    # 3. Prepare Data
    d=CFG["data"]
    base=d["base_dir"]
    images=os.path.join(base,d["images_subdir"])
    df=_load_test_df()
    ds=Flickr30kCLIPDataset(df,images)
    tf=build_image_transform(CFG["data"]["aug"],train=False)
    coll=Collator(tok,tf,d["max_text_len"])
    loader=torch.utils.data.DataLoader(ds,batch_size=d["batch_size"],shuffle=False,
                                       num_workers=0 if os.name=="nt" else d["num_workers"],
                                       pin_memory=d["pin_memory"],persistent_workers=False,collate_fn=coll)
    
    # 4. Run Encoding (Uses original function)
    print("[index] Encoding data…")
    T_all,V_all,names,caps=_encode_batches(model,loader,device)
    
    os.makedirs(out_dir,exist_ok=True)
    
    # 5. Process Vectors (Uses original functions)
    V_u,img_names,name_to_idx=_unique_images(V_all,names)
    mat_img=_l2n(_to_np(V_u))
    idx_img=_ip_index(mat_img)
    faiss.write_index(idx_img,os.path.join(out_dir,"image.faiss"))
    
    mat_txt=_l2n(_to_np(T_all))
    idx_txt=_ip_index(mat_txt)
    faiss.write_index(idx_txt,os.path.join(out_dir,"text.faiss"))
    
    meta={
        "data_mode":"test",
        "backend": backend, # might be 'unknown_app' if passed from flask, but acceptable
        "dims":int(mat_img.shape[1]),
        "counts":{"images":int(mat_img.shape[0]),"texts":int(mat_txt.shape[0])},
        "limits":{"limit_images":0},
        "image":{"filenames":img_names},
        "text":{"filenames":names,"captions":caps},
        "image_row_of_filename":name_to_idx
    }
    with open(os.path.join(out_dir,"meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    print(f"[index] Wrote:\n- {os.path.join(out_dir,'image.faiss')}\n- {os.path.join(out_dir,'text.faiss')}\n- {os.path.join(out_dir,'meta.json')}")
    print(f"[index] Images indexed: {len(img_names)} | Texts indexed: {len(names)}")


if __name__=="__main__":
    main()