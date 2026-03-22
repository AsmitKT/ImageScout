#Major\tools\manual_query.py
import os,argparse,json,numpy as np,torch,matplotlib.pyplot as plt,textwrap
from typing import Tuple,List
import faiss
from PIL import Image
from config import CFG
from utils.devices import pick_device
from data.collate import build_tokenizer, build_image_transform
from models.clip_model import CLIPLike

def _load_indexes(index_dir:str):
    img_path=os.path.join(index_dir,"image.faiss")
    meta_path=os.path.join(index_dir,"meta.json")
    if not (os.path.exists(img_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Missing FAISS files in {index_dir}. Expected image.faiss, meta.json")
    idx_img=faiss.read_index(img_path)
    with open(meta_path,"r",encoding="utf-8") as f:
        meta=json.load(f)
    return idx_img,meta

def _load_model_and_tokenizer(device:torch.device):
    tok=build_tokenizer()
    # Inject EOS token and pass full config dict
    CFG["model"]["eos_token_id"] = tok.eos_token_id
    model=CLIPLike(CFG["model"]).to(device)
    return model,tok

def _load_ckpt(model:CLIPLike,device:torch.device,ckpt_path:str):
    if not (ckpt_path.endswith(".pt") and os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"Checkpoint not found or not a .pt file: {ckpt_path}")
    state=torch.load(ckpt_path,map_location=device,weights_only=False)
    model.load_state_dict(state["model_state"],strict=True)
    print(f"[ckpt] Loaded {ckpt_path}")

@torch.no_grad()
def _encode_text(model:CLIPLike,tokenizer,text:str,device:torch.device)->np.ndarray:
    toks=tokenizer([text],padding=True,truncation=True,max_length=CFG["data"]["max_text_len"],return_tensors="pt")
    T=model.encode_text(toks.input_ids.to(device),toks.attention_mask.to(device)).cpu()
    v=T.numpy().astype("float32",copy=False)
    v/=np.linalg.norm(v,axis=1,keepdims=True)+1e-10
    return v

@torch.no_grad()
def _encode_image_query(model:CLIPLike, img_transform, image_path:str, device:torch.device)->np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Query image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    # Apply the same transform used during validation/testing
    pixel_values = img_transform(img).unsqueeze(0).to(device)
    
    V = model.encode_image(pixel_values).cpu()
    v = V.numpy().astype("float32", copy=False)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
    return v

def _abs_image_path(rel_or_name:str,images_root:str)->str:
    if os.path.isabs(rel_or_name):
        return rel_or_name
    return os.path.abspath(os.path.join(images_root,rel_or_name))

def _open_rgb(path:str)->Image.Image:
    return Image.open(path).convert("RGB")

def _wrap(s:str,width:int=40)->str:
    return "\n".join(textwrap.wrap(s,width=width)) if s else s

def _show_t2i(text_query:str,I:np.ndarray,D:np.ndarray,meta:dict,topk:int,images_root:str):
    names=meta["image"]["filenames"]
    K=min(topk,I.shape[1])
    cols=min(5,K)
    rows=int(np.ceil(K/cols))
    fig=plt.figure(figsize=(12.0,8.0),dpi=100)
    fig.suptitle(f'Text → Image | Query: "{text_query}"',y=0.98)
    for k in range(K):
        rid=int(I[0][k])
        score=float(D[0][k])
        fn=names[rid]
        path=_abs_image_path(fn,images_root)
        try:
            img=_open_rgb(path)
        except Exception:
            img=Image.new("RGB",(CFG["model"]["image_size"],CFG["model"]["image_size"]),(220,220,220))
            fn=f"{fn} [missing]"
        ax=fig.add_subplot(rows,cols,k+1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(_wrap(f"#{k+1} • {score:.3f}\n{fn}",width=40),fontsize=9)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show(block=True)

def _show_i2i(query_img_path:str,I:np.ndarray,D:np.ndarray,meta:dict,topk:int,images_root:str):
    names=meta["image"]["filenames"]
    K=min(topk,I.shape[1])
    
    # +1 column for the query image
    cols=min(5,K) + 1 
    rows=int(np.ceil((K+1)/cols)) # Adjust rows if wrapping occurs
    
    fig=plt.figure(figsize=(14.0, 6.0),dpi=100)
    fig.suptitle(f'Image → Image | Query: {os.path.basename(query_img_path)}',y=0.98)
    
    # Plot Query Image first
    ax_q = fig.add_subplot(1, cols, 1)
    try:
        q_img = _open_rgb(query_img_path)
        ax_q.imshow(q_img)
        ax_q.set_title("QUERY IMAGE", fontsize=10, fontweight='bold', color='blue')
    except Exception:
        ax_q.text(0.5, 0.5, "Query Image\nNot Found", ha='center', va='center')
    ax_q.axis("off")

    # Plot Results
    for k in range(K):
        rid=int(I[0][k])
        score=float(D[0][k])
        fn=names[rid]
        path=_abs_image_path(fn,images_root)
        try:
            img=_open_rgb(path)
        except Exception:
            img=Image.new("RGB",(CFG["model"]["image_size"],CFG["model"]["image_size"]),(220,220,220))
            fn=f"{fn} [missing]"
            
        # Offset index by 2 because subplot 1 is query
        ax=fig.add_subplot(1, cols, k+2)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(_wrap(f"#{k+1} • {score:.3f}\n{fn}",width=25),fontsize=8)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show(block=True)

def main():
    p=argparse.ArgumentParser(description="Manual FAISS query; t2i and i2i modes.")
    p.add_argument("--ckpt",required=True)
    p.add_argument("--mode", choices=["t2i", "i2i"], default="t2i", help="Search mode")
    p.add_argument("--text",type=str,default=None, help="Query text (for t2i)")
    p.add_argument("--image",type=str,default=None, help="Path to query image (for i2i)")
    args=p.parse_args()

    topk = 10
    index_dir = os.path.splitext(args.ckpt)[0]

    device,backend=pick_device()
    print("Using device:",device,"| backend:",backend)
    model,tok=_load_model_and_tokenizer(device)
    _load_ckpt(model,device,args.ckpt)
    model.eval()
    
    idx_img,meta=_load_indexes(index_dir)
    images_root=os.path.join(CFG["data"]["base_dir"],CFG["data"]["images_subdir"])
    
    if args.mode == "t2i":
        if not args.text:
            raise SystemExit("For t2i mode you must pass --text")
        print(f"[Query] Text: '{args.text}'")
        q=_encode_text(model,tok,args.text,device)
        D,I=idx_img.search(q,max(1,topk))
        _print_results(I, D, meta, topk)
        _show_t2i(args.text,I,D,meta,topk,images_root)

    elif args.mode == "i2i":
        if not args.image:
            raise SystemExit("For i2i mode you must pass --image")
        print(f"[Query] Image: '{args.image}'")
        
        # Build transform (Val mode)
        tf = build_image_transform(CFG["data"]["aug"], train=False)
        q=_encode_image_query(model, tf, args.image, device)
        
        D,I=idx_img.search(q,max(1,topk))
        _print_results(I, D, meta, topk)
        _show_i2i(args.image,I,D,meta,topk,images_root)

def _print_results(I, D, meta, topk):
    print("\nTop results:")
    for k in range(min(topk,I.shape[1])):
        rid=int(I[0][k]);score=float(D[0][k]);fn=meta["image"]["filenames"][rid]
        print(f"  #{k+1}: score={score:.4f} file={fn}")

if __name__=="__main__":
    main()