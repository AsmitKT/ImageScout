#Major\data\dataset.py
import os,ast
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def _assert_exists(path:str,hint:str=""):
    if not os.path.exists(path):
        msg=f"Path not found: {path}"
        if hint: msg+=f"\nHint: {hint}"
        parent=os.path.dirname(path) or "."
        if os.path.isdir(parent):
            try:
                listing="\n".join(sorted(os.listdir(parent)))
                msg+=f"\nDirectory listing of {parent}:\n{listing}"
            except Exception:
                pass
        raise FileNotFoundError(msg)


def load_flickr30k(base_dir:str,csv_name="captions.csv"):
    _assert_exists(base_dir,"Set CFG['data']['base_dir'] to your dataset folder.")
    csv_path=os.path.join(base_dir,csv_name)
    _assert_exists(csv_path,"Check CFG['data']['csv_name'].")
    df=pd.read_csv(csv_path)
    req=["raw","filename","split"]
    for k in req:
        if k not in df.columns:
            raise ValueError(f"CSV must contain columns {req}. Found {list(df.columns)}")
    try:
        df["captions"]=df["raw"].apply(ast.literal_eval)
    except Exception as e:
        raise ValueError("Failed to parse 'raw' as list of captions.") from e
    if "sentids" in df.columns:
        df["sentids"]=df["sentids"].apply(ast.literal_eval)
    rows=[]
    for _,r in df.iterrows():
        caps=r["captions"] if isinstance(r["captions"],(list,tuple)) else [str(r["captions"])]
        for j,cap in enumerate(caps):
            rows.append({
                "filename":r["filename"],
                "caption":cap,
                "img_id":r.get("img_id",None),
                "sentid":r["sentids"][j] if "sentids" in df.columns and isinstance(r["sentids"],(list,tuple)) and len(r["sentids"])>j else None,
                "split":r["split"]
            })
    return pd.DataFrame(rows)


class Flickr30kCLIPDataset(Dataset):
    def __init__(self,df_cap,images_dir:str):
        self.df=df_cap.reset_index(drop=True)
        self.images_dir=images_dir
        _assert_exists(self.images_dir,"Expected an 'images' folder inside base_dir.")
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]
        img_path=os.path.join(self.images_dir,r["filename"])
        _assert_exists(img_path,"Is the 'filename' correct and file present in images/?")
        img=Image.open(img_path).convert("RGB")
        return{
            "caption":r["caption"],
            "image":img,
            "filename":r["filename"],
            "img_id":r.get("img_id",None),
            "sentid":r.get("sentid",None)
        }