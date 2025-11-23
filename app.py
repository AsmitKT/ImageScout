import os
import json
import torch
import numpy as np
import faiss
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

# --- Project Imports ---
from config import CFG
from utils.devices import pick_device
from models.clip_model import CLIPLike
from data.collate import build_tokenizer, build_image_transform

# --- Tool Imports ---
from tools.build_index import build_index_from_app
from tools.manual_query import _load_indexes

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHECKPOINT_DIR'] = 'checkpoints'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHECKPOINT_DIR'], exist_ok=True)

# --- Global State ---
STATE = {
    "model": None,
    "tokenizer": None,
    "index_img": None,
    "meta": None,
    "device": None,
    "current_ckpt_name": None
}

def load_model_globally(ckpt_filename):
    ckpt_path = os.path.join(app.config['CHECKPOINT_DIR'], ckpt_filename)
    device, _ = pick_device()
    STATE["device"] = device
    
    tok = build_tokenizer()
    CFG["model"]["eos_token_id"] = tok.eos_token_id
    model = CLIPLike(CFG["model"]).to(device)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    
    STATE["model"] = model
    STATE["tokenizer"] = tok
    STATE["current_ckpt_name"] = ckpt_filename
    print(f"[System] Model loaded: {ckpt_filename}")

# --- Routes ---

@app.route('/')
def index():
    ckpt_dir = app.config['CHECKPOINT_DIR']
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')] if os.path.exists(ckpt_dir) else []
    return render_template('index.html', ckpts=ckpts)

@app.route('/load_checkpoint', methods=['POST'])
def load_checkpoint():
    ckpt_filename = request.form.get('ckpt')
    if not ckpt_filename:
        return redirect(url_for('index'))
    
    ckpt_path = os.path.join(app.config['CHECKPOINT_DIR'], ckpt_filename)
    index_dir = os.path.splitext(ckpt_path)[0]
    
    try:
        # 1. Load Model (into RAM)
        load_model_globally(ckpt_filename)
        
        # 2. Check Index
        if not (os.path.exists(os.path.join(index_dir, "image.faiss")) and 
                os.path.exists(os.path.join(index_dir, "meta.json"))):
            
            flash("Index not found. Creating new FAISS index... (Check console)")
            
            # Use the new wrapper in build_index.py without reloading model
            build_index_from_app(
                ckpt_path=ckpt_path,
                model=STATE["model"],
                tok=STATE["tokenizer"],
                device=STATE["device"]
            )
        
        # 3. Load Index (Using function from manual_query.py)
        idx_img, meta = _load_indexes(index_dir)
        STATE["index_img"] = idx_img
        STATE["meta"] = meta
        
        return redirect(url_for('search_page'))
    except Exception as e:
        flash(f"Error: {str(e)}")
        print(e)
        return redirect(url_for('index'))

@app.route('/search', methods=['GET', 'POST'])
def search_page():
    if STATE["model"] is None:
        return redirect(url_for('index'))
        
    results = []
    query_display = None
    mode = "t2i"

    if request.method == 'POST':
        mode = request.form.get('mode')
        topk = 10
        device = STATE["device"]
        model = STATE["model"]
        index = STATE["index_img"]
        meta = STATE["meta"]
        
        query_vec = None
        
        if mode == 't2i':
            text = request.form.get('text_query')
            query_display = text
            if text:
                tok = STATE["tokenizer"]
                tokens = tok([text], padding=True, truncation=True, max_length=CFG["data"]["max_text_len"], return_tensors="pt")
                with torch.no_grad():
                    T = model.encode_text(tokens.input_ids.to(device), tokens.attention_mask.to(device)).cpu()
                v = T.numpy().astype("float32")
                v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
                query_vec = v
                
        elif mode == 'i2i':
            if 'image_query' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['image_query']
            if file.filename != '':
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                query_display = filename
                
                tf = build_image_transform(CFG["data"]["aug"], train=False)
                img = Image.open(path).convert("RGB")
                pixel_values = tf(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    V = model.encode_image(pixel_values).cpu()
                v = V.numpy().astype("float32")
                v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
                query_vec = v

        if query_vec is not None:
            D, I = index.search(query_vec, topk)
            filenames = meta["image"]["filenames"]
            for k in range(topk):
                rid = int(I[0][k])
                score = float(D[0][k])
                fn = filenames[rid]
                results.append({"rank": k+1, "score": f"{score:.3f}", "filename": fn})

    return render_template('search.html', 
                           results=results, 
                           mode=mode, 
                           query_display=query_display,
                           ckpt_name=STATE["current_ckpt_name"])

@app.route('/images/<path:filename>')
def serve_image(filename):
    images_root = os.path.join(CFG["data"]["base_dir"], CFG["data"]["images_subdir"])
    return send_from_directory(images_root, filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)