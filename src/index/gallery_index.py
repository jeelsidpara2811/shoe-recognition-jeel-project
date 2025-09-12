import os
import glob
import json
import hashlib

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch


def _dir_fingerprint(paths):
    s = "|".join([f"{os.path.basename(p)}:{os.path.getsize(p)}" for p in paths])
    return hashlib.md5(s.encode()).hexdigest()

def build_gallery_index(gallery_dir, model, preprocess):
    os.makedirs("cache", exist_ok=True)
    all_files = glob.glob(os.path.join(gallery_dir, "*"))
    paths = sorted([p for p in all_files if os.path.isfile(p)])
    if not paths: 
        return [], np.zeros((0,512)), None

    fp = _dir_fingerprint(paths)
    feats_p = f"cache/{fp}_feats.npy"
    paths_p = f"cache/{fp}_paths.json"

    if os.path.exists(feats_p) and os.path.exists(paths_p):
        feats = np.load(feats_p)
        with open(paths_p, "r", encoding="utf-8") as f: paths = json.load(f)
    else:
        feats = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            with torch.no_grad():
                x = preprocess(img).unsqueeze(0)
                if torch.cuda.is_available(): x = x.cuda()
                f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy()[0])
        feats = np.vstack(feats) if feats else np.zeros((0,512))
        np.save(feats_p, feats); json.dump(paths, open(paths_p, "w", encoding="utf-8"))

    nn = None
    if len(feats) > 0:
        nn = NearestNeighbors(n_neighbors=min(8, len(paths)), metric="cosine").fit(feats)
    return paths, feats, nn
