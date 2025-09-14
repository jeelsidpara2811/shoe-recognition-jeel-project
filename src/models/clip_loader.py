# src/models/clip_loader.py
import os, torch, open_clip

CACHE_DIR = r"C:\hf-cache-openclip"  
os.makedirs(CACHE_DIR, exist_ok=True)

CANDIDATES = [
    ("RN50", "laion400m_e32"),          
    ("RN50", "yfcc15m"),                 
    ("ViT-B-32", "laion2b_s34b_b79k"),   
]

def load_clip():
    last_err = None
    for arch, tag in CANDIDATES:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=tag, cache_dir=CACHE_DIR
            )
            tokenizer = open_clip.get_tokenizer(arch)
            model.eval().to("cpu")   # force CPU, avoid GPU OOM
            return model, tokenizer, preprocess
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any CLIP model: {last_err}")
