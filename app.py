import io, os
import streamlit as st
from PIL import Image
import numpy as np
import torch
import json


from src.models.clip_loader import load_clip
from src.infer.zero_shot import top1
from src.index.gallery_index import build_gallery_index
from src.vision.color import dominant_color_swatch, rgb2hex

st.set_page_config(page_title="Shoe-Recognition", page_icon="|", layout="wide")
st.title("Shoe Recognition & Similarity")

# keep state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.preprocess = None
if "gallery_paths" not in st.session_state:
    st.session_state.gallery_paths, st.session_state.gallery_feats, st.session_state.nn = [], None, None

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Setup")
    if st.button("1) Load model", type="primary", use_container_width=True):
        with st.spinner("Loading CLIP… (first time downloads weights)"):
            m, tok, pre = load_clip()
            st.session_state.model, st.session_state.tokenizer, st.session_state.preprocess = m, tok, pre
        st.success("Model ready ✅")

    if st.button("2) Build/Refresh gallery index", use_container_width=True):
        if st.session_state.model is None:
            st.warning("Load the model first.")
        else:
            with st.spinner("Indexing ./gallery …"):
                p, f, nn = build_gallery_index("gallery", st.session_state.model, st.session_state.preprocess)
                st.session_state.gallery_paths, st.session_state.gallery_feats, st.session_state.nn = p, f, nn
            st.success(f"Indexed {len(st.session_state.gallery_paths)} images ✅")
    st.caption("Tip: put ~40 JPG/PNG shoes in ./gallery before indexing.")

# ---- Labels ----
CATEGORIES = ["sneaker","boot","sandal","loafer","heel"]
CLOSURE   = ["lace-up","slip-on","zip","buckle","velcro"]
TOE       = ["round toe","pointed toe","square toe","open toe"]
MATERIAL  = ["leather","suede","textile","mesh","synthetic"]
COLORS    = ["black","white","brown","beige","blue","red","green","grey"]

# ---- Main ----
left, right = st.columns([1,1.2], vertical_alignment="top")
with left:
    st.subheader("Upload a shoe image")
    up = st.file_uploader("JPG/PNG only", type=["jpg","jpeg","png"])
    pil = None
    if up: 
        pil = Image.open(io.BytesIO(up.read())).convert("RGB")
        st.image(pil, caption="Input", use_container_width=True)

with right:
    st.subheader("Status")
    st.write(f"Model loaded: **{st.session_state.model is not None}**")
    st.write(f"Indexed images: **{len(st.session_state.gallery_paths)}**")

st.markdown("---")

# Inference only when ready
if st.session_state.model is not None and pil is not None:
    m, tok, pre = st.session_state.model, st.session_state.tokenizer, st.session_state.preprocess
    with st.spinner("Predicting attributes…"):
        cat, p_cat, _ = top1(pil, CATEGORIES, "a product photo of a", m, tok, pre)
        clo, _, _    = top1(pil, CLOSURE,   "a", m, tok, pre)
        toe, _, _    = top1(pil, TOE,       "a", m, tok, pre)
        mat, _, _    = top1(pil, MATERIAL,  "made of", m, tok, pre)
        colg, _, _   = top1(pil, COLORS,    "a", m, tok, pre)
        dom_rgb      = dominant_color_swatch(pil, k=3)

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("### Predicted attributes")
        st.write(f"- **Category:** {cat} ({p_cat:.2f})")
        st.write(f"- **Closure:** {clo}")
        st.write(f"- **Toe:** {toe}")
        st.write(f"- **Material:** {mat}")
        st.write(f"- **Color (pred):** {colg}")
    with c2:
        st.markdown("### Dominant color (pixel)")
        st.color_picker(" ", value=rgb2hex(dom_rgb), key="cp", label_visibility="collapsed")

    st.markdown("---")
    st.subheader("Looks-like search")
    if st.session_state.nn is None or len(st.session_state.gallery_paths) == 0:
        st.info("Build the gallery index in the sidebar to enable similarity search.")
    else:
        with st.spinner("Searching neighbors…"):
            with torch.no_grad():
                xq = pre(pil).unsqueeze(0)
                if torch.cuda.is_available(): xq = xq.cuda()
                fq = m.encode_image(xq)
                fq = fq / fq.norm(dim=-1, keepdim=True)
                fq = fq.cpu().numpy()
            dists, idxs = st.session_state.nn.kneighbors(fq, return_distance=True)
                        # --- Download JSON button (attributes + neighbors) ---
            result = {
                "category": {"label": cat, "confidence": round(p_cat, 4)},
                "closure": clo,
                "toe": toe,
                "material": mat,
                "color_pred": colg,
                "dominant_color_hex": rgb2hex(dom_rgb),
                "neighbors": [
                    {
                        "path": st.session_state.gallery_paths[ii],
                        "cosine_distance": float(di)
                    }
                    for di, ii in zip(dists[0], idxs[0])
                ],
            }
            st.download_button(
                "Download results (JSON)",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name="shoesnap_result.json",
                mime="application/json"
            )

        grid = st.columns(4)
        for i, (di, ii) in enumerate(zip(dists[0], idxs[0])):
            with grid[i % 4]:
                path = st.session_state.gallery_paths[ii]
                st.image(path, use_container_width=True, caption=f"{os.path.basename(path)} | dist {di:.3f}")
elif pil is not None and st.session_state.model is None:
    st.warning("Click **Load model** in the sidebar first.")
