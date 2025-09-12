# Shoe Recognition with CLIP + Streamlit

This project is an **AI-powered shoe recognition system** built with **CLIP model** and **Streamlit**.  
You can upload a shoe image, and the app will show the **most similar shoes** from a local gallery.  

It demonstrates **image embeddings, similarity search, and zero-shot learning** in a clean, interactive web app.

---

## Features
- Upload any shoe image and find **look-alike matches** from your gallery  
- Uses **CLIP embeddings** for zero-shot image similarity  
- **Preprocessing & caching** for faster performance  
- **Dominant color extraction** from images  
- Built with **modular, production-style code structure**  

---

## Project Structure
```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── gallery/                # Shoe image database (reference gallery)
├── cache/                  # Cached embeddings + NN index (auto-generated)
├── src/
│ ├── index/
│ │ └── gallery_index.py    # Builds gallery index, caches embeddings
│ ├── infer/
│ │ └── zero_shot.py        # Inference functions (e.g., top-1 search)
│ ├── models/
│ │ └── clip_loader.py      # Loads CLIP model + tokenizer
│ └── vision/
│ └── color.py              # Extracts dominant color from images
└── .venv/                  # Local Python virtual environment (not for Git)
```

## Installation

1. Clone the repository:
```
git clone https://github.com/username/shoe-recognition.git
cd shoe-recognition
```
2. Create a virtual environment:
```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the app locally:
```
streamlit run app.py
```
Steps:
1.  Start the Streamlit server, load the model and refresh the gallery index
2.  Upload a shoe image via the sidebar.
3.  The system finds the Top-K most similar shoes from your gallery.
4.  View similarity scores + dominant color extraction.

## Skills Demonstrated

-   Python & Streamlit for interactive AI apps
-   Computer Vision & Preprocessing
-   CLIP embeddings & zero-shot inference
-   Similarity search (cosine / Euclidean)
-   Caching & efficient data handling
-   Modular software design

## Example Demo
