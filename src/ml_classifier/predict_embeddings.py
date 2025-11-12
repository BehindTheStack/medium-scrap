"""
Predict labels using semantic embeddings model.

Usage:
    python -m src.ml_classifier.predict_embeddings --input ./outputs/Netflix_timeline.json --output ./outputs/Netflix_timeline_refined.json
"""
from __future__ import annotations
import json
from pathlib import Path
import argparse
import warnings

import joblib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

warnings.filterwarnings('ignore')

CACHE_DIR = Path(__file__).resolve().parent
MODEL_PATH = CACHE_DIR / "model_embeddings.joblib"
MLB_PATH = CACHE_DIR / "mlb_embeddings.joblib"


def read_timeline(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported timeline JSON structure")


def extract_text(entry: dict) -> str:
    parts = []
    if entry.get("title"):
        parts.append(entry["title"])
    if entry.get("snippet"):
        parts.append(entry["snippet"])
    p = entry.get("path")
    if p:
        try:
            md = Path(p)
            if md.exists():
                txt = md.read_text(encoding="utf-8")
                parts.append(txt[:8000])  # Limit content
        except Exception:
            pass
    return "\n".join(parts)


def predict(args):
    timeline = read_timeline(Path(args.input))
    if not MODEL_PATH.exists() or not MLB_PATH.exists():
        raise SystemExit("Model artifacts not found. Run train_embeddings.py first.")

    # Load artifacts
    print("Loading model...")
    model_data = joblib.load(MODEL_PATH)
    clf = model_data["classifier"]
    embedder_name = model_data["embedder_name"]
    mlb = joblib.load(MLB_PATH)
    
    print(f"Loading sentence transformer: {embedder_name}")
    embedder = SentenceTransformer(embedder_name)
    
    print(f"Processing {len(timeline)} posts...")
    
    # Batch processing for efficiency
    texts = []
    valid_indices = []
    
    for idx, e in enumerate(timeline):
        text = extract_text(e)
        if text.strip():
            texts.append(text)
            valid_indices.append(idx)
    
    print(f"Valid posts with text: {len(texts)}")
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Predict
    print("Predicting labels...")
    import numpy as np
    
    # Get decision scores
    scores = clf.decision_function(embeddings)
    # Convert to probabilities
    probs = 1 / (1 + np.exp(-scores))
    
    # Build output
    out = []
    pred_idx = 0
    
    for idx, e in enumerate(tqdm(timeline, desc="building output")):
        e2 = e.copy()
        
        if idx not in valid_indices:
            # No text, keep empty
            e2["layers"] = []
        else:
            # Get predictions for this post
            post_probs = probs[pred_idx]
            
            # Apply threshold
            threshold = args.threshold
            labels = [
                c for c, s in zip(mlb.classes_, post_probs) 
                if s >= threshold
            ]
            
            e2["layers"] = labels
            pred_idx += 1
        
        out.append(e2)
    
    # Write output
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"count": len(out), "entries": out}, f, ensure_ascii=False, indent=2)
    
    print(f"\nWrote predictions to {args.output}")
    
    # Stats
    total_labels = sum(len(e["layers"]) for e in out)
    avg_labels = total_labels / len(out)
    print(f"Average labels per post: {avg_labels:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./outputs/Netflix_timeline.json")
    parser.add_argument("--output", default="./outputs/Netflix_timeline_refined.json")
    parser.add_argument("--threshold", type=float, default=0.7, help="Higher threshold = fewer, more confident labels (0.7 recommended)")
    args = parser.parse_args()
    predict(args)
