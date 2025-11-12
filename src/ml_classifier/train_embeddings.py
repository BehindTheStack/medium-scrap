"""
Train a multi-label classifier using semantic embeddings instead of TF-IDF.
Uses sentence-transformers to capture contextual meaning, not just keywords.

Usage:
    python -m src.ml_classifier.train_embeddings --input ../outputs/Netflix_timeline.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
from tqdm import tqdm
import sys
from pathlib import Path as _Path

# Try to import centralized cleaner
try:
    src_root = _Path(__file__).parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from presentation.helpers.text_cleaner import clean_markdown
except Exception:
    def clean_markdown(x):
        return x or ""

warnings.filterwarnings('ignore')

CACHE_DIR = Path(__file__).resolve().parent
MODEL_PATH = CACHE_DIR / "model_embeddings.joblib"
EMBEDDER_NAME = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dims
MLB_PATH = CACHE_DIR / "mlb_embeddings.joblib"


def read_timeline(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported timeline JSON structure")


def extract_text(entry: dict) -> str:
    """Build a text field from title, snippet, and file content."""
    parts = []
    if entry.get("title"):
        parts.append(entry["title"])
    if entry.get("snippet"):
        parts.append(entry["snippet"])
    
    # Read markdown content
    p = entry.get("path")
    if p:
        try:
            md = Path(p)
            if md.exists():
                txt = md.read_text(encoding="utf-8")
                # Keep full cleaned content (no hard truncation). Caller can decide to chunk.
                parts.append(clean_markdown(txt))
        except Exception:
            pass
    return "\n".join(parts)


def build_dataset(entries: List[dict]):
    rows = []
    for e in tqdm(entries, desc="building dataset"):
        text = extract_text(e)
        labels = e.get("layers") or []
        rows.append({"text": text, "labels": labels, "meta": e})
    df = pd.DataFrame(rows)
    # drop empty texts
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    return df


def train(args):
    timeline_path = Path(args.input)
    entries = read_timeline(timeline_path)
    df = build_dataset(entries)
    if df.empty:
        raise SystemExit("No text data found to train on.")

    print(f"Dataset size: {len(df)} posts")
    
    # Prepare labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["labels"])
    print(f"Number of unique labels: {len(mlb.classes_)}")
    print(f"Labels: {list(mlb.classes_)}")
    
    # Load sentence transformer
    print(f"Loading sentence transformer: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    
    # Generate embeddings
    print("Generating embeddings (this may take a few minutes)...")
    X_texts = df["text"].tolist()
    X = embedder.encode(
        X_texts, 
        show_progress_bar=True, 
        batch_size=32,
        convert_to_numpy=True
    )
    print(f"Embeddings shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42
    )

    # Train classifier
    print("Training classifier on embeddings...")
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    )
    clf.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = clf.predict(X_test)
    
    print("\nPer-label metrics:")
    report = classification_report(
        y_test, y_pred, 
        target_names=mlb.classes_,
        zero_division=0
    )
    print(report)
    
    hl = hamming_loss(y_test, y_pred)
    print(f"\nHamming Loss: {hl:.4f}")

    # Save artifacts
    joblib.dump({
        "classifier": clf,
        "embedder_name": EMBEDDER_NAME
    }, MODEL_PATH)
    joblib.dump(mlb, MLB_PATH)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved label binarizer to {MLB_PATH}")
    print(f"\nNote: The embedder '{EMBEDDER_NAME}' will be loaded from HuggingFace at prediction time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./outputs/Netflix_timeline.json")
    args = parser.parse_args()
    train(args)
