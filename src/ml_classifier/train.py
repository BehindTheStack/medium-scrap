"""
Train a multi-label classifier to predict post categories (layers) from markdown posts.
Saves a model and vectorizer to src/ml_classifier/model.joblib and vectorizer.joblib.

Usage:
    python -m src.ml_classifier.train --input ../outputs/Netflix_timeline.json

This is a pragmatic first-pass: TF-IDF + OneVsRest(LogisticRegression) using existing labels
as weak/noisy supervision.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

CACHE_DIR = Path(__file__).resolve().parent
MODEL_PATH = CACHE_DIR / "model.joblib"
VECT_PATH = CACHE_DIR / "vectorizer.joblib"
MLB_PATH = CACHE_DIR / "mlb.joblib"


def read_timeline(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # timeline files may be either a list or an object with an "entries" key
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported timeline JSON structure")


def extract_text(entry: dict) -> str:
    """Build a text field from title, snippet, and file content if available."""
    parts = []
    if entry.get("title"):
        parts.append(entry["title"])
    if entry.get("snippet"):
        parts.append(entry["snippet"])
    # try to read markdown file if path exists
    p = entry.get("path")
    if p:
        try:
            md = Path(p)
            if md.exists():
                txt = md.read_text(encoding="utf-8")
                parts.append(txt)
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

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["labels"])
    X_texts = df["text"].tolist()

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")
    X = vectorizer.fit_transform(X_texts)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear"))
    print("Training classifier...")
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)

    # Report per-label
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    # Save artifacts
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)
    joblib.dump(mlb, MLB_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECT_PATH}")
    print(f"Saved label binarizer to {MLB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./outputs/Netflix_timeline.json", help="path to timeline json")
    args = parser.parse_args()
    train(args)
