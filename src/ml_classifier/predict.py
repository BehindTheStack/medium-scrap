"""
Load saved model and predict labels for posts. Outputs JSON with predicted layers and confidences.

Usage:
    python -m src.ml_classifier.predict --input ./outputs/Netflix_timeline.json --output ./outputs/Netflix_timeline_refined.json
"""
from __future__ import annotations
import json
from pathlib import Path
import argparse
import joblib
from tqdm import tqdm

CACHE_DIR = Path(__file__).resolve().parent
MODEL_PATH = CACHE_DIR / "model.joblib"
VECT_PATH = CACHE_DIR / "vectorizer.joblib"
MLB_PATH = CACHE_DIR / "mlb.joblib"


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
                parts.append(txt)
        except Exception:
            pass
    return "\n".join(parts)


def predict(args):
    timeline = read_timeline(Path(args.input))
    if not MODEL_PATH.exists() or not VECT_PATH.exists() or not MLB_PATH.exists():
        raise SystemExit("Model artifacts not found. Run train.py first.")

    clf = joblib.load(MODEL_PATH)
    vect = joblib.load(VECT_PATH)
    mlb = joblib.load(MLB_PATH)

    out = []
    for e in tqdm(timeline, desc="predicting"):
        text = extract_text(e)
        if not text.strip():
            e2 = e.copy()
            e2["layers"] = []
            out.append(e2)
            continue
        X = vect.transform([text])
        # predict_proba may not be available for some estimators; OneVsRestClassifier with LogisticRegression supports it
        try:
            probs = clf.decision_function(X)
            # convert to probabilities via sigmoid if necessary
            import numpy as np
            probs = 1 / (1 + np.exp(-probs))
        except Exception:
            try:
                probs = clf.predict_proba(X)
            except Exception:
                probs = clf.predict(X)
        # handle shapes
        if hasattr(probs, "shape") and probs.shape[1] == len(mlb.classes_):
            scores = probs[0].tolist()
        else:
            # fallback: binary predictions
            preds = clf.predict(X)[0]
            scores = [float(p) for p in preds]

        # thresholding
        threshold = args.threshold
        labels = [c for c, s in zip(mlb.classes_, scores) if s >= threshold]

        # Replace original layers with predicted ones
        e2 = e.copy()
        e2["layers"] = labels
        out.append(e2)

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump({"count": len(out), "entries": out}, f, ensure_ascii=False, indent=2)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./outputs/Netflix_timeline.json")
    parser.add_argument("--output", default="./outputs/Netflix_timeline_refined.json")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    predict(args)
