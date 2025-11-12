"""
Script to train (if needed) and run predictions to create a refined timeline JSON.
Uses semantic embeddings (sentence-transformers) for better context understanding.

Outputs to: ./outputs/Netflix_timeline_refined.json

Usage:
    python scripts/reclassify_posts.py
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "Netflix_timeline.json"
OUTPUT = ROOT / "outputs" / "Netflix_timeline_refined.json"

ML_MODULE = "src.ml_classifier"


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    print("=" * 80)
    print("ML-Based Post Reclassification Pipeline (Semantic Embeddings)")
    print("=" * 80)
    
    # 1) train with embeddings
    print("\nStep 1/2: Training model with semantic embeddings")
    print("(This will download a 80MB model from HuggingFace on first run)")
    run([sys.executable, "-m", f"{ML_MODULE}.train_embeddings", "--input", str(INPUT)])

    # 2) predict with embeddings
    print("\nStep 2/2: Predicting and writing refined timeline")
    run([sys.executable, "-m", f"{ML_MODULE}.predict_embeddings", "--input", str(INPUT), "--output", str(OUTPUT)])

    print("\n" + "=" * 80)
    print("✅ Done! Refined timeline written to:", OUTPUT)
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Restart API: cd webapp && docker-compose restart api")
    print("  2. Open frontend: http://localhost:3000")


if __name__ == "__main__":
    main()
