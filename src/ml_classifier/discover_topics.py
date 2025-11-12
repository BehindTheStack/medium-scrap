"""
Discover natural topics/clusters in Netflix Tech Blog posts using unsupervised learning.
Instead of forcing predefined categories, let the data tell us what the real topics are.

Approach:
1. Generate semantic embeddings for all posts
2. Apply clustering (K-means or HDBSCAN) to find natural groupings
3. Extract top keywords/phrases per cluster to create descriptive labels
4. Generate new timeline JSON with discovered labels

Usage:
    python -m src.ml_classifier.discover_topics --input ./outputs/Netflix_timeline.json --output ./outputs/Netflix_timeline_discovered.json --n-topics 12
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import argparse
import warnings
from collections import Counter

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings('ignore')

CACHE_DIR = Path(__file__).resolve().parent
EMBEDDER_NAME = "all-MiniLM-L6-v2"


def read_timeline(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported timeline JSON structure")


def extract_text(entry: dict) -> str:
    """Extract text from post."""
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
                # Take more content for better topic discovery
                parts.append(txt[:15000])
        except Exception:
            pass
    return "\n".join(parts)


def extract_keywords_from_cluster(texts: List[str], top_n: int = 10) -> List[str]:
    """Extract top keywords from a cluster of texts using TF-IDF."""
    if not texts:
        return []
    
    # Custom stopwords - remove generic/uninformative words
    custom_stops = set([
        'netflix', 'com', 'https', 'http', 'www', 'medium', 'source', 
        'github', 'tech', 'blog', 'post', 'article', 'read', 'min',
        'netflixtechblog', 'org', 'html', 'png', 'jpg', 'jpeg'
    ])
    
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words=list(set(list(TfidfVectorizer(stop_words='english').get_stop_words()) + list(custom_stops))),
        ngram_range=(1, 2),
        min_df=2
    )
    
    try:
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across all docs in cluster
        scores = np.asarray(X.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        
        return [feature_names[i] for i in top_indices]
    except Exception:
        # Fallback to word frequency
        words = []
        for text in texts:
            words.extend(text.lower().split())
        return [w for w, _ in Counter(words).most_common(top_n)]


def create_label_from_keywords(keywords: List[str], cluster_id: int) -> str:
    """Create a human-readable label from keywords."""
    # Take top 2-3 most distinctive keywords
    if len(keywords) >= 2:
        # Clean and format
        top = [k.replace('_', ' ').title() for k in keywords[:2]]
        return '/'.join(top)
    elif keywords:
        return keywords[0].replace('_', ' ').title()
    else:
        return f"Topic_{cluster_id}"


def discover_topics(args):
    print("=" * 80)
    print("Netflix Tech Blog - Topic Discovery (Unsupervised)")
    print("=" * 80)
    
    # Load data
    timeline_path = Path(args.input)
    entries = read_timeline(timeline_path)
    print(f"\nLoaded {len(entries)} posts")
    
    # Extract texts
    print("Extracting text from posts...")
    texts = []
    valid_entries = []
    
    for e in tqdm(entries, desc="reading posts"):
        text = extract_text(e)
        if text.strip():
            texts.append(text)
            valid_entries.append(e)
    
    print(f"Valid posts with text: {len(texts)}")
    
    # Generate embeddings
    print(f"\nLoading sentence transformer: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    
    print("Generating embeddings...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Clustering
    n_clusters = args.n_topics
    print(f"\nClustering into {n_clusters} topics using K-means...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Analyze clusters
    print("\n" + "=" * 80)
    print("Discovered Topics")
    print("=" * 80)
    
    cluster_info = {}
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
        cluster_entries = [valid_entries[i] for i, m in enumerate(mask) if m]
        
        # Extract keywords
        keywords = extract_keywords_from_cluster(cluster_texts, top_n=10)
        label = create_label_from_keywords(keywords, cluster_id)
        
        cluster_info[cluster_id] = {
            'label': label,
            'keywords': keywords,
            'size': len(cluster_texts),
            'entries': cluster_entries
        }
        
        print(f"\n📁 Topic {cluster_id}: {label}")
        print(f"   Posts: {len(cluster_texts)}")
        print(f"   Keywords: {', '.join(keywords[:8])}")
        
        # Show sample titles
        sample_titles = [e.get('title', 'Untitled')[:80] for e in cluster_entries[:3]]
        print(f"   Examples:")
        for title in sample_titles:
            print(f"     - {title}")
    
    # Option to review and rename labels interactively
    if args.interactive:
        print("\n" + "=" * 80)
        print("Interactive Label Refinement")
        print("=" * 80)
        print("Review the discovered topics and optionally rename them.")
        print("Press Enter to keep the auto-generated label, or type a new one.\n")
        
        for cluster_id in range(n_clusters):
            info = cluster_info[cluster_id]
            current_label = info['label']
            print(f"\nTopic {cluster_id}: {current_label}")
            print(f"Keywords: {', '.join(info['keywords'][:5])}")
            new_label = input(f"New label (or Enter to keep): ").strip()
            if new_label:
                cluster_info[cluster_id]['label'] = new_label
                print(f"✓ Updated to: {new_label}")
    
    # Generate output JSON with new labels
    print("\n" + "=" * 80)
    print("Generating refined timeline with discovered topics")
    print("=" * 80)
    
    # Create mapping of entry to cluster label
    entry_to_cluster = {}
    for cluster_id, info in cluster_info.items():
        for entry in info['entries']:
            # Use path as unique ID
            entry_id = entry.get('path')
            if entry_id:
                entry_to_cluster[entry_id] = info['label']
    
    # Build output
    output_entries = []
    for entry in entries:
        entry_id = entry.get('path')
        new_entry = entry.copy()
        
        if entry_id and entry_id in entry_to_cluster:
            # Single label per post (primary topic)
            new_entry['layers'] = [entry_to_cluster[entry_id]]
        else:
            new_entry['layers'] = []
        
        output_entries.append(new_entry)
    
    # Write output
    output_path = Path(args.output)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump({
            'count': len(output_entries),
            'entries': output_entries,
            'metadata': {
                'method': 'unsupervised_clustering',
                'n_topics': n_clusters,
                'embedder': EMBEDDER_NAME,
                'topics': {
                    str(cid): {
                        'label': info['label'],
                        'keywords': info['keywords'],
                        'size': info['size']
                    }
                    for cid, info in cluster_info.items()
                }
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Wrote refined timeline to: {output_path}")
    
    # Save cluster model for future use
    model_path = CACHE_DIR / "topic_model.joblib"
    joblib.dump({
        'kmeans': kmeans,
        'cluster_info': cluster_info,
        'embedder_name': EMBEDDER_NAME
    }, model_path)
    print(f"✅ Saved topic model to: {model_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total posts: {len(entries)}")
    print(f"Posts with topics: {sum(1 for e in output_entries if e['layers'])}")
    print(f"Average labels per post: {sum(len(e['layers']) for e in output_entries) / len(output_entries):.2f}")
    print(f"\nTopic distribution:")
    for cid, info in sorted(cluster_info.items(), key=lambda x: x[1]['size'], reverse=True):
        print(f"  {info['label']:30s} {info['size']:3d} posts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./outputs/Netflix_timeline.json")
    parser.add_argument("--output", default="./outputs/Netflix_timeline_discovered.json")
    parser.add_argument("--n-topics", type=int, default=12, help="Number of topics to discover (8-15 recommended)")
    parser.add_argument("--interactive", action="store_true", help="Interactively rename discovered topics")
    args = parser.parse_args()
    discover_topics(args)
