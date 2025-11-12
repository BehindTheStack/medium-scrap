#!/usr/bin/env python3
"""
Timeline builder that reads directly from database.
No more markdown files needed!

Usage:
    python scripts/build_timeline_v2.py --source netflix
    python scripts/build_timeline_v2.py --source airbnb
"""
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infrastructure.pipeline_db import PipelineDB

OUTPUTS_DIR = ROOT / "outputs"

# Architecture layer keywords
LAYER_KEYWORDS = {
    "Infrastructure": [
        "aws", "ec2", "kubernetes", "k8s", "infrastructure", "cloud", 
        "container", "docker", "deployment", "cicd", "terraform", "ansible"
    ],
    "Backend APIs": [
        "api", "graphql", "rest", "restful", "backend", "microservice", 
        "grpc", "service mesh", "endpoint"
    ],
    "Data Infrastructure": [
        "data mesh", "data platform", "database", "kafka", "spark",
        "etl", "data warehouse", "bigquery", "cassandra", "elasticsearch",
        "flink", "airflow", "dbt"
    ],
    "Frontend & UI": [
        "ui", "frontend", "react", "vue", "angular", "web", "ux", 
        "css", "javascript", "typescript", "mobile", "ios", "android"
    ],
    "ML & AI": [
        "machine learning", "ml", "ai", "model", "training", "inference",
        "recommendation", "personalization", "deep learning", "neural network",
        "tensorflow", "pytorch", "llm", "generative ai"
    ],
    "Observability": [
        "observability", "monitoring", "tracing", "logging", "metrics",
        "alert", "slo", "sli", "prometheus", "grafana", "datadog"
    ],
    "Security": [
        "security", "authentication", "authorization", "encryption",
        "oauth", "compliance", "vulnerability", "penetration test"
    ],
    "Platform Engineering": [
        "platform", "developer experience", "internal tools", "backstage",
        "service catalog", "developer portal"
    ],
}


def classify_layers(text: str) -> list:
    """Classify text into architecture layers based on keywords"""
    text_lower = text.lower()
    matches = {}
    
    for layer, keywords in LAYER_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            matches[layer] = score
    
    # Return layers sorted by score
    return [layer for layer, _ in sorted(matches.items(), key=lambda x: x[1], reverse=True)]


def build_timeline(source):
    """Build timeline from database"""
    db = PipelineDB()
    
    # Get posts with content from database
    posts = db.get_posts_with_content(source=source)
    
    if not posts:
        print(f"❌ No posts with content found for source '{source}'")
        return None
    
    publication = posts[0]['publication']
    print(f"📚 Processing {len(posts)} posts from {publication}")
    
    entries = []
    
    for post in posts:
        # Get content
        content_md = post.get('content_markdown', '')
        if not content_md:
            continue
        
        # Extract date
        date = None
        if post.get('published_at'):
            try:
                date = datetime.fromisoformat(str(post['published_at']).replace('Z', '+00:00')).date()
            except:
                pass
        
        # Get tags
        try:
            tags = json.loads(post.get('tags', '[]')) if isinstance(post.get('tags'), str) else post.get('tags', [])
        except:
            tags = []
        
        # Classify into layers
        classify_text = f"{post.get('title', '')} {content_md[:4000]} {' '.join(tags)}"
        layers = classify_layers(classify_text)
        
        if not layers:
            layers = ["Other"]
        
        # Create snippet from markdown (first paragraph)
        lines = content_md.split('\n')
        snippet = ''
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('---'):
                snippet = line[:200]
                break
        
        entries.append({
            'id': post['id'],
            'title': post.get('title', 'Untitled'),
            'date': date.isoformat() if date else None,
            'layers': layers,
            'snippet': snippet,
            'url': post.get('url', ''),
            'author': post.get('author', 'Unknown'),
            'reading_time': post.get('reading_time', 0),
            'categories': tags[:5],
            'is_technical': post.get('is_technical', False),
            'technical_score': post.get('technical_score', 0.0),
            'code_blocks': post.get('code_blocks', 0),
        })
    
    # Sort by date
    entries.sort(key=lambda e: (e['date'] is None, e['date'] or '9999-12-31'), reverse=False)
    
    # Group by layer
    per_layer = {}
    for e in entries:
        for layer in e['layers']:
            per_layer.setdefault(layer, []).append(e)
    
    return {
        'count': len(entries),
        'publication': publication,
        'source': source,
        'posts': entries,
        'per_layer': per_layer,
        'stats': {
            'total_posts': len(entries),
            'technical_posts': sum(1 for e in entries if e.get('is_technical')),
            'layers': {layer: len(items) for layer, items in per_layer.items()},
            'date_range': {
                'earliest': min((e['date'] for e in entries if e['date']), default=None),
                'latest': max((e['date'] for e in entries if e['date']), default=None),
            }
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build timeline from database')
    parser.add_argument('--source', required=True, help='Source key (e.g., netflix, airbnb)')
    args = parser.parse_args()
    
    timeline = build_timeline(args.source)
    
    if not timeline:
        exit(1)
    
    # Output filenames
    out_json = OUTPUTS_DIR / f"{args.source}_timeline.json"
    out_md = OUTPUTS_DIR / f"{args.source}_timeline.md"
    
    # Write JSON
    out_json.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"✅ Wrote JSON: {out_json}")
    
    # Write Markdown summary
    md_lines = [f"# {timeline['publication']} Timeline\n"]
    md_lines.append(f"**Total Posts**: {timeline['stats']['total_posts']}")
    md_lines.append(f"**Technical Posts**: {timeline['stats']['technical_posts']}\n")
    
    if timeline['stats']['date_range']['earliest']:
        md_lines.append(f"**Date Range**: {timeline['stats']['date_range']['earliest']} to {timeline['stats']['date_range']['latest']}\n")
    
    md_lines.append('\n## Posts by Layer\n')
    
    for layer, items in sorted(timeline['per_layer'].items(), key=lambda x: len(x[1]), reverse=True):
        md_lines.append(f'\n### {layer} ({len(items)} posts)\n')
        for item in items[:10]:
            date = item['date'] or 'unknown'
            title = item['title'][:80]
            md_lines.append(f"- **{date}** — {title}")
        
        if len(items) > 10:
            md_lines.append(f"  _... and {len(items) - 10} more posts_\n")
    
    out_md.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f"✅ Wrote Markdown: {out_md}")
    
    # Print stats
    print(f"\n📊 Timeline Statistics:")
    print(f"   Total: {timeline['stats']['total_posts']} posts")
    print(f"   Technical: {timeline['stats']['technical_posts']} posts")
    print(f"\n   Layers:")
    for layer, count in sorted(timeline['stats']['layers'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / timeline['stats']['total_posts']) * 100
        print(f"      - {layer}: {count} ({pct:.1f}%)")
    
    print(f"\n✨ Timeline created successfully!")
