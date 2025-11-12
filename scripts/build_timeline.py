#!/usr/bin/env python3
"""
Generic timeline builder for any Medium publication.
Reads directly from the database instead of markdown files.

Usage:
    python scripts/build_timeline.py --source netflix
    python scripts/build_timeline.py --source airbnb
    python scripts/build_timeline.py --publication "Netflix Tech Blog"
"""
import os
import re
import json
import yaml
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infrastructure.pipeline_db import PipelineDB

OUTPUTS_DIR = ROOT / "outputs"

# Architecture layer keywords (same as Netflix)
LAYER_KEYWORDS = {
    "Infrastructure": [
        "aws", "ec2", "kubernetes", "k8s", "titus", "spinnaker", 
        "infrastructure", "cloud", "multi-cloud", "gcp", "azure", 
        "container", "docker", "deployment", "cicd", "terraform"
    ],
    "Backend APIs": [
        "api", "graphql", "rest", "restful", "backend", "edge", 
        "zuul", "gateway", "microservice", "microservices", "grpc",
        "service mesh", "endpoint", "backend service"
    ],
    "Data Infrastructure": [
        "data mesh", "data platform", "data pipeline", "database",
        "iceberg", "presto", "druid", "kafka", "flink", "spark",
        "etl", "ingest", "ingestion", "data warehouse", "bigquery",
        "redshift", "snowflake", "cassandra", "elasticsearch"
    ],
    "Frontend & UI": [
        "ui", "frontend", "html5", "playback", "tv", "react", 
        "vue", "angular", "web", "ux", "user interface", "css",
        "javascript", "typescript", "mobile app", "ios", "android"
    ],
    "Video/Encoding": [
        "encode", "encoding", "av1", "x264", "x265", "video", 
        "codec", "4k", "hdr", "streaming video", "transcoding",
        "video processing", "media", "playback"
    ],
    "Observability": [
        "observability", "tracing", "atlas", "monitor", "monitoring",
        "alert", "alerting", "slo", "sli", "canary", "telemetry",
        "traces", "metrics", "logging", "apm", "prometheus",
        "grafana", "datadog", "new relic"
    ],
    "ML & Data Science": [
        "machine learning", "ml", "recommen", "recommender", 
        "model", "training", "inference", "personalization",
        "data science", "artificial intelligence", "ai", 
        "deep learning", "neural network", "tensorflow", "pytorch"
    ],
    "Security": [
        "security", "fido", "bug bounty", "auth", "authentication",
        "authorization", "token", "encryption", "credential",
        "ssl", "tls", "oauth", "compliance", "gdpr", "vulnerability"
    ],
    "Platform Engineering": [
        "platform", "platform engineering", "paas", "developer experience",
        "developer tools", "internal tools", "backstage", "portal",
        "platform team", "service catalog"
    ],
}

DATE_PATTERNS = [
    (re.compile(r"([A-Z][a-z]{2} \d{1,2}, \d{4})"), "%b %d, %Y"),
    (re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4})"), "%B %d, %Y"),
    (re.compile(r"(\d{4}-\d{2}-\d{2})"), "%Y-%m-%d"),
    (re.compile(r"(\d{1,2}/\d{1,2}/\d{4})"), "%m/%d/%Y"),
]
def parse_date_from_text(text):
    """Try to extract a date from text using multiple patterns"""
    for rx, fmt in DATE_PATTERNS:
        m = rx.search(text)
        if m:
            s = m.group(1)
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                continue
    return None


def extract_title_and_snippet(md_text):
    """Extract title (first H1) and snippet (first paragraph) from markdown
    
    Also extracts YAML frontmatter if present.
    """
    frontmatter = {}
    content = md_text
    
    # Check for YAML frontmatter
    if md_text.startswith('---\n'):
        parts = md_text.split('---\n', 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1])
                content = parts[2]  # Content after frontmatter
            except Exception:
                pass
    
    title = None
    snippet = None
    lines = content.splitlines()
    
    # Try to get title from frontmatter first
    if frontmatter.get('title'):
        title = frontmatter['title']
    
    # Otherwise extract from first H1
    if not title:
        for i, ln in enumerate(lines[:20]):
            if ln.startswith('# '):
                title = ln.lstrip('# ').strip()
                # snippet: first non-empty paragraph after title
                for j in range(i+1, min(i+30, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('[') and not lines[j].startswith('!') and not lines[j].startswith('#'):
                        snippet = lines[j].strip()
                        break
                break
    
    if not title:
        # fallback: first non-empty line
        for ln in lines:
            if ln.strip():
                title = ln.strip()
                break
    
    # Get snippet if not found
    if not snippet:
        for ln in lines[:50]:
            if ln.strip() and not ln.startswith('#') and not ln.startswith('[') and not ln.startswith('!'):
                snippet = ln.strip()
                if len(snippet) > 20:  # Ignore very short lines
                    break
    
    return title or "(untitled)", snippet or "", frontmatter


def classify_layers(text):
    """Classify text into architecture layers based on keyword matching"""
    text_l = text.lower()
    layers = set()
    
    for layer, keys in LAYER_KEYWORDS.items():
        for k in keys:
            if k.lower() in text_l:
                layers.add(layer)
                break
    
    if not layers:
        layers.add('Uncategorized')
    
    return sorted(layers)


def build_timeline(publication_name: str):
    """Build timeline for a given publication"""
    publication_dir = OUTPUTS_DIR / publication_name
    
    if not publication_dir.exists():
        print(f"❌ Error: Publication directory not found: {publication_dir}")
        print(f"   Make sure you've scraped posts with: python main.py collect {publication_name.lower()}")
        return None
    
    # Find all markdown files
    md_files = list(publication_dir.glob('*.md'))
    
    if not md_files:
        print(f"❌ Error: No markdown files found in {publication_dir}")
        print(f"   Directory exists but contains no .md files")
        return None
    
    print(f"📚 Found {len(md_files)} markdown files in {publication_name}")
    
    entries = []
    
    for p in sorted(md_files):
        try:
            md_text = p.read_text(encoding='utf-8')
        except Exception:
            try:
                md_text = p.read_text(encoding='latin-1')
            except Exception as e:
                print(f"⚠️  Warning: Could not read {p.name}: {e}")
                continue
        
        title, snippet, frontmatter = extract_title_and_snippet(md_text)
        
        # Try to enrich classification with JSON metadata
        enrichment = ''
        json_path = p.with_suffix('.json')
        meta = frontmatter.copy() if frontmatter else {}  # Start with frontmatter
        
        if json_path.exists():
            try:
                json_meta = json.loads(json_path.read_text(encoding='utf-8'))
                # Merge JSON metadata (JSON takes precedence if keys conflict)
                meta.update(json_meta)
                # Add tags/topics/description to enrichment
                if isinstance(meta.get('tags'), list):
                    enrichment += ' ' + ' '.join([str(x) for x in meta.get('tags')])
                if isinstance(meta.get('topics'), list):
                    enrichment += ' ' + ' '.join([str(x) for x in meta.get('topics')])
                if isinstance(meta.get('description'), str):
                    enrichment += ' ' + meta.get('description')
            except Exception:
                pass
        
        # Try to find date - prioritize: 1) frontmatter, 2) JSON, 3) text parsing
        date = None
        
        # 1. Check frontmatter first
        if frontmatter.get('published_at'):
            try:
                date = datetime.fromisoformat(str(frontmatter['published_at']).replace('Z', '+00:00')).date()
            except Exception:
                pass
        
        # 2. Check JSON metadata
        if not date and meta:
            for k in ('publishedAt', 'date', 'created_at', 'published_at', 'createdAt'):
                if k in meta:
                    try:
                        # Try ISO format
                        date = datetime.fromisoformat(str(meta[k]).replace('Z', '+00:00')).date()
                        break
                    except Exception:
                        try:
                            # Try simple date format
                            date = datetime.strptime(str(meta[k])[:10], '%Y-%m-%d').date()
                            break
                        except Exception:
                            pass
        
        # 3. Fallback to parsing from markdown text
        if not date:
            date = parse_date_from_text(md_text)
        
        # Get URL from metadata (frontmatter or JSON)
        url = frontmatter.get('url') or meta.get('url', '')
        
        # Get author from metadata
        author = frontmatter.get('author') or meta.get('author', 'Unknown')
        
        # Get reading time
        reading_time = frontmatter.get('reading_time') or meta.get('reading_time', 0)
        
        # Get categories/tags from metadata
        categories = []
        if meta:
            if isinstance(meta.get('tags'), list):
                categories = meta.get('tags', [])
            elif isinstance(meta.get('topics'), list):
                categories = meta.get('topics', [])
        
        # Classify into layers (using title + content + metadata)
        classify_text = ' '.join([title or '', md_text[:4000], enrichment])
        layers = classify_layers(classify_text)
        
        entries.append({
            'path': str(p.relative_to(ROOT)),
            'title': title,
            'date': date.isoformat() if date else None,
            'layers': layers,
            'snippet': snippet[:200] if snippet else '',  # Limit snippet length
            'url': url,
            'categories': categories[:5],  # Limit to 5 categories
        })
    
    # Sort by date (unknown dates at end)
    entries.sort(key=lambda e: (e['date'] is None, e['date'] or '9999-12-31'), reverse=False)
    
    # Build per-layer index
    per_layer = {}
    for e in entries:
        for layer in e['layers']:
            per_layer.setdefault(layer, []).append(e)
    
    return {
        'count': len(entries),
        'publication': publication_name,
        'entries': entries,
        'per_layer': per_layer,
        'stats': {
            'total_posts': len(entries),
            'layers': {layer: len(posts) for layer, posts in per_layer.items()},
            'date_range': {
                'earliest': min((e['date'] for e in entries if e['date']), default=None),
                'latest': max((e['date'] for e in entries if e['date']), default=None),
            }
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build timeline for any Medium publication')
    parser.add_argument('--publication', required=True, help='Publication name (e.g., "Airbnb Engineering")')
    parser.add_argument('--output-name', help='Custom output filename (default: auto-generated from publication name)')
    args = parser.parse_args()
    
    publication = args.publication
    timeline = build_timeline(publication)
    
    if timeline is None:
        exit(1)
    
    # Generate output filename
    if args.output_name:
        out_name = args.output_name
    else:
        # Convert "Airbnb Engineering" -> "airbnb_timeline.json"
        safe_name = publication.lower().split()[0]  # Take first word
        out_name = f"{safe_name}_timeline.json"
    
    out_json = OUTPUTS_DIR / out_name
    out_md = OUTPUTS_DIR / out_name.replace('.json', '.md')
    
    # Write JSON
    out_json.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"✅ Wrote JSON timeline: {out_json}")
    
    # Write markdown summary
    md_lines = [f'# {publication} Timeline by Layer\n']
    md_lines.append(f"**Total Posts**: {timeline['stats']['total_posts']}\n")
    
    if timeline['stats']['date_range']['earliest']:
        md_lines.append(f"**Date Range**: {timeline['stats']['date_range']['earliest']} to {timeline['stats']['date_range']['latest']}\n")
    
    md_lines.append('\n## Posts by Layer\n')
    
    for layer, items in sorted(timeline['per_layer'].items(), key=lambda x: len(x[1]), reverse=True):
        md_lines.append(f'\n### {layer} ({len(items)} posts)\n')
        for it in items[:10]:  # Show first 10 posts per layer
            date = it['date'] or 'unknown'
            title = it['title'][:80]  # Truncate long titles
            md_lines.append(f"- **{date}** — {title}")
        
        if len(items) > 10:
            md_lines.append(f"  _... and {len(items) - 10} more posts_")
    
    out_md.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f"✅ Wrote Markdown summary: {out_md}")
    
    # Print stats
    print(f"\n📊 Timeline Statistics:")
    print(f"   Total Posts: {timeline['stats']['total_posts']}")
    print(f"   Layers:")
    for layer, count in sorted(timeline['stats']['layers'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / timeline['stats']['total_posts']) * 100
        print(f"      - {layer}: {count} ({percentage:.1f}%)")
    
    print(f"\n✨ Next step: Run ML classification")
    print(f"   python scripts/reclassify_posts.py --input {out_json} --output {out_json.with_name(out_name.replace('_timeline', '_timeline_refined'))}")
