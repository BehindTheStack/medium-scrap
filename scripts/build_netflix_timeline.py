#!/usr/bin/env python3
"""Build a timeline per layer/architecture from Netflix Markdown outputs.

Outputs:
 - outputs/Netflix_timeline.json
 - outputs/Netflix_timeline.md

Heuristics: keyword -> layer mapping; posts can map to multiple layers.
"""
import os
import re
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETFLIX_DIR = ROOT / "outputs" / "Netflix Tech Blog"
OUT_JSON = ROOT / "outputs" / "Netflix_timeline.json"
OUT_MD = ROOT / "outputs" / "Netflix_timeline.md"

# Simple keyword -> layer mapping (lowercase keys)
LAYER_KEYWORDS = {
    "data": ["data mesh", "data platform", "data pipeline", "data", "iceberg", "presto", "druid", "kafka", "flink", "etl", "ingest", "ingestion"],
    "streaming": ["stream", "streaming", "mantis", "kafka", "flink", "stream-processing", "real-time", "streaming sql", "kinesis"],
    "video/encoding": ["encode", "encoding", "av1", "x264", "x265", "video", "codec", "4k", "hdr"],
    "infrastructure": ["aws", "ec2", "kubernetes", "titus", "spinnaker", "infrastructure", "cloud", "multi-cloud", "gcp", "azure", "container", "docker"],
    "platform": ["platform", "platform engineering", "paas", "p a a s", "mesh", "platform team", "platform engineering"],
    "frontend/ui": ["ui", "frontend", "html5", "playback", "tv", "react", "web", "ux", "user interface"],
    "api/backend": ["api", "graphql", "rest", "backend", "edge", "zuul", "gateway", "microservice", "microservices"],
    "observability": ["observability", "tracing", "atlas", "monitor", "alert", "slo", "canary", "telemetry", "traces", "metrics"],
    "ml/data-science": ["machine learning", "ml", "recommen", "recommender", "model", "training", "inference", "personalization"],
    "security": ["security", "fido", "bug bounty", "auth", "token", "encryption", "credential"],
}

DATE_PATTERNS = [
    (re.compile(r"([A-Z][a-z]{2} \d{1,2}, \d{4})"), "%b %d, %Y"),
    (re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4})"), "%B %d, %Y"),
    (re.compile(r"(\d{4}-\d{2}-\d{2})"), "%Y-%m-%d"),
]


def parse_date_from_text(text):
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
    # Title: first H1
    title = None
    snippet = None
    lines = md_text.splitlines()
    for i, ln in enumerate(lines[:20]):
        if ln.startswith('# '):
            title = ln.lstrip('# ').strip()
            # snippet: first non-empty paragraph after title
            for j in range(i+1, i+30):
                if j < len(lines) and lines[j].strip() and not lines[j].startswith('[') and not lines[j].startswith('!'):
                    snippet = lines[j].strip()
                    break
            break
    if not title:
        # fallback: first non-empty line
        for ln in lines:
            if ln.strip():
                title = ln.strip()
                break
    return title or "(untitled)", snippet or ""


def classify_layers(text):
    text_l = text.lower()
    layers = set()
    for layer, keys in LAYER_KEYWORDS.items():
        for k in keys:
            if k.lower() in text_l:
                layers.add(layer)
                break
    if not layers:
        layers.add('uncategorized')
    return sorted(layers)


def build_timeline():
    entries = []
    for p in sorted(NETFLIX_DIR.glob('*.md')):
        try:
            md_text = p.read_text(encoding='utf-8')
        except Exception:
            md_text = p.read_text(encoding='latin-1')
        title, snippet = extract_title_and_snippet(md_text)
        # try to enrich the classification text with JSON metadata (tags, description)
        enrichment = ''
        json_path = p.with_suffix('.json')
        meta = {}
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding='utf-8'))
                # add tags/topics/description to enrichment
                if isinstance(meta.get('tags'), list):
                    enrichment += ' ' + ' '.join([str(x) for x in meta.get('tags')])
                if isinstance(meta.get('topics'), list):
                    enrichment += ' ' + ' '.join([str(x) for x in meta.get('topics')])
                if isinstance(meta.get('description'), str):
                    enrichment += ' ' + meta.get('description')
            except Exception:
                meta = {}
        # try to find date in associated json first (common keys)
        date = None
        if meta:
            for k in ('publishedAt', 'date', 'created_at', 'published_at'):
                if k in meta:
                    try:
                        # try iso parse, fallback to simple date
                        date = datetime.fromisoformat(str(meta[k])).date()
                        break
                    except Exception:
                        try:
                            # try parsing common formats
                            date = datetime.strptime(str(meta[k]), '%Y-%m-%d').date()
                            break
                        except Exception:
                            pass
        if not date:
            date = parse_date_from_text(md_text)
        # include title and metadata enrichment when classifying
        classify_text = ' '.join([title or '', md_text[:4000], enrichment])
        layers = classify_layers(classify_text)
        entries.append({
            'path': str(p),
            'title': title,
            'date': date.isoformat() if date else None,
            'layers': layers,
            'snippet': snippet,
        })
    # sort by date (unknown dates at end)
    entries.sort(key=lambda e: (e['date'] is None, e['date'] or ''), reverse=False)
    # build per-layer index
    per_layer = {}
    for e in entries:
        for layer in e['layers']:
            per_layer.setdefault(layer, []).append(e)
    return {'entries': entries, 'per_layer': per_layer}


if __name__ == '__main__':
    timeline = build_timeline()
    # write JSON
    OUT_JSON.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding='utf-8')
    # write simple markdown grouped by layer
    md_lines = ['# Netflix Timeline by Layer\n']
    for layer, items in sorted(timeline['per_layer'].items()):
        md_lines.append(f'## {layer} ({len(items)})\n')
        for it in items:
            date = it['date'] or 'unknown'
            md_lines.append(f"- **{date}** — [{it['title']}]({it['path']}) — {it['snippet']}")
        md_lines.append('')
    OUT_MD.write_text('\n'.join(md_lines), encoding='utf-8')
    print('Wrote', OUT_JSON, 'and', OUT_MD)