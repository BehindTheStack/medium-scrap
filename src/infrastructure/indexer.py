"""Simple inverted index for persisted posts.

Index format (JSON):
{
  "terms": {"word": [{"id": "<post_id>", "title": "...", "path": "..."}, ...]},
  "meta": {"posts": {"<post_id>": {"title":..., "path":...}}}
}

This is intentionally small and filesystem-backed (JSON) so it is easy to
inspect and extend. Tokenization is minimal (split on non-alphanum).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def update_index(index_path: str, post_meta: Dict[str, Any]):
    """Update (or create) the index with post_meta.

    post_meta expected fields: id, title, markdown (path), classifier (optional), assets
    """
    p = Path(index_path)
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
    else:
        data = {"terms": {}, "meta": {"posts": {}}}

    post_id = post_meta.get('id')
    title = post_meta.get('title') or ''
    path = post_meta.get('markdown') or post_meta.get('metadata')

    # store meta
    data.setdefault('meta', {}).setdefault('posts', {})[post_id] = {"title": title, "path": path}

    # index tokens from title and optionally classifier reasons
    tokens = set(_tokenize(title))
    # also index classifier reasons if present
    classifier = post_meta.get('classifier') or {}
    for r in classifier.get('reasons', []):
        tokens.update(_tokenize(r))

    for t in tokens:
        postings = data['terms'].setdefault(t, [])
        # Avoid duplicates
        if not any(p.get('id') == post_id for p in postings):
            postings.append({"id": post_id, "title": title, "path": path})

    # Save back
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


__all__ = ["update_index"]
