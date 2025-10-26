"""Persistence helpers for saving markdown, JSON metadata and assets for posts."""
from __future__ import annotations

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from urllib.parse import urlparse

from .http_transport import HttpxTransport


def _safe_filename(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or 'asset'
    return name.replace('?', '_').replace('&', '_')


def download_assets(assets: List[Dict], dest_dir: str) -> List[str]:
    """Download assets (images) to dest_dir. Returns list of local file paths."""
    os.makedirs(dest_dir, exist_ok=True)
    transport = HttpxTransport()
    saved = []
    for asset in assets:
        src = asset.get('src')
        if not src:
            continue
        filename = _safe_filename(src)
        out_path = os.path.join(dest_dir, filename)
        try:
            resp = transport.get(src, headers={}, follow_redirects=True)
            if resp.status_code == 200:
                with open(out_path, 'wb') as f:
                    f.write(resp.content)
                saved.append(out_path)
        except Exception:
            continue
    return saved


def persist_markdown_and_metadata(post, markdown: str, assets: List[Dict], output_dir: str) -> Dict[str, str]:
    """Persist markdown and JSON metadata for a post.

    post: domain Post entity (expects .id.value and .title)
    Returns dict with paths for markdown, json and assets_dir
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    post_key = getattr(post.id, 'value', str(post.id))
    safe_title = ''.join(c for c in getattr(post, 'title', post_key) if c.isalnum() or c in (' ', '-', '_')).strip()
    base_name = f"{post_key}_{safe_title[:40].replace(' ', '_')}"

    md_path = out_dir / f"{base_name}.md"
    json_path = out_dir / f"{base_name}.json"

    # Save markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    # Download assets into assets/<base_name>/
    assets_dir = out_dir / 'assets' / base_name
    saved_assets = []
    if assets:
        saved_assets = download_assets(assets, str(assets_dir))

    # Save metadata
    meta = {
        'id': post_key,
        'title': getattr(post, 'title', None),
        'slug': getattr(post, 'slug', None),
        'author': getattr(getattr(post, 'author', None), 'name', None),
        'markdown': str(md_path),
        'assets': saved_assets
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {'markdown': str(md_path), 'metadata': str(json_path), 'assets_dir': str(assets_dir)}


__all__ = ["persist_markdown_and_metadata", "download_assets"]
