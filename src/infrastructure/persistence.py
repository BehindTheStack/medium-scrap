"""Persistence helpers for saving markdown, JSON metadata and assets for posts."""
from __future__ import annotations

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from urllib.parse import urlparse

from .http_transport import HttpxTransport
from .indexer import update_index
import re


def _safe_filename(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or 'asset'
    return name.replace('?', '_').replace('&', '_')


def download_assets(assets: List[Dict], dest_dir: str) -> List[str]:
    """Download assets (images) to dest_dir.

    Parameters
    ----------
    assets: List[Dict]
        List of asset descriptors. Each item should contain at least a 'src' key.
    dest_dir: str
        Destination directory where assets will be saved.

    Returns
    -------
    List[str]
        List of saved file paths (relative to the filesystem).
    """
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


def persist_markdown_and_metadata(post, markdown: str, assets: List[Dict], output_dir: str, *, code_blocks: Optional[List[Dict]] = None, classifier: Optional[Dict] = None) -> Dict[str, str]:
    """Persist markdown and JSON metadata for a post.

    This helper writes three primary artifacts for a post:
    - a Markdown file with the rendered content
    - a JSON metadata file containing title, slug, author, code_blocks and classifier
    - a local assets directory with downloaded images/files referenced by the post

    Parameters
    ----------
    post: object
        Domain Post entity (must expose .id.value and optional .title, .slug, .author).
    markdown: str
        The Markdown body to write.
    assets: List[Dict]
        Structured list of assets as produced by the extractor (each item should include 'src' and 'filename').
    output_dir: str
        Directory under which the artifacts will be stored.
    code_blocks: Optional[List[Dict]]
        Extracted code blocks to include in metadata.
    classifier: Optional[Dict]
        Classifier output (e.g., {'is_technical': True, 'score': 0.9, 'reasons': [...]})

    Returns
    -------
    Dict[str, str]
        Paths for written artifacts: {'markdown': ..., 'metadata': ..., 'assets_dir': ...}
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    post_key = getattr(post.id, 'value', str(post.id))
    safe_title = ''.join(c for c in getattr(post, 'title', post_key) if c.isalnum() or c in (' ', '-', '_')).strip()
    base_name = f"{post_key}_{safe_title[:40].replace(' ', '_')}"

    md_path = out_dir / f"{base_name}.md"
    json_path = out_dir / f"{base_name}.json"

    # Clean markdown to remove site navigation/footers introduced by HTML->MD conversion
    def _clean_markdown(md: str) -> str:
        """Light-weight, best-effort cleaning of converted markdown.

        - Start at first H1 if present
        - Remove common sign-in / nav short lines
        - Trim repeated 'Written by' blocks and 'Responses' sections
        - Collapse multiple blank lines
        """
        lines = md.splitlines()
        # start at first H1
        for i, ln in enumerate(lines):
            if ln.strip().startswith('# '):
                lines = lines[i:]
                break

        # drop obvious nav/footer short markers
        nav_markers = ['Sitemap', 'Open in app', 'Sign up', 'Sign in', 'Medium Logo', '[Write]', '[Search]']
        cleaned = []
        for ln in lines:
            if any(m.lower() in ln.lower() for m in nav_markers):
                continue
            if ln.strip() == '![]()':
                continue
            cleaned.append(ln)

        text = '\n'.join(cleaned)
        # remove duplicated author/byline panels and trailing responses/footer
        if '\n## Written by' in text:
            text = text.split('\n## Written by')[0]
        if 'See all responses' in text:
            text = text.split('See all responses')[0]
        # collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"

    cleaned_md = _clean_markdown(markdown)

    # Save markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_md)

    # Download assets into assets/<base_name>/
    assets_dir = out_dir / 'assets' / base_name
    saved_assets = []
    if assets:
        saved_assets = download_assets(assets, str(assets_dir))

    # Update inverted index (best-effort)
    try:
        post_meta = {
            'id': post_key,
            'title': getattr(post, 'title', None),
            'markdown': str(md_path),
            'metadata': str(json_path),
            'assets': saved_assets,
            'classifier': classifier or {}
        }
        update_index(os.path.join(output_dir, 'index.json'), post_meta)
    except Exception:
        # Don't fail persistence on index problems
        pass

    # Save metadata
    meta = {
        'id': post_key,
        'title': getattr(post, 'title', None),
        'slug': getattr(post, 'slug', None),
        'author': getattr(getattr(post, 'author', None), 'name', None),
        'markdown': str(md_path),
        'assets': saved_assets,
        'code_blocks': code_blocks or [],
        'classifier': classifier or {}
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {'markdown': str(md_path), 'metadata': str(json_path), 'assets_dir': str(assets_dir)}


__all__ = ["persist_markdown_and_metadata", "download_assets"]
