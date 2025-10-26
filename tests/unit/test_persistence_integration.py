import json
import os
from types import SimpleNamespace

import pytest

from src.infrastructure import persistence


class FakeResp:
    def __init__(self, status_code=200, content=b'bytes'):
        self.status_code = status_code
        self.content = content


class FakeTransport:
    def get(self, url, headers=None, follow_redirects=True):
        return FakeResp(200, b'PNGDATA')


def make_post():
    p = SimpleNamespace()
    p.id = SimpleNamespace(value='abc123')
    p.title = 'Test Post'
    p.slug = 'test-post'
    p.author = SimpleNamespace(name='Author')
    return p


def test_persist_markdown_and_metadata_writes_files(tmp_path, monkeypatch):
    # Patch transport used by persistence
    monkeypatch.setattr(persistence, 'HttpxTransport', lambda: FakeTransport())

    post = make_post()
    md = '# Hello\nThis is content'
    assets = [{'src': 'http://example.com/image.png', 'filename': 'image.png', 'alt': 'img'}]

    out = persistence.persist_markdown_and_metadata(post, md, assets, str(tmp_path))

    # Paths returned
    assert os.path.exists(out['markdown'])
    assert os.path.exists(out['metadata'])
    assert os.path.exists(out['assets_dir'])

    # Metadata content
    with open(out['metadata'], 'r', encoding='utf-8') as f:
        meta = json.load(f)
    assert meta['id'] == 'abc123'
    assert meta['title'] == 'Test Post'
    assert isinstance(meta['code_blocks'], list)

    # Index updated (best-effort)
    idx = tmp_path / 'index.json'
    assert idx.exists()
    with open(idx, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Expect tokens derived from title or content
    assert isinstance(data, dict)
