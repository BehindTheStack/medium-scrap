import json
from pathlib import Path

from src.infrastructure.indexer import update_index


def test_update_index(tmp_path):
    idx = tmp_path / 'index.json'
    post_meta = {'id': 'abc123', 'title': 'Hello World', 'markdown': '/tmp/abc.md', 'classifier': {'reasons': ['code_blocks:1']}}
    update_index(str(idx), post_meta)

    assert idx.exists()
    data = json.loads(idx.read_text(encoding='utf-8'))
    assert 'hello' in data['terms']
    assert any(p['id'] == 'abc123' for p in data['terms']['hello'])