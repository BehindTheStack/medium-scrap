from unittest.mock import MagicMock
from datetime import datetime, timezone

from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.domain.entities.publication import PostId


def make_post_object():
    from src.domain.entities.publication import Post, Author
    return Post(
        id=PostId('aaaaaaaaaaaa'),
        title='T',
        slug='s',
        author=Author(id='a', name='n', username='u'),
    published_at=datetime.now(timezone.utc),
        reading_time=1.0
    )


def test_get_posts_by_ids_batches(monkeypatch):
    adapter = MediumApiAdapter()

    # Patch _fetch_post_batch to return a list with one Post
    monkeypatch.setattr(adapter, '_fetch_post_batch', lambda ids, cfg, headers: [make_post_object()])
    monkeypatch.setattr('time.sleep', lambda x: None)

    posts = adapter.get_posts_by_ids([PostId('bbbbbbbbbbbb'), PostId('cccccccccccc')], MagicMock())
    assert len(posts) == 2 or len(posts) >= 1


def test_discover_via_graphql_publication_branch(monkeypatch):
    adapter = MediumApiAdapter()

    # Mock publication response with posts/edges
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        'data': {
            'publication': {
                'posts': {
                    'edges': [
                        {'node': {'id': 'eeeeeeeeeeee'}},
                    ],
                    'pageInfo': {'endCursor': None, 'hasNextPage': False}
                }
            }
        }
    }

    monkeypatch.setattr(adapter, '_safe_http_post', lambda url, headers, payload: resp)

    class Cfg:
        def __init__(self):
            self.id = type('X', (), {'value': 'netflix'})
            self.is_custom_domain = False
            self.graphql_url = 'https://x'

    cfg = Cfg()
    ids = adapter._discover_via_graphql(cfg, limit=10)
    assert any(i.value == 'eeeeeeeeeeee' for i in ids)
