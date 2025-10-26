from types import SimpleNamespace
import pytest

from src.domain.services.publication_service import PostDiscoveryService


class DummyRepo:
    def __init__(self, discover_ids=None, posts_by_id=None):
        self._discover_ids = discover_ids or []
        self._posts_by_id = posts_by_id or {}

    def discover_post_ids(self, config, limit):
        return self._discover_ids[:limit] if limit is not None else self._discover_ids

    def get_posts_by_ids(self, ids, config):
        return [self._posts_by_id[i] for i in ids if i in self._posts_by_id]

    def get_posts_from_publication_feed(self, config, limit):
        # return empty to simulate last-resort path
        return []

    def fetch_post_html(self, post, config):
        if getattr(post, 'id', None) and post.id.value in self._posts_by_id:
            return '<p>ok</p>'
        raise Exception('not found')


def make_post(id_value, title='t'):
    # Minimal Post-like object with expected attributes
    return SimpleNamespace(id=SimpleNamespace(value=id_value), title=title, content_html=None)


def test_discover_posts_prefers_auto(monkeypatch):
    # repo will discover two ids
    p1 = make_post('a1')
    p2 = make_post('b2')
    repo = DummyRepo(discover_ids=['a1','b2'], posts_by_id={'a1': p1, 'b2': p2})
    service = PostDiscoveryService(repo)

    events = []

    def cb(e):
        events.append(e)

    cfg = SimpleNamespace(has_known_posts=False, known_post_ids=[])
    posts = service.discover_posts_intelligently(cfg, limit=10, prefer_auto_discovery=True, progress_callback=cb)
    assert len(posts) == 2
    assert any(ev.get('phase') == 'discovered_ids' for ev in events)


def test_known_ids_fallback():
    # when discover returns nothing but config has known posts
    p3 = make_post('k1')
    repo = DummyRepo(discover_ids=[], posts_by_id={'k1': p3})
    service = PostDiscoveryService(repo)
    cfg = SimpleNamespace(has_known_posts=True, known_post_ids=['k1'])
    posts = service.discover_posts_intelligently(cfg, limit=5, prefer_auto_discovery=False)
    assert len(posts) == 1


def test_enrich_posts_with_html(monkeypatch):
    p = make_post('x1')
    repo = DummyRepo(posts_by_id={'x1': p})
    service = PostDiscoveryService(repo)

    events = []

    def cb(e):
        events.append(e)

    cfg = SimpleNamespace()
    enriched = service.enrich_posts_with_html([p], cfg, progress_callback=cb)
    assert len(enriched) == 1
    assert enriched[0].content_html == '<p>ok</p>'
    assert any(e.get('phase') == 'enriched_post' for e in events)
