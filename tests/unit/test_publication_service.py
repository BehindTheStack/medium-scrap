from types import SimpleNamespace

from src.domain.services.publication_service import PostDiscoveryService


class FakeRepo:
    def __init__(self):
        self._ids = ['p1', 'p2']

    def discover_post_ids(self, config, limit):
        return self._ids[:limit]

    def get_posts_by_ids(self, ids, config):
        posts = []
        for i in ids:
            p = SimpleNamespace()
            p.id = SimpleNamespace(value=i)
            posts.append(p)
        return posts

    def get_posts_from_publication_feed(self, config, limit):
        p = SimpleNamespace()
        p.id = SimpleNamespace(value='feed1')
        return [p]

    def fetch_post_html(self, post, config):
        return '<html></html>'


def make_config(has_known=False):
    cfg = SimpleNamespace()
    cfg.has_known_posts = has_known
    cfg.known_post_ids = []
    return cfg


def test_discover_posts_intelligently_calls_progress_callback():
    repo = FakeRepo()
    svc = PostDiscoveryService(repo)
    events = []

    def cb(e):
        events.append(e)

    cfg = make_config(has_known=False)
    posts = svc.discover_posts_intelligently(cfg, limit=2, prefer_auto_discovery=True, progress_callback=cb)

    assert len(posts) == 2
    # Expect progress callback to have been called for discovered_ids and fetched_posts
    assert any(e.get('phase') == 'discovered_ids' for e in events)
    assert any(e.get('phase') == 'fetched_posts' for e in events)


def test_enrich_posts_with_html_reports_each_post():
    repo = FakeRepo()
    svc = PostDiscoveryService(repo)
    posts = repo.get_posts_by_ids(['p1', 'p2'], None)
    events = []

    def cb(e):
        events.append(e)

    enriched = svc.enrich_posts_with_html(posts, None, progress_callback=cb)
    assert len(enriched) == 2
    assert all(getattr(p, 'content_html', None) is not None for p in enriched)
    assert any(e.get('phase') == 'enriched_post' for e in events)
