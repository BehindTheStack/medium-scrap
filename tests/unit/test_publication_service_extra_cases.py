from types import SimpleNamespace
from src.domain.services.publication_service import PostDiscoveryService


class RepoRaiseFeed:
    def discover_post_ids(self, config, limit):
        return []

    def get_posts_by_ids(self, ids, config):
        return []

    def get_posts_from_publication_feed(self, config, limit):
        raise Exception('feed error')

    def fetch_post_html(self, post, config):
        raise Exception('fetch error')


def test_discover_all_fallbacks_and_feed_exception():
    repo = RepoRaiseFeed()
    svc = PostDiscoveryService(repo)
    cfg = SimpleNamespace(has_known_posts=False, known_post_ids=[])
    posts = svc.discover_posts_intelligently(cfg, limit=5, prefer_auto_discovery=True)
    # feed exception should cause an empty list instead of crash
    assert posts == []


def test_progress_callback_exception_is_swallowed():
    # progress callback that raises
    events = []

    def bad_cb(e):
        events.append(e)
        raise RuntimeError('boom')

    class RepoThing:
        def discover_post_ids(self, config, limit):
            return ['x']

        def get_posts_by_ids(self, ids, config):
            return [SimpleNamespace(id=SimpleNamespace(value='x'), content_html=None)]

        def fetch_post_html(self, post, config):
            return '<p>ok</p>'

    svc = PostDiscoveryService(RepoThing())
    cfg = SimpleNamespace(has_known_posts=False, known_post_ids=[])
    # Should not raise despite callback raising
    posts = svc.discover_posts_intelligently(cfg, limit=1, prefer_auto_discovery=True, progress_callback=bad_cb)
    assert posts
