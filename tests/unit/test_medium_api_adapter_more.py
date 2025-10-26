from types import SimpleNamespace
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from datetime import datetime, timezone


def test_build_queries_and_headers():
    adapter = MediumApiAdapter(transport=SimpleNamespace())
    post_query = adapter._build_post_query(['p1', 'p2'])
    assert 'operationName' in post_query and 'variables' in post_query

    pub_query = adapter._build_publication_query('@user', 10, cursor='c')
    assert pub_query['operationName'] == 'UserProfileQuery'

    pub_query2 = adapter._build_publication_query('publication', 5)
    assert pub_query2['operationName'] == 'PublicationPostsQuery'

    cfg = SimpleNamespace(is_custom_domain=True, domain='ex.com')
    headers = adapter._get_headers_for_config(cfg)
    assert 'origin' in headers and 'referer' in headers


def test_parse_post_and_fetch_batch(monkeypatch):
    # Create a fake transport that returns a JSON payload
    class FakeResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def json(self):
            return self._data

    class FakeTransport:
        def post(self, url, headers=None, json=None):
            payload = {
                'data': {
                    'postResults': [
                        {
                            '__typename': 'Post',
                            'id': 'abcdef123456',
                            'title': 'T',
                            'uniqueSlug': 'slug',
                            'firstPublishedAt': int(datetime.now(timezone.utc).timestamp() * 1000),
                            'latestPublishedAt': None,
                            'readingTime': 3,
                            'creator': {'id': 'a', 'name': 'Auth', 'username': 'auth'},
                            'extendedPreviewContent': {'subtitle': 's'}
                        }
                    ]
                }
            }
            return FakeResp(payload)

    adapter = MediumApiAdapter(transport=FakeTransport())
    # create a minimal config with graphql_url
    cfg = SimpleNamespace(graphql_url='http://x', is_custom_domain=False, id=SimpleNamespace(value='pub'))
    posts = adapter._fetch_post_batch(['abcdef123456'], cfg, headers={})
    assert len(posts) == 1
    p = posts[0]
    assert p.slug == 'slug' and p.title == 'T'


def test_discover_via_html_scraping(monkeypatch):
    # Fake transport.get to return HTML with 12-char hex ids
    class FakeResp:
        def __init__(self, text):
            self.status_code = 200
            self.text = text

    class FakeTransport:
        def __init__(self):
            self.calls = 0

        def get(self, url, headers=None, follow_redirects=True):
            self.calls += 1
            if self.calls > 1:
                return FakeResp('')
            # include two 12-char hex ids
            return FakeResp('abcdef123456 and 7890ab123456')

    transport = FakeTransport()
    adapter = MediumApiAdapter(transport=transport)
    cfg = SimpleNamespace(is_custom_domain=False, id=SimpleNamespace(value='pub'), domain=None)
    ids = adapter._discover_via_html_scraping(cfg, limit=5)
    assert len(ids) >= 1
