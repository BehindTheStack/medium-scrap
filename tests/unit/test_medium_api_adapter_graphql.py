from types import SimpleNamespace
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter


def test_discover_via_graphql_user_profile_pagination():
    # Simulate two-page user profile responses
    class FakeResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def json(self):
            return self._data

    class FakeTransport:
        def __init__(self):
            self.calls = 0

        def post(self, url, headers=None, json=None):
            self.calls += 1
            if self.calls == 1:
                return FakeResp({'data': {'userResult': {'homepagePostsConnection': {'posts': [{'id': 'a1'}], 'pagingInfo': {'next': {'from': 'c1'}}}}}})
            return FakeResp({'data': {'userResult': {'homepagePostsConnection': {'posts': [{'id': 'b2'}], 'pagingInfo': {}}}}})

    transport = FakeTransport()
    adapter = MediumApiAdapter(transport=transport)
    cfg = SimpleNamespace(id=SimpleNamespace(value='@me'))
    ids = adapter._discover_via_graphql(cfg, limit=10)
    assert len(ids) >= 1


def test_discover_via_graphql_publication_pagination():
    class FakeResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def json(self):
            return self._data

    class FakeTransport:
        def __init__(self):
            self.calls = 0

        def post(self, url, headers=None, json=None):
            self.calls += 1
            if self.calls == 1:
                return FakeResp({'data': {'publication': {'posts': {'edges': [{'node': {'id': 'p1'}}], 'pageInfo': {'hasNextPage': True, 'endCursor': 'c1'}}}}})
            return FakeResp({'data': {'publication': {'posts': {'edges': [{'node': {'id': 'p2'}}], 'pageInfo': {'hasNextPage': False}}}}})

    transport = FakeTransport()
    adapter = MediumApiAdapter(transport=transport)
    cfg = SimpleNamespace(id=SimpleNamespace(value='pub'))
    ids = adapter._discover_via_graphql(cfg, limit=10)
    assert len(ids) >= 1


def test_discover_via_publication_all_pagination():
    class FakeResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def json(self):
            return self._data

    class FakeTransport:
        def __init__(self):
            self.calls = 0

        def post(self, url, headers=None, json=None):
            self.calls += 1
            if self.calls == 1:
                data = {'data': {'publication': {'publicationPostsConnection': {'edges': [{'node': {'id': 'x1', 'title': 'T', 'uniqueSlug': 's', 'firstPublishedAt': 1, 'latestPublishedAt': None, 'readingTime': 1, 'creator': {'id': 'a', 'name': 'A', 'username': 'a'}, 'extendedPreviewContent': {'subtitle': ''}}], 'pageInfo': {'hasNextPage': True, 'endCursor': 'c1'}}}}}
                return FakeResp(data)
            data = {'data': {'publication': {'publicationPostsConnection': {'edges': [], 'pageInfo': {'hasNextPage': False}}}}}
            return FakeResp(data)

    transport = FakeTransport()
    adapter = MediumApiAdapter(transport=transport)
    cfg = SimpleNamespace(is_custom_domain=False, id=SimpleNamespace(value='pub'), graphql_url='u')
    ids = adapter._discover_via_publication_all(cfg, limit=10)
    # Should return list of PostId-like objects
    assert isinstance(ids, list)
