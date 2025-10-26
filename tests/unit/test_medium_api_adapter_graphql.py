from types import SimpleNamespace
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter


def test_discover_via_graphql_user_profile_pagination():
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
                return FakeResp({
                    'data': {
                        'userResult': {
                            'homepagePostsConnection': {
                                'posts': [{'id': 'abcdef123456'}],
                                'pagingInfo': {'next': {'from': 'c1'}}
                            }
                        }
                    }
                })
            return FakeResp({
                'data': {
                    'userResult': {
                        'homepagePostsConnection': {
                            'posts': [{'id': '7890ab123456'}],
                            'pagingInfo': {}
                        }
                    }
                }
            })

    transport = FakeTransport()
    adapter = MediumApiAdapter(transport=transport)
    cfg = SimpleNamespace(id=SimpleNamespace(value='@me'), is_custom_domain=False, domain=None, graphql_url='u')
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
                return FakeResp({
                    'data': {
                        'publication': {
                            'posts': {
                                'edges': [{'node': {'id': '111111111111'}}],
                                'pageInfo': {'hasNextPage': True, 'endCursor': 'c1'}
                            }
                        }
                    }
                })
            return FakeResp({
                'data': {
                    'publication': {
                        'posts': {
                            'edges': [{'node': {'id': '222222222222'}}],
                            'pageInfo': {'hasNextPage': False}
                        }
                    }
                }
            })

    transport = FakeTransport()
    adapter = MediumApiAdapter(transport=transport)
    cfg = SimpleNamespace(id=SimpleNamespace(value='pub'), is_custom_domain=False, domain=None, graphql_url='u')
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
                data = {
                    'data': {
                        'publication': {
                            'publicationPostsConnection': {
                                'edges': [
                                    {
                                        'node': {
                                            'id': '333333333333',
                                            'title': 'T',
                                            'uniqueSlug': 's',
                                            'firstPublishedAt': 1,
                                            'latestPublishedAt': None,
                                            'readingTime': 1,
                                            'creator': {'id': 'a', 'name': 'A', 'username': 'a'},
                                            'extendedPreviewContent': {'subtitle': ''}
                                        }
                                    }
                                ],
                                'pageInfo': {'hasNextPage': True, 'endCursor': 'c1'}
                            }
                        }
                    }
                }
                return FakeResp(data)
            data = {'data': {'publication': {'publicationPostsConnection': {'edges': [], 'pageInfo': {'hasNextPage': False}}}}}
            return FakeResp(data)

    transport = FakeTransport()
    adapter = MediumApiAdapter(transport=transport)
    cfg = SimpleNamespace(is_custom_domain=False, id=SimpleNamespace(value='pub'), graphql_url='u')
    ids = adapter._discover_via_publication_all(cfg, limit=10)
    assert isinstance(ids, list)
