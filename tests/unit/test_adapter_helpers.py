from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.domain.entities.publication import PostId


def test_build_queries_and_parse_post():
    adapter = MediumApiAdapter()

    pub_query = adapter._build_publication_query('netflix', limit=5, cursor=None)
    assert pub_query['operationName'] == 'PublicationPostsQuery'
    assert pub_query['variables']['publicationId'] == 'netflix'

    user_query = adapter._build_publication_query('@user', limit=3, cursor=None)
    assert user_query['operationName'] == 'UserProfileQuery'
    assert user_query['variables']['username'] == 'user'

    post_query = adapter._build_post_query(['aaaaaaaaaaaa'])
    assert post_query['operationName'].startswith('PublicationSectionPostsQuery')

    # Test _parse_post
    post_data = {
        'id': 'bbbbbbbbbbbb',
        'title': 'Title',
        'uniqueSlug': 'slug',
    'firstPublishedAt': int(datetime.now(timezone.utc).timestamp() * 1000),
        'readingTime': 3.5,
        'creator': {'id': 'a', 'name': 'Auth', 'username': 'auth'},
        'extendedPreviewContent': {'subtitle': 'sub'}
    }

    post = adapter._parse_post(post_data)
    assert post.id.value == 'bbbbbbbbbbbb'
    assert post.title == 'Title'
    assert post.reading_time == 3.5


def test_fetch_post_batch_uses_post_results(monkeypatch):
    adapter = MediumApiAdapter()

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {'data': {'postResults': [
        {'__typename': 'Post', 'id': 'cccccccccccc', 'title': 'T', 'uniqueSlug': 's', 'firstPublishedAt': 0, 'readingTime': 1.0, 'creator': {'id': 'a','name':'n','username':'u'}, 'extendedPreviewContent': {'subtitle': ''}}
    ]}}

    # Patch safe post to return our response
    monkeypatch.setattr(adapter, '_safe_http_post', lambda url, headers, payload: response)

    # Provide a minimal config-like object with graphql_url attribute
    class Cfg:
        graphql_url = 'https://example.com'

    posts = adapter._fetch_post_batch(['cccccccccccc'], Cfg(), {})
    assert len(posts) == 1
    assert posts[0].id.value == 'cccccccccccc'


def test_discover_via_graphql_user_profile(monkeypatch):
    adapter = MediumApiAdapter()

    # Mock a user profile response
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        'data': {
            'userResult': {
                'homepagePostsConnection': {
                    'posts': [
                        {'id': 'dddddddddddd', 'title': 'U', 'uniqueSlug': 'u', 'firstPublishedAt': 0, 'readingTime': 2.0, 'creator': {'id':'a','name':'n','username':'u'}}
                    ],
                    'pagingInfo': {'next': None}
                }
            }
        }
    }

    monkeypatch.setattr(adapter, '_safe_http_post', lambda url, headers, payload: resp)

    # Build a fake config-like object with id starting with @
    class Cfg:
        def __init__(self):
            self.id = type('X', (), {'value': '@user'})
            self.is_custom_domain = False
            self.domain = 'medium.com'
            self.graphql_url = 'https://medium.com/graphql'

    cfg = Cfg()

    ids = adapter._discover_via_graphql(cfg, limit=10)
    assert any(isinstance(i, PostId) for i in ids)
    assert 'dddddddddddd' in [i.value for i in ids]
