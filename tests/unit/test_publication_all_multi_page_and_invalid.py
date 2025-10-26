from unittest.mock import patch, MagicMock

from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.domain.entities.publication import PublicationId


def test_publication_all_multi_page_fetches_multiple_pages():
    """Adapter should follow hasNextPage and collect IDs across multiple pages"""
    repo = InMemoryPublicationRepository()
    config = repo.get_by_id(PublicationId('netflix'))
    adapter = MediumApiAdapter()

    page1 = {
        "data": {
            "publication": {
                "publicationPostsConnection": {
                    "edges": [
                        {"node": {"id": "aaaaaaaaaaaa", "title": "First"}}
                    ],
                    "pageInfo": {"endCursor": "cursor1", "hasNextPage": True}
                }
            }
        }
    }

    page2 = {
        "data": {
            "publication": {
                "publicationPostsConnection": {
                    "edges": [
                        {"node": {"id": "bbbbbbbbbbbb", "title": "Second"}}
                    ],
                    "pageInfo": {"endCursor": None, "hasNextPage": False}
                }
            }
        }
    }

    mock_resp1 = MagicMock()
    mock_resp1.json.return_value = page1
    mock_resp1.status_code = 200

    mock_resp2 = MagicMock()
    mock_resp2.json.return_value = page2
    mock_resp2.status_code = 200

    with patch('httpx.post') as mock_post:
        mock_post.side_effect = [mock_resp1, mock_resp2]
        post_ids = adapter._discover_via_publication_all(config, limit=10)

    values = [p.value for p in post_ids]
    assert "aaaaaaaaaaaa" in values
    assert "bbbbbbbbbbbb" in values
    assert len(values) == 2


def test_publication_all_skips_invalid_ids_and_continues():
    """Adapter should skip invalid Post IDs and still return valid ones"""
    repo = InMemoryPublicationRepository()
    config = repo.get_by_id(PublicationId('netflix'))
    adapter = MediumApiAdapter()

    page = {
        "data": {
            "publication": {
                "publicationPostsConnection": {
                    "edges": [
                        {"node": {"id": "cccccccccccc", "title": "Good"}},
                        {"node": {"id": "invalid", "title": "Bad"}},
                    ],
                    "pageInfo": {"endCursor": None, "hasNextPage": False}
                }
            }
        }
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = page
    mock_resp.status_code = 200

    with patch('httpx.post') as mock_post:
        mock_post.return_value = mock_resp
        post_ids = adapter._discover_via_publication_all(config, limit=10)

    values = [p.value for p in post_ids]
    assert "cccccccccccc" in values
    # invalid should be skipped
    assert "invalid" not in values
    assert len(values) == 1
