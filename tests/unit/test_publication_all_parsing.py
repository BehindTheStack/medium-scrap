from unittest.mock import patch, MagicMock

from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.domain.entities.publication import PublicationId


def test_publication_all_parsing_extracts_ids():
    """Focused unit test: mocked PublicationContentDataQuery payload should yield post IDs"""
    repo = InMemoryPublicationRepository()
    config = repo.get_by_id(PublicationId('netflix'))
    adapter = MediumApiAdapter()

    mock_response_page1 = {
        "data": {
            "publication": {
                "publicationPostsConnection": {
                    "edges": [
                        {
                            "node": {
                                "id": "6dcc91058d8d",
                                "title": "From Facts & Metrics to Media Machine Learning",
                                "uniqueSlug": "from-facts-metrics-to-media-machine-learning-6dcc91058d8d",
                                "firstPublishedAt": 1724236800000,
                                "readingTime": 5.0,
                                "creator": {
                                    "name": "Netflix Technology Blog",
                                    "username": "netflixtechblog"
                                }
                            }
                        },
                        {
                            "node": {
                                "id": "33073e260a38",
                                "title": "ML Observability: Bringing Transparency to Payments and Beyond",
                                "uniqueSlug": "ml-observability-33073e260a38",
                                "firstPublishedAt": 1724062800000,
                                "readingTime": 9.0,
                                "creator": {
                                    "name": "Netflix Technology Blog",
                                    "username": "netflixtechblog"
                                }
                            }
                        }
                    ],
                    "pageInfo": {
                        "endCursor": "next_cursor_token",
                        "hasNextPage": False
                    }
                }
            }
        }
    }

    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_page1
    mock_response.status_code = 200

    with patch('httpx.post') as mock_post:
        mock_post.return_value = mock_response
        post_ids = adapter._discover_via_publication_all(config, limit=10)

    # We expect the two IDs to be discovered and returned as PostId objects
    values = [p.value for p in post_ids]
    assert "6dcc91058d8d" in values
    assert "33073e260a38" in values
