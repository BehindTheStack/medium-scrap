from unittest.mock import patch, MagicMock

from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.domain.entities.publication import PublicationId


@patch('httpx.Client')
def test_discover_via_publication_all_extracts_ids(mock_client_class):
    """_discover_via_publication_all should extract PostId values from publicationPostsConnection edges"""
    # Setup
    repo = InMemoryPublicationRepository()
    config = repo.get_by_id(PublicationId('netflix'))
    adapter = MediumApiAdapter()

    # Build a mocked response similar to what GraphQL returns
    mock_response_data = {
        "data": {
            "publication": {
                "publicationPostsConnection": {
                    "edges": [
                        {"node": {"id": "6dcc91058d8d", "title": "A", "uniqueSlug": "a", "firstPublishedAt": 0, "readingTime": 1.0, "creator": {"name": "N", "username": "n"}}},
                        {"node": {"id": "33073e260a38", "title": "B", "uniqueSlug": "b", "firstPublishedAt": 0, "readingTime": 2.0, "creator": {"name": "N", "username": "n"}}}
                    ],
                    "pageInfo": {"endCursor": "next_cursor_token", "hasNextPage": False}
                }
            }
        }
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data

    # Configure the client mock to return an object whose post() returns our mock_response
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    # The Client context manager should return mock_client
    mock_client_class.return_value.__enter__.return_value = mock_client

    # Execute
    post_ids = adapter._discover_via_publication_all(config, limit=10)

    # Assert we found the expected PostId values
    values = [p.value for p in post_ids]
    assert "6dcc91058d8d" in values
    assert "33073e260a38" in values


@patch('httpx.Client')
def test_discover_handles_invalid_ids_and_pagination(mock_client_class):
    """Ensure invalid IDs are skipped and pagination end is respected"""
    repo = InMemoryPublicationRepository()
    config = repo.get_by_id(PublicationId('netflix'))
    adapter = MediumApiAdapter()

    # First page contains one valid and one invalid id
    mock_response_page1 = MagicMock()
    mock_response_page1.status_code = 200
    mock_response_page1.json.return_value = {
        "data": {
            "publication": {
                "publicationPostsConnection": {
                    "edges": [
                        {"node": {"id": "ac15cada49ef", "title": "Valid", "uniqueSlug": "v", "firstPublishedAt": 0, "readingTime": 1.0, "creator": {}}},
                        {"node": {"id": "invalid_id!", "title": "Invalid", "uniqueSlug": "i", "firstPublishedAt": 0, "readingTime": 1.0, "creator": {}}}
                    ],
                    "pageInfo": {"endCursor": "cursor1", "hasNextPage": False}
                }
            }
        }
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response_page1
    mock_client_class.return_value.__enter__.return_value = mock_client

    post_ids = adapter._discover_via_publication_all(config, limit=10)

    # Only the valid id should be present
    values = [p.value for p in post_ids]
    assert "ac15cada49ef" in values
    assert not any(v == "invalid_id!" for v in values)
