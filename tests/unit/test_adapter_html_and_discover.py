from unittest.mock import patch, MagicMock
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.domain.entities.publication import PublicationId


def test_discover_post_ids_prioritizes_publication_all(monkeypatch):
    adapter = MediumApiAdapter()

    # Patch methods to simulate first strategy returning ids
    monkeypatch.setattr(adapter, '_discover_via_publication_all', lambda config, limit: [PublicationId('netflix')])
    monkeypatch.setattr(adapter, '_discover_via_graphql', lambda config, limit: [])
    monkeypatch.setattr(adapter, '_discover_via_html_scraping', lambda config, limit: [])

    class Cfg:
        pass
    cfg = Cfg()
    cfg.id = type('X', (), {'value':'netflix'})

    ids = adapter.discover_post_ids(cfg, limit=5)
    # We return PublicationId instances if first method returns them; ensure truthy
    assert ids


def test_discover_via_html_scraping_extracts_ids(monkeypatch):
    adapter = MediumApiAdapter()

    # Simulate html content with two 12-char hex IDs
    html = '... 61a538d717a9 ... e027cb313f8f ...'

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html

    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp

    # Patch httpx.Client to context manager returning our mock_client
    class FakeClient:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return mock_client
        def __exit__(self, exc_type, exc, tb):
            return False

    with patch('httpx.Client', return_value=FakeClient()):
        class Cfg:
            is_custom_domain = False
            id = type('X', (), {'value': 'somepub'})
        cfg = Cfg()
        ids = adapter._discover_via_html_scraping(cfg, limit=5)
        assert any(i.value == '61a538d717a9' for i in ids)
        assert any(i.value == 'e027cb313f8f' for i in ids)
