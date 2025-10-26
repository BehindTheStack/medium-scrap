from types import SimpleNamespace
import builtins
import io
import pytest

from src.presentation.cli import CLIController
from src.application.use_cases.scrape_posts import ScrapePostsRequest, ScrapePostsResponse


def test_scrape_from_config_uses_custom_domain(monkeypatch):
    # Fake SourceConfigManager to return a source with custom_domain True
    class FakeSource:
        def __init__(self):
            self.description = 'd'
            self.name = 'custom.domain.com'
            self.custom_domain = True
            self.auto_discover = False

        def get_publication_name(self):
            return 'custom.domain.com'

    class FakeManager:
        def get_source(self, key):
            return FakeSource()

        def get_defaults(self):
            return {'limit': 5, 'output_dir': 'outputs', 'skip_session': True}

    monkeypatch.setattr('src.presentation.cli.SourceConfigManager', FakeManager)

    # Spy on CLIController.scrape_posts to capture call
    calls = []

    def fake_scrape_posts(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr('src.presentation.cli.CLIController.scrape_posts', fake_scrape_posts)

    controller = CLIController(scrape_posts_use_case=SimpleNamespace())
    controller.scrape_from_config(source_key='x')
    assert calls, 'scrape_posts should have been called'


def test_scrape_bulk_collection_calls_each(monkeypatch):
    class FakeBulk:
        def __init__(self):
            self.description = 'bulk'
            self.sources = ['a', 'b']

    class FakeManager:
        def get_bulk_config(self, key):
            return FakeBulk()

        def get_defaults(self):
            return {'limit': 2, 'output_dir': 'outputs'}

    monkeypatch.setattr('src.presentation.cli.SourceConfigManager', FakeManager)

    called = []

    def fake_scrape_from_config(self, source_key, **kwargs):
        called.append(source_key)

    monkeypatch.setattr('src.presentation.cli.CLIController.scrape_from_config', fake_scrape_from_config)

    controller = CLIController(scrape_posts_use_case=SimpleNamespace())
    controller.scrape_bulk_collection(bulk_key='bulk')
    assert called == ['a', 'b']


def test_execute_with_progress_fallback_no_callback(monkeypatch):
    # Use a FakeUseCase whose execute signature does not accept progress_callback
    class NoCallbackUseCase:
        def execute(self, request):
            return ScrapePostsResponse(posts=[], publication_config=None, total_posts_found=0, discovery_method='none')

    controller = CLIController(scrape_posts_use_case=NoCallbackUseCase())
    req = ScrapePostsRequest(publication_name='p')
    resp = controller._execute_with_progress(req, skip_session=True)
    assert isinstance(resp, ScrapePostsResponse)


def test_handle_failed_response_prints(monkeypatch, capsys):
    controller = CLIController(scrape_posts_use_case=SimpleNamespace())
    resp = ScrapePostsResponse(posts=[], publication_config=None, total_posts_found=0, discovery_method='none')
    controller._handle_failed_response(resp)
    captured = capsys.readouterr()
    assert 'No posts found' in captured.out or '❌' in captured.out


def test_save_to_file_error_swallowed(monkeypatch, tmp_path, capsys):
    controller = CLIController(scrape_posts_use_case=SimpleNamespace())

    # Monkeypatch builtins.open to raise
    def bad_open(*args, **kwargs):
        raise IOError('disk full')

    monkeypatch.setattr(builtins, 'open', bad_open)
    controller._save_to_file('content', str(tmp_path / 'file.txt'))
    captured = capsys.readouterr()
    assert 'Error saving file' in captured.out or 'Error' in captured.out
