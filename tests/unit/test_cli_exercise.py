from types import SimpleNamespace
from pathlib import Path
from src.presentation.cli import CLIController, PostFormatter
from src.application.use_cases.scrape_posts import ScrapePostsResponse


def test_cli_exercise_many_paths(monkeypatch, tmp_path):
    # Fake manager with a source and bulk config
    class FakeSource:
        def __init__(self):
            self.type = 'publication'
            self.name = 'p'
            self.description = 'd'
            self.auto_discover = False
            self.custom_domain = False

        def get_publication_name(self):
            return 'p'

    class FakeManager:
        def list_sources(self):
            return {'p': FakeSource()}

        def list_bulk_collections(self):
            return {'bulk': SimpleNamespace(description='b', sources=['p'])}

        def get_source(self, key):
            return FakeSource()

        def get_defaults(self):
            return {'output_dir': str(tmp_path), 'limit': 2, 'skip_session': True}

        def get_bulk_config(self, key):
            return SimpleNamespace(description='b', sources=['p'])

    monkeypatch.setattr('src.presentation.cli.SourceConfigManager', FakeManager)

    # Fake use case for controller
    class FakeUseCase:
        def execute(self, request, progress_callback=None):
            post = SimpleNamespace(
                id=SimpleNamespace(value='id1'), title='T', author=SimpleNamespace(name='A'),
                reading_time=3, published_at=None, slug='s', content_html='<p>ok</p>'
            )
            return ScrapePostsResponse(posts=[post], publication_config=SimpleNamespace(name='Pub'), total_posts_found=1, discovery_method='auto')

    # Monkeypatch extractor and persistence to safe no-ops
    monkeypatch.setattr('src.presentation.cli.content_extractor.html_to_markdown', lambda h: ('md', [], []))
    monkeypatch.setattr('src.presentation.cli.content_extractor.classify_technical', lambda h, c: {'is_technical': True, 'score': 0.5, 'reasons': []})
    monkeypatch.setattr('src.presentation.cli.persist_markdown_and_metadata', lambda *a, **k: {'markdown': str(tmp_path / '1.md')})

    controller = CLIController(FakeUseCase())

    # list_sources should not raise
    controller.list_sources()

    # scrape_from_config will call scrape_posts; monkeypatch scrape_posts to avoid nested heavy calls
    called = []

    def fake_scrape_posts(self, **kwargs):
        called.append(kwargs)

    monkeypatch.setattr('src.presentation.cli.CLIController.scrape_posts', fake_scrape_posts)
    controller.scrape_from_config(source_key='p')
    assert called

    # scrape_bulk_collection will iterate sources and call our fake scrape_from_config
    controller.scrape_bulk_collection(bulk_key='bulk')

    # Test _handle_successful_response for json and ids formats
    resp = ScrapePostsResponse(posts=[], publication_config=SimpleNamespace(name='Pub'), total_posts_found=0, discovery_method='none')
    # JSON format with output file -> _save_to_file path
    controller._handle_successful_response(resp, format_type='json', output_file=str(tmp_path / 'out.json'), mode='metadata')
    # ids format
    controller._handle_successful_response(resp, format_type='ids', output_file=None, mode='metadata')
