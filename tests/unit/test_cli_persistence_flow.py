from types import SimpleNamespace
import time
from src.presentation.cli import CLIController
from src.application.use_cases.scrape_posts import ScrapePostsResponse


def test_cli_persistence_flow(monkeypatch, tmp_path):
    # Prepare two fake posts: one with HTML and one without
    post_with_html = SimpleNamespace(
        id=SimpleNamespace(value='id1'),
        title='This is a long title that will be truncated by the UI for display purposes',
        slug='slug-1',
        reading_time=5,
        content_html='<p>Hello</p>',
        author=SimpleNamespace(name='Auth'),
        published_at=None
    )

    post_no_html = SimpleNamespace(
        id=SimpleNamespace(value='id2'),
        title='No HTML here',
        slug='s2',
        reading_time=None,
        content_html=None,
        author=SimpleNamespace(name='B'),
        published_at=None
    )

    # Fake use case that returns both posts
    class FakeUseCase:
        def execute(self, request, progress_callback=None):
            return ScrapePostsResponse(
                posts=[post_with_html, post_no_html],
                publication_config=SimpleNamespace(name='TestPub'),
                total_posts_found=2,
                discovery_method='test'
            )

    # Monkeypatch SourceConfigManager to provide output defaults
    class FakeManager:
        def get_defaults(self):
            return {'output_dir': str(tmp_path)}

        def list_sources(self):
            return {}

        def list_bulk_collections(self):
            return {}

    monkeypatch.setattr('src.presentation.cli.SourceConfigManager', FakeManager)

    # Capture calls to persistence
    persisted = []

    def fake_persist(post, md, assets, output_dir, code_blocks=None, classifier=None):
        out = str(tmp_path / f"{post.id.value}.md")
        persisted.append({'post': post.id.value, 'md': out})
        return {'markdown': out}

    monkeypatch.setattr('src.presentation.cli.persist_markdown_and_metadata', fake_persist)

    # Monkeypatch extractor
    def fake_md(html):
        return ('# title', [], [{'code': 'c', 'language': 'python'}])

    monkeypatch.setattr('src.presentation.cli.content_extractor.html_to_markdown', fake_md)
    monkeypatch.setattr('src.presentation.cli.content_extractor.classify_technical', lambda h, c: {'is_technical': True, 'score': 0.8, 'reasons': []})

    controller = CLIController(FakeUseCase())

    # Run the flow (mode 'full' triggers persistence)
    controller.scrape_posts(publication='TestPub', limit=2, format_type='table', custom_ids=None, auto_discover=False, skip_session=True, output_file=None, mode='full')

    # Ensure persist was called once (only for the post that had HTML)
    assert len(persisted) == 1
    assert persisted[0]['post'] == 'id1'
