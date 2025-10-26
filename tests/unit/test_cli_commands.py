import yaml
from click.testing import CliRunner
from unittest.mock import MagicMock
import time

from src.presentation import cli as cli_module
from src.infrastructure.config.source_manager import SourceConfigManager
from src.application.use_cases.scrape_posts import ScrapePostsResponse
from src.domain.entities.publication import PublicationConfig, PublicationId, PublicationType, PostId
from src.domain.entities.publication import Author, Post
from datetime import datetime, timezone


def make_post():
    return Post(
        id=PostId('aaaaaaaaaaaa'),
        title='T',
        slug='s',
        author=Author(id='a', name='n', username='u'),
        published_at=datetime.now(timezone.utc),
        reading_time=1.0
    )


def test_cli_no_args_shows_error():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, [])
    assert result.exit_code == 0
    assert 'Must specify --publication' in result.output


def test_cli_publication_triggers_scrape(tmp_path, monkeypatch):
    # Monkeypatch the various classes used in cli() to simple factories
    monkeypatch.setattr('src.infrastructure.adapters.medium_api_adapter.MediumApiAdapter', lambda: MagicMock())
    monkeypatch.setattr('src.infrastructure.external.repositories.InMemoryPublicationRepository', lambda: MagicMock())
    monkeypatch.setattr('src.infrastructure.external.repositories.MediumSessionRepository', lambda: MagicMock())
    monkeypatch.setattr('src.domain.services.publication_service.PostDiscoveryService', lambda repo: MagicMock())
    monkeypatch.setattr('src.domain.services.publication_service.PublicationConfigService', lambda repo: MagicMock())

    # Replace ScrapePostsUseCase so cli() will instantiate our fake that returns a response
    class FakeUseCase:
        def __init__(self, *a, **k):
            pass

        def execute(self, request):
            cfg = PublicationConfig(
                id=PublicationId('testpub'),
                name='testpub',
                type=PublicationType.MEDIUM_HOSTED,
                domain='medium.com',
                graphql_url='https://x',
                known_post_ids=[]
            )
            return ScrapePostsResponse(posts=[make_post()], publication_config=cfg, total_posts_found=1, discovery_method='x')

    monkeypatch.setattr('src.presentation.cli.ScrapePostsUseCase', FakeUseCase)

    runner = CliRunner()
    out_file = tmp_path / 'out.json'
    result = runner.invoke(cli_module.cli, ['--publication', 'testpub', '--limit', '1', '--format', 'json', '--output', str(out_file)])
    assert result.exit_code == 0
    # file should be created by CLI flow
    assert out_file.exists()


def test_cli_all_flag_prints_warning(monkeypatch):
    # Patch the heavy constructors to no-op
    monkeypatch.setattr('src.infrastructure.adapters.medium_api_adapter.MediumApiAdapter', lambda: MagicMock())
    monkeypatch.setattr('src.infrastructure.external.repositories.InMemoryPublicationRepository', lambda: MagicMock())
    monkeypatch.setattr('src.infrastructure.external.repositories.MediumSessionRepository', lambda: MagicMock())
    monkeypatch.setattr('src.domain.services.publication_service.PostDiscoveryService', lambda repo: MagicMock())
    monkeypatch.setattr('src.domain.services.publication_service.PublicationConfigService', lambda repo: MagicMock())

    class FakeUseCase:
        def __init__(self, *a, **k):
            pass
        def execute(self, request):
            cfg = PublicationConfig(
                id=PublicationId('testpub'), name='testpub', type=PublicationType.MEDIUM_HOSTED, domain='medium.com', graphql_url='https://x', known_post_ids=[]
            )
            return ScrapePostsResponse(posts=[make_post()], publication_config=cfg, total_posts_found=1, discovery_method='x')

    monkeypatch.setattr('src.presentation.cli.ScrapePostsUseCase', FakeUseCase)

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ['--publication', 'testpub', '--all'])
    assert result.exit_code == 0
    assert 'Collecting ALL posts' in result.output


def test_add_source_overwrite_prompt_cancel(tmp_path):
    # Create a config with an existing source 'p'
    cfg_file = tmp_path / 'medium_sources.yaml'
    cfg_file.write_text(yaml.safe_dump({'sources': {'p': {'type': 'publication', 'name': 'pname', 'description': '', 'auto_discover': True, 'custom_domain': False}}, 'defaults': {}}), encoding='utf-8')

    # Monkeypatch manager creation to use our file
    monkey = MagicMock()
    # Use real manager so validate_source and add_or_update_source behave
    monkeypatch_manager = lambda: SourceConfigManager(config_path=cfg_file)

    # Replace the manager inside cli_module for this test
    cli_module.SourceConfigManager = monkeypatch_manager

    runner = CliRunner()
    # Provide 'n' input to the confirmation prompt
    result = runner.invoke(cli_module.cli, ['add-source', '--key', 'p', '--name', 'pname2', '--type', 'publication', '--description', 'd'], input='n\n')
    assert result.exit_code == 0
    assert 'Cancelled' in result.output
