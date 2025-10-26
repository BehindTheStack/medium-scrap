import yaml
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import MagicMock
import time

from src.presentation import cli as cli_module
from src.infrastructure.config.source_manager import SourceConfigManager
from src.domain.entities.publication import PublicationId, PublicationType, PublicationConfig, PostId
from src.application.use_cases.scrape_posts import ScrapePostsResponse
from src.domain.entities.publication import Author, Post
from datetime import datetime


def make_post():
    return Post(
        id=PostId('aaaaaaaaaaaa'),
        title='T',
        slug='s',
        author=Author(id='a', name='n', username='u'),
        published_at=datetime.utcnow(),
        reading_time=1.0
    )


def test_cli_list_and_scrape_from_config_and_bulk(tmp_path, monkeypatch):
    # Prepare temp YAML config
    cfg_file = tmp_path / 'medium_sources.yaml'
    data = {
        'sources': {
            'testsrc': {
                'type': 'publication',
                'name': 'testpub',
                'description': 'desc',
                'auto_discover': True,
                'custom_domain': False
            }
        },
        'defaults': {'limit': 5, 'output_dir': str(tmp_path)} ,
        'bulk_collections': {'b': {'description': 'bulk', 'sources': ['testsrc']}}
    }
    cfg_file.write_text(yaml.safe_dump(data), encoding='utf-8')

    # Monkeypatch SourceConfigManager in cli module to return manager pointing to tmp file
    monkeypatch.setattr(cli_module, 'SourceConfigManager', lambda : SourceConfigManager(config_path=cfg_file))

    # Fake use case that returns a response
    class FakeUseCase:
        def execute(self, request):
            return ScrapePostsResponse(posts=[make_post()], publication_config=PublicationConfig(
                id=PublicationId('testpub'), name='testpub', type=PublicationType.MEDIUM_HOSTED,
                domain='medium.com', graphql_url='https://x', known_post_ids=[]), total_posts_found=1, discovery_method='x')

    controller = cli_module.CLIController(FakeUseCase())

    # List sources
    controller.list_sources()

    # Monkeypatch time.sleep to fast-forward progress loops
    monkeypatch.setattr(time, 'sleep', lambda x: None)

    # scrape_from_config should run without raising
    controller.scrape_from_config('testsrc', limit=1, format_type='json', output_file=str(tmp_path / 'out.json'))

    # scrape_bulk_collection
    controller.scrape_bulk_collection('b', limit=1, format_type='json')


def test_cli_add_source_command(tmp_path, monkeypatch):
    cfg_file = tmp_path / 'medium_sources.yaml'
    cfg_file.write_text(yaml.safe_dump({'sources': {}, 'defaults': {}}), encoding='utf-8')

    monkeypatch.setattr(cli_module, 'SourceConfigManager', lambda : SourceConfigManager(config_path=cfg_file))

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ['add-source', '--key', 'p', '--name', 'pname', '--type', 'publication', '--description', 'd', '--yes'])
    assert result.exit_code == 0

    # Verify file updated
    content = yaml.safe_load(cfg_file.read_text(encoding='utf-8'))
    assert 'p' in content['sources']
