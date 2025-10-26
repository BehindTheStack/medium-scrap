import yaml
from pathlib import Path

from src.infrastructure.config.source_manager import SourceConfigManager


def test_add_and_get_source(tmp_path):
    cfg_file = tmp_path / 'medium_sources.yaml'
    # Create initial minimal config
    initial = {'sources': {}, 'defaults': {'limit': 10, 'output_dir': 'outputs'}}
    cfg_file.write_text(yaml.safe_dump(initial), encoding='utf-8')

    manager = SourceConfigManager(config_path=cfg_file)
    manager.add_or_update_source('testkey', {
        'type': 'publication',
        'name': 'testname',
        'description': 'desc',
        'auto_discover': True,
        'custom_domain': False
    })

    src = manager.get_source('testkey')
    assert src.name == 'testname'

    all_sources = manager.list_sources()
    assert 'testkey' in all_sources

    defaults = manager.get_defaults()
    assert defaults.get('limit') == 10
