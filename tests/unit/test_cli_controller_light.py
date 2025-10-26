from click.testing import CliRunner
from src.presentation import cli as cli_module
from types import SimpleNamespace


def test_add_source_command_creates_entry(monkeypatch, tmp_path):
    # Patch SourceConfigManager to use a temp config path and avoid disk collisions
    class FakeManager:
        def __init__(self):
            self.config_path = str(tmp_path / 'medium_sources.yaml')

        def validate_source(self, key):
            return False

        def add_or_update_source(self, source_key, source_data):
            # write a small YAML file to simulate persistence
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(f"{source_key}: {source_data}\n")

    monkeypatch.setattr('src.presentation.cli.SourceConfigManager', FakeManager)

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ['add-source', '--key', 'pinterest', '--name', 'pinterest', '--description', 'desc'])
    assert result.exit_code == 0
    assert 'added/updated' in result.output.lower() or '✅' in result.output


def test_cli_mode_coercion_warns(monkeypatch):
    # Run cli with mode ids but format table to exercise coercion warnings
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ['--publication', 'netflix', '--mode', 'ids', '--format', 'table'])
    # exit_code 0 because controller early exits due to requiring publication flow running real adapters
    assert result.exit_code == 0
    assert 'mode' in result.output.lower()
