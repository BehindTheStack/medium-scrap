# Universal Medium Scraper — Complete Edition

This repository provides a CLI tool to scrape posts from Medium publications and custom domains (for example, engineering blogs). It follows Clean Architecture and provides:

- Intelligent discovery (auto-discovery, known IDs, fallbacks)
- Support for custom domains and usernames
- A rich CLI (progress indicators and formatted output via Rich)
- YAML-based configuration for sources and bulk collections

This README is written to be fully reproducible: installation, configuration, commands, testing and troubleshooting are covered below.

Table of contents
- Overview
- Quick start
- Configuration (medium_sources.yaml)
- CLI usage and examples
- How it works (high-level)
- Tests
- Troubleshooting
- Project layout
- Contributing and license

## Overview

- Entry point: `main.py` (calls `src.presentation.cli.cli()`)
- Config file: `medium_sources.yaml` (expected in repo root)
- Default output folder: `outputs/`
- Python: 3.9+ (see `pyproject.toml`)

This tool resolves publication definitions, discovers post IDs (auto-discovery or fallbacks), fetches post details via adapters, and presents results in table/JSON/IDs formats.

## Quick start

1. Create and activate a virtual environment (Linux/macOS):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install editable package (recommended):

```bash
pip install --upgrade pip
pip install -e .
```

3. Verify CLI is available and view help:

```bash
python main.py --help
```

Notes:
- Main dependencies are declared in `pyproject.toml` (httpx, click, rich, pyyaml, pytest, requests, etc.).
- Installing in editable mode lets you make code changes without reinstalling.

## Configuration (`medium_sources.yaml`)

The file `medium_sources.yaml` configures named sources and bulk collections. The included example in the repo contains many predefined keys (netflix, nytimes, airbnb, etc.).

Minimal example structure:

```yaml
sources:
	netflix:
		type: publication
		name: netflix
		description: "Netflix Technology Blog"
		auto_discover: true
		custom_domain: false

defaults:
	limit: 50
	skip_session: true
	format: json
	output_dir: "outputs"

bulk_collections:
	tech_giants:
		description: "Major tech blogs"
		sources: [netflix, airbnb]
```

Key notes:
- `type`: `publication` or `username`.
- `name`: publication name, domain, or `@username`. If `type` is `username` and `name` lacks `@`, the code will add it.
- `custom_domain`: set to `true` for domains like `open.nytimes.com`.

Use `python main.py --list-sources` to list keys and descriptions from the YAML file.

## CLI usage and examples

Run from project root or after installing into your venv.

- List configured sources:

```bash
python main.py --list-sources
```

- Scrape a configured source and save JSON:

```bash
python main.py --source nytimes --limit 20 --format json --output outputs/nytimes_posts.json
```

- Scrape a publication directly:

```bash
python main.py --publication netflix --limit 5 --format table
```

- Bulk collection (group from YAML):

```bash
python main.py --bulk tech_giants --limit 10 --format json
```

- Auto-discovery and skip session (production-ready):

```bash
python main.py --publication pinterest --auto-discover --skip-session --format json --output results.json
```

- Custom post IDs (comma-separated). Each must be exactly 12 alphanumeric characters:

```bash
python main.py --publication netflix --custom-ids "ac15cada49ef,64c786c2a3ac" --format json
```

Flags summary:

- `-p, --publication` TEXT
- `-s, --source` TEXT
- `-b, --bulk` TEXT
- `--list-sources`
- `-o, --output` TEXT
- `-f, --format` [table|json|ids]
- `--custom-ids` TEXT (comma-separated)
- `--skip-session`
- `--limit` INTEGER
- `--all` (collect all posts)

### Managing sources via CLI

You can add or update sources directly from the CLI using the `add-source` subcommand. This writes to `medium_sources.yaml` and is useful when you want to quickly register a publication without editing the YAML manually.

Example — add Pinterest:

```bash
python main.py add-source \
	--key pinterest \
	--type publication \
	--name pinterest \
	--description "Pinterest Engineering" \
	--auto-discover
```

Notes:
- `add-source` persists the change to `medium_sources.yaml` in the repo root.
- The command is implemented to avoid loading optional network adapters, so it can run even if dependencies like `httpx` are not installed.
- To see the result, run `python main.py --list-sources`.

Interactive behavior and safety

- If the source key you pass already exists in `medium_sources.yaml`, the CLI will ask for confirmation before overwriting. This prevents accidental data loss when updating an existing source.

	Example (interactive prompt shown):

	```text
	$ python main.py add-source --key pinterest --type publication --name pinterest --description "Pinterest Engineering"
	Source 'pinterest' already exists. Overwrite? [y/N]: y
	✅ Source 'pinterest' added/updated in medium_sources.yaml
	```

- To skip the interactive prompt and assume confirmation, use `--yes` (or `-y`):

	```bash
	python main.py add-source --key pinterest --type publication --name pinterest --description "Pinterest Engineering" --yes
	```

- The CLI subcommand writes a normalized YAML entry (ensures booleans and required keys). It creates the `sources` block if it does not exist.

- After adding/updating a source you can:
	- run `python main.py --list-sources` to see the configured keys and descriptions; or
	- open `medium_sources.yaml` to inspect the persisted entry.

Implementation notes (for maintainers)

- The `add-source` subcommand avoids importing network adapters (e.g. `httpx`) when invoked so it can be used on systems where optional runtime dependencies are not installed.
- The command is implemented in `src/presentation/cli.py` and uses `SourceConfigManager.add_or_update_source` (in `src/infrastructure/config/source_manager.py`) to persist changes.


## How it works (high-level)

1. The CLI bootstraps concrete adapters and repositories (e.g. `MediumApiAdapter`, `InMemoryPublicationRepository`, `MediumSessionRepository`).
2. It creates domain services: `PostDiscoveryService`, `PublicationConfigService`.
3. `ScrapePostsUseCase` orchestrates the flow: resolve config, initialize session (unless skipped), handle custom IDs or auto-discovery, collect posts.
4. The use case returns `ScrapePostsResponse` containing `Post` entities. The CLI formats and optionally saves the response.

## Tests

Run all tests:

```bash
python -m pytest tests/ -v
```

Run only unit or integration tests:

```bash
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
```

Coverage reports are configured in `pyproject.toml` and generate `htmlcov/`.

## Test coverage

![coverage](https://img.shields.io/badge/coverage-84%25-yellow)

Latest test run (local): TOTAL coverage 84%.

- HTML report: `htmlcov/index.html` (generated by pytest-cov)
- XML report: `coverage.xml`

Notes:

- Coverage is computed with pytest-cov. The HTML report lives in `htmlcov/` after running `pytest --cov=src --cov-report=html`.
- Some adapter branches remain partially covered; see `src/infrastructure/adapters/medium_api_adapter.py` for areas to target next.
- If you want a live badge that updates automatically, add CI with coverage upload (Codecov or Coveralls). See the `Add CI with GitHub Actions` task in the project TODO.

## Troubleshooting & important notes

- Missing `medium_sources.yaml`: `SourceConfigManager` raises `FileNotFoundError` when calling `--source` or `--list-sources`.
- Custom IDs validation: `PostId` requires exactly 12 alphanumeric characters; invalid IDs will raise a validation error.
- Empty result / errors: the use case catches exceptions and returns an empty response; the CLI prints helpful troubleshooting tips. Use logging or run in a development environment to debug further.
- Output directory: default `outputs/` (CLI will create it if missing).

## Files to inspect when extending or debugging

- `main.py` — entry point
- `src/presentation/cli.py` — CLI orchestration, formatting and progress UI
- `src/application/use_cases/scrape_posts.py` — main use case
- `src/domain/entities/publication.py` — domain entities
- `src/infrastructure/config/source_manager.py` — YAML loader
- `src/infrastructure/adapters/medium_api_adapter.py` — adapter for external API logic

## Contributing

The repository already includes `CONTRIBUTING.md` with development workflow, testing and coding standards. Please follow it when contributing.

## License

MIT — see `LICENSE` for details.

---

Next steps I can take (pick one):
- run the test suite and report results
- run a sample scrape and include a short JSON example
- add a `--verbose` flag to the CLI to improve debug output
