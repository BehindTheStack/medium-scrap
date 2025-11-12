"""
CLI Presentation Layer - Main Entry Point
Clean routing to modular commands
"""

import click
from rich.console import Console

# Import commands
from .commands.reprocess_ml_command import reprocess_ml_command


@click.group()
def cli():
    """🚀 Medium Scraper CLI - Engineering Blog Content Pipeline"""
    pass


# Register commands
cli.add_command(reprocess_ml_command)


# Placeholder commands (to be refactored from old cli.py)
@cli.command('full')
@click.option('--bulk', '-b', help='Bulk collection name (e.g., tech_giants)')
@click.option('--source', '-s', help='Single source name (e.g., netflix)')
@click.option('--limit', '-l', type=int, help='Limit posts per source')
def full_command(bulk, source, limit):
    """
    🎯 Complete pipeline: scrape → enrich → classify
    
    [TO BE REFACTORED]
    """
    console = Console()
    console.print("[yellow]⚠️  Full command needs refactoring - use reprocess-ml for ML features[/yellow]")
    console.print()
    console.print("[cyan]Available:[/cyan]")
    console.print("  • uv run python main.py reprocess-ml --all")


@cli.command('etl')
@click.option('--source', '-s', required=True, help='Source name')
@click.option('--limit', '-l', type=int, help='Limit posts')
def etl_command(source, limit):
    """
    📊 ETL with ML classification
    
    [TO BE REFACTORED]
    """
    console = Console()
    console.print("[yellow]⚠️  ETL command needs refactoring - use reprocess-ml for ML features[/yellow]")


if __name__ == "__main__":
    cli()
