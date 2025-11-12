"""
CLI Presentation Layer - Main Entry Point
Clean routing to modular commands
"""

import click
from rich.console import Console

# Import commands
from .commands.reprocess_ml_command import reprocess_ml_command
from .commands.full_command import full_command


@click.group()
def cli():
    """🚀 Medium Scraper CLI - Engineering Blog Content Pipeline"""
    pass


# Register commands
cli.add_command(reprocess_ml_command)
cli.add_command(full_command)


# Placeholder ETL command (to be refactored)
@cli.command('etl')
@click.option('--source', '-s', required=True, help='Source name')
@click.option('--limit', '-l', type=int, help='Limit posts')
def etl_command(source, limit):
    """
    📊 ETL with ML classification
    
    [TO BE REFACTORED - usar full command por enquanto]
    """
    console = Console()
    console.print("[yellow]⚠️  ETL command needs refactoring[/yellow]")
    console.print("[cyan]Use: uv run python main.py full --source", source, "[/cyan]")


if __name__ == "__main__":
    cli()
