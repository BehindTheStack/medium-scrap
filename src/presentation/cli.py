"""
CLI Presentation Layer - Main Entry Point
Clean routing to modular commands
"""

import click

# Import commands
from .commands.reprocess_ml_command import reprocess_ml_command
from .commands.full_command import full_command
from .commands.etl_command import etl_command


@click.group()
def cli():
    """🚀 Medium Scraper CLI - Engineering Blog Content Pipeline"""
    pass


# Register commands
cli.add_command(full_command)
cli.add_command(etl_command)
cli.add_command(reprocess_ml_command)


if __name__ == "__main__":
    cli()
