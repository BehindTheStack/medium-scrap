"""
Collect Command - Post Collection and Updates

This command focuses solely on collecting and updating posts from Medium sources.
No ML processing, no exports - just data collection.

Author: BehindTheStack Team
License: MIT
"""

import time
from typing import Dict, List, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from src.infrastructure.config.source_manager import SourceConfigManager
from src.infrastructure.pipeline_db import PipelineDB
from src.infrastructure.content_extractor import (
    html_to_markdown,
    extract_code_blocks,
    classify_technical
)
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.http_transport import HttpxTransport


@click.command('collect')
@click.option(
    '--source',
    '-s',
    help='Collect from specific source only (e.g., netflix, airbnb)'
)
@click.option(
    '--all',
    'all_sources',
    is_flag=True,
    help='Collect from ALL sources in medium_sources.yaml'
)
@click.option(
    '--limit',
    '-l',
    type=int,
    help='Limit number of posts per source (for testing)'
)
@click.option(
    '--update',
    is_flag=True,
    help='Update existing posts (re-fetch metadata like claps, tags)'
)
def collect_command(
    source: Optional[str],
    all_sources: bool,
    limit: Optional[int],
    update: bool
) -> None:
    """
    Collect posts from Medium sources.
    
    This command:
    - Fetches posts from Medium API
    - Extracts content (HTML → Markdown)
    - Classifies as technical/non-technical
    - Stores in database
    
    Does NOT:
    - Run ML processing
    - Export data
    - Generate reports
    
    Examples:
        # Collect from one source
        uv run python main.py collect --source netflix
        
        # Collect from all sources
        uv run python main.py collect --all
        
        # Update existing posts
        uv run python main.py collect --source netflix --update
        
        # Test with limit
        uv run python main.py collect --source airbnb --limit 10
    """
    console = Console()
    
    _print_header(console)
    
    # Validate arguments
    if not source and not all_sources:
        console.print("[red]❌ Error: Specify --source <name> or --all[/red]")
        console.print("[dim]Example: uv run python main.py collect --source netflix[/dim]")
        return
    
    if source and all_sources:
        console.print("[red]❌ Error: Cannot use both --source and --all[/red]")
        return
    
    # Load source configuration
    try:
        source_manager = SourceConfigManager()
    except Exception as e:
        console.print(f"[red]❌ Failed to load source configuration: {e}[/red]")
        return
    
    # Determine which sources to process
    if all_sources:
        sources_to_collect = list(source_manager.get_all_sources().keys())
        console.print(f"[cyan]📋 Collecting from all {len(sources_to_collect)} sources[/cyan]")
    else:
        if not source_manager.has_source(source):
            available = ', '.join(source_manager.get_all_sources().keys())
            console.print(f"[red]❌ Source '{source}' not found in configuration[/red]")
            console.print(f"[dim]Available sources: {available}[/dim]")
            return
        sources_to_collect = [source]
        console.print(f"[cyan]📋 Collecting from: {source}[/cyan]")
    
    if limit:
        console.print(f"[dim]⚠ Limiting to {limit} posts per source[/dim]")
    
    console.print()
    
    # Initialize services
    db = PipelineDB()
    transport = HttpxTransport()
    api = MediumApiAdapter(transport)
    
    # Process each source
    total_stats = {
        'sources_processed': 0,
        'posts_collected': 0,
        'posts_updated': 0,
        'posts_failed': 0
    }
    
    start_time = time.time()
    
    for idx, source_key in enumerate(sources_to_collect, 1):
        source_config = source_manager.get_source(source_key)
        
        console.print(f"[bold magenta]{'=' * 70}[/bold magenta]")
        console.print(f"[bold magenta][{idx}/{len(sources_to_collect)}] {source_key.upper()}[/bold magenta]")
        console.print(f"[bold magenta]{'=' * 70}[/bold magenta]")
        console.print()
        
        stats = _collect_posts(
            console=console,
            db=db,
            api=api,
            source_key=source_key,
            source_config=source_config,
            limit=limit,
            update=update
        )
        
        total_stats['sources_processed'] += 1
        total_stats['posts_collected'] += stats['collected']
        total_stats['posts_updated'] += stats['updated']
        total_stats['posts_failed'] += stats['failed']
        
        console.print()
    
    # Final summary
    total_time = time.time() - start_time
    _print_summary(console, total_stats, total_time)


def _print_header(console: Console) -> None:
    """Print command header"""
    console.print()
    console.print("╔" + "═" * 70 + "╗", style="bold cyan")
    console.print("║" + " " * 70 + "║", style="bold cyan")
    console.print(
        "║" + "  📦 COLLECT - Post Collection & Updates".center(70) + "║",
        style="bold cyan"
    )
    console.print(
        "║" + "  Fetch posts from Medium sources".center(70) + "║",
        style="dim cyan"
    )
    console.print("║" + " " * 70 + "║", style="bold cyan")
    console.print("╚" + "═" * 70 + "╝", style="bold cyan")
    console.print()


def _collect_posts(
    console: Console,
    db: PipelineDB,
    api: MediumApiAdapter,
    source_key: str,
    source_config,
    limit: Optional[int],
    update: bool
) -> Dict[str, int]:
    """
    Collect posts from a single source.
    
    Returns:
        Dictionary with collection statistics
    """
    stats = {
        'collected': 0,
        'updated': 0,
        'failed': 0
    }
    
    publication_name = source_config.get_publication_name()
    
    console.print(f"[cyan]🔍 Fetching posts from publication: {publication_name}[/cyan]")
    
    # Fetch posts from API
    try:
        posts = api.fetch_publication_posts(publication_name, limit=limit)
        console.print(f"[green]✓ Found {len(posts)} posts[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to fetch posts: {e}[/red]")
        return stats
    
    if not posts:
        console.print("[yellow]⚠ No posts found[/yellow]")
        return stats
    
    console.print()
    console.print(f"[cyan]📝 Processing {len(posts)} posts...[/cyan]")
    
    # Process posts with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing", total=len(posts))
        
        for post in posts:
            try:
                # Check if post exists
                existing = db.get_post(post['id'])
                
                if existing and not update:
                    progress.update(task, advance=1)
                    continue
                
                # Extract content
                if post.get('content', {}).get('bodyModel'):
                    html_content = api.render_post_html(post['content']['bodyModel'])
                    markdown_content, _, code_blocks = html_to_markdown(html_content)
                    is_technical = classify_technical(
                        html_content,
                        code_blocks,
                        tags=post.get('tags', [])
                    )
                else:
                    markdown_content = None
                    is_technical = False
                
                # Prepare post data
                post_data = {
                    'id': post['id'],
                    'title': post.get('title', ''),
                    'subtitle': post.get('content', {}).get('subtitle', ''),
                    'url': f"https://medium.com/p/{post['id']}",
                    'published_at': post.get('firstPublishedAt'),
                    'updated_at': post.get('latestPublishedAt'),
                    'author': post.get('creator', {}).get('name', ''),
                    'author_id': post.get('creator', {}).get('id', ''),
                    'tags': post.get('tags', []),
                    'claps': post.get('clapCount', 0),
                    'reading_time': post.get('readingTime', 0),
                    'source': source_key,
                    'content_markdown': markdown_content,
                    'is_technical': is_technical
                }
                
                # Save or update
                if existing:
                    db.update_post(post['id'], post_data)
                    stats['updated'] += 1
                else:
                    db.save_post(post_data)
                    stats['collected'] += 1
                
            except Exception as e:
                console.print(f"[red]Error processing post {post.get('id')}: {e}[/red]")
                stats['failed'] += 1
            
            progress.update(task, advance=1)
    
    # Print source summary
    console.print()
    console.print(f"[green]✅ {source_key} completed:[/green]")
    console.print(f"  • New posts: {stats['collected']}")
    console.print(f"  • Updated posts: {stats['updated']}")
    if stats['failed'] > 0:
        console.print(f"  • Failed: {stats['failed']}")
    
    return stats


def _print_summary(console: Console, stats: Dict[str, int], total_time: float) -> None:
    """Print final summary table"""
    console.print()
    console.print("[bold green]" + "=" * 70 + "[/bold green]")
    console.print("[bold green]📊 COLLECTION SUMMARY[/bold green]")
    console.print("[bold green]" + "=" * 70 + "[/bold green]")
    console.print()
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Sources Processed", str(stats['sources_processed']))
    table.add_row("Posts Collected (New)", str(stats['posts_collected']))
    table.add_row("Posts Updated", str(stats['posts_updated']))
    
    if stats['posts_failed'] > 0:
        table.add_row("Failed", str(stats['posts_failed']), style="red")
    
    total_posts = stats['posts_collected'] + stats['posts_updated']
    table.add_row("", "")
    table.add_row("Total Posts Processed", str(total_posts), style="bold green")
    
    console.print(table)
    console.print()
    console.print(f"[dim]⏱️  Total time: {total_time:.1f}s[/dim]")
    console.print(f"[dim]⚡ Average: {total_time/max(total_posts, 1):.2f}s per post[/dim]")
    console.print()


# Export for CLI registration
__all__ = ['collect_command']
