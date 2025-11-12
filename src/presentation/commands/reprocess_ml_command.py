"""Reprocess ML Command for existing posts.

This module provides a CLI command to reprocess existing posts from the
database with the new ML discovery approach (NO hardcoded keywords):

• Clustering → Layers
• NER → Tech Stack  
• NER+ngrams → Patterns (dynamic)
• Embeddings → Solutions (semantic)
• Q&A → Problem
• Q&A → Approach

Author: BehindTheStack Team
License: MIT
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

from src.infrastructure.pipeline_db import PipelineDB
from src.presentation.helpers.ml_processor import MLProcessor


# Constants
MIN_CONTENT_LENGTH = 100
SAVE_PROGRESS_INTERVAL = 10


@click.command('reprocess-ml')
@click.option(
    '--source',
    '-s',
    help='Reprocess specific source only'
)
@click.option(
    '--all',
    'all_sources',
    is_flag=True,
    help='Reprocess ALL sources'
)
@click.option(
    '--limit',
    '-l',
    type=int,
    help='Limit posts per source (for testing)'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force reprocess even if already has ML data'
)
def reprocess_ml_command(
    source: Optional[str],
    all_sources: bool,
    limit: Optional[int],
    force: bool
) -> None:
    """Reprocess posts with NEW ML discovery approach.
    
    Reprocesses existing posts FROM DATABASE that have content_markdown 
    with the new ML approach (NO hardcoded keywords).
    
    Args:
        source: Specific source name to reprocess
        all_sources: Whether to process all available sources
        limit: Maximum number of posts per source to process
        force: Force reprocess even if already has ML data
        
    Returns:
        None
        
    Examples:
        # Reprocess one source
        uv run python main.py reprocess-ml --source netflix
        
        # Test with limit
        uv run python main.py reprocess-ml --source netflix --limit 50
        
        # Reprocess ALL sources
        uv run python main.py reprocess-ml --all
        
        # Force reprocess everything (even if already has ML)
        uv run python main.py reprocess-ml --all --force
    """
    console = Console()
    db = PipelineDB()
    
    _print_header(console)
    
    # Get available sources from database
    available_sources = _get_available_sources(db, console)
    if not available_sources:
        return
    
    # Determine which sources to process
    sources_to_process = _determine_sources_to_process(
        console,
        all_sources,
        source,
        available_sources
    )
    if not sources_to_process:
        return
    
    console.print()
    
    # Initialize ML processor
    ml_processor = MLProcessor(console)
    ml_processor.load_models()
    
    # Process each source
    total_reprocessed, total_failed = _process_all_sources(
        console,
        db,
        ml_processor,
        sources_to_process,
        force,
        limit
    )
    
    # Final summary
    _print_final_summary(
        console,
        sources_to_process,
        total_reprocessed,
        total_failed
    )


def _print_header(console: Console) -> None:
    """Print command header with description.
    
    Args:
        console: Rich console for output
        
    Returns:
        None
    """
    console.print()
    console.print("[bold cyan]🔄 ML Reprocessing - New Approach[/bold cyan]")
    console.print(
        "[dim]Using: Clustering + NER + Q&A "
        "(NO hardcoded keywords!)[/dim]"
    )
    console.print("[dim]Source: Database posts with content_markdown[/dim]")
    console.print()


def _get_available_sources(
    db: PipelineDB,
    console: Console
) -> List[str]:
    """Get list of available sources from database.
    
    Args:
        db: Database instance
        console: Rich console for output
        
    Returns:
        List of source names that have content_markdown, empty if none
    """
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT source FROM posts "
            "WHERE content_markdown IS NOT NULL ORDER BY source"
        )
        available_sources = [row['source'] for row in cursor.fetchall()]
    
    if not available_sources:
        console.print(
            "[red]❌ No posts with content_markdown found in database[/red]"
        )
        return []
    
    console.print(
        f"[dim]Available sources in DB: {', '.join(available_sources)}[/dim]"
    )
    console.print()
    
    return available_sources


def _determine_sources_to_process(
    console: Console,
    all_sources: bool,
    source: Optional[str],
    available_sources: List[str]
) -> Optional[List[str]]:
    """Determine which sources to process based on flags.
    
    Args:
        console: Rich console for output
        all_sources: Whether to process all sources
        source: Specific source name if provided
        available_sources: List of available source names
        
    Returns:
        List of source names to process, or None if invalid input
    """
    if all_sources:
        console.print(
            f"[cyan]Processing ALL {len(available_sources)} "
            f"sources from database[/cyan]"
        )
        return available_sources
    
    if source:
        if source not in available_sources:
            console.print(
                f"[red]❌ Source '{source}' not found in database[/red]"
            )
            console.print(
                f"[yellow]Available: {', '.join(available_sources)}[/yellow]"
            )
            return None
        return [source]
    
    # No source specified
    console.print("[red]❌ Must specify --source or --all[/red]")
    console.print("[yellow]Examples:[/yellow]")
    console.print("  uv run python main.py reprocess-ml --source netflix")
    console.print("  uv run python main.py reprocess-ml --all")
    return None


def _process_all_sources(
    console: Console,
    db: PipelineDB,
    ml_processor: MLProcessor,
    sources: List[str],
    force: bool,
    limit: Optional[int]
) -> tuple[int, int]:
    """Process all specified sources with ML.
    
    Args:
        console: Rich console for output
        db: Database instance
        ml_processor: ML processor instance
        sources: List of source names to process
        force: Whether to force reprocessing
        limit: Optional limit on posts per source
        
    Returns:
        Tuple of (total_reprocessed, total_failed)
    """
    total_reprocessed = 0
    total_failed = 0
    
    for idx, src in enumerate(sources, 1):
        _print_source_header(console, idx, len(sources), src)
        
        # Get posts to process
        posts = _get_posts_to_process(db, src, force, limit)
        
        if not posts:
            console.print(
                f"[green]✅ No posts need ML reprocessing for {src}[/green]"
            )
            console.print()
            continue
        
        console.print(
            f"[yellow]Found {len(posts)} posts to reprocess[/yellow]"
        )
        console.print()
        
        # Prepare and process entries
        entries = _prepare_entries_for_ml(posts)
        
        if not entries:
            console.print(
                f"[yellow]No valid content to process for {src}[/yellow]"
            )
            console.print()
            continue
        
        # Process with ML
        reprocessed, failed = _process_source_with_ml(
            console,
            db,
            ml_processor,
            src,
            entries
        )
        
        total_reprocessed += reprocessed
        total_failed += failed
        
        console.print()
    
    return total_reprocessed, total_failed


def _print_source_header(
    console: Console,
    current: int,
    total: int,
    source: str
) -> None:
    """Print header for source being processed.
    
    Args:
        console: Rich console for output
        current: Current source index (1-based)
        total: Total number of sources
        source: Source name
        
    Returns:
        None
    """
    console.print(f"[magenta]{'='*60}[/magenta]")
    console.print(f"[magenta][{current}/{total}] {source}[/magenta]")
    console.print(f"[magenta]{'='*60}[/magenta]")
    console.print()


def _get_posts_to_process(
    db: PipelineDB,
    source: str,
    force: bool,
    limit: Optional[int]
) -> List[Dict[str, Any]]:
    """Get posts that need ML processing from database.
    
    Args:
        db: Database instance
        source: Source name to query
        force: Whether to reprocess all posts
        limit: Optional limit on number of posts
        
    Returns:
        List of post dictionaries from database
    """
    with db._get_connection() as conn:
        cursor = conn.cursor()
        
        if force:
            # Reprocess all posts with content
            query = """
                SELECT * FROM posts 
                WHERE source = ? AND content_markdown IS NOT NULL
                ORDER BY published_at DESC
            """
            params: List[Any] = [source]
        else:
            # Only posts without ML data
            query = """
                SELECT * FROM posts 
                WHERE source = ? 
                AND content_markdown IS NOT NULL
                AND (tech_stack IS NULL OR patterns IS NULL 
                     OR ml_classified = 0)
                ORDER BY published_at DESC
            """
            params = [source]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def _prepare_entries_for_ml(
    posts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Prepare post entries for ML processing.
    
    Filters posts by minimum content length and converts database
    posts to ML processor entry format.
    
    Args:
        posts: List of post dictionaries from database
        
    Returns:
        List of prepared entries for ML processor
    """
    entries_for_ml = []
    
    for post in posts:
        md = post.get('content_markdown', '')
        if not md or len(md) < MIN_CONTENT_LENGTH:
            continue
        
        date = _parse_published_date(post.get('published_at'))
        
        entries_for_ml.append({
            'id': post['id'],
            'title': post.get('title', 'Untitled'),
            'date': date.isoformat() if date else None,
            'content_markdown': md,
            'content': md,  # Backward compatibility
            'path': post.get('url', ''),
        })
    
    return entries_for_ml


def _parse_published_date(published_at: Any) -> Optional[datetime]:
    """Parse published_at field to datetime.
    
    Args:
        published_at: Published date string or None
        
    Returns:
        Parsed datetime.date object or None if parsing fails
    """
    if not published_at:
        return None
    
    try:
        date_str = str(published_at).replace('Z', '+00:00')
        return datetime.fromisoformat(date_str).date()
    except (ValueError, AttributeError):
        return None


def _process_source_with_ml(
    console: Console,
    db: PipelineDB,
    ml_processor: MLProcessor,
    source: str,
    entries: List[Dict[str, Any]]
) -> tuple[int, int]:
    """Process single source with ML extraction.
    
    Args:
        console: Rich console for output
        db: Database instance
        ml_processor: ML processor instance
        source: Source name being processed
        entries: List of entries to process
        
    Returns:
        Tuple of (reprocessed_count, failed_count)
    """
    console.print(
        f"[cyan]🤖 Running ML discovery on "
        f"{len(entries)} posts...[/cyan]"
    )
    console.print()
    
    try:
        # Create incremental save callback
        save_callback = _create_save_callback(console, db, len(entries))
        
        # Run ML processing with incremental save
        ml_processor.process_posts(
            entries,
            save_callback=save_callback
        )
        
        saved_count = save_callback.keywords['count']
        console.print(
            f"[green]✅ Processed and saved "
            f"{saved_count} posts[/green]"
        )
        return saved_count, 0
        
    except Exception as e:
        console.print(
            f"[red]❌ ML processing failed for {source}: {e}[/red]"
        )
        import traceback
        traceback.print_exc()
        return 0, 1


def _create_save_callback(
    console: Console,
    db: PipelineDB,
    total_entries: int
) -> Callable[[str, Dict[str, Any]], None]:
    """Create callback function for incremental post saving.
    
    Args:
        console: Rich console for progress output
        db: Database instance for saving
        total_entries: Total number of entries being processed
        
    Returns:
        Callback function that saves posts incrementally
    """
    saved_count = {'count': 0}  # Mutable counter
    
    def save_post_callback(post_id: str, ml_data: Dict[str, Any]) -> None:
        """Save each post immediately after ML processing.
        
        Args:
            post_id: Post ID to update
            ml_data: ML discovery data to save
            
        Returns:
            None
        """
        db.update_ml_discovery(post_id, ml_data)
        saved_count['count'] += 1
        
        if saved_count['count'] % SAVE_PROGRESS_INTERVAL == 0:
            console.print(
                f"[dim]💾 Saved {saved_count['count']}/"
                f"{total_entries} posts...[/dim]",
                end='\r'
            )
    
    # Attach counter to function for access in outer scope
    save_post_callback.keywords = saved_count
    
    return save_post_callback


def _print_final_summary(
    console: Console,
    sources: List[str],
    total_reprocessed: int,
    total_failed: int
) -> None:
    """Print final summary table.
    
    Args:
        console: Rich console for output
        sources: List of source names processed
        total_reprocessed: Total number of posts reprocessed
        total_failed: Total number of failed sources
        
    Returns:
        None
    """
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print("[bold green]🎉 Reprocessing Complete![/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print()
    
    table = Table(show_header=True, border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow", justify="right")
    
    table.add_row("Sources Processed", str(len(sources) - total_failed))
    table.add_row("Posts Reprocessed", str(total_reprocessed))
    
    if total_failed > 0:
        table.add_row("Failed Sources", str(total_failed))
    
    console.print(table)
    console.print()
    
    console.print("[cyan]✨ All posts now have ML-discovered data![/cyan]")
    console.print()
