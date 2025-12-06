"""Reprocess ML Command for existing posts.

This module provides a CLI command to reprocess existing posts from the
database with ML discovery approaches:

LEGACY MODE (default):
• Clustering → Layers
• NER (BERT) → Tech Stack  
• NER+ngrams → Patterns (dynamic)
• Embeddings → Solutions (semantic)
• Q&A → Problem
• Q&A → Approach

MODERN MODE (--modern):
• GLiNER → Zero-shot tech extraction
• Semantic Embeddings → Architecture patterns
• Optional LLM → Deep extraction (--use-llm)

Author: BehindTheStack Team
License: MIT
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

from src.infrastructure.pipeline_db import PipelineDB
from src.presentation.helpers.ml_processor import MLProcessor, ModernMLProcessor


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
@click.option(
    '--batch-size',
    '-b',
    type=int,
    default=50,
    help='Batch size for optimized processing (default: 50)'
)
@click.option(
    '--technical-only',
    is_flag=True,
    help='Process only technical posts (is_technical=True)'
)
@click.option(
    '--legacy',
    is_flag=True,
    help='Use legacy processing (all at once, slower)'
)
@click.option(
    '--modern',
    is_flag=True,
    help='Use modern hybrid extraction (GLiNER + Semantic Patterns)'
)
@click.option(
    '--use-llm',
    is_flag=True,
    help='Enable LLM extraction with --modern (requires Ollama)'
)
def reprocess_ml_command(
    source: Optional[str],
    all_sources: bool,
    limit: Optional[int],
    force: bool,
    batch_size: int,
    technical_only: bool,
    legacy: bool,
    modern: bool,
    use_llm: bool
) -> None:
    """Reprocess posts with ML discovery approach.
    
    Two modes available:
    
    DEFAULT: Legacy approach (BERT NER + n-grams + QA)
    
    --modern: Modern hybrid approach (2024):
    - GLiNER for zero-shot tech extraction
    - Semantic embeddings for pattern detection
    - Optional LLM for deep extraction
    
    Args:
        source: Specific source name to reprocess
        all_sources: Whether to process all available sources
        limit: Maximum number of posts per source to process
        force: Force reprocess even if already has ML data
        batch_size: Posts per batch for extraction (default: 50)
        legacy: Use old processing method (all at once)
        modern: Use modern GLiNER + Semantic approach
        use_llm: Enable LLM extraction (with --modern)
        
    Returns:
        None
        
    Examples:
        # Reprocess one source (legacy)
        uv run python main.py reprocess-ml --source netflix
        
        # Modern extraction (recommended)
        uv run python main.py reprocess-ml --source netflix --modern
        
        # Modern with LLM deep extraction
        uv run python main.py reprocess-ml --source netflix --modern --use-llm
        
        # Force reprocess everything with modern
        uv run python main.py reprocess-ml --all --force --modern
    """
    console = Console()
    db = PipelineDB()
    
    _print_header(console, optimized=not legacy, modern=modern)
    
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
    
    # Initialize appropriate ML processor
    if modern:
        ml_processor = ModernMLProcessor(console)
        ml_processor.load_models(use_llm=use_llm)
    else:
        ml_processor = MLProcessor(console)
        ml_processor.load_models()
    
    # Process each source
    total_reprocessed, total_failed = _process_all_sources(
        console,
        db,
        ml_processor,
        sources_to_process,
        force,
        limit,
        technical_only,
        use_optimized=not legacy,
        batch_size=batch_size,
        modern=modern,
        use_llm=use_llm
    )
    
    # Final summary
    _print_final_summary(
        console,
        sources_to_process,
        total_reprocessed,
        total_failed
    )


def _print_header(console: Console, optimized: bool = True, modern: bool = False) -> None:
    """Print beautiful command header"""
    console.print()
    console.print("╔" + "═" * 78 + "╗", style="bold cyan")
    console.print("║" + " " * 78 + "║", style="bold cyan")
    
    if modern:
        title = "🔬 REPROCESS ML - Modern Hybrid Extraction (2024)"
        subtitle = "  GLiNER + Semantic Patterns + Optional LLM"
    elif optimized:
        title = "🚀 REPROCESS ML - Optimized Discovery (Hybrid Approach)"
        subtitle = "  Global Clustering + Batch Extraction"
    else:
        title = "🔬 REPROCESS ML - Legacy Discovery (Full Processing)"
        subtitle = "  All posts processed together"
    
    console.print(
        "║" + f"  {title}".center(78) + "║",
        style="bold cyan"
    )
    console.print(
        "║" + subtitle.center(78) + "║",
        style="dim cyan"
    )
    console.print("║" + " " * 78 + "║", style="bold cyan")
    console.print("╚" + "═" * 78 + "╝", style="bold cyan")
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
    ml_processor,
    sources: List[str],
    force: bool,
    limit: Optional[int],
    technical_only: bool,
    use_optimized: bool = True,
    batch_size: int = 50,
    modern: bool = False,
    use_llm: bool = False
) -> tuple[int, int]:
    """Process all specified sources with ML.
    
    Args:
        console: Rich console for output
        db: Database instance
        ml_processor: ML processor instance (MLProcessor or ModernMLProcessor)
        sources: List of source names to process
        force: Whether to force reprocessing
        limit: Maximum posts per source
        technical_only: Process only technical posts
        use_optimized: Whether to use optimized batch processing
        batch_size: Posts per batch (for optimized mode)
        modern: Use modern extraction pipeline
        use_llm: Enable LLM extraction (with modern)
        
    Returns:
        Tuple of (total_reprocessed, total_failed)
    """
    total_reprocessed = 0
    total_failed = 0
    
    for idx, src in enumerate(sources, 1):
        _print_source_header(console, idx, len(sources), src)
        
        # Get posts to process
        posts = _get_posts_to_process(db, src, force, limit, technical_only, modern)
        
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
            entries,
            use_optimized,
            batch_size,
            modern=modern,
            use_llm=use_llm
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
    limit: Optional[int],
    technical_only: bool = False,
    modern: bool = False
) -> List[Dict[str, Any]]:
    """Get posts that need ML processing from database.
    
    Args:
        db: Database instance
        source: Source name to query
        force: Whether to reprocess all posts
        limit: Optional limit on number of posts
        technical_only: Filter only technical posts
        modern: Use modern model version
        
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
            """
            params: List[Any] = [source]
            
            # Filter technical posts if requested
            if technical_only:
                query += " AND is_technical = 1"
            
            query += " ORDER BY published_at DESC"
        else:
            # Only posts without ML data for this model version
            model_version = 'modern-v1' if modern else 'legacy-v1'
            query = """
                SELECT p.* FROM posts p
                LEFT JOIN ml_discoveries md ON p.id = md.post_id AND md.model_version = ?
                WHERE p.source = ? 
                AND p.content_markdown IS NOT NULL
                AND md.id IS NULL
            """
            params = [model_version, source]
            
            # Filter technical posts if requested
            if technical_only:
                query += " AND is_technical = 1"
            
            query += " ORDER BY published_at DESC"
        
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
    ml_processor,
    source: str,
    entries: List[Dict[str, Any]],
    use_optimized: bool = True,
    batch_size: int = 50,
    modern: bool = False,
    use_llm: bool = False
) -> tuple[int, int]:
    """Process single source with ML extraction.
    
    Args:
        console: Rich console for output
        db: Database instance
        ml_processor: ML processor instance
        source: Source name being processed
        entries: List of entries to process
        use_optimized: Whether to use optimized batch processing
        batch_size: Posts per batch (for optimized mode)
        modern: Use modern extraction pipeline
        use_llm: Enable LLM extraction (with modern)
        
    Returns:
        Tuple of (reprocessed_count, failed_count)
    """
    if modern:
        mode = "Modern (GLiNER + Patterns" + (" + LLM)" if use_llm else ")")
    elif use_optimized:
        mode = "Optimized (Hybrid)"
    else:
        mode = "Legacy (Full)"
        
    console.print(
        f"[cyan]🤖 Running ML discovery on "
        f"{len(entries)} posts ({mode})...[/cyan]"
    )
    console.print()
    
    try:
        # Determine model version and pipeline type
        if modern:
            model_version = 'modern-llm-v1' if use_llm else 'modern-v1'
            pipeline_type = 'modern-llm' if use_llm else 'modern'
        else:
            model_version = 'legacy-v1'
            pipeline_type = 'legacy'
        
        # Create incremental save callback
        save_callback = _create_save_callback(console, db, len(entries), model_version, pipeline_type)
        
        # Run ML processing with appropriate method
        if modern:
            # Use modern hybrid processor
            ml_processor.process_posts(
                entries,
                save_callback=save_callback,
                use_llm=use_llm
            )
        elif use_optimized:
            ml_processor.process_posts_optimized(
                entries,
                save_callback=save_callback,
                batch_size=batch_size
            )
        else:
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
    total_entries: int,
    model_version: str,
    pipeline_type: str
) -> Callable[[str, Dict[str, Any]], None]:
    """Create callback function for incremental post saving.
    
    Args:
        console: Rich console for progress output
        db: Database instance for saving
        total_entries: Total number of entries being processed
        model_version: ML model version (e.g., 'modern-v1', 'legacy-v1')
        pipeline_type: Pipeline type ('modern', 'modern-llm', 'legacy')
        
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
        db.update_ml_discovery(post_id, ml_data, model_version, pipeline_type)
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
