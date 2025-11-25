"""Full Command - Complete pipeline execution.

This module provides the complete end-to-end pipeline command that:
1. Scrapes posts from Medium API
2. Saves posts to database
3. Enriches content (HTML → Markdown)
4. Performs ML discovery
5. Generates timeline files

The pipeline orchestrates all steps in sequence, handling errors gracefully
and providing comprehensive progress reporting.

Author: BehindTheStack Team
License: MIT
"""

import subprocess
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table

from src.application.use_cases.scrape_posts import (
    ScrapePostsRequest,
    ScrapePostsUseCase
)
from src.domain.services.publication_service import (
    PostDiscoveryService,
    PublicationConfigService
)
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.config.source_manager import SourceConfigManager
from src.infrastructure.external.repositories import (
    InMemoryPublicationRepository,
    MediumSessionRepository
)
from src.infrastructure.pipeline_db import PipelineDB


# Constants
COLLECTION_MODE = 'metadata'
SEPARATOR_WIDTH = 50


@click.command('full')
@click.option(
    '--source',
    '-s',
    help='Process specific source (default: ALL from YAML)'
)
@click.option(
    '--limit',
    '-l',
    type=int,
    help='Limit posts per source'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force reprocess ML even if already classified'
)
@click.option(
    '--technical-only',
    is_flag=True,
    help='Process and export only technical posts'
)
def full_command(
    source: Optional[str],
    limit: Optional[int],
    force: bool,
    technical_only: bool
) -> None:
    """FULL Pipeline: Collect → Enrich → ML → Timeline.
    
    Complete workflow:
    1. Scrape posts from Medium API
    2. Save to database
    3. Enrich content (HTML → Markdown)
    4. ML discovery (tech stack, patterns, etc)
    5. Generate timeline
    
    Args:
        source: Specific source to process (None = all from YAML)
        limit: Optional limit on posts per source
        force: Force reprocess ML even if already classified
        technical_only: Process and export only technical posts
        
    Returns:
        None
        
    Examples:
        # Process ALL sources from YAML
        uv run python main.py full
        
        # Process one source
        uv run python main.py full --source netflix
        
        # Limit posts
        uv run python main.py full --source netflix --limit 50
        
        # Force reprocess only technical posts
        uv run python main.py full --force --technical-only
    """
    console = Console()
    db = PipelineDB()
    config_manager = SourceConfigManager()
    
    # Determine sources to process
    sources = _get_sources_to_process(config_manager, source, console)
    if not sources:
        return
    
    _print_pipeline_header(console, len(sources))
    
    # Process each source
    stats = _process_all_sources(console, db, sources, limit, force, technical_only)
    
    # Print final summary
    _print_final_summary(console, sources, stats)


def _get_sources_to_process(
    config_manager: SourceConfigManager,
    source: Optional[str],
    console: Console
) -> Optional[List[str]]:
    """Get list of sources to process.
    
    Args:
        config_manager: Source configuration manager
        source: Specific source name or None for all
        console: Rich console for output
        
    Returns:
        List of source names or None if error
    """
    if source:
        return [source]
    
    try:
        sources_dict = config_manager.list_sources()
        sources = list(sources_dict.keys())
        
        if not sources:
            console.print("[red]❌ No sources found in YAML[/red]")
            return None
        
        return sources
        
    except Exception as e:
        console.print(f"[red]❌ Failed to load sources from YAML: {e}[/red]")
        return None


def _print_pipeline_header(console: Console, num_sources: int) -> None:
    """Print pipeline header.
    
    Args:
        console: Rich console for output
        num_sources: Number of sources to process
        
    Returns:
        None
    """
    console.print(
        f"\n[bold cyan]🚀 FULL Pipeline: {num_sources} source(s)[/bold cyan]\n"
    )


def _process_all_sources(
    console: Console,
    db: PipelineDB,
    sources: List[str],
    limit: Optional[int],
    force: bool,
    technical_only: bool
) -> Dict[str, Any]:
    """Process all sources through complete pipeline.
    
    Args:
        console: Rich console for output
        db: Database instance
        sources: List of source names to process
        limit: Optional limit on posts per source
        force: Force reprocess ML even if already classified
        technical_only: Process and export only technical posts
        
    Returns:
        Statistics dictionary with counts and failures
    """
    stats = {
        'collected': 0,
        'enriched': 0,
        'timelines': 0,
        'failed': []
    }
    
    for idx, src in enumerate(sources, 1):
        _print_source_header(console, idx, len(sources), src)
        
        try:
            result = _process_single_source(console, db, src, limit, force, technical_only)
            
            # Update statistics
            stats['collected'] += result['new_posts']
            if result['enriched']:
                stats['enriched'] += 1
                stats['timelines'] += 1
            
            if not result['success']:
                stats['failed'].append(src)
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]\n")
            import traceback
            traceback.print_exc()
            stats['failed'].append(src)
        
        console.print()
    
    return stats


def _print_source_header(
    console: Console,
    current: int,
    total: int,
    source: str
) -> None:
    """Print header for source being processed.
    
    Args:
        console: Rich console for output
        current: Current source index
        total: Total number of sources
        source: Source name
        
    Returns:
        None
    """
    console.print(f"[magenta]{'='*SEPARATOR_WIDTH}[/magenta]")
    console.print(f"[magenta][{current}/{total}] {source}[/magenta]")
    console.print(f"[magenta]{'='*SEPARATOR_WIDTH}[/magenta]\n")


def _process_single_source(
    console: Console,
    db: PipelineDB,
    source: str,
    limit: Optional[int],
    force: bool,
    technical_only: bool
) -> Dict[str, Any]:
    """Process single source through pipeline.
    
    Args:
        console: Rich console for output
        db: Database instance
        source: Source name
        limit: Optional limit on posts
        force: Force reprocess ML even if already classified
        technical_only: Process and export only technical posts
        
    Returns:
        Dictionary with processing results
    """
    # Phase 1: Collect posts
    collected_posts = _collect_posts(console, source, limit)
    
    # Phase 2: Save to database
    new_posts, updated_posts = _save_posts_to_db(
        console,
        db,
        source,
        collected_posts
    )
    
    # Check if we should continue
    if not _has_posts_to_process(console, db, source, new_posts):
        return {'success': False, 'new_posts': 0, 'enriched': False}
    
    # Phase 3: Enrich with ETL
    enriched = _run_etl_pipeline(console, source, limit, force, technical_only)
    
    return {
        'success': True,
        'new_posts': new_posts,
        'enriched': enriched
    }


def _collect_posts(
    console: Console,
    source: str,
    limit: Optional[int]
) -> List[Any]:
    """Collect posts from Medium API.
    
    Args:
        console: Rich console for output
        source: Source name
        limit: Optional limit on posts
        
    Returns:
        List of collected Post objects
    """
    with console.status("[blue]📥 Collecting...[/blue]", spinner="dots"):
        # Setup repositories and services
        post_repo = MediumApiAdapter()
        pub_repo = InMemoryPublicationRepository()
        sess_repo = MediumSessionRepository()
        
        svc = PostDiscoveryService(post_repo)
        cfg_svc = PublicationConfigService(pub_repo)
        use_case = ScrapePostsUseCase(svc, cfg_svc, sess_repo)
        
        # Execute scraping
        req = ScrapePostsRequest(
            publication_name=source,
            limit=limit,
            auto_discover=True,
            skip_session=True,
            mode=COLLECTION_MODE
        )
        resp = use_case.execute(req)
    
    if resp.success and resp.posts:
        return resp.posts
    
    return []


def _save_posts_to_db(
    console: Console,
    db: PipelineDB,
    source: str,
    posts: List[Any]
) -> Tuple[int, int]:
    """Save collected posts to database.
    
    Args:
        console: Rich console for output
        db: Database instance
        source: Source name
        posts: List of Post objects to save
        
    Returns:
        Tuple of (new_posts_count, updated_posts_count)
    """
    if not posts:
        return 0, 0
    
    console.print(
        f"[dim]💾 Saving {len(posts)} posts to database...[/dim]"
    )
    
    new_posts = 0
    updated_posts = 0
    
    for post in posts:
        try:
            existing = db.post_exists(post.id.value)
            
            # Skip if already enriched (preserve content)
            if _is_already_enriched(db, source, post.id.value):
                updated_posts += 1
                continue
            
            # Save or update post
            db.add_or_update_post(_create_post_dict(source, post))
            
            if existing:
                updated_posts += 1
            else:
                new_posts += 1
                
        except Exception:
            pass
    
    console.print(
        f"[green]✅ {new_posts} new, {updated_posts} updated "
        f"({len(posts)} total)[/green]"
    )
    
    return new_posts, updated_posts


def _is_already_enriched(db: PipelineDB, source: str, post_id: str) -> bool:
    """Check if post is already enriched.
    
    Args:
        db: Database instance
        source: Source name
        post_id: Post ID
        
    Returns:
        True if post has content_markdown, False otherwise
    """
    existing_post = next(
        (p for p in db.get_posts_by_source(source) if p['id'] == post_id),
        None
    )
    
    return existing_post and bool(existing_post.get('content_markdown'))


def _create_post_dict(source: str, post: Any) -> Dict[str, Any]:
    """Create post dictionary for database.
    
    Args:
        source: Source name
        post: Post object
        
    Returns:
        Dictionary with post data
    """
    return {
        'id': post.id.value,
        'source': source,
        'publication': source,  # Will be updated if available
        'title': post.title,
        'author': post.author.name if post.author else None,
        'url': getattr(post, 'url', None) or f"https://medium.com/p/{post.id.value}",
        'published_at': str(post.published_at) if hasattr(
            post,
            'published_at'
        ) else None,
        'reading_time': getattr(post, 'reading_time', 0),
        'claps': getattr(post, 'claps', None),
        'tags': getattr(post, 'tags', []) or [],
        'collection_mode': COLLECTION_MODE,
        'has_markdown': False
    }


def _has_posts_to_process(
    console: Console,
    db: PipelineDB,
    source: str,
    new_posts: int
) -> bool:
    """Check if there are posts to process.
    
    Args:
        console: Rich console for output
        db: Database instance
        source: Source name
        new_posts: Number of new posts collected
        
    Returns:
        True if there are NEW posts or posts needing enrichment/ML, False otherwise
    """
    if new_posts > 0:
        console.print(f"[green]✓ {new_posts} new posts to process[/green]")
        return True
    
    # Check for posts needing enrichment
    posts_needing_enrichment = db.get_posts_by_source(source)
    # Filter posts without content_markdown
    needing_enrichment = [
        p for p in posts_needing_enrichment 
        if not p.get('content_markdown')
    ]
    
    if needing_enrichment:
        console.print(
            f"[yellow]⚠️  No new posts, but {len(needing_enrichment)} "
            f"need enrichment[/yellow]"
        )
        return True
    
    # Check for posts needing ML
    posts_needing_ml = db.get_posts_needing_ml(source=source)
    if posts_needing_ml:
        console.print(
            f"[yellow]⚠️  No new posts, but {len(posts_needing_ml)} "
            f"need ML classification[/yellow]"
        )
        return True
    
    # All posts are fully processed
    total_posts = len(posts_needing_enrichment)
    if total_posts > 0:
        console.print(
            f"[green]✅ All {total_posts} posts already fully processed[/green]"
        )
    else:
        console.print("[red]❌ No posts (new or existing)[/red]")
    
    console.print()
    return False


def _run_etl_pipeline(
    console: Console,
    source: str,
    limit: Optional[int],
    force: bool,
    technical_only: bool
) -> bool:
    """Run ETL pipeline for source.
    
    Args:
        console: Rich console for output
        source: Source name
        limit: Optional limit on posts
        force: Force reprocess ML even if already classified
        technical_only: Process and export only technical posts
        
    Returns:
        True if successful, False otherwise
    """
    console.print("[blue]🔄 Enriching...[/blue]")
    
    # Build ETL command
    cmd = ["uv", "run", "python", "main.py", "etl", "--source", source]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if technical_only:
        cmd.append("--technical-only")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show live output
            text=True
            # No timeout - internal process, let it run as long as needed
        )
        
        if result.returncode == 0:
            console.print("[green]✅ Enriched & Timeline created[/green]")
            return True
        else:
            console.print("[yellow]⚠️  Enrich partially failed[/yellow]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ ETL failed: {e}[/red]")
        return False


def _print_final_summary(
    console: Console,
    sources: List[str],
    stats: Dict[str, Any]
) -> None:
    """Print final pipeline summary.
    
    Args:
        console: Rich console for output
        sources: List of all sources processed
        stats: Statistics dictionary
        
    Returns:
        None
    """
    console.print(f"[bold green]{'='*SEPARATOR_WIDTH}[/bold green]")
    console.print("[bold green]🎉 DONE![/bold green]")
    console.print(f"[bold green]{'='*SEPARATOR_WIDTH}[/bold green]\n")
    
    # Create summary table
    table = Table(show_header=True, border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow", justify="right")
    
    sources_processed = len(sources) - len(stats['failed'])
    
    table.add_row("Sources Processed", str(sources_processed))
    table.add_row("Posts Collected", str(stats['collected']))
    table.add_row("Sources Enriched", str(stats['enriched']))
    table.add_row("Timelines Created", str(stats['timelines']))
    
    if stats['failed']:
        table.add_row("Failed", str(len(stats['failed'])))
    
    console.print(table)
    
    # Show failures
    if stats['failed']:
        console.print(
            f"\n[yellow]⚠️  Failed: {', '.join(stats['failed'])}[/yellow]"
        )
    
    console.print("\n[cyan]✨ Check outputs/ folder for timelines![/cyan]\n")
