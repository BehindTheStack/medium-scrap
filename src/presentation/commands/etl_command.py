"""ETL Command - Extract, Transform, Load with ML.

This module provides the main ETL pipeline command that:
1. Enriches posts with HTML and Markdown content
2. Performs ML discovery (tech stack, patterns, solutions)
3. Generates timeline in JSON and Markdown formats

The pipeline processes posts from the database and enriches them with:
- Content extraction (HTML, Markdown)
- Technical classification
- ML-based discovery (clustering, NER, Q&A)
- Timeline generation with layer grouping

Author: BehindTheStack Team
License: MIT
"""

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table

from src.domain.entities.publication import Author, Post, PostId
from src.infrastructure import content_extractor
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.config.source_manager import SourceConfigManager
from src.infrastructure.external.repositories import (
    InMemoryPublicationRepository
)
from src.infrastructure.pipeline_db import PipelineDB
from src.presentation.helpers.ml_processor import MLProcessor
from src.presentation.helpers.progress_display import ProgressDisplay
from src.presentation.helpers.text_cleaner import clean_markdown


# Constants
SLOW_MODE_DELAY = 60  # seconds between requests
NORMAL_MODE_DELAY = 3  # seconds between requests
PROGRESS_DISPLAY_INTERVAL = 10
MAX_SNIPPET_LENGTH = 200
MAX_TITLE_DISPLAY_LENGTH = 80
TIMELINE_PREVIEW_LIMIT = 10
TOP_ITEMS_DISPLAY_LIMIT = 5
MIN_CONTENT_LENGTH = 100


@click.command('etl')
@click.option(
    '--source',
    '-s',
    required=True,
    help='Source to process (e.g., netflix)'
)
@click.option(
    '--limit',
    '-l',
    type=int,
    help='Limit posts to process (for testing)'
)
@click.option(
    '--slow-mode',
    is_flag=True,
    help='Ultra slow mode: 1 post per minute (avoid rate limiting)'
)
@click.option(
    '--technical-only',
    is_flag=True,
    help='Export only technical posts in timeline'
)
def etl_command(
    source: str,
    limit: Optional[int],
    slow_mode: bool,
    technical_only: bool
) -> None:
    """ETL Pipeline: Enrich + ML Discovery + Timeline.
    
    Process posts already in database:
    1. Enrich with HTML and Markdown (if missing)
    2. ML discovery (tech stack, patterns, solutions, etc)
    3. Generate timeline JSON + Markdown
    
    Args:
        source: Source name to process (e.g., 'netflix', 'airbnb')
        limit: Optional limit on number of posts to process
        slow_mode: Enable slow processing (1 post/min) to avoid rate limits
        technical_only: Export only technical posts in timeline
        
    Returns:
        None
        
    Examples:
        # Process all Netflix posts
        uv run python main.py etl --source netflix
        
        # Test with 20 posts
        uv run python main.py etl --source netflix --limit 20
        
        # Slow mode (avoid rate limits)
        uv run python main.py etl --source netflix --slow-mode
        
        # Export only technical posts
        uv run python main.py etl --source netflix --technical-only
    """
    console = Console()
    db = PipelineDB()
    
    _print_etl_header(console, source, slow_mode)
    
    # Step 1: Enrich posts with HTML & Markdown
    _enrich_posts(console, db, source, limit, slow_mode)
    
    # Step 2: ML Discovery & Timeline Generation
    _generate_timeline_with_ml(console, db, source, technical_only)


def _print_etl_header(
    console: Console,
    source: str,
    slow_mode: bool
) -> None:
    """Print ETL pipeline header.
    
    Args:
        console: Rich console for output
        source: Source name being processed
        slow_mode: Whether slow mode is enabled
        
    Returns:
        None
    """
    if slow_mode:
        console.print()
        console.print(
            "[yellow]⚠️  SLOW MODE enabled: 1 post per minute "
            "to avoid rate limiting[/yellow]"
        )
        console.print(
            "[dim]This will be VERY slow but safer for "
            "avoiding HTTP 429 errors[/dim]"
        )
        console.print()
    
    console.print()
    console.print("[bold cyan]🚀 ETL Pipeline: Enrich + Timeline[/bold cyan]")
    console.print(f"[dim]Source: {source}[/dim]")
    console.print()


def _enrich_posts(
    console: Console,
    db: PipelineDB,
    source: str,
    limit: Optional[int],
    slow_mode: bool
) -> None:
    """Enrich posts with HTML and Markdown content.
    
    Fetches HTML content, converts to Markdown, performs technical
    classification, and updates database records.
    
    Args:
        console: Rich console for output
        db: Database instance
        source: Source name to process
        limit: Optional limit on posts to process
        slow_mode: Whether to use slow processing mode
        
    Returns:
        None
    """
    console.print(
        "[bold blue]Step 1/2: Enriching with HTML & Markdown...[/bold blue]"
    )
    
    adapter = MediumApiAdapter()
    config_manager = SourceConfigManager()
    
    # Get posts needing enrichment
    total_posts, posts_to_enrich = _get_posts_needing_enrichment(
        db,
        source,
        limit
    )
    
    already_enriched = total_posts - len(posts_to_enrich)
    
    if not posts_to_enrich:
        console.print(
            f"[green]✅ All {total_posts} posts already enriched![/green]"
        )
        console.print()
        return
    
    if already_enriched > 0:
        console.print(
            f"[dim]ℹ️  {already_enriched} posts already enriched, "
            f"skipping...[/dim]"
        )
    
    console.print(f"[yellow]Processing {len(posts_to_enrich)} posts...[/yellow]")
    console.print()
    
    # Process each post
    enriched, failed, failure_reasons = _process_enrichment(
        console,
        db,
        adapter,
        config_manager,
        source,
        posts_to_enrich,
        slow_mode
    )
    
    _print_enrichment_summary(console, enriched, failed, failure_reasons)


def _get_posts_needing_enrichment(
    db: PipelineDB,
    source: str,
    limit: Optional[int]
) -> Tuple[int, List[Dict[str, Any]]]:
    """Get posts that need HTML/Markdown enrichment.
    
    Args:
        db: Database instance
        source: Source name to query
        limit: Optional limit on results
        
    Returns:
        Tuple of (total_posts, posts_to_enrich)
    """
    with db._get_connection() as conn:
        cursor = conn.cursor()
        
        # Count total posts
        cursor.execute(
            "SELECT COUNT(*) FROM posts WHERE source = ?",
            [source]
        )
        total_posts = cursor.fetchone()[0]
        
        # Get posts that need enrichment
        query = (
            "SELECT * FROM posts WHERE source = ? "
            "AND (content_html IS NULL OR content_markdown IS NULL)"
        )
        params: List[Any] = [source]
        
        if limit:
            query += " ORDER BY published_at DESC LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        posts_to_enrich = [dict(row) for row in cursor.fetchall()]
    
    return total_posts, posts_to_enrich


def _process_enrichment(
    console: Console,
    db: PipelineDB,
    adapter: MediumApiAdapter,
    config_manager: SourceConfigManager,
    source: str,
    posts: List[Dict[str, Any]],
    slow_mode: bool
) -> Tuple[int, int, Dict[str, int]]:
    """Process enrichment for all posts.
    
    Args:
        console: Rich console for output
        db: Database instance
        adapter: Medium API adapter
        config_manager: Source configuration manager
        source: Source name
        posts: List of posts to enrich
        slow_mode: Whether to use slow processing
        
    Returns:
        Tuple of (enriched_count, failed_count, failure_reasons_dict)
    """
    # Parallelized enrichment: fetch (threads) + parse (processes) + serial DB writes
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    import multiprocessing

    total = len(posts)
    enriched = 0
    failed = 0
    failure_reasons: Dict[str, int] = {}

    # Determine workers
    fetch_workers = 1 if slow_mode else min(8, max(2, (multiprocessing.cpu_count() // 2)))
    parse_workers = max(1, multiprocessing.cpu_count() // 2)

    console.print(f"[dim]Fetching HTML with {fetch_workers} threads and parsing with {parse_workers} processes...[/dim]")

    # Prepare post entities and configs
    post_entities = {}
    post_configs = {}
    for post_data in posts:
        cfg = _get_post_config(config_manager, source, post_data.get('publication'))
        post = _create_post_entity(post_data)
        post_entities[post_data['id']] = post
        post_configs[post_data['id']] = cfg

    # Phase 1: fetch HTML in threads (with live progress)
    html_map: Dict[str, Optional[str]] = {}
    import random
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        fetch_task = progress.add_task("Fetching HTML", total=total)

        with ThreadPoolExecutor(max_workers=fetch_workers) as tpool:
            future_to_pid = {}

            def _fetch_wrapper(post_obj, cfg_obj):
                # Small jitter to avoid bursty simultaneous requests
                try:
                    time.sleep(random.uniform(0.0, 0.8))
                except Exception:
                    pass
                return adapter.fetch_post_html(post_obj, cfg_obj)

            for post_data in posts:
                pid = post_data['id']
                post = post_entities[pid]
                cfg = post_configs[pid]
                future = tpool.submit(_fetch_wrapper, post, cfg)
                future_to_pid[future] = pid

            for fut in as_completed(future_to_pid):
                pid = future_to_pid[fut]
                try:
                    html = fut.result()
                except Exception as e:
                    html = None
                    reason = str(e)[:200]
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    console.print(f"[yellow]⚠ Fetch failed for {pid}: {reason}[/yellow]")
                html_map[pid] = html
                progress.update(fetch_task, advance=1, description=f"Fetching {pid}")

    # Phase 2: parse HTML -> markdown in processes (with live progress)
    parse_futures = {}
    # Count how many items will be parsed
    html_items = [(pid, html) for pid, html in html_map.items() if html]
    total_parse = len(html_items)
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

    parsed_map: Dict[str, Tuple[str, List[Dict], List[Dict]]] = {}
    if total_parse > 0:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            parse_task = progress.add_task("Parsing HTML", total=total_parse)

            with ProcessPoolExecutor(max_workers=parse_workers) as ppool:
                for pid, html in html_items:
                    # Submit parser; content_extractor.html_to_markdown is picklable as module-level
                    parse_futures[ppool.submit(content_extractor.html_to_markdown, html)] = pid

                for fut in as_completed(parse_futures):
                    pid = parse_futures[fut]
                    try:
                        md, assets, code_blocks = fut.result()
                        parsed_map[pid] = (md, assets, code_blocks)
                    except Exception as e:
                        reason = str(e)[:200]
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                        console.print(f"[yellow]⚠ Parse failed for {pid}: {reason}[/yellow]")
                    progress.update(parse_task, advance=1, description=f"Parsing {pid}")

    # Phase 3: assemble results and write to DB serially
    processed = 0
    for i, post_data in enumerate(posts, 1):
        pid = post_data['id']
        _show_enrichment_progress(console, i, total)

        html = html_map.get(pid)
        if not html:
            failed += 1
            console.print(f"[yellow]⚠ No HTML fetched for post {pid}; skipping.[/yellow]")
            continue

        parsed = parsed_map.get(pid)
        if not parsed:
            failed += 1
            continue

        try:
            md, assets, code_blocks = parsed

            # Get tags from post_data for classification
            post_tags = post_data.get('tags') or []
            classification = content_extractor.classify_technical(html, code_blocks, tags=post_tags)

            # Clean text for ML processing
            text_only = re.sub(r'[#*`\[\]()]+', ' ', md)
            text_only = re.sub(r'\s+', ' ', text_only).strip()

            # Update post data
            post_data['content_html'] = html
            post_data['content_markdown'] = md
            post_data['content_text'] = clean_markdown(text_only)
            post_data['is_technical'] = classification.get('is_technical')
            post_data['technical_score'] = classification.get('score')
            post_data['code_blocks'] = len(code_blocks)
            post_data['metadata'] = {
                'classifier': classification,
                'code_blocks': code_blocks,
                'assets': assets
            }

            db.add_or_update_post(post_data)
            enriched += 1
            processed += 1

            # Respect small delay after DB write if slow_mode to reduce load
            if slow_mode:
                time.sleep(SLOW_MODE_DELAY)
            else:
                time.sleep(NORMAL_MODE_DELAY)

        except Exception as e:
            failed += 1
            reason = str(e)[:50]
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return enriched, failed, failure_reasons


def _show_enrichment_progress(
    console: Console,
    current: int,
    total: int
) -> None:
    """Show enrichment progress.
    
    Args:
        console: Rich console for output
        current: Current post index
        total: Total posts to process
        
    Returns:
        None
    """
    if current % PROGRESS_DISPLAY_INTERVAL == 1 or current == total:
        percentage = current * 100 // total
        console.print(
            f"[cyan]Progress: {current}/{total} posts ({percentage}%)[/cyan]",
            end='\r'
        )


def _enrich_single_post(
    db: PipelineDB,
    adapter: MediumApiAdapter,
    config_manager: SourceConfigManager,
    source: str,
    post_data: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Enrich a single post with content and classification.
    
    Args:
        db: Database instance
        adapter: Medium API adapter
        config_manager: Source configuration manager
        source: Source name
        post_data: Post data dictionary
        
    Returns:
        Tuple of (success, failure_reason)
    """
    try:
        # Get configuration
        config = _get_post_config(
            config_manager,
            source,
            post_data['publication']
        )
        
        # Create Post entity
        post = _create_post_entity(post_data)
        
        # Fetch and process HTML
        html = adapter.fetch_post_html(post, config)
        if not html:
            return False, "No HTML returned from API"
        
        # Convert to markdown and extract metadata
        md, assets, code_blocks = content_extractor.html_to_markdown(html)
        
        # Get tags from post_data for classification
        post_tags = post_data.get('tags') or []
        classification = content_extractor.classify_technical(
            html,
            code_blocks,
            tags=post_tags
        )
        
        # Clean text for ML processing
        text_only = re.sub(r'[#*`\[\]()]+', ' ', md)
        text_only = re.sub(r'\s+', ' ', text_only).strip()
        
        # Update post data
        post_data['content_html'] = html
        post_data['content_markdown'] = md
        post_data['content_text'] = clean_markdown(text_only)
        post_data['is_technical'] = classification.get('is_technical')
        post_data['technical_score'] = classification.get('score')
        post_data['code_blocks'] = len(code_blocks)
        post_data['metadata'] = {
            'classifier': classification,
            'code_blocks': code_blocks,
            'assets': assets
        }
        
        db.add_or_update_post(post_data)
        return True, None
        
    except Exception as e:
        reason = str(e)[:50]
        return False, reason


def _get_post_config(
    config_manager: SourceConfigManager,
    source: str,
    publication: str
) -> Any:
    """Get configuration for post source.
    
    Args:
        config_manager: Source configuration manager
        source: Source name
        publication: Publication name
        
    Returns:
        Configuration object for the publication
    """
    try:
        sources = config_manager.load_sources()
        source_config = sources.get('sources', {}).get(source)
        
        if source_config:
            repo = InMemoryPublicationRepository()
            return repo.create_generic_config(
                source_config.get('publication', publication)
            )
        else:
            raise ValueError("Config not found")
    except Exception:
        repo = InMemoryPublicationRepository()
        return repo.create_generic_config(publication)


def _create_post_entity(post_data: Dict[str, Any]) -> Post:
    """Create Post entity from post data.
    
    Args:
        post_data: Post data dictionary from database
        
    Returns:
        Post entity instance
    """
    author_name = post_data.get('author') or 'Unknown'
    
    # Extract slug from URL
    slug = ''
    if post_data.get('url'):
        url_parts = post_data['url'].rstrip('/').split('/')
        if url_parts:
            slug = url_parts[-1]
    
    # If no slug from URL or slug is just the ID, generate a slug
    if not slug or not slug.strip():
        # Generate slug from title or use ID as fallback
        title = post_data.get('title', '').strip()
        if title:
            # Create slug from title (first 50 chars, replace spaces with hyphens)
            slug = title[:50].lower().replace(' ', '-').replace('_', '-')
            # Remove special characters except hyphens
            slug = ''.join(c for c in slug if c.isalnum() or c == '-')
            # Remove multiple consecutive hyphens
            slug = '-'.join(filter(None, slug.split('-')))
        
        # If still no slug, use ID
        if not slug:
            slug = post_data['id']
    
    return Post(
        id=PostId(post_data['id']),
        title=post_data['title'] or 'Untitled',
        slug=slug,
        author=Author(
            id='unknown',
            name=author_name,
            username=author_name.lower().replace(' ', '_')
        ),
        published_at=datetime.now(),
        reading_time=post_data.get('reading_time', 0)
    )


def _print_enrichment_summary(
    console: Console,
    enriched: int,
    failed: int,
    failure_reasons: Dict[str, int]
) -> None:
    """Print enrichment summary.
    
    Args:
        console: Rich console for output
        enriched: Number of successfully enriched posts
        failed: Number of failed posts
        failure_reasons: Dictionary of failure reasons and counts
        
    Returns:
        None
    """
    console.print()
    console.print(f"[green]✅ Enriched: {enriched}[/green]", end="")
    
    if failed > 0:
        console.print(f" [yellow]| Failed: {failed}[/yellow]")
        if failure_reasons:
            console.print("\n[yellow]Failure reasons:[/yellow]")
            sorted_reasons = sorted(
                failure_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for reason, count in sorted_reasons:
                console.print(f"  • {reason}: [red]{count}[/red]")
    else:
        console.print()
    
    console.print()


def _generate_timeline_with_ml(
    console: Console,
    db: PipelineDB,
    source: str,
    technical_only: bool = False
) -> None:
    """Generate timeline with ML discovery using hybrid approach.
    
    Hybrid Strategy:
    1. Get ONLY posts needing ML (not yet classified)
    2. Cluster ALL posts globally for accurate TF-IDF
    3. Extract features in batches for efficiency
    4. Save incrementally with progress tracking
    
    Args:
        console: Rich console for output
        db: Database instance
        source: Source name to process
        technical_only: Export only technical posts in timeline
        
    Returns:
        None
    """
    console.print(
        "[bold blue]Step 2/2: ML Discovery & "
        "Timeline Generation...[/bold blue]"
    )
    console.print(
        "[dim]Using: Clustering + NER + Q&A "
        "(NO hardcoded keywords!)[/dim]"
    )
    console.print()
    
    # Get ONLY posts needing ML (incremental processing)
    posts_needing_ml = db.get_posts_needing_ml(source=source)
    
    if not posts_needing_ml:
        console.print("[green]✅ All posts already ML classified![/green]")
        console.print("[dim]Run with --force to reprocess[/dim]")
        console.print()
        # Still generate timeline from existing ML data
        all_posts = db.get_posts_with_content(source=source)
        if all_posts:
            _save_timeline(console, all_posts, source, all_posts, technical_only)
        return
    
    # Filter to ONLY technical posts (skip creative/non-eng content)
    technical_posts = [
        p for p in posts_needing_ml 
        if p.get('is_technical', False) and p.get('technical_score', 0) >= 0.3
    ]
    
    skipped_non_technical = len(posts_needing_ml) - len(technical_posts)
    
    if skipped_non_technical > 0:
        console.print(
            f"[dim]ℹ️  Skipping {skipped_non_technical} non-technical posts "
            f"(focusing on engineering content)[/dim]"
        )
    
    if not technical_posts:
        console.print(
            "[yellow]⚠️  No technical posts to process with ML[/yellow]"
        )
        console.print()
        return
    
    console.print(
        f"[yellow]Processing {len(technical_posts)} technical posts "
        f"with ML classification...[/yellow]"
    )
    console.print()
    
    # Prepare entries for ML (using only technical posts)
    entries_for_ml = _prepare_ml_entries(technical_posts)
    
    if not entries_for_ml:
        console.print("[red]❌ No posts with valid content![/red]")
        return
    
    # Run ML processing with batch optimization
    _run_ml_processing_optimized(console, db, entries_for_ml, batch_size=50)
    
    # Generate and save timeline from ALL posts (including newly processed)
    all_posts = db.get_posts_with_content(source=source)
    _save_timeline(console, all_posts, source, all_posts, technical_only)


def _prepare_ml_entries(
    posts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Prepare posts for ML processing.
    
    Args:
        posts: List of post dictionaries from database
        
    Returns:
        List of entries formatted for ML processor
    """
    entries_for_ml = []
    
    for post in posts:
        md = post.get('content_markdown', '')
        if not md:
            continue
        
        date = _parse_post_date(post.get('published_at'))
        
        entries_for_ml.append({
            'id': post['id'],
            'title': post.get('title', 'Untitled'),
            'date': date.isoformat() if date else None,
            'content': md,
            'url': post.get('url', ''),
            'author': post.get('author', 'Unknown'),
            'reading_time': post.get('reading_time', 0),
            'is_technical': post.get('is_technical', False),
            'technical_score': post.get('technical_score', 0.0),
            'code_blocks': post.get('code_blocks', 0),
        })
    
    return entries_for_ml


def _parse_post_date(published_at: Any) -> Optional[datetime]:
    """Parse published_at to datetime.
    
    Args:
        published_at: Published date string or None
        
    Returns:
        Parsed datetime.date or None if parsing fails
    """
    if not published_at:
        return None
    
    try:
        return datetime.fromisoformat(
            str(published_at).replace('Z', '+00:00')
        ).date()
    except Exception:
        return None


def _run_ml_processing(
    console: Console,
    db: PipelineDB,
    entries: List[Dict[str, Any]]
) -> None:
    """Run ML processing on entries.
    
    Args:
        console: Rich console for output
        db: Database instance
        entries: List of entries to process
        
    Returns:
        None
        
    Raises:
        Exception: If ML processing fails
    """
    ml_processor = MLProcessor(console)
    ml_processor.load_models()

    try:
        # Process posts and use incremental save callback so each post is
        # persisted immediately after extraction. This makes the ML stage
        # resumable if the process dies mid-run.
        console.print("[cyan]🤖 Running ML processing (incremental save)...[/cyan]")
        ml_processor.process_posts(entries, save_callback=db.update_ml_discovery)
        console.print("[green]✓ ML processing completed (incremental saves applied)[/green]")
        console.print()

    except Exception as e:
        console.print(f"[red]❌ ML Discovery failed: {e}[/red]")
        raise


def _run_ml_processing_optimized(
    console: Console,
    db: PipelineDB,
    entries: List[Dict[str, Any]],
    batch_size: int = 50
) -> None:
    """Run ML processing with hybrid optimization.
    
    Strategy:
    1. Cluster ALL posts globally (accurate TF-IDF)
    2. Extract features in batches (GPU efficiency)
    3. Save incrementally (memory efficiency + resumable)
    
    Args:
        console: Rich console for output
        db: Database instance
        entries: List of entries to process
        batch_size: Number of posts per batch for extraction
        
    Returns:
        None
        
    Raises:
        Exception: If ML processing fails
    """
    import torch
    from pathlib import Path
    import sys
    
    # Add ML classifier path
    ml_path = Path(__file__).parent.parent.parent / 'ml_classifier'
    sys.path.insert(0, str(ml_path))
    
    from discover_enriched import (
        extract_tech_stack, extract_patterns, extract_solutions,
        extract_problem, extract_approach
    )
    from ..helpers.text_cleaner import clean_markdown
    
    ml_processor = MLProcessor(console)
    ml_processor.load_models()
    
    try:
        # STEP 1: Global clustering (TF-IDF needs all posts for accurate IDF)
        console.print("[cyan]📊 Step 1/6: Global clustering for topic layers...[/cyan]")
        texts = [e.get('content', '') for e in entries]
        cluster_result = ml_processor.cluster_topics(entries, texts)
        
        # STEP 2: Extract features in batches
        total_batches = (len(entries) + batch_size - 1) // batch_size
        console.print(f"[cyan]🤖 Extracting features in {total_batches} batches of {batch_size}...[/cyan]")
        console.print()
        
        stats = {
            'tech_count': 0,
            'patterns_count': 0,
            'solutions_count': 0,
            'problems_count': 0,
            'approaches_count': 0
        }
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(entries))
            batch = entries[start_idx:end_idx]
            
            console.print(f"[dim]Batch {batch_num + 1}/{total_batches} ({len(batch)} posts)[/dim]")
            
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Tech Stack (NER - efficient in batch)
                task = progress.add_task("  Tech Stack", total=len(batch))
                for entry in batch:
                    entry['tech_stack'] = extract_tech_stack(entry['content'], ml_processor.ner_pipeline)
                    stats['tech_count'] += len(entry['tech_stack'])
                    progress.update(task, advance=1)
                
                # Patterns (NER + embeddings)
                task = progress.add_task("  Patterns  ", total=len(batch))
                for entry in batch:
                    entry['patterns'] = extract_patterns(entry['content'], ml_processor.ner_pipeline, ml_processor.embedder)
                    stats['patterns_count'] += len(entry['patterns'])
                    progress.update(task, advance=1)
                
                # Solutions (embeddings)
                task = progress.add_task("  Solutions ", total=len(batch))
                for entry in batch:
                    entry['solutions'] = extract_solutions(entry['content'], entry.get('tech_stack', []), ml_processor.embedder)
                    stats['solutions_count'] += len(entry['solutions'])
                    progress.update(task, advance=1)
                
                # Problems (Q&A)
                task = progress.add_task("  Problems  ", total=len(batch))
                for entry in batch:
                    entry['problem'] = extract_problem(entry['content'], ml_processor.qa_pipeline)
                    if entry['problem']:
                        stats['problems_count'] += 1
                    progress.update(task, advance=1)
                
                # Approaches (Q&A)
                task = progress.add_task("  Approaches", total=len(batch))
                for entry in batch:
                    entry['approach'] = extract_approach(entry['content'], ml_processor.qa_pipeline)
                    if entry['approach']:
                        stats['approaches_count'] += 1
                    progress.update(task, advance=1)
            
            # Save batch to database
            console.print(f"[dim]  💾 Saving batch {batch_num + 1}...[/dim]")
            for entry in batch:
                ml_data = {
                    'layers': entry.get('layers', []),
                    'tech_stack': entry.get('tech_stack', []),
                    'patterns': entry.get('patterns', []),
                    'solutions': entry.get('solutions', []),
                    'problem': entry.get('problem'),
                    'approach': entry.get('approach')
                }
                db.update_ml_discovery(entry['id'], ml_data)
            
            # Clear GPU cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            console.print()
        
        # Print final statistics
        console.print("[green]✅ ML Processing Complete![/green]")
        console.print(f"[dim]  • Tech Stack: {stats['tech_count']} items[/dim]")
        console.print(f"[dim]  • Patterns: {stats['patterns_count']} items[/dim]")
        console.print(f"[dim]  • Solutions: {stats['solutions_count']} items[/dim]")
        console.print(f"[dim]  • Problems: {stats['problems_count']} posts[/dim]")
        console.print(f"[dim]  • Approaches: {stats['approaches_count']} posts[/dim]")
        console.print()
        
    except Exception as e:
        console.print(f"[red]❌ ML Discovery failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise
        console.print()
        
    except Exception as e:
        console.print(f"[red]❌ ML Discovery failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise


def _parse_json_field(field: Any) -> Any:
    """Parse JSON field from database.
    
    Args:
        field: Field value (can be string or already parsed)
        
    Returns:
        Parsed object or None if parsing fails
    """
    if field is None:
        return None
    
    if isinstance(field, str):
        try:
            return json.loads(field)
        except Exception:
            return None
    
    return field


def _convert_posts_to_entries(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert database posts to timeline entry format.
    
    Args:
        posts: List of posts from database with ML data
        
    Returns:
        List of entries formatted for timeline export
    """
    entries = []
    for post in posts:
        date = _parse_post_date(post.get('published_at'))
        
        entry = {
            'id': post['id'],
            'title': post.get('title', 'Untitled'),
            'date': date.isoformat() if date else None,
            'url': post.get('url'),
            'author': post.get('author', 'Unknown'),
            'reading_time': post.get('reading_time', 0),
            'content': post.get('content_markdown', ''),
            'is_technical': post.get('is_technical', False),
            'technical_score': post.get('technical_score', 0.0),
            'code_blocks': post.get('code_blocks', 0),
            # ML data
            'layers': _parse_json_field(post.get('ml_layers')),
            'tech_stack': _parse_json_field(post.get('tech_stack')),
            'patterns': _parse_json_field(post.get('patterns')),
            'solutions': _parse_json_field(post.get('solutions')),
            'problem': post.get('problem'),
            'approach': post.get('approach'),
        }
        entries.append(entry)
    
    return entries


def _save_timeline(
    console: Console,
    entries: List[Dict[str, Any]],
    source: str,
    posts: List[Dict[str, Any]],
    technical_only: bool = False
) -> None:
    """Save timeline to JSON and Markdown files.
    
    Args:
        console: Rich console for output
        entries: List of processed entries (can be DB posts or prepared entries)
        source: Source name
        posts: Original posts from database
        technical_only: Export only technical posts
        
    Returns:
        None
    """
    # Convert DB posts to entries if needed (check if entries have 'date' field)
    if entries and 'date' not in entries[0]:
        entries = _convert_posts_to_entries(entries)
    
    # Filter technical posts if requested
    if technical_only:
        entries = [e for e in entries if e.get('is_technical', False)]
        console.print(f"[dim]ℹ️  Exporting only {len(entries)} technical posts[/dim]")
    
    # Prepare clean entries (remove large content field)
    clean_entries = _prepare_clean_entries(entries)
    
    # Sort by date
    clean_entries.sort(
        key=lambda e: (e['date'] is None, e['date'] or '9999-12-31')
    )
    
    # Build timeline structure
    timeline = _build_timeline_structure(clean_entries, source, posts)
    
    # Save files
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    json_file = output_dir / f"{source}_timeline.json"
    md_file = output_dir / f"{source}_timeline.md"
    
    _save_json_timeline(json_file, timeline)
    _save_markdown_timeline(md_file, timeline)
    
    console.print(f"[green]✅ {json_file.name}[/green]")
    console.print(f"[green]✅ {md_file.name}[/green]")
    console.print()
    
    # Print summary
    _print_summary(console, timeline, source)


def _prepare_clean_entries(
    entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Prepare clean entries without large content fields.
    
    Args:
        entries: List of entries with content
        
    Returns:
        List of clean entries with snippets instead of full content
    """
    clean_entries = []
    
    for e in entries:
        entry_copy = e.copy()
        entry_copy.pop('content', None)
        
        # Add snippet
        md = e.get('content', '')
        snippet = _extract_snippet(md)
        entry_copy['snippet'] = snippet
        
        clean_entries.append(entry_copy)
    
    return clean_entries


def _extract_snippet(content: str) -> str:
    """Extract snippet from content.
    
    Args:
        content: Full content markdown
        
    Returns:
        Short snippet (max 200 chars)
    """
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('---'):
            return line[:MAX_SNIPPET_LENGTH]
    
    return ''


def _build_timeline_structure(
    entries: List[Dict[str, Any]],
    source: str,
    posts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build timeline data structure.
    
    Args:
        entries: List of clean entries
        source: Source name
        posts: Original posts from database
        
    Returns:
        Timeline dictionary with all metadata and stats
    """
    # Group by layer
    per_layer: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        for layer in e.get('layers', ['Uncategorized']):
            per_layer.setdefault(layer, []).append(e)
    
    # Count technologies and patterns
    all_techs = []
    all_patterns = []
    for e in entries:
        tech_stack = e.get('tech_stack') or []
        patterns = e.get('patterns') or []
        all_techs.extend([t['name'] for t in tech_stack if t])
        all_patterns.extend([p['pattern'] for p in patterns if p])
    
    tech_counter = Counter(all_techs)
    pattern_counter = Counter(all_patterns)
    
    return {
        'count': len(entries),
        'publication': posts[0]['publication'] if posts else source,
        'source': source,
        'posts': entries,
        'per_layer': per_layer,
        'stats': {
            'total_posts': len(entries),
            'technical_posts': sum(
                1 for e in entries if e.get('is_technical')
            ),
            'layers': {
                layer: len(items) for layer, items in per_layer.items()
            },
            'ml_discovery': {
                'method': 'clustering + ner + qa',
                'n_topics': len(per_layer),
                'total_tech_mentions': len(all_techs),
                'unique_technologies': len(tech_counter),
                'total_patterns': len(all_patterns),
                'unique_patterns': len(pattern_counter),
                'posts_with_solutions': sum(
                    1 for e in entries if e.get('solutions')
                ),
            },
            'top_technologies': [
                {'name': k, 'count': v}
                for k, v in tech_counter.most_common(10)
            ],
            'top_patterns': [
                {'pattern': k, 'count': v}
                for k, v in pattern_counter.most_common(10)
            ],
        }
    }


def _save_json_timeline(file_path: Path, timeline: Dict[str, Any]) -> None:
    """Save timeline as JSON file.
    
    Args:
        file_path: Path to save JSON file
        timeline: Timeline dictionary
        
    Returns:
        None
    """
    file_path.write_text(
        json.dumps(timeline, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )


def _save_markdown_timeline(
    file_path: Path,
    timeline: Dict[str, Any]
) -> None:
    """Save timeline as Markdown file.
    
    Args:
        file_path: Path to save Markdown file
        timeline: Timeline dictionary
        
    Returns:
        None
    """
    md_lines = [f"# {timeline['publication']} Timeline\n"]
    md_lines.append(
        f"**Total**: {timeline['stats']['total_posts']} posts"
    )
    md_lines.append(
        f"**Technical**: {timeline['stats']['technical_posts']} posts\n"
    )
    md_lines.append('\n## Architecture Layers\n')
    
    # Sort layers by size
    sorted_layers = sorted(
        timeline['per_layer'].items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for layer, items in sorted_layers:
        md_lines.append(f'\n### {layer} ({len(items)})\n')
        
        # Show preview of posts
        for item in items[:TIMELINE_PREVIEW_LIMIT]:
            date = item['date'] or 'unknown'
            title = item['title'][:MAX_TITLE_DISPLAY_LENGTH]
            md_lines.append(f"- **{date}** — {title}")
        
        if len(items) > TIMELINE_PREVIEW_LIMIT:
            remaining = len(items) - TIMELINE_PREVIEW_LIMIT
            md_lines.append(f"  _... +{remaining} more_\n")
    
    file_path.write_text('\n'.join(md_lines), encoding='utf-8')


def _print_summary(
    console: Console,
    timeline: Dict[str, Any],
    source: str
) -> None:
    """Print final summary table.
    
    Args:
        console: Rich console for output
        timeline: Timeline dictionary with stats
        source: Source name
        
    Returns:
        None
    """
    console.print("[bold green]🎉 Complete![/bold green]")
    console.print()
    
    # Main stats table
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total Posts", str(timeline['stats']['total_posts']))
    table.add_row(
        "Technical Posts",
        str(timeline['stats']['technical_posts'])
    )
    table.add_row("In Timeline", str(timeline['count']))
    
    # ML discovery stats
    if 'ml_discovery' in timeline['stats']:
        ml_stats = timeline['stats']['ml_discovery']
        table.add_row("", "")
        table.add_row("ML Topics", str(ml_stats['n_topics']))
        table.add_row("Tech Mentions", str(ml_stats['total_tech_mentions']))
        table.add_row("Unique Techs", str(ml_stats['unique_technologies']))
        table.add_row("Patterns Found", str(ml_stats['total_patterns']))
        table.add_row("With Solutions", str(ml_stats['posts_with_solutions']))
    
    console.print(table)
    console.print()
    
    # Topics/Layers
    _print_discovered_layers(console, timeline)
    
    # Top technologies
    _print_top_technologies(console, timeline)
    
    # Top patterns
    _print_top_patterns(console, timeline)


def _print_discovered_layers(
    console: Console,
    timeline: Dict[str, Any]
) -> None:
    """Print discovered architecture layers.
    
    Args:
        console: Rich console for output
        timeline: Timeline dictionary
        
    Returns:
        None
    """
    console.print("[bold]📊 Discovered Topics (Layers):[/bold]")
    
    sorted_layers = sorted(
        timeline['stats']['layers'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for layer, count in sorted_layers:
        console.print(f"   • {layer}: [green]{count}[/green]")
    
    console.print()


def _print_top_technologies(
    console: Console,
    timeline: Dict[str, Any]
) -> None:
    """Print top technologies discovered.
    
    Args:
        console: Rich console for output
        timeline: Timeline dictionary
        
    Returns:
        None
    """
    if not timeline['stats'].get('top_technologies'):
        return
    
    console.print("[bold]🔧 Top Technologies:[/bold]")
    
    top_techs = timeline['stats']['top_technologies']
    for tech in top_techs[:TOP_ITEMS_DISPLAY_LIMIT]:
        console.print(f"   • {tech['name']}: [green]{tech['count']}[/green]")
    
    console.print()


def _print_top_patterns(
    console: Console,
    timeline: Dict[str, Any]
) -> None:
    """Print top patterns discovered.
    
    Args:
        console: Rich console for output
        timeline: Timeline dictionary
        
    Returns:
        None
    """
    if not timeline['stats'].get('top_patterns'):
        return
    
    console.print("[bold]🏗️  Top Patterns:[/bold]")
    
    top_patterns = timeline['stats']['top_patterns']
    for pattern in top_patterns[:TOP_ITEMS_DISPLAY_LIMIT]:
        console.print(
            f"   • {pattern['pattern']}: [green]{pattern['count']}[/green]"
        )
    
    console.print()
