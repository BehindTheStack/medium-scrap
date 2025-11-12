"""Export Timeline Command - Export complete timeline with ML discovery data.

This module provides CLI commands to export timelines with ML discovery data
in JSON and Markdown formats. Supports single source or combined exports.

The exported timeline includes:
- Complete ML discovery data (layers, tech stack, patterns, solutions)
- Technical classification scores
- Aggregated statistics
- Layer-based grouping

Author: BehindTheStack Team
License: MIT
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn
)

from src.infrastructure.pipeline_db import PipelineDB


# Constants
MIN_CONTENT_LENGTH = 100
MAX_SNIPPET_LENGTH = 200
COMBINED_FILENAME = "all_sources_timeline"
DEFAULT_OUTPUT_DIR = "outputs"
TIMELINE_VERSION = "2.0"
TOP_TECH_LIMIT = 20
TOP_TECH_COMBINED_LIMIT = 30
MARKDOWN_PREVIEW_LIMIT = 15
MARKDOWN_COMBINED_PREVIEW_LIMIT = 20


@click.command('export-timeline')
@click.option(
    '--source',
    '-s',
    help='Export specific source only'
)
@click.option(
    '--all',
    'all_sources',
    is_flag=True,
    help='Export ALL sources'
)
@click.option(
    '--combined',
    '-c',
    is_flag=True,
    help='Combine all sources in one file (only with --all)'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output directory (default: outputs/)'
)
@click.option(
    '--format',
    '-f',
    type=click.Choice(['json', 'both']),
    default='both',
    help='Output format'
)
def export_timeline_command(
    source: Optional[str],
    all_sources: bool,
    combined: bool,
    output: Optional[str],
    format: str
) -> None:
    """Export Timeline with Complete ML Discovery Data.
    
    IMPORTANT: Run 'reprocess-ml' FIRST to classify posts with ML!
    This command exports ONLY posts with ml_classified=1.
    
    Export timeline JSON with all ML discoveries:
    - Layers (clustering)
    - Tech Stack (NER)
    - Patterns (NER + n-grams)
    - Solutions (semantic similarity)
    - Problem & Approach (Q&A)
    
    Args:
        source: Specific source name to export
        all_sources: Export all available sources
        combined: Combine all sources in one file (requires all_sources)
        output: Custom output directory path
        format: Output format ('json' or 'both')
        
    Returns:
        None
        
    Examples:
        # First: Process with ML
        uv run python main.py reprocess-ml --all
        
        # Then: Export one source
        uv run python main.py export-timeline --source netflix
        
        # Export ALL sources (separate files)
        uv run python main.py export-timeline --all
        
        # Export ALL sources COMBINED in one file
        uv run python main.py export-timeline --all --combined
        
        # Custom output directory
        uv run python main.py export-timeline --all --output ./timelines
        
        # JSON only (no markdown)
        uv run python main.py export-timeline --all --format json
    """
    console = Console()
    db = PipelineDB()
    
    output_dir = Path(output) if output else Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    _print_export_header(console)
    
    # Check ML classification status
    ml_status = _check_ml_status(db, console)
    if ml_status is None:
        return
    
    # Get available sources
    available_sources = _get_available_sources(db, console)
    if not available_sources:
        return
    
    # Determine which sources to export
    sources_to_export = _determine_sources_to_export(
        console,
        all_sources,
        source,
        available_sources
    )
    if not sources_to_export:
        return
    
    console.print()
    
    # Export based on mode
    if all_sources and combined:
        _export_combined_mode(db, sources_to_export, output_dir, format, console)
    else:
        _export_separate_mode(db, sources_to_export, output_dir, format, console)


def _print_export_header(console: Console) -> None:
    """Print export command header.
    
    Args:
        console: Rich console for output
        
    Returns:
        None
    """
    console.print()
    console.print(
        "[bold cyan]📊 Timeline Export - ML Discovery Edition[/bold cyan]"
    )
    console.print(
        "[dim]Format: Complete ML data "
        "(tech_stack, patterns, solutions, problem, approach)[/dim]"
    )
    console.print()


def _check_ml_status(
    db: PipelineDB,
    console: Console
) -> Optional[Tuple[int, int]]:
    """Check ML classification status in database.
    
    Args:
        db: Database instance
        console: Rich console for output
        
    Returns:
        Tuple of (total_posts, posts_with_ml) or None if no ML data
    """
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN ml_classified = 1 THEN 1 ELSE 0 END) as with_ml
            FROM posts 
            WHERE content_markdown IS NOT NULL
        """)
        row = cursor.fetchone()
        total_posts = row['total']
        posts_with_ml = row['with_ml'] or 0
    
    if posts_with_ml == 0:
        console.print(
            f"[red]⚠️  WARNING: 0/{total_posts} posts have "
            f"ML classification![/red]"
        )
        console.print(
            "[yellow]Run 'reprocess-ml' first to classify "
            "posts with ML[/yellow]"
        )
        console.print(
            "[dim]Example: uv run python main.py reprocess-ml --all[/dim]"
        )
        console.print()
        return None
    
    percentage = posts_with_ml * 100 // total_posts
    console.print(
        f"[green]✓ Found {posts_with_ml}/{total_posts} posts with "
        f"ML data ({percentage}%)[/green]"
    )
    console.print()
    
    return total_posts, posts_with_ml


def _get_available_sources(
    db: PipelineDB,
    console: Console
) -> List[str]:
    """Get available sources from database.
    
    Args:
        db: Database instance
        console: Rich console for output
        
    Returns:
        List of source names, empty if none found
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
            "[red]❌ No sources with content found in database[/red]"
        )
        return []
    
    return available_sources


def _determine_sources_to_export(
    console: Console,
    all_sources: bool,
    source: Optional[str],
    available_sources: List[str]
) -> Optional[List[str]]:
    """Determine which sources to export.
    
    Args:
        console: Rich console for output
        all_sources: Whether to export all sources
        source: Specific source name if provided
        available_sources: List of available source names
        
    Returns:
        List of source names to export, or None if invalid input
    """
    if all_sources:
        sources = available_sources
        console.print(
            f"[cyan]Exporting ALL {len(sources)} sources[/cyan]"
        )
        return sources
    
    if source:
        if source not in available_sources:
            console.print(f"[red]❌ Source '{source}' not found[/red]")
            console.print(
                f"[yellow]Available: {', '.join(available_sources)}[/yellow]"
            )
            return None
        return [source]
    
    # No source specified
    console.print("[red]❌ Must specify --source or --all[/red]")
    console.print("[yellow]Examples:[/yellow]")
    console.print("  uv run python main.py export-timeline --source netflix")
    console.print("  uv run python main.py export-timeline --all")
    return None


def _export_combined_mode(
    db: PipelineDB,
    sources: List[str],
    output_dir: Path,
    format: str,
    console: Console
) -> None:
    """Export all sources combined in one file.
    
    Args:
        db: Database instance
        sources: List of source names to export
        output_dir: Output directory path
        format: Output format ('json' or 'both')
        console: Rich console for output
        
    Returns:
        None
    """
    console.print(
        f"[cyan]Combining {len(sources)} sources into one file...[/cyan]"
    )
    
    timeline = _build_combined_timeline(db, sources, console)
    
    if not timeline:
        return
    
    # Save files
    if format in ['json', 'both']:
        _save_json_timeline(
            output_dir / f"{COMBINED_FILENAME}.json",
            timeline
        )
    
    if format == 'both':
        _save_combined_markdown(
            output_dir / f"{COMBINED_FILENAME}.md",
            timeline
        )
    
    console.print()
    console.print("[bold green]✅ Exported combined timeline[/bold green]")
    console.print(
        f"[cyan]📁 Output: {output_dir.absolute()}/"
        f"{COMBINED_FILENAME}.json[/cyan]"
    )
    console.print()


def _export_separate_mode(
    db: PipelineDB,
    sources: List[str],
    output_dir: Path,
    format: str,
    console: Console
) -> None:
    """Export sources as separate files.
    
    Args:
        db: Database instance
        sources: List of source names to export
        output_dir: Output directory path
        format: Output format ('json' or 'both')
        console: Rich console for output
        
    Returns:
        None
    """
    exported_count = 0
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Exporting sources",
            total=len(sources)
        )
        
        for src in sources:
            _export_source_timeline(db, src, output_dir, format, console)
            exported_count += 1
            progress.update(task, advance=1)
    
    console.print()
    console.print(
        f"[bold green]✅ Exported {exported_count} source(s)[/bold green]"
    )
    console.print(f"[cyan]📁 Output directory: {output_dir.absolute()}[/cyan]")
    console.print()


def _build_combined_timeline(
    db: PipelineDB,
    sources: List[str],
    console: Console
) -> Optional[Dict[str, Any]]:
    """Build combined timeline from multiple sources.
    
    Args:
        db: Database instance
        sources: List of source names
        console: Rich console for output
        
    Returns:
        Timeline dictionary or None if no data
    """
    all_entries = []
    sources_stats = {}
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Loading sources",
            total=len(sources)
        )
        
        for src in sources:
            posts = _get_ml_classified_posts(db, src)
            
            if not posts:
                progress.update(task, advance=1)
                continue
            
            source_entries = _build_entries_from_posts(posts, src)
            
            sources_stats[src] = {
                'count': len(source_entries),
                'publication': posts[0].get('publication', src) if posts else src
            }
            
            all_entries.extend(source_entries)
            progress.update(task, advance=1)
    
    if not all_entries:
        console.print(
            "[yellow]⚠️  No posts with ML data found "
            "in any source[/yellow]"
        )
        return None
    
    # Sort by date
    all_entries.sort(
        key=lambda e: (e['date'] is None, e['date'] or '9999-12-31')
    )
    
    # Build timeline structure
    return _create_combined_timeline_structure(
        all_entries,
        sources_stats,
        len(sources)
    )


def _get_ml_classified_posts(
    db: PipelineDB,
    source: str
) -> List[Dict[str, Any]]:
    """Get posts with ML classification from database.
    
    Args:
        db: Database instance
        source: Source name
        
    Returns:
        List of post dictionaries
    """
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM posts 
            WHERE source = ? 
            AND content_markdown IS NOT NULL
            AND ml_classified = 1
            ORDER BY published_at DESC
        """, (source,))
        return [dict(row) for row in cursor.fetchall()]


def _build_entries_from_posts(
    posts: List[Dict[str, Any]],
    source: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Build timeline entries from posts.
    
    Args:
        posts: List of post dictionaries
        source: Optional source name (for combined mode)
        
    Returns:
        List of timeline entry dictionaries
    """
    entries = []
    
    for post in posts:
        md = post.get('content_markdown', '')
        if not md or len(md) < MIN_CONTENT_LENGTH:
            continue
        
        date = _parse_published_date(post.get('published_at'))
        snippet = _extract_snippet_from_markdown(md)
        
        # Parse ML data
        ml_data = {
            'layers': _parse_json_field(post.get('layers')) or [],
            'tech_stack': _parse_json_field(post.get('tech_stack')) or [],
            'patterns': _parse_json_field(post.get('patterns')) or [],
            'solutions': _parse_json_field(post.get('solutions')) or [],
            'problem': post.get('problem'),
            'approach': post.get('approach'),
            'ml_classified': bool(post.get('ml_classified', 0)),
        }
        
        entry = {
            'id': post['id'],
            'title': post.get('title', 'Untitled'),
            'date': date.isoformat() if date else None,
            'url': post.get('url'),
            'author': post.get('author', 'Unknown'),
            'reading_time': post.get('reading_time', 0),
            'snippet': snippet,
            'is_technical': post.get('is_technical', False),
            'technical_score': post.get('technical_score', 0.0),
            'code_blocks': post.get('code_blocks', 0),
            'ml_discovery': ml_data
        }
        
        # Add source field for combined mode
        if source:
            entry['source'] = source
            entry['publication'] = post.get('publication', source)
        
        entries.append(entry)
    
    return entries


def _parse_published_date(published_at: Any) -> Optional[datetime]:
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


def _extract_snippet_from_markdown(content: str) -> str:
    """Extract snippet from markdown content.
    
    Args:
        content: Full markdown content
        
    Returns:
        Short snippet (max 200 chars)
    """
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('---'):
            return line[:MAX_SNIPPET_LENGTH]
    
    return ''


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


def _create_combined_timeline_structure(
    entries: List[Dict[str, Any]],
    sources_stats: Dict[str, Dict[str, Any]],
    total_sources: int
) -> Dict[str, Any]:
    """Create combined timeline structure.
    
    Args:
        entries: List of timeline entries
        sources_stats: Statistics per source
        total_sources: Total number of sources
        
    Returns:
        Complete timeline dictionary
    """
    # Group by layer
    per_layer = _group_entries_by_layer(entries)
    
    # Aggregate statistics
    stats = _aggregate_ml_statistics(entries, per_layer)
    
    return {
        'version': TIMELINE_VERSION,
        'exported_at': datetime.now().isoformat(),
        'type': 'combined',
        'total_sources': total_sources,
        'count': len(entries),
        'sources': sources_stats,
        'posts': entries,
        'per_layer': per_layer,
        'stats': stats
    }


def _export_source_timeline(
    db: PipelineDB,
    source: str,
    output_dir: Path,
    format: str,
    console: Console
) -> None:
    """Export timeline for a single source.
    
    Args:
        db: Database instance
        source: Source name
        output_dir: Output directory path
        format: Output format ('json' or 'both')
        console: Rich console for output
        
    Returns:
        None
    """
    posts = _get_ml_classified_posts(db, source)
    
    if not posts:
        console.print(
            f"[yellow]⚠️  {source}: No posts with ML data "
            f"(run 'reprocess-ml' first)[/yellow]"
        )
        return
    
    entries = _build_entries_from_posts(posts)
    
    if not entries:
        console.print(f"[yellow]⚠️  {source}: No valid entries[/yellow]")
        return
    
    # Sort by date
    entries.sort(
        key=lambda e: (e['date'] is None, e['date'] or '9999-12-31')
    )
    
    # Build timeline
    timeline = _create_source_timeline_structure(entries, source, posts)
    
    # Save files
    if format in ['json', 'both']:
        _save_json_timeline(
            output_dir / f"{source}_timeline.json",
            timeline
        )
    
    if format == 'both':
        _save_source_markdown(
            output_dir / f"{source}_timeline.md",
            timeline
        )


def _create_source_timeline_structure(
    entries: List[Dict[str, Any]],
    source: str,
    posts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create timeline structure for single source.
    
    Args:
        entries: List of timeline entries
        source: Source name
        posts: Original posts from database
        
    Returns:
        Complete timeline dictionary
    """
    # Group by layer
    per_layer = _group_entries_by_layer(entries)
    
    # Aggregate statistics
    stats = _aggregate_ml_statistics(entries, per_layer)
    
    return {
        'version': TIMELINE_VERSION,
        'exported_at': datetime.now().isoformat(),
        'source': source,
        'publication': posts[0].get('publication', source) if posts else source,
        'count': len(entries),
        'posts': entries,
        'per_layer': per_layer,
        'stats': stats
    }


def _group_entries_by_layer(
    entries: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Group entries by architecture layer.
    
    Args:
        entries: List of timeline entries
        
    Returns:
        Dictionary mapping layer names to entry lists
    """
    per_layer: Dict[str, List[Dict[str, Any]]] = {}
    
    for e in entries:
        layers = e['ml_discovery']['layers']
        if not layers:
            layers = ['Uncategorized']
        
        for layer in layers:
            per_layer.setdefault(layer, []).append(e)
    
    return per_layer


def _aggregate_ml_statistics(
    entries: List[Dict[str, Any]],
    per_layer: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Aggregate ML statistics from entries.
    
    Args:
        entries: List of timeline entries
        per_layer: Entries grouped by layer
        
    Returns:
        Statistics dictionary
    """
    all_techs = []
    all_patterns = []
    posts_with_ml = 0
    
    for e in entries:
        ml = e['ml_discovery']
        if ml['ml_classified']:
            posts_with_ml += 1
        
        all_techs.extend([t['name'] for t in ml.get('tech_stack', [])])
        all_patterns.extend([p['pattern'] for p in ml.get('patterns', [])])
    
    tech_counter = Counter(all_techs)
    pattern_counter = Counter(all_patterns)
    
    # Determine top items limit based on mode
    is_combined = any('source' in e for e in entries)
    tech_limit = TOP_TECH_COMBINED_LIMIT if is_combined else TOP_TECH_LIMIT
    
    return {
        'total_posts': len(entries),
        'technical_posts': sum(1 for e in entries if e.get('is_technical')),
        'posts_with_ml': posts_with_ml,
        'layers': {
            layer: len(items) for layer, items in per_layer.items()
        },
        'ml_discovery': {
            'method': 'clustering + ner + qa + semantic',
            'n_topics': len(per_layer),
            'total_tech_mentions': len(all_techs),
            'unique_technologies': len(tech_counter),
            'total_patterns': len(all_patterns),
            'unique_patterns': len(pattern_counter),
            'posts_with_solutions': sum(
                1 for e in entries if e['ml_discovery'].get('solutions')
            ),
            'posts_with_problem': sum(
                1 for e in entries if e['ml_discovery'].get('problem')
            ),
            'posts_with_approach': sum(
                1 for e in entries if e['ml_discovery'].get('approach')
            ),
        },
        'top_technologies': [
            {'name': k, 'count': v}
            for k, v in tech_counter.most_common(tech_limit)
        ],
        'top_patterns': [
            {'pattern': k, 'count': v}
            for k, v in pattern_counter.most_common(tech_limit)
        ],
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


def _save_source_markdown(
    file_path: Path,
    timeline: Dict[str, Any]
) -> None:
    """Save source timeline as Markdown.
    
    Args:
        file_path: Path to save Markdown file
        timeline: Timeline dictionary
        
    Returns:
        None
    """
    md_lines = _build_source_markdown_lines(timeline)
    file_path.write_text('\n'.join(md_lines), encoding='utf-8')


def _save_combined_markdown(
    file_path: Path,
    timeline: Dict[str, Any]
) -> None:
    """Save combined timeline as Markdown.
    
    Args:
        file_path: Path to save Markdown file
        timeline: Timeline dictionary
        
    Returns:
        None
    """
    md_lines = _build_combined_markdown_lines(timeline)
    file_path.write_text('\n'.join(md_lines), encoding='utf-8')


def _build_source_markdown_lines(timeline: Dict[str, Any]) -> List[str]:
    """Build Markdown lines for source timeline.
    
    Args:
        timeline: Timeline dictionary
        
    Returns:
        List of Markdown lines
    """
    stats = timeline['stats']
    ml = stats['ml_discovery']
    
    md_lines = [
        f"# {timeline['publication']} - Engineering Timeline\n",
        f"**Version**: {timeline['version']} (ML Discovery Edition)",
        f"**Exported**: {timeline['exported_at']}",
        f"**Total Posts**: {stats['total_posts']}",
        f"**Technical Posts**: {stats['technical_posts']}",
        f"**With ML Data**: {stats['posts_with_ml']}\n",
        "---\n",
        "## 📊 ML Discovery Statistics\n",
        f"- **Topics Discovered**: {ml['n_topics']}",
        f"- **Technologies Mentioned**: {ml['total_tech_mentions']} "
        f"({ml['unique_technologies']} unique)",
        f"- **Patterns Found**: {ml['total_patterns']} "
        f"({ml['unique_patterns']} unique)",
        f"- **Posts with Solutions**: {ml['posts_with_solutions']}",
        f"- **Posts with Problem**: {ml['posts_with_problem']}",
        f"- **Posts with Approach**: {ml['posts_with_approach']}\n",
        "---\n",
        "## 🔧 Top Technologies\n"
    ]
    
    for tech in stats['top_technologies'][:10]:
        md_lines.append(f"- **{tech['name']}**: {tech['count']} mentions")
    
    md_lines.extend([
        "\n---\n",
        "## 🏗️ Top Patterns\n"
    ])
    
    for pattern in stats['top_patterns'][:10]:
        md_lines.append(
            f"- **{pattern['pattern']}**: {pattern['count']} occurrences"
        )
    
    md_lines.extend([
        "\n---\n",
        "## 📚 Posts by Architecture Layer\n"
    ])
    
    _add_layer_sections(
        md_lines,
        timeline['per_layer'],
        MARKDOWN_PREVIEW_LIMIT
    )
    
    return md_lines


def _build_combined_markdown_lines(timeline: Dict[str, Any]) -> List[str]:
    """Build Markdown lines for combined timeline.
    
    Args:
        timeline: Timeline dictionary
        
    Returns:
        List of Markdown lines
    """
    stats = timeline['stats']
    ml = stats['ml_discovery']
    
    md_lines = [
        "# All Sources - Engineering Timeline\n",
        f"**Version**: {timeline['version']} (ML Discovery Edition)",
        f"**Type**: Combined ({timeline['total_sources']} sources)",
        f"**Exported**: {timeline['exported_at']}",
        f"**Total Posts**: {stats['total_posts']}",
        f"**Technical Posts**: {stats['technical_posts']}",
        f"**With ML Data**: {stats['posts_with_ml']}\n",
        "---\n",
        "## 📚 Sources Included\n"
    ]
    
    sorted_sources = sorted(
        timeline['sources'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    for src, info in sorted_sources:
        md_lines.append(
            f"- **{info['publication']}** ({src}): {info['count']} posts"
        )
    
    md_lines.extend([
        "\n---\n",
        "## 📊 ML Discovery Statistics\n",
        f"- **Topics Discovered**: {ml['n_topics']}",
        f"- **Technologies Mentioned**: {ml['total_tech_mentions']} "
        f"({ml['unique_technologies']} unique)",
        f"- **Patterns Found**: {ml['total_patterns']} "
        f"({ml['unique_patterns']} unique)",
        f"- **Posts with Solutions**: {ml['posts_with_solutions']}",
        f"- **Posts with Problem**: {ml['posts_with_problem']}",
        f"- **Posts with Approach**: {ml['posts_with_approach']}\n",
        "---\n",
        "## 🔧 Top Technologies (All Sources)\n"
    ])
    
    for tech in stats['top_technologies'][:20]:
        md_lines.append(f"- **{tech['name']}**: {tech['count']} mentions")
    
    md_lines.extend([
        "\n---\n",
        "## 🏗️ Top Patterns (All Sources)\n"
    ])
    
    for pattern in stats['top_patterns'][:20]:
        md_lines.append(
            f"- **{pattern['pattern']}**: {pattern['count']} occurrences"
        )
    
    md_lines.extend([
        "\n---\n",
        "## 📚 Posts by Architecture Layer\n"
    ])
    
    _add_layer_sections(
        md_lines,
        timeline['per_layer'],
        MARKDOWN_COMBINED_PREVIEW_LIMIT,
        is_combined=True
    )
    
    return md_lines


def _add_layer_sections(
    md_lines: List[str],
    per_layer: Dict[str, List[Dict[str, Any]]],
    preview_limit: int,
    is_combined: bool = False
) -> None:
    """Add layer sections to Markdown lines.
    
    Args:
        md_lines: List of Markdown lines to append to
        per_layer: Entries grouped by layer
        preview_limit: Maximum posts to show per layer
        is_combined: Whether this is combined mode (includes source)
        
    Returns:
        None (modifies md_lines in place)
    """
    sorted_layers = sorted(
        per_layer.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for layer, items in sorted_layers:
        md_lines.append(f"\n### {layer} ({len(items)} posts)\n")
        
        for item in items[:preview_limit]:
            date_str = item['date'] or 'unknown'
            title = item['title'][:70 if is_combined else 80]
            url = item['url'] or '#'
            
            if is_combined and 'source' in item:
                src = item['source']
                md_lines.append(
                    f"- **{date_str}** [{src}] — [{title}]({url})"
                )
            else:
                md_lines.append(f"- **{date_str}** — [{title}]({url})")
        
        if len(items) > preview_limit:
            remaining = len(items) - preview_limit
            md_lines.append(f"\n  _... and {remaining} more posts_\n")
