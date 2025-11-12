"""
ETL Command - Extract, Transform, Load with ML
Enrich posts + Generate timeline with ML discovery
"""

import json
import re
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import click
from rich.console import Console
from rich.table import Table

from ...infrastructure.pipeline_db import PipelineDB
from ...infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from ...infrastructure.config.source_manager import SourceConfigManager
from ...infrastructure import content_extractor
from ...domain.entities.publication import Post, PostId, Author
from ..helpers.ml_processor import MLProcessor
from ..helpers.progress_display import ProgressDisplay
from ..helpers.text_cleaner import clean_markdown


@click.command('etl')
@click.option('--source', '-s', required=True, help='Source to process (e.g., netflix)')
@click.option('--limit', '-l', type=int, help='Limit posts to process (for testing)')
@click.option('--slow-mode', is_flag=True, help='Ultra slow mode: 1 post per minute (avoid rate limiting)')
def etl_command(source, limit, slow_mode):
    """
    🚀 ETL Pipeline: Enrich + ML Discovery + Timeline
    
    Process posts already in database:
    1. Enrich with HTML and Markdown (if missing)
    2. ML discovery (tech stack, patterns, solutions, etc)
    3. Generate timeline JSON + Markdown
    
    Examples:
    
    \b
    # Process all Netflix posts
    uv run python main.py etl --source netflix
    
    \b
    # Test with 20 posts
    uv run python main.py etl --source netflix --limit 20
    
    \b
    # Slow mode (avoid rate limits)
    uv run python main.py etl --source netflix --slow-mode
    """
    console = Console()
    db = PipelineDB()
    
    if slow_mode:
        console.print()
        console.print("[yellow]⚠️  SLOW MODE enabled: 1 post per minute to avoid rate limiting[/yellow]")
        console.print("[dim]This will be VERY slow but safer for avoiding HTTP 429 errors[/dim]")
        console.print()
    
    console.print()
    console.print("[bold cyan]🚀 ETL Pipeline: Enrich + Timeline[/bold cyan]")
    console.print(f"[dim]Source: {source}[/dim]")
    console.print()
    
    # Step 1: Enrich posts
    _enrich_posts(console, db, source, limit, slow_mode)
    
    # Step 2: ML Discovery & Timeline
    _generate_timeline_with_ml(console, db, source)


def _enrich_posts(console: Console, db: PipelineDB, source: str, limit: int, slow_mode: bool):
    """Step 1: Enrich posts with HTML & Markdown"""
    console.print("[bold blue]Step 1/2: Enriching with HTML & Markdown...[/bold blue]")
    
    adapter = MediumApiAdapter()
    config_manager = SourceConfigManager()
    
    with db._get_connection() as conn:
        cursor = conn.cursor()
        
        # Count total posts
        cursor.execute("SELECT COUNT(*) FROM posts WHERE source = ?", [source])
        total_posts = cursor.fetchone()[0]
        
        # Get posts that need enrichment
        query = "SELECT * FROM posts WHERE source = ? AND (content_html IS NULL OR content_markdown IS NULL)"
        params = [source]
        if limit:
            query += " ORDER BY published_at DESC LIMIT ?"
            params.append(limit)
        cursor.execute(query, params)
        posts_to_enrich = [dict(row) for row in cursor.fetchall()]
    
    already_enriched = total_posts - len(posts_to_enrich)
    
    if not posts_to_enrich:
        console.print(f"[green]✅ All {total_posts} posts already enriched![/green]")
        console.print()
        return
    
    if already_enriched > 0:
        console.print(f"[dim]ℹ️  {already_enriched} posts already enriched, skipping...[/dim]")
    console.print(f"[yellow]Processing {len(posts_to_enrich)} posts...[/yellow]")
    console.print()
    
    enriched = 0
    failed = 0
    failure_reasons = {}
    
    for i, post_data in enumerate(posts_to_enrich, 1):
        # Show progress
        if i % 10 == 1 or i == len(posts_to_enrich):
            console.print(
                f"[cyan]Progress: {i}/{len(posts_to_enrich)} posts ({i*100//len(posts_to_enrich)}%)[/cyan]",
                end='\r'
            )
        
        # Add delay between posts
        if i > 1:
            delay = 60 if slow_mode else 3
            time.sleep(delay)
        
        try:
            # Get config
            try:
                sources = config_manager.load_sources()
                source_config = sources.get('sources', {}).get(source)
                if source_config:
                    from ...infrastructure.external.repositories import InMemoryPublicationRepository
                    repo = InMemoryPublicationRepository()
                    config = repo.create_generic_config(
                        source_config.get('publication', post_data['publication'])
                    )
                else:
                    raise ValueError("Config not found")
            except:
                from ...infrastructure.external.repositories import InMemoryPublicationRepository
                repo = InMemoryPublicationRepository()
                config = repo.create_generic_config(post_data['publication'])
            
            author_name = post_data.get('author') or 'Unknown'
            post = Post(
                id=PostId(post_data['id']),
                title=post_data['title'] or 'Untitled',
                slug=post_data.get('url', '').split('/')[-1] if post_data.get('url') else post_data['id'],
                author=Author(
                    id='unknown',
                    name=author_name,
                    username=author_name.lower().replace(' ', '_')
                ),
                published_at=datetime.now(),
                reading_time=post_data.get('reading_time', 0)
            )
            
            html = adapter.fetch_post_html(post, config)
            if not html:
                failed += 1
                reason = "No HTML returned from API"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                continue
            
            md, assets, code_blocks = content_extractor.html_to_markdown(html)
            classification = content_extractor.classify_technical(html, code_blocks)
            text_only = re.sub(r'[#*`\[\]()]+', ' ', md)
            text_only = re.sub(r'\s+', ' ', text_only).strip()
            
            post_data['content_html'] = html
            post_data['content_markdown'] = md
            # Keep full cleaned content for ML processing (no hard truncation)
            post_data['content_text'] = clean_markdown(text_only)
            post_data['has_markdown'] = True
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
            
        except Exception as e:
            failed += 1
            reason = str(e)[:50]
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    console.print()
    console.print(f"[green]✅ Enriched: {enriched}[/green]", end="")
    if failed > 0:
        console.print(f" [yellow]| Failed: {failed}[/yellow]")
        if failure_reasons:
            console.print("\n[yellow]Failure reasons:[/yellow]")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                console.print(f"  • {reason}: [red]{count}[/red]")
    else:
        console.print()
    
    console.print()


def _generate_timeline_with_ml(console: Console, db: PipelineDB, source: str):
    """Step 2: ML Discovery & Timeline Generation"""
    console.print("[bold blue]Step 2/2: ML Discovery & Timeline Generation...[/bold blue]")
    console.print("[dim]Using: Clustering + NER + Q&A (NO hardcoded keywords!)[/dim]")
    console.print()
    
    posts = db.get_posts_with_content(source=source)
    
    if not posts:
        console.print("[red]❌ No posts with content![/red]")
        return
    
    console.print(f"[yellow]Processing {len(posts)} posts with ML...[/yellow]")
    console.print()
    
    # Prepare data for ML
    entries_for_ml = []
    for post in posts:
        md = post.get('content_markdown', '')
        if not md:
            continue
        
        date = None
        if post.get('published_at'):
            try:
                date = datetime.fromisoformat(
                    str(post['published_at']).replace('Z', '+00:00')
                ).date()
            except:
                pass
        
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
    
    if not entries_for_ml:
        console.print("[red]❌ No posts with valid content![/red]")
        return
    
    # Run ML processing
    ml_processor = MLProcessor(console)
    ml_processor.load_models()
    
    try:
        stats = ml_processor.process_posts(entries_for_ml)
        
        # Save ML data to database
        console.print("[cyan]💾 Saving ML data to database...[/cyan]")
        with ProgressDisplay.create_simple_progress(console) as progress:
            task = progress.add_task("[cyan]Saving to DB", total=len(entries_for_ml))
            for entry in entries_for_ml:
                ml_data = {
                    'layers': entry.get('layers', []),
                    'tech_stack': entry.get('tech_stack', []),
                    'patterns': entry.get('patterns', []),
                    'solutions': entry.get('solutions', []),
                    'problem': entry.get('problem'),
                    'approach': entry.get('approach')
                }
                db.update_ml_discovery(entry['id'], ml_data)
                progress.update(task, advance=1)
        
        console.print(f"[green]✓ Saved ML data to database[/green]")
        console.print()
        
    except Exception as e:
        console.print(f"[red]❌ ML Discovery failed: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    # Generate timeline files
    _save_timeline(console, entries_for_ml, source, posts)


def _save_timeline(console: Console, entries: list, source: str, posts: list):
    """Save timeline to JSON and Markdown"""
    # Remove 'content' field (too large)
    clean_entries = []
    for e in entries:
        entry_copy = e.copy()
        entry_copy.pop('content', None)
        
        # Add snippet
        md = e.get('content', '')
        lines = md.split('\n')
        snippet = ''
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('---'):
                snippet = line[:200]
                break
        entry_copy['snippet'] = snippet
        
        clean_entries.append(entry_copy)
    
    clean_entries.sort(key=lambda e: (e['date'] is None, e['date'] or '9999-12-31'))
    
    # Group by layer
    per_layer = {}
    for e in clean_entries:
        for layer in e.get('layers', ['Uncategorized']):
            per_layer.setdefault(layer, []).append(e)
    
    # Count tech stack and patterns
    all_techs = []
    all_patterns = []
    for e in clean_entries:
        all_techs.extend([t['name'] for t in e.get('tech_stack', [])])
        all_patterns.extend([p['pattern'] for p in e.get('patterns', [])])
    
    tech_counter = Counter(all_techs)
    pattern_counter = Counter(all_patterns)
    
    timeline = {
        'count': len(clean_entries),
        'publication': posts[0]['publication'] if posts else source,
        'source': source,
        'posts': clean_entries,
        'per_layer': per_layer,
        'stats': {
            'total_posts': len(clean_entries),
            'technical_posts': sum(1 for e in clean_entries if e.get('is_technical')),
            'layers': {layer: len(items) for layer, items in per_layer.items()},
            'ml_discovery': {
                'method': 'clustering + ner + qa',
                'n_topics': len(per_layer),
                'total_tech_mentions': len(all_techs),
                'unique_technologies': len(tech_counter),
                'total_patterns': len(all_patterns),
                'unique_patterns': len(pattern_counter),
                'posts_with_solutions': sum(1 for e in clean_entries if e.get('solutions')),
            },
            'top_technologies': [{'name': k, 'count': v} for k, v in tech_counter.most_common(10)],
            'top_patterns': [{'pattern': k, 'count': v} for k, v in pattern_counter.most_common(10)],
        }
    }
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    json_file = output_dir / f"{source}_timeline.json"
    md_file = output_dir / f"{source}_timeline.md"
    
    # Save JSON
    json_file.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding='utf-8')
    
    # Save Markdown
    md_lines = [f"# {timeline['publication']} Timeline\n"]
    md_lines.append(f"**Total**: {timeline['stats']['total_posts']} posts")
    md_lines.append(f"**Technical**: {timeline['stats']['technical_posts']} posts\n")
    md_lines.append('\n## Architecture Layers\n')
    
    for layer, items in sorted(per_layer.items(), key=lambda x: len(x[1]), reverse=True):
        md_lines.append(f'\n### {layer} ({len(items)})\n')
        for item in items[:10]:
            md_lines.append(f"- **{item['date'] or 'unknown'}** — {item['title'][:80]}")
        if len(items) > 10:
            md_lines.append(f"  _... +{len(items) - 10} more_\n")
    
    md_file.write_text('\n'.join(md_lines), encoding='utf-8')
    
    console.print(f"[green]✅ {json_file.name}[/green]")
    console.print(f"[green]✅ {md_file.name}[/green]")
    console.print()
    
    # Summary
    _print_summary(console, timeline, source)


def _print_summary(console: Console, timeline: dict, source: str):
    """Print final summary table"""
    console.print("[bold green]🎉 Complete![/bold green]")
    console.print()
    
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total Posts", str(timeline['stats']['total_posts']))
    table.add_row("Technical Posts", str(timeline['stats']['technical_posts']))
    table.add_row("In Timeline", str(timeline['count']))
    
    # ML stats
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
    
    console.print("[bold]📊 Discovered Topics (Layers):[/bold]")
    for layer, count in sorted(timeline['stats']['layers'].items(), key=lambda x: x[1], reverse=True):
        console.print(f"   • {layer}: [green]{count}[/green]")
    console.print()
    
    # Show top technologies
    if timeline['stats'].get('top_technologies'):
        console.print("[bold]🔧 Top Technologies:[/bold]")
        for tech in timeline['stats']['top_technologies'][:5]:
            console.print(f"   • {tech['name']}: [green]{tech['count']}[/green]")
        console.print()
    
    # Show top patterns
    if timeline['stats'].get('top_patterns'):
        console.print("[bold]🏗️  Top Patterns:[/bold]")
        for pattern in timeline['stats']['top_patterns'][:5]:
            console.print(f"   • {pattern['pattern']}: [green]{pattern['count']}[/green]")
        console.print()

