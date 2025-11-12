"""
Full Command - Complete pipeline
Scraping → Database → ETL → Timeline
"""

import subprocess
import click
from rich.console import Console
from rich.table import Table

from ...infrastructure.pipeline_db import PipelineDB
from ...infrastructure.config.source_manager import SourceConfigManager
from ...infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from ...infrastructure.external.repositories import InMemoryPublicationRepository, MediumSessionRepository
from ...domain.services.publication_service import PostDiscoveryService, PublicationConfigService
from ...application.use_cases.scrape_posts import ScrapePostsUseCase, ScrapePostsRequest


@click.command('full')
@click.option('--source', '-s', help='Process specific source (default: ALL from YAML)')
@click.option('--limit', '-l', type=int, help='Limit posts per source')
def full_command(source, limit):
    """
    🚀 FULL Pipeline: Collect → Enrich → ML → Timeline
    
    Complete workflow:
    1. Scrape posts from Medium API
    2. Save to database
    3. Enrich content (HTML → Markdown)
    4. ML discovery (tech stack, patterns, etc)
    5. Generate timeline
    
    Examples:
    
    \b
    # Process ALL sources from YAML
    uv run python main.py full
    
    \b
    # Process one source
    uv run python main.py full --source netflix
    
    \b
    # Limit posts
    uv run python main.py full --source netflix --limit 50
    """
    console = Console()
    db = PipelineDB()
    config_manager = SourceConfigManager()
    
    # Get sources
    if source:
        sources = [source]
    else:
        try:
            sources_dict = config_manager.list_sources()
            sources = list(sources_dict.keys())
            if not sources:
                console.print("[red]❌ No sources found in YAML[/red]")
                return
        except Exception as e:
            console.print(f"[red]❌ Failed to load sources from YAML: {e}[/red]")
            return
    
    console.print(f"\n[bold cyan]🚀 FULL Pipeline: {len(sources)} source(s)[/bold cyan]\n")
    
    stats = {'collected': 0, 'enriched': 0, 'timelines': 0, 'failed': []}
    
    for idx, src in enumerate(sources, 1):
        console.print(f"[magenta]{'='*50}[/magenta]")
        console.print(f"[magenta][{idx}/{len(sources)}] {src}[/magenta]")
        console.print(f"[magenta]{'='*50}[/magenta]\n")
        
        try:
            # Phase 1: Collect posts from Medium API
            with console.status("[blue]📥 Collecting...[/blue]", spinner="dots"):
                post_repo = MediumApiAdapter()
                pub_repo = InMemoryPublicationRepository()
                sess_repo = MediumSessionRepository()
                
                svc = PostDiscoveryService(post_repo)
                cfg_svc = PublicationConfigService(pub_repo)
                use_case = ScrapePostsUseCase(svc, cfg_svc, sess_repo)
                
                req = ScrapePostsRequest(
                    publication_name=src,
                    limit=limit,
                    auto_discover=True,
                    skip_session=True,
                    mode='metadata'
                )
                resp = use_case.execute(req)
            
            posts_collected = 0
            if resp.success and len(resp.posts) > 0:
                # Phase 2: Save to DB
                console.print(f"[dim]💾 Saving {len(resp.posts)} posts to database...[/dim]")
                new_posts = 0
                updated_posts = 0
                
                for post in resp.posts:
                    try:
                        # Check if post already exists
                        existing = db.post_exists(post.id.value)
                        
                        # If exists and already enriched, skip update (preserve enrichment)
                        if existing:
                            existing_post = next(
                                (p for p in db.get_posts_by_source(src) if p['id'] == post.id.value),
                                None
                            )
                            if existing_post and existing_post.get('content_markdown'):
                                # Already enriched, skip to preserve content
                                updated_posts += 1
                                continue
                        
                        db.add_or_update_post({
                            'id': post.id.value,
                            'source': src,
                            'publication': resp.publication_config.name if resp.publication_config else src,
                            'title': post.title,
                            'author': post.author.name if post.author else None,
                            'url': getattr(post, 'url', None),
                            'published_at': str(post.published_at) if hasattr(post, 'published_at') else None,
                            'reading_time': getattr(post, 'reading_time', 0),
                            'claps': getattr(post, 'claps', None),
                            'tags': getattr(post, 'tags', []),
                            'collection_mode': 'metadata',
                            'has_markdown': False
                        })
                        
                        if existing:
                            updated_posts += 1
                        else:
                            new_posts += 1
                    except:
                        pass
                
                console.print(f"[green]✅ {new_posts} new, {updated_posts} updated ({len(resp.posts)} total)[/green]")
                posts_collected = len(resp.posts)
                stats['collected'] += new_posts
            else:
                # Collection failed, check if we have existing posts
                existing_posts = db.get_posts_by_source(src)
                    
                if existing_posts:
                    console.print(f"[yellow]⚠️  No new posts collected, but found {len(existing_posts)} existing[/yellow]")
                else:
                    console.print(f"[red]❌ No posts (new or existing)[/red]\n")
                    stats['failed'].append(src)
                    continue
            
            # Phase 3: Enrich - call etl command
            console.print("[blue]🔄 Enriching...[/blue]")
            cmd = ["uv", "run", "python", "main.py", "etl", "--source", src]
            if limit:
                cmd.extend(["--limit", str(limit)])
            
            result = subprocess.run(
                cmd,
                capture_output=False,  # Show live output with Rich progress bars
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                console.print(f"[green]✅ Enriched & Timeline created[/green]")
                stats['enriched'] += 1
                stats['timelines'] += 1
            else:
                console.print(f"[yellow]⚠️  Enrich partially failed[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]\n")
            import traceback
            traceback.print_exc()
            stats['failed'].append(src)
        
        console.print()
    
    # Final summary
    console.print(f"[bold green]{'='*50}[/bold green]")
    console.print(f"[bold green]🎉 DONE![/bold green]")
    console.print(f"[bold green]{'='*50}[/bold green]\n")
    
    table = Table(show_header=True, border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow", justify="right")
    
    table.add_row("Sources Processed", str(len(sources) - len(stats['failed'])))
    table.add_row("Posts Collected", str(stats['collected']))
    table.add_row("Sources Enriched", str(stats['enriched']))
    table.add_row("Timelines Created", str(stats['timelines']))
    if stats['failed']:
        table.add_row("Failed", str(len(stats['failed'])))
    
    console.print(table)
    
    if stats['failed']:
        console.print(f"\n[yellow]⚠️  Failed: {', '.join(stats['failed'])}[/yellow]")
    
    console.print(f"\n[cyan]✨ Check outputs/ folder for timelines![/cyan]\n")

