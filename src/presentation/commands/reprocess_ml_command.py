"""
Reprocess ML Command
Reprocess existing posts with new ML discovery approach
"""

import click
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table

from ...infrastructure.pipeline_db import PipelineDB
from ..helpers.ml_processor import MLProcessor
from ..helpers.progress_display import ProgressDisplay


@click.command('reprocess-ml')
@click.option('--source', '-s', help='Reprocess specific source only')
@click.option('--all', 'all_sources', is_flag=True, help='Reprocess ALL sources')
@click.option('--limit', '-l', type=int, help='Limit posts per source (for testing)')
@click.option('--force', is_flag=True, help='Force reprocess even if already has ML data')
def reprocess_ml_command(source, all_sources, limit, force):
    """
    🔄 Reprocess posts with NEW ML discovery approach
    
    Reprocesses existing posts FROM DATABASE that have content_markdown 
    with the new ML approach (NO hardcoded keywords):
    
    • Clustering → Layers
    • NER → Tech Stack
    • NER+ngrams → Patterns (dynamic)
    • Embeddings → Solutions (semantic)
    • Q&A → Problem
    • Q&A → Approach
    
    Examples:
    
    \b
    # Reprocess one source
    uv run python main.py reprocess-ml --source netflix
    
    \b
    # Test with limit
    uv run python main.py reprocess-ml --source netflix --limit 50
    
    \b
    # Reprocess ALL sources
    uv run python main.py reprocess-ml --all
    
    \b
    # Force reprocess everything (even if already has ML)
    uv run python main.py reprocess-ml --all --force
    """
    console = Console()
    db = PipelineDB()
    
    console.print()
    console.print("[bold cyan]🔄 ML Reprocessing - New Approach[/bold cyan]")
    console.print("[dim]Using: Clustering + NER + Q&A (NO hardcoded keywords!)[/dim]")
    console.print("[dim]Source: Database posts with content_markdown[/dim]")
    console.print()
    
    # Get available sources from database
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT source FROM posts WHERE content_markdown IS NOT NULL ORDER BY source")
        available_sources = [row['source'] for row in cursor.fetchall()]
    
    if not available_sources:
        console.print("[red]❌ No posts with content_markdown found in database[/red]")
        return
    
    console.print(f"[dim]Available sources in DB: {', '.join(available_sources)}[/dim]")
    console.print()
    
    # Determine which sources to process
    if all_sources:
        sources = available_sources
        console.print(f"[cyan]Processing ALL {len(sources)} sources from database[/cyan]")
    elif source:
        if source not in available_sources:
            console.print(f"[red]❌ Source '{source}' not found in database[/red]")
            console.print(f"[yellow]Available: {', '.join(available_sources)}[/yellow]")
            return
        sources = [source]
    else:
        console.print("[red]❌ Must specify --source or --all[/red]")
        console.print("[yellow]Examples:[/yellow]")
        console.print("  uv run python main.py reprocess-ml --source netflix")
        console.print("  uv run python main.py reprocess-ml --all")
        return
    
    console.print()
    
    # Initialize ML processor
    ml_processor = MLProcessor(console)
    ml_processor.load_models()
    
    total_reprocessed = 0
    total_failed = 0
    
    for idx, src in enumerate(sources, 1):
        console.print(f"[magenta]{'='*60}[/magenta]")
        console.print(f"[magenta][{idx}/{len(sources)}] {src}[/magenta]")
        console.print(f"[magenta]{'='*60}[/magenta]")
        console.print()
        
        # Get posts with content (ready for ML)
        with db._get_connection() as conn:
            cursor = conn.cursor()
            
            if force:
                # Reprocess all posts with content
                query = """
                    SELECT * FROM posts 
                    WHERE source = ? AND content_markdown IS NOT NULL
                    ORDER BY published_at DESC
                """
                params = [src]
            else:
                # Only posts without ML data
                query = """
                    SELECT * FROM posts 
                    WHERE source = ? 
                    AND content_markdown IS NOT NULL
                    AND (tech_stack IS NULL OR patterns IS NULL OR ml_classified = 0)
                    ORDER BY published_at DESC
                """
                params = [src]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            posts_to_process = [dict(row) for row in cursor.fetchall()]
        
        if not posts_to_process:
            console.print(f"[green]✅ No posts need ML reprocessing for {src}[/green]")
            console.print()
            continue
        
        console.print(f"[yellow]Found {len(posts_to_process)} posts to reprocess[/yellow]")
        console.print()
        
        # Prepare entries for ML
        entries_for_ml = []
        for post in posts_to_process:
            md = post.get('content_markdown', '')
            if not md or len(md) < 100:
                continue
            
            date = None
            if post.get('published_at'):
                try:
                    date = datetime.fromisoformat(str(post['published_at']).replace('Z', '+00:00')).date()
                except:
                    pass
            
            entries_for_ml.append({
                'id': post['id'],
                'title': post.get('title', 'Untitled'),
                'date': date.isoformat() if date else None,
                'content': md,
                'path': post.get('url', ''),
            })
        
        if not entries_for_ml:
            console.print(f"[yellow]No valid content to process for {src}[/yellow]")
            console.print()
            continue
        
        console.print(f"[cyan]🤖 Running ML discovery on {len(entries_for_ml)} posts...[/cyan]")
        console.print()
        
        try:
            # Run ML processing
            stats = ml_processor.process_posts(entries_for_ml)
            
            # Save to database with progress
            console.print("[cyan]💾 Saving ML data to database...[/cyan]")
            with ProgressDisplay.create_simple_progress(console) as progress:
                task = progress.add_task("[cyan]Saving to DB", total=len(entries_for_ml))
                saved = 0
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
                    saved += 1
                    progress.update(task, advance=1)
            
            console.print(f"[green]✅ Saved ML data for {saved} posts[/green]")
            total_reprocessed += saved
            
        except Exception as e:
            console.print(f"[red]❌ ML processing failed for {src}: {e}[/red]")
            import traceback
            traceback.print_exc()
            total_failed += 1
        
        console.print()
    
    # Final summary
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]🎉 Reprocessing Complete![/bold green]")
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
