#!/usr/bin/env python3
"""
Extract content from existing *_posts.json files.
This script reads post IDs from JSON files and fetches only the content,
creating markdown files without re-scraping metadata.

Usage:
    python scripts/extract_content_from_json.py
    python scripts/extract_content_from_json.py --source netflix
    python scripts/extract_content_from_json.py --limit 50
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infrastructure.http_transport import HttpTransport
from src.infrastructure import content_extractor
from src.infrastructure.persistence import persist_markdown_and_metadata
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()

SOURCES = [
    "netflix", "airbnb", "lyft", "wise", "tinder", "skyscanner",
    "kickstarter", "medium", "nytimes", "olx", "deezer", "pinterest"
]


def load_posts_from_json(json_file: Path) -> List[Dict]:
    """Load posts from JSON file"""
    try:
        with json_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'posts' in data:
                return data['posts']
            elif isinstance(data, dict) and 'entries' in data:
                return data['entries']
            else:
                return []
    except Exception as e:
        console.print(f"[red]Error loading {json_file}: {e}[/red]")
        return []


def extract_content_for_post(post_data: Dict, transport: HttpTransport, source: str) -> bool:
    """Extract and save content for a single post"""
    try:
        post_id = post_data.get('id')
        if not post_id:
            return False
        
        # Check if markdown already exists
        output_dir = ROOT / "outputs" / source
        md_file = output_dir / f"{post_id}_{post_data.get('title', 'untitled').replace(' ', '_')[:50]}.md"
        
        if md_file.exists():
            return True  # Skip if already exists
        
        # Get content HTML (either from JSON or fetch it)
        content_html = post_data.get('content_html')
        
        if not content_html or content_html == 'null' or len(str(content_html)) < 50:
            # Need to fetch content
            try:
                # Try to get from Medium API
                url = post_data.get('url', '')
                if url:
                    response = transport.get_post_content(url)
                    content_html = response.get('content', {}).get('bodyModel', {}).get('paragraphs', [])
                else:
                    return False
            except Exception:
                return False
        
        if not content_html:
            return False
        
        # Convert HTML to Markdown
        md, assets, code_blocks = content_extractor.html_to_markdown(content_html)
        
        # Classify technical level
        classification = content_extractor.classify_technical(content_html, code_blocks)
        
        # Persist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved = persist_markdown_and_metadata(
            post_data=post_data,
            md_content=md,
            assets=assets,
            classification=classification,
            output_dir=output_dir
        )
        
        return bool(saved)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to extract {post_data.get('id')}: {e}[/yellow]")
        return False


def process_source(source: str, limit: int = None, transport: HttpTransport = None):
    """Process a single source"""
    json_file = ROOT / "outputs" / f"{source}_posts.json"
    
    if not json_file.exists():
        console.print(f"[yellow]⊘ Skipping {source}: JSON file not found[/yellow]")
        return 0, 0
    
    posts = load_posts_from_json(json_file)
    
    if not posts:
        console.print(f"[yellow]⊘ Skipping {source}: No posts in JSON[/yellow]")
        return 0, 0
    
    if limit:
        posts = posts[:limit]
    
    console.print(f"\n[bold blue]Processing {source}: {len(posts)} posts[/bold blue]")
    
    if transport is None:
        transport = HttpTransport()
    
    success = 0
    failed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[cyan]Extracting content...", total=len(posts))
        
        for post in posts:
            if extract_content_for_post(post, transport, source):
                success += 1
            else:
                failed += 1
            
            progress.update(task, advance=1)
    
    console.print(f"[green]✓[/green] {source}: {success} extracted, {failed} failed")
    
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Extract content from existing JSON files")
    parser.add_argument("--source", help="Process single source (e.g., netflix)")
    parser.add_argument("--limit", type=int, help="Limit number of posts per source")
    args = parser.parse_args()
    
    console.print("\n[bold cyan]╔═══════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║                                                       ║[/bold cyan]")
    console.print("[bold cyan]║      Content Extraction from Existing JSON Files     ║[/bold cyan]")
    console.print("[bold cyan]║                                                       ║[/bold cyan]")
    console.print("[bold cyan]╚═══════════════════════════════════════════════════════╝[/bold cyan]\n")
    
    # Initialize HTTP transport once
    transport = HttpTransport()
    
    if args.source:
        sources = [args.source]
    else:
        sources = SOURCES
    
    total_success = 0
    total_failed = 0
    
    for source in sources:
        success, failed = process_source(source, args.limit, transport)
        total_success += success
        total_failed += failed
    
    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold green]✅ Extraction Complete![/bold green]\n")
    console.print(f"[bold]Statistics:[/bold]")
    console.print(f"  [green]✓[/green] Successfully extracted: {total_success}")
    console.print(f"  [red]✗[/red] Failed: {total_failed}")
    console.print(f"  [blue]→[/blue] Total processed: {total_success + total_failed}")
    console.print("\n[bold cyan]Next step:[/bold cyan]")
    console.print("  bash scripts/process_all_publications.sh")
    console.print()


if __name__ == "__main__":
    main()
