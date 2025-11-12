"""
Export Timeline Command
Export complete timeline with ML discovery data
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

from ...infrastructure.pipeline_db import PipelineDB


@click.command('export-timeline')
@click.option('--source', '-s', help='Export specific source only')
@click.option('--all', 'all_sources', is_flag=True, help='Export ALL sources')
@click.option('--combined', '-c', is_flag=True, help='Combine all sources in one file (only with --all)')
@click.option('--output', '-o', type=click.Path(), help='Output directory (default: outputs/)')
@click.option('--format', '-f', type=click.Choice(['json', 'both']), default='both', help='Output format')
def export_timeline_command(source, all_sources, combined, output, format):
    """
    📊 Export Timeline with Complete ML Discovery Data
    
    ⚠️  IMPORTANT: Run 'reprocess-ml' FIRST to classify posts with ML!
    This command exports ONLY posts with ml_classified=1
    
    Export timeline JSON with all ML discoveries:
    - Layers (clustering)
    - Tech Stack (NER)
    - Patterns (NER + n-grams)
    - Solutions (semantic similarity)
    - Problem & Approach (Q&A)
    
    Examples:
    
    \b
    # First: Process with ML
    uv run python main.py reprocess-ml --all
    
    \b
    # Then: Export one source
    uv run python main.py export-timeline --source netflix
    
    \b
    # Export ALL sources (separate files)
    uv run python main.py export-timeline --all
    
    \b
    # Export ALL sources COMBINED in one file
    uv run python main.py export-timeline --all --combined
    
    \b
    # Custom output directory
    uv run python main.py export-timeline --all --output ./timelines
    
    \b
    # JSON only (no markdown)
    uv run python main.py export-timeline --all --format json
    """
    console = Console()
    db = PipelineDB()
    
    output_dir = Path(output) if output else Path("outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    console.print()
    console.print("[bold cyan]📊 Timeline Export - ML Discovery Edition[/bold cyan]")
    console.print("[dim]Format: Complete ML data (tech_stack, patterns, solutions, problem, approach)[/dim]")
    console.print()
    
    # Check ML classification status
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
        console.print(f"[red]⚠️  WARNING: 0/{total_posts} posts have ML classification![/red]")
        console.print("[yellow]Run 'reprocess-ml' first to classify posts with ML[/yellow]")
        console.print("[dim]Example: uv run python main.py reprocess-ml --all[/dim]")
        console.print()
        return
    
    console.print(f"[green]✓ Found {posts_with_ml}/{total_posts} posts with ML data ({posts_with_ml*100//total_posts}%)[/green]")
    console.print()
    
    # Get sources
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT source FROM posts WHERE content_markdown IS NOT NULL ORDER BY source")
        available_sources = [row['source'] for row in cursor.fetchall()]
    
    if not available_sources:
        console.print("[red]❌ No sources with content found in database[/red]")
        return
    
    if all_sources:
        sources = available_sources
        if combined:
            console.print(f"[cyan]Exporting ALL {len(sources)} sources COMBINED into one file[/cyan]")
        else:
            console.print(f"[cyan]Exporting ALL {len(sources)} sources (separate files)[/cyan]")
    elif source:
        if source not in available_sources:
            console.print(f"[red]❌ Source '{source}' not found[/red]")
            console.print(f"[yellow]Available: {', '.join(available_sources)}[/yellow]")
            return
        sources = [source]
    else:
        console.print("[red]❌ Must specify --source or --all[/red]")
        console.print("[yellow]Examples:[/yellow]")
        console.print("  uv run python main.py export-timeline --source netflix")
        console.print("  uv run python main.py export-timeline --all")
        return
    
    console.print()
    
    # Combined mode: one file with all sources
    if all_sources and combined:
        _export_combined_timeline(db, sources, output_dir, format, console)
        console.print()
        console.print(f"[bold green]✅ Exported combined timeline[/bold green]")
        console.print(f"[cyan]📁 Output: {output_dir.absolute()}/all_sources_timeline.json[/cyan]")
        console.print()
        return
    
    # Separate files mode
    exported_count = 0
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Exporting sources", total=len(sources))
        
        for src in sources:
            _export_source_timeline(db, src, output_dir, format, console)
            exported_count += 1
            progress.update(task, advance=1)
    
    console.print()
    console.print(f"[bold green]✅ Exported {exported_count} source(s)[/bold green]")
    console.print(f"[cyan]📁 Output directory: {output_dir.absolute()}[/cyan]")
    console.print()


def _export_combined_timeline(db: PipelineDB, sources: list, output_dir: Path, format: str, console: Console):
    """Export all sources combined in one file"""
    
    all_entries = []
    sources_stats = {}
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading sources", total=len(sources))
        
        for src in sources:
            # Get posts with ML data
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE source = ? 
                    AND content_markdown IS NOT NULL
                    AND ml_classified = 1
                    ORDER BY published_at DESC
                """, (src,))
                posts = [dict(row) for row in cursor.fetchall()]
            
            if not posts:
                progress.update(task, advance=1)
                continue
            
            # Build entries for this source
            source_entries = []
            for post in posts:
                md = post.get('content_markdown', '')
                if not md or len(md) < 100:
                    continue
                
                # Parse date
                date = None
                if post.get('published_at'):
                    try:
                        date = datetime.fromisoformat(
                            str(post['published_at']).replace('Z', '+00:00')
                        ).date()
                    except:
                        pass
                
                # Extract snippet
                lines = md.split('\n')
                snippet = ''
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('---'):
                        snippet = line[:200]
                        break
                
                # Parse ML data
                tech_stack = _parse_json_field(post.get('tech_stack'))
                patterns = _parse_json_field(post.get('patterns'))
                solutions = _parse_json_field(post.get('solutions'))
                layers = _parse_json_field(post.get('layers'))
                
                entry = {
                    'id': post['id'],
                    'source': src,  # Add source field!
                    'publication': post.get('publication', src),
                    'title': post.get('title', 'Untitled'),
                    'date': date.isoformat() if date else None,
                    'url': post.get('url'),
                    'author': post.get('author', 'Unknown'),
                    'reading_time': post.get('reading_time', 0),
                    'snippet': snippet,
                    'is_technical': post.get('is_technical', False),
                    'technical_score': post.get('technical_score', 0.0),
                    'code_blocks': post.get('code_blocks', 0),
                    'ml_discovery': {
                        'layers': layers or [],
                        'tech_stack': tech_stack or [],
                        'patterns': patterns or [],
                        'solutions': solutions or [],
                        'problem': post.get('problem'),
                        'approach': post.get('approach'),
                        'ml_classified': bool(post.get('ml_classified', 0)),
                    }
                }
                
                source_entries.append(entry)
            
            # Store stats for this source
            sources_stats[src] = {
                'count': len(source_entries),
                'publication': posts[0].get('publication', src) if posts else src
            }
            
            all_entries.extend(source_entries)
            progress.update(task, advance=1)
    
    if not all_entries:
        console.print("[yellow]⚠️  No posts with ML data found in any source[/yellow]")
        return
    
    # Sort all entries by date
    all_entries.sort(key=lambda e: (e['date'] is None, e['date'] or '9999-12-31'))
    
    # Aggregate statistics across ALL sources
    per_layer = {}
    all_techs = []
    all_patterns = []
    posts_with_ml = 0
    
    for e in all_entries:
        ml = e['ml_discovery']
        if ml['ml_classified']:
            posts_with_ml += 1
        
        layers = ml.get('layers', [])
        if not layers:
            layers = ['Uncategorized']
        for layer in layers:
            per_layer.setdefault(layer, []).append(e)
        
        all_techs.extend([t['name'] for t in ml.get('tech_stack', [])])
        all_patterns.extend([p['pattern'] for p in ml.get('patterns', [])])
    
    tech_counter = Counter(all_techs)
    pattern_counter = Counter(all_patterns)
    
    # Build combined timeline
    timeline = {
        'version': '2.0',
        'exported_at': datetime.now().isoformat(),
        'type': 'combined',
        'total_sources': len(sources_stats),
        'count': len(all_entries),
        
        'sources': sources_stats,
        'posts': all_entries,
        'per_layer': per_layer,
        
        'stats': {
            'total_posts': len(all_entries),
            'technical_posts': sum(1 for e in all_entries if e.get('is_technical')),
            'posts_with_ml': posts_with_ml,
            
            'layers': {
                layer: len(items) 
                for layer, items in per_layer.items()
            },
            
            'ml_discovery': {
                'method': 'clustering + ner + qa + semantic',
                'n_topics': len(per_layer),
                'total_tech_mentions': len(all_techs),
                'unique_technologies': len(tech_counter),
                'total_patterns': len(all_patterns),
                'unique_patterns': len(pattern_counter),
                'posts_with_solutions': sum(
                    1 for e in all_entries 
                    if e['ml_discovery'].get('solutions')
                ),
                'posts_with_problem': sum(
                    1 for e in all_entries 
                    if e['ml_discovery'].get('problem')
                ),
                'posts_with_approach': sum(
                    1 for e in all_entries 
                    if e['ml_discovery'].get('approach')
                ),
            },
            
            'top_technologies': [
                {'name': k, 'count': v} 
                for k, v in tech_counter.most_common(30)
            ],
            'top_patterns': [
                {'pattern': k, 'count': v} 
                for k, v in pattern_counter.most_common(30)
            ],
        }
    }
    
    # Save JSON
    if format in ['json', 'both']:
        json_file = output_dir / "all_sources_timeline.json"
        json_file.write_text(
            json.dumps(timeline, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    
    # Save Markdown
    if format == 'both':
        md_file = output_dir / "all_sources_timeline.md"
        _generate_combined_markdown(timeline, md_file)


def _export_source_timeline(db: PipelineDB, source: str, output_dir: Path, format: str, console: Console):
    """Export timeline for a single source"""
    
    # Get ONLY posts with ML classification (ml_classified=1)
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM posts 
            WHERE source = ? 
            AND content_markdown IS NOT NULL
            AND ml_classified = 1
            ORDER BY published_at DESC
        """, (source,))
        posts = [dict(row) for row in cursor.fetchall()]
    
    if not posts:
        console.print(f"[yellow]⚠️  {source}: No posts with ML data (run 'reprocess-ml' first)[/yellow]")
        return
    
    # Build timeline entries
    entries = []
    for post in posts:
        md = post.get('content_markdown', '')
        if not md or len(md) < 100:
            continue
        
        # Parse date
        date = None
        if post.get('published_at'):
            try:
                date = datetime.fromisoformat(
                    str(post['published_at']).replace('Z', '+00:00')
                ).date()
            except:
                pass
        
        # Extract snippet
        lines = md.split('\n')
        snippet = ''
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('---'):
                snippet = line[:200]
                break
        
        # Parse ML data from database (JSON strings)
        tech_stack = _parse_json_field(post.get('tech_stack'))
        patterns = _parse_json_field(post.get('patterns'))
        solutions = _parse_json_field(post.get('solutions'))
        layers = _parse_json_field(post.get('layers'))
        
        entry = {
            'id': post['id'],
            'title': post.get('title', 'Untitled'),
            'date': date.isoformat() if date else None,
            'url': post.get('url'),
            'author': post.get('author', 'Unknown'),
            'reading_time': post.get('reading_time', 0),
            'snippet': snippet,
            
            # Technical classification
            'is_technical': post.get('is_technical', False),
            'technical_score': post.get('technical_score', 0.0),
            'code_blocks': post.get('code_blocks', 0),
            
            # ML Discovery Data (NEW!)
            'ml_discovery': {
                'layers': layers or [],
                'tech_stack': tech_stack or [],
                'patterns': patterns or [],
                'solutions': solutions or [],
                'problem': post.get('problem'),
                'approach': post.get('approach'),
                'ml_classified': bool(post.get('ml_classified', 0)),
            }
        }
        
        entries.append(entry)
    
    if not entries:
        console.print(f"[yellow]⚠️  {source}: No valid entries[/yellow]")
        return
    
    # Sort by date
    entries.sort(key=lambda e: (e['date'] is None, e['date'] or '9999-12-31'))
    
    # Group by layers
    per_layer = {}
    for e in entries:
        layers = e['ml_discovery']['layers']
        if not layers:
            layers = ['Uncategorized']
        for layer in layers:
            per_layer.setdefault(layer, []).append(e)
    
    # Aggregate statistics
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
    
    # Build timeline structure
    timeline = {
        'version': '2.0',  # New version with ML data
        'exported_at': datetime.now().isoformat(),
        'source': source,
        'publication': posts[0].get('publication', source) if posts else source,
        'count': len(entries),
        
        'posts': entries,
        'per_layer': per_layer,
        
        'stats': {
            'total_posts': len(entries),
            'technical_posts': sum(1 for e in entries if e.get('is_technical')),
            'posts_with_ml': posts_with_ml,
            
            'layers': {
                layer: len(items) 
                for layer, items in per_layer.items()
            },
            
            'ml_discovery': {
                'method': 'clustering + ner + qa + semantic',
                'n_topics': len(per_layer),
                'total_tech_mentions': len(all_techs),
                'unique_technologies': len(tech_counter),
                'total_patterns': len(all_patterns),
                'unique_patterns': len(pattern_counter),
                'posts_with_solutions': sum(
                    1 for e in entries 
                    if e['ml_discovery'].get('solutions')
                ),
                'posts_with_problem': sum(
                    1 for e in entries 
                    if e['ml_discovery'].get('problem')
                ),
                'posts_with_approach': sum(
                    1 for e in entries 
                    if e['ml_discovery'].get('approach')
                ),
            },
            
            'top_technologies': [
                {'name': k, 'count': v} 
                for k, v in tech_counter.most_common(20)
            ],
            'top_patterns': [
                {'pattern': k, 'count': v} 
                for k, v in pattern_counter.most_common(20)
            ],
        }
    }
    
    # Save JSON
    if format in ['json', 'both']:
        json_file = output_dir / f"{source}_timeline.json"
        json_file.write_text(
            json.dumps(timeline, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    
    # Save Markdown
    if format == 'both':
        md_file = output_dir / f"{source}_timeline.md"
        _generate_markdown(timeline, md_file)


def _generate_markdown(timeline: dict, md_file: Path):
    """Generate Markdown summary"""
    md_lines = [
        f"# {timeline['publication']} - Engineering Timeline\n",
        f"**Version**: {timeline['version']} (ML Discovery Edition)",
        f"**Exported**: {timeline['exported_at']}",
        f"**Total Posts**: {timeline['stats']['total_posts']}",
        f"**Technical Posts**: {timeline['stats']['technical_posts']}",
        f"**With ML Data**: {timeline['stats']['posts_with_ml']}\n",
        "---\n",
        "## 📊 ML Discovery Statistics\n",
        f"- **Topics Discovered**: {timeline['stats']['ml_discovery']['n_topics']}",
        f"- **Technologies Mentioned**: {timeline['stats']['ml_discovery']['total_tech_mentions']} ({timeline['stats']['ml_discovery']['unique_technologies']} unique)",
        f"- **Patterns Found**: {timeline['stats']['ml_discovery']['total_patterns']} ({timeline['stats']['ml_discovery']['unique_patterns']} unique)",
        f"- **Posts with Solutions**: {timeline['stats']['ml_discovery']['posts_with_solutions']}",
        f"- **Posts with Problem**: {timeline['stats']['ml_discovery']['posts_with_problem']}",
        f"- **Posts with Approach**: {timeline['stats']['ml_discovery']['posts_with_approach']}\n",
        "---\n",
        "## 🔧 Top Technologies\n"
    ]
    
    for tech in timeline['stats']['top_technologies'][:10]:
        md_lines.append(f"- **{tech['name']}**: {tech['count']} mentions")
    
    md_lines.append("\n---\n")
    md_lines.append("## 🏗️ Top Patterns\n")
    
    for pattern in timeline['stats']['top_patterns'][:10]:
        md_lines.append(f"- **{pattern['pattern']}**: {pattern['count']} occurrences")
    
    md_lines.append("\n---\n")
    md_lines.append("## 📚 Posts by Architecture Layer\n")
    
    for layer, items in sorted(
        timeline['per_layer'].items(),
        key=lambda x: len(x[1]),
        reverse=True
    ):
        md_lines.append(f"\n### {layer} ({len(items)} posts)\n")
        for item in items[:15]:  # Show first 15
            date_str = item['date'] or 'unknown'
            md_lines.append(f"- **{date_str}** — [{item['title'][:80]}]({item['url'] or '#'})")
        
        if len(items) > 15:
            md_lines.append(f"\n  _... and {len(items) - 15} more posts_\n")
    
    md_file.write_text('\n'.join(md_lines), encoding='utf-8')


def _generate_combined_markdown(timeline: dict, md_file: Path):
    """Generate Markdown summary for combined timeline"""
    md_lines = [
        f"# All Sources - Engineering Timeline\n",
        f"**Version**: {timeline['version']} (ML Discovery Edition)",
        f"**Type**: Combined ({timeline['total_sources']} sources)",
        f"**Exported**: {timeline['exported_at']}",
        f"**Total Posts**: {timeline['stats']['total_posts']}",
        f"**Technical Posts**: {timeline['stats']['technical_posts']}",
        f"**With ML Data**: {timeline['stats']['posts_with_ml']}\n",
        "---\n",
        "## 📚 Sources Included\n"
    ]
    
    for src, info in sorted(timeline['sources'].items(), key=lambda x: x[1]['count'], reverse=True):
        md_lines.append(f"- **{info['publication']}** ({src}): {info['count']} posts")
    
    md_lines.extend([
        "\n---\n",
        "## 📊 ML Discovery Statistics\n",
        f"- **Topics Discovered**: {timeline['stats']['ml_discovery']['n_topics']}",
        f"- **Technologies Mentioned**: {timeline['stats']['ml_discovery']['total_tech_mentions']} ({timeline['stats']['ml_discovery']['unique_technologies']} unique)",
        f"- **Patterns Found**: {timeline['stats']['ml_discovery']['total_patterns']} ({timeline['stats']['ml_discovery']['unique_patterns']} unique)",
        f"- **Posts with Solutions**: {timeline['stats']['ml_discovery']['posts_with_solutions']}",
        f"- **Posts with Problem**: {timeline['stats']['ml_discovery']['posts_with_problem']}",
        f"- **Posts with Approach**: {timeline['stats']['ml_discovery']['posts_with_approach']}\n",
        "---\n",
        "## 🔧 Top Technologies (All Sources)\n"
    ])
    
    for tech in timeline['stats']['top_technologies'][:20]:
        md_lines.append(f"- **{tech['name']}**: {tech['count']} mentions")
    
    md_lines.append("\n---\n")
    md_lines.append("## 🏗️ Top Patterns (All Sources)\n")
    
    for pattern in timeline['stats']['top_patterns'][:20]:
        md_lines.append(f"- **{pattern['pattern']}**: {pattern['count']} occurrences")
    
    md_lines.append("\n---\n")
    md_lines.append("## 📚 Posts by Architecture Layer\n")
    
    for layer, items in sorted(
        timeline['per_layer'].items(),
        key=lambda x: len(x[1]),
        reverse=True
    ):
        md_lines.append(f"\n### {layer} ({len(items)} posts)\n")
        for item in items[:20]:  # Show first 20
            date_str = item['date'] or 'unknown'
            src = item['source']
            md_lines.append(f"- **{date_str}** [{src}] — [{item['title'][:70]}]({item['url'] or '#'})")
        
        if len(items) > 20:
            md_lines.append(f"\n  _... and {len(items) - 20} more posts_\n")
    
    md_file.write_text('\n'.join(md_lines), encoding='utf-8')


def _parse_json_field(field):
    """Parse JSON field from database (can be string or already parsed)"""
    if field is None:
        return None
    if isinstance(field, str):
        try:
            return json.loads(field)
        except:
            return None
    return field
