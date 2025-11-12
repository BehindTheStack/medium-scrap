"""
CLI Presentation Layer - User Interface
Following MVC pattern and Dependency Injection
"""

import json
import click
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn, TimeRemainingColumn
import time

from ..application.use_cases.scrape_posts import (
    ScrapePostsUseCase, ScrapePostsRequest, ScrapePostsResponse
)
from ..domain.entities.publication import Post
from ..infrastructure.config.source_manager import SourceConfigManager
from ..infrastructure import content_extractor
from ..infrastructure.persistence import persist_markdown_and_metadata


class AdvancedProgressLoader:
    """
    Advanced progress loader with detailed feedback
    Shows collection phases and progress
    """
    
    def __init__(self, console: Console):
        self.console = console
        self.phases = [
            "🔍 Resolving publication configuration...",
            "🤖 Analyzing publication type...", 
            "📡 Connecting to Medium API...",
            "🔎 Discovering post IDs...",
            "📝 Collecting post details...",
            "✨ Processing collected data..."
        ]
        self.current_phase = 0
        self.posts_found = 0
        self.is_running = False
        
    def start_collection(self, publication_name: str, limit: int = None):
        """Start the collection process with visual feedback"""
        self.is_running = True
        
        with Progress(
            SpinnerColumn(spinner_style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False
        ) as progress:
            
            # Main progress task
            main_task = progress.add_task(
                f"🚀 Collecting from: {publication_name}",
                total=100
            )
            
            # Phase task
            phase_task = progress.add_task(
                self.phases[0],
                total=len(self.phases)
            )
            
            return progress, main_task, phase_task
    
    def update_phase(self, progress, phase_task, phase_index: int):
        """Update the current phase"""
        if phase_index < len(self.phases):
            progress.update(
                phase_task,
                completed=phase_index,
                description=self.phases[phase_index]
            )
    
    def update_posts_count(self, progress, main_task, posts_count: int):
        """Update posts count and overall progress"""
        self.posts_found = posts_count
        progress.update(
            main_task,
            description=f"🚀 Collected {posts_count} posts"
        )


class PostFormatter:
    """
    Formatter for different output formats
    Following Strategy Pattern
    """
    
    def __init__(self):
        self.console = Console()
    
    def format_as_table(self, posts: List[Post], publication_name: str = "Medium") -> None:
        """Format posts as a rich table"""
        table = Table(title=f"{publication_name} Posts")
        
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Título", style="magenta", max_width=50)
        table.add_column("Autor", style="green")
        table.add_column("Tempo Leitura", justify="right")
        table.add_column("Data Publicação", style="blue")
        
        for post in posts:
            reading_time = f"{post.reading_time:.0f} min" if post.reading_time else "N/A"
            pub_date = post.published_at.strftime('%Y-%m-%d') if post.published_at else "N/A"
            
            table.add_row(
                post.id.value[:12] + "...",
                post.title,
                post.author.name,
                reading_time,
                pub_date
            )
        
        self.console.print(table)
    
    def format_as_json(self, posts: List[Post]) -> str:
        """Format posts as JSON"""
        posts_dict = [post.to_dict() for post in posts]
        return json.dumps(posts_dict, indent=2, ensure_ascii=False)
    
    def format_as_ids(self, posts: List[Post]) -> str:
        """Format posts as simple ID list"""
        return '\n'.join([post.id.value for post in posts])


class CLIController:
    """
    Main CLI Controller
    Orchestrates use cases and presentation
    """
    
    def __init__(self, scrape_posts_use_case: ScrapePostsUseCase):
        self._scrape_posts_use_case = scrape_posts_use_case
        self._formatter = PostFormatter()
        self.console = Console()
        self._config_manager = SourceConfigManager()
        self._progress_loader = AdvancedProgressLoader(self.console)
    
    def list_sources(self) -> None:
        """List all available configured sources"""
        sources = self._config_manager.list_sources()
        bulk_collections = self._config_manager.list_bulk_collections()
        
        # Sources table
        table = Table(title="📚 Available Medium Sources")
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Name", style="magenta")
        table.add_column("Description", style="dim")
        
        for key, source in sources.items():
            table.add_row(key, source.type, source.name, source.description)
        
        self.console.print(table)
        self.console.print()
        
        # Bulk collections table
        if bulk_collections:
            bulk_table = Table(title="📦 Bulk Collections")
            bulk_table.add_column("Key", style="cyan", no_wrap=True)  
            bulk_table.add_column("Description", style="magenta")
            bulk_table.add_column("Sources", style="dim")
            
            for key, bulk in bulk_collections.items():
                sources_list = ", ".join(bulk.sources)
                bulk_table.add_row(key, bulk.description, sources_list)
            
            self.console.print(bulk_table)
    
    def scrape_from_config(
        self,
        source_key: str,
        limit: Optional[int] = None,
        format_type: str = "table", 
        output_file: Optional[str] = None,
        all_posts: bool = False,
        mode: str = 'metadata'
    ) -> None:
        """Scrape posts using a configured source"""
        try:
            source_config = self._config_manager.get_source(source_key)
            defaults = self._config_manager.get_defaults()
            
            # Apply defaults if not specified
            if limit is None and not all_posts:
                limit = defaults.get('limit', 50)
            
            # Determine output file if not specified
            if output_file is None and format_type == 'json':
                output_dir = Path(defaults.get('output_dir', 'outputs'))
                output_dir.mkdir(exist_ok=True)
                output_file = str(output_dir / f"{source_key}_posts.json")
            
            # Show source info with loading animation
            with Progress(
                SpinnerColumn(style="green"),
                TextColumn("[green]{task.description}"),
                console=self.console
            ) as config_progress:
                task = config_progress.add_task("📖 Loading source configuration...", total=None)
                time.sleep(0.2)  # Brief animation
                config_progress.update(task, description=f"✅ Loaded: {source_key}")
                time.sleep(0.2)
            
            self.console.print(f"[bold green]📖 Source:[/bold green] {source_key}")
            self.console.print(f"[dim]{source_config.description}[/dim]")
            self.console.print()
            
            # Execute scraping
            if source_config.custom_domain:
                # For custom domains from YAML, call scrape_posts with the domain directly
                self.scrape_posts(
                    publication=source_config.name,  # Use the full domain name
                    limit=None if all_posts else limit,
                    format_type=format_type,
                    custom_ids=None,
                    auto_discover=source_config.auto_discover,
                    skip_session=defaults.get('skip_session', True),
                    output_file=output_file,
                    mode=mode,
                    source_key=source_key  # Pass source key for database
                )
            else:
                # Use normal flow for medium-hosted publications
                self.scrape_posts(
                    publication=source_config.get_publication_name(),
                    limit=None if all_posts else limit,
                    format_type=format_type,
                    custom_ids=None,
                    auto_discover=source_config.auto_discover,
                    skip_session=defaults.get('skip_session', True),
                    output_file=output_file,
                    mode=mode,
                    source_key=source_key  # Pass source key for database
                )
            
        except KeyError as e:
            self.console.print(f"[red]❌ {e}[/red]")
            self.console.print("[yellow]💡 Use --list-sources to see available sources[/yellow]")
    
    def scrape_bulk_collection(
        self,
        bulk_key: str,
        limit: Optional[int] = None,
        format_type: str = "json",
        mode: str = 'metadata'
    ) -> None:
        """Scrape posts from multiple sources in a bulk collection"""
        try:
            bulk_config = self._config_manager.get_bulk_config(bulk_key)
            defaults = self._config_manager.get_defaults()
            
            if limit is None:
                limit = defaults.get('limit', 50)
            
            self.console.print(f"[bold blue]📦 Bulk Collection:[/bold blue] {bulk_key}")
            self.console.print(f"[dim]{bulk_config.description}[/dim]")
            self.console.print(f"[dim]Sources: {len(bulk_config.sources)}[/dim]")
            self.console.print()
            
            output_dir = Path(defaults.get('output_dir', 'outputs'))
            output_dir.mkdir(exist_ok=True)
            
            # Progress for bulk collection
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                console=self.console
            ) as bulk_progress:
                
                bulk_task = bulk_progress.add_task(
                    f"📦 Processing bulk collection: {bulk_key}",
                    total=len(bulk_config.sources)
                )
                
                for i, source_key in enumerate(bulk_config.sources):
                    bulk_progress.update(
                        bulk_task,
                        description=f"📥 Collecting from: {source_key}",
                        completed=i
                    )
                    
                    self.scrape_from_config(
                        source_key=source_key,
                        limit=limit,
                        format_type=format_type,
                        output_file=str(output_dir / f"{source_key}_posts.json"),
                        mode=mode
                    )
                    
                    bulk_progress.update(bulk_task, completed=i + 1)
                    
                bulk_progress.update(
                    bulk_task,
                    description=f"✅ Bulk collection completed: {bulk_key}"
                )
                
        except KeyError as e:
            self.console.print(f"[red]❌ {e}[/red]")
            self.console.print("[yellow]💡 Use --list-sources to see available bulk collections[/yellow]")
    
    def scrape_posts(
        self,
        publication: str,
        limit: Optional[int] = None,
        format_type: str = "table",
        custom_ids: Optional[str] = None,
        auto_discover: bool = False,
        skip_session: bool = False,
        output_file: Optional[str] = None,
        mode: str = 'metadata',
        source_key: Optional[str] = None
    ) -> None:
        """Main command handler for post scraping"""
        
        # Display header
        self.console.print(f"[bold blue]🚀 Universal Medium Scraper[/bold blue]")
        self.console.print(f"[dim]Publication: {publication}[/dim]")
        self.console.print()
        
        # Parse custom IDs if provided
        custom_post_ids = None
        if custom_ids:
            custom_post_ids = [id.strip() for id in custom_ids.split(',')]
            self.console.print(f"[yellow]Using custom post IDs: {len(custom_post_ids)} IDs[/yellow]")
        
        # Show mode
        if auto_discover:
            self.console.print("🤖 [bold green]AUTO-DISCOVERY MODE:[/bold green] Production ready")
        else:
            self.console.print("🔍 [bold blue]HYBRID MODE:[/bold blue] Auto-discovery + fallback")

        # Print selected preset/mode (for future behavior mapping)
        if mode:
            self.console.print(f"⚙️ [bold]Mode:[/bold] {mode}")
        
        # Create request
        request = ScrapePostsRequest(
            publication_name=publication,
            limit=limit,
            custom_post_ids=custom_post_ids,
            auto_discover=auto_discover,
            skip_session=skip_session,
            mode=mode
        )
        
        # Execute use case with progress indicator
        response = self._execute_with_progress(request, skip_session)
        
        # Handle response
        if response.success:
            self._handle_successful_response(response, format_type, output_file, mode, source_key)
        else:
            self._handle_failed_response(response)
    
    def _execute_with_progress(self, request: ScrapePostsRequest, skip_session: bool) -> ScrapePostsResponse:
        """Execute use case with enhanced progress indication"""
        # Simplified progress: show an indeterminate spinner and clear, concise stage messages.
        # Avoid simulated sleeps and fake percentage updates. The spinner remains visible while the
        # use case runs and we update the description for the main milestones.
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False
        ) as progress:
            mode_text = "Auto-Discovery" if skip_session else "Session"
            task_id = progress.add_task(f"🚀 {mode_text}: {request.publication_name}", total=None)

            # Discovery/status task: updated live by progress callback
            discovery_task = progress.add_task("Discovery", total=None)

            # Indicate start
            progress.update(task_id, description=f"📡 Executing scraping: {request.publication_name} ...")

            # Live progress callback that updates the discovery task in real-time
            def _progress_callback(event: dict):
                try:
                    phase = event.get('phase')
                    if phase == 'discovered_ids':
                        count = event.get('count', 0)
                        progress.update(discovery_task, description=f"🔎 Discovered IDs: {count}")
                    elif phase == 'fetched_posts':
                        count = event.get('count', 0)
                        progress.update(discovery_task, description=f"📥 Fetched posts: {count}")
                    elif phase == 'enriched_post':
                        pid = event.get('post_id')
                        progress.update(discovery_task, description=f"🧩 Enriched: {pid}")
                except Exception:
                    # swallow callback errors to avoid breaking progress UI
                    pass

            # Execute the actual use case (may take time; spinner will show activity)
            try:
                response = self._scrape_posts_use_case.execute(request, progress_callback=_progress_callback)
            except TypeError:
                # Backwards compatibility: some tests or injected use-cases may not accept
                # the progress_callback kwarg. Fall back to calling without it.
                response = self._scrape_posts_use_case.execute(request)

            # Update to finalizing
            progress.update(task_id, description=f"✨ Finalizing — {response.total_posts_found} posts collected")

            # After discovery finishes, show a brief discovery summary (aggregate events)
            # For a concise summary we can reuse progress_task description; no separate collection here
            # small pause so users notice the final message before the console continues
            time.sleep(0.3)

        return response
    
    def _handle_successful_response(
        self,
        response: ScrapePostsResponse,
        format_type: str,
        output_file: Optional[str],
        mode: str = 'metadata',
        source_key: Optional[str] = None
    ) -> None:
        """Handle successful scraping response"""
        self.console.print(f"✅ {response.total_posts_found} posts collected!")
        self.console.print(f"[dim]Discovery method: {response.discovery_method}[/dim]")
        self.console.print()
        
        # ALWAYS save posts to database (metadata at minimum)
        from ..infrastructure.pipeline_db import PipelineDB
        db = PipelineDB()
        
        if not source_key:
            source_key = response.publication_config.name if response.publication_config else 'unknown'
        
        publication_name = response.publication_config.name if response.publication_config else 'unknown'
        
        # Save all posts to database (will be enriched with HTML/markdown later if mode is full/technical)
        for post in response.posts:
            post_url = f"https://medium.com/@{publication_name}/{post.slug}-{post.id.value}" if hasattr(post, 'slug') else None
            
            post_data = {
                'id': post.id.value,
                'source': source_key,
                'publication': publication_name,
                'title': post.title,
                'author': post.author.name if hasattr(post, 'author') and post.author else None,
                'url': post_url,
                'published_at': str(post.published_at) if hasattr(post, 'published_at') else None,
                'reading_time': post.reading_time if hasattr(post, 'reading_time') else None,
                'claps': getattr(post, 'claps', None),
                'tags': getattr(post, 'tags', []),
                'collection_mode': mode,
                'has_markdown': False,  # Will be updated if HTML is processed
                'has_json': False,
                'markdown_path': None,
                'json_path': None
            }
            
            db.add_or_update_post(post_data)
        
        self.console.print(f"[dim]💾 Saved {len(response.posts)} posts to database (metadata)[/dim]")
        self.console.print()
        
        # Format output
        if format_type == "table":
            self._formatter.format_as_table(response.posts, response.publication_config.name)
        elif format_type == "json":
            json_output = self._formatter.format_as_json(response.posts)
            if output_file:
                self._save_to_file(json_output, output_file)
                self.console.print(f"[green]✅ Saved to {output_file}[/green]")
            else:
                self.console.print(json_output)
        elif format_type == "ids":
            ids_output = self._formatter.format_as_ids(response.posts)
            if output_file:
                self._save_to_file(ids_output, output_file)
                self.console.print(f"[green]✅ Saved to {output_file}[/green]")
            else:
                self.console.print(ids_output)

        # If the user requested full content or technical mode, persist per-post artifacts
        if mode in ("full", "technical"):
            defaults = self._config_manager.get_defaults()
            output_base = defaults.get('output_dir', 'outputs')
            publication_dir = Path(output_base) / (response.publication_config.name if response.publication_config else 'publication')
            publication_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine source_key from publication name if not provided
            if not source_key:
                source_key = response.publication_config.name if response.publication_config else 'unknown'
            
            # Use Rich progress with a main task for total posts and a per-post subtask for steps.
            # Steps per post: convert -> download assets -> persist files -> index update
            steps_per_post = 4
            with Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("{task.fields[current_post]}", justify="left"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
                transient=False
            ) as progress:
                main_task = progress.add_task("Overall", total=len(response.posts), current_post="")

                posts_done = 0
                cumulative_time = 0.0
                persisted_messages = []

                for post in response.posts:
                    html = getattr(post, 'content_html', None)
                    if not html:
                        # advance the main counter even if there's nothing to do
                        posts_done += 1
                        progress.update(main_task, advance=1, current_post=f"Skipping {post.id.value} (no HTML)")
                        continue
                    # Start timing for ETA
                    start_time = time.time()

                    # Use a title-friendly label (truncate to 48 chars) and include slug/reading_time
                    slug = getattr(post, 'slug', None) or ''
                    rt = getattr(post, 'reading_time', None)
                    rt_label = f"{int(rt)}m" if isinstance(rt, (int, float)) else "N/A"
                    title_head = (post.title[:48] + '...') if post.title and len(post.title) > 48 else (post.title or post.id.value)
                    slug_part = (slug[:20] + '...') if slug and len(slug) > 20 else slug
                    title_label = f"{title_head} [{slug_part}] ({rt_label})"

                    subtask = progress.add_task(f"{title_label}", total=steps_per_post, current_post=f"{title_label} — starting")

                    try:
                        # Step 1: convert HTML -> Markdown
                        progress.update(subtask, description="convert: HTML -> Markdown", current_post=f"Post {post.id.value} — converting")
                        md, assets, code_blocks = content_extractor.html_to_markdown(html)
                        progress.update(subtask, advance=1)

                        # Step 2: classification
                        progress.update(subtask, description="classify: technical heuristics", current_post=f"Post {post.id.value} — classifying")
                        classification = content_extractor.classify_technical(html, code_blocks)
                        progress.update(subtask, advance=1)

                        # Step 3: persist to database (UPDATE with HTML/markdown)
                        progress.update(subtask, description="persist: save to database", current_post=f"Post {post.id.value} — persisting")
                        from ..infrastructure.pipeline_db import PipelineDB
                        import re
                        
                        db = PipelineDB()
                        
                        # Extract plain text for search (first 5000 chars, no markdown syntax)
                        text_only = re.sub(r'[#*`\[\]()]+', ' ', md)
                        text_only = re.sub(r'\s+', ' ', text_only).strip()
                        
                        # Get existing post and update with HTML/markdown
                        existing_post = db.get_post(post.id.value)
                        if existing_post:
                            # Update existing post with HTML/markdown content
                            existing_post['content_html'] = html
                            existing_post['content_markdown'] = md
                            existing_post['content_text'] = text_only[:5000]
                            existing_post['collection_mode'] = 'technical' if mode == 'technical' else 'full'
                            existing_post['is_technical'] = classification.get('is_technical')
                            existing_post['technical_score'] = classification.get('score')
                            existing_post['code_blocks'] = len(code_blocks)
                            existing_post['has_markdown'] = True
                            existing_post['metadata'] = {
                                'classifier': classification,
                                'code_blocks': code_blocks,
                                'assets': assets
                            }
                            
                            db.add_or_update_post(existing_post)
                        else:
                            # Fallback: create new (shouldn't happen if metadata save worked)
                            publication_name = response.publication_config.name if response.publication_config else 'unknown'
                            post_url = f"https://medium.com/@{publication_name}/{post.slug}-{post.id.value}" if hasattr(post, 'slug') else None
                            
                            post_data = {
                                'id': post.id.value,
                                'source': source_key,
                                'publication': publication_name,
                                'title': post.title,
                                'author': post.author.name if hasattr(post, 'author') and post.author else None,
                                'url': post_url,
                                'published_at': str(post.published_at) if hasattr(post, 'published_at') else None,
                                'reading_time': post.reading_time if hasattr(post, 'reading_time') else None,
                                'claps': getattr(post, 'claps', None),
                                'tags': getattr(post, 'tags', []),
                                'content_html': html,
                                'content_markdown': md,
                                'content_text': text_only[:5000],
                                'collection_mode': 'technical' if mode == 'technical' else 'full',
                                'is_technical': classification.get('is_technical'),
                                'technical_score': classification.get('score'),
                                'code_blocks': len(code_blocks),
                                'has_markdown': True,
                                'has_json': False,
                                'markdown_path': None,  # No longer saving to disk
                                'json_path': None,
                                'metadata': {
                                    'classifier': classification,
                                    'code_blocks': code_blocks,
                                    'assets': assets
                                }
                            }
                            
                            db.add_or_update_post(post_data)
                        
                        progress.update(subtask, advance=1)

                        # Step 4: finalize
                        progress.update(subtask, description="finalize: done", current_post=f"Post {post.id.value} — finalizing")
                        progress.update(subtask, advance=1)

                        # timing & ETA update using exponential moving average (EMA)
                        elapsed = time.time() - start_time
                        posts_done += 1
                        # Update EMA
                        if cumulative_time == 0.0:
                            # initialize EMA to first elapsed
                            ema = elapsed
                        else:
                            # alpha controls responsiveness; 0.3 is a reasonable default
                            alpha = 0.3
                            ema = alpha * elapsed + (1 - alpha) * ema
                        cumulative_time += elapsed
                        remaining = max(0, len(response.posts) - posts_done)
                        eta = remaining * ema
                        eta_str = f"ETA: {int(eta)}s" if eta >= 1 else "ETA: <1s"

                        # Mark post done in overall task and update current_post with ETA
                        progress.update(main_task, advance=1, current_post=f"{title_label} — done ({eta_str})")
                        # Remove subtask to keep UI tidy
                        progress.remove_task(subtask)
                        persisted_messages.append(f"✅ Saved post {post.id.value} to database")

                    except Exception as e:
                        progress.update(main_task, advance=1, current_post=f"Failed {post.id.value}")
                        # ensure subtask removed
                        try:
                            progress.remove_task(subtask)
                        except Exception:
                            pass
                        persisted_messages.append(f"⚠️ Failed to save post {post.id.value}: {e}")

                # After progress context is closed, print persisted messages in a concise block
                if persisted_messages:
                    self.console.print()
                    for m in persisted_messages:
                        self.console.print(m)
    
    def _handle_failed_response(self, response: ScrapePostsResponse) -> None:
        """Handle failed scraping response"""
        self.console.print("[red]❌ No posts found![/red]")
        self.console.print("[yellow]💡 Troubleshooting tips:[/yellow]")
        self.console.print("• Check publication name spelling")
        self.console.print("• Try --auto-discover for latest posts")
        self.console.print("• Use --skip-session for faster execution")
        self.console.print("• Provide --custom-ids if you have specific post IDs")
    
    def _save_to_file(self, content: str, filename: str) -> None:
        """Save content to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.console.print(f"[red]Error saving file: {e}[/red]")


# CLI Command Interface
@click.group(invoke_without_command=True)
@click.option('--publication', '-p', help='Publication name (netflix, pinterest, or any publication)')
@click.option('--source', '-s', help='Use configured source from YAML (e.g., --source netflix)')
@click.option('--bulk', '-b', help='Run bulk collection (e.g., --bulk tech_giants)')
@click.option('--list-sources', is_flag=True, help='List all available configured sources')
@click.option('--output', '-o', help='Output file for saving results')
@click.option('--format', '-f', 'format_type', type=click.Choice(['table', 'json', 'ids', 'md']), 
              default='table', help='Output format')
@click.option('--mode', '-m', 'mode', type=click.Choice(['ids', 'metadata', 'full', 'technical']),
              default='metadata', help='Operation mode/preset (ids|metadata|full|technical)')
@click.option('--custom-ids', help='Comma-separated list of specific post IDs')
@click.option('--skip-session', is_flag=True, help='Skip session initialization (faster)')
@click.option('--limit', type=int, help='Maximum number of posts to collect')
@click.option('--all', 'all_posts', is_flag=True, help='Collect ALL posts from publication (no limit)')
@click.option('--auto-discover', is_flag=True, 
              help='Force auto-discovery mode (production ready)')
@click.option('--index', is_flag=True, help='Update local search index for persisted posts')
@click.pass_context
def cli(ctx, publication, source, bulk, list_sources, output, format_type, mode, custom_ids, skip_session, limit, all_posts, auto_discover, index):
    """
    Universal Medium Scraper - Enterprise Edition
    
    Scrape any Medium publication with intelligent discovery.
    
    Examples:
    
    \b
    # Quick scrape with fallback  
    python main.py --publication netflix --limit 5
    
    \b
    # Production auto-discovery mode
    python main.py --publication pinterest --auto-discover --skip-session --format json
    
    \b
    # Use configured source
    python main.py --source netflix --limit 10
    
    \b
    # Use username from config
    python main.py --source skyscanner --all --format json
    
    \b 
    # Bulk collection from multiple sources
    python main.py --bulk tech_giants --limit 20
    
    \b
    # List all available sources
    python main.py --list-sources
    
    \b
    # Custom post IDs
    python main.py --publication netflix --custom-ids "ac15cada49ef,64c786c2a3ac"
    
    \b
    # Collect ALL posts from publication
    python main.py --publication netflix --all --skip-session --format json --output all_posts.json
    """
    
    # If a subcommand was invoked, skip the main CLI wiring (allows subcommands to run without loading adapters)
    if ctx.invoked_subcommand is not None:
        return

    # Create controller first for potential source listing
    from ..infrastructure.adapters.medium_api_adapter import MediumApiAdapter
    from ..infrastructure.external.repositories import InMemoryPublicationRepository, MediumSessionRepository
    from ..domain.services.publication_service import PostDiscoveryService, PublicationConfigService
    
    # Wire up dependencies
    post_repository = MediumApiAdapter()
    publication_repository = InMemoryPublicationRepository()
    session_repository = MediumSessionRepository()
    
    post_discovery_service = PostDiscoveryService(post_repository)
    publication_config_service = PublicationConfigService(publication_repository)
    
    scrape_posts_use_case = ScrapePostsUseCase(
        post_discovery_service,
        publication_config_service,
        session_repository
    )
    
    controller = CLIController(scrape_posts_use_case)
    # Map simple legacy cues to presets when user didn't explicitly change mode
    # If user asked for ids format or provided custom IDs, prefer the 'ids' preset
    if mode == 'metadata':
        if format_type == 'ids' or (custom_ids and custom_ids.strip()):
            mode = 'ids'

    # Validate and coerce mode/format combinations (simple, low-maintenance rules)
    effective_mode = mode
    effective_format = format_type

    # 1) mode 'ids' only makes sense with format 'ids' => coerce format
    if effective_mode == 'ids' and effective_format != 'ids':
        click.echo(f"⚠️  Mode 'ids' selected but format='{effective_format}' was requested; forcing --format ids for consistency.")
        effective_format = 'ids'

    # 2) format 'md' implies full content; if user didn't request full/technical, switch to 'full'
    if effective_format == 'md' and effective_mode not in ('full', 'technical'):
        click.echo(f"⚠️  Format 'md' requires full content. Switching mode '{effective_mode}' -> 'full'.")
        effective_mode = 'full'

    # 3) warn when asking for full/technical but choosing table output (table shows summary only)
    if effective_mode in ('full', 'technical') and effective_format == 'table':
        click.echo("⚠️  Mode '{0}' produces full content; --format table shows only a summary in the terminal.\n    Consider using --format json or --format md to persist full content.".format(effective_mode))

    # Apply the effective values forward
    mode = effective_mode
    format_type = effective_format

    # Handle source listing
    if list_sources:
        controller.list_sources()
        return
        
    # Handle bulk collection
    if bulk:
        controller.scrape_bulk_collection(
            bulk_key=bulk,
            limit=limit,
            format_type=format_type,
            mode=mode
        )
        return
    
    # Handle configured source
    if source:
        controller.scrape_from_config(
            source_key=source,
            limit=limit,
            format_type=format_type,
            output_file=output,
            all_posts=all_posts,
            mode=mode
        )
        return
    
    # Require publication if not using source or bulk
    if not publication:
        click.echo("❌ Error: Must specify --publication, --source, or --bulk")
        click.echo("💡 Use --list-sources to see available configured sources")
        return
    
    # Handle --all flag
    if all_posts:
        limit = None
        auto_discover = True  # Force auto-discover for complete collection
        click.echo("🌟 Collecting ALL posts from publication (this may take a while)...")
    
    # Execute traditional publication scraping
    controller.scrape_posts(
        publication=publication,
        limit=limit,
        format_type=format_type,
        custom_ids=custom_ids,
        auto_discover=auto_discover,
        skip_session=skip_session,
        output_file=output,
        mode=mode
    )


@cli.command('pipeline')
@click.option('--bulk', '-b', default='tech_giants', help='Bulk collection to process (default: tech_giants)')
@click.option('--skip-collect', is_flag=True, help='Skip data collection phase')
@click.option('--skip-ml', is_flag=True, help='Skip ML training and classification')
@click.option('--skip-webapp', is_flag=True, help='Skip WebApp integration')
@click.option('--limit', type=int, help='Limit posts per source (for testing)')
def pipeline(bulk, skip_collect, skip_ml, skip_webapp, limit):
    """
    🚀 Run the COMPLETE pipeline: Collect → Timeline → ML → WebApp
    
    This command automates the entire data processing workflow:
    
    \b
    Phase 1: Data Collection (with markdown extraction)
    Phase 2: Timeline Building (keyword-based classification)
    Phase 3: ML Training & Classification (architecture layers)
    Phase 4: WebApp Integration (copy + restart)
    
    Examples:
    
    \b
    # Full pipeline with all tech giants
    python main.py pipeline
    
    \b
    # Test with limited posts
    python main.py pipeline --limit 10
    
    \b
    # Skip collection (use existing data)
    python main.py pipeline --skip-collect
    
    \b
    # Only collect and build timelines
    python main.py pipeline --skip-ml --skip-webapp
    
    \b
    # Custom bulk collection
    python main.py pipeline --bulk custom_domains
    """
    import subprocess
    import time
    from pathlib import Path
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.table import Table
    
    console = Console()
    start_time = time.time()
    
    # Configuration
    config_manager = SourceConfigManager()
    output_dir = Path("outputs")
    webapp_dir = Path("webapp")
    
    console.print(Panel.fit(
        "🚀 [bold cyan]Medium Scraping Pipeline[/bold cyan] - Full Automation",
        border_style="cyan"
    ))
    console.print()
    
    # Get sources from bulk collection
    try:
        bulk_config = config_manager.get_bulk_config(bulk)
        sources = bulk_config.sources  # Access as attribute, not dict
        console.print(f"📦 Bulk Collection: [cyan]{bulk}[/cyan]")
        console.print(f"   {bulk_config.description}")
        console.print(f"📋 Sources: [green]{len(sources)}[/green] publications")
        console.print()
    except KeyError:
        console.print(f"[red]❌ Bulk collection '{bulk}' not found in medium_sources.yaml[/red]")
        console.print("[yellow]💡 Available collections:[/yellow]")
        bulks = config_manager.list_bulk_collections()
        for key in bulks.keys():
            console.print(f"   - {key}")
        return
    
    results = {
        'collected': 0,
        'timelines': 0,
        'ml_classified': 0,
        'webapp_copied': 0,
        'failed': []
    }
    
    # Initialize database
    from ..infrastructure.pipeline_db import PipelineDB
    db = PipelineDB()
    
    # =========================================================================
    # PHASE 1: Data Collection
    # =========================================================================
    if not skip_collect:
        console.print("[bold magenta]▶ Phase 1:[/bold magenta] Data Collection with Markdown Extraction")
        console.print("─" * 60)
        console.print()
        
        # Create progress table
        from rich.live import Live
        from rich.layout import Layout
        
        def create_progress_table():
            """Create a table showing collection progress"""
            table = Table(title="📊 Collection Progress", show_header=True, header_style="bold cyan")
            table.add_column("Source", style="cyan", width=15)
            table.add_column("Status", width=12)
            table.add_column("Posts", justify="right", width=8)
            table.add_column("With MD", justify="right", width=8)
            table.add_column("Progress", width=12)
            
            for source in sources:
                existing_posts = db.get_posts_by_source(source)
                posts_with_markdown = [p for p in existing_posts if p['has_markdown']]
                total = len(existing_posts)
                with_md = len(posts_with_markdown)
                
                if total == 0:
                    status = "🔵 Queued"
                    progress = "—"
                elif with_md == total:
                    status = "✅ Complete"
                    progress = "100%"
                else:
                    status = "🔄 Partial"
                    progress = f"{(with_md*100//total) if total else 0}%"
                
                table.add_row(source, status, str(total), str(with_md), progress)
            
            return table
        
        # Show initial state
        console.print(create_progress_table())
        console.print()
        
        for idx, source in enumerate(sources, 1):
            console.print(f"[cyan][{idx}/{len(sources)}][/cyan] Collecting: [green]{source}[/green]")
            
            # Check database for existing posts
            existing_posts = db.get_posts_by_source(source)
            # IMPORTANT: Check actual content_markdown, not just the flag!
            posts_with_markdown = [p for p in existing_posts if p.get('content_markdown')]
            
            if existing_posts and not limit:
                console.print(f"   [dim]📊 Database: {len(existing_posts)} posts, {len(posts_with_markdown)} with markdown[/dim]")
                
                # Skip if all posts have markdown
                if len(posts_with_markdown) == len(existing_posts) and len(existing_posts) > 0:
                    console.print(f"   [green]✅ Skipping - all posts already collected with markdown[/green]")
                    console.print()
                    results['collected'] += len(existing_posts)
                    continue
            
            # Build command
            cmd = [
                "uv", "run", "python", "main.py",
                "--source", source,
                "--all",
                "--mode", "technical",
                "--format", "json",
                "--skip-session",
                "--output", str(output_dir / f"{source}_posts.json")
            ]
            
            if limit:
                cmd.extend(["--limit", str(limit)])
            
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
            console.print()
            
            # Create live status panel
            from rich.live import Live
            from threading import Thread
            import time as time_module
            
            status_running = True
            
            def update_status():
                """Update status in background while subprocess runs"""
                start = time_module.time()
                while status_running:
                    elapsed = int(time_module.time() - start)
                    # Query DB for current progress
                    current_posts = db.get_posts_by_source(source)
                    current_with_md = len([p for p in current_posts if p['has_markdown']])
                    
                    status_text = (
                        f"⏱️  Elapsed: {elapsed}s | "
                        f"📊 Database: {len(current_posts)} posts | "
                        f"📝 With markdown: {current_with_md}"
                    )
                    
                    # Print inline update (overwrite previous line)
                    print(f"\r[dim]{status_text}[/dim]", end='', flush=True)
                    time_module.sleep(2)
            
            # Start status monitor in background
            monitor_thread = Thread(target=update_status, daemon=True)
            monitor_thread.start()
            
            try:
                # Run without capturing output so Rich progress bars show through
                result = subprocess.run(
                    cmd,
                    timeout=1800  # 30 minutes timeout
                )
                
                # Stop status monitor
                status_running = False
                monitor_thread.join(timeout=1)
                print()  # New line after inline updates
                console.print()  # Add spacing after subprocess output
                
                if result.returncode == 0:
                    # Count posts
                    json_file = output_dir / f"{source}_posts.json"
                    if json_file.exists():
                        import json
                        with open(json_file) as f:
                            posts = json.load(f)
                            post_count = len(posts)
                        console.print(f"[green]✅ Collected {post_count} posts from {source}[/green]")
                        results['collected'] += post_count
                        
                        # Sync to database (optimized - build index first)
                        console.print(f"[dim]   Syncing to database...[/dim]")
                        
                        # Build markdown index (one glob instead of N globs)
                        md_index = {}
                        for md_file in output_dir.glob("**/*.md"):
                            post_id = md_file.stem.split('_')[0]
                            md_index[post_id] = md_file
                        
                        # Batch insert/update
                        for post in posts:
                            post_id = post.get('id')
                            if not post_id:
                                continue
                            
                            # IMPORTANT: Check if post already has content in database
                            # The sync should not overwrite posts that already have HTML/markdown
                            existing_post = db.get_post(post_id)
                            if existing_post and existing_post.get('content_markdown'):
                                # Post already has content, skip sync for this post
                                continue
                            
                            md_file = md_index.get(post_id)
                            
                            post_data = {
                                'id': post_id,
                                'source': source,
                                'publication': post.get('publication', source),
                                'title': post.get('title'),
                                'author': post.get('author'),
                                'url': post.get('url'),
                                'published_at': post.get('published_at') or post.get('publishedAt'),
                                'reading_time': post.get('reading_time'),
                                'claps': post.get('claps'),
                                'tags': post.get('tags', []),
                                'collection_mode': 'technical',
                                'has_json': True,
                                'json_path': str(json_file),
                                'has_markdown': md_file is not None,
                                'markdown_path': str(md_file) if md_file else None
                            }
                            
                            # Check for technical classification
                            if md_file:
                                json_path = md_file.with_suffix('.json')
                                if json_path.exists():
                                    try:
                                        with open(json_path) as jf:
                                            meta = json.load(jf)
                                            classifier = meta.get('classifier', {})
                                            post_data['is_technical'] = classifier.get('is_technical')
                                            post_data['technical_score'] = classifier.get('score')
                                            post_data['code_blocks'] = len(meta.get('code_blocks', []))
                                    except:
                                        pass
                            
                            db.add_or_update_post(post_data)
                        
                        console.print(f"[dim]   ✅ Synced {post_count} posts to database[/dim]")
                    else:
                        console.print(f"[yellow]⚠️  No output file created for {source}[/yellow]")
                else:
                    console.print(f"[red]❌ Failed to collect from {source}[/red]")
                    results['failed'].append(f"collect:{source}")
            except subprocess.TimeoutExpired:
                status_running = False
                console.print(f"[red]❌ Timeout collecting from {source}[/red]")
                results['failed'].append(f"collect:{source}:timeout")
            except Exception as e:
                status_running = False
                console.print(f"[red]❌ Error collecting from {source}: {e}[/red]")
                results['failed'].append(f"collect:{source}:{str(e)}")
            
            # Show updated progress table
            console.print()
            console.print(create_progress_table())
            console.print()
        
        console.print(f"[blue]ℹ️  Collection complete: {results['collected']} sources processed[/blue]")
        console.print()
    else:
        console.print("[blue]ℹ️  Skipping data collection (--skip-collect)[/blue]")
        console.print()
    
    # =========================================================================
    # PHASE 2: Timeline Building
    # =========================================================================
    console.print("[bold magenta]▶ Phase 2:[/bold magenta] Timeline Building (Keyword-Based)")
    console.print("─" * 60)
    console.print()
    
    # Create progress tracking table for timelines
    def create_timeline_table():
        """Create a table showing timeline building progress"""
        table = Table(title="🕐 Timeline Progress", show_header=True, header_style="bold cyan")
        table.add_column("Source", style="cyan", width=15)
        table.add_column("Posts w/ Content", justify="right", width=15)
        table.add_column("Timeline", width=12)
        table.add_column("Status", width=20)
        
        for source in sources:
            posts_with_content = db.get_posts_with_content(source=source)
            timeline_file = output_dir / f"{source}_timeline.json"
            
            if not posts_with_content:
                status = "⚠️  No content"
                timeline_status = "—"
            elif timeline_file.exists():
                status = "✅ Complete"
                timeline_status = "Exists"
            else:
                status = "🔵 Pending"
                timeline_status = "—"
            
            table.add_row(source, str(len(posts_with_content)), timeline_status, status)
        
        return table
    
    console.print(create_timeline_table())
    console.print()
    
    # Get publication names from sources
    publications = []
    for source in sources:
        try:
            source_config = config_manager.get_source(source)
            # Use description field, extract publication name (before ' - ')
            pub_name = source_config.description.split(' - ')[0].strip()
            publications.append((source, pub_name))
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not get config for {source}: {e}[/yellow]")
            publications.append((source, source))
    
    for source, pub_name in publications:
        console.print(f"Building timeline for: [cyan]{pub_name}[/cyan] ([dim]source: {source}[/dim])")
        
        # Check if timeline already exists
        timeline_file = output_dir / f"{source}_timeline.json"
        if timeline_file.exists() and not limit:
            console.print(f"   [green]✅ Timeline already exists - skipping[/green]")
            console.print()
            results['timelines'] += 1
            continue
        
        # Check if posts with content exist in database
        posts_with_content = db.get_posts_with_content(source=source)
        if not posts_with_content:
            console.print(f"[yellow]⚠️  Skipping: No posts with content in database[/yellow]")
            console.print()
            continue
        
        console.print(f"   Found {len(posts_with_content)} posts with content in database")
        
        # Build timeline from database
        cmd = ["python", "scripts/build_timeline_v2.py", "--source", source]
        
        try:
            result = subprocess.run(cmd, timeout=300)
            
            console.print()  # Add spacing
            
            if result.returncode == 0:
                if timeline_file.exists():
                    import json
                    with open(timeline_file) as f:
                        timeline = json.load(f)
                        post_count = len(timeline.get('posts', []))
                    console.print(f"[green]✅ Timeline created with {post_count} posts[/green]")
                    results['timelines'] += 1
                    
                    # Update database - mark posts as in_timeline
                    for post in timeline.get('posts', []):
                        post_id = post.get('id')
                        if post_id and db.post_exists(post_id):
                            existing = db.get_post(post_id)
                            existing['in_timeline'] = True
                            db.add_or_update_post(existing)
                else:
                    console.print(f"[yellow]⚠️  Timeline file not created[/yellow]")
            else:
                console.print(f"[red]❌ Failed to build timeline[/red]")
                results['failed'].append(f"timeline:{source}")
        except Exception as e:
            console.print(f"[red]❌ Error building timeline: {e}[/red]")
            results['failed'].append(f"timeline:{source}:{str(e)}")
        
        console.print()
    
    # Show updated timeline table
    console.print(create_timeline_table())
    console.print()
    console.print(f"[blue]ℹ️  Timelines built: {results['timelines']}[/blue]")
    console.print()
    
    # =========================================================================
    # PHASE 3: ML Training & Classification
    # =========================================================================
    if not skip_ml:
        console.print("[bold magenta]▶ Phase 3:[/bold magenta] ML Training & Classification")
        console.print("─" * 60)
        console.print()
        
        # Find all timeline files
        timeline_files = list(output_dir.glob("*_timeline.json"))
        
        if not timeline_files:
            console.print("[yellow]⚠️  No timeline files found. Skipping ML phase.[/yellow]")
        else:
            # Show ML progress panel
            ml_table = Table(title="🤖 ML Classification Progress", show_header=True, header_style="bold magenta")
            ml_table.add_column("Task", style="cyan", width=30)
            ml_table.add_column("Status", width=40)
            
            ml_table.add_row("1. Model Training", "🔵 Pending")
            ml_table.add_row("2. Timeline Classification", "⏸️  Waiting")
            ml_table.add_row("3. Database Update", "⏸️  Waiting")
            
            console.print(ml_table)
            console.print()
            
            # Train on Netflix (largest dataset)
            netflix_timeline = output_dir / "netflix_timeline.json"
            model_file = output_dir / "ml_model.pkl"
            
            if netflix_timeline.exists():
                console.print("[yellow]🎓 Training ML model on Netflix timeline...[/yellow]")
                
                cmd = [
                    "python", "scripts/train_ml_classifier.py",
                    "--input", str(netflix_timeline),
                    "--output", str(model_file)
                ]
                
                try:
                    result = subprocess.run(cmd, timeout=600)
                    
                    console.print()  # Add spacing
                    
                    if result.returncode == 0 and model_file.exists():
                        console.print("[green]✅ ML model trained successfully[/green]")
                        console.print()
                        
                        # Update table
                        ml_table = Table(title="🤖 ML Classification Progress", show_header=True, header_style="bold magenta")
                        ml_table.add_column("Task", style="cyan", width=30)
                        ml_table.add_column("Status", width=40)
                        
                        ml_table.add_row("1. Model Training", "✅ Complete")
                        ml_table.add_row("2. Timeline Classification", f"🔄 Processing {len(timeline_files)} timelines")
                        ml_table.add_row("3. Database Update", "⏸️  Waiting")
                        
                        console.print(ml_table)
                        console.print()
                        
                        # Classify all timelines
                        console.print("[yellow]🏷️  Classifying all timelines with ML model...[/yellow]")
                        console.print()
                        
                        classified_count = 0
                        for idx, timeline_file in enumerate(timeline_files, 1):
                            basename = timeline_file.stem
                            refined_file = output_dir / f"{basename}_refined.json"
                            
                            console.print(f"   [{idx}/{len(timeline_files)}] Processing: [cyan]{timeline_file.name}[/cyan]")
                            
                            cmd = [
                                "python", "scripts/classify_timeline.py",
                                "--input", str(timeline_file),
                                "--model", str(model_file),
                                "--output", str(refined_file)
                            ]
                            
                            try:
                                result = subprocess.run(cmd, timeout=120)
                                
                                if result.returncode == 0 and refined_file.exists():
                                    console.print(f"[green]   ✅ Refined: {refined_file.name}[/green]")
                                    results['ml_classified'] += 1
                                    classified_count += 1
                                else:
                                    console.print(f"[red]   ❌ Failed to classify[/red]")
                                    results['failed'].append(f"ml:{timeline_file.name}")
                            except Exception as e:
                                console.print(f"[red]   ❌ Error: {e}[/red]")
                                results['failed'].append(f"ml:{timeline_file.name}:{str(e)}")
                        
                        console.print()
                        
                        # Final ML summary table
                        ml_table = Table(title="🤖 ML Classification Complete", show_header=True, header_style="bold green")
                        ml_table.add_column("Task", style="cyan", width=30)
                        ml_table.add_column("Status", width=40)
                        
                        ml_table.add_row("1. Model Training", "✅ Complete")
                        ml_table.add_row("2. Timeline Classification", f"✅ {classified_count}/{len(timeline_files)} timelines")
                        ml_table.add_row("3. Database Update", "✅ Complete")
                        
                        console.print(ml_table)
                        console.print()
                        console.print(f"[blue]ℹ️  ML classification complete: {results['ml_classified']} timelines[/blue]")
                    else:
                        console.print("[red]❌ ML training failed[/red]")
                        results['failed'].append("ml:training")
                except Exception as e:
                    console.print(f"[red]❌ ML training error: {e}[/red]")
            else:
                console.print("[yellow]⚠️  Netflix timeline not found. Skipping ML training.[/yellow]")
        
        console.print()
    else:
        console.print("[blue]ℹ️  Skipping ML training and classification (--skip-ml)[/blue]")
        console.print()
    
    # =========================================================================
    # PHASE 4: WebApp Integration
    # =========================================================================
    if not skip_webapp:
        console.print("[bold magenta]▶ Phase 4:[/bold magenta] WebApp Integration")
        console.print("─" * 60)
        console.print()
        
        if not webapp_dir.exists():
            console.print(f"[yellow]⚠️  WebApp directory not found: {webapp_dir}[/yellow]")
        else:
            webapp_data_dir = webapp_dir / "api" / "data"
            webapp_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy refined timelines
            refined_files = list(output_dir.glob("*_timeline_refined.json"))
            
            if not refined_files:
                console.print("[yellow]⚠️  No refined timeline files to copy[/yellow]")
            else:
                import shutil
                
                for refined_file in refined_files:
                    try:
                        shutil.copy(refined_file, webapp_data_dir / refined_file.name)
                        console.print(f"[green]✅ Copied: {refined_file.name}[/green]")
                        results['webapp_copied'] += 1
                    except Exception as e:
                        console.print(f"[red]❌ Failed to copy {refined_file.name}: {e}[/red]")
                
                console.print()
                console.print(f"[blue]ℹ️  Copied {results['webapp_copied']} files to WebApp[/blue]")
                
                # Restart API container
                if (webapp_dir / "docker-compose.yml").exists():
                    console.print()
                    console.print("[yellow]Restarting WebApp API...[/yellow]")
                    
                    try:
                        result = subprocess.run(
                            ["docker", "compose", "restart", "api"],
                            cwd=str(webapp_dir),
                            timeout=30
                        )
                        
                        console.print()  # Add spacing
                        
                        if result.returncode == 0:
                            console.print("[green]✅ API container restarted[/green]")
                        else:
                            console.print("[yellow]⚠️  Could not restart API container[/yellow]")
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Could not restart API: {e}[/yellow]")
        
        console.print()
    else:
        console.print("[blue]ℹ️  Skipping WebApp integration (--skip-webapp)[/blue]")
        console.print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    # Get database stats
    db_stats = db.get_stats()
    
    console.print(Panel.fit(
        f"📊 [bold cyan]Pipeline Summary[/bold cyan]\n\n"
        f"⏱️  Execution Time: [yellow]{minutes}m {seconds}s[/yellow]\n\n"
        f"Pipeline Results:\n"
        f"  • Sources collected: [green]{results['collected']}[/green]\n"
        f"  • Timelines built: [green]{results['timelines']}[/green]\n"
        f"  • ML classifications: [green]{results['ml_classified']}[/green]\n"
        f"  • WebApp files copied: [green]{results['webapp_copied']}[/green]\n"
        f"  • Failed operations: [red]{len(results['failed'])}[/red]\n\n"
        f"Database Status:\n"
        f"  • Total posts: [cyan]{db_stats['total_posts']}[/cyan]\n"
        f"  • With markdown: [green]{db_stats['with_markdown']}[/green]\n"
        f"  • In timeline: [blue]{db_stats['in_timeline']}[/blue]\n"
        f"  • ML classified: [magenta]{db_stats['ml_classified']}[/magenta]\n"
        f"  • Technical posts: [yellow]{db_stats['technical_posts']}[/yellow]",
        border_style="cyan"
    ))
    
    if results['failed']:
        console.print()
        console.print("[yellow]Failed operations:[/yellow]")
        for failure in results['failed']:
            console.print(f"  • {failure}")
    
    console.print()
    console.print("[bold green]✨ Pipeline Complete![/bold green]")
    console.print()
    console.print("[cyan]Next Steps:[/cyan]")
    console.print("  1. View database stats: [dim]python main.py db stats[/dim]")
    console.print("  2. View timelines:      [dim]ls -lh outputs/*_timeline*.json[/dim]")
    console.print("  3. Check WebApp:        [dim]cd webapp && docker compose ps[/dim]")
    console.print("  4. Browse UI:           [dim]http://localhost:3000[/dim]")
    console.print()


@cli.command('add-source')
@click.option('--key', required=True, help='Key name to use in medium_sources.yaml (e.g. pinterest)')
@click.option('--type', 'stype', default='publication', type=click.Choice(['publication', 'username']), help='Type: publication or username')
@click.option('--name', required=True, help='Publication name, domain, or @username')
@click.option('--description', default='', help='Short description for the source')
@click.option('--auto-discover/--no-auto-discover', default=True, help='Enable auto-discover for this source')
@click.option('--custom-domain/--no-custom-domain', default=False, help='Mark this source as a custom domain')
@click.option('--yes', '-y', is_flag=True, help='Assume yes when overwriting an existing source')
def add_source(key, stype, name, description, auto_discover, custom_domain, yes):
    """Add or update a source in `medium_sources.yaml`.

    Example:
      python main.py add-source --key pinterest --type publication --name pinterest --description "Pinterest Engineering" --auto-discover
    """
    try:
        manager = SourceConfigManager()
        exists = manager.validate_source(key)

        if exists and not yes:
            confirm = click.confirm(f"Source '{key}' already exists. Overwrite?", default=False)
            if not confirm:
                click.echo("Cancelled — no changes made.")
                return

        manager.add_or_update_source(
            source_key=key,
            source_data={
                'type': stype,
                'name': name,
                'description': description,
                'auto_discover': auto_discover,
                'custom_domain': custom_domain
            }
        )
        click.echo(f"✅ Source '{key}' added/updated in {manager.config_path}")
    except Exception as e:
        click.echo(f"❌ Failed to add/update source: {e}")


# ============================================================================
# DATABASE MANAGEMENT COMMANDS
# ============================================================================

@cli.group('db')
def db_group():
    """Database management commands"""
    pass


@db_group.command('stats')
def db_stats():
    """Show database statistics"""
    from ..infrastructure.pipeline_db import PipelineDB
    
    console = Console()
    db = PipelineDB()
    stats = db.get_stats()
    
    console.print()
    console.print(Panel.fit(
        "📊 [bold cyan]Pipeline Database Statistics[/bold cyan]",
        border_style="cyan"
    ))
    console.print()
    
    # Overview table
    table = Table(title="Overview", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Total Posts", str(stats['total_posts']))
    table.add_row("With Markdown", str(stats['with_markdown']))
    table.add_row("In Timeline", str(stats['in_timeline']))
    table.add_row("ML Classified", str(stats['ml_classified']))
    table.add_row("Technical Posts", str(stats['technical_posts']))
    table.add_row("", "")
    table.add_row("Needs Markdown", str(stats['needs_markdown']), style="yellow")
    table.add_row("Needs ML", str(stats['needs_ml']), style="yellow")
    
    console.print(table)
    console.print()
    
    # By source table
    if stats['by_source']:
        table = Table(title="Posts by Source", show_header=True)
        table.add_column("Source", style="cyan")
        table.add_column("Posts", justify="right", style="green")
        
        for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True):
            table.add_row(source, str(count))
        
        console.print(table)
        console.print()


@db_group.command('inspect')
@click.option('--source', help='Filter by source')
@click.option('--limit', type=int, default=20, help='Limit results')
def db_inspect(source, limit):
    """Inspect posts in database"""
    from ..infrastructure.pipeline_db import PipelineDB
    
    console = Console()
    db = PipelineDB()
    
    if source:
        posts = db.get_posts_by_source(source)
        title = f"Posts from {source}"
    else:
        # Get all posts
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM posts ORDER BY collected_at DESC LIMIT {limit}")
            posts = [dict(row) for row in cursor.fetchall()]
        title = f"Recent Posts (limit {limit})"
    
    console.print()
    
    if not posts:
        console.print("[yellow]No posts found[/yellow]")
        return
    
    # Show posts limited
    display_posts = posts[:limit] if len(posts) > limit else posts
    
    table = Table(title=title, show_header=True)
    table.add_column("ID", style="dim", width=12)
    table.add_column("Title", style="cyan", width=40)
    table.add_column("MD", justify="center", width=3)
    table.add_column("Timeline", justify="center", width=8)
    table.add_column("ML", justify="center", width=3)
    table.add_column("Tech", justify="center", width=4)
    
    for post in display_posts:
        table.add_row(
            post['id'][:12],
            (post['title'] or 'Untitled')[:40],
            "✅" if post['has_markdown'] else "❌",
            "✅" if post['in_timeline'] else "❌",
            "✅" if post['ml_classified'] else "❌",
            "✅" if post['is_technical'] else "❌"
        )
    
    console.print(table)
    
    if len(posts) > limit:
        console.print(f"\n[dim]Showing {limit} of {len(posts)} posts[/dim]")
    
    console.print()


@db_group.command('sync')
@click.option('--source', help='Sync specific source only')
def db_sync(source):
    """Sync JSON files into database"""
    from ..infrastructure.pipeline_db import PipelineDB
    from ..infrastructure.config.source_manager import SourceConfigManager
    
    console = Console()
    db = PipelineDB()
    source_manager = SourceConfigManager()
    
    console.print("[yellow]🔄 Syncing JSON files into database...[/yellow]")
    console.print()
    
    outputs_dir = Path("outputs")
    
    if source:
        json_files = [outputs_dir / f"{source}_posts.json"]
    else:
        json_files = list(outputs_dir.glob("*_posts.json"))
    
    if not json_files:
        console.print("[red]No JSON files found[/red]")
        return
    
    total_synced = 0
    total_new = 0
    total_updated = 0
    
    for json_file in json_files:
        if not json_file.exists():
            continue
        
        source_key = json_file.stem.replace('_posts', '')
        console.print(f"Processing: [cyan]{source_key}[/cyan]")
        
        # Get publication name from YAML config
        try:
            source_config = source_manager.get_source(source_key)
            publication_name = source_config.name
        except KeyError:
            # Fallback to source_key if not in config
            publication_name = source_key
            console.print(f"  [dim]⚠️  Source not in config, using '{source_key}' as publication[/dim]")
        
        try:
            with open(json_file) as f:
                posts = json.load(f)
            
            for post in posts:
                post_id = post.get('id')
                if not post_id:
                    continue
                
                exists = db.post_exists(post_id)
                
                # Use publication from JSON if present, otherwise from config
                publication = post.get('publication', publication_name)
                
                post_data = {
                    'id': post_id,
                    'source': source_key,
                    'publication': publication,
                    'title': post.get('title'),
                    'author': post.get('author'),
                    'url': post.get('url'),
                    'published_at': post.get('published_at') or post.get('publishedAt'),
                    'reading_time': post.get('reading_time'),
                    'claps': post.get('claps'),
                    'tags': post.get('tags', []),
                    'collection_mode': 'metadata',
                    'has_json': True,
                    'json_path': str(json_file)
                }
                
                # Check if markdown exists - search in all subdirectories
                md_files = list(outputs_dir.glob(f"**/{post_id}_*.md"))
                if md_files:
                    post_data['has_markdown'] = True
                    post_data['markdown_path'] = str(md_files[0])
                    
                    # Read markdown content into database
                    try:
                        content_md = md_files[0].read_text(encoding='utf-8')
                        post_data['content_markdown'] = content_md
                        # Extract plain text for search (first 5000 chars, no markdown syntax)
                        import re
                        text_only = re.sub(r'[#*`\[\]()]+', ' ', content_md)
                        text_only = re.sub(r'\s+', ' ', text_only).strip()
                        post_data['content_text'] = text_only[:5000]
                    except:
                        pass
                    
                    # Try to get technical classification from JSON
                    json_path = md_files[0].with_suffix('.json')
                    if json_path.exists():
                        try:
                            with open(json_path) as jf:
                                meta = json.load(jf)
                                classifier = meta.get('classifier', {})
                                post_data['is_technical'] = classifier.get('is_technical')
                                post_data['technical_score'] = classifier.get('score')
                                post_data['code_blocks'] = len(meta.get('code_blocks', []))
                                # Store full metadata as JSON
                                post_data['metadata'] = meta
                        except:
                            pass
                
                db.add_or_update_post(post_data)
                
                if exists:
                    total_updated += 1
                else:
                    total_new += 1
                
                total_synced += 1
            
            console.print(f"  [green]✅ Synced {len(posts)} posts[/green]")
        
        except Exception as e:
            console.print(f"  [red]❌ Error: {e}[/red]")
    
    console.print()
    console.print(f"[green]✨ Sync complete![/green]")
    console.print(f"  Total synced: {total_synced}")
    console.print(f"  New: {total_new}")
    console.print(f"  Updated: {total_updated}")
    console.print()


@db_group.command('needs-markdown')
@click.option('--source', help='Get posts needing markdown for specific source')
def db_needs_markdown(source):
    """List posts that need markdown extraction"""
    from ..infrastructure.pipeline_db import PipelineDB
    
    console = Console()
    db = PipelineDB()
    
    posts = db.get_posts_needing_markdown(source)
    
    console.print()
    console.print(f"[yellow]Posts needing markdown extraction: {len(posts)}[/yellow]")
    console.print()
    
    if not posts:
        console.print("[green]All posts have markdown! ✅[/green]")
        return
    
    # Group by source
    by_source = {}
    for post in posts:
        src = post['source']
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(post)
    
    table = Table(show_header=True)
    table.add_column("Source", style="cyan")
    table.add_column("Posts Without Markdown", justify="right", style="yellow")
    
    for src, posts_list in sorted(by_source.items()):
        table.add_row(src, str(len(posts_list)))
    
    console.print(table)
    console.print()
    console.print("[dim]Tip: Run with --source to re-collect specific source[/dim]")
    console.print()


@db_group.command('needs-ml')
def db_needs_ml():
    """List posts that need ML classification"""
    from ..infrastructure.pipeline_db import PipelineDB
    
    console = Console()
    db = PipelineDB()
    
    posts = db.get_posts_needing_ml_classification()
    
    console.print()
    console.print(f"[yellow]Posts needing ML classification: {len(posts)}[/yellow]")
    console.print()
    
    if not posts:
        console.print("[green]All posts are ML classified! ✅[/green]")
        return
    
    # Show sample
    table = Table(title="Sample (first 20)", show_header=True)
    table.add_column("ID", style="dim", width=12)
    table.add_column("Source", style="cyan", width=15)
    table.add_column("Title", style="white", width=50)
    
    for post in posts[:20]:
        table.add_row(
            post['id'][:12],
            post['source'],
            (post['title'] or 'Untitled')[:50]
        )
    
    console.print(table)
    
    if len(posts) > 20:
        console.print(f"\n[dim]Showing 20 of {len(posts)} posts[/dim]")
    
    console.print()


@db_group.command('clean')
@click.option('--days', type=int, default=7, help='Delete cache older than N days')
def db_clean(days):
    """Clean old cache entries"""
    from ..infrastructure.pipeline_db import PipelineDB
    from datetime import datetime, timedelta
    
    console = Console()
    db = PipelineDB()
    
    console.print(f"[yellow]Cleaning cache older than {days} days...[/yellow]")
    
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
        deleted = cursor.rowcount
    
    console.print(f"[green]✅ Deleted {deleted} cache entries[/green]")
    console.print()


@cli.command('enrich')
@click.option('--source', '-s', help='Filter by source (e.g., netflix)')
@click.option('--publication', '-p', help='Filter by publication name')
@click.option('--limit', '-l', type=int, help='Limit number of posts to enrich')
@click.option('--force', is_flag=True, help='Re-fetch HTML even for posts that already have it')
def enrich_posts_command(source, publication, limit, force):
    """
    Enrich posts in database with HTML and Markdown content.
    
    This command fetches HTML for posts that don't have it yet,
    converts to Markdown, and updates the database.
    
    Examples:
    
    \b
    # Enrich all posts from Netflix
    uv run python main.py enrich --source netflix
    
    \b
    # Enrich first 10 posts from any source
    uv run python main.py enrich --limit 10
    
    \b
    # Re-fetch HTML for all Netflix posts
    uv run python main.py enrich --source netflix --force
    """
    from ..infrastructure.pipeline_db import PipelineDB
    from ..infrastructure.adapters.medium_api_adapter import MediumApiAdapter
    from ..infrastructure.config.source_manager import SourceConfigManager
    from ..domain.entities.publication import Post, PostId, Author
    from datetime import datetime, timezone
    import re
    
    console = Console()
    db = PipelineDB()
    adapter = MediumApiAdapter()
    config_manager = SourceConfigManager()
    
    console.print()
    console.print("[bold blue]🔄 Enriching Posts with HTML & Markdown[/bold blue]")
    console.print()
    
    # Get posts needing enrichment
    with db._get_connection() as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM posts WHERE 1=1"
        params = []
        
        if not force:
            query += " AND (content_html IS NULL OR content_markdown IS NULL)"
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if publication:
            query += " AND publication = ?"
            params.append(publication)
        
        query += " ORDER BY published_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        posts_to_enrich = [dict(row) for row in cursor.fetchall()]
    
    if not posts_to_enrich:
        console.print("[green]✅ All posts already have HTML and Markdown![/green]")
        console.print()
        return
    
    console.print(f"[yellow]Found {len(posts_to_enrich)} posts to enrich[/yellow]")
    console.print()
    
    # Process each post
    enriched_count = 0
    failed_count = 0
    
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        task = progress.add_task("Enriching posts...", total=len(posts_to_enrich))
        
        for post_data in posts_to_enrich:
            post = None  # Initialize to avoid UnboundLocalError
            try:
                # Get publication config
                source_key = post_data['source']
                
                try:
                    # Try to get config from YAML
                    sources = config_manager.load_sources()
                    source_config = sources.get('sources', {}).get(source_key)
                    if source_config:
                        from ..infrastructure.external.repositories import InMemoryPublicationRepository
                        repo = InMemoryPublicationRepository()
                        config = repo.create_generic_config(source_config.get('publication', post_data['publication']))
                    else:
                        raise ValueError("Config not found")
                except:
                    # Create generic config
                    from ..infrastructure.external.repositories import InMemoryPublicationRepository
                    repo = InMemoryPublicationRepository()
                    config = repo.create_generic_config(post_data['publication'])
                
                # Create Post object
                author_name = post_data.get('author') or 'Unknown'
                post = Post(
                    id=PostId(post_data['id']),
                    title=post_data['title'] or 'Untitled',
                    slug=post_data.get('url', '').split('/')[-1] if post_data.get('url') else post_data['id'],
                    author=Author(id='unknown', name=author_name, username=author_name.lower().replace(' ', '_')),
                    published_at=datetime.now(timezone.utc),
                    reading_time=post_data.get('reading_time', 0)
                )
                
                # Fetch HTML
                html = adapter.fetch_post_html(post, config)
                
                if not html:
                    failed_count += 1
                    progress.update(task, advance=1, description=f"❌ Failed: {post.id.value[:12]}")
                    continue
                
                # Convert to Markdown
                md, assets, code_blocks = content_extractor.html_to_markdown(html)
                
                # Classify
                classification = content_extractor.classify_technical(html, code_blocks)
                
                # Extract plain text for search
                text_only = re.sub(r'[#*`\[\]()]+', ' ', md)
                text_only = re.sub(r'\s+', ' ', text_only).strip()
                
                # Update database
                post_data['content_html'] = html
                post_data['content_markdown'] = md
                post_data['content_text'] = text_only[:5000]
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
                
                enriched_count += 1
                progress.update(task, advance=1, description=f"✅ Enriched: {post.id.value[:12]}")
                
            except Exception as e:
                failed_count += 1
                post_id = post.id.value[:12] if post else post_data.get('id', 'unknown')[:12]
                progress.update(task, advance=1, description=f"❌ Error: {post_id} - {str(e)[:30]}")
    
    console.print()
    console.print(f"[green]✅ Successfully enriched: {enriched_count} posts[/green]")
    if failed_count > 0:
        console.print(f"[red]❌ Failed: {failed_count} posts[/red]")
    console.print()


@cli.command('full')
@click.option('--source', '-s', help='Process specific source (default: ALL from YAML)')
@click.option('--limit', '-l', type=int, help='Limit posts per source')
def full_command(source, limit):
    """🚀 FULL: Process ALL sources OR one - Collect + Enrich + Label + Timeline"""
    from ..infrastructure.pipeline_db import PipelineDB
    from ..infrastructure.config.source_manager import SourceConfigManager
    
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
        
        # Just call ETL for each source (reuse existing logic)
        try:
            # Import inline to avoid circular deps
            from ..infrastructure.adapters.medium_api_adapter import MediumApiAdapter
            from ..infrastructure.external.repositories import InMemoryPublicationRepository, MediumSessionRepository
            from ..domain.services.publication_service import PostDiscoveryService, PublicationConfigService
            from ..application.use_cases.scrape_posts import ScrapePostsUseCase, ScrapePostsRequest
            
            # Phase 1: Always try to collect new posts
            with console.status("[blue]📥 Collecting...[/blue]", spinner="dots"):
                post_repo = MediumApiAdapter()
                pub_repo = InMemoryPublicationRepository()
                sess_repo = MediumSessionRepository()
                
                svc = PostDiscoveryService(post_repo)
                cfg_svc = PublicationConfigService(pub_repo)
                use_case = ScrapePostsUseCase(svc, cfg_svc, sess_repo)
                
                req = ScrapePostsRequest(publication_name=src, limit=limit, auto_discover=True, skip_session=True, mode='metadata')
                resp = use_case.execute(req)
            
            posts_collected = 0
            if resp.success and len(resp.posts) > 0:
                # Phase 2: Save to DB (add_or_update handles duplicates)
                console.print(f"[dim]💾 Saving {len(resp.posts)} posts to database...[/dim]")
                new_posts = 0
                updated_posts = 0
                
                for post in resp.posts:
                    try:
                        # Check if post already exists
                        existing = db.post_exists(post.id.value)
                        
                        # If exists and already enriched, skip update (preserve enrichment)
                        if existing:
                            existing_post = next((p for p in db.get_posts_by_source(src) if p['id'] == post.id.value), None)
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
            
            # Phase 3: Enrich - use subprocess to call etl
            console.print("[blue]🔄 Enriching...[/blue]")
            import subprocess
            result = subprocess.run(
                ["uv", "run", "python", "main.py", "etl", "--source", src, "--limit", str(limit)] if limit else ["uv", "run", "python", "main.py", "etl", "--source", src],
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


@cli.command('etl')
@click.option('--source', '-s', required=True, help='Source to process (e.g., netflix)')
@click.option('--limit', '-l', type=int, help='Limit posts to process (for testing)')
@click.option('--slow-mode', is_flag=True, help='Ultra slow mode: 1 post per minute (avoid rate limiting)')
def etl_command(source, limit, slow_mode):
    """
    🚀 Simple ETL: Enrich existing posts + Generate timeline
    
    This command processes posts already in database:
    1. Enrich with HTML and Markdown (if missing)
    2. Classify technically (automatic)
    3. Generate timeline with ML layer classification
    
    Examples:
    
    \b
    # Process all Netflix posts
    uv run python main.py etl --source netflix
    
    \b
    # Test with 20 posts
    uv run python main.py etl --source netflix --limit 20
    """
    from ..infrastructure.pipeline_db import PipelineDB
    from ..infrastructure.adapters.medium_api_adapter import MediumApiAdapter
    from ..infrastructure.config.source_manager import SourceConfigManager
    from ..domain.entities.publication import Post, PostId, Author
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    import re
    
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
    
    # Step 1: Enrich
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
    else:
        if already_enriched > 0:
            console.print(f"[dim]ℹ️  {already_enriched} posts already enriched, skipping...[/dim]")
        console.print(f"[yellow]Processing {len(posts_to_enrich)} posts...[/yellow]")
        
        enriched = 0
        failed = 0
        failure_reasons = {}  # Track failure reasons
        
        # Use simpler progress to avoid flickering
        console.print()
        import time
        for i, post_data in enumerate(posts_to_enrich, 1):
            # Show progress every 10 posts or at the end
            if i % 10 == 1 or i == len(posts_to_enrich):
                console.print(f"[cyan]Progress: {i}/{len(posts_to_enrich)} posts ({i*100//len(posts_to_enrich)}%)[/cyan]", end='\r')
            
            # Add delay BEFORE processing (skip first post)
            if i > 1:
                time.sleep(3)  # 3 seconds between posts
            
            post = None
            try:
                # Get config
                try:
                    sources = config_manager.load_sources()
                    source_config = sources.get('sources', {}).get(source)
                    if source_config:
                        from ..infrastructure.external.repositories import InMemoryPublicationRepository
                        repo = InMemoryPublicationRepository()
                        config = repo.create_generic_config(source_config.get('publication', post_data['publication']))
                    else:
                        raise ValueError("Config not found")
                except:
                    from ..infrastructure.external.repositories import InMemoryPublicationRepository
                    repo = InMemoryPublicationRepository()
                    config = repo.create_generic_config(post_data['publication'])
                
                author_name = post_data.get('author') or 'Unknown'
                post = Post(
                    id=PostId(post_data['id']),
                    title=post_data['title'] or 'Untitled',
                    slug=post_data.get('url', '').split('/')[-1] if post_data.get('url') else post_data['id'],
                    author=Author(id='unknown', name=author_name, username=author_name.lower().replace(' ', '_')),
                    published_at=datetime.now(timezone.utc),
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
                post_data['content_text'] = text_only[:5000]
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
                reason = str(e)[:50]  # Truncate long errors
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        console.print()  # New line after progress
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
    
    # Step 2: ML-Based Discovery & Timeline
    console.print("[bold blue]Step 2/2: ML Discovery & Timeline Generation...[/bold blue]")
    console.print("[dim]Using: Clustering + NER + Q&A (NO hardcoded keywords!)[/dim]")
    console.print()
    
    posts = db.get_posts_with_content(source=source)
    
    if not posts:
        console.print("[red]❌ No posts with content![/red]")
        return
    
    console.print(f"[yellow]Processing {len(posts)} posts with ML...[/yellow]")
    
    # Prepare data for ML discovery
    entries_for_ml = []
    for post in posts:
        md = post.get('content_markdown', '')
        if not md:
            continue
        
        date = None
        if post.get('published_at'):
            try:
                date = datetime.fromisoformat(str(post['published_at']).replace('Z', '+00:00')).date()
            except:
                pass
        
        tags = json.loads(post.get('tags', '[]')) if isinstance(post.get('tags'), str) else post.get('tags', [])
        
        lines = md.split('\n')
        snippet = ''
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('---'):
                snippet = line[:200]
                break
        
        entries_for_ml.append({
            'id': post['id'],
            'title': post.get('title', 'Untitled'),
            'date': date.isoformat() if date else None,
            'content': md,  # Full content for ML
            'snippet': snippet,
            'url': post.get('url', ''),
            'author': post.get('author', 'Unknown'),
            'reading_time': post.get('reading_time', 0),
            'is_technical': post.get('is_technical', False),
            'technical_score': post.get('technical_score', 0.0),
            'code_blocks': post.get('code_blocks', 0),
            'path': post.get('url', ''),  # For ML clustering
        })
    
    if not entries_for_ml:
        console.print("[red]❌ No posts with content to process![/red]")
        return
    
    # Run ML discovery
    console.print("[cyan]🤖 Running ML discovery (this may take a few minutes)...[/cyan]")
    console.print()
    
    try:
        # Import ML discovery module
        import sys
        from pathlib import Path
        ml_path = Path(__file__).parent.parent / 'ml_classifier'
        sys.path.insert(0, str(ml_path))
        
        from discover_enriched import (
            load_embedder, load_ner_pipeline, load_qa_pipeline,
            extract_tech_stack, extract_patterns, extract_solutions,
            extract_problem, extract_approach
        )
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Load models
        console.print("[dim]Loading embedder...[/dim]")
        embedder = load_embedder()
        
        console.print("[dim]Loading NER model...[/dim]")
        ner_pipeline = load_ner_pipeline()
        
        console.print("[dim]Loading Q&A model...[/dim]")
        qa_pipeline = load_qa_pipeline()
        
        console.print()
        
        # Extract texts for processing
        texts = [e['content'] for e in entries_for_ml]
        
        # 1. Clustering for layers
        console.print("[cyan]1/4 Clustering for topic discovery...[/cyan]")
        embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
        
        n_clusters = min(8, len(entries_for_ml))  # Max 8 clusters (like original layers)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Extract keywords per cluster
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        cluster_info = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
            top_indices = cluster_tfidf.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            # Auto-label from top keywords
            label = keywords[0].replace('_', ' ').title()
            
            cluster_info[cluster_id] = {
                'label': label,
                'keywords': keywords,
                'size': int(cluster_mask.sum())
            }
        
        # Assign layers to entries
        for i, entry in enumerate(entries_for_ml):
            entry['layers'] = [cluster_info[cluster_labels[i]]['label']]
        
        console.print(f"[green]✓ Discovered {n_clusters} topics[/green]")
        console.print()
        
        # 2. Tech Stack Extraction (NER)
        console.print("[cyan]2/4 Extracting tech stack (NER)...[/cyan]")
        for i, entry in enumerate(entries_for_ml):
            if i % 50 == 0:
                console.print(f"[dim]  Processing {i+1}/{len(entries_for_ml)}[/dim]", end='\r')
            
            tech_stack = extract_tech_stack(entry['content'], ner_pipeline)
            entry['tech_stack'] = tech_stack
        
        total_techs = sum(len(e.get('tech_stack', [])) for e in entries_for_ml)
        console.print(f"[green]✓ Extracted {total_techs} technology mentions[/green]")
        console.print()
        
        # 3. Pattern Extraction (NER + Semantic)
        console.print("[cyan]3/4 Extracting architectural patterns (NO hardcoded list)...[/cyan]")
        for i, entry in enumerate(entries_for_ml):
            if i % 50 == 0:
                console.print(f"[dim]  Processing {i+1}/{len(entries_for_ml)}[/dim]", end='\r')
            
            patterns = extract_patterns(entry['content'], ner_pipeline, embedder)
            entry['patterns'] = patterns
        
        total_patterns = sum(len(e.get('patterns', [])) for e in entries_for_ml)
        console.print(f"[green]✓ Extracted {total_patterns} architectural patterns[/green]")
        console.print()
        
        # 4. Solution Mining + Problem/Approach Extraction (Q&A)
        console.print("[cyan]4/4 Mining solutions + extracting problem/approach (Q&A)...[/cyan]")
        for entry in entries_for_ml:
            tech_stack = entry.get('tech_stack', [])
            content = entry['content']
            
            # Extract solutions (semantic similarity)
            solutions = extract_solutions(content, tech_stack, embedder)
            entry['solutions'] = solutions
            
            # Extract problem (Q&A)
            problem = extract_problem(content, qa_pipeline)
            entry['problem'] = problem
            
            # Extract approach (Q&A)
            approach = extract_approach(content, qa_pipeline)
            entry['approach'] = approach
        
        total_solutions = sum(len(e.get('solutions', [])) for e in entries_for_ml)
        posts_with_problem = sum(1 for e in entries_for_ml if e.get('problem'))
        posts_with_approach = sum(1 for e in entries_for_ml if e.get('approach'))
        console.print(f"[green]✓ Mined {total_solutions} solution descriptions[/green]")
        console.print(f"[green]✓ Extracted problem from {posts_with_problem} posts[/green]")
        console.print(f"[green]✓ Extracted approach from {posts_with_approach} posts[/green]")
        console.print()
        
        # 5. Save ML data to database
        console.print("[cyan]5/5 Saving ML discoveries to database...[/cyan]")
        saved_count = 0
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
            saved_count += 1
        
        console.print(f"[green]✓ Saved ML data for {saved_count} posts to database[/green]")
        console.print()
        
        # Remove 'content' field (too large for JSON)
        entries = []
        for e in entries_for_ml:
            entry_copy = e.copy()
            entry_copy.pop('content', None)
            entry_copy.pop('path', None)
            entries.append(entry_copy)
        
    except Exception as e:
        console.print(f"[red]❌ ML Discovery failed: {e}[/red]")
        console.print("[yellow]Falling back to basic timeline...[/yellow]")
        import traceback
        traceback.print_exc()
        
        # Fallback: simple entries without ML
        entries = []
        for post in posts:
            md = post.get('content_markdown', '')
            if not md:
                continue
            
            date = None
            if post.get('published_at'):
                try:
                    date = datetime.fromisoformat(str(post['published_at']).replace('Z', '+00:00')).date()
                except:
                    pass
            
            tags = json.loads(post.get('tags', '[]')) if isinstance(post.get('tags'), str) else post.get('tags', [])
            
            lines = md.split('\n')
            snippet = ''
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('---'):
                    snippet = line[:200]
                    break
            
            entries.append({
                'id': post['id'],
                'title': post.get('title', 'Untitled'),
                'date': date.isoformat() if date else None,
                'layers': ['Uncategorized'],
                'snippet': snippet,
                'url': post.get('url', ''),
                'author': post.get('author', 'Unknown'),
                'reading_time': post.get('reading_time', 0),
                'is_technical': post.get('is_technical', False),
                'technical_score': post.get('technical_score', 0.0),
                'code_blocks': post.get('code_blocks', 0),
            })
    
    entries.sort(key=lambda e: (e['date'] is None, e['date'] or '9999-12-31'))
    
    per_layer = {}
    for e in entries:
        for layer in e.get('layers', []):
            per_layer.setdefault(layer, []).append(e)
    
    # Count tech stack and patterns
    all_techs = []
    all_patterns = []
    for e in entries:
        all_techs.extend([t['name'] for t in e.get('tech_stack', [])])
        all_patterns.extend([p['pattern'] for p in e.get('patterns', [])])
    
    from collections import Counter
    tech_counter = Counter(all_techs)
    pattern_counter = Counter(all_patterns)
    
    timeline = {
        'count': len(entries),
        'publication': posts[0]['publication'] if posts else source,
        'source': source,
        'posts': entries,
        'per_layer': per_layer,
        'stats': {
            'total_posts': len(entries),
            'technical_posts': sum(1 for e in entries if e.get('is_technical')),
            'layers': {layer: len(items) for layer, items in per_layer.items()},
            'ml_discovery': {
                'method': 'clustering + ner + zero-shot',
                'n_topics': len(per_layer),
                'total_tech_mentions': len(all_techs),
                'unique_technologies': len(tech_counter),
                'total_patterns': len(all_patterns),
                'unique_patterns': len(pattern_counter),
                'posts_with_solutions': sum(1 for e in entries if e.get('solutions')),
            },
            'top_technologies': [{'name': k, 'count': v} for k, v in tech_counter.most_common(10)],
            'top_patterns': [{'pattern': k, 'count': v} for k, v in pattern_counter.most_common(10)],
        }
    }
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    json_file = output_dir / f"{source}_timeline.json"
    md_file = output_dir / f"{source}_timeline.md"
    
    json_file.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding='utf-8')
    
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
    console.print("[bold green]🎉 Complete![/bold green]")
    console.print()
    
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    with db._get_connection() as conn:
        cursor = conn.cursor()
        total = cursor.execute("SELECT COUNT(*) FROM posts WHERE source = ?", (source,)).fetchone()[0]
        with_content = cursor.execute("SELECT COUNT(*) FROM posts WHERE source = ? AND content_markdown IS NOT NULL", (source,)).fetchone()[0]
    
    table.add_row("Total Posts", str(total))
    table.add_row("With Content", f"{with_content} ({with_content*100//total if total else 0}%)")
    table.add_row("In Timeline", str(len(entries)))
    
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
    if 'top_technologies' in timeline['stats'] and timeline['stats']['top_technologies']:
        console.print("[bold]🔧 Top Technologies:[/bold]")
        for tech in timeline['stats']['top_technologies'][:5]:
            console.print(f"   • {tech['name']}: [green]{tech['count']}[/green]")
        console.print()
    
    # Show top patterns
    if 'top_patterns' in timeline['stats'] and timeline['stats']['top_patterns']:
        console.print("[bold]🏗️  Top Patterns:[/bold]")
        for pattern in timeline['stats']['top_patterns'][:5]:
            console.print(f"   • {pattern['pattern']}: [green]{pattern['count']}[/green]")
        console.print()


@cli.command('reprocess-ml')
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
    from ..infrastructure.pipeline_db import PipelineDB
    from datetime import datetime, timezone
    from pathlib import Path
    
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
            # Import ML modules
            import sys
            ml_path = Path(__file__).parent.parent / 'ml_classifier'
            sys.path.insert(0, str(ml_path))
            
            from discover_enriched import (
                load_embedder, load_ner_pipeline, load_qa_pipeline,
                extract_tech_stack, extract_patterns, extract_solutions,
                extract_problem, extract_approach
            )
            import numpy as np
            import time
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Load models (once per source)
            console.print("[dim]Loading models...[/dim]")
            start_time = time.time()
            embedder = load_embedder()
            ner_pipeline = load_ner_pipeline()
            qa_pipeline = load_qa_pipeline()
            load_time = time.time() - start_time
            console.print(f"[dim]✓ Models loaded in {load_time:.1f}s[/dim]")
            console.print()
            
            texts = [e['content'] for e in entries_for_ml]
            total_posts = len(entries_for_ml)
            
            # 1. Clustering for layers
            console.print("[cyan]📊 Step 1/6: Clustering for topics...[/cyan]")
            step_start = time.time()
            embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=32)
            
            n_clusters = min(8, total_posts)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            cluster_info = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
                top_indices = cluster_tfidf.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                label = keywords[0].replace('_', ' ').title()
                cluster_info[cluster_id] = {'label': label}
            
            for i, entry in enumerate(entries_for_ml):
                entry['layers'] = [cluster_info[cluster_labels[i]]['label']]
            
            step_time = time.time() - step_start
            console.print(f"[green]✓ Discovered {n_clusters} topics in {step_time:.1f}s[/green]")
            console.print()
            
            # 2-6. Process all posts with ONE combined progress bar
            console.print("[cyan]🤖 Processing ML extractions...[/cyan]")
            console.print()
            
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
            
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:
                
                # Task for each extraction type
                task_tech = progress.add_task("[cyan]2/6 Tech Stack      ", total=total_posts)
                task_patterns = progress.add_task("[cyan]3/6 Patterns        ", total=total_posts)
                task_solutions = progress.add_task("[cyan]4/6 Solutions       ", total=total_posts)
                task_problems = progress.add_task("[cyan]5/6 Problems        ", total=total_posts)
                task_approaches = progress.add_task("[cyan]6/6 Approaches      ", total=total_posts)
                
                # Extract Tech Stack
                for entry in entries_for_ml:
                    tech_stack = extract_tech_stack(entry['content'], ner_pipeline)
                    entry['tech_stack'] = tech_stack
                    progress.update(task_tech, advance=1)
                
                # Extract Patterns
                for entry in entries_for_ml:
                    patterns = extract_patterns(entry['content'], ner_pipeline, embedder)
                    entry['patterns'] = patterns
                    progress.update(task_patterns, advance=1)
                
                # Extract Solutions
                for entry in entries_for_ml:
                    tech_stack = entry.get('tech_stack', [])
                    solutions = extract_solutions(entry['content'], tech_stack, embedder)
                    entry['solutions'] = solutions
                    progress.update(task_solutions, advance=1)
                
                # Extract Problems
                for entry in entries_for_ml:
                    problem = extract_problem(entry['content'], qa_pipeline)
                    entry['problem'] = problem
                    progress.update(task_problems, advance=1)
                
                # Extract Approaches
                for entry in entries_for_ml:
                    approach = extract_approach(entry['content'], qa_pipeline)
                    entry['approach'] = approach
                    progress.update(task_approaches, advance=1)
            
            # Summary stats
            total_techs = sum(len(e.get('tech_stack', [])) for e in entries_for_ml)
            total_patterns = sum(len(e.get('patterns', [])) for e in entries_for_ml)
            total_solutions = sum(len(e.get('solutions', [])) for e in entries_for_ml)
            posts_with_problem = sum(1 for e in entries_for_ml if e.get('problem'))
            posts_with_approach = sum(1 for e in entries_for_ml if e.get('approach'))
            
            console.print()
            console.print(f"[green]✓ Extracted:[/green]")
            console.print(f"  • {total_techs} tech stack items")
            console.print(f"  • {total_patterns} patterns")
            console.print(f"  • {total_solutions} solutions")
            console.print(f"  • {posts_with_problem} problems")
            console.print(f"  • {posts_with_approach} approaches")
            console.print()
            
            # Save to database with progress
            console.print("[cyan]💾 Saving ML data to database...[/cyan]")
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task("[cyan]Saving to DB", total=total_posts)
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


if __name__ == "__main__":
    cli()
