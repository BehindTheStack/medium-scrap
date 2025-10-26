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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
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
                    mode=mode
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
                    mode=mode
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
        mode: str = 'metadata'
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
            self._handle_successful_response(response, format_type, output_file, mode)
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
        mode: str = 'metadata'
    ) -> None:
        """Handle successful scraping response"""
        self.console.print(f"✅ {response.total_posts_found} posts collected!")
        self.console.print(f"[dim]Discovery method: {response.discovery_method}[/dim]")
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

                        # Step 3: persist markdown, metadata and assets
                        progress.update(subtask, description="persist: write files & assets", current_post=f"Post {post.id.value} — persisting")
                        saved = persist_markdown_and_metadata(
                            post, md, assets, str(publication_dir), code_blocks=code_blocks, classifier=classification
                        )
                        progress.update(subtask, advance=1)

                        # Step 4: finalize/index (persistence already updates index but show step)
                        progress.update(subtask, description="finalize: index update", current_post=f"Post {post.id.value} — finalizing")
                        # No-op: persist_markdown_and_metadata already updated index; we keep the step for UX
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
                        persisted_messages.append(f"✅ Persisted post {post.id.value} -> {saved['markdown']}")

                    except Exception as e:
                        progress.update(main_task, advance=1, current_post=f"Failed {post.id.value}")
                        # ensure subtask removed
                        try:
                            progress.remove_task(subtask)
                        except Exception:
                            pass
                        persisted_messages.append(f"⚠️ Failed to persist post {post.id.value}: {e}")

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


if __name__ == "__main__":
    cli()
