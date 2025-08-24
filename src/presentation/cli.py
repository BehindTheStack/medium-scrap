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
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..application.use_cases.scrape_posts import (
    ScrapePostsUseCase, ScrapePostsRequest, ScrapePostsResponse
)
from ..domain.entities.publication import Post
from ..infrastructure.config.source_manager import SourceConfigManager


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
        all_posts: bool = False
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
            
            # Show source info
            self.console.print(f"[bold green]📖 Using configured source:[/bold green] {source_key}")
            self.console.print(f"[dim]{source_config.description}[/dim]")
            self.console.print()
            
            # Execute scraping
            self.scrape_posts(
                publication=source_config.get_publication_name(),
                limit=None if all_posts else limit,
                format_type=format_type,
                custom_ids=None,
                auto_discover=source_config.auto_discover,
                skip_session=defaults.get('skip_session', True),
                output_file=output_file
            )
            
        except KeyError as e:
            self.console.print(f"[red]❌ {e}[/red]")
            self.console.print("[yellow]💡 Use --list-sources to see available sources[/yellow]")
    
    def scrape_bulk_collection(
        self,
        bulk_key: str,
        limit: Optional[int] = None,
        format_type: str = "json"
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
            
            for source_key in bulk_config.sources:
                self.console.print(f"[blue]📥 Collecting from:[/blue] {source_key}")
                self.scrape_from_config(
                    source_key=source_key,
                    limit=limit,
                    format_type=format_type,
                    output_file=str(output_dir / f"{source_key}_posts.json")
                )
                self.console.print()
                
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
        output_file: Optional[str] = None
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
        
        # Create request
        request = ScrapePostsRequest(
            publication_name=publication,
            limit=limit,
            custom_post_ids=custom_post_ids,
            auto_discover=auto_discover,
            skip_session=skip_session
        )
        
        # Execute use case with progress indicator
        response = self._execute_with_progress(request, skip_session)
        
        # Handle response
        if response.success:
            self._handle_successful_response(response, format_type, output_file)
        else:
            self._handle_failed_response(response)
    
    def _execute_with_progress(self, request: ScrapePostsRequest, skip_session: bool) -> ScrapePostsResponse:
        """Execute use case with progress indication"""
        if skip_session:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Collecting posts...", total=None)
                response = self._scrape_posts_use_case.execute(request)
                progress.update(task, description=f"Posts collected: {response.total_posts_found}")
            return response
        else:
            self.console.print(f"[blue]🔄 Initializing session...[/blue]")
            response = self._scrape_posts_use_case.execute(request)
            self.console.print()
            return response
    
    def _handle_successful_response(
        self, 
        response: ScrapePostsResponse, 
        format_type: str, 
        output_file: Optional[str]
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
@click.command()
@click.option('--publication', '-p', help='Publication name (netflix, pinterest, or any publication)')
@click.option('--source', '-s', help='Use configured source from YAML (e.g., --source netflix)')
@click.option('--bulk', '-b', help='Run bulk collection (e.g., --bulk tech_giants)')
@click.option('--list-sources', is_flag=True, help='List all available configured sources')
@click.option('--output', '-o', help='Output file for saving results')
@click.option('--format', '-f', 'format_type', type=click.Choice(['table', 'json', 'ids']), 
              default='table', help='Output format')
@click.option('--custom-ids', help='Comma-separated list of specific post IDs')
@click.option('--skip-session', is_flag=True, help='Skip session initialization (faster)')
@click.option('--limit', type=int, help='Maximum number of posts to collect')
@click.option('--all', 'all_posts', is_flag=True, help='Collect ALL posts from publication (no limit)')
@click.option('--auto-discover', is_flag=True, 
              help='Force auto-discovery mode (production ready)')
def cli(publication, source, bulk, list_sources, output, format_type, custom_ids, skip_session, limit, all_posts, auto_discover):
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
    
    # Handle source listing
    if list_sources:
        controller.list_sources()
        return
        
    # Handle bulk collection
    if bulk:
        controller.scrape_bulk_collection(
            bulk_key=bulk,
            limit=limit,
            format_type=format_type
        )
        return
    
    # Handle configured source
    if source:
        controller.scrape_from_config(
            source_key=source,
            limit=limit,
            format_type=format_type,
            output_file=output,
            all_posts=all_posts
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
        output_file=output
    )


if __name__ == "__main__":
    cli()
