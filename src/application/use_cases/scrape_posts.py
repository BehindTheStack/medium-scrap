"""
Application Use Cases - Orchestration Layer
Following Use Case pattern and Clean Architecture principles
"""

from dataclasses import dataclass
from typing import List, Optional
from ...domain.entities.publication import Post, PublicationConfig
from ...domain.services.publication_service import PostDiscoveryService, PublicationConfigService
from ...domain.repositories.base import PostRepository, SessionRepository


@dataclass
class ScrapePostsRequest:
    """Request DTO for post scraping use case"""
    publication_name: str
    limit: Optional[int] = None
    custom_post_ids: Optional[List[str]] = None
    auto_discover: bool = False
    skip_session: bool = False
    mode: str = 'metadata'


@dataclass
class ScrapePostsResponse:
    """Response DTO for post scraping use case"""
    posts: List[Post]
    publication_config: PublicationConfig
    total_posts_found: int
    discovery_method: str
    
    @property
    def success(self) -> bool:
        return len(self.posts) > 0


class ScrapePostsUseCase:
    """
    Primary Use Case for scraping Medium posts
    Orchestrates domain services and repositories
    """
    
    def __init__(
        self,
        post_discovery_service: PostDiscoveryService,
        publication_config_service: PublicationConfigService,
        session_repository: SessionRepository
    ):
        self._post_discovery_service = post_discovery_service
        self._publication_config_service = publication_config_service
        self._session_repository = session_repository
    
    def execute(self, request: ScrapePostsRequest, progress_callback=None) -> ScrapePostsResponse:
        """
        Execute the post scraping use case
        Follows Command pattern for request handling
        """
        try:
            # 1. Resolve publication configuration
            config = self._publication_config_service.resolve_publication_config(
                request.publication_name
            )
            
            # 2. Initialize session if needed
            discovery_method = "unknown"
            if not request.skip_session:
                self._session_repository.initialize_session(config)
            
            # 3. Handle custom post IDs
            if request.custom_post_ids:
                posts = self._handle_custom_ids(request.custom_post_ids, config)
                discovery_method = "custom_ids"
            else:
                # 4. Use intelligent post discovery
                posts = self._post_discovery_service.discover_posts_intelligently(
                    config=config,
                    limit=request.limit,
                    prefer_auto_discovery=request.auto_discover,
                    progress_callback=progress_callback
                )
                discovery_method = self._determine_discovery_method(config, request.auto_discover)

            # If full content is requested, enrich posts with HTML
            if request.mode in ('full', 'technical') and posts:
                try:
                    posts = self._post_discovery_service.enrich_posts_with_html(posts, config, progress_callback=progress_callback)
                except Exception:
                    # Ignore enrichment failures
                    pass
            
            return ScrapePostsResponse(
                posts=posts,
                publication_config=config,
                total_posts_found=len(posts),
                discovery_method=discovery_method
            )
            
        except Exception as e:
            # Return empty response on error - could implement proper error handling
            return ScrapePostsResponse(
                posts=[],
                publication_config=config if 'config' in locals() else None,
                total_posts_found=0,
                discovery_method="error",
            )
    
    def _handle_custom_ids(self, custom_ids: List[str], config: PublicationConfig) -> List[Post]:
        """Handle custom post IDs provided by user"""
        from ..domain.entities.publication import PostId
        from ..infrastructure.adapters.medium_api_adapter import MediumApiAdapter
        
        # Convert string IDs to PostId objects
        post_ids = [PostId(id_str) for id_str in custom_ids]
        
        # Use repository to get posts
        # Note: In a real implementation, we'd inject this properly
        adapter = MediumApiAdapter()
        return adapter.get_posts_by_ids(post_ids, config)
    
    def _determine_discovery_method(self, config: PublicationConfig, auto_discover: bool) -> str:
        """Determine which discovery method was likely used"""
        if auto_discover:
            return "auto_discovery"
        elif config.has_known_posts:
            return "known_ids_fallback"
        else:
            return "publication_feed"


@dataclass
class GetSupportedPublicationsRequest:
    """Request for getting supported publications"""
    pass


@dataclass
class GetSupportedPublicationsResponse:
    """Response with supported publications"""
    publications: List[PublicationConfig]
    total_count: int


class GetSupportedPublicationsUseCase:
    """Use Case for retrieving supported publications"""
    
    def __init__(self, publication_config_service: PublicationConfigService):
        self._publication_config_service = publication_config_service
    
    def execute(self, request: GetSupportedPublicationsRequest) -> GetSupportedPublicationsResponse:
        """Get all supported publications"""
        publications = self._publication_config_service.get_all_supported_publications()
        
        return GetSupportedPublicationsResponse(
            publications=publications,
            total_count=len(publications)
        )
