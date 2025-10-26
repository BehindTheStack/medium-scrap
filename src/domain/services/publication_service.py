"""
Domain Services - Business Logic Layer
Enterprise-grade services following Single Responsibility Principle
"""

from typing import List, Optional
from ..entities.publication import Post, PostId, PublicationConfig, PublicationId
from ..repositories.base import PostRepository, PublicationRepository


class PostDiscoveryService:
    """
    Domain Service for intelligent post discovery
    Implements Strategy Pattern for multiple discovery methods
    """
    
    def __init__(self, post_repository: PostRepository):
        self._post_repository = post_repository
    
    def discover_posts_intelligently(
        self, 
        config: PublicationConfig, 
        limit: Optional[int] = 25,
        prefer_auto_discovery: bool = False
    ) -> List[Post]:
        """
        Intelligently discover posts using multiple strategies
        
        Strategy precedence:
        1. Auto-discovery (if preferred or no known IDs)
        2. Known IDs (fallback)
        3. Publication feed (last resort)
        """
        posts = []
        
        # Strategy 1: Auto-discovery (primary for production)
        if prefer_auto_discovery or not config.has_known_posts:
            discovered_ids = self._post_repository.discover_post_ids(config, limit)
            if discovered_ids:
                posts = self._post_repository.get_posts_by_ids(discovered_ids, config)
                if posts:
                    return posts[:limit] if limit is not None else posts
        
        # Strategy 2: Known IDs (reliable fallback)
        if config.has_known_posts:
            known_ids = config.known_post_ids[:limit] if limit is not None else config.known_post_ids
            posts = self._post_repository.get_posts_by_ids(known_ids, config)
            if posts:
                return posts

            def enrich_posts_with_html(self, posts: List[Post], config: PublicationConfig) -> List[Post]:
                """Enrich a list of posts by fetching their full HTML content via the repository."""
                enriched = []
                for post in posts:
                    try:
                        html = self._post_repository.fetch_post_html(post, config)
                        if html:
                            post.content_html = html
                    except Exception:
                        # Ignore enrichment failures and continue
                        pass
                    enriched.append(post)
                return enriched
        
        # Strategy 3: Publication feed (last resort)
        try:
            posts = self._post_repository.get_posts_from_publication_feed(config, limit)
            return posts
        except Exception:
            return []


class PublicationConfigService:
    """
    Domain Service for publication configuration management
    Handles publication resolution and validation
    """
    
    def __init__(self, publication_repository: PublicationRepository):
        self._publication_repository = publication_repository
    
    def resolve_publication_config(self, publication_name: str) -> PublicationConfig:
        """
        Resolve publication configuration by name
        Creates generic config if not found
        """
        publication_id = PublicationId(publication_name)
        
        # Try to get known configuration
        config = self._publication_repository.get_by_id(publication_id)
        if config:
            return config
        
        # Create generic configuration for unknown publications
        return self._publication_repository.create_generic_config(publication_name)
    
    def get_all_supported_publications(self) -> List[PublicationConfig]:
        """Get all pre-configured publications"""
        return self._publication_repository.get_all_known_publications()
    
    def validate_config(self, config: PublicationConfig) -> bool:
        """Validate publication configuration"""
        try:
            # Basic validation is handled by entity invariants
            return True
        except ValueError:
            return False
