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
        prefer_auto_discovery: bool = False,
        progress_callback=None
    ) -> List[Post]:
        """
        Intelligently discover posts using multiple strategies.

        This method implements a small strategy chain:
        1. Auto-discovery (preferred when enabled or when no known IDs are present).
        2. Known IDs from configuration (fallback when available).
        3. Publication feed scraping (last resort).

        Parameters
        ----------
        config: PublicationConfig
            Publication configuration used to discover or fetch posts.
        limit: Optional[int]
            Maximum number of posts to return. None means no limit.
        prefer_auto_discovery: bool
            When True, prefer auto-discovery even if known IDs are present.
        progress_callback: Optional[callable]
            Optional callable that receives lightweight events during discovery
            and fetching. Used by the CLI to update live progress. Events are
            best-effort and should not raise exceptions.

        Returns
        -------
        List[Post]
            List of discovered or fetched Post entities.
        """
        posts = []
        
        # Strategy 1: Auto-discovery (primary for production)
        if prefer_auto_discovery or not config.has_known_posts:
            discovered_ids = self._post_repository.discover_post_ids(config, limit)
            if progress_callback:
                try:
                    progress_callback({'phase': 'discovered_ids', 'count': len(discovered_ids)})
                except Exception:
                    pass
            if discovered_ids:
                posts = self._post_repository.get_posts_by_ids(discovered_ids, config)
                if progress_callback:
                    try:
                        progress_callback({'phase': 'fetched_posts', 'count': len(posts)})
                    except Exception:
                        pass
                if posts:
                    return posts[:limit] if limit is not None else posts
        
        # Strategy 2: Known IDs (reliable fallback)
        if config.has_known_posts:
            known_ids = config.known_post_ids[:limit] if limit is not None else config.known_post_ids
            posts = self._post_repository.get_posts_by_ids(known_ids, config)
            if posts:
                return posts
            
        
        # Strategy 3: Publication feed (last resort)
        try:
            posts = self._post_repository.get_posts_from_publication_feed(config, limit)
            return posts
        except Exception:
            return []

    def enrich_posts_with_html(self, posts: List[Post], config: PublicationConfig, progress_callback=None) -> List[Post]:
        """Enrich a list of posts by fetching their full HTML content via the repository.

        Parameters
        ----------
        posts: List[Post]
            Posts to enrich (mutated in-place with .content_html when fetched).
        config: PublicationConfig
            Publication configuration used to fetch HTML content.
        progress_callback: Optional[callable]
            Optional callable invoked with {'phase': 'enriched_post', 'post_id': <id>} for
            each post successfully enriched. The callback should be non-blocking.

        Returns
        -------
        List[Post]
            The list of posts with content_html populated when available.
        """
        enriched = []
        for post in posts:
            try:
                html = self._post_repository.fetch_post_html(post, config)
                if html:
                    post.content_html = html
                    if progress_callback:
                        try:
                            progress_callback({'phase': 'enriched_post', 'post_id': post.id.value})
                        except Exception:
                            pass
            except Exception:
                # Ignore enrichment failures and continue
                pass
            enriched.append(post)
        return enriched


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
