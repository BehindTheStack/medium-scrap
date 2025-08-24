"""
Domain Repository Interfaces
Following Repository Pattern and Dependency Inversion Principle
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.publication import Post, PostId, PublicationConfig, PublicationId


class PublicationRepository(ABC):
    """
    Repository interface for publication data access
    Abstracts data source implementation
    """
    
    @abstractmethod
    def get_by_id(self, publication_id: PublicationId) -> Optional[PublicationConfig]:
        """Get publication configuration by ID"""
        pass
    
    @abstractmethod
    def get_all_known_publications(self) -> List[PublicationConfig]:
        """Get all pre-configured publications"""
        pass
    
    @abstractmethod
    def create_generic_config(self, publication_name: str) -> PublicationConfig:
        """Create generic configuration for unknown publications"""
        pass


class PostRepository(ABC):
    """
    Repository interface for post data access
    Handles post retrieval from various sources
    """
    
    @abstractmethod
    def get_posts_by_ids(self, post_ids: List[PostId], config: PublicationConfig) -> List[Post]:
        """Retrieve posts by their IDs"""
        pass
    
    @abstractmethod
    def discover_post_ids(self, config: PublicationConfig, limit: Optional[int] = 25) -> List[PostId]:
        """Automatically discover post IDs from publication"""
        pass
    
    @abstractmethod
    def get_posts_from_publication_feed(self, config: PublicationConfig, limit: Optional[int] = 25) -> List[Post]:
        """Get posts directly from publication feed"""
        pass


class SessionRepository(ABC):
    """
    Repository interface for session management
    Handles authentication and session state
    """
    
    @abstractmethod
    def initialize_session(self, config: PublicationConfig) -> bool:
        """Initialize session for publication access"""
        pass
    
    @abstractmethod
    def is_session_valid(self) -> bool:
        """Check if current session is valid"""
        pass
