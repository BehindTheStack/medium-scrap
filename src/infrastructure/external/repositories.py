"""
Concrete Repository Implementations
Data access layer with predefined configurations
"""

from typing import List, Optional, Dict
from ...domain.entities.publication import (
    PublicationConfig, PublicationId, PublicationType, PostId
)
from ...domain.repositories.base import PublicationRepository


class InMemoryPublicationRepository(PublicationRepository):
    """
    In-memory repository for publication configurations
    Contains predefined publication settings
    """
    
    def __init__(self):
        self._publications: Dict[str, PublicationConfig] = self._load_predefined_publications()
    
    def get_by_id(self, publication_id: PublicationId) -> Optional[PublicationConfig]:
        """Get publication configuration by ID"""
        return self._publications.get(publication_id.value)
    
    def get_all_known_publications(self) -> List[PublicationConfig]:
        """Get all pre-configured publications"""
        return list(self._publications.values())
    
    def create_generic_config(self, publication_name: str) -> PublicationConfig:
        """Create generic configuration for unknown publications"""
        # Detect if it's a custom domain or medium-hosted
        if publication_name.startswith('@'):
            # User profile - keep the @ in the ID for URL construction
            return PublicationConfig(
                id=PublicationId(publication_name),  # Keep the @ 
                name=publication_name[1:].title().replace('-', ' '),  # Remove @ from display name
                type=PublicationType.MEDIUM_HOSTED,
                domain="medium.com",
                graphql_url="https://medium.com/_/graphql",
                known_post_ids=[]
            )
        elif '.' in publication_name:
            # Assume custom domain
            domain = publication_name if publication_name.startswith(('http://', 'https://')) else f"{publication_name}.com"
            domain = domain.replace('https://', '').replace('http://', '')
            
            return PublicationConfig(
                id=PublicationId(publication_name),
                name=publication_name.title().replace('-', ' '),
                type=PublicationType.CUSTOM_DOMAIN,
                domain=domain,
                graphql_url=f"https://{domain}/_/graphql",
                known_post_ids=[]
            )
        else:
            # Assume medium-hosted publication
            return PublicationConfig(
                id=PublicationId(publication_name),
                name=publication_name.title().replace('-', ' '),
                type=PublicationType.MEDIUM_HOSTED,
                domain="medium.com",
                graphql_url="https://medium.com/_/graphql",
                known_post_ids=[]
            )
    
    def _load_predefined_publications(self) -> Dict[str, PublicationConfig]:
        """Load predefined publication configurations"""
        # Netflix Tech Blog
        netflix_ids = [
            "ac15cada49ef", "64c786c2a3ac", "422d6218fdf1", "3052540e231d", "ee13a06f9c78",
            "e735e6ce8f7d", "6b4d4410b88f", "8ebdda0b2db4", "fd78328ee0bb", "946b9b3cd300",
            "222ac5d23576", "da6805341642", "2d2e6b6d205d", "256629c9386b", "039d5efd115b",
            "6a727c1ae2e5", "cba6c7ed49df", "37f9f88c152d", "4e5e6310e359", "f326b0589102",
            "e9e0cb15f2ba", "9340b879176a", "4b0ce22a8a96", "b8ba072ddeeb"
        ]
        
        # Pinterest Engineering  
        pinterest_ids = [
            "b34ac9e3bdd9", "bef9af9dabf4", "39a36d5e82c4", "4038b9e837a0",
            "0248efe4fd52", "3253d2432a0c", "9e0b9d35a11f", "8a99e6c8e6b7", 
            "86dcc6d5fce9"
        ]
        
        netflix_config = PublicationConfig(
            id=PublicationId("netflix"),
            name="Netflix Tech Blog",
            type=PublicationType.CUSTOM_DOMAIN,
            domain="netflixtechblog.com",
            graphql_url="https://netflixtechblog.com/_/graphql",
            known_post_ids=[PostId(pid) for pid in netflix_ids]
        )
        
        pinterest_config = PublicationConfig(
            id=PublicationId("pinterest"),
            name="Pinterest Engineering",
            type=PublicationType.MEDIUM_HOSTED,
            domain="medium.com",
            graphql_url="https://medium.com/_/graphql",
            known_post_ids=[PostId(pid) for pid in pinterest_ids]
        )
        
        return {
            "netflix": netflix_config,
            "pinterest": pinterest_config
        }


class MediumSessionRepository:
    """Repository for managing Medium sessions"""
    
    def __init__(self):
        self._session_active = False
    
    def initialize_session(self, config: PublicationConfig) -> bool:
        """Initialize session for publication access"""
        try:
            # Session initialization logic would go here
            # For now, just simulate successful initialization
            self._session_active = True
            return True
        except Exception:
            return False
    
    def is_session_valid(self) -> bool:
        """Check if current session is valid"""
        return self._session_active
