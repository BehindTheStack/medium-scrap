"""
Domain Entities - Core Business Objects
Enterprise-grade entities following SOLID principles
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from enum import Enum


class PublicationType(Enum):
    """Type of Medium publication hosting"""
    CUSTOM_DOMAIN = "custom_domain"        # e.g., netflixtechblog.com
    MEDIUM_HOSTED = "medium_hosted"        # e.g., medium.com/publication


@dataclass(frozen=True)
class PublicationId:
    """Value Object for Publication Identification"""
    value: str
    
    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("Publication ID must be a non-empty string")


@dataclass(frozen=True)
class PostId:
    """Value Object for Post Identification"""
    value: str
    
    def __post_init__(self):
        # Basic validation - trust the Medium API for format
        if not self.value:
            raise ValueError("Post ID cannot be empty")
        if not self.value.strip():
            raise ValueError("Post ID cannot be whitespace")


@dataclass(frozen=True)
class Author:
    """Value Object representing a post author"""
    id: str
    name: str
    username: str
    
    def __post_init__(self):
        if not all([self.id, self.name, self.username]):
            raise ValueError("All author fields are required")


@dataclass
class Post:
    """
    Core Post Entity - Aggregate Root
    Represents a Medium post with all its properties
    """
    id: PostId
    title: str
    slug: str
    author: Author
    published_at: datetime
    reading_time: float
    subtitle: Optional[str] = None
    content_html: Optional[str] = None
    latest_published_at: Optional[datetime] = None
    url: Optional[str] = None
    tags: Optional[List[str]] = None
    claps: Optional[int] = None
    
    def __post_init__(self):
        if not self.title:
            raise ValueError("Post title is required")
        if not self.slug:
            raise ValueError("Post slug is required")
        if self.reading_time < 0:
            raise ValueError("Reading time must be non-negative")
    
    @property
    def is_recently_updated(self) -> bool:
        """Check if post was updated after initial publication"""
        return (self.latest_published_at is not None and 
                self.latest_published_at > self.published_at)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id.value,
            "title": self.title,
            "slug": self.slug,
            "author": {
                "id": self.author.id,
                "name": self.author.name,
                "username": self.author.username
            },
            "published_at": self.published_at.isoformat(),
            "latest_published_at": self.latest_published_at.isoformat() if self.latest_published_at else None,
            "reading_time": self.reading_time,
            "subtitle": self.subtitle,
            "content_html": self.content_html
        }


@dataclass
class PublicationConfig:
    """
    Publication Configuration Entity
    Contains all metadata needed to scrape a publication
    """
    id: PublicationId
    name: str
    type: PublicationType
    domain: str
    graphql_url: str
    known_post_ids: List[PostId]
    
    def __post_init__(self):
        if not all([self.name, self.domain, self.graphql_url]):
            raise ValueError("All publication config fields are required")
        if not self.graphql_url.startswith(('http://', 'https://')):
            raise ValueError("GraphQL URL must be a valid URL")
    
    @property
    def has_known_posts(self) -> bool:
        """Check if publication has known post IDs"""
        return len(self.known_post_ids) > 0
    
    @property
    def is_custom_domain(self) -> bool:
        """Check if publication uses custom domain"""
        return self.type == PublicationType.CUSTOM_DOMAIN
