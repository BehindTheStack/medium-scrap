"""
Unit Tests for Domain Entities
Following AAA pattern (Arrange, Act, Assert)
"""

import pytest
from datetime import datetime

from src.domain.entities.publication import (
    Post, PostId, Author, PublicationConfig, PublicationId, PublicationType
)


class TestPostId:
    """Test PostId value object"""
    
    def test_valid_post_id_creation(self):
        # Arrange & Act
        post_id = PostId("ac15cada49ef")
        
        # Assert
        assert post_id.value == "ac15cada49ef"
    
    def test_invalid_post_id_length(self):
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="Post ID must be exactly 12 characters"):
            PostId("invalid")
    
    def test_invalid_post_id_non_alphanumeric(self):
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="Post ID must be exactly 12 characters"):
            PostId("invalid-id!")


class TestAuthor:
    """Test Author value object"""
    
    def test_valid_author_creation(self):
        # Arrange & Act
        author = Author(id="123", name="John Doe", username="johndoe")
        
        # Assert
        assert author.id == "123"
        assert author.name == "John Doe"
        assert author.username == "johndoe"
    
    def test_invalid_author_empty_fields(self):
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="All author fields are required"):
            Author(id="", name="John", username="john")


class TestPost:
    """Test Post entity"""
    
    def test_valid_post_creation(self):
        # Arrange
        post_id = PostId("ac15cada49ef")
        author = Author(id="123", name="John Doe", username="johndoe")
        published_at = datetime.now()
        
        # Act
        post = Post(
            id=post_id,
            title="Test Post",
            slug="test-post",
            author=author,
            published_at=published_at,
            reading_time=5.5
        )
        
        # Assert
        assert post.id == post_id
        assert post.title == "Test Post"
        assert post.reading_time == 5.5
        assert not post.is_recently_updated
    
    def test_post_recently_updated(self):
        # Arrange
        post_id = PostId("ac15cada49ef")
        author = Author(id="123", name="John Doe", username="johndoe")
        published_at = datetime(2023, 1, 1)
        latest_at = datetime(2023, 1, 2)
        
        # Act
        post = Post(
            id=post_id,
            title="Test Post",
            slug="test-post",
            author=author,
            published_at=published_at,
            reading_time=5.5,
            latest_published_at=latest_at
        )
        
        # Assert
        assert post.is_recently_updated
    
    def test_post_to_dict(self):
        # Arrange
        post_id = PostId("ac15cada49ef")
        author = Author(id="123", name="John Doe", username="johndoe")
        published_at = datetime(2023, 1, 1, 12, 0, 0)
        
        post = Post(
            id=post_id,
            title="Test Post",
            slug="test-post",
            author=author,
            published_at=published_at,
            reading_time=5.5,
            subtitle="Test subtitle"
        )
        
        # Act
        result = post.to_dict()
        
        # Assert
        assert result["id"] == "ac15cada49ef"
        assert result["title"] == "Test Post"
        assert result["author"]["name"] == "John Doe"
        assert result["reading_time"] == 5.5


class TestPublicationConfig:
    """Test PublicationConfig entity"""
    
    def test_valid_custom_domain_publication(self):
        # Arrange
        pub_id = PublicationId("netflix")
        post_ids = [PostId("ac15cada49ef")]
        
        # Act
        config = PublicationConfig(
            id=pub_id,
            name="Netflix Tech Blog",
            type=PublicationType.CUSTOM_DOMAIN,
            domain="netflixtechblog.com",
            graphql_url="https://netflixtechblog.com/_/graphql",
            known_post_ids=post_ids
        )
        
        # Assert
        assert config.is_custom_domain
        assert config.has_known_posts
        assert config.name == "Netflix Tech Blog"
    
    def test_valid_medium_hosted_publication(self):
        # Arrange
        pub_id = PublicationId("pinterest")
        
        # Act
        config = PublicationConfig(
            id=pub_id,
            name="Pinterest Engineering",
            type=PublicationType.MEDIUM_HOSTED,
            domain="medium.com",
            graphql_url="https://medium.com/_/graphql",
            known_post_ids=[]
        )
        
        # Assert
        assert not config.is_custom_domain
        assert not config.has_known_posts
    
    def test_invalid_graphql_url(self):
        # Arrange
        pub_id = PublicationId("test")
        
        # Act & Assert
        with pytest.raises(ValueError, match="GraphQL URL must be a valid URL"):
            PublicationConfig(
                id=pub_id,
                name="Test",
                type=PublicationType.CUSTOM_DOMAIN,
                domain="test.com",
                graphql_url="invalid-url",
                known_post_ids=[]
            )
