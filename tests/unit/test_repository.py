"""
Unit tests for InMemoryPublicationRepository
"""

import pytest
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.domain.entities.publication import PublicationId, PublicationType


class TestPublicationRepository:
    """Test publication repository functionality"""
    
    def test_repository_initialization(self):
        """Test that repository basic functionality works"""
        repo = InMemoryPublicationRepository()
        
        # Test that Netflix config exists
        netflix = repo.get_by_id(PublicationId("netflix"))
        assert netflix is not None
        assert netflix.name == "Netflix Tech Blog"
        assert netflix.type == PublicationType.CUSTOM_DOMAIN
    
    def test_generic_config_user_creation(self):
        """Test user config generation (like @SkyscannerEng)"""
        repo = InMemoryPublicationRepository()
        
        # Test with @ prefix
        config = repo.create_generic_config("@SkyscannerEng")
        assert config.id.value == "@SkyscannerEng"
        assert config.type == PublicationType.MEDIUM_HOSTED
        assert config.domain == "medium.com"
        assert "medium.com" in config.graphql_url
    
    def test_generic_config_custom_domain(self):
        """Test custom domain detection (like Netflix)"""
        repo = InMemoryPublicationRepository()
        
        # Test domain detection
        config = repo.create_generic_config("netflixtechblog.com")
        assert config.type == PublicationType.CUSTOM_DOMAIN
        assert "netflixtechblog.com" in config.domain
        assert "netflixtechblog.com" in config.graphql_url
    
    def test_generic_config_medium_hosted(self):
        """Test medium-hosted publication detection"""
        repo = InMemoryPublicationRepository()
        
        # Test regular publication name
        config = repo.create_generic_config("some-publication")
        assert config.type == PublicationType.MEDIUM_HOSTED
        assert config.domain == "medium.com"
        assert "medium.com" in config.graphql_url
    
    def test_malformed_username_handling(self):
        """Test handling of malformed usernames"""
        repo = InMemoryPublicationRepository()
        
        # Test username without @
        config = repo.create_generic_config("UsernameWithoutAt")
        assert config.type == PublicationType.MEDIUM_HOSTED
        
        # Test with multiple @
        config = repo.create_generic_config("@@DoubleAt")
        assert config.id.value == "@@DoubleAt"
    
    def test_domain_prefix_handling(self):
        """Test domain with http/https prefixes"""
        repo = InMemoryPublicationRepository()
        
        # Test with https prefix
        config = repo.create_generic_config("https://example.com")
        assert "example.com" in config.domain
        assert config.type == PublicationType.CUSTOM_DOMAIN
        
        # Test with http prefix
        config = repo.create_generic_config("http://test.com")
        assert "test.com" in config.domain
        assert config.type == PublicationType.CUSTOM_DOMAIN
