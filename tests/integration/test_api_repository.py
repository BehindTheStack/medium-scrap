"""
Integration tests for API Adapter with Repository
"""

import pytest
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.domain.entities.publication import PublicationId


class TestApiRepositoryIntegration:
    """Test API adapter integration with repository configurations"""
    
    def test_netflix_api_configuration(self):
        """Test API adapter with Netflix configuration"""
        adapter = MediumApiAdapter()
        repo = InMemoryPublicationRepository()
        
        # Get Netflix config from repository
        netflix_config = repo.get_by_id(PublicationId("netflix"))
        assert netflix_config is not None
        
        # Test API adapter can handle this config
        headers = adapter._get_headers_for_config(netflix_config)
        assert "user-agent" in headers
        assert "content-type" in headers
        
        # Test query building for custom domain
        query = adapter._build_publication_query("netflix", limit=10, cursor=None)
        assert query["operationName"] == "PublicationPostsQuery"
        assert query["variables"]["publicationId"] == "netflix"
    
    def test_user_profile_api_configuration(self):
        """Test API adapter with user profile configuration"""
        adapter = MediumApiAdapter()
        repo = InMemoryPublicationRepository()
        
        # Create user config
        user_config = repo.create_generic_config("@SkyscannerEng")
        assert user_config is not None
        
        # Test API adapter can handle user config
        headers = adapter._get_headers_for_config(user_config)
        assert "user-agent" in headers
        assert "content-type" in headers
        
        # Test query building for user profile
        query = adapter._build_publication_query("@SkyscannerEng", limit=10, cursor=None)
        assert query["operationName"] == "UserProfileQuery"
        assert query["variables"]["username"] == "SkyscannerEng"  # @ removed
    
    def test_custom_domain_api_configuration(self):
        """Test API adapter with custom domain configuration"""
        adapter = MediumApiAdapter()
        repo = InMemoryPublicationRepository()
        
        # Create custom domain config
        custom_config = repo.create_generic_config("example.com")
        assert custom_config is not None
        
        # Test API adapter can handle custom domain
        headers = adapter._get_headers_for_config(custom_config)
        assert "user-agent" in headers
        assert "content-type" in headers
        
        # Custom domain should use publication query
        query = adapter._build_publication_query("example.com", limit=10, cursor=None)
        assert query["operationName"] == "PublicationPostsQuery"
