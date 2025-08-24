"""
Unit tests for MediumApiAdapter components
"""

import pytest
from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.domain.entities.publication import PublicationId


class TestMediumApiAdapter:
    """Test API adapter functionality"""
    
    def test_api_adapter_initialization(self):
        """Test that API adapter can be initialized"""
        adapter = MediumApiAdapter()
        assert adapter is not None
    
    def test_api_adapter_properties(self):
        """Test that MediumApiAdapter initializes correctly"""
        adapter = MediumApiAdapter()
        
        assert adapter is not None
        assert hasattr(adapter, '_base_headers')
        assert "user-agent" in adapter._base_headers
        assert "content-type" in adapter._base_headers
    
    def test_user_query_structure(self):
        """Test user query building (based on debug_skyscanner findings)"""
        adapter = MediumApiAdapter()
        
        # Test user profile query building
        query = adapter._build_publication_query("@SkyscannerEng", limit=10, cursor=None)
        
        assert query["operationName"] == "UserProfileQuery"
        assert "username" in query["variables"]
        assert query["variables"]["username"] == "SkyscannerEng"  # @ removed
        assert "UserProfileQuery" in query["query"]
        assert "userResult" in query["query"]
        assert "homepagePostsConnection" in query["query"]
    
    def test_publication_query_structure(self):
        """Test publication query building"""
        adapter = MediumApiAdapter()
        
        # Test publication query building
        query = adapter._build_publication_query("netflix", limit=10, cursor=None)
        
        assert query["operationName"] == "PublicationPostsQuery" 
        assert "publicationId" in query["variables"]
        assert query["variables"]["publicationId"] == "netflix"
    
    def test_headers_generation(self):
        """Test header generation for different publication types"""
        adapter = MediumApiAdapter()
        
        # Netflix (custom domain)
        repo = InMemoryPublicationRepository()
        netflix_config = repo.get_by_id(PublicationId("netflix"))
        
        headers = adapter._get_headers_for_config(netflix_config)
        assert "user-agent" in headers
        assert "content-type" in headers
        assert headers["content-type"] == "application/json"
    
    def test_graphql_query_variables_validation(self):
        """Test GraphQL query variables validation"""
        adapter = MediumApiAdapter()
        
        # Test with valid publication
        query = adapter._build_publication_query("netflix", limit=50, cursor=None)
        
        assert "variables" in query
        assert query["variables"]["first"] == 50  # Uses 'first' not 'limit'
        assert query["variables"]["publicationId"] == "netflix"
        
        # Test cursor handling
        query_with_cursor = adapter._build_publication_query("netflix", limit=25, cursor="test_cursor")
        assert query_with_cursor["variables"]["after"] == "test_cursor"
    
    def test_user_profile_query_username_processing(self):
        """Test username processing for user profiles"""
        adapter = MediumApiAdapter()
        
        # Test that @ symbol is properly handled
        query = adapter._build_publication_query("@SkyscannerEng", limit=10, cursor=None)
        
        # Username should have @ removed
        assert query["variables"]["username"] == "SkyscannerEng"
        assert query["operationName"] == "UserProfileQuery"
    
    def test_post_id_extraction_structure(self):
        """Test that adapter has proper structure for post ID extraction"""
        adapter = MediumApiAdapter()
        
        # Test that adapter can be initialized and has expected methods
        assert hasattr(adapter, '_base_headers')
        assert isinstance(adapter._base_headers, dict)
        
        # Test expected GraphQL query structure
        query = adapter._build_publication_query("test", limit=10, cursor=None)
        assert "query" in query
        assert "variables" in query
        assert "operationName" in query
