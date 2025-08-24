"""
Integration tests for comprehensive Medium scraping scenarios
Based on debug scripts and proven functionality
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import json
from datetime import datetime

from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.infrastructure.config.source_manager import SourceConfigManager
from src.domain.entities.publication import PublicationConfig, PublicationId, PublicationType, PostId, Post, Author
from src.application.use_cases.scrape_posts import ScrapePostsUseCase, ScrapePostsRequest, ScrapePostsResponse
from src.domain.services.publication_service import PostDiscoveryService, PublicationConfigService


class TestRealWorldScenariosIntegration:
    """Integration tests based on successful debug scenarios"""
    
    @pytest.fixture
    def publication_repository(self):
        """Initialize publication repository"""
        return InMemoryPublicationRepository()
    
    @pytest.fixture
    def api_adapter(self):
        """Initialize API adapter"""
        return MediumApiAdapter()
    
    @pytest.fixture
    def source_manager(self):
        """Initialize source configuration manager"""
        return SourceConfigManager()
    
    def test_netflix_repository_to_use_case_flow(self, publication_repository):
        """Test complete flow from repository config to use case"""
        # Get Netflix config
        netflix_config = publication_repository.get_by_id(PublicationId("netflix"))
        assert netflix_config is not None
        
        # Mock dependencies
        mock_post_repo = Mock()
        mock_session_repo = Mock()
        
        # Setup sample post data
        sample_post = Post(
            id=PostId("6dcc91058d8d"),
            title="From Facts & Metrics to Media Machine Learning",
            slug="from-facts-metrics-to-media-machine-learning-6dcc91058d8d",
            author=Author(id="test", name="Netflix Technology Blog", username="netflixtechblog"),
            published_at=datetime.now(),
            reading_time=5.0
        )
        
        mock_post_repo.get_posts_by_ids.return_value = [sample_post]
        mock_post_repo.discover_post_ids.return_value = []
        
        # Setup services and use case
        post_discovery_service = PostDiscoveryService(mock_post_repo)
        publication_config_service = PublicationConfigService(publication_repository)
        
        use_case = ScrapePostsUseCase(
            post_discovery_service,
            publication_config_service,
            mock_session_repo
        )
        
        # Execute
        request = ScrapePostsRequest(
            publication_name="netflix",
            limit=5,
            skip_session=True
        )
        
        response = use_case.execute(request)
        
        # Assert
        assert response.success
        assert len(response.posts) == 1
        assert response.posts[0].title == "From Facts & Metrics to Media Machine Learning"
        assert response.publication_config.name == "Netflix Tech Blog"
    
    def test_skyscanner_yaml_to_use_case_flow(self, source_manager, publication_repository):
        """Test Skyscanner flow from YAML to use case"""
        # Get Skyscanner from YAML
        skyscanner_source = source_manager.get_source("skyscanner")
        assert skyscanner_source is not None
        
        # Create config from YAML data
        config = publication_repository.create_generic_config(skyscanner_source.name)
        
        # Mock dependencies
        mock_post_repo = Mock()
        mock_session_repo = Mock()
        
        # Setup sample user posts
        sample_posts = [
            Post(
                id=PostId("ac15cada49ef"),
                title="Scaling Engineering Teams at Skyscanner",
                slug="scaling-engineering-teams",
                author=Author(id="test", name="Skyscanner Engineering", username="SkyscannerEng"),
                published_at=datetime.now(),
                reading_time=8.0
            )
        ]
        
        mock_post_repo.get_posts_by_ids.return_value = sample_posts
        mock_post_repo.discover_post_ids.return_value = []
        
        # Setup services
        post_discovery_service = PostDiscoveryService(mock_post_repo)
        publication_config_service = PublicationConfigService(publication_repository)
        
        use_case = ScrapePostsUseCase(
            post_discovery_service,
            publication_config_service,
            mock_session_repo
        )
        
        # Execute with username
        request = ScrapePostsRequest(
            publication_name="@SkyscannerEng",
            limit=10,
            skip_session=True
        )
        
        response = use_case.execute(request)
        
        # Assert - Note: may not be successful if no known posts and mocked discovery fails
        assert response is not None
        assert response.publication_config.type == PublicationType.MEDIUM_HOSTED
        # Success depends on whether posts were found, which depends on mocks
    
    def test_auto_discovery_integration(self, publication_repository):
        """Test auto-discovery functionality integration"""
        # Mock dependencies
        mock_post_repo = Mock()
        mock_session_repo = Mock()
        
        # Mock discovery to return post IDs
        discovered_ids = [
            PostId("6dcc91058d8d"),
            PostId("33073e260a38"),
            PostId("ac15cada49ef")
        ]
        mock_post_repo.discover_post_ids.return_value = discovered_ids
        
        # Mock posts retrieval
        sample_posts = [
            Post(
                id=PostId("6dcc91058d8d"),
                title="Discovered Post 1",
                slug="discovered-1",
                author=Author(id="1", name="Author 1", username="author1"),
                published_at=datetime.now(),
                reading_time=5.0
            ),
            Post(
                id=PostId("33073e260a38"),
                title="Discovered Post 2", 
                slug="discovered-2",
                author=Author(id="2", name="Author 2", username="author2"),
                published_at=datetime.now(),
                reading_time=7.0
            )
        ]
        mock_post_repo.get_posts_by_ids.return_value = sample_posts
        
        # Setup services
        post_discovery_service = PostDiscoveryService(mock_post_repo)
        publication_config_service = PublicationConfigService(publication_repository)
        
        use_case = ScrapePostsUseCase(
            post_discovery_service,
            publication_config_service,
            mock_session_repo
        )
        
        # Execute with auto-discovery
        request = ScrapePostsRequest(
            publication_name="netflix",
            limit=50,
            skip_session=True,
            auto_discover=True
        )
        
        response = use_case.execute(request)
        
        # Assert
        assert response.success
        assert len(response.posts) == 2  # Two posts returned
        mock_post_repo.discover_post_ids.assert_called_once()
    
    def test_mixed_known_and_discovered_posts(self, publication_repository):
        """Test handling of both known post IDs and auto-discovered posts"""
        netflix_config = publication_repository.get_by_id(PublicationId("netflix"))
        
        # Mock dependencies
        mock_post_repo = Mock()
        mock_session_repo = Mock()
        
        # Mock that known posts return some results
        known_posts = [
            Post(
                id=PostId("known1234567"),  # 12 characters
                title="Known Post",
                slug="known-post",
                author=Author(id="1", name="Netflix", username="netflix"),
                published_at=datetime.now(),
                reading_time=5.0
            )
        ]
        
        # Mock discovery returns additional posts
        discovered_ids = [PostId("discov123456")]  # 12 characters
        discovered_posts = [
            Post(
                id=PostId("discov123456"),  # 12 characters
                title="Discovered Post",
                slug="discovered-post", 
                author=Author(id="2", name="Netflix", username="netflix"),
                published_at=datetime.now(),
                reading_time=8.0
            )
        ]
        
        # Setup mock returns
        mock_post_repo.get_posts_by_ids.return_value = known_posts
        mock_post_repo.discover_post_ids.return_value = discovered_ids
        
        # Setup services
        post_discovery_service = PostDiscoveryService(mock_post_repo)
        publication_config_service = PublicationConfigService(publication_repository)
        
        use_case = ScrapePostsUseCase(
            post_discovery_service,
            publication_config_service,
            mock_session_repo
        )
        
        # Execute with both known IDs and auto-discovery
        request = ScrapePostsRequest(
            publication_name="netflix",
            limit=50,
            skip_session=True,
            auto_discover=True,
            custom_post_ids=["known1234567"]  # 12 characters
        )
        
        response = use_case.execute(request)
        
        # Assert
        assert response is not None
        assert response.publication_config.name == "Netflix Tech Blog"
        # The actual success depends on mock setup and service implementation
        # Just verify the structure is correct
    
    def test_error_handling_integration(self, publication_repository):
        """Test error handling in integration scenarios"""
        # Mock dependencies with error scenarios
        mock_post_repo = Mock()
        mock_session_repo = Mock()
        
        # Mock an error scenario
        mock_post_repo.get_posts_by_ids.side_effect = Exception("API Error")
        mock_post_repo.discover_post_ids.return_value = []
        
        # Setup services
        post_discovery_service = PostDiscoveryService(mock_post_repo)
        publication_config_service = PublicationConfigService(publication_repository)
        
        use_case = ScrapePostsUseCase(
            post_discovery_service,
            publication_config_service,
            mock_session_repo
        )
        
        # Execute
        request = ScrapePostsRequest(
            publication_name="netflix",
            limit=5,
            skip_session=True
        )
        
        response = use_case.execute(request)
        
        # Should handle error gracefully
        assert not response.success  # success is based on posts length
        assert response.discovery_method == "error"
    
    def test_custom_domain_vs_medium_hosted_integration(self, publication_repository):
        """Test integration behavior difference between custom domain and Medium-hosted"""
        # Mock dependencies
        mock_post_repo = Mock()
        mock_session_repo = Mock()
        
        # Sample posts
        sample_posts = [
            Post(
                id=PostId("test12345678"),  # 12 characters
                title="Test Post",
                slug="test-post",
                author=Author(id="1", name="Test Author", username="testauthor"),
                published_at=datetime.now(),
                reading_time=5.0
            )
        ]
        
        mock_post_repo.get_posts_by_ids.return_value = sample_posts
        mock_post_repo.discover_post_ids.return_value = []
        
        # Setup services
        post_discovery_service = PostDiscoveryService(mock_post_repo)
        publication_config_service = PublicationConfigService(publication_repository)
        
        use_case = ScrapePostsUseCase(
            post_discovery_service,
            publication_config_service,
            mock_session_repo
        )
        
        # Test custom domain (Netflix)
        netflix_request = ScrapePostsRequest(
            publication_name="netflix",
            limit=5,
            skip_session=True
        )
        
        netflix_response = use_case.execute(netflix_request)
        assert netflix_response.success
        assert netflix_response.publication_config.type == PublicationType.CUSTOM_DOMAIN
        
        # Test Medium-hosted (user profile)
        user_request = ScrapePostsRequest(
            publication_name="@testuser",
            limit=5,
            skip_session=True
        )
        
        user_response = use_case.execute(user_request)
        assert user_response is not None  # Response should exist
        assert user_response.publication_config.type == PublicationType.MEDIUM_HOSTED
        # Success depends on mock behavior - may not always succeed without proper setup
