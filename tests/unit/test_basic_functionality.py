"""
Basic unit tests for core components
"""

import pytest
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.domain.entities.publication import PublicationId, PublicationType


def test_repository_basic():
    """Test that repository basic functionality works"""
    repo = InMemoryPublicationRepository()
    
    # Test that Netflix config exists
    netflix = repo.get_by_id(PublicationId("netflix"))
    assert netflix is not None
    assert netflix.name == "Netflix Tech Blog"
    assert netflix.type == PublicationType.CUSTOM_DOMAIN
    
    # Test generic config creation
    config = repo.create_generic_config("@TestUser")
    assert config is not None
    assert config.id.value == "@TestUser"
    assert config.type == PublicationType.MEDIUM_HOSTED


def test_source_manager_basic():
    """Test that source manager basic functionality works"""
    from src.infrastructure.config.source_manager import SourceConfigManager
    
    manager = SourceConfigManager()
    sources = manager.list_sources()
    
    # Should have sources
    assert len(sources) > 0
    
    # Netflix should exist
    netflix = manager.get_source("netflix")
    assert netflix is not None
    assert netflix.custom_domain == True


def test_api_adapter_initialization():
    """Test that API adapter can be initialized"""
    from src.infrastructure.adapters.medium_api_adapter import MediumApiAdapter
    
    adapter = MediumApiAdapter()
    assert adapter is not None
