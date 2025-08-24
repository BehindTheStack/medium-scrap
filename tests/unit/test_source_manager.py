"""
Unit tests for SourceConfigManager (YAML configuration)
"""

import pytest
from src.infrastructure.config.source_manager import SourceConfigManager


class TestSourceConfigManager:
    """Test YAML configuration management"""
    
    def test_source_manager_initialization(self):
        """Test that source manager initializes correctly"""
        manager = SourceConfigManager()
        sources = manager.list_sources()
        
        # Should have sources
        assert len(sources) > 0
    
    def test_netflix_yaml_config(self):
        """Test Netflix YAML config (we know this works)"""
        manager = SourceConfigManager()
        netflix = manager.get_source("netflix")
        
        assert netflix is not None
        assert netflix.type == "publication"
        assert netflix.name == "netflix"
        assert netflix.custom_domain == True
        assert netflix.auto_discover == True
    
    def test_skyscanner_yaml_config(self):
        """Test Skyscanner YAML config (we know this works)"""
        manager = SourceConfigManager()
        skyscanner = manager.get_source("skyscanner")
        
        assert skyscanner is not None
        assert skyscanner.type == "username"
        assert skyscanner.name == "@SkyscannerEng"
        assert skyscanner.auto_discover == True
    
    def test_all_yaml_sources_valid(self):
        """Test that all YAML sources have valid structure"""
        manager = SourceConfigManager()
        sources = manager.list_sources()  # Returns Dict[str, SourceConfig]
        
        assert len(sources) >= 5  # Should have multiple sources
        
        for source_name, config in sources.items():
            assert config is not None
            assert config.type in ["username", "publication"]
            assert isinstance(config.auto_discover, bool)
            assert len(config.description) > 0
            
            # Validate specific fields based on type
            if config.type == "username":
                assert config.name.startswith("@")
            elif config.type == "publication":
                assert isinstance(config.custom_domain, bool)
    
    def test_bulk_collections_structure(self):
        """Test bulk collection structure"""
        manager = SourceConfigManager()
        bulk_collections = manager.list_bulk_collections()  # Correct method name
        
        assert len(bulk_collections) > 0
        
        for collection_name, collection in bulk_collections.items():
            assert len(collection.sources) > 0
            assert len(collection.description) > 0
            
            # Verify all sources in bulk collection exist
            for source_name in collection.sources:
                source = manager.get_source(source_name)
                assert source is not None
    
    def test_empty_source_handling(self):
        """Test handling of empty/invalid source names"""
        manager = SourceConfigManager()
        
        # Test nonexistent source - should raise KeyError
        with pytest.raises(KeyError):
            manager.get_source("nonexistent_source")
