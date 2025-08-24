"""
Integration tests for YAML ↔ Repository integration
"""

import pytest
from src.infrastructure.config.source_manager import SourceConfigManager
from src.infrastructure.external.repositories import InMemoryPublicationRepository
from src.domain.entities.publication import PublicationId, PublicationType


class TestYamlRepositoryIntegration:
    """Test integration between YAML configuration and repository"""
    
    def test_netflix_yaml_to_repository_integration(self):
        """Test Netflix is properly configured across systems (we know this works)"""
        # YAML side
        manager = SourceConfigManager()
        netflix_yaml = manager.get_source("netflix")
        
        # Repository side
        repo = InMemoryPublicationRepository()
        netflix_repo = repo.get_by_id(PublicationId("netflix"))
        
        # Both should exist and match
        assert netflix_yaml is not None
        assert netflix_repo is not None
        assert netflix_yaml.custom_domain == True
        assert netflix_repo.type == PublicationType.CUSTOM_DOMAIN
        assert netflix_repo.name == "Netflix Tech Blog"
        assert netflix_repo.domain == "netflixtechblog.com"
        assert netflix_repo.is_custom_domain == True
        assert len(netflix_repo.known_post_ids) > 0
    
    def test_skyscanner_yaml_to_repository_integration(self):
        """Test Skyscanner user profile integration"""
        # YAML side
        manager = SourceConfigManager()
        skyscanner_yaml = manager.get_source("skyscanner")
        
        # Repository side - create config for user
        repo = InMemoryPublicationRepository()
        skyscanner_repo = repo.create_generic_config("@SkyscannerEng")
        
        # YAML config should be username type
        assert skyscanner_yaml.type == "username"
        assert skyscanner_yaml.name == "@SkyscannerEng"
        
        # Repository config should be medium hosted
        assert skyscanner_repo.type == PublicationType.MEDIUM_HOSTED
        assert skyscanner_repo.domain == "medium.com"
    
    def test_yaml_to_repository_mapping_consistency(self):
        """Test that YAML configs map to repository configs correctly"""
        manager = SourceConfigManager()
        repo = InMemoryPublicationRepository()
        
        # Test all configured sources
        sources = manager.list_sources()
        
        for source_name, yaml_config in sources.items():
            if yaml_config.type == "publication":
                # Should exist in repository
                repo_config = repo.get_by_id(PublicationId(source_name))
                
                if repo_config is not None:  # Pre-configured sources
                    if yaml_config.custom_domain:
                        assert repo_config.type == PublicationType.CUSTOM_DOMAIN
                    else:
                        assert repo_config.type == PublicationType.MEDIUM_HOSTED
            
            elif yaml_config.type == "username":
                # Should be able to create generic config
                repo_config = repo.create_generic_config(yaml_config.name)
                assert repo_config.type == PublicationType.MEDIUM_HOSTED
                assert repo_config.domain == "medium.com"
