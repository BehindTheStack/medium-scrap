"""
Configuration manager for Medium sources.
Handles YAML-based source definitions and bulk operations.
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class SourceConfig:
    """Configuration for a Medium source."""
    name: str
    type: str  # 'publication' or 'username'
    description: str
    auto_discover: bool = True
    custom_domain: bool = False
    
    def get_publication_name(self) -> str:
        """Get the publication name formatted correctly."""
        if self.type == 'username' and not self.name.startswith('@'):
            return f"@{self.name}"
        return self.name


@dataclass 
class BulkConfig:
    """Configuration for bulk collection operations."""
    description: str
    sources: List[str]


class SourceConfigManager:
    """Manages Medium source configurations from YAML file."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("medium_sources.yaml")
        self._config_data = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._config_data is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
                
        return self._config_data

    def _save_config(self) -> None:
        """Persist the in-memory configuration back to the YAML file."""
        if self._config_data is None:
            # Nothing to save
            return

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self._config_data, f, sort_keys=False, allow_unicode=True)

    def add_or_update_source(self, source_key: str, source_data: Dict[str, Any]) -> None:
        """Add a new source or update an existing one and persist changes.

        source_data should contain keys: type, name, description, optional auto_discover, custom_domain
        """
        config = self._load_config() or {}

        if 'sources' not in config or config['sources'] is None:
            config['sources'] = {}

        # Normalize booleans and required fields
        entry = {
            'type': source_data.get('type', 'publication'),
            'name': source_data.get('name'),
            'description': source_data.get('description', ''),
            'auto_discover': bool(source_data.get('auto_discover', True)),
            'custom_domain': bool(source_data.get('custom_domain', False))
        }

        config['sources'][source_key] = entry
        self._config_data = config
        self._save_config()
    
    def get_source(self, source_key: str) -> SourceConfig:
        """Get configuration for a specific source."""
        config = self._load_config()
        
        if source_key not in config.get('sources', {}):
            raise KeyError(f"Source '{source_key}' not found in configuration")
            
        source_data = config['sources'][source_key]
        return SourceConfig(
            name=source_data['name'],
            type=source_data['type'],
            description=source_data['description'],
            auto_discover=source_data.get('auto_discover', True),
            custom_domain=source_data.get('custom_domain', False)
        )
    
    def list_sources(self) -> Dict[str, SourceConfig]:
        """List all available sources."""
        config = self._load_config()
        sources = {}
        
        for key, data in config.get('sources', {}).items():
            sources[key] = SourceConfig(
                name=data['name'],
                type=data['type'], 
                description=data['description'],
                auto_discover=data.get('auto_discover', True),
                custom_domain=data.get('custom_domain', False)
            )
            
        return sources
    
    def get_bulk_config(self, bulk_key: str) -> BulkConfig:
        """Get configuration for bulk operations."""
        config = self._load_config()
        
        if bulk_key not in config.get('bulk_collections', {}):
            raise KeyError(f"Bulk collection '{bulk_key}' not found in configuration")
            
        bulk_data = config['bulk_collections'][bulk_key]
        return BulkConfig(
            description=bulk_data['description'],
            sources=bulk_data['sources']
        )
    
    def list_bulk_collections(self) -> Dict[str, BulkConfig]:
        """List all available bulk collections."""
        config = self._load_config()
        bulk_collections = {}
        
        for key, data in config.get('bulk_collections', {}).items():
            bulk_collections[key] = BulkConfig(
                description=data['description'],
                sources=data['sources']
            )
            
        return bulk_collections
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default settings."""
        config = self._load_config()
        return config.get('defaults', {})
    
    def validate_source(self, source_key: str) -> bool:
        """Validate that a source exists and is properly configured."""
        try:
            source = self.get_source(source_key)
            return bool(source.name and source.type in ['publication', 'username'])
        except KeyError:
            return False
