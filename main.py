#!/usr/bin/env python3
"""
Universal Medium Scraper - Enterprise Edition
Main entry point following Clean Architecture principles

Architecture Layers:
- Domain: Core business logic and entities
- Application: Use cases and orchestration  
- Infrastructure: External integrations and adapters
- Presentation: User interface (CLI)
"""

from src.presentation.cli import cli

if __name__ == "__main__":
    cli()
