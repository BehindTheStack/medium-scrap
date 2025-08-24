# Universal Medium Scraper - Enterprise Edition

🏢 **Enterprise-grade Architecture with Netflix/Spotify patterns**

[![Tests](https://img.shields.io/badge/tests-73%20passing-brightgreen)](#-testing)
[![Coverage](https://img.shields.io/badge/coverage-44%25-yellow)](#-coverage)
[![Clean Architecture](https://img.shields.io/badge/architecture-clean-blue)](#-clean-architecture)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#-installation)

## 🚀 Overview

Universal Medium scraper built with **Clean Architecture**, **SOLID Principles**, and **Design Patterns** used by companies like Netflix and Spotify. Supports any Medium publication with intelligent post discovery and modern visual interface.

### ✨ What's New in v2.0

- 🎨 **Enhanced Visual Interface**: Animated loader with progress phases
- 🌐 **Custom Domains**: Full support for `.engineering`, `.tech`, etc.
- 📊 **Progress Tracking**: Real-time progress bars
- 🧪 **73 Tests**: Complete suite of unit and integration tests
- 📝 **YAML Config**: Flexible configuration system
- 🚀 **Bulk Collections**: Collect from multiple sources simultaneously

## 🏗️ Clean Architecture

### Layers

```
src/
├── domain/               # Pure business rules
│   ├── entities/         # Domain entities (Post, Author, Publication)
│   ├── repositories/     # Repository interfaces
│   └── services/         # Domain services
├── application/          # Application use cases
│   └── use_cases/        # Use case implementations
├── infrastructure/       # External adapters
│   ├── adapters/         # External API adapters (GraphQL)
│   ├── config/           # YAML configuration management
│   └── external/         # Concrete implementations
└── presentation/         # User interface
    └── cli.py            # CLI Controller with Rich UI
```

### Implemented Patterns

- **Repository Pattern**: Data access abstraction
- **Strategy Pattern**: Different discovery strategies
- **Command Pattern**: Use cases as commands
- **Adapter Pattern**: External API integration
- **Dependency Injection**: Dependency inversion
- **Factory Pattern**: Configuration creation
- **Observer Pattern**: Progress tracking system

## 🎯 Features

### Core Features
- ✅ **Intelligent Discovery**: Auto-discovery + known IDs + fallback
- ✅ **Custom Domains**: Netflix, Kickstarter, etc. 
- ✅ **User Profiles**: @SkyscannerEng, @TinderEng, etc.
- ✅ **Medium Publications**: Pinterest, Airbnb, Uber, etc.
- ✅ **Complete Pagination**: Collects ALL available posts
- ✅ **Rate Limiting**: Respects API limits

### Interface & UX  
- 🎨 **Rich CLI**: Modern visual interface with colors and emojis
- 📊 **Progress Bars**: Animated loader with detailed phases
- 🎭 **Multiple Formats**: Table, JSON, IDs
- 📁 **Auto Output**: Automatic saving to `outputs/`
- 🔄 **Bulk Operations**: Batch processing

### Configuration & Flexibility
- 📝 **YAML Sources**: Configure reusable sources
- 🎛️ **Flexible Parameters**: Limit, format, mode, etc.
- 🔧 **Custom Domains**: Automatic support for any domain
- 📦 **Bulk Collections**: Predefined source groups

## 🛠️ Installation

```bash
# Clone the repository
git clone <repo-url>
cd medium-scrap

# Install with uv
uv sync
```

## 📖 Usage

### Basic Commands

```bash
# Quick Netflix scraping
python main.py --publication netflix --limit 5

# Auto-discovery (production mode)
python main.py --publication pinterest --auto-discover --skip-session --format json

# Custom IDs
python main.py --publication netflix --custom-ids "ac15cada49ef,64c786c2a3ac"

# Any publication
python main.py --publication unknown-blog --auto-discover --limit 10
```

### Complete Options

```bash
-p, --publication TEXT         Publication name (netflix, pinterest, or any)
-o, --output TEXT              File to save results
-f, --format [table|json|ids]  Output format
--custom-ids TEXT              Specific IDs list (comma-separated)
--skip-session                 Skip session initialization (faster)
--limit INTEGER                Maximum number of posts
--auto-discover                Force auto-discovery mode (production ready)
--help                         Show help
```

## 🧪 Testing

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v
```

## 📋 Supported Publications

### Pre-configured
- **Netflix Tech Blog** (`netflix`)
- **Pinterest Engineering** (`pinterest`)

### Universal Discovery
- Any Medium publication can be automatically discovered
- Use `--auto-discover` for non-preconfigured publications

## 🏢 Enterprise Patterns

### SOLID Principles

- **Single Responsibility**: Each class has one responsibility
- **Open/Closed**: Extensible without modification
- **Liskov Substitution**: Subtypes replace base types
- **Interface Segregation**: Specific interfaces
- **Dependency Inversion**: Abstract dependencies

### Clean Architecture

- **Domain Layer**: Framework-independent business rules
- **Application Layer**: Application use cases
- **Infrastructure Layer**: Implementation details
- **Presentation Layer**: User interface

## 🚀 Usage Examples

### Example 1: Basic Scraping
```bash
python main.py --publication netflix --limit 3 --format table
```

### Example 2: Production Mode
```bash
python main.py --publication pinterest --auto-discover --skip-session --format json --output results.json
```

### Example 3: Specific IDs
```bash
python main.py --publication netflix --custom-ids "ac15cada49ef,64c786c2a3ac" --format json
```

## 📁 Project Structure

```
medium-scrap/
├── src/
│   ├── domain/
│   │   ├── entities/
│   │   │   └── publication.py      # Domain entities
│   │   ├── repositories/
│   │   │   └── base.py             # Repository interfaces
│   │   └── services/
│   │       └── publication_service.py  # Domain services
│   ├── application/
│   │   └── use_cases/
│   │       └── scrape_posts.py     # Main use cases
│   ├── infrastructure/
│   │   ├── adapters/
│   │   │   └── medium_api_adapter.py   # API adapter
│   │   └── external/
│   │       └── repositories.py     # Concrete repositories
│   └── presentation/
│       └── cli.py                  # CLI interface
├── tests/
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── main.py                        # Entry point
├── pyproject.toml                 # Project configuration
└── README.md                      # This documentation
```

## 🎯 Architecture Benefits

1. **Testability**: Isolated tests for each layer
2. **Maintainability**: Clear separation of responsibilities
3. **Extensibility**: Easy addition of new features
4. **Scalability**: Architecture prepared for growth
5. **Quality**: Standards used by tier-1 companies

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for complete details.

### MIT License Summary
- ✅ **Commercial Use**: Allowed for commercial projects
- ✅ **Modification**: Can modify source code
- ✅ **Distribution**: Can distribute modified versions
- ✅ **Private Use**: Can use for private projects
- ⚠️ **Liability**: Software provided "as is", no warranties

---

**Built with Clean Architecture and enterprise-grade patterns** 🏢✨
