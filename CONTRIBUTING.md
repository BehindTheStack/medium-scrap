# Contributing to Medium Scraper

Thank you for your interest in contributing to Medium Scraper! This guide will help you get started.

## 🚀 How to Contribute

### 1. Development Environment Setup

#### Prerequisites
- Python 3.10 or higher
- Git
- UV (recommended package manager) or pip

#### Clone and Setup
```bash
# Clone the repository
git clone https://github.com/BehindTheStack/medium-scrap.git
cd medium-scrap

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### 2. Project Architecture

The project follows **Clean Architecture** and **Domain-Driven Design** principles:

```
src/
├── domain/                 # Business rules
│   ├── entities/          # Core entities (Post, Author, etc.)
│   ├── repositories/      # Repository interfaces
│   └── services/          # Domain services
├── application/           # Use cases
│   └── use_cases/        # Business logic orchestration
├── infrastructure/       # Technical implementations
│   ├── adapters/         # External API adapters
│   ├── config/           # Configuration management
│   └── external/         # Repositories and integrations
└── presentation/         # User interface
    └── cli.py            # Command line interface
```

### 3. Running Tests

We have a complete organized test suite:

```bash
# All organized tests
pytest tests/unit/ tests/integration/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/unit/ tests/integration/ --cov=src --cov-report=html
```

#### Test Structure
- **Unit Tests** (`tests/unit/`): Test components in isolation
- **Integration Tests** (`tests/integration/`): Test complete flows

### 4. Code Standards

#### Code Style
- Follow PEP 8
- Use type hints whenever possible
- Documentation in docstrings following Google standard

#### Class Example:
```python
"""
Module docstring explaining the purpose
"""

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ExampleEntity:
    """
    Example entity following domain patterns
    """
    id: str
    name: str
    optional_field: Optional[str] = None
    
    def validate(self) -> None:
        """Validate entity rules"""
        if not self.id:
            raise ValueError("ID is required")
```

#### Commit Patterns
We use Conventional Commits:

```
feat: add support for custom domains
fix: resolve pagination issue in API adapter
docs: update README with new features
test: add integration tests for scraping
refactor: improve error handling in CLI
style: format code according to PEP 8
```

### 5. Types of Contributions

#### 🐛 Reporting Bugs
- Use the bug issue template
- Include steps to reproduce
- Provide environment information
- Add error logs when possible

#### ✨ Proposing Features
- Use the feature issue template
- Explain the use case
- Provide examples of how it would be used
- Consider architecture impacts

#### 🔧 Contributing Code

##### For New Features:
1. **Create an issue** discussing the feature
2. **Fork the repository**
3. **Create a specific branch**: `feature/feature-name`
4. **Implement** following existing architecture
5. **Add tests** (unit and/or integration)
6. **Update documentation** if necessary
7. **Create a Pull Request**

##### For Bug Fixes:
1. **Create an issue** describing the bug
2. **Fork the repository**
3. **Create a branch**: `fix/bug-name`
4. **Fix the bug**
5. **Add test** that reproduces and validates the fix
6. **Create a Pull Request**

### 6. Adding New Publications

#### Via YAML (Recommended)
Add to `medium_sources.yaml`:

```yaml
new-publication:
  type: publication  # or username for user profiles
  name: domain.com   # or @username
  description: "Publication description"
  auto_discover: true
  custom_domain: true  # if custom domain
```

#### Via CLI (convenient)

You can add or update sources using the built-in CLI `add-source` subcommand. This is useful for quickly registering a publication without editing the YAML file by hand.

Example:

```bash
python main.py add-source \
    --key pinterest \
    --type publication \
    --name pinterest \
    --description "Pinterest Engineering" \
    --auto-discover
```

Notes:
- The command writes to `medium_sources.yaml` in the repository root and will create the `sources` section if it does not exist.
- The implementation avoids importing optional network adapters when running this subcommand, so it can run even if HTTP dependencies (like `httpx`) are not installed.
- After running `add-source`, verify changes with `python main.py --list-sources` or by inspecting `medium_sources.yaml`.


#### Programmatically
For publications with specific logic, add to repository:

```python
# In src/infrastructure/external/repositories.py
def _load_predefined_publications(self):
    # Add your custom configuration
    new_config = PublicationConfig(
        id=PublicationId("new-pub"),
        name="New Publication",
        type=PublicationType.CUSTOM_DOMAIN,
        domain="domain.com",
        graphql_url="https://domain.com/_/graphql",
        known_post_ids=[]
    )
```

### 7. Testing Your Changes

#### Basic Functional Test
```bash
# Test with known publication
python main.py --publication netflix --limit 5 --format table --skip-session

# Test with configured source
python main.py --source netflix --limit 3 --format json

# Test with custom domain
python main.py --publication example.com --limit 5 --skip-session
```

#### Integration Testing
```bash
# Run test suite
pytest tests/integration/test_comprehensive_scenarios.py -v

# Test specific to your changes
pytest tests/unit/test_[your_module].py -v
```

### 8. Pull Request Guidelines

#### PR Checklist
- [ ] Code follows project standards
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Commits follow Conventional Commits
- [ ] Branch is up to date with main
- [ ] No merge conflicts

#### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## How to Test
1. Steps to test the change
2. Specific commands
3. Expected results

## Checklist
- [ ] Tests passing
- [ ] Code reviewed
- [ ] Documentation updated
```

### 9. Data Structures

#### Main Entities
```python
@dataclass
class Post:
    """Represents a Medium post"""
    id: PostId
    title: str
    slug: str
    author: Author
    published_at: datetime
    reading_time: float

@dataclass
class PublicationConfig:
    """Publication configuration"""
    id: PublicationId
    name: str
    type: PublicationType
    domain: str
    graphql_url: str
    known_post_ids: List[PostId]
```

### 10. Debugging and Logs

#### Local Debug
```bash
# Enable verbose logs (if implemented)
python main.py --publication netflix --limit 5 --verbose

# Use Python debug mode
python -m pdb main.py --publication netflix --limit 5
```

#### Log Structure
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Important information")
logger.debug("Debug details")
logger.warning("Warning about something")
logger.error("Recoverable error")
```

### 11. Performance and Optimization

#### Guidelines
- **Rate Limiting**: Respect Medium API limits
- **Caching**: Consider caching for unchanging data
- **Pagination**: Implement efficient pagination
- **Error Handling**: Handle errors gracefully

#### Rate Limiting Example
```python
import time

def with_rate_limit(self, delay: float = 0.5):
    """Apply rate limiting between requests"""
    time.sleep(delay)
    # Your logic here
```

### 12. Useful Resources

#### Documentation
- [Rich Library](https://rich.readthedocs.io/) - User interface
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Pytest](https://docs.pytest.org/) - Testing framework

#### Development Tools
```bash
# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### 13. Community and Support

#### Where to Get Help
- **Issues**: For bugs and feature requests
- **Discussions**: For general questions
- **Wiki**: Additional documentation

#### How to Report Problems
1. Check if the problem has already been reported
2. Use the appropriate issue template
3. Provide as much context as possible
4. Include versions and environment

---

## 📜 Licensing

### License Agreement
By contributing to this project, you agree that your contributions will be licensed under the same [MIT License](LICENSE) as the project.

### What this means:
- ✅ Your contributions can be used commercially
- ✅ They can be modified and redistributed
- ✅ You retain copyright of your original contributions
- ⚠️ You guarantee you have the right to license your contributions

### CLA (Contributor License Agreement)
No separate CLA signing is required. The MIT license is sufficient and clear about rights and responsibilities.

---

## 📝 Final Notes

- **Be respectful** with other contributors
- **Keep discussions constructive** in issues and PRs
- **Document your changes** adequately
- **Test before submitting** changes

Thank you for contributing to make Medium Scraper even better! 🚀

---

**Need help?** Open an issue or start a discussion. We're here to help! 😊
```

### 10. Debugging e Logs

#### Debug Local
```bash
# Habilite logs verbose (se implementado)
python main.py --publication netflix --limit 5 --verbose

# Use modo debug do Python
python -m pdb main.py --publication netflix --limit 5
```

#### Estrutura de Logs
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Informação importante")
logger.debug("Detalhes para debug")
logger.warning("Aviso sobre algo")
logger.error("Erro recuperável")
```

### 11. Performance e Otimização

#### Diretrizes
- **Rate Limiting**: Respeite limites da API do Medium
- **Caching**: Considere cache para dados que não mudam
- **Pagination**: Implemente paginação eficiente
- **Error Handling**: Trate erros graciosamente

#### Exemplo de Rate Limiting
```python
import time

def with_rate_limit(self, delay: float = 0.5):
    """Aplica rate limiting entre requests"""
    time.sleep(delay)
    # Sua lógica aqui
```

### 12. Recursos Úteis

#### Documentação
- [Rich Library](https://rich.readthedocs.io/) - Interface de usuário
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Pytest](https://docs.pytest.org/) - Framework de testes

#### Ferramentas de Desenvolvimento
```bash
# Formatação de código
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### 13. Comunidade e Suporte

#### Onde Buscar Ajuda
- **Issues**: Para bugs e feature requests
- **Discussions**: Para perguntas gerais
- **Wiki**: Documentação adicional

#### Como Reportar Problemas
1. Verifique se o problema já foi reportado
2. Use o template de issue apropriado
3. Forneça o máximo de contexto possível
4. Inclua versões e ambiente

---

## � Licenciamento

### Concordância com a Licença
Ao contribuir com este projeto, você concorda que suas contribuições serão licenciadas sob a mesma [Licença MIT](LICENSE) do projeto.

### O que isso significa:
- ✅ Suas contribuições podem ser usadas comercialmente
- ✅ Podem ser modificadas e redistribuídas
- ✅ Você mantém o copyright de suas contribuições originais
- ⚠️ Você garante que tem direito de licenciar suas contribuições

### CLA (Contributor License Agreement)
Não é necessário assinar um CLA separado. A licença MIT é suficiente e clara sobre os direitos e responsabilidades.

---

## �📝 Notas Finais

- **Seja respeitoso** com outros contribuidores
- **Mantenha discussões construtivas** em issues e PRs
- **Documente suas mudanças** adequadamente
- **Teste antes de submeter** alterações

Obrigado por contribuir para tornar o Medium Scraper ainda melhor! 🚀

---

**Precisa de ajuda?** Abra uma issue ou inicie uma discussão. Estamos aqui para ajudar! 😊
