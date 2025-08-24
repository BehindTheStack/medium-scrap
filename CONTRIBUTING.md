# Contributing to Medium Scraper

Obrigado por seu interesse em contribuir para o Medium Scraper! Este guia ajudará você a começar.

## 🚀 Como Contribuir

### 1. Configuração do Ambiente de Desenvolvimento

#### Pré-requisitos
- Python 3.10 ou superior
- Git
- UV (gerenciador de pacotes recomendado) ou pip

#### Clone e Configuração
```bash
# Clone o repositório
git clone https://github.com/BehindTheStack/medium-scrap.git
cd medium-scrap

# Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instale as dependências
pip install -e .
```

### 2. Arquitetura do Projeto

O projeto segue os princípios da **Clean Architecture** e **Domain-Driven Design**:

```
src/
├── domain/                 # Regras de negócio
│   ├── entities/          # Entidades principais (Post, Author, etc.)
│   ├── repositories/      # Interfaces dos repositórios
│   └── services/          # Serviços de domínio
├── application/           # Casos de uso
│   └── use_cases/        # Orquestração da lógica de negócio
├── infrastructure/       # Implementações técnicas
│   ├── adapters/         # Adaptadores para APIs externas
│   ├── config/           # Gerenciamento de configuração
│   └── external/         # Repositórios e integrações
└── presentation/         # Interface do usuário
    └── cli.py            # Interface de linha de comando
```

### 3. Executando os Testes

Temos uma suíte completa de testes organizados:

```bash
# Todos os testes organizados
pytest tests/unit/ tests/integration/ -v

# Apenas testes unitários
pytest tests/unit/ -v

# Apenas testes de integração
pytest tests/integration/ -v

# Com cobertura
pytest tests/unit/ tests/integration/ --cov=src --cov-report=html
```

#### Estrutura dos Testes
- **Testes Unitários** (`tests/unit/`): Testam componentes isoladamente
- **Testes de Integração** (`tests/integration/`): Testam fluxos completos

### 4. Padrões de Código

#### Estilo de Código
- Seguimos PEP 8
- Usamos type hints sempre que possível
- Documentação em docstrings seguindo o padrão Google

#### Exemplo de Classe:
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

#### Padrões de Commit
Usamos Conventional Commits:

```
feat: add support for custom domains
fix: resolve pagination issue in API adapter
docs: update README with new features
test: add integration tests for scraping
refactor: improve error handling in CLI
style: format code according to PEP 8
```

### 5. Tipos de Contribuição

#### 🐛 Reportando Bugs
- Use o template de issue para bugs
- Inclua passos para reproduzir
- Forneça informações do ambiente
- Adicione logs de erro quando possível

#### ✨ Propondo Features
- Use o template de issue para features
- Explique o caso de uso
- Forneça exemplos de como seria usado
- Considere impactos na arquitetura

#### 🔧 Contribuindo com Código

##### Para Novas Features:
1. **Crie uma issue** discutindo a feature
2. **Fork o repositório**
3. **Crie uma branch** específica: `feature/nome-da-feature`
4. **Implemente** seguindo a arquitetura existente
5. **Adicione testes** (unitários e/ou integração)
6. **Atualize documentação** se necessário
7. **Crie um Pull Request**

##### Para Bug Fixes:
1. **Crie uma issue** descrevendo o bug
2. **Fork o repositório**
3. **Crie uma branch**: `fix/nome-do-bug`
4. **Corrija o bug**
5. **Adicione teste** que reproduza e valide a correção
6. **Crie um Pull Request**

### 6. Adicionando Novas Publicações

#### Via YAML (Recomendado)
Adicione ao `medium_sources.yaml`:

```yaml
nova-publicacao:
  type: publication  # ou username para perfis de usuário
  name: domain.com   # ou @username
  description: "Descrição da publicação"
  auto_discover: true
  custom_domain: true  # se for domínio personalizado
```

#### Programaticamente
Para publicações com lógica específica, adicione ao repositório:

```python
# Em src/infrastructure/external/repositories.py
def _load_predefined_publications(self):
    # Adicione sua configuração personalizada
    nova_config = PublicationConfig(
        id=PublicationId("nova-pub"),
        name="Nova Publicação",
        type=PublicationType.CUSTOM_DOMAIN,
        domain="domain.com",
        graphql_url="https://domain.com/_/graphql",
        known_post_ids=[]
    )
```

### 7. Testando Suas Mudanças

#### Teste Funcional Básico
```bash
# Teste com publicação conhecida
python main.py --publication netflix --limit 5 --format table --skip-session

# Teste com fonte configurada
python main.py --source netflix --limit 3 --format json

# Teste com domínio customizado
python main.py --publication example.com --limit 5 --skip-session
```

#### Teste de Integração
```bash
# Execute a suíte de testes
pytest tests/integration/test_comprehensive_scenarios.py -v

# Teste específico do que você mudou
pytest tests/unit/test_[seu_modulo].py -v
```

### 8. Pull Request Guidelines

#### Checklist do PR
- [ ] Código segue os padrões do projeto
- [ ] Testes adicionados/atualizados
- [ ] Documentação atualizada
- [ ] Commits seguem Conventional Commits
- [ ] Branch está atualizada com main
- [ ] Sem conflitos de merge

#### Template do PR
```markdown
## Descrição
Breve descrição das mudanças

## Tipo de Mudança
- [ ] Bug fix
- [ ] Nova feature
- [ ] Documentação
- [ ] Refatoração

## Como Testar
1. Passos para testar a mudança
2. Comandos específicos
3. Resultados esperados

## Checklist
- [ ] Testes passando
- [ ] Código revisado
- [ ] Documentação atualizada
```

### 9. Estrutura de Dados

#### Entidades Principais
```python
@dataclass
class Post:
    """Representa um post do Medium"""
    id: PostId
    title: str
    slug: str
    author: Author
    published_at: datetime
    reading_time: float

@dataclass
class PublicationConfig:
    """Configuração de uma publicação"""
    id: PublicationId
    name: str
    type: PublicationType
    domain: str
    graphql_url: str
    known_post_ids: List[PostId]
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

## 📝 Notas Finais

- **Seja respeitoso** com outros contribuidores
- **Mantenha discussões construtivas** em issues e PRs
- **Documente suas mudanças** adequadamente
- **Teste antes de submeter** alterações

Obrigado por contribuir para tornar o Medium Scraper ainda melhor! 🚀

---

**Precisa de ajuda?** Abra uma issue ou inicie uma discussão. Estamos aqui para ajudar! 😊
