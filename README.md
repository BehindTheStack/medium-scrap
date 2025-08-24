# Universal Medium Scraper - Enterprise Edition

🏢 **Arquitetura Enterprise-grade com padrões Netflix/Spotify**

[![Tests](https://img.shields.io/badge/tests-73%20passing-brightgreen)](#-testes)
[![Coverage](https://img.shields.io/badge/coverage-44%25-yellow)](#-cobertura)
[![Clean Architecture](https://img.shields.io/badge/architecture-clean-blue)](#-arquitetura-clean)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#-instalação)

## 🚀 Visão Geral

Scraper universal do Medium construído com **Clean Architecture**, **SOLID Principles** e **Design Patterns** utilizados por empresas como Netflix e Spotify. Suporta qualquer publicação do Medium com descoberta inteligente de posts e interface visual moderna.

### ✨ Novidades v2.0

- 🎨 **Interface Visual Melhorada**: Loader animado com fases de progresso
- 🌐 **Domínios Customizados**: Suporte completo para `.engineering`, `.tech`, etc.
- 📊 **Progress Tracking**: Barra de progresso em tempo real
- 🧪 **73 Testes**: Suíte completa de testes unitários e integração
- 📝 **YAML Config**: Sistema de configuração flexível
- 🚀 **Bulk Collections**: Colete de múltiplas fontes simultaneamente

## 🏗️ Arquitetura Clean

### Camadas

```
src/
├── domain/               # Regras de negócio puras
│   ├── entities/         # Entidades de domínio (Post, Author, Publication)
│   ├── repositories/     # Interfaces dos repositórios
│   └── services/         # Serviços de domínio
├── application/          # Casos de uso da aplicação
│   └── use_cases/        # Implementação dos casos de uso
├── infrastructure/       # Adaptadores externos
│   ├── adapters/         # Adaptadores para APIs externas (GraphQL)
│   ├── config/           # Gerenciamento de configuração YAML
│   └── external/         # Implementações concretas
└── presentation/         # Interface do usuário
    └── cli.py            # Controller CLI com Rich UI
```

### Padrões Implementados

- **Repository Pattern**: Abstração de acesso a dados
- **Strategy Pattern**: Diferentes estratégias de descoberta
- **Command Pattern**: Casos de uso como comandos
- **Adapter Pattern**: Integração com API externa
- **Dependency Injection**: Inversão de dependências
- **Factory Pattern**: Criação de configurações
- **Observer Pattern**: Sistema de progress tracking

## 🎯 Recursos

### Core Features
- ✅ **Descoberta Inteligente**: Auto-discovery + IDs conhecidos + fallback
- ✅ **Domínios Customizados**: Netflix, Kickstarter, etc. 
- ✅ **Perfis de Usuário**: @SkyscannerEng, @TinderEng, etc.
- ✅ **Publicações Medium**: Pinterest, Airbnb, Uber, etc.
- ✅ **Paginação Completa**: Coleta TODOS os posts disponíveis
- ✅ **Rate Limiting**: Respeita limites da API

### Interface & UX  
- 🎨 **Rich CLI**: Interface visual moderna com cores e emojis
- 📊 **Progress Bars**: Loader animado com fases detalhadas
- 🎭 **Multiple Formats**: Table, JSON, IDs
- 📁 **Auto Output**: Salvamento automático em `outputs/`
- 🔄 **Bulk Operations**: Processamento em lote

### Configuração & Flexibilidade
- 📝 **YAML Sources**: Configure fontes reutilizáveis
- 🎛️ **Flexible Parameters**: Limite, formato, modo, etc.
- 🔧 **Custom Domains**: Suporte automático para qualquer domínio
- 📦 **Bulk Collections**: Grupos de fontes predefinidos
## 🛠️ Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd medium-scrap

# Instale com uv
uv sync
```

## � Uso

### Comandos Básicos

```bash
# Scraping rápido do Netflix
python main.py --publication netflix --limit 5

# Auto-descoberta (modo produção)
python main.py --publication pinterest --auto-discover --skip-session --format json

# IDs customizados
python main.py --publication netflix --custom-ids "ac15cada49ef,64c786c2a3ac"

# Qualquer publicação
python main.py --publication unknown-blog --auto-discover --limit 10
```

### Opções Completas

```bash
-p, --publication TEXT         Nome da publicação (netflix, pinterest, ou qualquer)
-o, --output TEXT              Arquivo para salvar resultados
-f, --format [table|json|ids]  Formato de saída
--custom-ids TEXT              Lista de IDs específicos (separados por vírgula)
--skip-session                 Pular inicialização de sessão (mais rápido)
--limit INTEGER                Número máximo de posts
--auto-discover                Forçar modo auto-descoberta (pronto para produção)
--help                         Mostrar ajuda
```

## 🧪 Testes

```bash
# Todos os testes
python -m pytest tests/ -v

# Apenas testes unitários
python -m pytest tests/unit/ -v

# Apenas testes de integração
python -m pytest tests/integration/ -v
```

## 📋 Publicações Suportadas

### Pré-configuradas
- **Netflix Tech Blog** (`netflix`)
- **Pinterest Engineering** (`pinterest`)

### Descoberta Universal
- Qualquer publicação do Medium pode ser descoberta automaticamente
- Use `--auto-discover` para publicações não pré-configuradas

## 🏢 Padrões Enterprise

### Princípios SOLID

- **Single Responsibility**: Cada classe tem uma responsabilidade
- **Open/Closed**: Extensível sem modificação
- **Liskov Substitution**: Subtipos substituem tipos base
- **Interface Segregation**: Interfaces específicas
- **Dependency Inversion**: Dependências abstratas

### Clean Architecture

- **Domain Layer**: Regras de negócio independentes
- **Application Layer**: Casos de uso da aplicação
- **Infrastructure Layer**: Detalhes de implementação
- **Presentation Layer**: Interface do usuário

## 🚀 Exemplos de Uso

### Exemplo 1: Scraping Básico
```bash
python main.py --publication netflix --limit 3 --format table
```

### Exemplo 2: Modo Produção
```bash
python main.py --publication pinterest --auto-discover --skip-session --format json --output results.json
```

### Exemplo 3: IDs Específicos
```bash
python main.py --publication netflix --custom-ids "ac15cada49ef,64c786c2a3ac" --format json
```

## 📁 Estrutura do Projeto

```
medium-scrap/
├── src/
│   ├── domain/
│   │   ├── entities/
│   │   │   └── publication.py      # Entidades de domínio
│   │   ├── repositories/
│   │   │   └── base.py             # Interfaces dos repositórios
│   │   └── services/
│   │       └── publication_service.py  # Serviços de domínio
│   ├── application/
│   │   └── use_cases/
│   │       └── scrape_posts.py     # Casos de uso principais
│   ├── infrastructure/
│   │   ├── adapters/
│   │   │   └── medium_api_adapter.py   # Adaptador da API
│   │   └── external/
│   │       └── repositories.py     # Repositórios concretos
│   └── presentation/
│       └── cli.py                  # Interface CLI
├── tests/
│   ├── unit/                      # Testes unitários
│   └── integration/               # Testes de integração
├── main.py                        # Ponto de entrada
├── pyproject.toml                 # Configuração do projeto
└── README.md                      # Esta documentação
```

## 🎯 Benefícios da Arquitetura

1. **Testabilidade**: Testes isolados para cada camada
2. **Manutenibilidade**: Separação clara de responsabilidades
3. **Extensibilidade**: Fácil adição de novas funcionalidades
4. **Escalabilidade**: Arquitetura preparada para crescimento
5. **Qualidade**: Padrões utilizados por empresas tier-1

## 📄 Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE) - veja o arquivo [LICENSE](LICENSE) para detalhes completos.

### Resumo da Licença MIT
- ✅ **Uso Comercial**: Permitido uso em projetos comerciais
- ✅ **Modificação**: Pode modificar o código fonte
- ✅ **Distribuição**: Pode distribuir versões modificadas
- ✅ **Uso Privado**: Pode usar para projetos privados
- ⚠️ **Responsabilidade**: Software fornecido "como está", sem garantias

---

**Desenvolvido com Clean Architecture e padrões enterprise-grade** 🏢✨
