# Development Guide

This guide covers setting up the development environment and contributing to the async-toolformer-orchestrator project.

## 🚀 Quick Setup

### Prerequisites

- Python 3.10+ 
- Git
- Make (optional, for convenience commands)

### Environment Setup

1. **Clone and enter the repository**:
   ```bash
   git clone <your-fork-url>
   cd async-toolformer-orchestrator
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   make install-dev
   # OR manually:
   pip install -e ".[dev]"
   pre-commit install
   ```

## 🛠️ Development Workflow

### Making Changes

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** with regular commits

3. **Run checks before committing**:
   ```bash
   make check  # Runs lint, typecheck, and unit tests
   ```

4. **Run full test suite**:
   ```bash
   make test
   ```

### Code Quality

We maintain high code quality through automated tooling:

- **Ruff**: Fast linting and formatting
- **MyPy**: Static type checking  
- **Bandit**: Security vulnerability scanning
- **Pre-commit hooks**: Automated checks on every commit

#### Running Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make typecheck

# All checks at once
make check
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests  
├── e2e/           # End-to-end tests with real APIs
└── conftest.py    # Shared fixtures and configuration
```

### Running Tests

```bash
# All tests
make test

# By category  
make test-unit
make test-integration
make test-e2e

# With coverage
make test-coverage
```

### Writing Tests

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions without external APIs
- **E2E tests**: Test complete workflows with real LLM APIs (require API keys)

Example test:
```python
import pytest
from async_toolformer import AsyncOrchestrator, Tool

@pytest.mark.asyncio
async def test_parallel_execution(orchestrator):
    @Tool(description="Test tool")
    async def test_tool(x: int) -> int:
        return x * 2
    
    result = await orchestrator.execute_tools([test_tool], {"x": 5})
    assert result == 10
```

### Test Configuration

Set environment variables for E2E tests:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export REDIS_URL="redis://localhost:6379"  # For distributed rate limiting tests
```

## 📊 Performance & Benchmarking

### Running Benchmarks

```bash
make benchmark
```

### Profiling

```bash
make profile
```

### Performance Guidelines

- Use `asyncio.gather()` for truly parallel operations
- Implement proper rate limiting to avoid API throttling
- Use connection pooling for HTTP clients
- Profile memory usage for large-scale tool orchestration

## 🔧 Development Tools

### IDE Configuration

#### VS Code
Recommended extensions and settings in `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

#### PyCharm
- Enable Ruff plugin
- Configure pytest as test runner
- Set up mypy integration

### Debugging

#### Async Debugging
```python
import asyncio
import logging

# Enable asyncio debug mode  
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # Windows
logging.basicConfig(level=logging.DEBUG)

# Use breakpoint() in async functions
async def debug_tool():
    breakpoint()  # Works in async context
    await some_async_operation()
```

#### Tool Execution Debugging
```python
from async_toolformer import AsyncOrchestrator

orchestrator = AsyncOrchestrator(
    debug=True,  # Enables detailed logging
    trace_executions=True  # Traces tool call timing
)
```

## 🏗️ Architecture

### Core Components

- **AsyncOrchestrator**: Main coordination engine
- **Tool**: Decorator for registering async functions as tools
- **RateLimiter**: Manages API rate limits across services
- **SpeculationEngine**: Predicts and pre-fetches likely tool calls
- **ResultAggregator**: Combines results from parallel executions

### Design Principles

1. **Async-first**: Everything is built for asyncio
2. **Rate-limit aware**: Respects API limits without failing
3. **Error isolation**: One tool failure doesn't break others
4. **Observable**: Rich logging and metrics for monitoring
5. **Extensible**: Plugin system for custom tools and limiters

## 📝 Documentation

### Building Docs

```bash
make docs
make serve-docs  # Serves at http://localhost:8000
```

### Documentation Guidelines

- Use Google-style docstrings
- Include type hints for all public APIs
- Provide usage examples in docstrings
- Update README.md for user-facing changes

## 🐳 Docker Development

### Building Image

```bash
make docker-build
```

### Running Tests in Docker

```bash
make docker-test
```

### Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  app:
    build: .
    volumes:
      - .:/app
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## 🚀 Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Steps

1. **Update version** in `src/async_toolformer/__init__.py`
2. **Update CHANGELOG.md** with release notes
3. **Create release PR** and get approval
4. **Tag release**:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
5. **Build and release**:
   ```bash
   make release-test  # Test on PyPI test first
   make release       # Release to PyPI
   ```

## 🤝 Getting Help

- **GitHub Discussions**: Design questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time development chat
- **Email**: async-tools@yourdomain.com for sensitive issues

## 📚 Additional Resources

- [AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/claude/reference)
- [Pytest-AsyncIO Documentation](https://pytest-asyncio.readthedocs.io/)
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/)