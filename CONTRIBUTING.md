# Contributing to Async Toolformer Orchestrator

Thank you for your interest in contributing! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/async-toolformer-orchestrator.git
   cd async-toolformer-orchestrator
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## 🛠️ Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** with clear, focused commits
3. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```
4. **Run linting and formatting**:
   ```bash
   ruff check --fix .
   ruff format .
   ```
5. **Run type checking**:
   ```bash
   mypy src/
   ```

### Submitting Changes

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Link to related issues
   - Screenshots/examples if applicable

## 🧪 Testing

We use pytest with asyncio support:

```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest --cov=async_toolformer --cov-report=html
```

### Writing Tests

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **E2E tests**: Test full workflows with real LLM APIs

Example test structure:
```python
import pytest
from async_toolformer import AsyncOrchestrator

@pytest.mark.asyncio
async def test_parallel_execution():
    orchestrator = AsyncOrchestrator()
    # Your test here
    assert result.execution_time < 2.0
```

## 📝 Code Style

We use Ruff for linting and formatting:

- **Line length**: 88 characters
- **Import sorting**: isort-compatible
- **Type hints**: Required for public APIs
- **Docstrings**: Google style for public functions

Example:
```python
async def execute_tools(
    tools: list[Tool],
    max_parallel: int = 10,
) -> list[ToolResult]:
    """Execute multiple tools in parallel.
    
    Args:
        tools: List of tools to execute
        max_parallel: Maximum concurrent executions
        
    Returns:
        List of tool execution results
        
    Raises:
        RateLimitError: When rate limits are exceeded
    """
```

## 🎯 Contribution Areas

We welcome contributions in these areas:

### High Priority
- **LLM Provider Support**: Add support for new LLM APIs
- **Rate Limiting**: Improve rate limiting algorithms
- **Performance**: Optimize parallel execution
- **Testing**: Increase test coverage

### Medium Priority  
- **Documentation**: API docs, tutorials, examples
- **Monitoring**: Better observability and metrics
- **CLI Tools**: Command-line utilities
- **Error Handling**: More robust error recovery

### Future Ideas
- **Speculation Engine**: Smarter prediction algorithms
- **Distributed Execution**: Multi-node orchestration
- **Visual Tools**: GUI for orchestrator monitoring
- **Plugins**: Extensible tool system

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment details**:
   - Python version
   - Package version
   - Operating system

2. **Reproduction steps**:
   - Minimal code example
   - Expected behavior
   - Actual behavior

3. **Context**:
   - Error messages/stack traces
   - Related configuration
   - Recent changes

## 💡 Feature Requests

For feature requests:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and problem being solved
3. **Propose implementation** if you have ideas
4. **Consider backwards compatibility**

## 📋 Development Setup

### Prerequisites
- Python 3.10+
- Git
- Make (optional, for convenience commands)

### Optional Tools
- **Redis**: For distributed rate limiting tests
- **Docker**: For integration testing
- **Jupyter**: For interactive development

### Environment Variables
```bash
# For testing with real APIs
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# For Redis integration tests  
export REDIS_URL="redis://localhost:6379"
```

## 🏗️ Project Structure

```
async-toolformer-orchestrator/
├── src/async_toolformer/     # Main package
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
├── docs/                    # Documentation
├── examples/                # Example code
└── scripts/                 # Development scripts
```

## 🔍 Code Review Process

All contributions go through code review:

1. **Automated checks** must pass (CI/CD)
2. **Maintainer review** for code quality and design
3. **Testing verification** on multiple environments
4. **Documentation review** for user-facing changes

### Review Criteria
- **Correctness**: Does it work as intended?
- **Performance**: Does it maintain/improve performance?
- **Security**: Are there any security implications?
- **Maintainability**: Is the code clear and well-structured?
- **Documentation**: Are changes properly documented?

## 🎉 Recognition

Contributors are recognized through:
- **GitHub contributors list**
- **Changelog mentions** for significant contributions
- **Discord community highlights**
- **Conference talk acknowledgments**

## ❓ Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests  
- **Discord**: Real-time community chat
- **Email**: async-tools@yourdomain.com for sensitive issues

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make async tool orchestration better for everyone! 🚀