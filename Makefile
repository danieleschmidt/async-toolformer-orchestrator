.PHONY: install test lint demo clean

install:
	~/anaconda3/bin/pip install -e ".[dev]" -q

test:
	~/anaconda3/bin/python3 -m pytest tests/ -v

lint:
	~/anaconda3/bin/python3 -m ruff check src/ tests/

demo:
	~/anaconda3/bin/python3 examples/demo.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage *.egg-info
