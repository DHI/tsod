LIB = src/tsod

.PHONY: check build lint format test coverage docs clean

check: lint test

build: test
	uv build

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

coverage:
	uv run pytest --cov-report html --cov=$(LIB) tests/

docs:
	cd docs && uv run quartodoc build
	uv run quarto render docs

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf dist
	rm -rf docs/_build
