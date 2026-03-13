LIB = src/tsod

.PHONY: check build lint format test coverage docs examples clean

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

examples:
	$(MAKE) -C docs examples

docs:
	$(MAKE) -C docs build

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf dist
	$(MAKE) -C docs clean
