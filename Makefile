LIB = tsod

check: lint typecheck test

build: typecheck test
	python -m build

lint:
	uv run ruff check $(LIB)

format:
	uv run ruff format $(LIB)

test:
	uv run pytest --disable-warnings

typecheck:
	uv run mypy $(LIB)/

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

docs:
	cd docs && uv run quartodoc build
	uv run quarto render docs
