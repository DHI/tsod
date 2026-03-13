# Minimal makefile for Quarto documentation
#

.PHONY: help api examples build preview clean

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  api      to generate API documentation with quartodoc"
	@echo "  build    to build the documentation as HTML"
	@echo "  preview  to preview the documentation with live reload"
	@echo "  clean    to remove generated files"

api:
	uv run quartodoc build

examples:
	uv run python scripts/generate_examples_from_notebooks.py

build: api examples
	uv run quarto render

preview: api examples
	uv run quarto preview

clean:
	rm -rf _site .quarto objects.json
	find api -name "*.qmd" ! -name "index.qmd" -delete
	rm -f examples/*.ipynb
	rm -rf examples/*_files
