# Minimal makefile for Quarto documentation
#

.PHONY: help api build preview clean

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  api      to generate API documentation with quartodoc"
	@echo "  build    to build the documentation as HTML"
	@echo "  preview  to preview the documentation with live reload"
	@echo "  clean    to remove generated files"

api:
	uv run quartodoc build

build: api
	uv run quarto render

preview: api
	uv run quarto preview

clean:
	rm -rf _site objects.json
	rm -f api/*.qmd
