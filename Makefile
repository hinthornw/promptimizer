.PHONY: lint format


lint:
	uv run ruff check .
	uv run ruff format . --diff

format:
	uv run ruff check --fix .
	uv run ruff format .
