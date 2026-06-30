.PHONY: install lint format typecheck test serve inspect clean

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

serve:
	uv run dstools serve

inspect:
	uv run dstools inspect

clean:
	rm -rf build dist .eggs src/*.egg-info .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
