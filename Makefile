ARGS ?=

install-ollama:
	curl -fsSL https://ollama.com/install.sh | sh

pull-qwen:
	ollama serve & sleep 2 && ollama pull qwen3:0.6b

install: install-ollama pull-qwen 
	uv sync  
	
run:
	ollama serve & sleep 2 && uv run src $(ARGS)

debug:
	uv run python -m pdb $(ARGS)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

fclean: clean
	rm -rf .venv/

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run debug build clean fclean lint lint-strict
