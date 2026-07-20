ARGS ?=

install-ollama:
	curl -fsSL https://ollama.com/install.sh | sh

pull-qwen:
	ollama serve > /dev/null 2>&1 &
	sleep 2
	ollama pull qwen3:0.6b

install: pull-qwen 
	uv sync  
	
run:
	uv run python -m src $(ARGS)

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

.PHONY: install run debug build clean fclean lint
