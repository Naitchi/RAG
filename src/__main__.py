from .rag import RagSystem
import fire


def main() -> None:
    """Expose every `RagSystem` method as a Python Fire CLI command."""
    fire.Fire(RagSystem)


if __name__ == "__main__":
    main()
