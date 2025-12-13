"""Main entry point for the RAG pipeline CLI"""
import asyncio
import argparse
from src.cli.serve import start_session
from src.cli.build_index import build_vector_store
from src.rag.vector_store import VectorStore


def main():
    """Main CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Retrieve and generate answers from your documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\nExamples:
        python main.py build    Build vector store from scratch (clears existing data)
        python main.py serve    Start an interactive question-answering session
        """
    )

    parser.add_argument(
        'command',
        choices=['build_index', 'serve'],
        help='Command to execute'
    )

    args = parser.parse_args()

    if args.command == 'build_index':
        build_vector_store()
    elif args.command == 'serve':
        asyncio.run(start_session())


if __name__ == "__main__":
    main()
