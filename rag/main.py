"""Main entry point for the RAG pipeline CLI"""
import asyncio
import argparse
from cli.serve import start_session


def main():
    """Main CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Retrieve and generate answers from your documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\nExamples:
        python main.py serve    Start an interactive question-answering session
        """
    )

    parser.add_argument(
        'command',
        choices=['serve'],
        help='Command to execute'
    )

    args = parser.parse_args()

    if args.command == 'serve':
        asyncio.run(start_session())


if __name__ == "__main__":
    main()
