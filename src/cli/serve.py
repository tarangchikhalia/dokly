"""Interactive session for RAG pipeline"""
import asyncio
from src.rag.ollama import OllamaGenerator
from src.utils.config import settings


async def start_session():
    """Start an interactive RAG session.

    Displays configuration information and accepts user questions in a loop.
    Uses the RAG pipeline to generate answers based on indexed documents.
    """
    print("\n" + "="*60)
    print("RAG Pipeline - Interactive Session")
    print("="*60)

    # Display configuration information
    print("\nConfiguration:")
    print(f"  Embedding Model: {settings.EMBED_MODEL_ID}")
    print(f"  LLM Provider:    {settings.LLM_PROVIDER}")
    print(f"  LLM Model:       {settings.LLM_MODEL}")
    print(f"  LLM URL:         {settings.LLM_URL}")
    print(f"  Data Directory:  {settings.DATA_DIR}")
    print(f"  Vector Store:    {settings.VECTOR_STORE_PATH}")
    print("="*60)

    # Initialize the generator (which initializes vector store and retriever)
    print("\nInitializing RAG pipeline...")
    try:
        generator = OllamaGenerator()
        print("RAG pipeline initialized successfully!")
    except Exception as e:
        print(f"\nError initializing RAG pipeline: {e}")
        print("Please ensure:")
        print("  1. Vector store is built with documents")
        print("  2. Ollama is running at the configured URL")
        print("  3. The specified model is available")
        return

    # Display collection information
    try:
        doc_count = generator.vector_store.get_collection_count()
        print(f"Vector store contains {doc_count} document chunks")
    except Exception as e:
        print(f"Warning: Could not get document count: {e}")

    print("\nYou can now ask questions about your documents.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("-"*60)

    # Interactive question loop
    while True:
        try:
            # Get user question
            question = input("\nYour question: ").strip()

            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'q', '']:
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\nEnding session. Goodbye!")
                    break
                else:
                    continue

            # Generate answer using RAG
            print("\nGenerating answer...")
            answer = await generator.generate(question)

            # Display answer
            print("\nAnswer:")
            print("-"*60)
            print(answer)
            print("-"*60)

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing question: {e}")
            print("Please try again or type 'exit' to quit.")
