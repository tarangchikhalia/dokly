"""Vector database management commands"""
from ingestion.document_loader import DocumentLoader
from ingestion.vector_store import VectorStore


def build_vector_store():
    """Build the vector store from scratch by loading all documents.

    This will:
    1. Clear the existing vector store
    2. Load all supported documents from the data directory
    3. Create embeddings and build a new vector store
    """
    print("\n" + "="*60)
    print("Building Vector Store")
    print("="*60)

    try:
        # Initialize document loader
        print("\nInitializing document loader...")
        doc_loader = DocumentLoader()
        print(f"Data directory: {doc_loader.data_dir}")
        print(f"Supported extensions: {', '.join(doc_loader.supported_extensions)}")

        # Initialize vector store
        print("\nInitializing vector store...")
        vector_store = VectorStore()

        # Load all documents
        print("\nLoading documents...")
        all_docs = []
        doc_count = 0

        for docs in doc_loader.load_all_supported_documents():
            all_docs.extend(docs)
            doc_count += 1
            print(f"  Loaded document {doc_count}: {len(docs)} chunks")

        if not all_docs:
            print("\nNo documents found to build vector store.")
            print("Please ensure:")
            print("  1. Documents exist in the data directory")
            print("  2. Documents have supported extensions")
            return

        print(f"\nTotal documents loaded: {doc_count}")
        print(f"Total chunks: {len(all_docs)}")

        # Build vector store
        print("\nBuilding vector store (this may take a while)...")
        vector_store.build(all_docs)

        # Display final count
        final_count = vector_store.get_collection_count()
        print(f"\n{'='*60}")
        print(f"Vector store built successfully!")
        print(f"Total chunks in vector store: {final_count}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nError building vector store: {e}")
        print("Please check your configuration and try again.")
        raise

