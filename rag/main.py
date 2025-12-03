import asyncio
from ingestion.document_loader import DocumentLoader
from ingestion.vector_store import VectorStore
from generation.ollama import OllamaGenerator


async def main():
    # Initialize loader and vector store
    loader = DocumentLoader()
    vector_store = VectorStore()

    # Load all documents
    # print("Loading documents...")
    # all_docs = []
    # for docs in loader.load_all_supported_documents():
    #     all_docs.extend(docs)
    #     print(f"Loaded {len(docs)} documents from {docs[0].metadata.get('source', 'unknown')}")

    # print(f"\nTotal documents loaded: {len(all_docs)}")

    # Build vector database
    # if all_docs:
    #     print("\nBuilding vector database...")
    #     vector_store.build(all_docs)

    #     # Verify the build
    #     count = vector_store.get_collection_count()
    #     print(f"Vector database built successfully with {count} documents")
    # else:
    #     print("No documents to add to vector database")

    # Initialize RAG generator
    print("\n" + "="*50)
    print("Starting RAG Generation")
    print("="*50)

    generator = OllamaGenerator()

    # Example queries to test the RAG system
    queries = [
        "What is llama?",
        "Can you summarize the key points?",
    ]

    for query in queries:
        print(f"\nQuestion: {query}")
        print("-" * 50)
        response = await generator.generate(query)
        print(f"Answer: {response}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
