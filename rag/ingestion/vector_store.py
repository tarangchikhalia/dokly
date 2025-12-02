"""create and manage a vector store"""
import os
import shutil
from typing import Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from utils.config import settings


class VectorStore:
    def __init__(self):
        """Initialize the VectorStore with connection to ChromaDB.

        """
        self.name = settings.VECTOR_STORE_COLLECTION
        self.persist_directory = settings.VECTOR_STORE_PATH
        self.embedding_model_id = settings.EMBED_MODEL_ID

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_id,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize or connect to existing vector store
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Initialize or connect to the ChromaDB vector store."""
        if os.path.exists(self.persist_directory):
            # Connect to existing vector store
            self.vector_store = Chroma(
                collection_name=self.name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Connected to existing vector store at {self.persist_directory}")
        else:
            # Create new vector store directory
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store = Chroma(
                collection_name=self.name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Created new vector store at {self.persist_directory}")

    def build(self, docs: list[Document]) -> None:
        """Build a vector store from incoming list of Documents.

        This method creates a new vector store from scratch with the provided documents.
        If a vector store already exists, consider using update() instead.

        Args:
            docs: List of Document objects to add to the vector store
        """
        if not docs:
            print("Warning: No documents provided to build vector store")
            return

        # Clear existing collection if it exists
        try:
            self.vector_store.delete_collection()
            print(f"Cleared existing collection: {self.name}")
        except Exception as e:
            print(f"No existing collection to clear: {e}")

        # Reinitialize the vector store
        self.vector_store = Chroma(
            collection_name=self.name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Filter complex metadata before adding documents
        filtered_docs = filter_complex_metadata(docs)

        # Add documents to the vector store
        self.vector_store.add_documents(filtered_docs)
        print(f"Built vector store with {len(filtered_docs)} documents")

    def update(self, docs: list[Document]) -> None:
        """Update existing vector store with new documents.

        This method adds new documents to the existing vector store without
        removing existing documents.

        Args:
            docs: List of Document objects to add to the existing vector store
        """
        if not docs:
            print("Warning: No documents provided to update vector store")
            return

        # Filter complex metadata before adding documents
        filtered_docs = filter_complex_metadata(docs)

        # Add new documents to the existing vector store
        self.vector_store.add_documents(filtered_docs)
        print(f"Updated vector store with {len(filtered_docs)} new documents")

    def retrieve(self, top_k: int, input: str) -> list[Document]:
        """Retrieve top_k documents based on input query.

        Args:
            top_k: Number of top documents to retrieve
            input: Query string to search for similar documents

        Returns:
            List of top_k most similar Document objects
        """
        if not input:
            print("Warning: Empty query provided")
            return []

        # Perform similarity search
        results = self.vector_store.similarity_search(
            query=input,
            k=top_k
        )

        print(f"Retrieved {len(results)} documents for query: '{input}'")
        return results

    def retrieve_with_scores(self, top_k: int, input: str) -> list[tuple[Document, float]]:
        """Retrieve top_k documents with similarity scores.

        Args:
            top_k: Number of top documents to retrieve
            input: Query string to search for similar documents

        Returns:
            List of tuples containing (Document, similarity_score)
        """
        if not input:
            print("Warning: Empty query provided")
            return []

        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score(
            query=input,
            k=top_k
        )

        print(f"Retrieved {len(results)} documents with scores for query: '{input}'")
        return results

    def get_collection_count(self) -> int:
        """Get the total number of documents in the vector store.

        Returns:
            Number of documents in the collection
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return count
        except Exception as e:
            print(f"Error getting collection count: {e}")
            return 0
