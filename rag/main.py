from ingestion.document_loader import DocumentLoader
from ingestion.chunker import HuggingFaceChunker

def main():
    loader = DocumentLoader()
    for doc in loader.load_all_supported_documents():
        chunker = HuggingFaceChunker()
        chunk = chunker.chunk(doc.document)

if __name__ == "__main__":
    main()
