from ingestion.document_loader import DocumentLoader
from ingestion.chunker import HuggingFaceChunker

def main():
    loader = DocumentLoader()
    for docs in loader.load_all_supported_documents():
        print(docs[0].metadata)
      

if __name__ == "__main__":
    main()
