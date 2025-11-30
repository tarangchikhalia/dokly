from ingestion.document_loader import DocumentLoader
def main():
    loader = DocumentLoader()
    for result in loader.load_all_supported_documents():
        print(result)

if __name__ == "__main__":
    main()
