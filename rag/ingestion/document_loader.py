"""Scans the document directory and loads all document to docling documents"""
import os
from utils.config import settings
from docling.document_converter import DocumentConverter

class DocumentLoader():
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.supported_extensions = settings.SUPPORTED_EXTENSIONS
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
    def load_document(self, file_path):
        converter = DocumentConverter()
        document = converter.convert(file_path)
        return document
        
    def load_all_supported_documents(self):
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                ext = file.split('.')[-1]
                if ext.lower() in self.supported_extensions:
                    path = os.path.join(root, file)
                    document = self.load_document(path)
                    yield document
                    
        
