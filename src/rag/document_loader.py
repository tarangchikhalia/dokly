"""Scans the document directory and loads all document to docling documents"""
import os
from typing import Generator
from src.utils.config import settings
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


class DocumentLoader():
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.supported_extensions = settings.SUPPORTED_EXTENSIONS
        self.export_types = ExportType.DOC_CHUNKS
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
    def load_document(self, file_path: str) -> list[Document]:
        loader = DoclingLoader(
                                    file_path=file_path,
                                    export_type=self.export_types,
                                    chunker = HybridChunker(tokenizer = self._get_tokenizer())
                                    )
        docs = loader.load()
        return docs
        
    def load_all_supported_documents(self) -> Generator[list[Document], None, None]:
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                ext = file.split('.')[-1]
                if ext.lower() in self.supported_extensions:
                    path = os.path.join(root, file)
                    docs = self.load_document(path)
                    yield docs
    
    def _get_tokenizer(self) -> HuggingFaceTokenizer:
        tokenizer = HuggingFaceTokenizer(
            tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL_ID),
        )
        return tokenizer
        
