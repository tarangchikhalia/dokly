from utils.config import settings
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


class HuggingFaceChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL_ID),
            max_tokens = self.chunk_size,
        )

    def chunk(self, docling_document):
        """
        Chunk a Docling document using Hybrid chunking with HuggingFace tokenizer.

        Args:
            docling_document: A Docling document object

        Returns:
            list: A list of chunks
        """
        chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True
        )
        chunks = list(chunker.chunk(docling_document))
        return chunks
