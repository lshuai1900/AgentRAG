"""ingestion 模块"""
from .embedder import Embedder, get_embedder
from .parser import parse_document, detect_file_type
from .splitter import Chunk, split_pages, split_text

__all__ = [
    "Chunk",
    "Embedder",
    "get_embedder",
    "parse_document",
    "detect_file_type",
    "split_pages",
    "split_text",
]
