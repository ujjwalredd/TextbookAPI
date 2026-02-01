from dataclasses import dataclass
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTBOOKS_DIR = os.path.join(BASE_DIR, "textbooks")

# ── Book registry ────────────────────────────────────────────────────────────

BOOKS = {
    "principlesofdatascience": {
        "title": "Principles of Data Science",
        "pdf": "Principles-of-Data-Science-WEB.pdf",
    },
    "introductiontopythonprogramming": {
        "title": "Introduction to Python Programming",
        "pdf": "Introduction_to_Python_Programming_-_WEB.pdf",
    },
}


def get_book_pdf_path(book_id: str) -> str:
    return os.path.join(TEXTBOOKS_DIR, BOOKS[book_id]["pdf"])


def get_book_title(book_id: str) -> str:
    return BOOKS[book_id]["title"]


# ── Shared config ────────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:3b"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3

    # Book (set per-engine)
    book_title: str = ""
    pdf_path: str = ""
    cache_dir: str = ""

    # LLM generation options
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 384
    context_window: int = 2048

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys_file: str = ""

    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = os.path.join(BASE_DIR, ".cache")
        if not self.api_keys_file:
            self.api_keys_file = os.path.join(BASE_DIR, "api_keys.json")

    @staticmethod
    def for_book(book_id: str) -> "RAGConfig":
        """Create a RAGConfig pre-filled for a specific book."""
        return RAGConfig(
            book_title=get_book_title(book_id),
            pdf_path=get_book_pdf_path(book_id),
        )
