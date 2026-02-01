"""Core RAG engine — PDF extraction, chunking, vector store, LLM queries."""

import os
import json
import hashlib
import pickle
import logging
from typing import Generator

import pymupdf
import faiss
import requests
from sentence_transformers import SentenceTransformer

from .config import RAGConfig

logger = logging.getLogger("textbookapi.engine")


class RAGEngineError(Exception):
    pass


# ── PDF Extraction 

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    num_pages = len(doc)
    pages = [doc[i].get_text() for i in range(num_pages)]
    doc.close()

    text = "".join(pages)
    logger.info(f"Extracted {len(text):,} chars from {num_pages} pages")
    return text


# ── Chunking 

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                chunk = chunk[: break_point + 1]
                end = start + break_point + 1
        chunk = chunk.strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap

    logger.info(f"Created {len(chunks):,} text chunks")
    return chunks


# ── Vector Store 

def _pdf_hash(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read(65536))
        f.seek(0, 2)
        size = f.tell()
        h.update(str(size).encode())
        if size > 65536:
            f.seek(-65536, 2)
            h.update(f.read(65536))
    return h.hexdigest()


class VectorStore:
    def __init__(self, embedding_model: str, cache_dir: str):
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.cache_dir = cache_dir
        self.index = None
        self.chunks: list[str] = []

    def load_or_build(self, chunks: list[str], pdf_path: str):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"index_{_pdf_hash(pdf_path)}.pkl")

        if os.path.exists(cache_file):
            logger.info("Loading cached index from disk")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self.chunks = data["chunks"]
            embeddings = data["embeddings"]
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            logger.info(f"Loaded {self.index.ntotal} vectors from cache")
            return

        self.chunks = chunks
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            chunks, batch_size=128, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors")

        with open(cache_file, "wb") as f:
            pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
        logger.info("Index cached to disk")

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        query_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)
        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx < len(self.chunks)
        ]



def _check_ollama(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/api/tags", timeout=5).status_code == 200
    except requests.ConnectionError:
        return False


def _ensure_model(base_url: str, model: str):
    resp = requests.get(f"{base_url}/api/tags", timeout=10)
    models = [m["name"] for m in resp.json().get("models", [])]
    if any(model in m for m in models):
        logger.info(f"Model '{model}' available")
        return

    logger.info(f"Pulling model '{model}'...")
    pull = requests.post(
        f"{base_url}/api/pull", json={"name": model}, stream=True, timeout=600
    )
    for line in pull.iter_lines():
        if line:
            data = json.loads(line)
            logger.debug(f"Pull: {data.get('status', '')}")
    logger.info(f"Model '{model}' ready")


def _warmup_model(base_url: str, model: str):
    logger.info("Warming up LLM...")
    try:
        requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": "Hi", "options": {"num_predict": 1}, "keep_alive": "30m"},
            timeout=60,
        )
        logger.info("Model loaded into memory")
    except Exception:
        logger.warning("Warmup failed, first query may be slow")


# ── RAG Engine 

class RAGEngine:
    """Encapsulates the full RAG pipeline."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.store: VectorStore | None = None
        self.ready = False

    def initialize(self):
        """Load PDF, build index, warm up model. Call once at startup."""
        cfg = self.config

        if not _check_ollama(cfg.ollama_base_url):
            raise RAGEngineError(
                "Ollama is not running. Install from https://ollama.com "
                "and run: ollama serve"
            )

        _ensure_model(cfg.ollama_base_url, cfg.model_name)
        _warmup_model(cfg.ollama_base_url, cfg.model_name)

        text = extract_text_from_pdf(cfg.pdf_path)
        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)

        self.store = VectorStore(cfg.embedding_model, cfg.cache_dir)
        self.store.load_or_build(chunks, cfg.pdf_path)
        self.ready = True
        logger.info("RAG engine ready")

    def query(self, question: str, stream: bool = False, top_k: int | None = None):
        """
        Ask a question. Returns (answer, results) or (token_generator, results).
        results is a list of (chunk, score) tuples.
        """
        if not self.ready:
            raise RAGEngineError("Engine not initialized")

        k = top_k or self.config.top_k
        results = self.store.search(question, k)
        context = "\n\n".join(chunk for chunk, _ in results)

        if stream:
            return self._stream(question, context), results
        else:
            tokens = list(self._stream(question, context))
            return "".join(tokens), results

    def _stream(self, question: str, context: str) -> Generator[str, None, None]:
        """Yield tokens from Ollama one at a time."""
        cfg = self.config

        book_title = self.config.book_title
        system = (
            f"You are a helpful assistant for the book \"{book_title}\". "
            "Your job is to answer questions ONLY about this book using the provided context. "
            "Rules:\n"
            "- If the user sends a greeting (hi, hello, hey, etc.), reply with a friendly greeting "
            f"and say: \"Ask me anything about the book '{book_title}'!\"\n"
            "- If the user says bye/goodbye, reply with a friendly goodbye.\n"
            "- For all other questions, answer ONLY using the context provided below. "
            "If the answer is not in the context, say \"I couldn't find that in the book.\"\n"
            "- Be concise and accurate."
        )

        prompt = f"Context from the book:\n---\n{context}\n---\n\nUser: {question}\n\nAssistant:"

        try:
            resp = requests.post(
                f"{cfg.ollama_base_url}/api/generate",
                json={
                    "model": cfg.model_name,
                    "prompt": prompt,
                    "system": system,
                    "stream": True,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "num_predict": cfg.max_tokens,
                        "num_ctx": cfg.context_window,
                    },
                },
                stream=True,
                timeout=120,
            )
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        return
        except requests.ConnectionError:
            raise RAGEngineError("Cannot connect to Ollama")
        except Exception as e:
            raise RAGEngineError(f"LLM query failed: {e}")
