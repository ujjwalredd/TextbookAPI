"""
Python SDK client for the RAG API.

Usage:
    from textbookapi import Principlesofdatascience
    from textbookapi import Introductiontopythonprogramming

    client1 = Principlesofdatascience(api_key="ujjwal-xxx")
    client2 = Introductiontopythonprogramming(api_key="ujjwal-xxx")

    # Non-streaming
    result = client1.ask("What is data science?")
    print(result.answer)

    # Streaming
    for token in client2.ask("What is a variable?", stream=True):
        print(token, end="", flush=True)
"""

import json
from typing import Generator, Optional

import httpx


class textbookapiError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")


class SourceInfo:
    """A retrieved source passage."""
    def __init__(self, text: str, score: float):
        self.text = text
        self.score = score

    def __repr__(self):
        return f"SourceInfo(score={self.score:.3f}, text='{self.text[:60]}...')"


class QueryResult:
    """Response from a non-streaming query."""
    def __init__(self, answer: str, sources: list[SourceInfo], model: str, id: str):
        self.answer = answer
        self.sources = sources
        self.model = model
        self.id = id

    def __repr__(self):
        return f"QueryResult(answer='{self.answer[:80]}...', sources={len(self.sources)})"


# ── Base client ──────────────────────────────────────────────────────────────

class _BookClient:
    """
    Base client for the RAG API server. Subclassed per book.

    Args:
        api_key: Your API key (from api_keys.json).
        base_url: Server URL (default http://localhost:8000).
        timeout: Request timeout in seconds.
    """

    _book_id: str = ""  # overridden by subclasses

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def ask(
        self,
        question: str,
        stream: bool = False,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Ask a question about the book.

        Returns QueryResult for non-streaming, or a token generator for streaming.
        """
        payload: dict = {"question": question, "book": self._book_id, "stream": stream}
        if top_k is not None:
            payload["top_k"] = top_k
        if temperature is not None:
            payload["temperature"] = temperature

        if stream:
            return self._stream(payload)
        return self._query(payload)

    def _query(self, payload: dict) -> QueryResult:
        resp = self._client.post("/v1/query", json=payload)
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            raise textbookapiError(resp.status_code, detail)
        data = resp.json()
        return QueryResult(
            answer=data["answer"],
            sources=[SourceInfo(s["text"], s["score"]) for s in data.get("sources", [])],
            model=data.get("model", ""),
            id=data.get("id", ""),
        )

    def _stream(self, payload: dict) -> Generator[str, None, None]:
        """Yield tokens from the SSE stream."""
        with self._client.stream("POST", "/v1/query", json=payload) as resp:
            if resp.status_code != 200:
                raise textbookapiError(resp.status_code, "Stream request failed")
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                if "error" in chunk:
                    raise textbookapiError(500, chunk["error"])
                delta = chunk.get("delta", "")
                if delta:
                    yield delta

    def health(self) -> dict:
        """Check server health."""
        resp = self._client.get("/health")
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Per-book clients ─────────────────────────────────────────────────────────

class Principlesofdatascience(_BookClient):
    """Client for 'Principles of Data Science'."""
    _book_id = "principlesofdatascience"


class Introductiontopythonprogramming(_BookClient):
    """Client for 'Introduction to Python Programming'."""
    _book_id = "introductiontopythonprogramming"
