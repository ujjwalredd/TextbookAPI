"""FastAPI server with OpenAI-style API endpoints."""

import os
import json
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse

from .config import RAGConfig, BOOKS
from .engine import RAGEngine, RAGEngineError
from .models import QueryRequest, QueryResponse, StreamChunk, HealthResponse, BookStatus, Source
from .auth import APIKeyManager, require_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("textbookapi.server")

# ── Globals initialized at startup ───────────────────────────────────────────

engines: dict[str, RAGEngine] = {}
key_manager: APIKeyManager | None = None
_auth_dependency = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines, key_manager, _auth_dependency

    base_config = RAGConfig()

    # Auth
    key_manager = APIKeyManager(base_config.api_keys_file)
    _auth_dependency = require_api_key(key_manager)

    # Initialize one engine per book
    for book_id in BOOKS:
        config = RAGConfig.for_book(book_id)
        logger.info(f"Initializing book: {BOOKS[book_id]['title']}")
        logger.info(f"  PDF: {config.pdf_path}")

        eng = RAGEngine(config)
        eng.initialize()
        engines[book_id] = eng
        logger.info(f"  Book '{BOOKS[book_id]['title']}' ready.")

    logger.info(f"Server ready — {len(engines)} book(s) loaded.")

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="RAG API",
    description="Multi-book PDF Q&A powered by local LLM",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Server health check — no auth required."""
    books = []
    for book_id, eng in engines.items():
        books.append(BookStatus(
            name=BOOKS[book_id]["title"],
            status="ready" if eng.ready else "initializing",
            index_size=eng.store.index.ntotal if eng.ready else 0,
        ))

    all_ready = all(eng.ready for eng in engines.values())
    model = next(iter(engines.values())).config.model_name if engines else ""

    return HealthResponse(
        status="ready" if all_ready and engines else "initializing",
        model=model,
        books=books,
    )


@app.post("/v1/query")
async def query(request: QueryRequest):
    """Ask a question about a specific book. Supports streaming via SSE."""
    book_id = request.book.lower()

    if book_id not in engines:
        available = ", ".join(BOOKS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown book '{request.book}'. Available: {available}",
        )

    eng = engines[book_id]
    if not eng.ready:
        raise HTTPException(status_code=503, detail="Engine not ready for this book")

    try:
        if request.stream:
            return StreamingResponse(
                _stream_response(request, eng),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            answer, results = eng.query(
                request.question, stream=False, top_k=request.top_k
            )
            sources = [Source(text=chunk, score=score) for chunk, score in results]
            return QueryResponse(
                answer=answer,
                sources=sources,
                model=eng.config.model_name,
            )
    except RAGEngineError as e:
        raise HTTPException(status_code=503, detail=str(e))


async def _stream_response(request: QueryRequest, eng: RAGEngine):
    """SSE generator — yields data: {json} lines."""
    response_id = f"rag-{uuid.uuid4().hex[:12]}"
    loop = asyncio.get_event_loop()

    try:
        token_gen, results = eng.query(
            request.question, stream=True, top_k=request.top_k
        )

        # Bridge sync generator to async using a queue
        q: asyncio.Queue[str | None] = asyncio.Queue()

        def _produce():
            try:
                for token in token_gen:
                    loop.call_soon_threadsafe(q.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        loop.run_in_executor(None, _produce)

        while True:
            token = await q.get()
            if token is None:
                break
            chunk = StreamChunk(id=response_id, delta=token)
            yield f"data: {chunk.model_dump_json()}\n\n"

        done = StreamChunk(id=response_id, delta="", done=True)
        yield f"data: {done.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except RAGEngineError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ── Auth middleware ──────────────────────────────────────────────────────────

@app.middleware("http")
async def auth_middleware(request, call_next):
    """Validate API key for /v1/* routes."""
    if request.url.path.startswith("/v1/"):
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return _json_response(401, "Missing API key. Use: Authorization: Bearer ujjwal-xxx")
        token = auth[7:]
        if not key_manager or not key_manager.is_valid(token):
            return _json_response(401, "Invalid API key")
    return await call_next(request)


def _json_response(status: int, detail: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status, content={"detail": detail})


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "textbookapi.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
