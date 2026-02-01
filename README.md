# TextbookAPI

A retrieval-augmented generation (RAG) system that serves PDF textbooks as conversational Q&A APIs. It extracts content from PDF files, builds vector indexes for semantic search, and answers questions using a local LLM through Ollama. The system exposes a FastAPI server with streaming support and ships a Python SDK client for integration into any application.

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Python SDK](#python-sdk)
9. [Adding a New Book](#adding-a-new-book)
10. [Development](#development)
11. [Contributing](#contributing)
12. [License](#license)

## Architecture

```
                          Server
 ┌──────────────────────────────────────────────────┐
 │  FastAPI :8000                                   │
 │                                                  │
 │  ┌────────────────┐    ┌───────────────────────┐ │
 │  │  RAGEngine #1  │    │  RAGEngine #2         │ │
 │  │  FAISS Index   │    │  FAISS Index          │ │
 │  │  (Book A)      │    │  (Book B)             │ │
 │  └───────┬────────┘    └──────────┬────────────┘ │
 │          │                        │              │
 │          └───────┬────────────────┘              │
 │                  │                               │
 │          ┌───────▼────────┐                      │
 │          │  Ollama LLM    │                      │
 │          │  (qwen2.5:3b)  │                      │
 │          └────────────────┘                      │
 └──────────────────────┬──────────────────────────-┘
                        │ HTTP / SSE
                        │
 ┌──────────────────────▼──────────────────────────-┐
 │  Client Application                              │
 │  from textbookapi import Principlesofdatascience  │
 └──────────────────────────────────────────────────┘
```

The pipeline works as follows:

1. PDF text is extracted using PyMuPDF and split into overlapping chunks.
2. Chunks are embedded with `all-MiniLM-L6-v2` (sentence-transformers) and stored in a FAISS index.
3. On query, the question is embedded and the top matching chunks are retrieved.
4. Retrieved context and the question are sent to Ollama, which generates a streamed response.

Embeddings and FAISS indexes are cached to disk. Subsequent startups skip the embedding step entirely.

## Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| Ollama | Latest | Local LLM inference |
| pip | Latest | Package management |

Install Ollama from [https://ollama.com](https://ollama.com) and verify it is running:

```
ollama serve
```

The server will automatically pull the required model (`qwen2.5:3b`) on first startup if it is not already available.

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/your-username/textbookapi.git
cd textbookapi
pip install -r requirements.txt
```

### Dependencies

```
PyMuPDF>=1.24.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
requests>=2.31.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
httpx>=0.25.0
```

## Quick Start

### 1. Start Ollama

```
ollama serve
```

### 2. Start the API Server

```
python -m textbookapi.server
```

The server initializes a RAG engine for each registered book. On first run, this includes extracting text from PDFs, generating embeddings, and building FAISS indexes. The indexes are cached to `.cache/` for fast subsequent startups.

An API key is auto-generated on first run and written to `api_keys.json`. The key is printed in the server logs.

### 3. Query via the Python SDK

```python
from textbookapi import Principlesofdatascience

client = Principlesofdatascience(api_key="ujjwal-your-key-here")

# Streaming
for token in client.ask("What is data science?", stream=True):
    print(token, end="", flush=True)

# Non-streaming
result = client.ask("Explain supervised learning")
print(result.answer)
print(result.sources)
```

### 4. Run the Interactive Chatbot

```
python Test.py
```

This presents a book selection menu and opens an interactive chat session.

### 5. Direct Mode (No Server)

```
python LLM.py
```

This runs the RAG pipeline directly without the API server, useful for local testing.

## Project Structure

```
textbookapi/
    __init__.py         Package exports
    config.py           Book registry and RAGConfig dataclass
    engine.py           PDF extraction, chunking, FAISS vector store, Ollama queries
    server.py           FastAPI application with SSE streaming
    client.py           Python SDK with per-book client classes
    models.py           Pydantic request and response schemas
    auth.py             API key management and authentication middleware

textbooks/
    *.pdf               PDF textbook files

LLM.py                  CLI entry point for direct RAG queries
Test.py                 Interactive chatbot using the Python SDK
requirements.txt        Python dependencies
api_keys.json           Auto-generated API keys (created on first run)
.cache/                 Cached FAISS indexes (created on first run)
```

## Configuration

All configuration is centralized in `textbookapi/config.py`.

### Book Registry

Books are registered in the `BOOKS` dictionary:

```python
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
```

### RAG Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `qwen2.5:3b` | Ollama model for generation |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `chunk_size` | `1000` | Characters per text chunk |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |
| `top_k` | `3` | Number of passages retrieved per query |
| `temperature` | `0.3` | LLM sampling temperature |
| `max_tokens` | `384` | Maximum tokens in LLM response |
| `context_window` | `2048` | Ollama context window size |

## API Reference

### Health Check

```
GET /health
```

No authentication required. Returns server status and the state of all loaded books.

**Response:**

```json
{
  "status": "ready",
  "model": "qwen2.5:3b",
  "books": [
    {"name": "Principles of Data Science", "status": "ready", "index_size": 842},
    {"name": "Introduction to Python Programming", "status": "ready", "index_size": 615}
  ]
}
```

### Query

```
POST /v1/query
Authorization: Bearer ujjwal-your-key-here
Content-Type: application/json
```

**Request Body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | The question to ask (1 to 2000 characters) |
| `book` | string | Yes | Book identifier from the registry |
| `stream` | boolean | No | Enable server-sent events streaming (default: false) |
| `top_k` | integer | No | Number of passages to retrieve (1 to 10) |
| `temperature` | float | No | Sampling temperature (0.0 to 2.0) |

**Non-streaming Response:**

```json
{
  "id": "rag-a1b2c3d4e5f6",
  "answer": "Data science is an interdisciplinary field...",
  "sources": [
    {"text": "Chapter 1 introduces...", "score": 0.847}
  ],
  "model": "qwen2.5:3b",
  "created": 1706745600
}
```

**Streaming Response (SSE):**

```
data: {"id":"rag-a1b2c3d4e5f6","delta":"Data","done":false}
data: {"id":"rag-a1b2c3d4e5f6","delta":" science","done":false}
...
data: {"id":"rag-a1b2c3d4e5f6","delta":"","done":true}
data: [DONE]
```

### Authentication

All `/v1/*` endpoints require a Bearer token in the `Authorization` header. Keys are stored in `api_keys.json` and auto-generated on first server startup with the prefix `ujjwal-`.

## Python SDK

The SDK provides one client class per registered book. All classes share the same interface.

### Available Clients

| Class | Book |
|---|---|
| `Principlesofdatascience` | Principles of Data Science |
| `Introductiontopythonprogramming` | Introduction to Python Programming |

### Initialization

```python
from textbookapi import Principlesofdatascience

client = Principlesofdatascience(
    api_key="ujjwal-your-key-here",
    base_url="http://localhost:8000",  # optional, this is the default
    timeout=120.0,                     # optional, request timeout in seconds
)
```

### Asking Questions

```python
# Non-streaming: returns a QueryResult object
result = client.ask("What is a random variable?")
print(result.answer)
print(result.sources)   # list of SourceInfo objects
print(result.model)

# Streaming: returns a generator that yields tokens
for token in client.ask("Explain Bayes theorem", stream=True):
    print(token, end="", flush=True)

# With optional parameters
result = client.ask("What is regression?", top_k=5, temperature=0.5)
```

### Health Check

```python
status = client.health()
print(status)
```

### Resource Management

```python
# Manual cleanup
client.close()

# Or use as a context manager
with Principlesofdatascience(api_key="ujjwal-...") as client:
    result = client.ask("What is clustering?")
```

### Error Handling

```python
from textbookapi import Principlesofdatascience, textbookapiError

try:
    result = client.ask("What is PCA?")
except textbookapiError as e:
    print(e.status_code)
    print(e.message)
```

## Adding a New Book

To add a new textbook to the system:

1. Place the PDF file in the `textbooks/` directory.

2. Register the book in `textbookapi/config.py`:

```python
BOOKS = {
    # ... existing books ...
    "yourbook": {
        "title": "Your Book Title",
        "pdf": "Your-Book-Filename.pdf",
    },
}
```

3. Add a client class in `textbookapi/client.py`:

```python
class Yourbook(_BookClient):
    """Client for 'Your Book Title'."""
    _book_id = "yourbook"
```

4. Export the class in `textbookapi/__init__.py`:

```python
from .client import (
    Principlesofdatascience,
    Introductiontopythonprogramming,
    Yourbook,
    textbookapiError,
    QueryResult,
    SourceInfo,
)
```

5. Restart the server. The new book will be indexed on first startup and cached for subsequent runs.

## Development

### Running the Server in Development

```
python -m textbookapi.server
```

The server runs on `http://localhost:8000` by default. API documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

### Testing with curl

```
# Health check
curl http://localhost:8000/health

# Non-streaming query
curl -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer ujjwal-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is data science?", "book": "principlesofdatascience"}'

# Streaming query
curl -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer ujjwal-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is data science?", "book": "principlesofdatascience", "stream": true}'
```

### Clearing the Cache

Delete the `.cache/` directory to force re-indexing of all books on next startup:

```
rm -rf .cache/
```

## Contributing

Contributions are welcome. Please follow the guidelines below to keep the project consistent and maintainable.

### Getting Started

1. Fork the repository.
2. Create a feature branch from `main`:
   ```
   git checkout -b feature/your-feature-name
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make your changes.
5. Test your changes locally by starting the server and running queries.
6. Commit with a clear, descriptive message:
   ```
   git commit -m "Add support for custom embedding models"
   ```
7. Push to your fork and open a pull request.

### Guidelines

**Code Style**
  * Follow PEP 8 conventions.
  * Use type hints for function signatures.
  * Keep functions focused and short. If a function exceeds 40 lines, consider splitting it.

**Commits**
  * Write commit messages in the imperative mood ("Add feature", not "Added feature").
  * Each commit should represent a single logical change.
  * Reference issue numbers in commit messages where applicable.

**Pull Requests**
  * Provide a clear description of what the PR changes and why.
  * Keep PRs focused. One feature or fix per PR.
  * Ensure the server starts and responds to queries before submitting.
  * Update the README if your change affects usage, configuration, or the public API.

**Adding Books**
  * Follow the steps in [Adding a New Book](#adding-a-new-book).
  * Do not commit PDF files to the repository. Add them to `.gitignore`.
  * Only commit the configuration and client class changes.

**Reporting Issues**
  * Use GitHub Issues to report bugs or request features.
  * Include the Python version, OS, Ollama version, and full error traceback when reporting bugs.
  * Describe the expected behavior and the actual behavior.

### Areas for Contribution

The following areas are open for improvement:

  * **Testing**: Unit tests for the engine, server endpoints, and client SDK.
  * **Embedding models**: Support for alternative embedding models beyond `all-MiniLM-L6-v2`.
  * **LLM backends**: Support for OpenAI API, Anthropic API, or other inference providers alongside Ollama.
  * **Document formats**: Support for EPUB, DOCX, or HTML in addition to PDF.
  * **Chunk strategies**: Semantic chunking, section-aware splitting, or table extraction.
  * **Async engine**: Converting the synchronous RAG engine to fully async for better server performance.
  * **Docker**: Containerized deployment with Ollama bundled.
  * **Rate limiting**: Per-key rate limiting on the API.

## License
MIT License
