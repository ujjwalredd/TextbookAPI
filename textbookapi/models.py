"""Pydantic request/response schemas for the API."""

import time
import uuid
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    book: str = Field(..., min_length=1)
    stream: bool = Field(default=False)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class Source(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"rag-{uuid.uuid4().hex[:12]}")
    answer: str
    sources: list[Source]
    model: str
    created: int = Field(default_factory=lambda: int(time.time()))


class StreamChunk(BaseModel):
    id: str
    delta: str
    done: bool = False


class BookStatus(BaseModel):
    name: str
    status: str
    index_size: int


class HealthResponse(BaseModel):
    status: str
    model: str
    books: list[BookStatus]
