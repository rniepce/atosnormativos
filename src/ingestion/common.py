"""
Common utilities shared across ingestion scripts.

This module consolidates the duplicated extract_text, chunk_text, and
build_metadata_from_path helpers that were copy-pasted in every ingest_*.py.
New ingestion scripts should import from here instead of re-implementing.

It also provides centralised embedding functions (sync and async) that use
Azure OpenAI ``text-embedding-3-large`` with 1024 dimensions — the same
model and dimensionality used by the search backend.
"""

import asyncio
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Azure OpenAI embedding constants (must match src/backend/search.py)
# ---------------------------------------------------------------------------

AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
AZURE_EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1024"))


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(file_path: Path, timeout: int = 30) -> Optional[str]:
    """Extract text from .doc/.docx files.

    Uses macOS ``textutil`` when available; falls back to python-docx for
    .docx and ``antiword`` for .doc on Linux.
    """
    import shutil

    # Prefer textutil (macOS)
    if shutil.which("textutil"):
        try:
            result = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(file_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout extracting text from {file_path.name}")
            return None
        except Exception as e:
            logger.warning(f"Error extracting {file_path.name}: {e}")
            return None

    # Fallback: python-docx for .docx
    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        try:
            import docx

            doc = docx.Document(str(file_path))
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return None
        except Exception as e:
            logger.error(f"Error extracting .docx {file_path}: {e}")
            return None

    # Fallback: antiword for .doc
    if suffix == ".doc" and shutil.which("antiword"):
        try:
            result = subprocess.run(
                ["antiword", str(file_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
            )
            return result.stdout
        except Exception as e:
            logger.error(f"antiword failed for {file_path}: {e}")
            return None

    logger.error(f"Cannot extract text from {file_path} on this platform")
    return None


# ---------------------------------------------------------------------------
# Token-aware truncation
# ---------------------------------------------------------------------------

def truncate_to_token_limit(text: str, max_tokens: int = 7000) -> str:
    """Truncate *text* so it stays within the embedding model's token window.

    Uses a simple heuristic: 1 token ~ 4 characters in Portuguese.
    The limit for ``text-embedding-3-large`` is 8191 tokens — we default to
    8000 as a safety margin.
    """
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        logger.warning(
            "Truncating text from %d to %d chars (~%d tokens)",
            len(text), max_chars, max_tokens,
        )
        return text[:max_chars]
    return text


# ---------------------------------------------------------------------------
# Chunking — legal-structure aware
# ---------------------------------------------------------------------------

# Patterns ordered from coarsest to finest legal unit
_LEGAL_SPLIT_PATTERNS = [
    re.compile(r"(?=\nArt(?:igo)?\.?\s*\d+)"),          # Artigos
    re.compile(r"(?=\n§\s*\d+|(?=\nParágrafo))"),        # Parágrafos
    re.compile(r"(?=\n[IVX]+\s*[-–]|\n[a-z]\))"),        # Incisos / alíneas
    re.compile(r"\n\n"),                                  # Dupla quebra de linha
    re.compile(r"\n"),                                    # Quebra simples
]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split *text* into overlapping chunks of approximately *chunk_size* chars.

    The function tries to split along legal-document boundaries (articles,
    paragraphs, incisos) before falling back to plain character cuts.
    """
    text = text.strip()
    if not text:
        return []

    # --- 1. Find the best split pattern that produces >1 part ---------------
    parts: Optional[List[str]] = None
    for pattern in _LEGAL_SPLIT_PATTERNS:
        candidate = [p for p in pattern.split(text) if p.strip()]
        if len(candidate) > 1:
            parts = candidate
            break

    # Fallback: treat the whole text as a single part
    if parts is None:
        parts = [text]

    # --- 2. Accumulate parts into chunks of ~chunk_size ---------------------
    chunks: List[str] = []
    current = ""

    for part in parts:
        # If a single part already exceeds chunk_size, split by character
        if len(part) > chunk_size:
            # Flush what we have so far
            if current.strip():
                chunks.append(current.strip())
                current = ""
            # Character-level split for oversized part
            start = 0
            while start < len(part):
                piece = part[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                start += chunk_size - overlap
            continue

        if len(current) + len(part) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
            # Overlap: carry the tail of the previous chunk
            if overlap > 0 and current:
                current = current[-overlap:] + part
            else:
                current = part
        else:
            current += part

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:chunk_size]]


# ---------------------------------------------------------------------------
# Metadata from filesystem path
# ---------------------------------------------------------------------------

def build_metadata_from_path(file_path: Path) -> Dict[str, Any]:
    """Derive basic metadata (tipo, numero, ano, orgao) from the file path.

    The parent folder name is used as the document type.  Number and year
    are extracted from the filename via regex heuristics.
    """
    folder_name = file_path.parent.name
    filename = file_path.stem.lower()

    metadata: Dict[str, Any] = {
        "tipo": folder_name[:199],
        "numero": "0",
        "ano": 0,
        "orgao": "TJMG",
        "status": "VIGENTE",
        "assunto_resumo": folder_name,
        "tags": [],
    }

    # Try to extract number + year from the filename
    patterns = [
        r"(\d{3,5})(\d{4})$",       # re00011961
        r"(\d{1,4})[_\-](\d{4})",   # port_123_2024
        r"(\d{1,4})(\d{4})$",       # general fallback
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            metadata["numero"] = str(int(match.group(1)))
            year = int(match.group(2))
            if 1900 <= year <= 2030:
                metadata["ano"] = year
            break

    return metadata


# ---------------------------------------------------------------------------
# Embedding — synchronous  (for most ingestion scripts)
# ---------------------------------------------------------------------------

def _embed_with_retry_sync(
    texts: List[str],
    client,
    model: str = AZURE_EMBEDDING_MODEL,
    dimensions: int = AZURE_EMBEDDING_DIMENSIONS,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> List[List[float]]:
    """Call the Azure OpenAI embeddings endpoint with exponential backoff.

    Retries on ``RateLimitError`` and ``APIError``; re-raises all other
    exceptions immediately.
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model,
                dimensions=dimensions,
            )
            return [d.embedding for d in response.data]
        except (openai.RateLimitError, openai.APIError) as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Embedding API error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, delay,
            )
            time.sleep(delay)
    # Should not be reached, but just in case:
    raise RuntimeError("_embed_with_retry_sync exhausted retries")


def get_embeddings(
    texts: List[str],
    client,
    model: str = AZURE_EMBEDDING_MODEL,
    dimensions: int = AZURE_EMBEDDING_DIMENSIONS,
) -> List[List[float]]:
    """Generate embeddings for a list of texts using Azure OpenAI (sync).

    Each text is truncated to the model's token limit before being sent.
    Includes automatic retry with exponential backoff.
    """
    truncated = [truncate_to_token_limit(t) for t in texts]
    return _embed_with_retry_sync(truncated, client, model=model, dimensions=dimensions)


# ---------------------------------------------------------------------------
# Embedding — asynchronous  (for ingest_word_docs / ingest_lc59_ri)
# ---------------------------------------------------------------------------

async def _embed_with_retry_async(
    texts: List[str],
    client,
    model: str = AZURE_EMBEDDING_MODEL,
    dimensions: int = AZURE_EMBEDDING_DIMENSIONS,
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> List[List[float]]:
    """Async version of :func:`_embed_with_retry_sync`."""
    for attempt in range(max_retries):
        try:
            response = await client.embeddings.create(
                input=texts,
                model=model,
                dimensions=dimensions,
            )
            return [d.embedding for d in response.data]
        except (openai.RateLimitError, openai.APIError) as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Embedding API error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("_embed_with_retry_async exhausted retries")


async def get_embeddings_async(
    texts: List[str],
    client,
    model: str = AZURE_EMBEDDING_MODEL,
    dimensions: int = AZURE_EMBEDDING_DIMENSIONS,
) -> List[List[float]]:
    """Generate embeddings for a list of texts using Azure OpenAI (async).

    Each text is truncated to the model's token limit before being sent.
    Includes automatic retry with exponential backoff.
    """
    truncated = [truncate_to_token_limit(t) for t in texts]
    return await _embed_with_retry_async(truncated, client, model=model, dimensions=dimensions)
