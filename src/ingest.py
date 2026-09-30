# Ingestion: load a PDF or text file by path, split with LangChain, embed with a
# free local model via FastEmbed, and persist into a Chroma vector store.
import hashlib
import os
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/chroma")
DENSE_MODEL = os.getenv("DENSE_MODEL", "jinaai/jina-embeddings-v2-base-de")
CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR", "/data/fastembed_cache")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))


@lru_cache(maxsize=1)
def _embedder() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=DENSE_MODEL, cache_dir=CACHE_DIR)


@lru_cache(maxsize=1)
def store() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=_embedder(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _load(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyMuPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"unsupported file type '{suffix}' (only .pdf, .txt, .md)")


def _source_exists(source: str) -> bool:
    result = store().get(where={"source": source}, limit=1)
    return len(result.get("ids", [])) > 0


def ingest_file(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        print(f"[error] {path_str}: file not found")
        return
    if path.is_dir():
        print(f"[error] {path_str}: is a directory, expected a file")
        return
    if _source_exists(path.name):
        print(f"[skip] {path_str}: already indexed")
        return

    try:
        docs = _load(path)
    except ValueError as e:
        print(f"[error] {path_str}: {e}")
        return

    print(f"  loaded {len(docs)} pages")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = path.name

    print(f"  encoding {len(chunks)} chunks (this can take several minutes on CPU)...")
    store().add_documents(chunks)
    print(f"[ok] {path_str}: {len(chunks)} chunks indexed")


def _content_exists(content_hash: str) -> bool:
    result = store().get(where={"content_hash": content_hash}, limit=1)
    return len(result.get("ids", [])) > 0


def ingest_text(name: str, text: str, external_id: str | None = None) -> bool:
    """Index text that arrived from outside the filesystem — e.g. a file a user
    attached in Open WebUI, whose extracted content reaches us in the request body.

    Deduplication is on a hash of the content, not on `external_id`: Open WebUI
    re-sends the full context on every turn of a chat (so we must not re-index per
    message), and it mints a fresh resource id on every upload (so the same PDF
    uploaded twice must not become two copies). `external_id` is kept only for
    traceability. Returns True only if the text was newly indexed.
    """
    text = text.strip()
    if not text:
        return False

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if _content_exists(content_hash):
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents(
        [text],
        metadatas=[
            {"source": name, "content_hash": content_hash, "external_id": external_id or ""}
        ],
    )
    store().add_documents(chunks)
    return True


def ingest_path_as(name: str, path: Path, external_id: str | None = None) -> bool:
    """Index a file that already exists on disk, but under a different display name.

    Used for Open WebUI uploads: the original PDF sits in Open WebUI's volume, so we
    read it with PyMuPDF ourselves instead of accepting Open WebUI's flattened text.
    That is what preserves page numbers in citations. Deduplicates on content hash,
    like ingest_text. Returns True only if it was newly indexed.
    """
    docs = _load(path)
    full_text = "\n".join(d.page_content for d in docs).strip()
    if not full_text:
        return False

    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    if _content_exists(content_hash):
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        # Keep the page number the loader recorded; replace the on-disk path with the
        # name the user actually uploaded.
        chunk.metadata["source"] = name
        chunk.metadata["content_hash"] = content_hash
        chunk.metadata["external_id"] = external_id or ""

    store().add_documents(chunks)
    return True


def list_documents() -> list[str]:
    all_metas = store().get(include=["metadatas"]).get("metadatas", [])
    return sorted({m.get("source", "?") for m in all_metas if m})
