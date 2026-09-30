"""OpenAI-compatible HTTP facade, so Open WebUI can use the RAG as if it were a model.

Why this shape: Open WebUI only speaks HTTP, and its cleanest integration point is
an OpenAI-style /v1/chat/completions endpoint registered as a Direct Connection.
Retrieval happens here, server-side and unconditionally — the model never decides
whether to retrieve, never sees a tool, and never sees anything but the chunks we
selected. Open WebUI stays a chat frontend, and its own built-in RAG is bypassed.

Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from . import ask, ingest
from .ask import Evidence


# The id Open WebUI will show in its model picker.
logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("RAG_MODEL_ID", "local-rag")

# Diagnostic: dump each incoming request body to disk. Used to discover exactly what
# Open WebUI forwards when a user attaches a file. Off unless CAPTURE_REQUESTS=true.
CAPTURE_REQUESTS = os.getenv("CAPTURE_REQUESTS", "false").lower() == "true"
CAPTURE_PATH = os.getenv("CAPTURE_PATH", "/data/owui_requests.jsonl")

app = FastAPI(
    title="Local RAG — OpenAI-compatible facade",
    description="Grounded question answering over locally indexed documents.",
    version="0.1.0",
)


if CAPTURE_REQUESTS:

    @app.middleware("http")
    async def _capture(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/v1/chat/completions"):
            body = await request.body()
            with open(CAPTURE_PATH, "a", encoding="utf-8") as handle:
                handle.write(body.decode("utf-8", "replace") + "\n")
        return await call_next(request)


# --------------------------------------------------------------------------- #
# Request/response models                                                     #
# --------------------------------------------------------------------------- #

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    # Content may be a plain string or the multimodal list form.
    content: str | list[dict[str, Any]] | None = None

    def as_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n".join(
                str(part.get("text", ""))
                for part in self.content
                if part.get("type") == "text"
            )
        return ""


class ChatCompletionRequest(BaseModel):
    # Open WebUI sends many extra fields (temperature, tools, metadata, ...).
    # We ignore them rather than 422 on an unknown key.
    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _last_user_question(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.as_text().strip()
    return ""


# When a user attaches a file, Open WebUI does not forward the file itself. It
# extracts the text, wraps it in <context><source .../></context>, prepends its own
# "### Task:" instructions, and appends the real question after the closing tag:
#
#   ### Task:
#   ...Open WebUI's instructions...
#   <context>
#   <source id="1" name="report.pdf" resource-type="file" resource-id="<uuid>">TEXT</source>
#   </context>
#
#   Wie lautet das Upload-Codewort?
#
# We harvest that text into Chroma so the upload becomes part of the permanent
# corpus, then answer the real question through our own retrieval — otherwise the
# answer would come from Open WebUI's separate index and ours would never see it.
_CONTEXT_RE = re.compile(r"<context>(.*?)</context>", re.DOTALL)
_SOURCE_RE = re.compile(r"<source\b([^>]*)>(.*?)</source>", re.DOTALL)
_ATTR_RE = re.compile(r'(\S+?)="([^"]*)"')


@dataclass(frozen=True)
class Attachment:
    name: str
    external_id: str | None
    text: str


def _parse_owui_context(content: str) -> tuple[list[Attachment], str]:
    """Split an Open WebUI context prompt into its attachments and the real query.

    Returns ([], "") for anything that is not one of these prompts.
    """
    context_match = _CONTEXT_RE.search(content)
    if not context_match:
        return [], ""

    attachments = []
    for attrs_text, body in _SOURCE_RE.findall(context_match.group(1)):
        attrs = dict(_ATTR_RE.findall(attrs_text))
        attachments.append(
            Attachment(
                name=attrs.get("name") or "attachment",
                external_id=attrs.get("resource-id"),
                text=body.strip(),
            )
        )

    query = content[context_match.end():].strip()
    return attachments, query


# Open WebUI keeps the original upload in its own volume as "<resource-id>_<filename>".
# We mount that read-only, so we can parse the real PDF with PyMuPDF instead of using
# the flattened text Open WebUI sends — which is the only way to keep page numbers.
OWUI_UPLOADS = Path(os.getenv("OWUI_UPLOADS_DIR", "/owui/uploads"))
_READABLE_SUFFIXES = {".pdf", ".txt", ".md"}


def _original_upload(external_id: str | None) -> Path | None:
    """Locate the original file Open WebUI stored for this attachment, if available."""
    if not external_id or not OWUI_UPLOADS.is_dir():
        return None
    for candidate in sorted(OWUI_UPLOADS.glob(f"{external_id}_*")):
        if candidate.suffix.lower() in _READABLE_SUFFIXES:
            return candidate
    return None


def _ingest_attachment(attachment: "Attachment") -> bool:
    """Index an attachment, preferring the original file over Open WebUI's text.

    Falls back to the supplied text when the original is missing or unreadable — for
    file types we cannot parse but Open WebUI can, that fallback is the only path.
    """
    original = _original_upload(attachment.external_id)
    if original is not None:
        try:
            return ingest.ingest_path_as(
                attachment.name, original, attachment.external_id
            )
        except Exception:
            logger.warning("could not parse %s, using Open WebUI text", original)

    return ingest.ingest_text(
        attachment.name, attachment.text, attachment.external_id
    )


def _is_internal_task(messages: list[ChatMessage]) -> bool:
    """Detect Open WebUI's housekeeping calls (chat title / tag generation).

    Open WebUI reuses the selected model to name chats and suggest tags, prefixing
    those prompts with '### Task:'. Routing them through retrieval would be slow and
    would pollute chat titles with document text, so they bypass the RAG.

    Attached-file prompts carry the same '### Task:' prefix, so the presence of a
    <context> block is what separates the two — without that check every uploaded
    document would skip retrieval.
    """
    return any(
        "### Task:" in message.as_text() and "<context>" not in message.as_text()
        for message in messages
    )


def _to_langchain(messages: list[ChatMessage]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        text = message.as_text()
        if message.role == "system":
            converted.append(SystemMessage(content=text))
        elif message.role == "assistant":
            converted.append(AIMessage(content=text))
        else:
            converted.append(HumanMessage(content=text))
    return converted


def _format_citations(evidence: list[Evidence]) -> str:
    if not evidence:
        return ""
    lines = "\n".join(
        f"{e.index}. `{e.source}` — page {e.page}" for e in evidence
    )
    return f"\n\n---\n**Sources**\n{lines}\n"


def _completion_body(text: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


def _chunk_body(completion_id: str, delta: dict[str, Any], finish: str | None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


# --------------------------------------------------------------------------- #
# Endpoints                                                                   #
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only. Deliberately touches nothing — the container health check
    calls this on an interval, and opening Chroma would load the embedding model."""
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, Any]:
    """Richer smoke test: config plus what is currently indexed."""
    return {
        "status": "ok",
        "model": ask.OLLAMA_MODEL,
        "ollama_url": ask.OLLAMA_URL,
        "top_k": ask.TOP_K,
        "indexed_documents": ingest.list_documents(),
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """Open WebUI calls this to populate its model picker."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-rag",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Any:
    if _is_internal_task(request.messages):
        return _completion_body(ask.generate_plain(_to_langchain(request.messages)))

    question = _last_user_question(request.messages)

    # A file attached in the Open WebUI composer arrives as extracted text. Index it,
    # then fall through and answer from our own retrieval like any other question.
    attachments, real_query = _parse_owui_context(question)
    ingested = [a.name for a in attachments if _ingest_attachment(a)]
    if attachments:
        question = real_query

    if not question:
        if ingested:
            return _completion_body(
                f"Added to the index: {', '.join(ingested)}. Ask a question about it."
            )
        return _completion_body("Please ask a question about the indexed documents.")

    if request.stream:
        return StreamingResponse(
            _stream(question),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = ask.answer_question(question)
    return _completion_body(result.text + _format_citations(result.evidence))


def _stream(question: str) -> Iterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    yield _chunk_body(completion_id, {"role": "assistant", "content": ""}, None)

    evidence, tokens = ask.stream_answer(question)
    for token in tokens:
        yield _chunk_body(completion_id, {"content": token}, None)

    citations = _format_citations(evidence)
    if citations:
        yield _chunk_body(completion_id, {"content": citations}, None)

    yield _chunk_body(completion_id, {}, "stop")
    yield "data: [DONE]\n\n"
