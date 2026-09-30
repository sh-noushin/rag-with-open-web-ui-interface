# Retrieval + answering: cosine search in Chroma, then Ollama for the answer.
#
# The functions here are deliberately split so that both the CLI and the HTTP
# facade in src/api.py reuse the same retrieval and prompting logic:
#
#   retrieve()        -> the evidence, as structured records
#   build_messages()  -> the grounded prompt for a question + evidence
#   answer_question() -> evidence + generated answer (no printing)
#   stream_answer()   -> evidence, then the answer token by token
#   answer()          -> the CLI's printing wrapper
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .ingest import store


TOP_K = int(os.getenv("TOP_K", "6"))
# Chroma returns exactly TOP_K nearest neighbours whether or not they are relevant,
# so without a floor every answer cites TOP_K sources — mostly noise. On this corpus
# an on-topic chunk scores ~0.7-0.8 and unrelated ones ~0.2-0.3, so 0.4 separates
# them cleanly. Raise it if irrelevant sources still appear; lower it if valid
# sources go missing.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.4"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = "You are a helpful assistant for document-based question answering."


@dataclass(frozen=True)
class Evidence:
    """One retrieved chunk, with the metadata needed to cite it."""

    index: int
    source: str
    page: str
    text: str
    score: float

    @property
    def preview(self) -> str:
        return " ".join(self.text.split())[:150]


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    evidence: list[Evidence]


@lru_cache(maxsize=1)
def _model() -> ChatOllama:
    return ChatOllama(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_ctx=4096,
        keep_alive="24h",
    )


def _display_page(raw: object) -> str:
    """PyMuPDF numbers pages from 0; PDF readers show them from 1.

    Without this shift every citation is off by one against what the user sees when
    they open the document to check it. Text ingested without page structure (an
    Open WebUI upload we could not parse ourselves) has no page at all.
    """
    if raw is None or raw == "":
        return "?"
    try:
        return str(int(raw) + 1)
    except (TypeError, ValueError):
        return str(raw)


def retrieve(question: str, top_k: int | None = None) -> list[Evidence]:
    """Return the relevant chunks — not simply the top_k nearest ones.

    Anything scoring below SCORE_THRESHOLD is dropped, so a question the corpus does
    not cover yields no evidence at all rather than a list of unrelated chunks.
    """
    scored = store().similarity_search_with_relevance_scores(
        question, k=top_k or TOP_K
    )
    keep = [(d, s) for d, s in scored if s >= SCORE_THRESHOLD]
    return [
        Evidence(
            index=i,
            source=str(d.metadata.get("source", "?")),
            page=_display_page(d.metadata.get("page")),
            text=d.page_content,
            score=float(s),
        )
        for i, (d, s) in enumerate(keep, 1)
    ]


def build_messages(question: str, evidence: list[Evidence]) -> list:
    context = "\n\n".join(
        f"[Source {e.index}] (page {e.page} of {e.source})\n{e.text}" for e in evidence
    )
    combined_input = (
        f"Answer the following question using only the documents below. "
        f"If the answer is not in the documents, say you don't have enough information. "
        f"Answer in the same language as the question.\n\n"
        f"Documents:\n{context}\n\n"
        f"Question: {question}"
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=combined_input),
    ]


NO_EVIDENCE_TEXT = (
    "I could not find anything relevant to that question in the indexed documents."
)


def answer_question(question: str, top_k: int | None = None) -> Answer:
    """Retrieve, then generate. Returns the answer instead of printing it."""
    evidence = retrieve(question, top_k)
    if not evidence:
        return Answer(question=question, text=NO_EVIDENCE_TEXT, evidence=[])
    result = _model().invoke(build_messages(question, evidence))
    return Answer(question=question, text=str(result.content), evidence=evidence)


def stream_answer(
    question: str, top_k: int | None = None
) -> tuple[list[Evidence], Iterator[str]]:
    """Retrieve up front, then hand back a token iterator for the answer.

    Retrieval is done eagerly so callers can show sources before generation ends.
    """
    evidence = retrieve(question, top_k)
    if not evidence:
        return [], iter([NO_EVIDENCE_TEXT])

    def tokens() -> Iterator[str]:
        for part in _model().stream(build_messages(question, evidence)):
            text = part.content
            if text:
                yield str(text)

    return evidence, tokens()


def generate_plain(messages: list) -> str:
    """Call the LLM with no retrieval and no grounding prompt.

    Used only for Open WebUI's internal housekeeping calls (chat title and tag
    generation), which are not document questions and must not hit retrieval.
    """
    return str(_model().invoke(messages).content)


def answer(question: str) -> None:
    result = answer_question(question)

    if not result.evidence:
        print("\nNo relevant chunks found.\n")
        return

    print(f"\n--- Retrieved {len(result.evidence)} chunks ---")
    for e in result.evidence:
        print(f"  [{e.index}] {e.source} (page {e.page}): {e.preview}...")

    print(f"\nAnswer: {result.text}\n")
