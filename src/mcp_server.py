"""MCP server exposing the RAG as tools, over Streamable HTTP.

Streamable HTTP is deliberate: it is the only transport Open WebUI supports
natively (Admin Settings -> Integrations -> Add Connection). stdio and SSE would
need Open WebUI's mcpo proxy in front.

Note the trade-off. Registering these tools makes retrieval *agentic* — the model
decides whether and what to retrieve, so answers no longer carry the guarantee that
they were grounded in chunks we chose. Small models (qwen2.5:3b) are also unreliable
tool callers. For chatting in Open WebUI, prefer the OpenAI-compatible facade in
src/api.py, where retrieval always happens. This server's real value is giving other
MCP clients (Claude Code, Claude Desktop) query access to the same corpus.

Written against the MCP Python SDK 2.x API, where FastMCP was renamed to MCPServer.

Run:  python -m src.mcp_server        # serves http://0.0.0.0:9000/mcp
"""

import os

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from . import ask, ingest


HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "9000"))

# The SDK enables DNS-rebinding protection by default with an empty allow-list,
# which rejects every request whose Host header it has not been told about. Clients
# reach this server under several names (localhost from the host, the compose
# service name from another container), so list them explicitly.
ALLOWED_HOSTS = [
    h.strip()
    for h in os.getenv(
        "MCP_ALLOWED_HOSTS",
        f"localhost:{PORT},127.0.0.1:{PORT},rag-mcp:{PORT},host.docker.internal:{PORT}",
    ).split(",")
    if h.strip()
]

mcp = MCPServer(
    "local-rag",
    instructions=(
        "Tools for querying a local collection of indexed documents (PDF, TXT, MD). "
        "Use rag_search to read source passages, rag_ask for a cited answer."
    ),
)


@mcp.tool()
def rag_search(query: str, top_k: int = 6) -> list[dict[str, str]]:
    """Search the indexed documents and return the most relevant passages.

    Returns the raw evidence with its source filename and page number. Does not
    generate an answer — use this when you want to read the source text yourself.
    """
    return [
        {"source": e.source, "page": e.page, "text": e.text}
        for e in ask.retrieve(query, top_k=top_k)
    ]


@mcp.tool()
def rag_ask(question: str) -> dict[str, object]:
    """Answer a question using only the indexed documents, with source citations.

    Says it does not have enough information rather than guessing when the
    documents do not cover the question.
    """
    result = ask.answer_question(question)
    return {
        "answer": result.text,
        "citations": [
            {"source": e.source, "page": e.page, "quote": e.preview}
            for e in result.evidence
        ],
    }


@mcp.tool()
def rag_list_documents() -> list[str]:
    """List the filenames of every document currently indexed."""
    return ingest.list_documents()


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
        # No per-client session state to keep, and Open WebUI is multi-tenant.
        stateless_http=True,
        transport_security=TransportSecuritySettings(allowed_hosts=ALLOWED_HOSTS),
    )
