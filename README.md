# Minimal Local RAG

A small, local-first Retrieval-Augmented Generation app over your PDFs, TXT and Markdown files.

- **Chroma** (embedded — no separate container) storing dense vectors
- **FastEmbed** running a local embedding model, `jinaai/jina-embeddings-v2-base-de` by default (German-specialized, multilingual)
- **Ollama** on the Windows host for a local Qwen LLM
- Answers grounded in retrieved chunks, with source and page citations
- Three front doors: an interactive CLI, an **Open WebUI** chat UI, and an **MCP server**

> Retrieval is dense/semantic, not BM25. Questions can be phrased in your own words —
> they do not have to reuse the documents' exact wording.

## Prerequisites

Docker Desktop, plus Ollama installed on the Windows host (the containers reach it
through `host.docker.internal`). Pull a model once:

```powershell
ollama pull qwen2.5:7b
```

Set `OLLAMA_MODEL` in `.env` to use a different one. Copy `.env.example` to `.env` to
override any default.

## The three ways to use it

```powershell
docker compose up -d --build     # starts rag-api + Open WebUI
```

| | Command | Where |
| --- | --- | --- |
| Open WebUI chat | `docker compose up -d` (default) | http://localhost:3002 |
| Interactive CLI | `docker compose run --rm app` | terminal |
| MCP server | `docker compose --profile mcp up -d` | http://localhost:9000/mcp |

Ports 8010 / 3002 are used because `:8000` and `:3000` are already taken on this
machine by the separate `rag-pipeline` stack and another Open WebUI instance.
Override with `RAG_API_PORT` / `OPEN_WEBUI_PORT` in `.env`.

## Uploading documents from Open WebUI

Click **+** in the chat composer, pick a PDF, ask a question. The document is added to
the Chroma index **permanently** — later questions in any other chat find it, with no
attachment and no CLI.

This does not happen by itself. Open WebUI's file upload normally feeds Open WebUI's
*own* separate vector store, which would bypass this RAG entirely. Two pieces make it
work:

- `BYPASS_EMBEDDING_AND_RETRIEVAL=true` (set in compose) makes Open WebUI hand over the
  **full** extracted text of the attached file instead of snippets its own retriever
  picked.
- `src/api.py` parses the `<context><source>...</source></context>` block out of the
  incoming request to learn what was attached, ingests it, and then answers the real
  question through normal retrieval.
- Open WebUI's storage is mounted **read-only** at `/owui`. Uploads are kept there as
  `<resource-id>_<filename>`, so `rag-api` reads the **original PDF** and parses it with
  PyMuPDF rather than trusting Open WebUI's flattened text. That is what preserves page
  numbers in citations. The text from the request is only a fallback, used when the
  original is missing or is a format PyMuPDF cannot read.

Re-uploading the same file is safe: deduplication is on a hash of the content, because
Open WebUI issues a new `resource-id` on every upload and re-sends the whole context on
every turn of a chat.

Page numbers are shifted to 1-based on display — PyMuPDF counts from 0, PDF readers
count from 1, and an off-by-one makes a citation impossible to check. A citation
showing `page ?` means that document was indexed through the text fallback and has no
page structure.

To delete an uploaded document from the index:

```powershell
docker compose run --rm -T --entrypoint python app -c "from src.ingest import store; store().delete(where={'source': 'report.pdf'})"
```

### 1. Interactive CLI

```powershell
docker compose run --rm app
```

It first prompts repeatedly for a file to ingest (Enter on an empty line to move on),
then loops asking questions. Inside the container your documents are at
`/knowledge/<name>.pdf` — a read-only mount of your AiKnowledge folder.

There are no CLI subcommands — the prompt flow is the whole interface.

### 2. Open WebUI

```powershell
docker compose up -d
```

Open http://localhost:3002. `local-rag` is preselected in the model picker; ask a
question and you get the grounded answer with a **Sources** list underneath.

How it works: `src/api.py` exposes an OpenAI-compatible `/v1/chat/completions`, and
Open WebUI is registered against it as an OpenAI connection. **Retrieval always runs
server-side**, before the model is called — the model cannot choose to skip it, cannot
call tools, and never sees anything but the chunks we selected. Open WebUI's own
built-in RAG, Ollama passthrough and web search are all disabled in compose so nothing
can bypass or duplicate the Chroma index.

`WEBUI_AUTH=false` skips the login screen for local single-user use; remove it to get
the normal signup/login flow.

**Already running Open WebUI elsewhere?** Skip this service and point your existing
instance at the API instead: start only `docker compose up -d rag-api`, then in that
Open WebUI go to **Settings → Connections → OpenAI API** and add
`http://host.docker.internal:8010/v1` with any non-empty API key.

**Expect slow answers.** Measured on this machine with `qwen2.5:7b` and `TOP_K=6`:
roughly **4 minutes per question**, warm — and that is generation on CPU, not a
one-off cold start. Retrieval itself is fast; the time goes into the LLM chewing
through a ~2500-token context. Levers, in order of effect:

- **Use a smaller model.** `ollama pull qwen2.5:3b`, then set `OLLAMA_MODEL=qwen2.5:3b`
  in `.env`. A 3B model is roughly 2-3x faster than 7B on CPU.
- **Retrieve fewer chunks.** `TOP_K=3` roughly halves the prompt.
- **Shorten chunks.** `CHUNK_SIZE=800` (requires re-ingesting to take effect).
- **Use a GPU** via Ollama on the host, which is by far the biggest win if available.

Streaming is enabled, so tokens appear as they are produced rather than after a
4-minute blank wait — but the first token still lags by however long retrieval plus
prompt processing takes.

### 3. MCP server

```powershell
docker compose --profile mcp up -d
```

Serves Streamable HTTP at `http://localhost:9000/mcp` with three tools:

| Tool | Purpose |
| --- | --- |
| `rag_search` | return the most relevant passages, with source + page |
| `rag_ask` | return a cited answer |
| `rag_list_documents` | list indexed filenames |

Register it in Open WebUI under **Settings → Admin → Integrations → + Add Connection**,
type **MCP (Streamable HTTP)**. Streamable HTTP is the only transport Open WebUI
supports natively; stdio and SSE servers need Open WebUI's `mcpo` proxy in front.

**Prefer the Open WebUI path above for chat.** Registering these tools makes retrieval
*agentic* — the model decides whether and what to retrieve, so answers lose the
guarantee that they were grounded in chunks we picked, and small models are unreliable
tool callers. The MCP server's real value is giving other MCP clients (Claude Code,
Claude Desktop) query access to the same corpus.

Two gotchas worth knowing:

- The MCP Python SDK is now **2.x**, where `FastMCP` was renamed to `MCPServer`. Code
  written against the 1.x `mcp.server.fastmcp` import will not run.
- The SDK enables **DNS-rebinding protection by default with an empty allow-list**, so
  every hostname a client uses must be listed in `MCP_ALLOWED_HOSTS` or requests are
  rejected. The defaults cover localhost, `rag-mcp` and `host.docker.internal`.

## Run without Docker

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:7b

python -m src.cli                                        # interactive CLI
uvicorn src.api:app --host 0.0.0.0 --port 8010           # HTTP facade
python -m src.mcp_server                                 # MCP server
```

Set `CHROMA_DIR` and `FASTEMBED_CACHE_DIR` to writable local paths first — the
defaults (`/data/...`) only exist inside the container.

## Checking it works

```powershell
curl http://localhost:8010/status          # config + list of indexed documents
curl http://localhost:8010/v1/models       # what Open WebUI sees
```

## Files

| File | Purpose |
| --- | --- |
| `src/ingest.py`     | Load PDF/TXT/MD, split, embed, persist into Chroma |
| `src/ask.py`        | Retrieval, grounded prompt, and answer generation |
| `src/cli.py`        | Interactive command-line entry point |
| `src/api.py`        | OpenAI-compatible HTTP facade for Open WebUI |
| `src/mcp_server.py` | MCP server exposing the RAG as tools |

## Stopping and resetting

```powershell
docker compose --profile mcp down
docker compose --profile mcp down -v   # also wipes the index
```

Data lives in the `rag_data` volume (Chroma index + embedding model cache) and
`rag_open_webui_data` (Open WebUI chats and settings).
