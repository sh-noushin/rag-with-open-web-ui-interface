# App image: shared by the CLI, the HTTP facade, and the MCP server.
# Ollama runs separately (on the host by default — see docker-compose.yml).
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV PYTHONUNBUFFERED=1 \
    OLLAMA_URL=http://ollama:11434 \
    QDRANT_PATH=/data/qdrant \
    HF_HOME=/data/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1

# CMD, not ENTRYPOINT, so compose services can override the process
# (uvicorn for the API, src.mcp_server for MCP) while `docker compose run --rm app`
# still drops you into the interactive CLI.
CMD ["python", "-m", "src.cli"]
