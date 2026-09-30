# Interactive CLI: prompt for PDF/text file paths, then loop asking questions.
import sys

from . import ask, ingest


def _print_indexed() -> None:
    docs = ingest.list_documents()
    if docs:
        print("Already indexed:")
        for name in docs:
            print(f"  {name}")
        print()


def main() -> int:
    print("Local RAG (LangChain + Chroma + FastEmbed + Ollama). Free & local.\n")
    _print_indexed()

    while True:
        try:
            path = input("Path to a PDF/TXT file to ingest (Enter to continue): ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not path:
            break
        ingest.ingest_file(path)

    if not ingest.list_documents():
        print("No documents indexed. Exiting.")
        return 1

    print("\nReady. Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q.lower() in ("quit", "q", "exit"):
            return 0
        try:
            ask.answer(q)
        except KeyboardInterrupt:
            print("\n(interrupted)\n")
        except Exception as e:  # noqa: BLE001 — keep the loop alive on any error
            print(f"\n[error] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    sys.exit(main())
