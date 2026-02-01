"""CLI entry point for RAG PDF Q&A (direct, no API server)."""

import logging
from textbookapi.engine import RAGEngine, RAGEngineError
from textbookapi.config import RAGConfig, BOOKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    print("=" * 60)
    print("  RAG PDF Q&A System (Direct Mode)")
    print("=" * 60)
    print()
    print("Choose a book:")
    for i, (book_id, info) in enumerate(BOOKS.items(), 1):
        print(f"  {i}. {info['title']}")
    print()

    choice = input("Enter number: ").strip()
    book_ids = list(BOOKS.keys())
    idx = int(choice) - 1 if choice.isdigit() else 0
    idx = max(0, min(idx, len(book_ids) - 1))
    book_id = book_ids[idx]

    config = RAGConfig.for_book(book_id)
    print(f"\n  Model: Ollama + {config.model_name}")
    print(f"  Book: {config.book_title}")
    print("=" * 60)

    try:
        engine = RAGEngine(config)
        engine.initialize()
    except (RAGEngineError, FileNotFoundError) as e:
        print(f"\nError: {e}")
        return

    print("\n" + "=" * 60)
    print("  Ready! Ask questions about the book.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        print()
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            token_gen, results = engine.query(question, stream=True)
            print(f"  [Retrieved {len(results)} passages, "
                  f"score: {results[0][1]:.3f}]")
            print()
            print("Assistant: ", end="")
            for token in token_gen:
                print(token, end="", flush=True)
            print()
        except RAGEngineError as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
