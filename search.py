import sys
from vectorize import (
    load_embedding_model,
    load_qdrant_client,
    search_transcript,
    COLLECTION_NAME,
)
from transcribe import format_timestamp


def run_search():
    # loading model and connecting to existing local qdrant db
    embedding_model = load_embedding_model()
    client = load_qdrant_client()

    print()
    print("=" * 50)
    print("  LectureIQ - Semantic Search")
    print("=" * 50)
    print("  type your query to search the lecture transcript")
    print("  type 'exit' to quit")
    print("=" * 50)

    while True:
        print()
        query = input("search > ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("bye")
            break

        results = search_transcript(query, embedding_model, client, top_k=3)

        if not results:
            print("  no results found")
            continue

        print(f"\n  top {len(results)} matches for: '{query}'\n")
        for i, r in enumerate(results, 1):
            start_ts = format_timestamp(r.payload["start"])
            end_ts = format_timestamp(r.payload["end"])
            text = r.payload["text"]
            score = round(r.score, 3)
            print(f"  {i}. score: {score}  [{start_ts} -> {end_ts}]")
            print(f"     {text}")


if __name__ == "__main__":
    run_search()