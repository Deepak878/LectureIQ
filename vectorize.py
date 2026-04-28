import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Batch
from sentence_transformers import SentenceTransformer, CrossEncoder

# local qdrant folder
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "lecture_segments"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_embedding_model(model_name=EMBEDDING_MODEL):
    """loading sentence transformer model locally for generating embeddings"""
    return SentenceTransformer(model_name)

def load_reranker_model(model_name=RERANKER_MODEL):
    """loading cross-encoder model for re ranking candidates from qdrant"""
    return CrossEncoder(model_name)

def load_qdrant_client(db_path=QDRANT_PATH):
    """connecting to local qdrant using a context manager or explicit client"""
    return QdrantClient(path=db_path)

def setup_collection(client, collection_name=COLLECTION_NAME, vector_size=384):
    """creating qdrant collection if it doesnt exist"""
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f" created collection : {collection_name}")
    else:
        print(f" collection already exists: {collection_name}")

def embed_and_store(
    transcript_json_path, embedding_model, client, collection_name=COLLECTION_NAME
):
    """reads transcript json and embeds each segment, stores vectors in qdrant"""
    if not os.path.exists(transcript_json_path):
        print(f" transcript json not found: {transcript_json_path}")
        return False

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    segments = transcript_data.get("segments", [])
    if not segments:
        print(" no segments found in transcript json")
        return False

    print(f"embedding {len(segments)} transcript segments ...")

    texts = [seg["text"] for seg in segments]
    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Optimization: Use Batch upsert for faster Qdrant indexing
    client.upsert(
        collection_name=collection_name,
        points=Batch(
            ids=list(range(len(segments))),
            vectors=embeddings.tolist(),
            payloads=[
                {
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "source_file": os.path.basename(transcript_json_path),
                }
                for seg in segments
            ]
        )
    )
    print(f" stored {len(segments)} vectors in collection: '{collection_name}'")
    return True

def search_transcript(
    query,
    embedding_model,
    client,
    collection_name=COLLECTION_NAME,
    top_k=3,
    reranker=None,
):
    """semantic search over stored transcript vectors"""
    fetch_k = max(top_k * 4, 10) if reranker else top_k
    
    # Optimization: encode single query without unnecessary list wrapping if possible
    query_vector = embedding_model.encode(query).tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=fetch_k,
    )

    if not reranker or not results:
        return results[:top_k]

    pairs = [(query, r.payload["text"]) for r in results]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_k]]

def run_vectorize_pipeline(transcript_json_path):
    """main entry: load transcript json, embed segments and store in qdrant"""
    print("\n" + "="*50 + "\n  LectureIQ - Vectorize step\n" + "="*50)

    embedding_model = load_embedding_model()
    # Use 'with' to ensure the client connection closes properly if supported, 
    # or ensure it's handled in the main block.
    client = load_qdrant_client()
    setup_collection(client)

    success = embed_and_store(transcript_json_path, embedding_model, client)

    if success:
        print("\n" + "="*50 + "\n  vectorization complete")
        print(f"  source  : {transcript_json_path}\n  db path : {QDRANT_PATH}/\n  collection : {COLLECTION_NAME}")
        print("="*50)

    return embedding_model, client, success

def get_collection_stats(client, collection_name=COLLECTION_NAME):
    """
    Returns high-level statistics about the vector collection.
    Useful for verifying the health and size of the LectureIQ index.
    """
    try:
        info = client.get_collection(collection_name=collection_name)
        return {
            "status": info.status,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = (
            input("enter path to transcript json file: ").strip().strip('"').strip("'")
        )

    json_path = os.path.abspath(json_path)
    embedding_model, client, success = run_vectorize_pipeline(json_path)

    if success:
        # performing  quick search test after vectorizing
        print()
        print("quick search test")
        test_query = input(
            "enter a search query to test or, press enter to skip : "
        ).strip()

        if test_query:
            results = search_transcript(test_query, embedding_model, client)
            print(f"\ntop {len(results)} matches for: '{test_query}'\n")
            for r in results:
                start = r.payload["start"]
                end = r.payload["end"]
                text = r.payload["text"]
                score = round(r.score, 3)
                print(f"  score: {score}  [{start}s -> {end}s]  {text}")
