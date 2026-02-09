from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "eurlex"
MODEL_NAME = "BAAI/bge-m3"  # or jinaai/jina-embeddings-v2-base-multilingual

# ------------------------------------------------------------
# Search logic
# ------------------------------------------------------------


def search(query: str, limit: int = 5):
    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Embed the query
    query_vector = model.encode(query, normalize_embeddings=True)

    # Connect to Qdrant
    client = QdrantClient(QDRANT_URL)

    # Perform vector search
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )

    # Print results
    for idx, point in enumerate(results.points, start=1):
        print(f"\nResult {idx}:")
        print(f"Score: {point.score}")
        print(f"CELEX: {point.payload.get('celex')}")
        print(f"Chunk type: {point.payload.get('chunk_type')}")
        print(f"Text:\n{point.payload.get('text')[:500]}...")
        print("-" * 80)


if __name__ == "__main__":
    search("What is the legal basis for GDPR?")
