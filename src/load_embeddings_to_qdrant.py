import json
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

EMBEDDINGS_FILE = "./data/processed/embeddings.jsonl"
COLLECTION_NAME = "eurlex"
VECTOR_SIZE = 1024
BATCH_SIZE = 500
QDRANT_URL = "http://localhost:6333"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def make_uuid_from_string(s: str) -> str:
    """Deterministically convert any string into a UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------


def stream_embeddings_to_qdrant():
    client = QdrantClient(QDRANT_URL)

    # Delete collection if it exists
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' exists. Deleting...")
        client.delete_collection(COLLECTION_NAME)

    # Create new collection
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    print("Collection created. Starting upload...")

    points = []
    total_uploaded = 0

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            row = json.loads(line)

            # Convert your string ID → UUID
            point_id = make_uuid_from_string(row["id"])

            points.append(
                PointStruct(
                    id=point_id,
                    vector=row["embedding"],
                    payload={
                        "id": row["id"],  # keep original ID in payload
                        "celex": row["celex"],
                        "chunk_type": row["chunk_type"],
                        "text": row["text"],
                        "metadata": row["metadata"],
                    },
                )
            )

            # Upload in batches
            if len(points) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                total_uploaded += len(points)
                print(f"Uploaded {total_uploaded} vectors...")
                points = []

    # Final flush
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_uploaded += len(points)

    print(f"Done. Total vectors uploaded: {total_uploaded}")


if __name__ == "__main__":
    stream_embeddings_to_qdrant()
