from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"


model = SentenceTransformer("BAAI/bge-m3")
query = "What is the legal basis for GDPR?"
query_vec = model.encode(query, normalize_embeddings=True)

client = QdrantClient(QDRANT_URL)

results = client.search(
    collection_name="eurlex",
    query_vector=query_vec,
    limit=5,
)

for r in results:
    print(r.payload["text"])
