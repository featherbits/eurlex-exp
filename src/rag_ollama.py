from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
from sentence_transformers import CrossEncoder

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "eurlex"
EMBED_MODEL = "BAAI/bge-m3"  # or jinaai/jina-embeddings-v2-base-multilingual
OLLAMA_MODEL = "llama3"  # or mistral, qwen2.5, etc.

# ------------------------------------------------------------
# Init
# ------------------------------------------------------------

embedder = SentenceTransformer(EMBED_MODEL, cache_folder="./.hf_cache")
reranker = CrossEncoder("BAAI/bge-reranker-large", cache_folder="./.hf_cache")
client = QdrantClient(QDRANT_URL)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def rerank(query, results, top_n=5):
    """Rerank Qdrant results using a cross-encoder."""
    pairs = []
    payloads = []

    for r in results.points:
        text = r.payload["text"]
        pairs.append((query, text))
        payloads.append(r)

    # Cross-encoder scores
    scores = reranker.predict(pairs)

    # Sort by score descending
    ranked = sorted(zip(scores, payloads), key=lambda x: x[0], reverse=True)

    # Return top_n payloads
    return [p for _, p in ranked[:top_n]]


def build_context(results, max_chars=8000):
    """Merge retrieved chunks into a single context block."""
    context = ""
    for r in results.points:
        chunk = r.payload["text"]
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n\n"
    return context.strip()


def build_prompt(query, context):
    """RAG prompt template."""
    return f"""
You are an expert assistant for EU law.

Use ONLY the context below to answer the question.
If the answer is not in the context, say that you cannot find it.

Context:
{context}

Question:
{query}

Answer:
"""


def generate_with_ollama(prompt: str) -> str:
    """Call Ollama locally."""
    response = ollama.chat(
        model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ------------------------------------------------------------
# Main RAG function
# ------------------------------------------------------------


def rag(query: str, k: int = 20, final_k: int = 5):
    # 1. Embed query
    qvec = embedder.encode(query, normalize_embeddings=True)

    # 2. Retrieve from Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=k,
    )

    # 3. Rerank
    reranked = rerank(query, results, top_n=final_k)

    # 4. Build context
    context = "\n\n".join([p.payload["text"] for p in reranked])

    # 5. Build prompt
    prompt = build_prompt(query, context)

    # 6. Generate answer with Ollama
    answer = generate_with_ollama(prompt)

    return answer


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    question = "What is the legal basis for GDPR?"
    answer = rag(question)
    print(f"\n\nQuestion:")
    print(question)
    print(f"Answer:")
    print(answer)
