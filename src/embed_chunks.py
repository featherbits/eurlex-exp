import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import math
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

CHUNKS_FILE = "./data/processed/chunks.jsonl"
OUTPUT_FILE = "./data/processed/embeddings.jsonl"
CHECKPOINT_FILE = "./data/processed/embeddings_checkpoint.txt"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 256
device = "cpu"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint() -> int:
    if not Path(CHECKPOINT_FILE).exists():
        return 0
    with open(CHECKPOINT_FILE, "r") as f:
        return int(f.read().strip() or 0)


def save_checkpoint(next_index: int):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(next_index))


# ------------------------------------------------------------
# Embedding logic
# ------------------------------------------------------------


def process_embeddings():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    chunks = read_jsonl(CHUNKS_FILE)
    total_chunks = len(chunks)
    print(f"Loaded {total_chunks} chunks.")
    print(f"Batch size: {BATCH_SIZE}")

    start_idx = load_checkpoint()
    print(f"Resuming from chunk index: {start_idx}")

    # If starting from scratch, truncate output file if it exists
    if start_idx == 0 and Path(OUTPUT_FILE).exists():
        print(f"Starting fresh, truncating {OUTPUT_FILE}")
        Path(OUTPUT_FILE).unlink()

    # Main loop: drive batches by absolute indices, not batch number
    for batch_start in range(start_idx, total_chunks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        chunk_batch = chunks[batch_start:batch_end]
        texts = [c["text"] for c in chunk_batch]

        print(f"Embedding chunks {batch_start}-{batch_end-1} " f"({len(texts)} chunks)")

        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )

        output_rows = []
        for chunk, vector in zip(chunk_batch, vectors):
            output_rows.append(
                {
                    "id": chunk["id"],
                    "celex": chunk["celex"],
                    "chunk_type": chunk["chunk_type"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": vector.tolist(),
                }
            )

        append_jsonl(OUTPUT_FILE, output_rows)

        # checkpoint = index of next chunk to process
        save_checkpoint(batch_end)

    print("All chunks processed.")
    print(f"Embeddings written to {OUTPUT_FILE}")
    save_checkpoint(total_chunks)


if __name__ == "__main__":
    process_embeddings()
