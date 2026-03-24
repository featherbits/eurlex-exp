import json
import random
from pathlib import Path
from typing import List, Dict
import argparse


def load_chunks(path: Path) -> List[Dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def make_query(chunk: Dict) -> str:
    """Generate a synthetic query based on chunk type."""
    ctype = chunk.get("chunk_type")

    if ctype == "article":
        num = chunk.get("article_number")
        return f"Article {num}" if num else "Article"

    if ctype == "recital":
        num = chunk.get("recital_number")
        return f"Recital {num}" if num else "Recital"

    if ctype == "annex":
        idx = chunk.get("annex_index")
        return f"Annex {idx}" if idx is not None else "Annex"

    # fallback
    return chunk.get("chunk_type", "Section")


def build_pairs(
    chunks: List[Dict],
    negatives_per_positive: int = 4,
    seed: int = 42,
) -> List[Dict]:
    random.seed(seed)

    # Group by CELEX to avoid sampling negatives from same document
    by_celex = {}
    for ch in chunks:
        celex = ch.get("celex")
        if celex is None:
            raise ValueError("Chunk missing 'celex' field")
        by_celex.setdefault(celex, []).append(ch)

    all_chunks = chunks[:]  # flat list for negative sampling
    pairs = []

    for ch in chunks:
        celex = ch["celex"]
        text = ch.get("text", "").strip()
        if not text:
            continue

        query = make_query(ch)

        # Positive pair
        pairs.append(
            {
                "query": query,
                "passage": text,
                "label": 1,
                "celex": celex,
                "chunk_id": ch.get("id"),
            }
        )

        # Negative pairs
        negatives = []
        attempts = 0
        max_attempts = negatives_per_positive * 10

        while len(negatives) < negatives_per_positive and attempts < max_attempts:
            attempts += 1
            neg = random.choice(all_chunks)
            if neg["celex"] == celex:
                continue
            neg_text = neg.get("text", "").strip()
            if not neg_text:
                continue
            negatives.append(neg)

        for neg in negatives:
            pairs.append(
                {
                    "query": query,
                    "passage": neg.get("text", ""),
                    "label": 0,
                    "celex": celex,
                    "chunk_id": ch.get("id"),
                    "neg_celex": neg.get("celex"),
                    "neg_chunk_id": neg.get("id"),
                }
            )

    return pairs


def save_pairs(pairs: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default="./data/processed/chunks.jsonl")
    parser.add_argument(
        "--out", type=Path, default="./data/processed/eurlex_reranker_train.jsonl"
    )
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading chunks from {args.chunks} ...")
    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks")

    print("Building training pairs ...")
    pairs = build_pairs(
        chunks,
        negatives_per_positive=args.negatives_per_positive,
    )
    print(f"Built {len(pairs)} pairs")

    print(f"Saving to {args.out} ...")
    save_pairs(pairs, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
