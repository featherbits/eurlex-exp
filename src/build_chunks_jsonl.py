import os
import json
from pathlib import Path
from typing import Dict, Any, List

from ingestion.chunker import chunk_document


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_directory(input_dir: str, output_file: str):
    input_path = Path(input_dir)
    output_path = Path(output_file)

    all_chunks: List[Dict[str, Any]] = []

    for file in sorted(input_path.glob("*.json")):
        print(f"Processing {file.name}...")

        try:
            doc = load_json(file)
        except Exception as e:
            print(f"  ERROR reading {file}: {e}")
            continue

        try:
            chunks = chunk_document(doc)
        except Exception as e:
            print(f"  ERROR chunking {file}: {e}")
            continue

        all_chunks.extend(chunks)

    print(f"Writing {len(all_chunks)} chunks to {output_path}...")
    write_jsonl(output_path, all_chunks)
    print("Done.")


if __name__ == "__main__":
    # Adjust these paths as needed
    INPUT_DIR = "./data/cellar_json/en"
    OUTPUT_FILE = "./data/processed/chunks.jsonl"

    process_directory(INPUT_DIR, OUTPUT_FILE)
