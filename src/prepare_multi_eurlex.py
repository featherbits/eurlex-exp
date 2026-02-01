# src/prepare_multi_eurlex.py

from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CONFIG_PATH = BASE_DIR / "config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_space():
    """
    Build label2id.json using REAL EUROVOC codes from dataset metadata.
    """
    print("Loading MultiEURLEX metadata to build label space...")

    ds = load_dataset("coastalcph/multi_eurlex", "en")

    # Extract real EUROVOC codes from dataset metadata
    eurovoc_codes = ds["train"].features["labels"].feature.names

    # Build mapping: EUROVOC_code -> index
    label2id = {code: i for i, code in enumerate(eurovoc_codes)}

    with open(DATA_DIR / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print(f"Label space built. Total EUROVOC labels: {len(label2id)}")


def prepare_split(lang: str):
    """
    Tokenize dataset and convert labels to multi-hot vectors using real EUROVOC codes.
    """
    print(f"Preparing dataset for language: {lang}")

    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # Load dataset
    ds = load_dataset("coastalcph/multi_eurlex", lang)

    # Load label2id mapping
    with open(DATA_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)

    num_labels = len(label2id)

    def encode(ex):
        # Tokenize
        enc = tokenizer(
            ex["text"],
            truncation=True,
            max_length=cfg["max_length"],
            padding="max_length",
        )

        # Convert EUROVOC codes to multi-hot vector
        y = [0.0] * num_labels
        for lbl in ex["labels"]:
            eurovoc_code = ds["train"].features["labels"].feature.names[lbl]
            y[label2id[eurovoc_code]] = 1.0

        enc["labels"] = y
        return enc

    # Map with num_proc=1 for Windows compatibility
    train = ds["train"].map(encode, num_proc=1)
    val = ds["validation"].map(encode, num_proc=1)

    # Save to disk
    train.save_to_disk(str(DATA_DIR / f"train_{lang}"))
    val.save_to_disk(str(DATA_DIR / f"val_{lang}"))

    print(f"Saved tokenized dataset for {lang}")


if __name__ == "__main__":
    build_label_space()
    prepare_split("en")
    prepare_split("lv")
