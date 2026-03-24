import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


MAX_QUERY_WORDS = 64
MAX_PASSAGE_WORDS = 256
MAX_SEQ_LEN = 512


def truncate_text(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


class RerankerDataset(Dataset):
    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "query": item["query"],
            "passage": item["passage"],
            "label": item["label"],
        }

    def __init__(self, path: Path, max_samples: Optional[int] = None):
        self.examples = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                query = truncate_text(obj["query"], MAX_QUERY_WORDS)
                passage = truncate_text(obj["passage"], MAX_PASSAGE_WORDS)
                label = float(obj["label"])
                self.examples.append(
                    {
                        "query": query,
                        "passage": passage,
                        "label": label,
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RerankerCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        # batch is ALWAYS a list of dicts now
        queries = [ex["query"] for ex in batch]
        passages = [ex["passage"] for ex in batch]
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.float32)

        enc = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        enc["labels"] = labels
        return enc


class BCETrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        labels = labels.view(-1)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("./data/processed/eurlex_reranker_train.jsonl"),
    )
    parser.add_argument(
        "--model-name", type=str, default="cross-encoder/ms-marco-electra-base"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./data/models/eurlex-reranker-hf")
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--save-total-limit", type=int, default=3)

    args = parser.parse_args()

    print(f"[{datetime.now()}] Loading dataset...")
    train_dataset = RerankerDataset(args.train_file, args.max_samples)
    train_dataset = list(train_dataset)
    print(f"[{datetime.now()}] Loaded {len(train_dataset)} training pairs")

    print(f"[{datetime.now()}] Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="single_label_classification",
    )

    tokenizer.model_max_length = MAX_SEQ_LEN
    data_collator = RerankerCollator(tokenizer, max_length=MAX_SEQ_LEN)

    output_dir = str(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=int(0.1 * (len(train_dataset) // args.batch_size)),
        logging_steps=100,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=True,
        report_to=["none"],
        remove_unused_columns=False,  # 👈 critical line
    )

    # Auto-resume if a checkpoint exists
    resume_from_checkpoint = False
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"))
    if checkpoints:
        resume_from_checkpoint = True
        print(f"[{datetime.now()}] Resuming from checkpoint")

    trainer = BCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(f"[{datetime.now()}] Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"[{datetime.now()}] Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{datetime.now()}] Training complete.")


if __name__ == "__main__":
    main()
