# src/train_classifier.py

import json
import random
import numpy as np
import torch
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from torch.nn import BCEWithLogitsLoss


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ---------------------------------------------------------
# Load label mappings
# ---------------------------------------------------------
def load_label_maps():
    with open(DATA_DIR / "label2id.json", encoding="utf-8") as f:
        raw = json.load(f)
        label2id = {int(k): v for k, v in raw.items()}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------
# Create model with LoRA
# ---------------------------------------------------------
def create_model(num_labels: int, cfg):
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["query", "key", "value"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="SEQ_CLS",
    )

    return get_peft_model(base, lora_cfg)


# ---------------------------------------------------------
# Custom Trainer to force float labels
# ---------------------------------------------------------
class MultiLabelTrainer(Trainer):
    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        labels = inputs.pop("labels")
        labels = labels.float()  # ensure BCEWithLogits receives float targets

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int()
    labels = torch.tensor(labels).int()

    tp = (preds & labels).sum().item()
    fp = (preds & (1 - labels)).sum().item()
    fn = ((1 - preds) & labels).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"micro_f1": f1, "precision": precision, "recall": recall}


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def main(lang="en"):
    print(f"Training classifier for language: {lang}")

    # Load config
    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Load labels
    label2id, _ = load_label_maps()
    num_labels = len(label2id)

    # Load datasets
    train_ds = load_from_disk(str(DATA_DIR / f"train_{lang}"))
    val_ds = load_from_disk(str(DATA_DIR / f"val_{lang}"))

    # Create model
    model = create_model(num_labels, cfg)

    # fp16 only on CUDA
    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(MODELS_DIR / f"xlmr_lora_{lang}"),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        report_to="none",
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(MODELS_DIR / f"xlmr_lora_{lang}"))

    with open(
        MODELS_DIR / f"xlmr_lora_{lang}/label2id.json", "w", encoding="utf-8"
    ) as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main("en")
