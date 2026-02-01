# src/infer_classifier.py

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


class EurovocClassifier:
    def __init__(self, lang="en"):
        self.lang = lang

        # Load config
        with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        # Model directory
        self.model_dir = MODELS_DIR / f"xlmr_lora_{lang}"

        # Load label2id (EUROVOC_code -> index)
        with open(self.model_dir / "label2id.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.label2id = {str(k): int(v) for k, v in raw.items()}

        # Reverse mapping: index -> EUROVOC_code
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Load EUROVOC descriptions
        with open(DATA_DIR / "eurovoc_descriptions.json", "r", encoding="utf-8") as f:
            self.eurovoc_desc = json.load(f)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])

        # Load base model with correct classifier head
        base = AutoModelForSequenceClassification.from_pretrained(
            self.cfg["model_name"],
            num_labels=len(self.label2id),
            problem_type="multi_label_classification",
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base, str(self.model_dir))
        self.model.eval()

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def predict(self, text, threshold=0.5):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg["max_length"],
            padding="max_length",
            return_tensors="pt",
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Select labels above threshold
        pred_ids = [i for i, p in enumerate(probs) if p >= threshold]
        pred_codes = [self.id2label[i] for i in pred_ids]

        # Human-readable descriptions
        pred_pairs = [
            (code, self.eurovoc_desc.get(code, "Unknown label")) for code in pred_codes
        ]

        return pred_pairs, probs


if __name__ == "__main__":
    clf = EurovocClassifier("en")

    sample_text = """
    The regulation establishes rules for data protection, privacy, and the processing of personal information within the European Union.
    """

    labels, probs = clf.predict(sample_text)

    print("Predicted EUROVOC labels:")
    for code, desc in labels:
        print(f"- {code} → {desc}")
