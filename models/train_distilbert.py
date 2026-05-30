"""
Fine-tune DistilBERT for 3-class financial sentiment (negative/neutral/positive).

RUN THIS ON YOUR GPU:
    python run.py train

Datasets (public, via HuggingFace `datasets`), combined to exceed 10K samples:
  - financial_phrasebank   (~4.8K expert-labeled financial sentences)
  - zeroshot/twitter-financial-news-sentiment  (~11.9K finance tweets)
Both are remapped to our label order: 0=negative, 1=neutral, 2=positive.

Outputs (to config model.finetuned_dir):
  - model weights + tokenizer
  - metrics.json : f1_weighted, accuracy, per-class classification report,
                   confusion matrix, training history  (read by the dashboard)

Target: ~0.87 weighted F1.
"""
from __future__ import annotations

import json
from pathlib import Path

# IMPORTANT (Windows): `datasets`/pyarrow MUST be imported before torch/transformers.
# Otherwise a native DLL load-order conflict between pyarrow and torch crashes the
# interpreter with an access violation (exit code 0xC0000005). Keep this import first.
import datasets  # noqa: F401  (side-effect import for load order)

import numpy as np

from utils.config import CONFIG, get_path
from utils.logger import get_logger

log = get_logger("train_distilbert")

LABELS = CONFIG["model"]["labels"]            # ["negative","neutral","positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def _load_combined_dataset():
    """Load + remap + concatenate the public financial sentiment datasets."""
    from datasets import Value, concatenate_datasets, load_dataset

    # force a common schema so the two sources can be concatenated
    int_label = Value("int64")
    parts = []

    # --- Financial PhraseBank (labels already 0=neg,1=neu,2=pos) ----------
    try:
        fpb = load_dataset("financial_phrasebank", "sentences_50agree",
                           split="train", trust_remote_code=True)
        fpb = fpb.rename_column("sentence", "text")
        fpb = fpb.map(lambda x: {"label": int(x["label"])})
        fpb = fpb.select_columns(["text", "label"]).cast_column("label", int_label)
        parts.append(fpb)
        log.info("Financial PhraseBank: %d rows", len(fpb))
    except Exception as e:
        log.warning("Could not load financial_phrasebank (%s)", e)

    # --- Twitter financial news sentiment (0=Bearish,1=Bullish,2=Neutral) -
    try:
        tw = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        remap = {0: 0, 1: 2, 2: 1}   # Bearish->neg, Bullish->pos, Neutral->neu
        tw = tw.map(lambda x: {"label": remap[int(x["label"])]})
        tw = tw.select_columns(["text", "label"]).cast_column("label", int_label)
        parts.append(tw)
        log.info("Twitter financial news: %d rows", len(tw))
    except Exception as e:
        log.warning("Could not load twitter-financial-news-sentiment (%s)", e)

    if not parts:
        raise RuntimeError(
            "No training datasets could be loaded. Check your internet connection "
            "and `pip install datasets`."
        )

    combined = concatenate_datasets(parts).shuffle(seed=42)
    log.info("Combined dataset: %d rows total", len(combined))
    if len(combined) < 10000:
        log.warning("Combined dataset < 10K (%d). Resume claims 10K+.", len(combined))
    return combined


def _compute_metrics_fn():
    from sklearn.metrics import accuracy_score, f1_score

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }
    return compute


def train():
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from transformers import (
        AutoTokenizer, DataCollatorWithPadding,
        DistilBertForSequenceClassification, Trainer, TrainingArguments,
    )

    mcfg = CONFIG["model"]
    out_dir = get_path("models_dir") / "distilbert_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    log.info("CUDA available: %s (%s)", use_cuda,
             torch.cuda.get_device_name(0) if use_cuda else "CPU")

    # --- data ------------------------------------------------------------
    ds = _load_combined_dataset()
    split = ds.train_test_split(test_size=0.15, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base_model"])

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=mcfg["max_length"])

    split = split.map(tok, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        mcfg["base_model"], num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=mcfg["epochs"],
        per_device_train_batch_size=mcfg["batch_size_train"],
        per_device_eval_batch_size=mcfg["batch_size_train"] * 2,
        learning_rate=float(mcfg["learning_rate"]),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        fp16=use_cuda,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
        compute_metrics=_compute_metrics_fn(),
    )

    log.info("Starting fine-tuning...")
    trainer.train()

    # --- final evaluation + metrics artifacts ----------------------------
    log.info("Evaluating best model...")
    preds_out = trainer.predict(split["test"])
    y_true = preds_out.label_ids
    y_pred = np.argmax(preds_out.predictions, axis=-1)

    report = classification_report(
        y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "labels": LABELS,
        "classification_report": report,
        "confusion_matrix": cm,
        "n_train": len(split["train"]),
        "n_eval": len(split["test"]),
        "training_history": trainer.state.log_history,
        "source": "trained",
    }

    # save model + tokenizer + metrics
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info("DONE. Weighted F1 = %.4f | Accuracy = %.4f -> %s",
             metrics["f1_weighted"], metrics["accuracy"], out_dir)
    return metrics


if __name__ == "__main__":
    train()
