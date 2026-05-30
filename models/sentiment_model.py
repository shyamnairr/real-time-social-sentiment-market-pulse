"""
Sentiment inference with graceful degradation.

Backends (auto-selected, best first):
  1. "finetuned"  -> your fine-tuned DistilBERT in config.model.finetuned_dir
                     (used automatically once `python run.py train` has run).
  2. "lexicon"    -> a fast finance-aware keyword scorer, NO torch required.
                     Keeps the whole demo runnable with zero ML dependencies.

Public API:
    scorer = get_scorer()
    df = scorer.score_texts(["...", "..."])   # -> columns: label, score, confidence
        label      : "negative" | "neutral" | "positive"
        score      : signed sentiment in [-1, +1]  (P(pos) - P(neg) for the model)
        confidence : 0..1
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils.config import CONFIG, get_path
from utils.logger import get_logger

log = get_logger("sentiment_model")

LABELS = CONFIG["model"]["labels"]   # ["negative","neutral","positive"]

# --- Lexicon backend ------------------------------------------------------
_POS_WORDS = {
    "crush", "soar", "soars", "beat", "beats", "bullish", "upgrade", "upgrading",
    "upside", "moon", "rocket", "squeeze", "undervalued", "strong", "incredible",
    "stellar", "buy", "rally", "breakout", "surge", "gains", "outperform", "record",
    "loading", "conviction", "printing", "green", "demand", "growth", "monster",
    "diamond", "rip", "pump", "accumulating", "win", "winner", "raised",
}
_NEG_WORDS = {
    "dump", "dumping", "crash", "bearish", "downgrade", "disaster", "miss", "missed",
    "lawsuit", "probe", "bleeding", "selloff", "capitulation", "overvalued", "garbage",
    "toast", "collapse", "collapsing", "cratering", "deteriorating", "puts", "short",
    "shorting", "knife", "stopped", "loss", "losses", "red", "drop", "plunge", "weak",
    "avoid", "cut", "brutal", "bagholder", "coping", "feasting", "destroyed",
}
_NEGATORS = {"not", "no", "never", "isn't", "aren't", "don't", "won't"}
_TOKEN_RE = re.compile(r"[a-z']+")


class LexiconScorer:
    backend = "lexicon"

    def score_texts(self, texts) -> pd.DataFrame:
        labels, scores, confs = [], [], []
        for t in texts:
            toks = _TOKEN_RE.findall(str(t).lower())
            pos = neg = 0
            for i, w in enumerate(toks):
                prev = toks[i - 1] if i > 0 else ""
                flip = prev in _NEGATORS
                if w in _POS_WORDS:
                    neg += 1 if flip else 0
                    pos += 0 if flip else 1
                elif w in _NEG_WORDS:
                    pos += 1 if flip else 0
                    neg += 0 if flip else 1
            total = pos + neg
            score = 0.0 if total == 0 else (pos - neg) / total
            if score > 0.2:
                label = "positive"
            elif score < -0.2:
                label = "negative"
            else:
                label = "neutral"
            conf = min(1.0, 0.5 + 0.1 * total) if total else 0.5
            labels.append(label)
            scores.append(float(score))
            confs.append(float(conf))
        return pd.DataFrame({"label": labels, "score": scores, "confidence": confs})


# --- Fine-tuned DistilBERT backend ----------------------------------------
class TransformerScorer:
    backend = "finetuned"

    def __init__(self, model_dir):
        import torch
        from transformers import (AutoTokenizer,
                                  DistilBertForSequenceClassification)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = DistilBertForSequenceClassification.from_pretrained(
            str(model_dir)).to(self.device).eval()
        self.max_len = CONFIG["model"]["max_length"]
        self.batch = CONFIG["model"]["batch_size_infer"]
        # map model's label order to neg/neu/pos indices
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        log.info("Loaded fine-tuned DistilBERT on %s", self.device)

    def score_texts(self, texts) -> pd.DataFrame:
        texts = [str(t) for t in texts]
        all_labels, all_scores, all_conf = [], [], []
        neg_i = [i for i, l in self.id2label.items() if l == "negative"][0]
        pos_i = [i for i, l in self.id2label.items() if l == "positive"][0]
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch):
                chunk = texts[start:start + self.batch]
                enc = self.tokenizer(chunk, truncation=True, padding=True,
                                     max_length=self.max_len, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
                idx = probs.argmax(axis=1)
                for j, p in zip(idx, probs):
                    all_labels.append(self.id2label[int(j)])
                    all_scores.append(float(p[pos_i] - p[neg_i]))
                    all_conf.append(float(p.max()))
        return pd.DataFrame({"label": all_labels, "score": all_scores,
                             "confidence": all_conf})


_SCORER = None


def get_scorer(force_lexicon: bool = False):
    """Return a cached scorer, preferring the fine-tuned model if present."""
    global _SCORER
    if _SCORER is not None:
        return _SCORER

    model_dir = get_path("models_dir") / "distilbert_finetuned"
    has_model = (model_dir / "config.json").exists()

    if has_model and not force_lexicon:
        try:
            _SCORER = TransformerScorer(model_dir)
            return _SCORER
        except Exception as e:
            log.warning("Could not load fine-tuned model (%s) -> lexicon fallback", e)

    log.info("Using lexicon sentiment backend (no fine-tuned model found)")
    _SCORER = LexiconScorer()
    return _SCORER
