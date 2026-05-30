"""
Secondary prediction model: XGBoost (the "compare ML approaches" piece).

Predicts next-day direction (up/down) from engineered daily features and is
benchmarked against the simple DistilBERT-sentiment signal on the SAME test set,
so the dashboard can show: "DistilBERT-signal X% vs XGBoost Y%".

Time-based train/test split (earlier dates train, later dates test) to avoid
look-ahead leakage.

Artifacts (data/processed/):
  - xgboost_metrics.json    (accuracy, roc_auc, baseline comparison)
  - feature_importance.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.features import load_merged
from utils.config import CONFIG, get_path
from utils.logger import get_logger

log = get_logger("xgboost_model")

FEATURES = [
    "mean_sentiment", "sentiment_momentum", "post_volume", "volume_change",
    "pct_positive", "pct_negative", "daily_return", "day_of_week",
]


def train_xgboost(merged: pd.DataFrame | None = None):
    from sklearn.metrics import accuracy_score, roc_auc_score
    import xgboost as xgb

    merged = load_merged() if merged is None else merged
    if merged.empty or len(merged) < 200:
        log.warning("Not enough data for XGBoost")
        return {}

    df = merged.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df = df.dropna(subset=FEATURES + ["next_direction"])

    # time-based split: last 25% of dates = test
    cutoff = df["date"].quantile(0.75)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    if test.empty or train.empty:
        log.warning("Bad split for XGBoost")
        return {}

    Xtr, ytr = train[FEATURES], train["next_direction"]
    Xte, yte = test[FEATURES], test["next_direction"]

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42,
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(yte, preds))
    try:
        auc = float(roc_auc_score(yte, proba))
    except ValueError:
        auc = float("nan")

    # baseline: DistilBERT-sentiment signal on the SAME test rows
    acfg = CONFIG["analysis"]
    bull, bear = acfg["signal_bullish_threshold"], acfg["signal_bearish_threshold"]
    test_sig = test[(test["mean_sentiment"] > bull) | (test["mean_sentiment"] < bear)]
    base_acc = float((np.sign(test_sig["mean_sentiment"])
                      == np.sign(test_sig["next_return"])).mean()) if len(test_sig) else 0.0

    importance = (pd.DataFrame({"feature": FEATURES,
                                "importance": model.feature_importances_})
                  .sort_values("importance", ascending=False))

    metrics = {
        "xgb_accuracy": acc,
        "xgb_roc_auc": auc,
        "distilbert_signal_accuracy": base_acc,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_test_signals": int(len(test_sig)),
        "features": FEATURES,
    }

    out = get_path("processed_dir")
    with open(out / "xgboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    importance.to_csv(out / "feature_importance.csv", index=False)

    log.info("XGBoost acc=%.3f auc=%.3f | DistilBERT-signal acc=%.3f",
             acc, auc, base_acc)
    return metrics


if __name__ == "__main__":
    train_xgboost()
