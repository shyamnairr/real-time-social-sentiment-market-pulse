"""
Run the full analysis layer in one shot and persist all artifacts the dashboard reads:
  correlation -> anomaly -> backtest -> xgboost.

Loads the merged sentiment+price frame once and shares it across modules.
"""
from __future__ import annotations

from analysis.anomaly import detect_anomalies
from analysis.backtest import run_backtest
from analysis.correlation import compute_correlations
from analysis.features import load_merged
from models.xgboost_model import train_xgboost
from utils.logger import get_logger

log = get_logger("run_analysis")


def run_all_analysis():
    merged = load_merged()
    if merged.empty:
        log.error("No merged data. Run prices -> generate -> pipeline first.")
        return

    log.info("[1/4] correlation + lag analysis")
    compute_correlations(merged)

    log.info("[2/4] anomaly detection")
    detect_anomalies()

    log.info("[3/4] backtest signal accuracy")
    run_backtest(merged)

    log.info("[4/4] XGBoost secondary model")
    train_xgboost(merged)

    log.info("Analysis complete. Artifacts written to data/processed/")


if __name__ == "__main__":
    run_all_analysis()
