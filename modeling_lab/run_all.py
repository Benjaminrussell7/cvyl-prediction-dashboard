from __future__ import annotations

import subprocess
import sys
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent


SCRIPTS = [
    "01_build_historical_matchups.py",
    "02_generate_features.py",
    "03_create_targets.py",
    "04_benchmark_baselines.py",
    "05_evaluate_models.py",
    "06_train_interpretable_ml_baselines.py",
    "07_confidence_engine.py",
    "08_poisson_score_model.py",
    "09_hybrid_prediction_layer.py",
    "10_disagreement_analysis.py",
    "11_game_archetypes.py",
    "12_modeling_lab_summary.py",
    "13_rolling_backtest.py",
]


def main() -> None:
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(LAB_DIR / script)], check=True)


if __name__ == "__main__":
    main()
