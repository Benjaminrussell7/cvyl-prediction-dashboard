from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.export import export_csv


DEFAULT_POWER_RATING_CALIBRATION_CSV = "data/processed/cvyl_calibration_power_rating.csv"

POWER_RATING_CALIBRATION_COLUMNS = [
    "bucket",
    "games",
    "average_predicted_probability",
    "actual_win_rate",
    "calibration_gap",
]

BUCKETS = [
    ("50-55%", 0.50, 0.55),
    ("55-60%", 0.55, 0.60),
    ("60-65%", 0.60, 0.65),
    ("65-70%", 0.65, 0.70),
    ("70-75%", 0.70, 0.75),
    ("75%+", 0.75, 1.01),
]


def build_power_rating_calibration(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(columns=POWER_RATING_CALIBRATION_COLUMNS)

    frame = comparison.copy()
    frame["power_v3_recency_win_probability"] = pd.to_numeric(
        frame["power_v3_recency_win_probability"],
        errors="coerce",
    )
    frame["power_v3_recency_correct"] = frame["power_v3_recency_correct"].astype(bool)
    frame = frame.dropna(subset=["power_v3_recency_win_probability", "power_v3_recency_correct"])

    rows = []
    for bucket, lower, upper in BUCKETS:
        if bucket == "75%+":
            bucket_games = frame[
                (frame["power_v3_recency_win_probability"] >= lower)
                & (frame["power_v3_recency_win_probability"] <= 1.0)
            ]
        else:
            bucket_games = frame[
                (frame["power_v3_recency_win_probability"] >= lower)
                & (frame["power_v3_recency_win_probability"] < upper)
            ]

        games = int(len(bucket_games))
        if games == 0:
            average_probability = 0.0
            actual_win_rate = 0.0
        else:
            average_probability = float(bucket_games["power_v3_recency_win_probability"].mean())
            actual_win_rate = float(bucket_games["power_v3_recency_correct"].mean())
        rows.append(
            {
                "bucket": bucket,
                "games": games,
                "average_predicted_probability": average_probability,
                "actual_win_rate": actual_win_rate,
                "calibration_gap": actual_win_rate - average_probability,
            }
        )

    return pd.DataFrame(rows, columns=POWER_RATING_CALIBRATION_COLUMNS)


def export_power_rating_calibration(
    comparison: pd.DataFrame,
    output_path: str | Path = DEFAULT_POWER_RATING_CALIBRATION_CSV,
) -> Path:
    calibration = build_power_rating_calibration(comparison)
    return export_csv(calibration, output_path)
