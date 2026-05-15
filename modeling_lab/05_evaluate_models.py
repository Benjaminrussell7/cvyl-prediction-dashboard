from __future__ import annotations

import pandas as pd

from common import OUTPUTS, write_output


PREDICTIONS_INPUT = "baseline_predictions.csv"
TARGETS_INPUT = "targets.csv"
SUMMARY_OUTPUT = "evaluation_summary.csv"
CALIBRATION_OUTPUT = "calibration_summary.csv"

MODELS = {
    "elo": "elo_home_probability",
    "power_v3_recency": "power_v3_recency_home_probability",
    "power_v3_calibrated": "power_v3_calibrated_home_probability",
}


def evaluate_models(predictions: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = predictions.merge(
        targets[["game_id", "home_win"]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    summary_rows = []
    calibration_rows = []
    for model_name, probability_column in MODELS.items():
        probabilities = rows[probability_column].astype(float)
        actual = rows["home_win"].astype(float)
        predicted_home_win = probabilities >= 0.5
        actual_home_win = actual == 1.0
        summary_rows.append(
            {
                "model": model_name,
                "games": len(rows),
                "accuracy": float((predicted_home_win == actual_home_win).mean()),
                "average_confidence": float(probabilities.sub(0.5).abs().add(0.5).mean()),
                "brier_score": float(((probabilities - actual) ** 2).mean()),
            }
        )
        calibration_rows.extend(calibration_by_bucket(rows, model_name, probability_column))
    return pd.DataFrame(summary_rows), pd.DataFrame(calibration_rows)


def calibration_by_bucket(
    rows: pd.DataFrame,
    model_name: str,
    probability_column: str,
) -> list[dict[str, object]]:
    buckets = pd.IntervalIndex.from_tuples(
        [(0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.0)],
        closed="right",
    )
    favorite_probability = rows[probability_column].astype(float).map(lambda value: max(value, 1.0 - value))
    favorite_won = rows.apply(
        lambda row: row["home_win"] == 1.0
        if row[probability_column] >= 0.5
        else row["home_win"] == 0.0,
        axis=1,
    )
    bucketed = pd.DataFrame(
        {
            "favorite_probability": favorite_probability,
            "favorite_won": favorite_won.astype(float),
            "bucket": pd.cut(favorite_probability, buckets),
        }
    ).dropna(subset=["bucket"])
    output = []
    for bucket, group in bucketed.groupby("bucket", observed=True):
        output.append(
            {
                "model": model_name,
                "bucket": format_bucket(bucket),
                "games": len(group),
                "average_predicted_probability": float(group["favorite_probability"].mean()),
                "actual_win_rate": float(group["favorite_won"].mean()),
                "calibration_gap": float(group["favorite_won"].mean() - group["favorite_probability"].mean()),
            }
        )
    return output


def format_bucket(bucket: pd.Interval) -> str:
    left = int(round(bucket.left * 100))
    right = int(round(bucket.right * 100))
    if right == 100:
        return f"{left}%+"
    return f"{left}-{right}%"


def main() -> None:
    predictions = pd.read_csv(OUTPUTS / PREDICTIONS_INPUT)
    targets = pd.read_csv(OUTPUTS / TARGETS_INPUT)
    summary, calibration = evaluate_models(predictions, targets)
    summary_path = write_output(summary, SUMMARY_OUTPUT)
    calibration_path = write_output(calibration, CALIBRATION_OUTPUT)
    print(f"Wrote {summary_path}")
    print(f"Wrote {calibration_path}")


if __name__ == "__main__":
    main()
