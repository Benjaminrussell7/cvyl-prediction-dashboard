from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.calibration import BUCKETS, POWER_RATING_CALIBRATION_COLUMNS
from cvyl_scraper.export import export_csv
from cvyl_scraper.hybrid import DEFAULT_POWER_V2_LOGISTIC_SCALE, power_v2_win_probability
from cvyl_scraper.model_comparison import (
    _actual_winner,
    _completed_games,
    _comparison_keys,
    _home_result,
)
from cvyl_scraper.model_comparison_v3 import _prior_power_v3_rating, _team_game_rows


DEFAULT_MODEL_COMPARISON_V4_CALIBRATED_CSV = (
    "data/processed/cvyl_model_comparison_v4_calibrated.csv"
)
DEFAULT_MODEL_COMPARISON_V4_CALIBRATED_SUMMARY_CSV = (
    "data/processed/cvyl_model_comparison_v4_calibrated_summary.csv"
)
DEFAULT_POWER_RATING_V4_CALIBRATION_CSV = "data/processed/cvyl_calibration_power_rating_v4.csv"
DEFAULT_CALIBRATED_LOGISTIC_SCALE = 3.0
DEFAULT_MIN_PROBABILITY = 0.15
DEFAULT_MAX_PROBABILITY = 0.85

MODEL_COMPARISON_V4_CALIBRATED_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "actual_winner",
    "power_v3_recency_predicted_winner",
    "power_v3_recency_win_probability",
    "power_v3_calibrated_predicted_winner",
    "power_v3_calibrated_win_probability",
    "power_v3_recency_correct",
    "power_v3_calibrated_correct",
]

MODEL_COMPARISON_V4_CALIBRATED_SUMMARY_COLUMNS = [
    "total_games",
    "power_v3_recency_accuracy",
    "power_v3_calibrated_accuracy",
    "power_v3_recency_brier_score",
    "power_v3_calibrated_brier_score",
]


def calibrated_power_v3_probability(
    rating_difference: float,
    *,
    scale: float = DEFAULT_CALIBRATED_LOGISTIC_SCALE,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    max_probability: float = DEFAULT_MAX_PROBABILITY,
) -> float:
    if scale <= 0:
        raise ValueError("scale must be greater than zero.")
    if not 0 < min_probability < max_probability < 1:
        raise ValueError("probability caps must satisfy 0 < min < max < 1.")

    raw_probability = power_v2_win_probability(rating_difference, scale=scale)
    return min(max(raw_probability, min_probability), max_probability)


def calibrated_power_v4_probability(
    rating_difference: float,
    *,
    scale: float = DEFAULT_CALIBRATED_LOGISTIC_SCALE,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    max_probability: float = DEFAULT_MAX_PROBABILITY,
) -> float:
    return calibrated_power_v3_probability(
        rating_difference,
        scale=scale,
        min_probability=min_probability,
        max_probability=max_probability,
    )


def build_model_comparison_v4_calibrated_outputs(
    games: pd.DataFrame,
    *,
    baseline_logistic_scale: float = DEFAULT_POWER_V2_LOGISTIC_SCALE,
    calibrated_logistic_scale: float = DEFAULT_CALIBRATED_LOGISTIC_SCALE,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    max_probability: float = DEFAULT_MAX_PROBABILITY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_team_game_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, float]] = []

    completed = _completed_games(games)
    for _, game in completed.iterrows():
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        actual_winner = _actual_winner(home_team, away_team, home_score, away_score)
        rating_difference = _prior_power_v3_rating(home_team, prior_team_game_rows) - _prior_power_v3_rating(
            away_team,
            prior_team_game_rows,
        )

        baseline_home_probability = power_v2_win_probability(
            rating_difference,
            scale=baseline_logistic_scale,
        )
        calibrated_home_probability = calibrated_power_v3_probability(
            rating_difference,
            scale=calibrated_logistic_scale,
            min_probability=min_probability,
            max_probability=max_probability,
        )
        baseline_winner = home_team if baseline_home_probability >= 0.5 else away_team
        calibrated_winner = home_team if calibrated_home_probability >= 0.5 else away_team

        comparison_rows.append(
            {
                "game_date": game["game_date"].date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "actual_winner": actual_winner,
                "power_v3_recency_predicted_winner": baseline_winner,
                "power_v3_recency_win_probability": max(
                    baseline_home_probability,
                    1.0 - baseline_home_probability,
                ),
                "power_v3_calibrated_predicted_winner": calibrated_winner,
                "power_v3_calibrated_win_probability": max(
                    calibrated_home_probability,
                    1.0 - calibrated_home_probability,
                ),
                "power_v3_recency_correct": baseline_winner == actual_winner,
                "power_v3_calibrated_correct": calibrated_winner == actual_winner,
            }
        )
        probability_rows.append(
            {
                "home_result": _home_result(home_score, away_score),
                "power_v3_recency_home_probability": baseline_home_probability,
                "power_v3_calibrated_home_probability": calibrated_home_probability,
            }
        )
        prior_team_game_rows.extend(_team_game_rows(game))

    comparison = pd.DataFrame(comparison_rows, columns=MODEL_COMPARISON_V4_CALIBRATED_COLUMNS)
    validate_model_comparison_v4_calibrated(comparison, completed)
    probabilities = pd.DataFrame(probability_rows)
    summary = build_model_comparison_v4_calibrated_summary(comparison, probabilities)
    calibration = build_calibration_table(
        comparison,
        probability_column="power_v3_calibrated_win_probability",
        correct_column="power_v3_calibrated_correct",
    )
    return comparison, summary, calibration


def validate_model_comparison_v4_calibrated(
    comparison: pd.DataFrame,
    completed_games: pd.DataFrame,
) -> None:
    if len(comparison) != len(completed_games):
        raise ValueError(
            "Model comparison v4 row count must equal completed games count: "
            f"{len(comparison)} != {len(completed_games)}."
        )

    required_columns = [
        "power_v3_recency_predicted_winner",
        "power_v3_recency_win_probability",
        "power_v3_calibrated_predicted_winner",
        "power_v3_calibrated_win_probability",
    ]
    if comparison[required_columns].isna().any(axis=1).any():
        raise ValueError("All v4 comparison rows must include baseline and calibrated predictions.")

    if _comparison_keys(comparison) != _comparison_keys(completed_games):
        raise ValueError("Model comparison v4 rows must cover the same games as completed games input.")


def build_model_comparison_v4_calibrated_summary(
    comparison: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(
            [
                {
                    "total_games": 0,
                    "power_v3_recency_accuracy": 0.0,
                    "power_v3_calibrated_accuracy": 0.0,
                    "power_v3_recency_brier_score": 0.0,
                    "power_v3_calibrated_brier_score": 0.0,
                }
            ],
            columns=MODEL_COMPARISON_V4_CALIBRATED_SUMMARY_COLUMNS,
        )

    return pd.DataFrame(
        [
            {
                "total_games": int(len(comparison)),
                "power_v3_recency_accuracy": float(comparison["power_v3_recency_correct"].mean()),
                "power_v3_calibrated_accuracy": float(
                    comparison["power_v3_calibrated_correct"].mean()
                ),
                "power_v3_recency_brier_score": _brier(
                    probabilities["power_v3_recency_home_probability"],
                    probabilities["home_result"],
                ),
                "power_v3_calibrated_brier_score": _brier(
                    probabilities["power_v3_calibrated_home_probability"],
                    probabilities["home_result"],
                ),
            }
        ],
        columns=MODEL_COMPARISON_V4_CALIBRATED_SUMMARY_COLUMNS,
    )


def build_calibration_table(
    comparison: pd.DataFrame,
    *,
    probability_column: str,
    correct_column: str,
) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(columns=POWER_RATING_CALIBRATION_COLUMNS)

    frame = comparison.copy()
    frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame[correct_column] = frame[correct_column].astype(bool)
    frame = frame.dropna(subset=[probability_column, correct_column])

    rows = []
    for bucket, lower, upper in BUCKETS:
        if bucket == "75%+":
            bucket_games = frame[
                (frame[probability_column] >= lower) & (frame[probability_column] <= 1.0)
            ]
        else:
            bucket_games = frame[
                (frame[probability_column] >= lower) & (frame[probability_column] < upper)
            ]

        games = int(len(bucket_games))
        if games == 0:
            average_probability = 0.0
            actual_win_rate = 0.0
        else:
            average_probability = float(bucket_games[probability_column].mean())
            actual_win_rate = float(bucket_games[correct_column].mean())
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


def export_model_comparison_v4_calibrated_outputs(
    games: pd.DataFrame,
    comparison_output_path: str | Path = DEFAULT_MODEL_COMPARISON_V4_CALIBRATED_CSV,
    summary_output_path: str | Path = DEFAULT_MODEL_COMPARISON_V4_CALIBRATED_SUMMARY_CSV,
    calibration_output_path: str | Path = DEFAULT_POWER_RATING_V4_CALIBRATION_CSV,
    **kwargs,
) -> tuple[Path, Path, Path]:
    comparison, summary, calibration = build_model_comparison_v4_calibrated_outputs(
        games,
        **kwargs,
    )
    return (
        export_csv(comparison, comparison_output_path),
        export_csv(summary, summary_output_path),
        export_csv(calibration, calibration_output_path),
    )


def _brier(probabilities: pd.Series, results: pd.Series) -> float:
    return float(((probabilities - results) ** 2).mean())
