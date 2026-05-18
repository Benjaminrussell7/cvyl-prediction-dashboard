from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from cvyl_scraper.export import export_csv
from cvyl_scraper.hybrid import DEFAULT_POWER_V2_LOGISTIC_SCALE, power_v2_win_probability
from cvyl_scraper.model_comparison import (
    _actual_winner,
    _completed_games,
    _comparison_keys,
    _home_result,
)
from cvyl_scraper.model_comparison_v3 import _team_game_rows
from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency
from cvyl_scraper.power_v4_opponent_adjusted import build_power_ratings_v4_opponent_adjusted
from cvyl_scraper.probability_calibration import (
    build_calibration_table,
    calibrated_power_v4_probability,
)


DEFAULT_MODEL_COMPARISON_POWER_V4_CSV = "data/processed/cvyl_model_comparison_power_v4.csv"
DEFAULT_MODEL_COMPARISON_POWER_V4_SUMMARY_CSV = (
    "data/processed/cvyl_model_comparison_power_v4_summary.csv"
)
DEFAULT_POWER_V4_CALIBRATION_CSV = "data/processed/cvyl_calibration_power_v4_opponent_adjusted.csv"
DEFAULT_TOTAL_GOALS = 12.0

MODEL_COMPARISON_POWER_V4_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "actual_winner",
    "power_v3_recency_predicted_winner",
    "power_v3_recency_win_probability",
    "power_v4_predicted_winner",
    "power_v4_win_probability",
    "power_v3_recency_correct",
    "power_v4_correct",
    "power_v3_recency_predicted_margin",
    "power_v4_predicted_margin",
    "actual_margin",
    "predicted_total_goals",
    "actual_total_goals",
]

MODEL_COMPARISON_POWER_V4_SUMMARY_COLUMNS = [
    "total_games",
    "power_v3_recency_accuracy",
    "power_v4_accuracy",
    "power_v3_recency_brier_score",
    "power_v4_brier_score",
    "power_v3_recency_log_loss",
    "power_v4_log_loss",
    "power_v3_recency_margin_mae",
    "power_v4_margin_mae",
    "power_v3_recency_total_goals_mae",
    "power_v4_total_goals_mae",
]


def build_model_comparison_power_v4_outputs(
    games: pd.DataFrame,
    *,
    baseline_logistic_scale: float = DEFAULT_POWER_V2_LOGISTIC_SCALE,
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
        actual_margin = home_score - away_score
        actual_total_goals = home_score + away_score
        predicted_total_goals = _prior_total_goals(prior_team_game_rows)

        power_v3_home_rating = _prior_power_rating(
            home_team,
            prior_team_game_rows,
            rating_column="power_rating_v3_recency",
            builder=build_power_ratings_v3_recency,
        )
        power_v3_away_rating = _prior_power_rating(
            away_team,
            prior_team_game_rows,
            rating_column="power_rating_v3_recency",
            builder=build_power_ratings_v3_recency,
        )
        power_v3_margin = power_v3_home_rating - power_v3_away_rating
        power_v3_home_probability = power_v2_win_probability(
            power_v3_margin,
            scale=baseline_logistic_scale,
        )
        power_v3_winner = home_team if power_v3_home_probability >= 0.5 else away_team

        power_v4_home_rating = _prior_power_rating(
            home_team,
            prior_team_game_rows,
            rating_column="power_rating_v4",
            builder=build_power_ratings_v4_opponent_adjusted,
        )
        power_v4_away_rating = _prior_power_rating(
            away_team,
            prior_team_game_rows,
            rating_column="power_rating_v4",
            builder=build_power_ratings_v4_opponent_adjusted,
        )
        power_v4_margin = power_v4_home_rating - power_v4_away_rating
        power_v4_home_probability = calibrated_power_v4_probability(power_v4_margin)
        power_v4_winner = home_team if power_v4_home_probability >= 0.5 else away_team

        comparison_rows.append(
            {
                "game_date": game["game_date"].date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "actual_winner": actual_winner,
                "power_v3_recency_predicted_winner": power_v3_winner,
                "power_v3_recency_win_probability": max(
                    power_v3_home_probability,
                    1.0 - power_v3_home_probability,
                ),
                "power_v4_predicted_winner": power_v4_winner,
                "power_v4_win_probability": max(
                    power_v4_home_probability,
                    1.0 - power_v4_home_probability,
                ),
                "power_v3_recency_correct": power_v3_winner == actual_winner,
                "power_v4_correct": power_v4_winner == actual_winner,
                "power_v3_recency_predicted_margin": power_v3_margin,
                "power_v4_predicted_margin": power_v4_margin,
                "actual_margin": actual_margin,
                "predicted_total_goals": predicted_total_goals,
                "actual_total_goals": actual_total_goals,
            }
        )
        probability_rows.append(
            {
                "home_result": _home_result(home_score, away_score),
                "power_v3_recency_home_probability": power_v3_home_probability,
                "power_v4_home_probability": power_v4_home_probability,
            }
        )
        prior_team_game_rows.extend(_team_game_rows(game))

    comparison = pd.DataFrame(comparison_rows, columns=MODEL_COMPARISON_POWER_V4_COLUMNS)
    validate_model_comparison_power_v4(comparison, completed)
    probabilities = pd.DataFrame(probability_rows)
    summary = build_model_comparison_power_v4_summary(comparison, probabilities)
    calibration = build_calibration_table(
        comparison,
        probability_column="power_v4_win_probability",
        correct_column="power_v4_correct",
    )
    return comparison, summary, calibration


def validate_model_comparison_power_v4(
    comparison: pd.DataFrame,
    completed_games: pd.DataFrame,
) -> None:
    if len(comparison) != len(completed_games):
        raise ValueError(
            "Power v4 comparison row count must equal completed games count: "
            f"{len(comparison)} != {len(completed_games)}."
        )
    required_columns = [
        "power_v3_recency_predicted_winner",
        "power_v3_recency_win_probability",
        "power_v4_predicted_winner",
        "power_v4_win_probability",
    ]
    if comparison[required_columns].isna().any(axis=1).any():
        raise ValueError("All Power v4 comparison rows must include baseline and v4 predictions.")
    if _comparison_keys(comparison) != _comparison_keys(completed_games):
        raise ValueError("Power v4 comparison rows must cover the same completed games.")


def build_model_comparison_power_v4_summary(
    comparison: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(
            [
                {
                    column: 0.0
                    for column in MODEL_COMPARISON_POWER_V4_SUMMARY_COLUMNS
                }
            ],
            columns=MODEL_COMPARISON_POWER_V4_SUMMARY_COLUMNS,
        )

    total_goals_errors = (
        comparison["predicted_total_goals"] - comparison["actual_total_goals"]
    ).abs()
    return pd.DataFrame(
        [
            {
                "total_games": int(len(comparison)),
                "power_v3_recency_accuracy": float(comparison["power_v3_recency_correct"].mean()),
                "power_v4_accuracy": float(comparison["power_v4_correct"].mean()),
                "power_v3_recency_brier_score": _brier(
                    probabilities["power_v3_recency_home_probability"],
                    probabilities["home_result"],
                ),
                "power_v4_brier_score": _brier(
                    probabilities["power_v4_home_probability"],
                    probabilities["home_result"],
                ),
                "power_v3_recency_log_loss": _log_loss(
                    probabilities["power_v3_recency_home_probability"],
                    probabilities["home_result"],
                ),
                "power_v4_log_loss": _log_loss(
                    probabilities["power_v4_home_probability"],
                    probabilities["home_result"],
                ),
                "power_v3_recency_margin_mae": float(
                    (
                        comparison["power_v3_recency_predicted_margin"]
                        - comparison["actual_margin"]
                    )
                    .abs()
                    .mean()
                ),
                "power_v4_margin_mae": float(
                    (comparison["power_v4_predicted_margin"] - comparison["actual_margin"])
                    .abs()
                    .mean()
                ),
                "power_v3_recency_total_goals_mae": float(total_goals_errors.mean()),
                "power_v4_total_goals_mae": float(total_goals_errors.mean()),
            }
        ],
        columns=MODEL_COMPARISON_POWER_V4_SUMMARY_COLUMNS,
    )


def export_model_comparison_power_v4_outputs(
    games: pd.DataFrame,
    comparison_output_path: str | Path = DEFAULT_MODEL_COMPARISON_POWER_V4_CSV,
    summary_output_path: str | Path = DEFAULT_MODEL_COMPARISON_POWER_V4_SUMMARY_CSV,
    calibration_output_path: str | Path = DEFAULT_POWER_V4_CALIBRATION_CSV,
    **kwargs,
) -> tuple[Path, Path, Path]:
    comparison, summary, calibration = build_model_comparison_power_v4_outputs(games, **kwargs)
    return (
        export_csv(comparison, comparison_output_path),
        export_csv(summary, summary_output_path),
        export_csv(calibration, calibration_output_path),
    )


def _prior_power_rating(
    team: str,
    prior_team_game_rows: list[dict[str, object]],
    *,
    rating_column: str,
    builder,
) -> float:
    if not prior_team_game_rows:
        return 0.0

    power_ratings = builder(pd.DataFrame(prior_team_game_rows))
    matches = power_ratings[power_ratings["team"] == team]
    if matches.empty:
        return 0.0
    return float(matches.iloc[0][rating_column])


def _prior_total_goals(prior_team_game_rows: list[dict[str, object]]) -> float:
    if not prior_team_game_rows:
        return DEFAULT_TOTAL_GOALS
    frame = pd.DataFrame(prior_team_game_rows)
    frame["points_for"] = pd.to_numeric(frame["points_for"], errors="coerce")
    return float(frame["points_for"].dropna().mean() * 2.0)


def _brier(probabilities: pd.Series, results: pd.Series) -> float:
    return float(((probabilities - results) ** 2).mean())


def _log_loss(probabilities: pd.Series, results: pd.Series) -> float:
    clipped = probabilities.clip(1e-6, 1.0 - 1e-6)
    return float(
        -(
            results * clipped.map(math.log)
            + (1.0 - results) * (1.0 - clipped).map(math.log)
        ).mean()
    )
