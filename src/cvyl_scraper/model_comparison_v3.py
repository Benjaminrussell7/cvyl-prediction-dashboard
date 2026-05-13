from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.elo import (
    DEFAULT_K_FACTOR,
    DEFAULT_RECENCY_GROWTH_GAMES,
    DEFAULT_RECENCY_MIN_MULTIPLIER,
    DEFAULT_STARTING_ELO,
    recency_multiplier,
    updated_elo_pair,
)
from cvyl_scraper.export import export_csv
from cvyl_scraper.hybrid import DEFAULT_POWER_V2_LOGISTIC_SCALE, power_v2_win_probability
from cvyl_scraper.model_comparison import (
    _actual_winner,
    _completed_games,
    _comparison_keys,
    _home_result,
)
from cvyl_scraper.power_v2 import build_power_ratings_v2
from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency
from cvyl_scraper.prediction import elo_win_probability


DEFAULT_MODEL_COMPARISON_V3_CSV = "data/processed/cvyl_model_comparison_v3.csv"
DEFAULT_MODEL_COMPARISON_V3_SUMMARY_CSV = "data/processed/cvyl_model_comparison_v3_summary.csv"

MODEL_COMPARISON_V3_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "actual_winner",
    "elo_predicted_winner",
    "elo_win_probability",
    "power_v2_predicted_winner",
    "power_v2_win_probability",
    "power_v3_recency_predicted_winner",
    "power_v3_recency_win_probability",
    "elo_correct",
    "power_v2_correct",
    "power_v3_recency_correct",
]

MODEL_COMPARISON_V3_SUMMARY_COLUMNS = [
    "total_games",
    "elo_accuracy",
    "power_v2_accuracy",
    "power_v3_recency_accuracy",
    "elo_brier_score",
    "power_v2_brier_score",
    "power_v3_recency_brier_score",
]


def build_model_comparison_v3_outputs(
    games: pd.DataFrame,
    *,
    starting_elo: float = DEFAULT_STARTING_ELO,
    k_factor: float = DEFAULT_K_FACTOR,
    recency_min_multiplier: float = DEFAULT_RECENCY_MIN_MULTIPLIER,
    recency_growth_games: float = DEFAULT_RECENCY_GROWTH_GAMES,
    power_logistic_scale: float = DEFAULT_POWER_V2_LOGISTIC_SCALE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    elo_ratings: dict[str, float] = {}
    prior_team_game_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, float]] = []

    completed = _completed_games(games)
    for game_index, (_, game) in enumerate(completed.iterrows()):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        actual_winner = _actual_winner(home_team, away_team, home_score, away_score)

        pregame_home_elo = elo_ratings.get(home_team, starting_elo)
        pregame_away_elo = elo_ratings.get(away_team, starting_elo)
        elo_home_probability = elo_win_probability(pregame_home_elo, pregame_away_elo)
        elo_predicted_winner = home_team if elo_home_probability >= 0.5 else away_team

        power_v2_home_rating = _prior_power_v2_rating(home_team, prior_team_game_rows)
        power_v2_away_rating = _prior_power_v2_rating(away_team, prior_team_game_rows)
        power_v2_home_probability = power_v2_win_probability(
            power_v2_home_rating - power_v2_away_rating,
            scale=power_logistic_scale,
        )
        power_v2_predicted_winner = home_team if power_v2_home_probability >= 0.5 else away_team

        power_v3_home_rating = _prior_power_v3_rating(home_team, prior_team_game_rows)
        power_v3_away_rating = _prior_power_v3_rating(away_team, prior_team_game_rows)
        power_v3_home_probability = power_v2_win_probability(
            power_v3_home_rating - power_v3_away_rating,
            scale=power_logistic_scale,
        )
        power_v3_predicted_winner = home_team if power_v3_home_probability >= 0.5 else away_team

        comparison_rows.append(
            {
                "game_date": game["game_date"].date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "actual_winner": actual_winner,
                "elo_predicted_winner": elo_predicted_winner,
                "elo_win_probability": max(elo_home_probability, 1.0 - elo_home_probability),
                "power_v2_predicted_winner": power_v2_predicted_winner,
                "power_v2_win_probability": max(
                    power_v2_home_probability,
                    1.0 - power_v2_home_probability,
                ),
                "power_v3_recency_predicted_winner": power_v3_predicted_winner,
                "power_v3_recency_win_probability": max(
                    power_v3_home_probability,
                    1.0 - power_v3_home_probability,
                ),
                "elo_correct": elo_predicted_winner == actual_winner,
                "power_v2_correct": power_v2_predicted_winner == actual_winner,
                "power_v3_recency_correct": power_v3_predicted_winner == actual_winner,
            }
        )
        probability_rows.append(
            {
                "home_result": _home_result(home_score, away_score),
                "elo_home_probability": elo_home_probability,
                "power_v2_home_probability": power_v2_home_probability,
                "power_v3_recency_home_probability": power_v3_home_probability,
            }
        )

        game_recency_multiplier = recency_multiplier(
            game_index,
            min_multiplier=recency_min_multiplier,
            growth_games=recency_growth_games,
        )
        home_postgame_elo, away_postgame_elo = updated_elo_pair(
            pregame_home_elo,
            pregame_away_elo,
            home_score,
            away_score,
            k_factor=k_factor,
            recency_multiplier_value=game_recency_multiplier,
        )
        elo_ratings[home_team] = home_postgame_elo
        elo_ratings[away_team] = away_postgame_elo
        prior_team_game_rows.extend(_team_game_rows(game))

    comparison = pd.DataFrame(comparison_rows, columns=MODEL_COMPARISON_V3_COLUMNS)
    validate_model_comparison_v3(comparison, completed)
    summary = build_model_comparison_v3_summary(comparison, pd.DataFrame(probability_rows))
    return comparison, summary


def validate_model_comparison_v3(comparison: pd.DataFrame, completed_games: pd.DataFrame) -> None:
    if len(comparison) != len(completed_games):
        raise ValueError(
            "Model comparison v3 row count must equal completed games count: "
            f"{len(comparison)} != {len(completed_games)}."
        )

    required_model_columns = [
        "elo_predicted_winner",
        "elo_win_probability",
        "power_v2_predicted_winner",
        "power_v2_win_probability",
        "power_v3_recency_predicted_winner",
        "power_v3_recency_win_probability",
    ]
    missing_predictions = comparison[required_model_columns].isna().any(axis=1)
    if missing_predictions.any():
        raise ValueError("All v3 comparison rows must include ELO, Power v2, and Power v3 predictions.")

    if _comparison_keys(comparison) != _comparison_keys(completed_games):
        raise ValueError("Model comparison v3 rows must cover the same games as completed games input.")


def build_model_comparison_v3_summary(
    comparison: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(
            [
                {
                    "total_games": 0,
                    "elo_accuracy": 0.0,
                    "power_v2_accuracy": 0.0,
                    "power_v3_recency_accuracy": 0.0,
                    "elo_brier_score": 0.0,
                    "power_v2_brier_score": 0.0,
                    "power_v3_recency_brier_score": 0.0,
                }
            ],
            columns=MODEL_COMPARISON_V3_SUMMARY_COLUMNS,
        )

    return pd.DataFrame(
        [
            {
                "total_games": int(len(comparison)),
                "elo_accuracy": float(comparison["elo_correct"].mean()),
                "power_v2_accuracy": float(comparison["power_v2_correct"].mean()),
                "power_v3_recency_accuracy": float(
                    comparison["power_v3_recency_correct"].mean()
                ),
                "elo_brier_score": _brier(
                    probabilities["elo_home_probability"],
                    probabilities["home_result"],
                ),
                "power_v2_brier_score": _brier(
                    probabilities["power_v2_home_probability"],
                    probabilities["home_result"],
                ),
                "power_v3_recency_brier_score": _brier(
                    probabilities["power_v3_recency_home_probability"],
                    probabilities["home_result"],
                ),
            }
        ],
        columns=MODEL_COMPARISON_V3_SUMMARY_COLUMNS,
    )


def export_model_comparison_v3_outputs(
    games: pd.DataFrame,
    comparison_output_path: str | Path = DEFAULT_MODEL_COMPARISON_V3_CSV,
    summary_output_path: str | Path = DEFAULT_MODEL_COMPARISON_V3_SUMMARY_CSV,
    **kwargs,
) -> tuple[Path, Path]:
    comparison, summary = build_model_comparison_v3_outputs(games, **kwargs)
    return export_csv(comparison, comparison_output_path), export_csv(summary, summary_output_path)


def _prior_power_v2_rating(team: str, prior_team_game_rows: list[dict[str, object]]) -> float:
    if not prior_team_game_rows:
        return 0.0

    power_ratings = build_power_ratings_v2(pd.DataFrame(prior_team_game_rows))
    matches = power_ratings[power_ratings["team"] == team]
    if matches.empty:
        return 0.0
    return float(matches.iloc[0]["power_rating_v2"])


def _prior_power_v3_rating(team: str, prior_team_game_rows: list[dict[str, object]]) -> float:
    if not prior_team_game_rows:
        return 0.0

    power_ratings = build_power_ratings_v3_recency(pd.DataFrame(prior_team_game_rows))
    matches = power_ratings[power_ratings["team"] == team]
    if matches.empty:
        return 0.0
    return float(matches.iloc[0]["power_rating_v3_recency"])


def _brier(probabilities: pd.Series, results: pd.Series) -> float:
    return float(((probabilities - results) ** 2).mean())


def _team_game_rows(game: pd.Series) -> list[dict[str, object]]:
    home_team = str(game["home_team"])
    away_team = str(game["away_team"])
    home_score = int(game["home_score"])
    away_score = int(game["away_score"])
    return [
        {
            "game_id": game["game_id"],
            "team": home_team,
            "opponent": away_team,
            "points_for": home_score,
            "points_against": away_score,
            "game_date": game["game_date"],
            "status": "completed",
        },
        {
            "game_id": game["game_id"],
            "team": away_team,
            "opponent": home_team,
            "points_for": away_score,
            "points_against": home_score,
            "game_date": game["game_date"],
            "status": "completed",
        },
    ]
