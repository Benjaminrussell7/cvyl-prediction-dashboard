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
from cvyl_scraper.hybrid import (
    DEFAULT_POWER_V2_LOGISTIC_SCALE,
    hybrid_win_probability,
    power_v2_win_probability,
)
from cvyl_scraper.power_v2 import build_power_ratings_v2
from cvyl_scraper.prediction import elo_win_probability


DEFAULT_MODEL_COMPARISON_CSV = "data/processed/cvyl_model_comparison.csv"
DEFAULT_MODEL_COMPARISON_SUMMARY_CSV = "data/processed/cvyl_model_comparison_summary.csv"

MODEL_COMPARISON_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "actual_winner",
    "elo_predicted_winner",
    "elo_win_probability",
    "power_v2_predicted_winner",
    "power_v2_win_probability",
    "hybrid_predicted_winner",
    "hybrid_win_probability",
    "elo_correct",
    "power_v2_correct",
    "hybrid_correct",
]

MODEL_COMPARISON_SUMMARY_COLUMNS = [
    "total_games",
    "elo_accuracy",
    "power_v2_accuracy",
    "hybrid_accuracy",
    "elo_brier_score",
    "power_v2_brier_score",
    "hybrid_brier_score",
]


def build_model_comparison_outputs(
    games: pd.DataFrame,
    *,
    starting_elo: float = DEFAULT_STARTING_ELO,
    k_factor: float = DEFAULT_K_FACTOR,
    recency_min_multiplier: float = DEFAULT_RECENCY_MIN_MULTIPLIER,
    recency_growth_games: float = DEFAULT_RECENCY_GROWTH_GAMES,
    power_v2_logistic_scale: float = DEFAULT_POWER_V2_LOGISTIC_SCALE,
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
        elo_win_probability_value = max(elo_home_probability, 1.0 - elo_home_probability)

        power_home_rating = _prior_power_rating(home_team, prior_team_game_rows)
        power_away_rating = _prior_power_rating(away_team, prior_team_game_rows)
        power_home_probability = power_v2_win_probability(
            power_home_rating - power_away_rating,
            scale=power_v2_logistic_scale,
        )
        power_predicted_winner = home_team if power_home_probability >= 0.5 else away_team
        power_win_probability_value = max(power_home_probability, 1.0 - power_home_probability)
        hybrid_home_probability = hybrid_win_probability(
            pregame_home_elo - pregame_away_elo,
            power_home_rating - power_away_rating,
        )
        hybrid_predicted_winner = home_team if hybrid_home_probability >= 0.5 else away_team
        hybrid_win_probability_value = max(hybrid_home_probability, 1.0 - hybrid_home_probability)

        comparison_rows.append(
            {
                "game_date": game["game_date"].date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "actual_winner": actual_winner,
                "elo_predicted_winner": elo_predicted_winner,
                "elo_win_probability": elo_win_probability_value,
                "power_v2_predicted_winner": power_predicted_winner,
                "power_v2_win_probability": power_win_probability_value,
                "hybrid_predicted_winner": hybrid_predicted_winner,
                "hybrid_win_probability": hybrid_win_probability_value,
                "elo_correct": elo_predicted_winner == actual_winner,
                "power_v2_correct": power_predicted_winner == actual_winner,
                "hybrid_correct": hybrid_predicted_winner == actual_winner,
            }
        )
        probability_rows.append(
            {
                "home_result": _home_result(home_score, away_score),
                "elo_home_probability": elo_home_probability,
                "power_home_probability": power_home_probability,
                "hybrid_home_probability": hybrid_home_probability,
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

    comparison = pd.DataFrame(comparison_rows, columns=MODEL_COMPARISON_COLUMNS)
    probabilities = pd.DataFrame(probability_rows)
    validate_model_comparison(comparison, completed)
    summary = build_model_comparison_summary(comparison, probabilities)
    return comparison, summary


def validate_model_comparison(comparison: pd.DataFrame, completed_games: pd.DataFrame) -> None:
    if len(comparison) != len(completed_games):
        raise ValueError(
            "Model comparison row count must equal completed games count: "
            f"{len(comparison)} != {len(completed_games)}."
        )

    required_model_columns = [
        "elo_predicted_winner",
        "elo_win_probability",
        "power_v2_predicted_winner",
        "power_v2_win_probability",
        "hybrid_predicted_winner",
        "hybrid_win_probability",
    ]
    missing_predictions = comparison[required_model_columns].isna().any(axis=1)
    if missing_predictions.any():
        raise ValueError("All comparison rows must include ELO, Power v2, and Hybrid predictions.")

    completed_keys = _comparison_keys(completed_games)
    comparison_keys = _comparison_keys(comparison)
    if comparison_keys != completed_keys:
        raise ValueError("Model comparison rows must cover the same games as completed games input.")


def build_model_comparison_summary(
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
                    "hybrid_accuracy": 0.0,
                    "elo_brier_score": 0.0,
                    "power_v2_brier_score": 0.0,
                    "hybrid_brier_score": 0.0,
                }
            ],
            columns=MODEL_COMPARISON_SUMMARY_COLUMNS,
        )

    elo_brier = ((probabilities["elo_home_probability"] - probabilities["home_result"]) ** 2).mean()
    power_brier = (
        (probabilities["power_home_probability"] - probabilities["home_result"]) ** 2
    ).mean()
    hybrid_brier = (
        (probabilities["hybrid_home_probability"] - probabilities["home_result"]) ** 2
    ).mean()
    return pd.DataFrame(
        [
            {
                "total_games": int(len(comparison)),
                "elo_accuracy": float(comparison["elo_correct"].mean()),
                "power_v2_accuracy": float(comparison["power_v2_correct"].mean()),
                "hybrid_accuracy": float(comparison["hybrid_correct"].mean()),
                "elo_brier_score": float(elo_brier),
                "power_v2_brier_score": float(power_brier),
                "hybrid_brier_score": float(hybrid_brier),
            }
        ],
        columns=MODEL_COMPARISON_SUMMARY_COLUMNS,
    )


def export_model_comparison_outputs(
    games: pd.DataFrame,
    comparison_output_path: str | Path = DEFAULT_MODEL_COMPARISON_CSV,
    summary_output_path: str | Path = DEFAULT_MODEL_COMPARISON_SUMMARY_CSV,
    **kwargs,
) -> tuple[Path, Path]:
    comparison, summary = build_model_comparison_outputs(games, **kwargs)
    return export_csv(comparison, comparison_output_path), export_csv(summary, summary_output_path)


def _prior_power_rating(team: str, prior_team_game_rows: list[dict[str, object]]) -> float:
    if not prior_team_game_rows:
        return 0.0

    power_ratings = build_power_ratings_v2(pd.DataFrame(prior_team_game_rows))
    matches = power_ratings[power_ratings["team"] == team]
    if matches.empty:
        return 0.0
    return float(matches.iloc[0]["power_rating_v2"])


def _completed_games(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    completed = games[games["status"] == "completed"].copy()
    completed["game_date"] = pd.to_datetime(completed["game_date"], errors="coerce")
    completed["home_score"] = pd.to_numeric(completed["home_score"], errors="coerce")
    completed["away_score"] = pd.to_numeric(completed["away_score"], errors="coerce")
    completed = completed.dropna(
        subset=["game_id", "game_date", "home_team", "away_team", "home_score", "away_score"]
    )
    return completed.sort_values(
        by=["game_date", "game_time", "game_id"],
        ascending=[True, True, True],
        na_position="last",
        ignore_index=True,
    )


def _comparison_keys(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    keys = frame[["game_date", "home_team", "away_team"]].copy()
    keys["game_date"] = pd.to_datetime(keys["game_date"], errors="coerce").dt.date.astype(str)
    return list(keys.itertuples(index=False, name=None))


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
            "status": "completed",
        },
        {
            "game_id": game["game_id"],
            "team": away_team,
            "opponent": home_team,
            "points_for": away_score,
            "points_against": home_score,
            "status": "completed",
        },
    ]


def _actual_winner(home_team: str, away_team: str, home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return home_team
    if away_score > home_score:
        return away_team
    return "Tie"


def _home_result(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5
