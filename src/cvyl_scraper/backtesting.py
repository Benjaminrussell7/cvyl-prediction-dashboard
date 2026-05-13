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
from cvyl_scraper.prediction import elo_win_probability


DEFAULT_BACKTEST_CSV = "data/processed/cvyl_backtest.csv"
DEFAULT_BACKTEST_SUMMARY_CSV = "data/processed/cvyl_backtest_summary.csv"

BACKTEST_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "predicted_winner",
    "actual_winner",
    "predicted_win_probability",
    "prediction_correct",
    "pregame_home_elo",
    "pregame_away_elo",
]

BACKTEST_SUMMARY_COLUMNS = [
    "total_predictions",
    "accuracy",
    "average_confidence",
    "brier_score",
]


def build_backtest_outputs(
    games: pd.DataFrame,
    *,
    starting_elo: float = DEFAULT_STARTING_ELO,
    k_factor: float = DEFAULT_K_FACTOR,
    recency_min_multiplier: float = DEFAULT_RECENCY_MIN_MULTIPLIER,
    recency_growth_games: float = DEFAULT_RECENCY_GROWTH_GAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    completed = _completed_games(games)
    for game_index, (_, game) in enumerate(completed.iterrows()):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_elo = ratings.get(home_team, starting_elo)
        away_elo = ratings.get(away_team, starting_elo)
        home_win_probability = elo_win_probability(home_elo, away_elo)
        predicted_winner = home_team if home_win_probability >= 0.5 else away_team
        actual_winner = _actual_winner(home_team, away_team, game["home_score"], game["away_score"])
        predicted_win_probability = max(home_win_probability, 1.0 - home_win_probability)

        rows.append(
            {
                "game_id": game["game_id"],
                "game_date": game["game_date"].date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(game["home_score"]),
                "away_score": int(game["away_score"]),
                "predicted_winner": predicted_winner,
                "actual_winner": actual_winner,
                "predicted_win_probability": predicted_win_probability,
                "prediction_correct": predicted_winner == actual_winner,
                "pregame_home_elo": home_elo,
                "pregame_away_elo": away_elo,
            }
        )

        game_recency_multiplier = recency_multiplier(
            game_index,
            min_multiplier=recency_min_multiplier,
            growth_games=recency_growth_games,
        )
        home_postgame, away_postgame = updated_elo_pair(
            home_elo,
            away_elo,
            int(game["home_score"]),
            int(game["away_score"]),
            k_factor=k_factor,
            recency_multiplier_value=game_recency_multiplier,
        )
        ratings[home_team] = home_postgame
        ratings[away_team] = away_postgame

    backtest = pd.DataFrame(rows, columns=BACKTEST_COLUMNS)
    summary = build_backtest_summary(backtest)
    return backtest, summary


def build_backtest_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame(
            [{"total_predictions": 0, "accuracy": 0.0, "average_confidence": 0.0, "brier_score": 0.0}],
            columns=BACKTEST_SUMMARY_COLUMNS,
        )

    home_win_probabilities = backtest.apply(_home_win_probability_from_prediction, axis=1)
    home_results = backtest.apply(_home_result, axis=1)
    brier_score = ((home_win_probabilities - home_results) ** 2).mean()

    return pd.DataFrame(
        [
            {
                "total_predictions": int(len(backtest)),
                "accuracy": float(backtest["prediction_correct"].mean()),
                "average_confidence": float(backtest["predicted_win_probability"].mean()),
                "brier_score": float(brier_score),
            }
        ],
        columns=BACKTEST_SUMMARY_COLUMNS,
    )


def export_backtest_outputs(
    games: pd.DataFrame,
    backtest_output_path: str | Path = DEFAULT_BACKTEST_CSV,
    summary_output_path: str | Path = DEFAULT_BACKTEST_SUMMARY_CSV,
    *,
    starting_elo: float = DEFAULT_STARTING_ELO,
    k_factor: float = DEFAULT_K_FACTOR,
    recency_min_multiplier: float = DEFAULT_RECENCY_MIN_MULTIPLIER,
    recency_growth_games: float = DEFAULT_RECENCY_GROWTH_GAMES,
) -> tuple[Path, Path]:
    backtest, summary = build_backtest_outputs(
        games,
        starting_elo=starting_elo,
        k_factor=k_factor,
        recency_min_multiplier=recency_min_multiplier,
        recency_growth_games=recency_growth_games,
    )
    return export_csv(backtest, backtest_output_path), export_csv(summary, summary_output_path)


def _completed_games(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

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


def _actual_winner(home_team: str, away_team: str, home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return home_team
    if away_score > home_score:
        return away_team
    return "Tie"


def _home_win_probability_from_prediction(row: pd.Series) -> float:
    if row["predicted_winner"] == row["home_team"]:
        return float(row["predicted_win_probability"])
    return 1.0 - float(row["predicted_win_probability"])


def _home_result(row: pd.Series) -> float:
    if row["home_score"] > row["away_score"]:
        return 1.0
    if row["home_score"] < row["away_score"]:
        return 0.0
    return 0.5
