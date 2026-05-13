from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.export import export_csv


DEFAULT_POWER_RATINGS_V2_CSV = "data/processed/cvyl_power_ratings_v2.csv"
DEFAULT_MARGIN_CAP = 8.0
DEFAULT_SHRINKAGE_FACTOR = 5.0
OPPONENT_ADJUSTMENT_WEIGHT = 0.25

POWER_RATINGS_V2_COLUMNS = [
    "team",
    "games_played",
    "avg_points_for",
    "avg_points_against",
    "avg_margin",
    "adjusted_offense_rating",
    "adjusted_defense_rating",
    "adjusted_margin_rating",
    "power_rating_v2",
    "power_rank_v2",
    "confidence_tier",
    "shrinkage_multiplier",
]


def build_power_ratings_v2(
    team_games: pd.DataFrame,
    *,
    margin_cap: float = DEFAULT_MARGIN_CAP,
    shrinkage_factor: float = DEFAULT_SHRINKAGE_FACTOR,
) -> pd.DataFrame:
    if shrinkage_factor < 0:
        raise ValueError("shrinkage_factor must be non-negative.")

    completed = _completed_team_games(team_games)
    if completed.empty:
        return pd.DataFrame(columns=POWER_RATINGS_V2_COLUMNS)

    league_avg_points = float(completed["points_for"].mean())
    team_profiles = _team_profiles(completed)
    games = completed.merge(
        team_profiles[["team", "avg_points_for", "avg_points_against"]].rename(
            columns={
                "team": "opponent",
                "avg_points_for": "opponent_avg_points_for",
                "avg_points_against": "opponent_avg_points_against",
            }
        ),
        on="opponent",
        how="left",
    ).dropna(subset=["opponent_avg_points_for", "opponent_avg_points_against"])

    games["raw_margin"] = games["points_for"] - games["points_against"]
    games["capped_margin"] = games["raw_margin"].clip(lower=-margin_cap, upper=margin_cap)
    games["offense_game_rating"] = (
        games["points_for"] + (league_avg_points - games["opponent_avg_points_against"])
    ) - league_avg_points
    games["defense_game_rating"] = games["opponent_avg_points_for"] - games["points_against"]
    opponent_adjustment = (
        games["opponent_avg_points_for"] - games["opponent_avg_points_against"]
    ) * OPPONENT_ADJUSTMENT_WEIGHT
    games["adjusted_margin_game_rating"] = (games["capped_margin"] + opponent_adjustment).clip(
        lower=-margin_cap,
        upper=margin_cap,
    )

    ratings = (
        games.groupby("team", as_index=False)
        .agg(
            games_played=("points_for", "size"),
            avg_points_for=("points_for", "mean"),
            avg_points_against=("points_against", "mean"),
            avg_margin=("raw_margin", "mean"),
            adjusted_offense_rating=("offense_game_rating", "mean"),
            adjusted_defense_rating=("defense_game_rating", "mean"),
            adjusted_margin_rating=("adjusted_margin_game_rating", "mean"),
        )
    )
    ratings["shrinkage_multiplier"] = ratings["games_played"] / (
        ratings["games_played"] + shrinkage_factor
    )
    ratings["power_rating_v2"] = ratings["adjusted_margin_rating"] * ratings["shrinkage_multiplier"]
    ratings["confidence_tier"] = ratings["games_played"].map(_confidence_tier)
    ratings = ratings.sort_values(
        by=["power_rating_v2", "team"],
        ascending=[False, True],
        ignore_index=True,
    )
    ratings["power_rank_v2"] = range(1, len(ratings) + 1)
    return ratings[POWER_RATINGS_V2_COLUMNS]


def export_power_ratings_v2(
    team_games: pd.DataFrame,
    output_path: str | Path = DEFAULT_POWER_RATINGS_V2_CSV,
    *,
    margin_cap: float = DEFAULT_MARGIN_CAP,
    shrinkage_factor: float = DEFAULT_SHRINKAGE_FACTOR,
) -> Path:
    ratings = build_power_ratings_v2(
        team_games,
        margin_cap=margin_cap,
        shrinkage_factor=shrinkage_factor,
    )
    return export_csv(ratings, output_path)


def _completed_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame()

    completed = team_games[team_games["status"] == "completed"].copy()
    completed["points_for"] = pd.to_numeric(completed["points_for"], errors="coerce")
    completed["points_against"] = pd.to_numeric(completed["points_against"], errors="coerce")
    return completed.dropna(subset=["team", "opponent", "points_for", "points_against"])


def _team_profiles(completed: pd.DataFrame) -> pd.DataFrame:
    return completed.groupby("team", as_index=False).agg(
        avg_points_for=("points_for", "mean"),
        avg_points_against=("points_against", "mean"),
    )


def _confidence_tier(games_played: int) -> str:
    if games_played == 1:
        return "very low"
    if games_played <= 3:
        return "low"
    if games_played <= 5:
        return "medium"
    return "high"
