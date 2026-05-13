from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.export import export_csv
from cvyl_scraper.power_v2 import (
    DEFAULT_MARGIN_CAP,
    DEFAULT_SHRINKAGE_FACTOR,
    OPPONENT_ADJUSTMENT_WEIGHT,
)


DEFAULT_POWER_RATINGS_V3_RECENCY_CSV = "data/processed/cvyl_power_ratings_v3_recency.csv"
DEFAULT_RECENCY_MIN_WEIGHT = 0.85
DEFAULT_RECENCY_MAX_WEIGHT = 1.15

POWER_RATINGS_V3_RECENCY_COLUMNS = [
    "team",
    "games_played",
    "avg_points_for",
    "avg_points_against",
    "avg_margin",
    "adjusted_offense_rating",
    "adjusted_defense_rating",
    "adjusted_margin_rating",
    "power_rating_v3_recency",
    "power_rank_v3_recency",
    "confidence_tier",
    "shrinkage_multiplier",
    "average_recency_weight",
]


def build_power_ratings_v3_recency(
    team_games: pd.DataFrame,
    *,
    margin_cap: float = DEFAULT_MARGIN_CAP,
    shrinkage_factor: float = DEFAULT_SHRINKAGE_FACTOR,
    recency_min_weight: float = DEFAULT_RECENCY_MIN_WEIGHT,
    recency_max_weight: float = DEFAULT_RECENCY_MAX_WEIGHT,
) -> pd.DataFrame:
    if shrinkage_factor < 0:
        raise ValueError("shrinkage_factor must be non-negative.")
    if recency_min_weight <= 0 or recency_max_weight <= 0:
        raise ValueError("recency weights must be greater than zero.")
    if recency_min_weight > recency_max_weight:
        raise ValueError("recency_min_weight must be less than or equal to recency_max_weight.")

    completed = _completed_team_games(team_games)
    if completed.empty:
        return pd.DataFrame(columns=POWER_RATINGS_V3_RECENCY_COLUMNS)

    completed = add_recency_weights(
        completed,
        recency_min_weight=recency_min_weight,
        recency_max_weight=recency_max_weight,
    )
    league_avg_points = _weighted_average(completed["points_for"], completed["recency_weight"])
    team_profiles = _weighted_team_profiles(completed)
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
        games.groupby("team", sort=True)
        .apply(_weighted_rating_row, include_groups=False)
        .reset_index()
    )
    ratings["games_played"] = ratings["games_played"].astype(int)
    ratings["shrinkage_multiplier"] = ratings["games_played"] / (
        ratings["games_played"] + shrinkage_factor
    )
    ratings["power_rating_v3_recency"] = (
        ratings["adjusted_margin_rating"] * ratings["shrinkage_multiplier"]
    )
    ratings["confidence_tier"] = ratings["games_played"].map(_confidence_tier)
    ratings = ratings.sort_values(
        by=["power_rating_v3_recency", "team"],
        ascending=[False, True],
        ignore_index=True,
    )
    ratings["power_rank_v3_recency"] = range(1, len(ratings) + 1)
    return ratings[POWER_RATINGS_V3_RECENCY_COLUMNS]


def export_power_ratings_v3_recency(
    team_games: pd.DataFrame,
    output_path: str | Path = DEFAULT_POWER_RATINGS_V3_RECENCY_CSV,
    **kwargs,
) -> Path:
    ratings = build_power_ratings_v3_recency(team_games, **kwargs)
    return export_csv(ratings, output_path)


def add_recency_weights(
    completed_team_games: pd.DataFrame,
    *,
    recency_min_weight: float = DEFAULT_RECENCY_MIN_WEIGHT,
    recency_max_weight: float = DEFAULT_RECENCY_MAX_WEIGHT,
) -> pd.DataFrame:
    weighted = completed_team_games.copy()
    weighted = weighted.sort_values(
        by=[column for column in ["game_date", "game_id", "team"] if column in weighted.columns],
        kind="mergesort",
        ignore_index=True,
    )
    if len(weighted) == 1:
        weighted["recency_weight"] = recency_max_weight
        return weighted

    positions = pd.Series(range(len(weighted)), index=weighted.index, dtype="float64")
    step = (recency_max_weight - recency_min_weight) / (len(weighted) - 1)
    weighted["recency_weight"] = recency_min_weight + (positions * step)
    return weighted


def _completed_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame()

    completed = team_games[team_games["status"] == "completed"].copy()
    completed["points_for"] = pd.to_numeric(completed["points_for"], errors="coerce")
    completed["points_against"] = pd.to_numeric(completed["points_against"], errors="coerce")
    completed["game_date"] = pd.to_datetime(completed.get("game_date"), errors="coerce")
    return completed.dropna(subset=["team", "opponent", "points_for", "points_against"])


def _weighted_team_profiles(completed: pd.DataFrame) -> pd.DataFrame:
    return (
        completed.groupby("team", sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "avg_points_for": _weighted_average(
                        group["points_for"],
                        group["recency_weight"],
                    ),
                    "avg_points_against": _weighted_average(
                        group["points_against"],
                        group["recency_weight"],
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def _weighted_rating_row(group: pd.DataFrame) -> pd.Series:
    weights = group["recency_weight"]
    return pd.Series(
        {
            "games_played": len(group),
            "avg_points_for": _weighted_average(group["points_for"], weights),
            "avg_points_against": _weighted_average(group["points_against"], weights),
            "avg_margin": _weighted_average(group["raw_margin"], weights),
            "adjusted_offense_rating": _weighted_average(group["offense_game_rating"], weights),
            "adjusted_defense_rating": _weighted_average(group["defense_game_rating"], weights),
            "adjusted_margin_rating": _weighted_average(
                group["adjusted_margin_game_rating"],
                weights,
            ),
            "average_recency_weight": float(weights.mean()),
        }
    )


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float((values.astype(float) * weights.astype(float)).sum() / weights.astype(float).sum())


def _confidence_tier(games_played: int) -> str:
    if games_played == 1:
        return "very low"
    if games_played <= 3:
        return "low"
    if games_played <= 5:
        return "medium"
    return "high"
