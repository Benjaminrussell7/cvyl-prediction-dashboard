from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.export import export_csv
from cvyl_scraper.power_v2 import DEFAULT_MARGIN_CAP, DEFAULT_SHRINKAGE_FACTOR
from cvyl_scraper.power_v3_recency import (
    DEFAULT_RECENCY_MAX_WEIGHT,
    DEFAULT_RECENCY_MIN_WEIGHT,
    add_recency_weights,
    build_power_ratings_v3_recency,
)


DEFAULT_POWER_RATINGS_V4_CSV = "data/processed/cvyl_power_ratings_v4_opponent_adjusted.csv"

POWER_RATINGS_V4_COLUMNS = [
    "team",
    "games_played",
    "avg_points_for",
    "avg_points_against",
    "avg_margin",
    "adjusted_offense_rating",
    "adjusted_defense_rating",
    "adjusted_margin_rating",
    "avg_capped_margin",
    "avg_expected_margin_vs_opponent",
    "avg_performance_above_expectation",
    "power_rating_v4",
    "power_rank_v4",
    "confidence_tier",
    "shrinkage_multiplier",
    "average_recency_weight",
    "baseline_power_rating_v3_recency",
    "baseline_power_rank_v3_recency",
]


def build_power_ratings_v4_opponent_adjusted(
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
        return pd.DataFrame(columns=POWER_RATINGS_V4_COLUMNS)

    baseline = build_power_ratings_v3_recency(
        completed,
        margin_cap=margin_cap,
        shrinkage_factor=shrinkage_factor,
        recency_min_weight=recency_min_weight,
        recency_max_weight=recency_max_weight,
    )
    baseline_ratings = dict(
        zip(baseline["team"], baseline["power_rating_v3_recency"], strict=False)
    )

    weighted = add_recency_weights(
        completed,
        recency_min_weight=recency_min_weight,
        recency_max_weight=recency_max_weight,
    )
    weighted["raw_margin"] = weighted["points_for"] - weighted["points_against"]
    weighted["capped_margin"] = weighted["raw_margin"].clip(lower=-margin_cap, upper=margin_cap)
    weighted["opponent_power_v3_rating"] = (
        weighted["opponent"].map(baseline_ratings).fillna(0.0).astype(float)
    )
    weighted["expected_margin_vs_opponent"] = -weighted["opponent_power_v3_rating"]
    weighted["performance_above_expectation"] = (
        weighted["capped_margin"] - weighted["expected_margin_vs_opponent"]
    ).clip(lower=-margin_cap, upper=margin_cap)

    ratings = (
        weighted.groupby("team", sort=True)
        .apply(_weighted_opponent_adjusted_row, include_groups=False)
        .reset_index()
    )
    ratings["games_played"] = ratings["games_played"].astype(int)
    ratings["shrinkage_multiplier"] = ratings["games_played"] / (
        ratings["games_played"] + shrinkage_factor
    )
    ratings["power_rating_v4"] = (
        ratings["avg_performance_above_expectation"] * ratings["shrinkage_multiplier"]
    )
    ratings["confidence_tier"] = ratings["games_played"].map(_confidence_tier)

    baseline_support = baseline[
        [
            "team",
            "adjusted_offense_rating",
            "adjusted_defense_rating",
            "power_rating_v3_recency",
            "power_rank_v3_recency",
        ]
    ].rename(
        columns={
            "power_rating_v3_recency": "baseline_power_rating_v3_recency",
            "power_rank_v3_recency": "baseline_power_rank_v3_recency",
        }
    )
    ratings = ratings.merge(baseline_support, on="team", how="left")
    ratings["adjusted_margin_rating"] = ratings["avg_performance_above_expectation"]

    ratings = ratings.sort_values(
        by=["power_rating_v4", "team"],
        ascending=[False, True],
        ignore_index=True,
    )
    ratings["power_rank_v4"] = range(1, len(ratings) + 1)
    return ratings[POWER_RATINGS_V4_COLUMNS]


def export_power_ratings_v4_opponent_adjusted(
    team_games: pd.DataFrame,
    output_path: str | Path = DEFAULT_POWER_RATINGS_V4_CSV,
    **kwargs,
) -> Path:
    ratings = build_power_ratings_v4_opponent_adjusted(team_games, **kwargs)
    return export_csv(ratings, output_path)


def _completed_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame()

    completed = team_games[team_games["status"] == "completed"].copy()
    completed["points_for"] = pd.to_numeric(completed["points_for"], errors="coerce")
    completed["points_against"] = pd.to_numeric(completed["points_against"], errors="coerce")
    completed["game_date"] = pd.to_datetime(completed.get("game_date"), errors="coerce")
    return completed.dropna(subset=["team", "opponent", "points_for", "points_against"])


def _weighted_opponent_adjusted_row(group: pd.DataFrame) -> pd.Series:
    weights = group["recency_weight"]
    return pd.Series(
        {
            "games_played": len(group),
            "avg_points_for": _weighted_average(group["points_for"], weights),
            "avg_points_against": _weighted_average(group["points_against"], weights),
            "avg_margin": _weighted_average(group["raw_margin"], weights),
            "avg_capped_margin": _weighted_average(group["capped_margin"], weights),
            "avg_expected_margin_vs_opponent": _weighted_average(
                group["expected_margin_vs_opponent"],
                weights,
            ),
            "avg_performance_above_expectation": _weighted_average(
                group["performance_above_expectation"],
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
