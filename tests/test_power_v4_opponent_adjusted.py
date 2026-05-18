from __future__ import annotations

import pandas as pd

from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency
from cvyl_scraper.power_v4_opponent_adjusted import build_power_ratings_v4_opponent_adjusted


def test_power_v4_outputs_expected_columns_and_ranks() -> None:
    ratings = build_power_ratings_v4_opponent_adjusted(pd.DataFrame(_team_games()))

    assert list(ratings["power_rank_v4"]) == list(range(1, len(ratings) + 1))
    assert {
        "team",
        "power_rating_v4",
        "power_rank_v4",
        "avg_performance_above_expectation",
        "baseline_power_rating_v3_recency",
        "baseline_power_rank_v3_recency",
    }.issubset(ratings.columns)
    assert ratings["power_rating_v4"].notna().all()


def test_power_v4_preserves_power_v3_builder() -> None:
    team_games = pd.DataFrame(_team_games())
    before = build_power_ratings_v3_recency(team_games)

    build_power_ratings_v4_opponent_adjusted(team_games)
    after = build_power_ratings_v3_recency(team_games)

    pd.testing.assert_frame_equal(before, after)


def test_power_v4_rewards_performance_relative_to_opponent_strength() -> None:
    ratings = build_power_ratings_v4_opponent_adjusted(pd.DataFrame(_team_games()))

    avon = ratings[ratings["team"] == "Avon"].iloc[0]
    canton = ratings[ratings["team"] == "Canton"].iloc[0]

    assert avon["power_rating_v4"] > canton["power_rating_v4"]
    assert avon["avg_performance_above_expectation"] > canton["avg_performance_above_expectation"]


def _team_games() -> list[dict[str, object]]:
    return [
        _row("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
        _row("game-1", "2026-04-10", "Canton", "Avon", 5, 10),
        _row("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
        _row("game-2", "2026-04-12", "Granby", "Avon", 4, 8),
        _row("game-3", "2026-04-14", "Canton", "Granby", 9, 4),
        _row("game-3", "2026-04-14", "Granby", "Canton", 4, 9),
    ]


def _row(
    game_id: str,
    game_date: str,
    team: str,
    opponent: str,
    points_for: int,
    points_against: int,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "game_date": game_date,
        "team": team,
        "opponent": opponent,
        "points_for": points_for,
        "points_against": points_against,
        "win": int(points_for > points_against),
        "status": "completed",
        "season": 2026,
        "division": "12U",
    }
