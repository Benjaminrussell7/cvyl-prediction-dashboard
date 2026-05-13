from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from cvyl_scraper.power_v2 import build_power_ratings_v2
from cvyl_scraper.power_v3_recency import add_recency_weights, build_power_ratings_v3_recency


def test_power_v3_recency_outputs_are_deterministic() -> None:
    rows = [
        _team_game("Avon", "Canton", 6, 10, "2026-04-01", "game-1"),
        _team_game("Avon", "Granby", 11, 4, "2026-04-10", "game-2"),
        _team_game("Canton", "Avon", 10, 6, "2026-04-01", "game-1"),
        _team_game("Granby", "Avon", 4, 11, "2026-04-10", "game-2"),
    ]

    first = build_power_ratings_v3_recency(pd.DataFrame(rows))
    second = build_power_ratings_v3_recency(pd.DataFrame([rows[2], rows[0], rows[3], rows[1]]))

    assert_frame_equal(first, second)


def test_power_v3_recency_uses_completed_games_only() -> None:
    ratings = build_power_ratings_v3_recency(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 10, 4, "2026-04-01", "game-1"),
                _team_game("Avon", "Granby", 0, 20, "2026-04-08", "game-2", status="scheduled"),
                _team_game("Canton", "Avon", 4, 10, "2026-04-01", "game-1"),
            ]
        )
    )

    assert set(ratings["team"]) == {"Avon", "Canton"}
    assert set(ratings["games_played"]) == {1}


def test_power_v3_recency_weights_favor_recent_games() -> None:
    weighted = add_recency_weights(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 4, 10, "2026-04-01", "game-1"),
                _team_game("Avon", "Canton", 12, 5, "2026-04-20", "game-2"),
            ]
        ),
        recency_min_weight=0.8,
        recency_max_weight=1.2,
    )

    assert weighted.iloc[0]["recency_weight"] == 0.8
    assert weighted.iloc[-1]["recency_weight"] == 1.2


def test_power_v3_recency_preserves_v2_outputs() -> None:
    rows = pd.DataFrame(
        [
            _team_game("Avon", "Canton", 10, 4, "2026-04-01", "game-1"),
            _team_game("Canton", "Avon", 4, 10, "2026-04-01", "game-1"),
            _team_game("Avon", "Granby", 8, 5, "2026-04-08", "game-2"),
            _team_game("Granby", "Avon", 5, 8, "2026-04-08", "game-2"),
        ]
    )
    before = build_power_ratings_v2(rows)

    build_power_ratings_v3_recency(rows)
    after = build_power_ratings_v2(rows)

    assert_frame_equal(before, after)


def test_power_v3_recency_shrinkage_and_confidence_are_retained() -> None:
    ratings = build_power_ratings_v3_recency(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 12, 2, "2026-04-01", "game-1"),
                _team_game("Canton", "Avon", 2, 12, "2026-04-01", "game-1"),
            ]
        ),
        shrinkage_factor=5,
    ).set_index("team")

    assert ratings.loc["Avon", "shrinkage_multiplier"] == 1 / 6
    assert ratings.loc["Avon", "confidence_tier"] == "very low"
    assert abs(ratings.loc["Avon", "adjusted_margin_rating"]) <= 8


def _team_game(
    team: str,
    opponent: str,
    points_for: int,
    points_against: int,
    game_date: str,
    game_id: str,
    *,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "team": team,
        "opponent": opponent,
        "points_for": points_for,
        "points_against": points_against,
        "game_date": game_date,
        "status": status,
    }
