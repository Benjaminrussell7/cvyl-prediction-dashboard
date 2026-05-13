from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from cvyl_scraper.power_v2 import build_power_ratings_v2


def test_build_power_ratings_v2_rewards_stronger_scoring_margins() -> None:
    ratings = build_power_ratings_v2(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 10, 4),
                _team_game("Avon", "Granby", 9, 5),
                _team_game("Canton", "Avon", 4, 10),
                _team_game("Canton", "Granby", 4, 8),
                _team_game("Granby", "Avon", 5, 9),
                _team_game("Granby", "Canton", 8, 4),
            ]
        )
    ).set_index("team")

    assert ratings.loc["Avon", "power_rating_v2"] > ratings.loc["Canton", "power_rating_v2"]
    assert ratings.loc["Avon", "power_rank_v2"] < ratings.loc["Canton", "power_rank_v2"]


def test_build_power_ratings_v2_blowout_cap_limits_extreme_games() -> None:
    capped = build_power_ratings_v2(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 20, 0),
                _team_game("Canton", "Avon", 0, 20),
            ]
        ),
        margin_cap=6,
    ).set_index("team")

    assert abs(capped.loc["Avon", "adjusted_margin_rating"]) <= 6
    assert abs(capped.loc["Canton", "adjusted_margin_rating"]) <= 6
    assert capped.loc["Avon", "adjusted_margin_rating"] > capped.loc["Canton", "adjusted_margin_rating"]


def test_build_power_ratings_v2_is_deterministic() -> None:
    rows = [
        _team_game("Avon", "Canton", 10, 4),
        _team_game("Avon", "Granby", 9, 5),
        _team_game("Canton", "Avon", 4, 10),
        _team_game("Canton", "Granby", 4, 8),
        _team_game("Granby", "Avon", 5, 9),
        _team_game("Granby", "Canton", 8, 4),
    ]

    first = build_power_ratings_v2(pd.DataFrame(rows))
    second = build_power_ratings_v2(pd.DataFrame([rows[2], rows[0], rows[5], rows[1], rows[4], rows[3]]))

    assert_frame_equal(first, second)


def test_build_power_ratings_v2_uses_completed_games_only() -> None:
    ratings = build_power_ratings_v2(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 10, 4),
                _team_game("Avon", "Granby", 0, 20, status="scheduled"),
                _team_game("Canton", "Avon", 4, 10),
            ]
        )
    )

    assert set(ratings["team"]) == {"Avon", "Canton"}
    assert set(ratings["games_played"]) == {1}


def test_build_power_ratings_v2_regresses_one_game_teams_significantly() -> None:
    ratings = build_power_ratings_v2(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 14, 2),
                _team_game("Canton", "Avon", 2, 14),
            ]
        ),
        shrinkage_factor=5,
    ).set_index("team")

    assert ratings.loc["Avon", "shrinkage_multiplier"] == 1 / 6
    assert ratings.loc["Avon", "power_rating_v2"] < ratings.loc["Avon", "adjusted_margin_rating"]
    assert ratings.loc["Avon", "confidence_tier"] == "very low"


def test_build_power_ratings_v2_larger_samples_preserve_more_strength() -> None:
    ratings = build_power_ratings_v2(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 10, 4),
                _team_game("Avon", "Granby", 9, 4),
                _team_game("Avon", "Windsor", 8, 4),
                _team_game("Avon", "Berlin", 9, 5),
                _team_game("Avon", "Bristol", 10, 6),
                _team_game("Avon", "Farmington", 9, 3),
                _team_game("Canton", "Avon", 4, 10),
                _team_game("Granby", "Avon", 4, 9),
                _team_game("Windsor", "Avon", 4, 8),
                _team_game("Berlin", "Avon", 5, 9),
                _team_game("Bristol", "Avon", 6, 10),
                _team_game("Farmington", "Avon", 3, 9),
            ]
        ),
        shrinkage_factor=5,
    ).set_index("team")

    assert ratings.loc["Avon", "shrinkage_multiplier"] == 6 / 11
    assert ratings.loc["Avon", "confidence_tier"] == "high"
    assert ratings.loc["Avon", "power_rating_v2"] > 0
    assert ratings.loc["Avon", "power_rating_v2"] > ratings.loc["Canton", "power_rating_v2"]


def _team_game(
    team: str,
    opponent: str,
    points_for: int,
    points_against: int,
    *,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "team": team,
        "opponent": opponent,
        "points_for": points_for,
        "points_against": points_against,
        "status": status,
    }
