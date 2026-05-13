from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from cvyl_scraper.sos import build_sos


def test_build_sos_gives_higher_sos_for_stronger_opponents() -> None:
    ratings = _ratings()
    team_games = pd.DataFrame(
        [
            _team_game("Avon", "Strong A"),
            _team_game("Avon", "Strong B"),
            _team_game("Canton", "Weak A"),
            _team_game("Canton", "Weak B"),
        ]
    )

    sos = build_sos(team_games, ratings).set_index("team")

    assert sos.loc["Avon", "average_opponent_elo"] > sos.loc["Canton", "average_opponent_elo"]
    assert sos.loc["Avon", "sos_rank"] < sos.loc["Canton", "sos_rank"]


def test_build_sos_is_deterministic_for_shuffled_input() -> None:
    ratings = _ratings()
    rows = [
        _team_game("Avon", "Strong A"),
        _team_game("Avon", "Strong B"),
        _team_game("Canton", "Weak A"),
        _team_game("Canton", "Weak B"),
    ]

    first = build_sos(pd.DataFrame(rows), ratings)
    second = build_sos(pd.DataFrame([rows[2], rows[0], rows[3], rows[1]]), ratings)

    assert_frame_equal(first, second)


def test_build_sos_uses_completed_games_only() -> None:
    ratings = _ratings()
    team_games = pd.DataFrame(
        [
            _team_game("Avon", "Strong A"),
            _team_game("Avon", "Weak A", status="scheduled"),
        ]
    )

    sos = build_sos(team_games, ratings)

    assert len(sos) == 1
    assert sos.iloc[0]["team"] == "Avon"
    assert sos.iloc[0]["games_played"] == 1
    assert sos.iloc[0]["opponent_count"] == 1
    assert sos.iloc[0]["average_opponent_elo"] == 1600


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "Strong A", "elo": 1600.0, "games_played": 5},
            {"team": "Strong B", "elo": 1550.0, "games_played": 5},
            {"team": "Weak A", "elo": 1450.0, "games_played": 5},
            {"team": "Weak B", "elo": 1400.0, "games_played": 5},
        ]
    )


def _team_game(team: str, opponent: str, *, status: str = "completed") -> dict[str, object]:
    return {
        "team": team,
        "opponent": opponent,
        "points_for": 8,
        "points_against": 5,
        "status": status,
    }
