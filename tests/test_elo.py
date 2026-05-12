from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from cvyl_scraper.elo import build_elo_outputs


def test_build_elo_outputs_updates_games_chronologically() -> None:
    team_games = pd.DataFrame(
        [
            _team_game("game-2", "2026-04-12", "Avon", "Granby", 5, 7),
            _team_game("game-1", "2026-04-10", "Avon", "Canton", 8, 7),
            _team_game("game-2", "2026-04-12", "Granby", "Avon", 7, 5),
            _team_game("game-1", "2026-04-10", "Canton", "Avon", 7, 8),
        ]
    )

    _, history = build_elo_outputs(team_games, k_factor=20)

    avon_rows = history[history["team"] == "Avon"].sort_values("game_date")
    assert avon_rows.iloc[0]["game_id"] == "game-1"
    assert avon_rows.iloc[1]["game_id"] == "game-2"
    assert avon_rows.iloc[1]["pregame_elo"] == avon_rows.iloc[0]["postgame_elo"]


def test_build_elo_outputs_winner_gains_and_loser_loses_rating() -> None:
    team_games = pd.DataFrame(
        [
            _team_game("game-1", "2026-04-10", "Avon", "Canton", 8, 7),
            _team_game("game-1", "2026-04-10", "Canton", "Avon", 7, 8),
        ]
    )

    ratings, history = build_elo_outputs(team_games, k_factor=20)
    ratings_by_team = ratings.set_index("team")
    history_by_team = history.set_index("team")

    assert ratings_by_team.loc["Avon", "elo"] > 1500
    assert ratings_by_team.loc["Canton", "elo"] < 1500
    assert history_by_team.loc["Avon", "postgame_elo"] > history_by_team.loc["Avon", "pregame_elo"]
    assert (
        history_by_team.loc["Canton", "postgame_elo"]
        < history_by_team.loc["Canton", "pregame_elo"]
    )


def test_build_elo_outputs_are_deterministic_for_shuffled_input() -> None:
    rows = [
        _team_game("game-2", "2026-04-12", "Avon", "Granby", 5, 7),
        _team_game("game-1", "2026-04-10", "Avon", "Canton", 8, 7),
        _team_game("game-2", "2026-04-12", "Granby", "Avon", 7, 5),
        _team_game("game-1", "2026-04-10", "Canton", "Avon", 7, 8),
    ]
    ordered = pd.DataFrame(rows)
    shuffled = pd.DataFrame([rows[2], rows[0], rows[3], rows[1]])

    ordered_ratings, ordered_history = build_elo_outputs(ordered, k_factor=20)
    shuffled_ratings, shuffled_history = build_elo_outputs(shuffled, k_factor=20)

    assert_frame_equal(ordered_ratings, shuffled_ratings)
    assert_frame_equal(ordered_history, shuffled_history)


def test_build_elo_outputs_excludes_scheduled_games() -> None:
    team_games = pd.DataFrame(
        [
            _team_game("game-1", "2026-04-10", "Avon", "Canton", 8, 7),
            _team_game("game-1", "2026-04-10", "Canton", "Avon", 7, 8),
            _team_game(
                "game-2",
                "2026-04-12",
                "Granby",
                "Simsbury",
                None,
                None,
                status="scheduled",
            ),
            _team_game(
                "game-2",
                "2026-04-12",
                "Simsbury",
                "Granby",
                None,
                None,
                status="scheduled",
            ),
        ]
    )

    ratings, history = build_elo_outputs(team_games)

    assert set(ratings["team"]) == {"Avon", "Canton"}
    assert set(history["game_id"]) == {"game-1"}


def _team_game(
    game_id: str,
    game_date: str,
    team: str,
    opponent: str,
    points_for: int | None,
    points_against: int | None,
    *,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "team": team,
        "opponent": opponent,
        "points_for": points_for,
        "points_against": points_against,
        "win": points_for is not None and points_against is not None and points_for > points_against,
        "game_date": game_date,
        "season": 2026,
        "division": "14U Boys",
        "status": status,
    }
