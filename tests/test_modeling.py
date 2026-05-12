from __future__ import annotations

import pandas as pd

from cvyl_scraper.cleaning import build_canonical_games
from cvyl_scraper.modeling import build_team_games


def test_build_team_games_creates_two_rows_per_completed_game() -> None:
    games = build_canonical_games(
        pd.DataFrame(
            [
                {
                    "game_date": "2026-04-10",
                    "game_time": "6:00 PM",
                    "season": 2026,
                    "division": "14U Boys",
                    "home_team": "Avon",
                    "away_team": "Canton",
                    "home_score": 8,
                    "away_score": 7,
                    "source_name": "source",
                    "source_url": "https://example.com",
                },
                {
                    "game_date": "2026-04-12",
                    "game_time": "7:00 PM",
                    "season": 2026,
                    "division": "14U Boys",
                    "home_team": "Granby",
                    "away_team": "Simsbury",
                    "home_score": None,
                    "away_score": None,
                    "source_name": "source",
                    "source_url": "https://example.com",
                },
            ]
        )
    )

    team_games = build_team_games(games)

    assert len(team_games) == 2
    assert team_games["game_id"].nunique() == 1
    assert set(team_games["team"]) == {"Avon", "Canton"}
    assert set(team_games["opponent"]) == {"Avon", "Canton"}
    assert set(team_games["status"]) == {"completed"}


def test_build_team_games_orients_scores_and_win_loss_by_team() -> None:
    games = build_canonical_games(
        pd.DataFrame(
            [
                {
                    "game_date": "2026-04-10",
                    "game_time": "6:00 PM",
                    "season": 2026,
                    "division": "14U Boys",
                    "home_team": "Avon",
                    "away_team": "Canton",
                    "home_score": 8,
                    "away_score": 7,
                    "source_name": "source",
                    "source_url": "https://example.com",
                }
            ]
        )
    )

    team_games = build_team_games(games).set_index("team")

    assert team_games.loc["Avon", "opponent"] == "Canton"
    assert team_games.loc["Avon", "points_for"] == 8
    assert team_games.loc["Avon", "points_against"] == 7
    assert team_games.loc["Avon", "win"].item() is True

    assert team_games.loc["Canton", "opponent"] == "Avon"
    assert team_games.loc["Canton", "points_for"] == 7
    assert team_games.loc["Canton", "points_against"] == 8
    assert team_games.loc["Canton", "win"].item() is False


def test_build_team_games_preserves_chronological_order() -> None:
    games = build_canonical_games(
        pd.DataFrame(
            [
                {
                    "game_date": "2026-04-12",
                    "game_time": "6:00 PM",
                    "season": 2026,
                    "division": "14U Boys",
                    "home_team": "Granby",
                    "away_team": "Simsbury",
                    "home_score": 3,
                    "away_score": 6,
                    "source_name": "source",
                    "source_url": "https://example.com",
                },
                {
                    "game_date": "2026-04-10",
                    "game_time": "6:00 PM",
                    "season": 2026,
                    "division": "14U Boys",
                    "home_team": "Avon",
                    "away_team": "Canton",
                    "home_score": 8,
                    "away_score": 7,
                    "source_name": "source",
                    "source_url": "https://example.com",
                },
            ]
        )
    )

    team_games = build_team_games(games)

    assert team_games["game_date"].tolist() == [
        "2026-04-10",
        "2026-04-10",
        "2026-04-12",
        "2026-04-12",
    ]
