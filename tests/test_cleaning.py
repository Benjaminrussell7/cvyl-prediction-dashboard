from __future__ import annotations

import pandas as pd

from cvyl_scraper.cleaning import build_canonical_games, split_by_status


def test_build_canonical_games_deduplicates_mirrored_games() -> None:
    raw = pd.DataFrame(
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
                "source_name": "home_page",
                "source_url": "https://example.com/home",
            },
            {
                "game_date": "2026-04-10",
                "game_time": "6:00 PM",
                "season": 2026,
                "division": "14U Boys",
                "home_team": "Canton",
                "away_team": "Avon",
                "home_score": 7,
                "away_score": 8,
                "source_name": "away_page",
                "source_url": "https://example.com/away",
            },
        ]
    )

    games = build_canonical_games(raw)

    assert len(games) == 1
    assert games.iloc[0]["status"] == "completed"


def test_split_by_status_returns_completed_and_scheduled_games() -> None:
    raw = pd.DataFrame(
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
                "game_date": "2026-04-11",
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

    games = build_canonical_games(raw)
    completed, scheduled = split_by_status(games)

    assert len(completed) == 1
    assert len(scheduled) == 1
