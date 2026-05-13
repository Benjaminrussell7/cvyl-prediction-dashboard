from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from cvyl_scraper.trends import TREND_COLUMNS, build_trends


def test_build_trends_outputs_are_deterministic() -> None:
    rows = _trend_rows()

    first = build_trends(pd.DataFrame(rows))
    second = build_trends(pd.DataFrame([rows[3], rows[0], rows[5], rows[1], rows[4], rows[2]]))

    assert_frame_equal(first, second)


def test_build_trends_uses_recent_games_for_last_three_and_five() -> None:
    trends = build_trends(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 1, 10, "2026-04-01", "game-1"),
                _team_game("Avon", "Canton", 2, 10, "2026-04-02", "game-2"),
                _team_game("Avon", "Canton", 10, 4, "2026-04-03", "game-3"),
                _team_game("Avon", "Canton", 11, 4, "2026-04-04", "game-4"),
                _team_game("Avon", "Canton", 12, 4, "2026-04-05", "game-5"),
                _team_game("Avon", "Canton", 13, 4, "2026-04-06", "game-6"),
            ]
        )
    ).set_index("team")

    assert trends.loc["Avon", "last_3_win_pct"] == 1.0
    assert trends.loc["Avon", "last_5_win_pct"] == 0.8
    assert trends.loc["Avon", "recent_avg_margin"] == ((-8) + 6 + 7 + 8 + 9) / 5


def test_build_trends_uses_completed_games_only() -> None:
    trends = build_trends(
        pd.DataFrame(
            [
                _team_game("Avon", "Canton", 10, 4, "2026-04-01", "game-1"),
                _team_game("Avon", "Canton", 1, 20, "2026-04-02", "game-2", status="scheduled"),
            ]
        )
    )

    avon = trends[trends["team"] == "Avon"].iloc[0]
    assert avon["games_played"] == 1
    assert avon["recent_avg_margin"] == 6


def test_build_trends_columns_are_valid() -> None:
    trends = build_trends(pd.DataFrame(_trend_rows()))

    assert list(trends.columns) == TREND_COLUMNS
    assert trends["momentum_score"].notna().all()
    assert trends["momentum_label"].isin(["Surging", "Improving", "Steady", "Cooling"]).all()


def _trend_rows() -> list[dict[str, object]]:
    return [
        _team_game("Avon", "Canton", 10, 4, "2026-04-01", "game-1"),
        _team_game("Canton", "Avon", 4, 10, "2026-04-01", "game-1"),
        _team_game("Avon", "Granby", 8, 5, "2026-04-08", "game-2"),
        _team_game("Granby", "Avon", 5, 8, "2026-04-08", "game-2"),
        _team_game("Canton", "Granby", 6, 5, "2026-04-15", "game-3"),
        _team_game("Granby", "Canton", 5, 6, "2026-04-15", "game-3"),
    ]


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
        "win": points_for > points_against,
        "game_date": game_date,
        "status": status,
    }
