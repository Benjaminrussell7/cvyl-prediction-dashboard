from __future__ import annotations

import pandas as pd

from cvyl_scraper.historical_snapshots import (
    HISTORICAL_SNAPSHOT_COLUMNS,
    build_historical_snapshots,
    build_historical_storylines,
    longest_top_five_streaks,
    weekly_snapshot_dates,
)


def _game_rows(
    game_id: str,
    game_date: str,
    team_a: str,
    team_b: str,
    score_a: int,
    score_b: int,
) -> list[dict[str, object]]:
    return [
        {
            "game_id": game_id,
            "team": team_a,
            "opponent": team_b,
            "points_for": score_a,
            "points_against": score_b,
            "win": score_a > score_b,
            "game_date": game_date,
            "season": 2026,
            "division": "U12 Boys",
            "status": "completed",
        },
        {
            "game_id": game_id,
            "team": team_b,
            "opponent": team_a,
            "points_for": score_b,
            "points_against": score_a,
            "win": score_b > score_a,
            "game_date": game_date,
            "season": 2026,
            "division": "U12 Boys",
            "status": "completed",
        },
    ]


def _sample_team_games() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_game_rows("g1", "2026-04-01", "Avon", "Granby", 8, 4))
    rows.extend(_game_rows("g2", "2026-04-02", "RHAM", "Simsbury", 5, 3))
    rows.extend(_game_rows("g3", "2026-04-10", "Avon", "RHAM", 7, 6))
    rows.extend(_game_rows("g4", "2026-04-17", "Simsbury", "Granby", 6, 2))
    rows.append(
        {
            "game_id": "g5",
            "team": "Avon",
            "opponent": "Simsbury",
            "points_for": None,
            "points_against": None,
            "win": False,
            "game_date": "2026-04-20",
            "season": 2026,
            "division": "U12 Boys",
            "status": "scheduled",
        }
    )
    return pd.DataFrame(rows)


def test_historical_snapshots_use_weekly_cadence_and_columns() -> None:
    snapshots = build_historical_snapshots(_sample_team_games())

    assert list(snapshots.columns) == HISTORICAL_SNAPSHOT_COLUMNS
    assert snapshots["snapshot_label"].drop_duplicates().tolist() == ["Week 1", "Week 2", "Week 3"]
    assert snapshots["snapshot_date"].drop_duplicates().tolist() == [
        "2026-04-07",
        "2026-04-14",
        "2026-04-17",
    ]


def test_historical_snapshots_are_deterministic() -> None:
    rows = _sample_team_games()
    first = build_historical_snapshots(rows)
    second = build_historical_snapshots(rows.sample(frac=1.0, random_state=42))

    pd.testing.assert_frame_equal(first, second)


def test_historical_snapshots_do_not_include_future_games_in_prior_weeks() -> None:
    snapshots = build_historical_snapshots(_sample_team_games())

    week_1_avon = snapshots[
        (snapshots["snapshot_label"] == "Week 1") & (snapshots["team"] == "Avon")
    ].iloc[0]
    week_2_avon = snapshots[
        (snapshots["snapshot_label"] == "Week 2") & (snapshots["team"] == "Avon")
    ].iloc[0]

    assert week_1_avon["games_played"] == 1
    assert week_1_avon["wins"] == 1
    assert week_1_avon["losses"] == 0
    assert week_2_avon["games_played"] == 2
    assert week_2_avon["wins"] == 2
    assert week_2_avon["losses"] == 0


def test_weekly_snapshot_dates_avoid_daily_noise() -> None:
    dates = weekly_snapshot_dates(_sample_team_games())

    assert [date.date().isoformat() for date in dates] == [
        "2026-04-07",
        "2026-04-14",
        "2026-04-17",
    ]


def test_longest_top_five_streaks_counts_current_streak() -> None:
    snapshots = pd.DataFrame(
        [
            {"snapshot_date": "2026-04-01", "team": "Avon", "power_rank": 3},
            {"snapshot_date": "2026-04-08", "team": "Avon", "power_rank": 4},
            {"snapshot_date": "2026-04-15", "team": "Avon", "power_rank": 7},
            {"snapshot_date": "2026-04-22", "team": "Avon", "power_rank": 2},
            {"snapshot_date": "2026-04-01", "team": "RHAM", "power_rank": 1},
            {"snapshot_date": "2026-04-08", "team": "RHAM", "power_rank": 2},
            {"snapshot_date": "2026-04-15", "team": "RHAM", "power_rank": 4},
            {"snapshot_date": "2026-04-22", "team": "RHAM", "power_rank": 5},
        ]
    )

    streaks = longest_top_five_streaks(snapshots)

    assert streaks.set_index("team").loc["RHAM", "top_5_streak"] == 4
    assert streaks.set_index("team").loc["Avon", "top_5_streak"] == 1


def test_historical_storylines_are_deterministic() -> None:
    snapshots = build_historical_snapshots(_sample_team_games())

    first = build_historical_storylines(snapshots)
    second = build_historical_storylines(snapshots)

    pd.testing.assert_frame_equal(first, second)
    assert {"biggest_riser", "biggest_faller", "longest_top_5_streak"}.intersection(
        set(first["storyline_type"])
    )
