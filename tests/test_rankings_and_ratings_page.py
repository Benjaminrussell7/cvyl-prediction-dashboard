from __future__ import annotations

import importlib.util
from pathlib import Path

import altair as alt
import pandas as pd

PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "2_Rankings_and_Ratings.py"
SPEC = importlib.util.spec_from_file_location("rankings_and_ratings_page", PAGE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
rankings_page = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rankings_page)


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"snapshot_date": "2026-04-01", "snapshot_label": "Week 1", "team": "Avon", "power_rank": 5, "power_rating": 0.5},
            {"snapshot_date": "2026-04-01", "snapshot_label": "Week 1", "team": "RHAM", "power_rank": 2, "power_rating": 1.4},
            {"snapshot_date": "2026-04-01", "snapshot_label": "Week 1", "team": "Granby", "power_rank": 7, "power_rating": -0.2},
            {"snapshot_date": "2026-04-08", "snapshot_label": "Week 2", "team": "Avon", "power_rank": 3, "power_rating": 1.2},
            {"snapshot_date": "2026-04-08", "snapshot_label": "Week 2", "team": "RHAM", "power_rank": 4, "power_rating": 0.9},
            {"snapshot_date": "2026-04-08", "snapshot_label": "Week 2", "team": "Granby", "power_rank": 6, "power_rating": 0.0},
            {"snapshot_date": "2026-04-15", "snapshot_label": "Week 3", "team": "Avon", "power_rank": 1, "power_rating": 2.0},
            {"snapshot_date": "2026-04-15", "snapshot_label": "Week 3", "team": "RHAM", "power_rank": 6, "power_rating": 0.1},
            {"snapshot_date": "2026-04-15", "snapshot_label": "Week 3", "team": "Granby", "power_rank": 4, "power_rating": 0.8},
        ]
    )


def test_historical_snapshot_display_data_normalizes_columns() -> None:
    history = rankings_page.historical_snapshot_display_data(_snapshots())

    assert not history.empty
    assert pd.api.types.is_datetime64_any_dtype(history["snapshot_date"])
    assert history["power_rank"].tolist()[0] == 5


def test_historical_snapshot_frame_has_required_columns() -> None:
    assert rankings_page.historical_snapshot_frame_has_required_columns(_snapshots())
    assert not rankings_page.historical_snapshot_frame_has_required_columns(pd.DataFrame())
    assert not rankings_page.historical_snapshot_frame_has_required_columns(
        pd.DataFrame([{"snapshot_date": "2026-04-01", "team": "Avon"}])
    )


def test_load_historical_snapshots_uses_existing_dashboard_data(monkeypatch) -> None:
    snapshots = _snapshots()

    def fail_load_csv(filename: str) -> pd.DataFrame:
        raise AssertionError(f"unexpected fallback load for {filename}")

    monkeypatch.setattr(rankings_page.dashboard, "load_csv", fail_load_csv)

    loaded = rankings_page.load_historical_snapshots({"historical_snapshots": snapshots})

    pd.testing.assert_frame_equal(loaded, snapshots)


def test_load_historical_snapshots_falls_back_to_csv_loader(monkeypatch) -> None:
    snapshots = _snapshots()

    monkeypatch.setattr(rankings_page.dashboard, "load_csv", lambda filename: snapshots)

    loaded = rankings_page.load_historical_snapshots({})

    pd.testing.assert_frame_equal(loaded, snapshots)


def test_latest_rank_movement_identifies_risers_and_fallers() -> None:
    history = rankings_page.historical_snapshot_display_data(_snapshots())

    movement = rankings_page.latest_rank_movement(history)

    assert movement.iloc[0]["team"] == "Avon"
    assert movement.iloc[0]["rank_move"] == 2
    assert movement.iloc[-1]["team"] == "RHAM"
    assert movement.iloc[-1]["rank_move"] == -2


def test_multi_snapshot_rank_movement_tracks_three_snapshot_climbers_and_coolers() -> None:
    history = rankings_page.historical_snapshot_display_data(_snapshots())

    climbing = rankings_page.multi_snapshot_rank_movement(history, snapshots_back=3, direction="up")
    cooling = rankings_page.multi_snapshot_rank_movement(history, snapshots_back=3, direction="down")

    assert climbing.iloc[0]["team"] == "Avon"
    assert climbing.iloc[0]["rank_move"] == 4
    assert cooling.iloc[0]["team"] == "RHAM"
    assert cooling.iloc[0]["rank_move"] == -4


def test_top_five_streak_display_counts_current_top_tier_streak() -> None:
    history = rankings_page.historical_snapshot_display_data(_snapshots())

    streaks = rankings_page.top_five_streak_display(history)

    assert streaks.set_index("team").loc["Avon", "top_5_streak"] == 3
    assert "RHAM" not in set(streaks["team"])


def test_format_rank_move_uses_sports_friendly_language() -> None:
    assert rankings_page.format_rank_move(3) == "Up 3"
    assert rankings_page.format_rank_move(-2) == "Down 2"
    assert rankings_page.format_rank_move(0) == "No movement"


def test_historical_trend_charts_compile() -> None:
    history = rankings_page.historical_snapshot_display_data(_snapshots())
    movement = rankings_page.latest_rank_movement(history)
    streaks = rankings_page.top_five_streak_display(history)

    assert isinstance(rankings_page.rank_movement_chart(movement), alt.Chart)
    assert isinstance(rankings_page.top_five_streak_chart(streaks), alt.Chart)
    assert isinstance(rankings_page.rank_trajectory_chart(history, ["Avon", "RHAM"]), alt.Chart)
