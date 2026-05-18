from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from cvyl_scraper.model_comparison_power_v4 import build_model_comparison_power_v4_outputs
from cvyl_scraper.model_comparison_v3 import build_model_comparison_v3_outputs


def test_model_comparison_power_v4_outputs_are_deterministic() -> None:
    rows = [
        _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
        _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
        _game("game-3", "2026-04-14", "Granby", "Canton", 5, 6),
    ]

    first_comparison, first_summary, first_calibration = build_model_comparison_power_v4_outputs(
        pd.DataFrame(rows)
    )
    second_comparison, second_summary, second_calibration = build_model_comparison_power_v4_outputs(
        pd.DataFrame([rows[2], rows[0], rows[1]])
    )

    assert_frame_equal(first_comparison, second_comparison)
    assert_frame_equal(first_summary, second_summary)
    assert_frame_equal(first_calibration, second_calibration)


def test_model_comparison_power_v4_has_no_future_leakage() -> None:
    comparison, _, _ = build_model_comparison_power_v4_outputs(
        pd.DataFrame(
            [
                _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
                _game("game-3", "2026-04-14", "Avon", "Canton", 1, 12),
                _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
            ]
        )
    )

    first = comparison.iloc[0]
    second = comparison.iloc[1]
    assert first["home_team"] == "Avon"
    assert first["away_team"] == "Canton"
    assert first["power_v4_win_probability"] == 0.5
    assert second["home_team"] == "Avon"
    assert second["power_v4_win_probability"] != 0.5


def test_model_comparison_power_v4_rows_summary_and_calibration_are_valid() -> None:
    comparison, summary, calibration = build_model_comparison_power_v4_outputs(
        pd.DataFrame(
            [
                _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
                _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
                _game("game-4", "2026-04-16", "Avon", "Granby", None, None, status="scheduled"),
            ]
        )
    )

    assert list(comparison.columns) == [
        "game_date",
        "home_team",
        "away_team",
        "actual_winner",
        "power_v3_recency_predicted_winner",
        "power_v3_recency_win_probability",
        "power_v4_predicted_winner",
        "power_v4_win_probability",
        "power_v3_recency_correct",
        "power_v4_correct",
        "power_v3_recency_predicted_margin",
        "power_v4_predicted_margin",
        "actual_margin",
        "predicted_total_goals",
        "actual_total_goals",
    ]
    assert summary.iloc[0]["total_games"] == 2
    assert comparison["power_v4_win_probability"].between(0, 1).all()
    assert 0 <= summary.iloc[0]["power_v4_brier_score"] <= 1
    assert 0 <= summary.iloc[0]["power_v4_log_loss"]
    assert list(calibration["bucket"]) == ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]


def test_model_comparison_power_v4_preserves_existing_v3_outputs() -> None:
    games = pd.DataFrame(
        [
            _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
            _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
            _game("game-3", "2026-04-14", "Granby", "Canton", 5, 6),
        ]
    )
    before_comparison, before_summary = build_model_comparison_v3_outputs(games)

    build_model_comparison_power_v4_outputs(games)
    after_comparison, after_summary = build_model_comparison_v3_outputs(games)

    assert_frame_equal(before_comparison, after_comparison)
    assert_frame_equal(before_summary, after_summary)


def _game(
    game_id: str,
    game_date: str,
    home_team: str,
    away_team: str,
    home_score: int | None,
    away_score: int | None,
    *,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "game_date": game_date,
        "game_time": None,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "status": status,
    }
