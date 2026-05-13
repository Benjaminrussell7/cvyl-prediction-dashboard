from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from cvyl_scraper.model_comparison import (
    build_model_comparison_outputs,
    power_v2_win_probability,
)


def test_power_v2_win_probability_is_between_zero_and_one() -> None:
    assert 0 < power_v2_win_probability(-10) < 0.5
    assert power_v2_win_probability(0) == 0.5
    assert 0.5 < power_v2_win_probability(10) < 1


def test_model_comparison_hybrid_probabilities_are_between_zero_and_one() -> None:
    comparison, _ = build_model_comparison_outputs(
        pd.DataFrame(
            [
                _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
                _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
            ]
        )
    )

    assert comparison["hybrid_win_probability"].between(0, 1).all()


def test_model_comparison_outputs_are_deterministic() -> None:
    rows = [
        _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
        _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
        _game("game-3", "2026-04-14", "Granby", "Canton", 5, 6),
    ]

    first_comparison, first_summary = build_model_comparison_outputs(pd.DataFrame(rows))
    second_comparison, second_summary = build_model_comparison_outputs(
        pd.DataFrame([rows[2], rows[0], rows[1]])
    )

    assert_frame_equal(first_comparison, second_comparison)
    assert_frame_equal(first_summary, second_summary)


def test_model_comparison_has_no_future_leakage() -> None:
    comparison, _ = build_model_comparison_outputs(
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
    assert first["elo_win_probability"] == 0.5
    assert first["power_v2_win_probability"] == 0.5
    assert first["hybrid_win_probability"] == 0.5
    assert second["home_team"] == "Avon"
    assert second["power_v2_win_probability"] > 0.5
    assert second["hybrid_win_probability"] > 0.5


def test_model_comparison_rows_and_summary_are_valid() -> None:
    games = pd.DataFrame(
        [
            _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
            _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
            _game("game-4", "2026-04-16", "Avon", "Granby", None, None, status="scheduled"),
        ]
    )
    comparison, summary = build_model_comparison_outputs(games)
    completed_count = int(
        (
            (games["status"] == "completed")
            & games["home_score"].notna()
            & games["away_score"].notna()
        )
        .sum()
    )

    assert list(comparison.columns) == [
        "game_date",
        "home_team",
        "away_team",
        "actual_winner",
        "elo_predicted_winner",
        "elo_win_probability",
        "power_v2_predicted_winner",
        "power_v2_win_probability",
        "hybrid_predicted_winner",
        "hybrid_win_probability",
        "elo_correct",
        "power_v2_correct",
        "hybrid_correct",
    ]
    assert len(comparison) == completed_count
    assert summary.iloc[0]["total_games"] == completed_count
    assert comparison["elo_win_probability"].between(0, 1).all()
    assert comparison["power_v2_win_probability"].between(0, 1).all()
    assert comparison["hybrid_win_probability"].between(0, 1).all()
    assert comparison[
        [
            "elo_predicted_winner",
            "power_v2_predicted_winner",
            "hybrid_predicted_winner",
        ]
    ].notna().all().all()
    assert 0 <= summary.iloc[0]["elo_brier_score"] <= 1
    assert 0 <= summary.iloc[0]["power_v2_brier_score"] <= 1
    assert 0 <= summary.iloc[0]["hybrid_brier_score"] <= 1
    assert 0 <= summary.iloc[0]["hybrid_accuracy"] <= 1


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
