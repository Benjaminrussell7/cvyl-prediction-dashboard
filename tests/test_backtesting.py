from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from cvyl_scraper.backtesting import build_backtest_outputs


def test_build_backtest_outputs_uses_only_prior_games_for_pregame_elos() -> None:
    games = pd.DataFrame(
        [
            _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
            _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
        ]
    )

    backtest, _ = build_backtest_outputs(games, k_factor=20)

    first = backtest.iloc[0]
    second = backtest.iloc[1]
    assert first["game_id"] == "game-1"
    assert first["pregame_home_elo"] == 1500
    assert first["pregame_away_elo"] == 1500
    assert second["game_id"] == "game-2"
    assert second["pregame_home_elo"] > 1500
    assert second["pregame_away_elo"] == 1500


def test_build_backtest_outputs_processes_predictions_chronologically() -> None:
    games = pd.DataFrame(
        [
            _game("game-3", "2026-04-14", "Granby", "Canton", 5, 6),
            _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
            _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
            _game("game-4", "2026-04-15", "Avon", "Granby", None, None, status="scheduled"),
        ]
    )

    backtest, summary = build_backtest_outputs(games)

    assert backtest["game_id"].tolist() == ["game-1", "game-2", "game-3"]
    assert summary.iloc[0]["total_predictions"] == 3


def test_build_backtest_outputs_are_deterministic() -> None:
    rows = [
        _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
        _game("game-2", "2026-04-12", "Avon", "Granby", 8, 4),
        _game("game-3", "2026-04-14", "Granby", "Canton", 5, 6),
    ]
    ordered = pd.DataFrame(rows)
    shuffled = pd.DataFrame([rows[2], rows[0], rows[1]])

    first_backtest, first_summary = build_backtest_outputs(ordered)
    second_backtest, second_summary = build_backtest_outputs(shuffled)

    assert_frame_equal(first_backtest, second_backtest)
    assert_frame_equal(first_summary, second_summary)


def test_build_backtest_summary_metrics_are_present() -> None:
    backtest, summary = build_backtest_outputs(
        pd.DataFrame(
            [
                _game("game-1", "2026-04-10", "Avon", "Canton", 10, 5),
                _game("game-2", "2026-04-12", "Granby", "Avon", 4, 8),
            ]
        )
    )

    assert list(backtest.columns) == [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "predicted_winner",
        "actual_winner",
        "predicted_win_probability",
        "prediction_correct",
        "pregame_home_elo",
        "pregame_away_elo",
    ]
    assert list(summary.columns) == [
        "total_predictions",
        "accuracy",
        "average_confidence",
        "brier_score",
    ]
    assert 0 <= summary.iloc[0]["accuracy"] <= 1
    assert 0 <= summary.iloc[0]["average_confidence"] <= 1
    assert 0 <= summary.iloc[0]["brier_score"] <= 1


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
