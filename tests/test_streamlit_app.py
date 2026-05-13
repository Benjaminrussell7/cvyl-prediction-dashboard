from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
SPEC = importlib.util.spec_from_file_location("streamlit_app", APP_PATH)
assert SPEC is not None
assert SPEC.loader is not None
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def test_build_matchup_prediction_passes_current_prediction_arguments(monkeypatch) -> None:
    ratings = pd.DataFrame(
        [
            {"team": "Avon 12U", "elo": 1525.0, "games_played": 6},
            {"team": "Granby 12U", "elo": 1490.0, "games_played": 6},
        ]
    )
    team_games = pd.DataFrame(
        [
            {"team": "Avon 12U", "points_for": 8, "points_against": 5},
            {"team": "Granby 12U", "points_for": 6, "points_against": 8},
        ]
    )
    sos = pd.DataFrame(
        [
            {"team": "Avon 12U", "average_opponent_elo": 1510.0, "sos_rank": 2},
            {"team": "Granby 12U", "average_opponent_elo": 1490.0, "sos_rank": 4},
        ]
    )
    power_v2 = pd.DataFrame(
        [
            {"team": "Avon 12U", "power_rating_v2": 2.5, "power_rank_v2": 2},
            {"team": "Granby 12U", "power_rating_v2": 1.0, "power_rank_v2": 5},
        ]
    )
    captured = {}
    expected_prediction = object()

    def fake_predict_matchup(
        team_a,
        team_b,
        ratings,
        team_games,
        sos=None,
        power_v2=None,
        /,
    ):
        captured.update(
            {
                "team_a": team_a,
                "team_b": team_b,
                "ratings": ratings,
                "team_games": team_games,
                "sos": sos,
                "power_v2": power_v2,
            }
        )
        return expected_prediction

    monkeypatch.setattr(dashboard, "predict_matchup", fake_predict_matchup)

    prediction = dashboard.build_matchup_prediction(
        "Avon 12U",
        "Granby 12U",
        ratings,
        team_games,
        sos,
        power_v2,
    )

    assert prediction is expected_prediction
    assert captured["team_a"] == "Avon 12U"
    assert captured["team_b"] == "Granby 12U"
    assert captured["ratings"] is ratings
    assert captured["team_games"] is team_games
    assert captured["sos"] is sos
    assert captured["power_v2"] is power_v2


def test_build_matchup_prediction_converts_empty_optional_frames_to_none(monkeypatch) -> None:
    captured = {}

    def fake_predict_matchup(
        team_a,
        team_b,
        ratings,
        team_games,
        sos=None,
        power_v2=None,
        /,
    ):
        captured["sos"] = sos
        captured["power_v2"] = power_v2
        return object()

    monkeypatch.setattr(dashboard, "predict_matchup", fake_predict_matchup)

    dashboard.build_matchup_prediction(
        "Avon 12U",
        "Granby 12U",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert captured["sos"] is None
    assert captured["power_v2"] is None
