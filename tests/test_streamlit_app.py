from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SPEC = importlib.util.spec_from_file_location("streamlit_app", APP_PATH)
assert SPEC is not None
assert SPEC.loader is not None
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


REQUIRED_CSV_COLUMNS = {
    "cvyl_games.csv": {
        "game_date",
        "season",
        "division",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "status",
        "game_id",
    },
    "cvyl_team_games.csv": {
        "game_id",
        "team",
        "opponent",
        "points_for",
        "points_against",
        "win",
        "game_date",
        "season",
        "division",
        "status",
    },
    "cvyl_scheduled_games.csv": {
        "game_date",
        "game_time",
        "season",
        "division",
        "home_team",
        "away_team",
        "status",
        "game_id",
    },
    "cvyl_elo_ratings.csv": {
        "team",
        "elo",
        "games_played",
    },
    "cvyl_sos.csv": {
        "team",
        "games_played",
        "average_opponent_elo",
        "opponent_count",
        "sos_rank",
    },
    "cvyl_trends.csv": {
        "team",
        "games_played",
        "last_3_win_pct",
        "last_5_win_pct",
        "recent_avg_margin",
        "recent_offense_rating",
        "recent_defense_rating",
        "current_power_rank",
        "prior_power_rank",
        "power_rank_movement",
        "momentum_score",
        "momentum_label",
    },
    "cvyl_power_ratings_v2.csv": {
        "team",
        "games_played",
        "avg_points_for",
        "avg_points_against",
        "avg_margin",
        "adjusted_offense_rating",
        "adjusted_defense_rating",
        "adjusted_margin_rating",
        "power_rating_v2",
        "power_rank_v2",
        "confidence_tier",
        "shrinkage_multiplier",
    },
    "cvyl_power_ratings_v3_recency.csv": {
        "team",
        "games_played",
        "avg_points_for",
        "avg_points_against",
        "avg_margin",
        "adjusted_offense_rating",
        "adjusted_defense_rating",
        "adjusted_margin_rating",
        "power_rating_v3_recency",
        "power_rank_v3_recency",
        "confidence_tier",
        "shrinkage_multiplier",
        "average_recency_weight",
    },
    "cvyl_model_comparison_v3_summary.csv": {
        "total_games",
        "power_v3_recency_accuracy",
        "power_v3_recency_brier_score",
        "power_v2_accuracy",
        "power_v2_brier_score",
        "elo_accuracy",
        "elo_brier_score",
    },
    "cvyl_model_comparison_v4_calibrated_summary.csv": {
        "total_games",
        "power_v3_recency_accuracy",
        "power_v3_calibrated_accuracy",
        "power_v3_recency_brier_score",
        "power_v3_calibrated_brier_score",
    },
    "cvyl_model_comparison_v4_calibrated.csv": {
        "game_date",
        "home_team",
        "away_team",
        "actual_winner",
        "power_v3_recency_predicted_winner",
        "power_v3_recency_win_probability",
        "power_v3_calibrated_predicted_winner",
        "power_v3_calibrated_win_probability",
        "power_v3_recency_correct",
        "power_v3_calibrated_correct",
    },
    "cvyl_model_comparison_summary.csv": {
        "total_games",
        "power_v2_accuracy",
        "power_v2_brier_score",
        "elo_accuracy",
        "elo_brier_score",
    },
    "cvyl_calibration_power_rating.csv": {
        "bucket",
        "games",
        "average_predicted_probability",
        "actual_win_rate",
        "calibration_gap",
    },
    "cvyl_calibration_power_rating_v4.csv": {
        "bucket",
        "games",
        "average_predicted_probability",
        "actual_win_rate",
        "calibration_gap",
    },
}


def test_dashboard_required_processed_csv_contracts() -> None:
    for filename, required_columns in REQUIRED_CSV_COLUMNS.items():
        path = PROCESSED / filename
        assert path.exists(), f"Missing dashboard CSV: {path}"
        frame = pd.read_csv(path)
        missing_columns = required_columns - set(frame.columns)
        assert not missing_columns, f"{filename} missing columns: {sorted(missing_columns)}"
        assert not frame.empty, f"{filename} should not be empty"


def test_dashboard_processed_data_smoke_contract() -> None:
    ratings = dashboard.load_csv("cvyl_elo_ratings.csv")
    team_games = dashboard.load_csv("cvyl_team_games.csv")
    sos = dashboard.load_csv("cvyl_sos.csv")
    power_ratings = dashboard.load_csv(dashboard.PRIMARY_POWER_RATINGS_FILE)
    model_summary = dashboard.load_csv(dashboard.PRIMARY_MODEL_COMPARISON_SUMMARY_FILE)
    scheduled_games = dashboard.load_csv("cvyl_scheduled_games.csv")

    prediction = dashboard.build_matchup_prediction(
        "West Hartford 12U Green",
        "RHAM 12U",
        ratings,
        team_games,
        sos,
        power_ratings,
    )
    power_row = dashboard.find_team_row(power_ratings, "Avon 12U B")
    power_context = dashboard.matchup_power_context(
        "West Hartford 12U Green",
        "RHAM 12U",
        power_ratings,
    )

    assert prediction.team_a == "West Hartford 12U Green"
    assert prediction.team_b == "RHAM 12U"
    assert power_context["predicted_winner"] in {"West Hartford 12U Green", "RHAM 12U"}
    assert power_row is not None
    assert pd.notna(power_row["power_rank_v3_recency"])
    assert pd.notna(power_row["power_rating_v3_recency"])
    assert pd.notna(power_row["average_recency_weight"])
    assert dashboard.metric_value(model_summary, dashboard.POWER_ACCURACY_KEY, percentage=True) != "N/A"
    assert dashboard.metric_value(model_summary, dashboard.POWER_BRIER_KEY) != "N/A"
    assert pd.notna(model_summary.iloc[0]["power_v3_calibrated_accuracy"])
    assert pd.notna(model_summary.iloc[0]["power_v3_calibrated_brier_score"])

    weekly_matchups = dashboard.build_weekly_matchups(
        scheduled_games,
        ratings,
        team_games,
        sos,
        power_ratings,
        today="2026-05-13",
    )

    assert not weekly_matchups.empty
    assert {
        "Date",
        "Time",
        "Home",
        "Away",
        "Projected Winner",
        "Win Probability",
        "Edge",
        "Projected Spread",
        "Projected Total",
        "Confidence",
        "Note",
    }.issubset(weekly_matchups.columns)
    assert weekly_matchups["Date"].min() >= "2026-05-13"
    assert weekly_matchups["Date"].max() <= "2026-05-20"
    assert weekly_matchups["Projected Winner"].ne("").any()
    assert weekly_matchups["Edge"].isin(
        ["Toss-up", "Slight Edge", "Solid Favorite", "Strong Favorite", "Unavailable"]
    ).all()


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
    power_ratings = pd.DataFrame(
        [
            {"team": "Avon 12U", "power_rating_v3_recency": 2.5, "power_rank_v3_recency": 2},
            {"team": "Granby 12U", "power_rating_v3_recency": 1.0, "power_rank_v3_recency": 5},
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
        /,
    ):
        captured.update(
            {
                "team_a": team_a,
                "team_b": team_b,
                "ratings": ratings,
                "team_games": team_games,
                "sos": sos,
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
        power_ratings,
    )

    assert prediction is expected_prediction
    assert captured["team_a"] == "Avon 12U"
    assert captured["team_b"] == "Granby 12U"
    assert captured["ratings"] is ratings
    assert captured["team_games"] is team_games
    assert captured["sos"] is sos


def test_build_matchup_prediction_converts_empty_optional_frames_to_none(monkeypatch) -> None:
    captured = {}

    def fake_predict_matchup(
        team_a,
        team_b,
        ratings,
        team_games,
        sos=None,
        /,
    ):
        captured["sos"] = sos
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


def test_find_team_row_normalizes_power_rating_team_names() -> None:
    power_ratings = pd.DataFrame(
        [
            {
                "team": "Avon 12U B",
                "power_rank_v3_recency": 20,
                "power_rating_v3_recency": 0.50,
                "confidence_tier": "medium",
            }
        ]
    )

    row = dashboard.find_team_row(power_ratings, "  avon   12u b  ")

    assert row is not None
    assert int(row["power_rank_v3_recency"]) == 20


def test_matchup_power_context_uses_calibrated_probability() -> None:
    power_ratings = pd.DataFrame(
        [
            {"team": "Avon 12U", "power_rating_v3_recency": 2.0, "power_rank_v3_recency": 1},
            {"team": "Granby 12U", "power_rating_v3_recency": 0.0, "power_rank_v3_recency": 2},
        ]
    )

    context = dashboard.matchup_power_context("Avon 12U", "Granby 12U", power_ratings)

    assert context["team_a_probability"] > 0.62
    assert context["predicted_winner"] == "Avon 12U"


def test_filter_matchups_by_team_matches_home_or_away() -> None:
    matchups = pd.DataFrame(
        [
            {"Home": "West Hartford 12U Gold", "Away": "Avon 12U", "Projected Winner": "Avon 12U"},
            {"Home": "RHAM 12U", "Away": "Granby 12U", "Projected Winner": "RHAM 12U"},
            {"Home": "Canton 12U", "Away": "West Hartford 12U Green", "Projected Winner": "Canton 12U"},
        ]
    )

    filtered = dashboard.filter_matchups_by_team(matchups, "West Hartford")

    assert len(filtered) == 2
    assert set(filtered["Home"]) == {"West Hartford 12U Gold", "Canton 12U"}


def test_upcoming_scheduled_games_for_team_excludes_past_scheduled_games() -> None:
    games = pd.DataFrame(
        [
            {
                "game_date": "2026-05-12",
                "game_time": "6:00 PM",
                "home_team": "Avon 12U",
                "away_team": "Canton 12U",
                "status": "scheduled",
            },
            {
                "game_date": "2026-05-13",
                "game_time": "6:00 PM",
                "home_team": "Granby 12U",
                "away_team": "Avon 12U",
                "status": "scheduled",
            },
            {
                "game_date": "2026-05-20",
                "game_time": "6:00 PM",
                "home_team": "Avon 12U",
                "away_team": "RHAM 12U",
                "status": "scheduled",
            },
            {
                "game_date": "2026-05-21",
                "game_time": "6:00 PM",
                "home_team": "Avon 12U",
                "away_team": "Berlin 12U",
                "status": "completed",
            },
        ]
    )

    upcoming = dashboard.upcoming_scheduled_games_for_team(games, "Avon 12U", today="2026-05-13")

    assert list(upcoming["game_date"].dt.strftime("%Y-%m-%d")) == ["2026-05-13", "2026-05-20"]
    assert set(upcoming["status"]) == {"scheduled"}


def test_format_eastern_timestamp_labels_timezone_for_aware_and_naive_values() -> None:
    aware = dashboard.format_eastern_timestamp(datetime(2026, 5, 13, 15, 30, tzinfo=UTC))
    naive = dashboard.format_eastern_timestamp(datetime(2026, 5, 13, 15, 30))

    assert aware == "2026-05-13 11:30 AM ET"
    assert naive.endswith(" ET")


def test_prediction_edge_label_boundaries() -> None:
    assert dashboard.prediction_edge_label(0.50) == "Toss-up"
    assert dashboard.prediction_edge_label(0.549) == "Toss-up"
    assert dashboard.prediction_edge_label(0.55) == "Slight Edge"
    assert dashboard.prediction_edge_label(0.649) == "Slight Edge"
    assert dashboard.prediction_edge_label(0.65) == "Solid Favorite"
    assert dashboard.prediction_edge_label(0.749) == "Solid Favorite"
    assert dashboard.prediction_edge_label(0.75) == "Strong Favorite"
    assert dashboard.prediction_edge_label(0.25) == "Strong Favorite"
    assert dashboard.prediction_edge_label(None) == "Unavailable"
