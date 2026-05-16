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
    "cvyl_historical_snapshots.csv": {
        "snapshot_date",
        "snapshot_label",
        "team",
        "power_rank",
        "power_rating",
        "elo",
        "offense_strength",
        "defense_strength",
        "sos_rank",
        "momentum_score",
        "last3_win_pct",
        "last5_win_pct",
        "wins",
        "losses",
        "games_played",
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
        dashboard.load_csv("cvyl_trends.csv"),
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
        "Explanation",
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


def test_dashboard_badge_helpers_render_expected_labels() -> None:
    assert "Strong Favorite" in dashboard.edge_badge("Strong Favorite")
    assert "#166534" in dashboard.edge_badge("Strong Favorite")
    assert "Confidence: High" in dashboard.confidence_badge("high")
    assert "Confidence: Unavailable" in dashboard.confidence_badge(None)


def test_momentum_indicator_adds_direction_when_rank_moves() -> None:
    assert dashboard.momentum_indicator("Surging", 2) == "Surging ↑"
    assert dashboard.momentum_indicator("Cooling", -1) == "Cooling ↓"
    assert dashboard.momentum_indicator("Steady", 0) == "Steady"


def test_rank_movement_formatting_is_clear() -> None:
    assert dashboard.format_rank_movement(3) == "↑ +3"
    assert dashboard.format_rank_movement(-2) == "↓ -2"
    assert dashboard.format_rank_movement(0) == "→ 0"


def test_ordered_bar_chart_preserves_input_category_order() -> None:
    frame = pd.DataFrame(
        [
            {"team": "A", "score": 3.0},
            {"team": "B", "score": 2.0},
            {"team": "C", "score": 1.0},
        ]
    )

    chart_spec = dashboard.ordered_bar_chart(
        frame,
        x_column="team",
        y_column="score",
        y_title="Score",
    ).to_dict()

    assert chart_spec["encoding"]["x"]["sort"] == ["A", "B", "C"]


def test_matchup_probability_data_uses_current_power_context() -> None:
    probabilities = dashboard.matchup_probability_data(
        "Avon",
        "Granby",
        {
            "team_a_probability": 0.62,
            "team_b_probability": 0.38,
        },
    )

    assert list(probabilities["Team"]) == ["Avon", "Granby"]
    assert list(probabilities["Probability"]) == [0.62, 0.38]


def test_matchup_team_colors_are_consistent_across_charts() -> None:
    colors = dashboard.matchup_team_colors("Avon", "Granby")
    probabilities = pd.DataFrame(
        [
            {"Team": "Avon", "Probability": 0.62},
            {"Team": "Granby", "Probability": 0.38},
        ]
    )
    scores = pd.DataFrame(
        [
            {"Team": "Avon", "Projected Goals": 6.1},
            {"Team": "Granby", "Projected Goals": 4.8},
        ]
    )

    probability_spec = dashboard.matchup_probability_chart(probabilities, colors).to_dict()
    score_spec = dashboard.projected_score_chart(scores, colors).to_dict()

    assert probability_spec["encoding"]["color"]["scale"]["domain"] == ["Avon", "Granby"]
    assert score_spec["encoding"]["color"]["scale"]["domain"] == ["Avon", "Granby"]
    assert probability_spec["encoding"]["color"]["scale"]["range"] == score_spec["encoding"]["color"]["scale"]["range"]


def test_branding_registry_loads_processed_csv_directly() -> None:
    registry = dashboard.load_club_branding_registry()

    assert len(registry) >= 30
    assert any(club.club_name == "West Hartford Youth Lacrosse" for club in registry)
    assert any(club.logo_path for club in registry)


def test_team_branding_context_and_matchup_colors_use_shared_club_resolution() -> None:
    context = dashboard.team_branding_context("Glastonbury 12U Blue")
    colors = dashboard.matchup_team_colors("Glastonbury 12U Blue", "West Hartford 12U Green")

    assert context["club_name"] == "Glastonbury Lacrosse Club"
    assert context["logo_source"]
    assert context["branding_applied"] is True
    assert colors["Glastonbury 12U Blue"] != colors["West Hartford 12U Green"]
    assert colors["Glastonbury 12U Blue"] != ""


def test_matchup_strength_data_reads_power_rating_context() -> None:
    power_ratings = pd.DataFrame(
        [
            {
                "team": "Avon",
                "adjusted_offense_rating": 2.5,
                "adjusted_defense_rating": 1.2,
            },
            {
                "team": "Granby",
                "adjusted_offense_rating": 1.5,
                "adjusted_defense_rating": 0.8,
            },
        ]
    )

    strengths = dashboard.matchup_strength_data("Avon", "Granby", power_ratings)

    assert len(strengths) == 4
    assert set(strengths["Metric"]) == {"Offense Strength", "Defense Strength"}
    assert set(strengths["Team"]) == {"Avon", "Granby"}


def test_projected_score_data_labels_each_team() -> None:
    class Prediction:
        projected_team_a_goals = 4.8
        projected_team_b_goals = 5.0

    scores = dashboard.projected_score_data("West Hartford 12U Green", "RHAM 12U", Prediction())

    assert list(scores["Team"]) == ["West Hartford 12U Green", "RHAM 12U"]
    assert list(scores["Projected Goals"]) == [4.8, 5.0]


def test_team_profile_comparison_data_uses_existing_context() -> None:
    power_ratings = pd.DataFrame(
        [
            {
                "team": "Avon",
                "power_rank_v3_recency": 3,
                "power_rating_v3_recency": 2.25,
                "adjusted_offense_rating": 1.5,
                "adjusted_defense_rating": 0.7,
            },
            {
                "team": "Granby",
                "power_rank_v3_recency": 8,
                "power_rating_v3_recency": 0.75,
                "adjusted_offense_rating": 0.5,
                "adjusted_defense_rating": -0.2,
            },
        ]
    )
    trends = pd.DataFrame(
        [
            {"team": "Avon", "momentum_label": "Steady", "power_rank_movement": 2},
            {"team": "Granby", "momentum_label": "Cooling", "power_rank_movement": -1},
        ]
    )
    sos = pd.DataFrame(
        [
            {"team": "Avon", "sos_rank": 4},
            {"team": "Granby", "sos_rank": 10},
        ]
    )

    profile = dashboard.team_profile_comparison_data("Avon", "Granby", power_ratings, trends, sos)

    assert list(profile["Team"]) == ["Avon", "Granby"]
    assert list(profile["Power Rank"]) == ["3", "8"]
    assert list(profile["Recent Form"]) == ["Improving", "Cooling"]
    assert list(profile["Recent Rank Move"]) == ["↑ +2", "↓ -1"]
    assert list(profile["SOS Rank"]) == ["4", "10"]


def test_matchup_summary_card_data_includes_both_teams() -> None:
    class Prediction:
        projected_team_a_goals = 4.8
        projected_team_b_goals = 5.0

    power_ratings = pd.DataFrame(
        [
            {"team": "Avon", "power_rank_v3_recency": 3},
            {"team": "Granby", "power_rank_v3_recency": 8},
        ]
    )
    trends = pd.DataFrame(
        [
            {"team": "Avon", "momentum_label": "Surging", "power_rank_movement": 1},
            {"team": "Granby", "momentum_label": "Steady", "power_rank_movement": 0},
        ]
    )

    cards = dashboard.matchup_summary_card_data(
        "Avon",
        "Granby",
        Prediction(),
        {"team_a_probability": 0.62, "team_b_probability": 0.38},
        power_ratings,
        trends,
    )

    assert [card["Team"] for card in cards] == ["Avon", "Granby"]
    assert cards[0]["Win Probability"] == "62.0%"
    assert cards[0]["Projected Goals"] == "4.8"
    assert cards[0]["Power Rank"] == "3"
    assert cards[0]["Recent Form"] == "Surging"


def test_game_preview_data_is_deterministic_and_fan_friendly() -> None:
    class Prediction:
        confidence_level = "high"
        projected_team_a_goals = 8.2
        projected_team_b_goals = 5.1
        projected_total_goals = 13.3
        projected_margin = 3.1
        projected_spread = "Avon by 3.1"

    power_ratings = pd.DataFrame(
        [
            {
                "team": "Avon",
                "power_rank_v3_recency": 2,
                "adjusted_defense_rating": 4.0,
                "adjusted_offense_rating": 3.0,
            },
            {
                "team": "Granby",
                "power_rank_v3_recency": 9,
                "adjusted_defense_rating": 1.0,
                "adjusted_offense_rating": 2.0,
            },
        ]
    )
    trends = pd.DataFrame(
        [
            {"team": "Avon", "momentum_label": "Surging", "power_rank_movement": 2},
            {"team": "Granby", "momentum_label": "Steady", "power_rank_movement": 0},
        ]
    )
    sos = pd.DataFrame([{"team": "Avon", "sos_rank": 4}, {"team": "Granby", "sos_rank": 15}])
    context = {
        "team_a_probability": 0.71,
        "team_b_probability": 0.29,
        "predicted_winner": "Avon",
    }

    first = dashboard.game_preview_data("Avon", "Granby", Prediction(), context, power_ratings, trends, sos)
    second = dashboard.game_preview_data("Avon", "Granby", Prediction(), context, power_ratings, trends, sos)

    assert first == second
    assert first["favorite"] == "Avon"
    assert first["team_a_probability"] == "71.0%"
    assert first["team_a_score"] == "8.2"
    assert first["fan_outlook"] == "Strong Edge"
    assert first["game_style"] in {
        "Heavyweight Battle",
        "Defense vs Firepower",
        "Momentum Clash",
        "Emerging Contender Matchup",
    }
    assert first["observations"]
    assert len(first["keys"]) == 3
    assert all("Brier" not in key and "probability" not in key.lower() for key in first["keys"])


def test_keys_to_game_handles_missing_context_gracefully() -> None:
    class Prediction:
        projected_total_goals = 8.0
        projected_margin = 1.0

    context = {
        "team_a_probability": 0.52,
        "team_b_probability": 0.48,
        "predicted_winner": "Avon",
    }

    keys = dashboard.keys_to_game(
        "Avon",
        "Granby",
        Prediction(),
        context,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert len(keys) == 3
    assert keys[0].startswith("Which team settles in first?")
    assert any("lower-scoring" in key for key in keys)


def test_form_distribution_data_uses_display_form_states() -> None:
    trends = pd.DataFrame(
        [
            {"momentum_label": "Surging", "power_rank_movement": 1},
            {"momentum_label": "Steady", "power_rank_movement": 1},
            {"momentum_label": "Cooling", "power_rank_movement": 1},
            {"momentum_label": "Cooling", "power_rank_movement": -1},
        ]
    )

    form = dashboard.form_distribution_data(trends)

    counts = dict(zip(form["Form"], form["Teams"], strict=True))
    assert counts["Surging"] == 1
    assert counts["Improving"] == 1
    assert counts["Recovering"] == 1
    assert counts["Cooling"] == 1


def test_trend_cell_style_is_limited_and_readable() -> None:
    assert "#dcfce7" in dashboard.trend_cell_style("Surging")
    assert "#e0f2fe" in dashboard.trend_cell_style("Improving")
    assert "#fef3c7" in dashboard.trend_cell_style("Recovering")
    assert "#f3f4f6" in dashboard.trend_cell_style("Steady")
    assert "#fecaca" in dashboard.trend_cell_style("Cooling")
    assert dashboard.trend_cell_style("75%") == ""
    assert dashboard.trend_cell_style("2.5") == ""


def test_display_form_state_simplifies_mixed_momentum_labels() -> None:
    assert dashboard.display_form_state("Surging", 2) == "Surging"
    assert dashboard.display_form_state("Surging", -1) == "Cooling"
    assert dashboard.display_form_state("Steady", 1) == "Improving"
    assert dashboard.display_form_state("Steady", 0) == "Steady"
    assert dashboard.display_form_state("Cooling", 1) == "Recovering"
    assert dashboard.display_form_state("Cooling", -1) == "Cooling"


def _historical_storyline_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"snapshot_date": "2026-04-01", "snapshot_label": "Week 1", "team": "Avon", "power_rank": 10, "power_rating": -0.2, "offense_strength": 1.0, "defense_strength": 1.0, "momentum_score": 1.0, "games_played": 3},
            {"snapshot_date": "2026-04-01", "snapshot_label": "Week 1", "team": "RHAM", "power_rank": 1, "power_rating": 2.0, "offense_strength": 3.0, "defense_strength": 3.0, "momentum_score": 4.0, "games_played": 3},
            {"snapshot_date": "2026-04-01", "snapshot_label": "Week 1", "team": "Granby", "power_rank": 4, "power_rating": 1.0, "offense_strength": 1.0, "defense_strength": 5.0, "momentum_score": 2.0, "games_played": 3},
            {"snapshot_date": "2026-04-01", "snapshot_label": "Week 1", "team": "Simsbury", "power_rank": 8, "power_rating": 0.4, "offense_strength": 2.0, "defense_strength": 0.5, "momentum_score": 2.0, "games_played": 3},
            {"snapshot_date": "2026-04-08", "snapshot_label": "Week 2", "team": "Avon", "power_rank": 7, "power_rating": 0.6, "offense_strength": 2.0, "defense_strength": 2.0, "momentum_score": 3.0, "games_played": 4},
            {"snapshot_date": "2026-04-08", "snapshot_label": "Week 2", "team": "RHAM", "power_rank": 1, "power_rating": 2.1, "offense_strength": 3.1, "defense_strength": 3.5, "momentum_score": 4.2, "games_played": 4},
            {"snapshot_date": "2026-04-08", "snapshot_label": "Week 2", "team": "Granby", "power_rank": 5, "power_rating": 0.8, "offense_strength": 1.1, "defense_strength": 5.5, "momentum_score": 1.5, "games_played": 4},
            {"snapshot_date": "2026-04-08", "snapshot_label": "Week 2", "team": "Simsbury", "power_rank": 9, "power_rating": 0.2, "offense_strength": 1.7, "defense_strength": 0.4, "momentum_score": 0.5, "games_played": 4},
            {"snapshot_date": "2026-04-15", "snapshot_label": "Week 3", "team": "Avon", "power_rank": 3, "power_rating": 1.8, "offense_strength": 4.0, "defense_strength": 2.5, "momentum_score": 6.0, "games_played": 5},
            {"snapshot_date": "2026-04-15", "snapshot_label": "Week 3", "team": "RHAM", "power_rank": 1, "power_rating": 2.3, "offense_strength": 3.2, "defense_strength": 4.0, "momentum_score": 4.4, "games_played": 5},
            {"snapshot_date": "2026-04-15", "snapshot_label": "Week 3", "team": "Granby", "power_rank": 6, "power_rating": 0.6, "offense_strength": 1.2, "defense_strength": 6.0, "momentum_score": 0.0, "games_played": 5},
            {"snapshot_date": "2026-04-15", "snapshot_label": "Week 3", "team": "Simsbury", "power_rank": 12, "power_rating": -0.5, "offense_strength": 1.4, "defense_strength": 0.1, "momentum_score": -2.0, "games_played": 5},
        ]
    )


def test_historical_storyline_cards_are_snapshot_based_and_deterministic() -> None:
    history = dashboard.historical_snapshot_display_data(_historical_storyline_rows())

    first = dashboard.historical_storyline_cards(history)
    second = dashboard.historical_storyline_cards(history)
    labels = {card["label"] for card in first}

    assert first == second
    assert "Biggest Climber" in labels
    assert "Biggest Fall" in labels
    assert "Holding the Top Spot" in labels
    assert "Fastest Rising Offense" in labels
    assert "Defensive Team to Watch" in labels
    climber = next(card for card in first if card["label"] == "Biggest Climber")
    assert climber["headline"] == "Avon"
    assert "Climbed 7 spots" in climber["body"]


def test_historical_storylines_handle_limited_snapshot_history() -> None:
    one_week = _historical_storyline_rows()
    one_week = one_week[one_week["snapshot_label"] == "Week 1"]
    history = dashboard.historical_snapshot_display_data(one_week)

    assert dashboard.historical_storyline_cards(history) == []


def test_historical_rank_movement_and_sparkline_data() -> None:
    history = dashboard.historical_snapshot_display_data(_historical_storyline_rows())

    movement = dashboard.historical_rank_movement(history, snapshots_back=3)
    sparkline = dashboard.historical_rank_sparkline_data(history, "Avon")

    assert movement.set_index("team").loc["Avon", "rank_move"] == 7
    assert movement.set_index("team").loc["Simsbury", "rank_move"] == -4
    assert sparkline["snapshot_label"].tolist() == ["Week 1", "Week 2", "Week 3"]
    assert sparkline["power_rank"].tolist() == [10, 7, 3]
