from __future__ import annotations

import importlib.util
from pathlib import Path

import altair as alt
import pandas as pd

PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "1_Team_Deep_Dive.py"
SPEC = importlib.util.spec_from_file_location("team_deep_dive_page", PAGE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
team_page = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(team_page)


def test_expected_win_probability_from_pregame_elo() -> None:
    row = pd.Series({"pregame_elo": 1600.0, "opponent_pregame_elo": 1500.0})

    probability = team_page.expected_win_probability_from_elo(row)

    assert probability > 0.63
    assert probability < 0.65


def test_team_snapshot_includes_record_and_core_metrics() -> None:
    data = {
        "ratings": pd.DataFrame([{"team": "Avon", "elo": 1510.0, "games_played": 2}]),
        "power_ratings": pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "power_rank_v3_recency": 4,
                    "power_rating_v3_recency": 1.25,
                    "avg_points_for": 6.0,
                    "avg_points_against": 4.0,
                    "avg_margin": 2.0,
                }
            ]
        ),
        "sos": pd.DataFrame([{"team": "Avon", "sos_rank": 3}]),
        "trends": pd.DataFrame([{"team": "Avon", "momentum_label": "Steady", "power_rank_movement": 1}]),
        "team_games": pd.DataFrame(
            [
                {"team": "Avon", "status": "completed", "points_for": 6, "points_against": 4, "win": True},
                {"team": "Avon", "status": "completed", "points_for": 3, "points_against": 5, "win": False},
            ]
        ),
    }

    snapshot = team_page.team_snapshot("Avon", data)
    values = {item["label"]: item["value"] for item in snapshot}

    assert values["Current Rank"] == "4"
    assert values["Record"] == "1-1"
    assert values["Momentum"] == "Improving"


def test_recent_result_cards_adds_upset_indicator() -> None:
    data = {
        "team_games": pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "opponent": "Granby",
                    "status": "completed",
                    "points_for": 5,
                    "points_against": 4,
                    "win": True,
                    "game_date": "2026-05-01",
                    "pregame_elo": 1450.0,
                    "opponent_pregame_elo": 1550.0,
                }
            ]
        ),
        "power_ratings": pd.DataFrame([{"team": "Granby", "power_rank_v3_recency": 2}]),
    }

    cards = team_page.recent_result_cards("Avon", data)

    assert bool(cards.iloc[0]["upset"])
    assert cards.iloc[0]["opponent_rank"] == 2


def test_schedule_difficulty_label_uses_existing_power_distribution() -> None:
    power = pd.DataFrame(
        {
            "team": ["A", "B", "C"],
            "power_rating_v3_recency": [-1.0, 0.0, 2.0],
        }
    )

    assert team_page.schedule_difficulty_label(2.0, power) == "Difficult"
    assert team_page.schedule_difficulty_label(-1.0, power) == "Easy"


def test_league_comparison_data_merges_existing_sources() -> None:
    data = {
        "power_ratings": pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "power_rating_v3_recency": 1.2,
                    "power_rank_v3_recency": 4,
                    "games_played": 5,
                    "adjusted_offense_rating": 0.8,
                    "adjusted_defense_rating": 1.1,
                    "avg_margin": 2.0,
                }
            ]
        ),
        "ratings": pd.DataFrame([{"team": "Avon", "elo": 1510.0}]),
        "sos": pd.DataFrame([{"team": "Avon", "sos_rank": 3}]),
        "trends": pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "last_3_win_pct": 0.67,
                    "last_5_win_pct": 0.60,
                    "momentum_score": 2.5,
                    "momentum_label": "Steady",
                    "power_rank_movement": 1,
                }
            ]
        ),
    }

    comparison = team_page.league_comparison_data(data)

    assert comparison.iloc[0]["Team"] == "Avon"
    assert comparison.iloc[0]["Power Rating"] == 1.2
    assert comparison.iloc[0]["ELO"] == 1510.0
    assert comparison.iloc[0]["Momentum/Form"] == "Improving"


def test_available_comparison_metrics_includes_numeric_columns_only() -> None:
    comparison = pd.DataFrame(
        [
            {
                "Team": "Avon",
                "Power Rating": 1.2,
                "Power Rank": 4,
                "Offensive Strength": 0.8,
                "Momentum/Form": "Improving",
            }
        ]
    )

    metrics = team_page.available_comparison_metrics(comparison)

    assert "Power Rating" in metrics
    assert "Power Rank" in metrics
    assert "Offensive Strength" in metrics
    assert "Momentum/Form" not in metrics


def test_rank_and_sos_axes_are_reversed() -> None:
    assert team_page.axis_scale_for_metric("Power Rank").reverse is True
    assert team_page.axis_scale_for_metric("SOS Rank").reverse is True
    assert team_page.axis_scale_for_metric("Power Rating").reverse is alt.Undefined


def test_comparison_point_type_labels_selected_comparison_and_league() -> None:
    assert team_page.comparison_point_type("Avon", "Avon", ["Granby"]) == "Selected Team"
    assert team_page.comparison_point_type("Granby", "Avon", ["Granby"]) == "Comparison Team"
    assert team_page.comparison_point_type("RHAM", "Avon", ["Granby"]) == "League Team"


def test_league_comparison_chart_uses_point_type_encoding() -> None:
    comparison = pd.DataFrame(
        [
            {
                "Team": "Avon",
                "Offensive Strength": 1.0,
                "Defensive Strength": 2.0,
                "Power Rank": 3,
                "Power Rating": 1.2,
                "ELO": 1510.0,
                "Momentum/Form": "Surging",
                "Games Played": 5,
            },
            {
                "Team": "Granby",
                "Offensive Strength": 0.5,
                "Defensive Strength": 1.0,
                "Power Rank": 8,
                "Power Rating": 0.2,
                "ELO": 1490.0,
                "Momentum/Form": "Steady",
                "Games Played": 4,
            },
        ]
    )

    chart_spec = team_page.league_comparison_chart(
        comparison,
        "Avon",
        ["Granby"],
        "Offensive Strength",
        "Defensive Strength",
    ).to_dict()

    assert chart_spec["encoding"]["color"]["field"] == "Point Type"
    assert chart_spec["encoding"]["size"]["field"] == "Point Type"
    assert "Point Type" in {tooltip["field"] for tooltip in chart_spec["encoding"]["tooltip"]}
