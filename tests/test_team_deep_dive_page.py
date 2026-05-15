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


def test_team_historical_snapshots_filters_and_sorts_snapshots() -> None:
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-04-14",
                "snapshot_label": "Week 2",
                "team": "Avon",
                "power_rank": 2,
                "power_rating": 1.4,
            },
            {
                "snapshot_date": "2026-04-07",
                "snapshot_label": "Week 1",
                "team": "Avon",
                "power_rank": 4,
                "power_rating": 0.8,
            },
            {
                "snapshot_date": "2026-04-07",
                "snapshot_label": "Week 1",
                "team": "RHAM",
                "power_rank": 1,
                "power_rating": 1.8,
            },
        ]
    )

    rows = team_page.team_historical_snapshots("Avon", snapshots)

    assert rows["Team"].tolist() == ["Avon", "Avon"]
    assert rows["Snapshot"].tolist() == ["Week 1", "Week 2"]
    assert rows["Power Rank"].tolist() == [4, 2]
    assert rows["Power Rating"].tolist() == [0.8, 1.4]


def test_historical_trajectory_data_limits_comparison_teams_and_labels_points() -> None:
    snapshots = pd.DataFrame(
        [
            {"snapshot_date": "2026-04-07", "snapshot_label": "Week 1", "team": "Avon", "power_rank": 4, "power_rating": 0.8},
            {"snapshot_date": "2026-04-14", "snapshot_label": "Week 2", "team": "Avon", "power_rank": 2, "power_rating": 1.4},
            {"snapshot_date": "2026-04-07", "snapshot_label": "Week 1", "team": "RHAM", "power_rank": 1, "power_rating": 1.8},
            {"snapshot_date": "2026-04-14", "snapshot_label": "Week 2", "team": "RHAM", "power_rank": 3, "power_rating": 1.0},
            {"snapshot_date": "2026-04-07", "snapshot_label": "Week 1", "team": "Granby", "power_rank": 8, "power_rating": -0.2},
            {"snapshot_date": "2026-04-14", "snapshot_label": "Week 2", "team": "Granby", "power_rank": 5, "power_rating": 0.4},
            {"snapshot_date": "2026-04-07", "snapshot_label": "Week 1", "team": "Simsbury", "power_rank": 9, "power_rating": -0.4},
            {"snapshot_date": "2026-04-14", "snapshot_label": "Week 2", "team": "Simsbury", "power_rank": 6, "power_rating": 0.2},
            {"snapshot_date": "2026-04-07", "snapshot_label": "Week 1", "team": "Windsor", "power_rank": 10, "power_rating": -0.8},
            {"snapshot_date": "2026-04-14", "snapshot_label": "Week 2", "team": "Windsor", "power_rank": 7, "power_rating": 0.1},
        ]
    )

    rows = team_page.historical_trajectory_data(
        snapshots,
        "Avon",
        ["RHAM", "Granby", "Simsbury", "Windsor"],
    )

    assert set(rows["Team"]) == {"Avon", "RHAM", "Granby", "Simsbury"}
    assert set(rows[rows["Team"] == "Avon"]["Point Type"]) == {"Selected Team"}
    assert set(rows[rows["Team"] != "Avon"]["Point Type"]) == {"Comparison Team"}


def test_season_trajectory_summary_and_callouts() -> None:
    snapshots = pd.DataFrame(
        [
            {"Snapshot Date": pd.Timestamp("2026-04-07"), "Snapshot": "Week 1", "Power Rank": 9, "Power Rating": 0.1},
            {"Snapshot Date": pd.Timestamp("2026-04-14"), "Snapshot": "Week 2", "Power Rank": 6, "Power Rating": 0.7},
            {"Snapshot Date": pd.Timestamp("2026-04-21"), "Snapshot": "Week 3", "Power Rank": 3, "Power Rating": 1.5},
        ]
    )

    summary = team_page.season_trajectory_summary(snapshots)
    callouts = {item["label"]: item["value"] for item in team_page.season_trajectory_callouts(summary)}

    assert summary["start_rank"] == 9
    assert summary["current_rank"] == 3
    assert summary["net_movement"] == 6
    assert summary["best_rank"] == 3
    assert summary["worst_rank"] == 9
    assert callouts["Started Week 1"] == "#9"
    assert callouts["Current Rank"] == "#3"
    assert callouts["Net Movement"] == "↑ 6"
    assert team_page.season_trajectory_narrative(summary) == "Climbing steadily."


def test_season_trajectory_narrative_handles_cooling_and_top_tier() -> None:
    assert (
        team_page.season_trajectory_narrative(
            {
                "start_rank": 3,
                "current_rank": 4,
                "net_movement": -1,
                "recent_movement": -1,
            }
        )
        == "Holding near the top."
    )
    assert (
        team_page.season_trajectory_narrative(
            {
                "start_rank": 4,
                "current_rank": 10,
                "net_movement": -6,
                "recent_movement": -2,
            }
        )
        == "Cooling after a strong start."
    )


def test_historical_snapshot_chart_reverses_rank_axis() -> None:
    snapshots = pd.DataFrame(
        [
            {"Team": "Avon", "Snapshot Date": pd.Timestamp("2026-04-07"), "Snapshot": "Week 1", "Power Rank": 4, "Power Rating": 0.8},
            {"Team": "Avon", "Snapshot Date": pd.Timestamp("2026-04-14"), "Snapshot": "Week 2", "Power Rank": 2, "Power Rating": 1.4},
        ]
    )

    chart = team_page.historical_snapshot_chart(snapshots)
    chart_dict = chart.to_dict()

    assert chart_dict["vconcat"][0]["encoding"]["y"]["scale"]["reverse"] is True


def test_comparison_point_type_labels_selected_comparison_and_league() -> None:
    assert team_page.comparison_point_type("Avon", "Avon", ["Granby"]) == "Selected Team"
    assert team_page.comparison_point_type("Granby", "Avon", ["Granby"]) == "Comparison Team"
    assert team_page.comparison_point_type("RHAM", "Avon", ["Granby"]) == "League Team"


def test_team_storyline_falls_back_when_dashboard_notification_helper_missing(monkeypatch) -> None:
    monkeypatch.delattr(team_page.dashboard, "notification_phrase_for_team", raising=False)

    storyline = team_page.team_storyline_sentence(
        "Avon",
        "Improving",
        2,
        pd.DataFrame(),
    )

    assert storyline == "Momentum is trending upward."


def test_team_storytelling_uses_deterministic_narrative() -> None:
    data = {
        "power_ratings": pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "power_rank_v3_recency": 4,
                    "avg_points_for": 7.0,
                    "avg_points_against": 3.0,
                    "avg_margin": 4.0,
                    "adjusted_offense_rating": 2.0,
                    "adjusted_defense_rating": 3.0,
                }
            ]
        ),
        "trends": pd.DataFrame(
            [{"team": "Avon", "momentum_label": "Steady", "power_rank_movement": 1}]
        ),
        "sos": pd.DataFrame([{"team": "Avon", "sos_rank": 4}]),
        "team_games": pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "status": "completed",
                    "game_date": "2026-05-01",
                    "opponent": "Granby",
                    "points_for": 7,
                    "points_against": 3,
                    "win": True,
                }
            ]
        ),
    }

    story = team_page.team_storytelling("Avon", data)

    assert "headline" in story
    assert "identity" in story
    assert story["model_sees"]
    assert story == team_page.team_storytelling("Avon", data)
    sections = [story["headline"], story["identity"], story["storyline"], *story["model_sees"]]
    assert len(sections) == len(set(sections))


def test_team_storytelling_does_not_require_dashboard_text_helpers(monkeypatch) -> None:
    monkeypatch.delattr(team_page.dashboard, "notification_phrase_for_team", raising=False)
    monkeypatch.delattr(team_page.dashboard, "dedupe_text", raising=False)
    data = {
        "power_ratings": pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "power_rank_v3_recency": 4,
                    "avg_points_for": 7.0,
                    "avg_points_against": 3.0,
                    "avg_margin": 4.0,
                    "adjusted_offense_rating": 2.0,
                    "adjusted_defense_rating": 3.0,
                }
            ]
        ),
        "trends": pd.DataFrame(
            [{"team": "Avon", "momentum_label": "Steady", "power_rank_movement": 1}]
        ),
        "sos": pd.DataFrame([{"team": "Avon", "sos_rank": 4}]),
        "team_games": pd.DataFrame(),
    }

    story = team_page.team_storytelling("Avon", data)

    assert story["storyline"] == "Momentum is trending upward."
    assert len(story["model_sees"]) == len(set(story["model_sees"]))
    assert story["headline"] not in story["model_sees"]
    assert story["identity"] not in story["model_sees"]
    assert story["storyline"] not in story["model_sees"]


def test_local_dedupe_text_preserves_order() -> None:
    assert team_page.dedupe_text(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]


def test_local_dedupe_text_excludes_existing_section_text() -> None:
    assert team_page.dedupe_text(
        ["Strong defense.", "Momentum is trending upward.", "Strong defense."],
        excluded=["Momentum is trending upward."],
    ) == ["Strong defense."]


def test_build_team_narrative_uses_distinct_headline() -> None:
    data = {
        "power_ratings": pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "power_rank_v3_recency": 4,
                    "avg_points_for": 7.0,
                    "avg_points_against": 3.0,
                    "avg_margin": 4.0,
                    "adjusted_offense_rating": 2.0,
                    "adjusted_defense_rating": 3.0,
                }
            ]
        ),
        "trends": pd.DataFrame(
            [{"team": "Avon", "momentum_label": "Steady", "power_rank_movement": 1}]
        ),
        "sos": pd.DataFrame([{"team": "Avon", "sos_rank": 4}]),
        "team_games": pd.DataFrame(),
    }

    story = team_page.team_storytelling("Avon", data)
    headline = team_page.build_team_narrative("Avon", data)

    assert headline == story["headline"]
    assert headline != story["identity"]
    assert headline != story["storyline"]


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
