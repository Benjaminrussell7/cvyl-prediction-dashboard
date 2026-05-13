from __future__ import annotations

import pandas as pd

from cvyl_scraper.explanations import generate_matchup_explanation


def test_explanation_generation_is_deterministic() -> None:
    first = _explain("Avon", "Canton", "Avon")
    second = _explain("Avon", "Canton", "Avon")

    assert first == second


def test_explanation_generation_for_favored_team_uses_offensive_edge() -> None:
    explanation = _explain("Avon", "Canton", "Avon")

    assert "Avon" in explanation
    assert "scoring" in explanation


def test_explanation_generation_handles_balanced_matchup() -> None:
    explanation = generate_matchup_explanation(
        "Avon",
        "Canton",
        predicted_winner="Avon",
        win_probability=0.53,
        confidence_tier="Medium",
        power_ratings=pd.DataFrame(
            [
                {"team": "Avon", "power_rating_v3_recency": 1.1},
                {"team": "Canton", "power_rating_v3_recency": 1.0},
            ]
        ),
        trends=pd.DataFrame(),
        sos=pd.DataFrame(),
    )

    assert "toss-up" in explanation


def test_explanation_generation_adds_upset_warning_for_hot_underdog() -> None:
    explanation = generate_matchup_explanation(
        "Avon",
        "Canton",
        predicted_winner="Avon",
        win_probability=0.61,
        confidence_tier="Medium",
        power_ratings=pd.DataFrame(
            [
                {"team": "Avon", "power_rating_v3_recency": 2.0},
                {"team": "Canton", "power_rating_v3_recency": 1.0},
            ]
        ),
        trends=pd.DataFrame(
            [
                {"team": "Avon", "momentum_score": 0.0, "momentum_label": "Steady"},
                {"team": "Canton", "momentum_score": 5.0, "momentum_label": "Surging"},
            ]
        ),
        sos=pd.DataFrame(),
    )

    assert "upset potential" in explanation


def test_explanation_generation_handles_missing_data() -> None:
    explanation = generate_matchup_explanation(
        "Avon",
        "Canton",
        predicted_winner="Avon",
        win_probability=0.62,
        confidence_tier="Low",
        power_ratings=pd.DataFrame(),
        trends=pd.DataFrame(),
        sos=pd.DataFrame(),
    )

    assert explanation
    assert "Avon" in explanation


def _explain(team_a: str, team_b: str, favorite: str) -> str:
    return generate_matchup_explanation(
        team_a,
        team_b,
        predicted_winner=favorite,
        win_probability=0.68,
        confidence_tier="High",
        power_ratings=pd.DataFrame(
            [
                {"team": "Avon", "power_rating_v3_recency": 3.0},
                {"team": "Canton", "power_rating_v3_recency": 0.5},
            ]
        ),
        trends=pd.DataFrame(
            [
                {
                    "team": "Avon",
                    "recent_offense_rating": 11.0,
                    "recent_defense_rating": 5.0,
                    "momentum_score": 2.0,
                    "momentum_label": "Improving",
                },
                {
                    "team": "Canton",
                    "recent_offense_rating": 7.0,
                    "recent_defense_rating": 6.0,
                    "momentum_score": 0.0,
                    "momentum_label": "Steady",
                },
            ]
        ),
        sos=pd.DataFrame(
            [
                {"team": "Avon", "average_opponent_elo": 1510.0},
                {"team": "Canton", "average_opponent_elo": 1490.0},
            ]
        ),
    )
