from __future__ import annotations

import pandas as pd

from cvyl_scraper.prediction import (
    elo_win_probability,
    format_matchup_prediction,
    predict_matchup,
)


def test_elo_win_probability_equal_ratings_is_about_even() -> None:
    assert elo_win_probability(1500, 1500) == 0.5


def test_predict_matchup_favors_stronger_team() -> None:
    ratings = pd.DataFrame(
        [
            {"team": "West Hartford 12U Gold", "elo": 1600.0, "games_played": 8},
            {"team": "Canton 12U", "elo": 1450.0, "games_played": 8},
        ]
    )
    team_games = pd.DataFrame(
        [
            _team_game("West Hartford 12U Gold", 12, 4),
            _team_game("West Hartford 12U Gold", 10, 6),
            _team_game("Canton 12U", 5, 11),
            _team_game("Canton 12U", 6, 10),
        ]
    )

    prediction = predict_matchup("West Hartford 12U Gold", "Canton 12U", ratings, team_games)

    assert prediction.predicted_winner == "West Hartford 12U Gold"
    assert prediction.win_probability > 0.5
    assert prediction.team_a_win_probability > prediction.team_b_win_probability
    assert prediction.elo_difference == 150.0
    assert prediction.projected_margin > 0
    assert prediction.projected_spread.startswith("West Hartford 12U Gold by")


def test_predict_matchup_output_is_deterministic() -> None:
    ratings = pd.DataFrame(
        [
            {"team": "Avon 12U", "elo": 1525.0, "games_played": 8},
            {"team": "Granby 12U", "elo": 1490.0, "games_played": 8},
        ]
    )
    team_games = pd.DataFrame(
        [
            _team_game("Avon 12U", 8, 5),
            _team_game("Avon 12U", 10, 7),
            _team_game("Granby 12U", 6, 8),
            _team_game("Granby 12U", 7, 9),
        ]
    )

    first = predict_matchup("Avon 12U", "Granby 12U", ratings, team_games)
    second = predict_matchup("Avon 12U", "Granby 12U", ratings, team_games)

    assert first == second
    assert format_matchup_prediction(first) == format_matchup_prediction(second)


def test_predict_matchup_total_goals_is_numeric_and_reasonable() -> None:
    ratings = pd.DataFrame(
        [
            {"team": "Avon 12U", "elo": 1525.0, "games_played": 8},
            {"team": "Granby 12U", "elo": 1490.0, "games_played": 8},
        ]
    )
    team_games = pd.DataFrame(
        [
            _team_game("Avon 12U", 8, 5),
            _team_game("Avon 12U", 10, 7),
            _team_game("Granby 12U", 6, 8),
            _team_game("Granby 12U", 7, 9),
            _team_game("Avon 12U", None, None, status="scheduled"),
        ]
    )

    prediction = predict_matchup("Avon 12U", "Granby 12U", ratings, team_games)

    assert isinstance(prediction.projected_total_goals, float)
    assert 0 < prediction.projected_total_goals < 40
    assert prediction.projected_total_goals == 15.0


def test_predict_matchup_high_confidence_has_no_warning() -> None:
    prediction = predict_matchup(
        "Avon 12U",
        "Granby 12U",
        _ratings(5, 6),
        _team_games_for_prediction(),
    )

    assert prediction.confidence_level == "High"
    assert prediction.confidence_warning is None
    assert "Confidence: High" in format_matchup_prediction(prediction)
    assert "Warning:" not in format_matchup_prediction(prediction)


def test_predict_matchup_includes_sos_when_available() -> None:
    prediction = predict_matchup(
        "Avon 12U",
        "Granby 12U",
        _ratings(5, 6),
        _team_games_for_prediction(),
        pd.DataFrame(
            [
                {
                    "team": "Avon 12U",
                    "games_played": 5,
                    "average_opponent_elo": 1510.0,
                    "opponent_count": 4,
                    "sos_rank": 2,
                },
                {
                    "team": "Granby 12U",
                    "games_played": 6,
                    "average_opponent_elo": 1490.0,
                    "opponent_count": 5,
                    "sos_rank": 4,
                },
            ]
        ),
    )

    output = format_matchup_prediction(prediction)

    assert prediction.team_a_sos == 1510.0
    assert prediction.team_b_sos_rank == 4
    assert "Avon 12U SOS: 1510.0 opponent ELO (rank 2)" in output


def test_predict_matchup_includes_power_v2_context_when_available() -> None:
    prediction = predict_matchup(
        "Avon 12U",
        "Granby 12U",
        _ratings(5, 6),
        _team_games_for_prediction(),
        power_v2=pd.DataFrame(
            [
                {
                    "team": "Avon 12U",
                    "power_rating_v2": 3.25,
                    "power_rank_v2": 2,
                },
                {
                    "team": "Granby 12U",
                    "power_rating_v2": 1.10,
                    "power_rank_v2": 5,
                },
            ]
        ),
    )

    output = format_matchup_prediction(prediction)

    assert prediction.team_a_power_v2 == 3.25
    assert prediction.team_b_power_rank_v2 == 5
    assert "Avon 12U Power v2: 3.25 (rank 2)" in output


def test_predict_matchup_medium_confidence_has_warning() -> None:
    prediction = predict_matchup(
        "Avon 12U",
        "Granby 12U",
        _ratings(3, 4),
        _team_games_for_prediction(),
    )

    assert prediction.confidence_level == "Medium"
    assert prediction.confidence_warning is not None
    assert "Warning:" in format_matchup_prediction(prediction)


def test_predict_matchup_low_confidence_has_warning() -> None:
    prediction = predict_matchup(
        "Avon 12U",
        "Granby 12U",
        _ratings(2, 8),
        _team_games_for_prediction(),
    )

    assert prediction.confidence_level == "Low"
    assert prediction.confidence_warning is not None
    assert "Small sample size" in prediction.confidence_warning


def _team_game(
    team: str,
    points_for: int | None,
    points_against: int | None,
    *,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "team": team,
        "points_for": points_for,
        "points_against": points_against,
        "status": status,
    }


def _ratings(team_a_games_played: int, team_b_games_played: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "Avon 12U", "elo": 1525.0, "games_played": team_a_games_played},
            {"team": "Granby 12U", "elo": 1490.0, "games_played": team_b_games_played},
        ]
    )


def _team_games_for_prediction() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _team_game("Avon 12U", 8, 5),
            _team_game("Avon 12U", 10, 7),
            _team_game("Granby 12U", 6, 8),
            _team_game("Granby 12U", 7, 9),
        ]
    )
