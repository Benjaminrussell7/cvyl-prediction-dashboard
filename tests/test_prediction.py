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

    prediction = predict_matchup("West Hartford 12U Gold", "Canton 12U", ratings)

    assert prediction.predicted_winner == "West Hartford 12U Gold"
    assert prediction.win_probability > 0.5
    assert prediction.team_a_win_probability > prediction.team_b_win_probability
    assert prediction.elo_difference == 150.0


def test_predict_matchup_output_is_deterministic() -> None:
    ratings = pd.DataFrame(
        [
            {"team": "Avon 12U", "elo": 1525.0, "games_played": 8},
            {"team": "Granby 12U", "elo": 1490.0, "games_played": 8},
        ]
    )

    first = predict_matchup("Avon 12U", "Granby 12U", ratings)
    second = predict_matchup("Avon 12U", "Granby 12U", ratings)

    assert first == second
    assert format_matchup_prediction(first) == format_matchup_prediction(second)
