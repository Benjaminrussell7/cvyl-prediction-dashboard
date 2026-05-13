from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_ELO_RATINGS_CSV = "data/processed/cvyl_elo_ratings.csv"


@dataclass(frozen=True)
class MatchupPrediction:
    team_a: str
    team_b: str
    team_a_elo: float
    team_b_elo: float
    team_a_win_probability: float
    team_b_win_probability: float
    predicted_winner: str
    win_probability: float
    elo_difference: float


def predict_matchup_from_file(
    team_a: str,
    team_b: str,
    ratings_path: str | Path = DEFAULT_ELO_RATINGS_CSV,
) -> MatchupPrediction:
    ratings = pd.read_csv(ratings_path)
    return predict_matchup(team_a, team_b, ratings)


def predict_matchup(team_a: str, team_b: str, ratings: pd.DataFrame) -> MatchupPrediction:
    team_a_elo = _team_elo(team_a, ratings)
    team_b_elo = _team_elo(team_b, ratings)
    team_a_probability = elo_win_probability(team_a_elo, team_b_elo)
    team_b_probability = 1.0 - team_a_probability

    if team_a_probability >= team_b_probability:
        predicted_winner = team_a
        win_probability = team_a_probability
    else:
        predicted_winner = team_b
        win_probability = team_b_probability

    return MatchupPrediction(
        team_a=team_a,
        team_b=team_b,
        team_a_elo=team_a_elo,
        team_b_elo=team_b_elo,
        team_a_win_probability=team_a_probability,
        team_b_win_probability=team_b_probability,
        predicted_winner=predicted_winner,
        win_probability=win_probability,
        elo_difference=team_a_elo - team_b_elo,
    )


def elo_win_probability(team_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_elo - team_elo) / 400.0))


def format_matchup_prediction(prediction: MatchupPrediction) -> str:
    return "\n".join(
        [
            f"Matchup: {prediction.team_a} vs {prediction.team_b}",
            f"{prediction.team_a} ELO: {prediction.team_a_elo:.1f}",
            f"{prediction.team_b} ELO: {prediction.team_b_elo:.1f}",
            f"ELO difference ({prediction.team_a} - {prediction.team_b}): {prediction.elo_difference:.1f}",
            f"{prediction.team_a} win probability: {prediction.team_a_win_probability:.1%}",
            f"{prediction.team_b} win probability: {prediction.team_b_win_probability:.1%}",
            f"Predicted winner: {prediction.predicted_winner} ({prediction.win_probability:.1%})",
        ]
    )


def _team_elo(team_name: str, ratings: pd.DataFrame) -> float:
    matches = ratings[ratings["team"] == team_name]
    if matches.empty:
        raise ValueError(f"Team not found in ELO ratings: {team_name}")
    return float(matches.iloc[0]["elo"])
