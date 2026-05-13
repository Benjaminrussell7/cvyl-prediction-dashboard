from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_ELO_RATINGS_CSV = "data/processed/cvyl_elo_ratings.csv"
DEFAULT_TEAM_GAMES_CSV = "data/processed/cvyl_team_games.csv"


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
    projected_team_a_goals: float
    projected_team_b_goals: float
    projected_margin: float
    projected_spread: str
    projected_total_goals: float


def predict_matchup_from_file(
    team_a: str,
    team_b: str,
    ratings_path: str | Path = DEFAULT_ELO_RATINGS_CSV,
    team_games_path: str | Path = DEFAULT_TEAM_GAMES_CSV,
) -> MatchupPrediction:
    ratings = pd.read_csv(ratings_path)
    team_games = pd.read_csv(team_games_path)
    return predict_matchup(team_a, team_b, ratings, team_games)


def predict_matchup(
    team_a: str,
    team_b: str,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
) -> MatchupPrediction:
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

    team_a_projected_goals, team_b_projected_goals = _projected_goals(team_a, team_b, team_games)
    projected_margin = team_a_projected_goals - team_b_projected_goals
    projected_spread = _projected_spread(team_a, team_b, projected_margin)

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
        projected_team_a_goals=team_a_projected_goals,
        projected_team_b_goals=team_b_projected_goals,
        projected_margin=projected_margin,
        projected_spread=projected_spread,
        projected_total_goals=team_a_projected_goals + team_b_projected_goals,
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
            f"Projected spread: {prediction.projected_spread}",
            f"Projected total goals: {prediction.projected_total_goals:.1f}",
            (
                "Projected score: "
                f"{prediction.team_a} {prediction.projected_team_a_goals:.1f}, "
                f"{prediction.team_b} {prediction.projected_team_b_goals:.1f}"
            ),
        ]
    )


def _team_elo(team_name: str, ratings: pd.DataFrame) -> float:
    matches = ratings[ratings["team"] == team_name]
    if matches.empty:
        raise ValueError(f"Team not found in ELO ratings: {team_name}")
    return float(matches.iloc[0]["elo"])


def _projected_goals(team_a: str, team_b: str, team_games: pd.DataFrame) -> tuple[float, float]:
    completed = team_games[team_games["status"] == "completed"].copy()
    completed["points_for"] = pd.to_numeric(completed["points_for"], errors="coerce")
    completed["points_against"] = pd.to_numeric(completed["points_against"], errors="coerce")
    completed = completed.dropna(subset=["team", "points_for", "points_against"])

    team_a_profile = _scoring_profile(team_a, completed)
    team_b_profile = _scoring_profile(team_b, completed)
    team_a_goals = (team_a_profile["points_for"] + team_b_profile["points_against"]) / 2.0
    team_b_goals = (team_b_profile["points_for"] + team_a_profile["points_against"]) / 2.0
    return team_a_goals, team_b_goals


def _scoring_profile(team_name: str, team_games: pd.DataFrame) -> dict[str, float]:
    rows = team_games[team_games["team"] == team_name]
    if rows.empty:
        raise ValueError(f"Team not found in completed team-game data: {team_name}")
    return {
        "points_for": float(rows["points_for"].mean()),
        "points_against": float(rows["points_against"].mean()),
    }


def _projected_spread(team_a: str, team_b: str, projected_margin: float) -> str:
    if projected_margin > 0:
        return f"{team_a} by {abs(projected_margin):.1f}"
    if projected_margin < 0:
        return f"{team_b} by {abs(projected_margin):.1f}"
    return "Pick'em"
