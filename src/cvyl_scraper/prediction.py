from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cvyl_scraper.hybrid import hybrid_model_edge, hybrid_win_probability, power_v2_win_probability
from cvyl_scraper.team_matching import resolve_team_name


DEFAULT_ELO_RATINGS_CSV = "data/processed/cvyl_elo_ratings.csv"
DEFAULT_TEAM_GAMES_CSV = "data/processed/cvyl_team_games.csv"
DEFAULT_SOS_CSV = "data/processed/cvyl_sos.csv"
DEFAULT_POWER_RATINGS_V2_CSV = "data/processed/cvyl_power_ratings_v2.csv"


@dataclass(frozen=True)
class MatchupPrediction:
    team_a: str
    team_b: str
    team_a_elo: float
    team_b_elo: float
    team_a_games_played: int
    team_b_games_played: int
    team_a_sos: float | None
    team_b_sos: float | None
    team_a_sos_rank: int | None
    team_b_sos_rank: int | None
    team_a_power_v2: float | None
    team_b_power_v2: float | None
    team_a_power_rank_v2: int | None
    team_b_power_rank_v2: int | None
    power_v2_win_probability_team_a: float
    power_v2_win_probability_team_b: float
    power_v2_predicted_winner: str
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
    confidence_level: str
    confidence_warning: str | None
    hybrid_win_probability_team_a: float
    hybrid_win_probability_team_b: float
    hybrid_predicted_winner: str
    hybrid_model_edge: float


def predict_matchup_from_file(
    team_a: str,
    team_b: str,
    ratings_path: str | Path = DEFAULT_ELO_RATINGS_CSV,
    team_games_path: str | Path = DEFAULT_TEAM_GAMES_CSV,
    sos_path: str | Path = DEFAULT_SOS_CSV,
    power_v2_path: str | Path | None = DEFAULT_POWER_RATINGS_V2_CSV,
) -> MatchupPrediction:
    ratings = pd.read_csv(ratings_path)
    team_games = pd.read_csv(team_games_path)
    sos = _read_optional_sos(sos_path)
    power_v2 = _read_optional_power_v2(power_v2_path)
    return predict_matchup(team_a, team_b, ratings, team_games, sos, power_v2)


def predict_matchup(
    team_a: str,
    team_b: str,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame | None = None,
    power_v2: pd.DataFrame | None = None,
) -> MatchupPrediction:
    team_a_model_name = _resolve_from_frame(team_a, ratings)
    team_b_model_name = _resolve_from_frame(team_b, ratings)
    team_a_elo = _team_elo(team_a_model_name, ratings)
    team_b_elo = _team_elo(team_b_model_name, ratings)
    team_a_games_played = _team_games_played(team_a_model_name, ratings)
    team_b_games_played = _team_games_played(team_b_model_name, ratings)
    team_a_sos, team_a_sos_rank = _team_sos(team_a_model_name, sos)
    team_b_sos, team_b_sos_rank = _team_sos(team_b_model_name, sos)
    team_a_power_v2, team_a_power_rank_v2 = _team_power_v2(team_a_model_name, power_v2)
    team_b_power_v2, team_b_power_rank_v2 = _team_power_v2(team_b_model_name, power_v2)
    team_a_power_value = team_a_power_v2 or 0.0
    team_b_power_value = team_b_power_v2 or 0.0
    team_a_power_probability = power_v2_win_probability(team_a_power_value - team_b_power_value)
    team_b_power_probability = 1.0 - team_a_power_probability
    power_predicted_winner = team_a if team_a_power_probability >= 0.5 else team_b
    team_a_probability = elo_win_probability(team_a_elo, team_b_elo)
    team_b_probability = 1.0 - team_a_probability
    team_a_hybrid_probability = hybrid_win_probability(
        team_a_elo - team_b_elo,
        team_a_power_value - team_b_power_value,
    )
    team_b_hybrid_probability = 1.0 - team_a_hybrid_probability
    hybrid_edge = hybrid_model_edge(team_a_elo - team_b_elo, team_a_power_value - team_b_power_value)
    hybrid_predicted_winner = team_a if team_a_hybrid_probability >= 0.5 else team_b

    if team_a_probability >= team_b_probability:
        predicted_winner = team_a
        win_probability = team_a_probability
    else:
        predicted_winner = team_b
        win_probability = team_b_probability

    team_a_scoring_name = _resolve_from_frame(team_a, team_games)
    team_b_scoring_name = _resolve_from_frame(team_b, team_games)
    team_a_projected_goals, team_b_projected_goals = _projected_goals(
        team_a_scoring_name,
        team_b_scoring_name,
        team_games,
    )
    projected_margin = team_a_projected_goals - team_b_projected_goals
    projected_spread = _projected_spread(team_a, team_b, projected_margin)
    confidence_level = _confidence_level(team_a_games_played, team_b_games_played)

    return MatchupPrediction(
        team_a=team_a,
        team_b=team_b,
        team_a_elo=team_a_elo,
        team_b_elo=team_b_elo,
        team_a_games_played=team_a_games_played,
        team_b_games_played=team_b_games_played,
        team_a_sos=team_a_sos,
        team_b_sos=team_b_sos,
        team_a_sos_rank=team_a_sos_rank,
        team_b_sos_rank=team_b_sos_rank,
        team_a_power_v2=team_a_power_v2,
        team_b_power_v2=team_b_power_v2,
        team_a_power_rank_v2=team_a_power_rank_v2,
        team_b_power_rank_v2=team_b_power_rank_v2,
        power_v2_win_probability_team_a=team_a_power_probability,
        power_v2_win_probability_team_b=team_b_power_probability,
        power_v2_predicted_winner=power_predicted_winner,
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
        confidence_level=confidence_level,
        confidence_warning=_confidence_warning(confidence_level),
        hybrid_win_probability_team_a=team_a_hybrid_probability,
        hybrid_win_probability_team_b=team_b_hybrid_probability,
        hybrid_predicted_winner=hybrid_predicted_winner,
        hybrid_model_edge=hybrid_edge,
    )


def elo_win_probability(team_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_elo - team_elo) / 400.0))


def format_matchup_prediction(prediction: MatchupPrediction) -> str:
    return "\n".join(
        [
            f"Matchup: {prediction.team_a} vs {prediction.team_b}",
            f"{prediction.team_a} ELO: {prediction.team_a_elo:.1f} ({prediction.team_a_games_played} games)",
            f"{prediction.team_b} ELO: {prediction.team_b_elo:.1f} ({prediction.team_b_games_played} games)",
            f"{prediction.team_a} SOS: {_format_sos(prediction.team_a_sos, prediction.team_a_sos_rank)}",
            f"{prediction.team_b} SOS: {_format_sos(prediction.team_b_sos, prediction.team_b_sos_rank)}",
            *(_format_power_v2_lines(prediction)),
            "Power Rating prediction:",
            f"  {prediction.team_a} win probability: {prediction.power_v2_win_probability_team_a:.1%}",
            f"  {prediction.team_b} win probability: {prediction.power_v2_win_probability_team_b:.1%}",
            f"  Predicted winner: {prediction.power_v2_predicted_winner}",
            f"ELO difference ({prediction.team_a} - {prediction.team_b}): {prediction.elo_difference:.1f}",
            "ELO prediction:",
            f"  {prediction.team_a} win probability: {prediction.team_a_win_probability:.1%}",
            f"  {prediction.team_b} win probability: {prediction.team_b_win_probability:.1%}",
            f"  Predicted winner: {prediction.predicted_winner} ({prediction.win_probability:.1%})",
            "Hybrid prediction:",
            f"  {prediction.team_a} win probability: {prediction.hybrid_win_probability_team_a:.1%}",
            f"  {prediction.team_b} win probability: {prediction.hybrid_win_probability_team_b:.1%}",
            f"  Predicted winner: {prediction.hybrid_predicted_winner}",
            f"  Hybrid model edge: {prediction.hybrid_model_edge:.3f}",
            f"Confidence: {prediction.confidence_level}",
            *([f"Warning: {prediction.confidence_warning}"] if prediction.confidence_warning else []),
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
    matches = _team_matches(team_name, ratings)
    if matches.empty:
        raise ValueError(f"Team not found in ELO ratings: {team_name}")
    return float(matches.iloc[0]["elo"])


def _team_games_played(team_name: str, ratings: pd.DataFrame) -> int:
    matches = _team_matches(team_name, ratings)
    if matches.empty:
        raise ValueError(f"Team not found in ELO ratings: {team_name}")
    return int(matches.iloc[0]["games_played"])


def _team_sos(team_name: str, sos: pd.DataFrame | None) -> tuple[float | None, int | None]:
    if sos is None or sos.empty:
        return None, None
    matches = _team_matches(team_name, sos)
    if matches.empty:
        return None, None
    return float(matches.iloc[0]["average_opponent_elo"]), int(matches.iloc[0]["sos_rank"])


def _read_optional_sos(sos_path: str | Path) -> pd.DataFrame | None:
    path = Path(sos_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_optional_power_v2(power_v2_path: str | Path | None) -> pd.DataFrame | None:
    if power_v2_path is None:
        return None
    path = Path(power_v2_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_sos(average_opponent_elo: float | None, sos_rank: int | None) -> str:
    if average_opponent_elo is None or sos_rank is None:
        return "Unavailable"
    return f"{average_opponent_elo:.1f} opponent ELO (rank {sos_rank})"


def _team_power_v2(team_name: str, power_v2: pd.DataFrame | None) -> tuple[float | None, int | None]:
    if power_v2 is None or power_v2.empty:
        return None, None
    matches = _team_matches(team_name, power_v2)
    if matches.empty:
        return None, None
    return float(matches.iloc[0]["power_rating_v2"]), int(matches.iloc[0]["power_rank_v2"])


def _format_power_v2_lines(prediction: MatchupPrediction) -> list[str]:
    if (
        prediction.team_a_power_v2 is None
        or prediction.team_b_power_v2 is None
        or prediction.team_a_power_rank_v2 is None
        or prediction.team_b_power_rank_v2 is None
    ):
        return []
    return [
        (
            f"{prediction.team_a} Power Rating: {prediction.team_a_power_v2:.2f} "
            f"(rank {prediction.team_a_power_rank_v2})"
        ),
        (
            f"{prediction.team_b} Power Rating: {prediction.team_b_power_v2:.2f} "
            f"(rank {prediction.team_b_power_rank_v2})"
        ),
    ]


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
    rows = _team_matches(team_name, team_games)
    if rows.empty:
        raise ValueError(f"Team not found in completed team-game data: {team_name}")
    return {
        "points_for": float(rows["points_for"].mean()),
        "points_against": float(rows["points_against"].mean()),
    }


def _resolve_from_frame(team_name: str, frame: pd.DataFrame) -> str:
    if frame.empty or "team" not in frame.columns:
        return team_name
    return resolve_team_name(team_name, frame["team"].dropna().unique())


def _team_matches(team_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "team" not in frame.columns:
        return pd.DataFrame()
    resolved = _resolve_from_frame(team_name, frame)
    return frame[frame["team"] == resolved]


def _projected_spread(team_a: str, team_b: str, projected_margin: float) -> str:
    if projected_margin > 0:
        return f"{team_a} by {abs(projected_margin):.1f}"
    if projected_margin < 0:
        return f"{team_b} by {abs(projected_margin):.1f}"
    return "Pick'em"


def _confidence_level(team_a_games_played: int, team_b_games_played: int) -> str:
    if team_a_games_played >= 5 and team_b_games_played >= 5:
        return "High"
    if team_a_games_played >= 3 and team_b_games_played >= 3:
        return "Medium"
    return "Low"


def _confidence_warning(confidence_level: str) -> str | None:
    if confidence_level == "Medium":
        return "Moderate sample size; treat this prediction as directional."
    if confidence_level == "Low":
        return "Small sample size; prediction may be unstable."
    return None
