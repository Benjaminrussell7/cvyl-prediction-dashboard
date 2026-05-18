from __future__ import annotations

import pandas as pd


def generate_matchup_explanation(
    team_a: str,
    team_b: str,
    *,
    predicted_winner: str,
    win_probability: float,
    confidence_tier: str,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
    sos: pd.DataFrame,
) -> str:
    if not predicted_winner:
        return "Explanation unavailable until both teams have enough model data."

    underdog = team_b if predicted_winner == team_a else team_a
    favorite_power = _team_row(power_ratings, predicted_winner)
    underdog_power = _team_row(power_ratings, underdog)
    favorite_trend = _team_row(trends, predicted_winner)
    underdog_trend = _team_row(trends, underdog)
    favorite_sos = _team_row(sos, predicted_winner)
    underdog_sos = _team_row(sos, underdog)

    power_diff = _power_rating_value(favorite_power) - _power_rating_value(underdog_power)
    offense_diff = _value(favorite_trend, "recent_offense_rating") - _value(
        underdog_trend,
        "recent_offense_rating",
    )
    defense_diff = _value(underdog_trend, "recent_defense_rating") - _value(
        favorite_trend,
        "recent_defense_rating",
    )
    momentum_diff = _value(favorite_trend, "momentum_score") - _value(
        underdog_trend,
        "momentum_score",
    )
    sos_diff = _value(favorite_sos, "average_opponent_elo") - _value(
        underdog_sos,
        "average_opponent_elo",
    )
    favorite_form = _label(favorite_trend, "momentum_label")
    underdog_form = _label(underdog_trend, "momentum_label")

    if abs(power_diff) < 0.4 and win_probability < 0.56:
        return (
            f"{team_a} and {team_b} rate very close overall, so this looks like a toss-up. "
            f"Small swings in shooting or goalie play could decide it."
        )

    if momentum_diff < -3 or (underdog_form in {"Surging", "Improving"} and win_probability < 0.65):
        return (
            f"{predicted_winner} is the model favorite, but {underdog} has the recent-form profile "
            f"to keep this close. Treat this as a matchup with upset potential."
        )

    if favorite_form == "Surging" and momentum_diff >= 2:
        return (
            f"{predicted_winner} gets the edge from a strong Power Rating and recent momentum. "
            f"The recent form points to a team that is playing better now than earlier in the season."
        )

    if defense_diff >= 2:
        return (
            f"{predicted_winner} has the edge partly because its recent defense has allowed fewer goals. "
            f"That defensive profile should help if the game pace gets tight."
        )

    if offense_diff >= 2:
        return (
            f"{predicted_winner} is favored behind the stronger recent scoring profile. "
            f"If that offense travels, the matchup tilts their way."
        )

    if sos_diff >= 50:
        return (
            f"{predicted_winner} has been tested by a tougher recent schedule and still grades ahead. "
            f"That schedule context supports the model edge."
        )

    return (
        f"{predicted_winner} is favored by the Power Rating, but the edge is not automatic. "
        f"Confidence is {confidence_tier.lower()}, so recent execution still matters."
    )


def _team_row(frame: pd.DataFrame, team: str) -> pd.Series | None:
    if frame.empty or "team" not in frame.columns:
        return None
    matches = frame[frame["team"].map(_normalize_team_name) == _normalize_team_name(team)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _normalize_team_name(team: object) -> str:
    return " ".join(str(team).strip().casefold().split())


def _value(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row or pd.isna(row[column]):
        return 0.0
    return float(row[column])


def _power_rating_value(row: pd.Series | None) -> float:
    for column in ["power_rating_v4", "power_rating_v3_recency"]:
        if row is not None and column in row and pd.notna(row[column]):
            return float(row[column])
    return 0.0


def _label(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return ""
    return str(row[column]).strip()
