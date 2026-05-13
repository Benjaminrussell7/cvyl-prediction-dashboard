from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.export import export_csv
from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency


DEFAULT_TRENDS_CSV = "data/processed/cvyl_trends.csv"

TREND_COLUMNS = [
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
]


def build_trends(team_games: pd.DataFrame) -> pd.DataFrame:
    completed = _completed_team_games(team_games)
    if completed.empty:
        return pd.DataFrame(columns=TREND_COLUMNS)

    current_power = _safe_power_ratings(completed)
    current_ranks = _current_power_ranks(completed, current_power)
    prior_ranks = _prior_power_ranks(completed, current_power)
    rows = []
    for team, group in completed.groupby("team", sort=True):
        ordered = group.sort_values(["game_date", "game_id"], kind="mergesort")
        recent = ordered.tail(5)
        last_3 = ordered.tail(3)
        last_5_win_pct = float(recent["win"].astype(float).mean())
        last_3_win_pct = float(last_3["win"].astype(float).mean())
        recent_avg_margin = float((recent["points_for"] - recent["points_against"]).mean())
        recent_offense = float(recent["points_for"].mean())
        recent_defense = float(recent["points_against"].mean())
        movement = prior_ranks.get(team, current_ranks.get(team, 0)) - current_ranks.get(team, 0)
        momentum_score = _momentum_score(
            last_5_win_pct,
            recent_avg_margin,
            recent_offense,
            recent_defense,
            movement,
        )
        rows.append(
            {
                "team": team,
                "games_played": int(len(ordered)),
                "last_3_win_pct": last_3_win_pct,
                "last_5_win_pct": last_5_win_pct,
                "recent_avg_margin": recent_avg_margin,
                "recent_offense_rating": recent_offense,
                "recent_defense_rating": recent_defense,
                "current_power_rank": current_ranks.get(team),
                "prior_power_rank": prior_ranks.get(team),
                "power_rank_movement": movement,
                "momentum_score": momentum_score,
                "momentum_label": _momentum_label(momentum_score),
            }
        )

    trends = pd.DataFrame(rows, columns=TREND_COLUMNS)
    return trends.sort_values(
        ["momentum_score", "team"],
        ascending=[False, True],
        ignore_index=True,
    )


def export_trends(
    team_games: pd.DataFrame,
    output_path: str | Path = DEFAULT_TRENDS_CSV,
) -> Path:
    return export_csv(build_trends(team_games), output_path)


def _completed_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame()
    completed = team_games[team_games["status"] == "completed"].copy()
    completed["game_date"] = pd.to_datetime(completed["game_date"], errors="coerce")
    completed["points_for"] = pd.to_numeric(completed["points_for"], errors="coerce")
    completed["points_against"] = pd.to_numeric(completed["points_against"], errors="coerce")
    completed["win"] = completed["win"].astype(bool)
    return completed.dropna(subset=["team", "game_date", "points_for", "points_against"])


def _prior_power_ranks(completed: pd.DataFrame, current_power: pd.DataFrame) -> dict[str, int]:
    if current_power.empty:
        return _alphabetical_ranks(completed["team"].dropna().unique())

    current = current_power[["team", "power_rating_v3_recency"]].copy()
    prior_ratings: dict[str, float] = dict(zip(current["team"], current["power_rating_v3_recency"]))
    for team, group in completed.groupby("team", sort=True):
        ordered = group.sort_values(["game_date", "game_id"], kind="mergesort")
        if len(ordered) <= 1:
            continue
        latest_index = ordered.tail(1).index
        prior_rows = pd.concat([completed[completed["team"] != team], ordered.drop(latest_index)])
        prior_power = _safe_power_ratings(prior_rows)
        match = prior_power[prior_power["team"] == team]
        if not match.empty:
            prior_ratings[team] = float(match.iloc[0]["power_rating_v3_recency"])

    prior = pd.DataFrame(
        [{"team": team, "prior_power_rating": rating} for team, rating in prior_ratings.items()]
    ).sort_values(["prior_power_rating", "team"], ascending=[False, True], ignore_index=True)
    prior["prior_power_rank"] = range(1, len(prior) + 1)
    return {row["team"]: int(row["prior_power_rank"]) for _, row in prior.iterrows()}


def _safe_power_ratings(completed: pd.DataFrame) -> pd.DataFrame:
    try:
        return build_power_ratings_v3_recency(completed)
    except KeyError:
        return pd.DataFrame()


def _current_power_ranks(completed: pd.DataFrame, current_power: pd.DataFrame) -> dict[str, int]:
    if current_power.empty:
        return _alphabetical_ranks(completed["team"].dropna().unique())
    return {
        row["team"]: int(row["power_rank_v3_recency"]) for _, row in current_power.iterrows()
    }


def _alphabetical_ranks(teams) -> dict[str, int]:
    return {team: index for index, team in enumerate(sorted(map(str, teams)), start=1)}


def _momentum_score(
    last_5_win_pct: float,
    recent_avg_margin: float,
    recent_offense: float,
    recent_defense: float,
    rank_movement: int,
) -> float:
    return float(
        ((last_5_win_pct - 0.5) * 10.0)
        + recent_avg_margin
        + ((recent_offense - recent_defense) * 0.25)
        + (rank_movement * 0.2)
    )


def _momentum_label(score: float) -> str:
    if score >= 4:
        return "Surging"
    if score >= 1:
        return "Improving"
    if score <= -3:
        return "Cooling"
    return "Steady"
