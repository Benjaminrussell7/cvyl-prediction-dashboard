from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.export import export_csv


DEFAULT_SOS_CSV = "data/processed/cvyl_sos.csv"

SOS_COLUMNS = [
    "team",
    "games_played",
    "average_opponent_elo",
    "opponent_count",
    "sos_rank",
]


def build_sos(team_games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame(columns=SOS_COLUMNS)

    completed = team_games[team_games["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(columns=SOS_COLUMNS)

    rating_lookup = ratings[["team", "elo"]].rename(
        columns={"team": "opponent", "elo": "opponent_elo"}
    )
    completed = completed.merge(rating_lookup, on="opponent", how="left")
    completed = completed.dropna(subset=["team", "opponent", "opponent_elo"])

    sos = (
        completed.groupby("team", as_index=False)
        .agg(
            games_played=("opponent_elo", "size"),
            average_opponent_elo=("opponent_elo", "mean"),
            opponent_count=("opponent", "nunique"),
        )
        .sort_values(
            by=["average_opponent_elo", "team"],
            ascending=[False, True],
            ignore_index=True,
        )
    )
    sos["sos_rank"] = range(1, len(sos) + 1)
    return sos[SOS_COLUMNS]


def export_sos(
    team_games: pd.DataFrame,
    ratings: pd.DataFrame,
    output_path: str | Path = DEFAULT_SOS_CSV,
) -> Path:
    return export_csv(build_sos(team_games, ratings), output_path)
