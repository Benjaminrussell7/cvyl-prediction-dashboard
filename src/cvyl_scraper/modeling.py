from __future__ import annotations

import pandas as pd


TEAM_GAME_COLUMNS = [
    "game_id",
    "team",
    "opponent",
    "points_for",
    "points_against",
    "win",
    "game_date",
    "season",
    "division",
    "status",
]


def build_team_games(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=TEAM_GAME_COLUMNS)

    completed = games[games["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(columns=TEAM_GAME_COLUMNS)

    completed["game_date"] = pd.to_datetime(completed["game_date"], errors="coerce")
    completed = completed.dropna(
        subset=["game_id", "game_date", "home_team", "away_team", "home_score", "away_score"]
    )

    home_rows = pd.DataFrame(
        {
            "game_id": completed["game_id"],
            "team": completed["home_team"],
            "opponent": completed["away_team"],
            "points_for": completed["home_score"],
            "points_against": completed["away_score"],
            "game_date": completed["game_date"],
            "season": completed["season"],
            "division": completed["division"],
            "status": completed["status"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "game_id": completed["game_id"],
            "team": completed["away_team"],
            "opponent": completed["home_team"],
            "points_for": completed["away_score"],
            "points_against": completed["home_score"],
            "game_date": completed["game_date"],
            "season": completed["season"],
            "division": completed["division"],
            "status": completed["status"],
        }
    )

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games["points_for"] = pd.to_numeric(team_games["points_for"], errors="coerce").astype(
        "Int64"
    )
    team_games["points_against"] = pd.to_numeric(
        team_games["points_against"], errors="coerce"
    ).astype("Int64")
    team_games["win"] = team_games["points_for"] > team_games["points_against"]

    team_games = team_games.sort_values(
        by=["game_date", "game_id", "team"],
        ascending=[True, True, True],
        na_position="last",
    )
    team_games["game_date"] = team_games["game_date"].dt.date.astype(str)

    return team_games[TEAM_GAME_COLUMNS].reset_index(drop=True)
