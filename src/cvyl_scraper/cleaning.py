from __future__ import annotations

import hashlib
import re

import pandas as pd


GAME_COLUMNS = [
    "game_date",
    "game_time",
    "season",
    "division",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "status",
    "source_name",
    "source_url",
    "game_id",
]


def build_canonical_games(raw_games: pd.DataFrame) -> pd.DataFrame:
    if raw_games.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)

    games = raw_games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    games["game_time"] = games["game_time"].map(_clean_optional_text)
    games["division"] = games["division"].map(_clean_optional_text)
    games["source_name"] = games["source_name"].map(_clean_optional_text)
    games["source_url"] = games["source_url"].map(_clean_optional_text)
    games["home_team"] = games["home_team"].map(normalize_team_name)
    games["away_team"] = games["away_team"].map(normalize_team_name)
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce").astype("Int64")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce").astype("Int64")
    games["status"] = games.apply(_status_for_game, axis=1)
    games["status_priority"] = games["status"].map({"completed": 0, "scheduled": 1})

    games = games.dropna(subset=["game_date", "home_team", "away_team"])
    games = games[games["home_team"] != games["away_team"]]
    games["dedupe_key"] = games.apply(_game_key, axis=1)
    games = games.sort_values(
        by=["status_priority", "game_date", "game_time", "source_name"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    games = games.drop_duplicates(subset=["dedupe_key"], keep="first")
    games["game_id"] = games["dedupe_key"].map(_stable_id)

    return games[GAME_COLUMNS].sort_values(
        by=["game_date", "game_time", "home_team", "away_team"],
        na_position="last",
    )


def split_by_status(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = games[games["status"] == "completed"].copy()
    scheduled = games[games["status"] == "scheduled"].copy()
    return completed, scheduled


def normalize_team_name(value: object) -> str | None:
    cleaned = _clean_optional_text(value)
    if cleaned is None:
        return None
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _status_for_game(row: pd.Series) -> str:
    if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
        return "completed"
    return "scheduled"


def _game_key(row: pd.Series) -> str:
    teams = sorted([str(row["home_team"]).lower(), str(row["away_team"]).lower()])
    return "|".join(
        [
            f"date:{row['game_date']}",
            f"time:{row['game_time'] or ''}",
            f"season:{_clean_optional_text(row['season']) or ''}",
            f"division:{str(row['division'] or '').lower()}",
            f"teams:{teams[0]}@{teams[1]}",
        ]
    )


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _clean_optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or None
