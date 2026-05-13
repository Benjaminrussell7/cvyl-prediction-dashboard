from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


DEFAULT_CANONICAL_GAMES_CSV = "data/processed/cvyl_games.csv"
DEFAULT_TEAM_IDENTITY_AUDIT_CSV = "data/processed/team_identity_audit.csv"

TEAM_IDENTITY_AUDIT_COLUMNS = [
    "team_name",
    "games_played",
    "completed_games",
    "scheduled_games",
    "source_count",
    "source_names",
    "opponent_count",
    "opponents",
    "appears_as_source_team",
    "has_12u_suffix",
    "possible_duplicate_group",
]

TEAM_VARIANT_TERMS = {
    "a",
    "b",
    "black",
    "blue",
    "gold",
    "green",
    "grey",
    "gray",
    "red",
    "white",
}


def export_team_identity_audit(
    input_path: str | Path = DEFAULT_CANONICAL_GAMES_CSV,
    output_path: str | Path = DEFAULT_TEAM_IDENTITY_AUDIT_CSV,
) -> Path:
    games = pd.read_csv(input_path)
    audit = build_team_identity_audit(games)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False)
    return path


def build_team_identity_audit(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=TEAM_IDENTITY_AUDIT_COLUMNS)

    appearances = pd.concat(
        [_team_appearances(games, "home_team", "away_team"), _team_appearances(games, "away_team", "home_team")],
        ignore_index=True,
    )
    appearances = appearances.dropna(subset=["team_name"])
    appearances["team_name"] = appearances["team_name"].map(_clean_text)
    appearances["opponent"] = appearances["opponent"].map(_clean_text)
    appearances = appearances[appearances["team_name"] != ""]

    rows: list[dict[str, object]] = []
    for team_name, team_rows in appearances.groupby("team_name", sort=True):
        source_names = _sorted_unique(team_rows["source_name"])
        opponents = _sorted_unique(team_rows["opponent"])
        rows.append(
            {
                "team_name": team_name,
                "games_played": int(team_rows["game_id"].nunique()),
                "completed_games": int((team_rows["status"] == "completed").sum()),
                "scheduled_games": int((team_rows["status"] == "scheduled").sum()),
                "source_count": len(source_names),
                "source_names": "; ".join(source_names),
                "opponent_count": len(opponents),
                "opponents": "; ".join(opponents),
                "appears_as_source_team": _appears_as_source_team(team_name, source_names),
                "has_12u_suffix": _has_12u_suffix(team_name),
                "possible_duplicate_group": possible_duplicate_group(team_name),
            }
        )

    return pd.DataFrame(rows, columns=TEAM_IDENTITY_AUDIT_COLUMNS)


def possible_duplicate_group(team_name: object) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", str(team_name).lower()).split()
    base_words = [
        word
        for word in words
        if word not in {"12u", "u12", "junior", "juniors"} and word not in TEAM_VARIANT_TERMS
    ]
    if not base_words:
        base_words = words
    return "_".join(base_words)


def _team_appearances(games: pd.DataFrame, team_column: str, opponent_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_name": games[team_column],
            "opponent": games[opponent_column],
            "game_id": games["game_id"],
            "status": games["status"],
            "source_name": games["source_name"],
        }
    )


def _appears_as_source_team(team_name: str, source_names: list[str]) -> bool:
    safe_team_name = _safe_name(team_name)
    return safe_team_name in {_safe_name(source_name) for source_name in source_names}


def _has_12u_suffix(team_name: str) -> bool:
    return bool(re.search(r"\b(12u|u12)\b", team_name, re.IGNORECASE))


def _sorted_unique(values: pd.Series) -> list[str]:
    cleaned = {_clean_text(value) for value in values if pd.notna(value) and _clean_text(value)}
    return sorted(cleaned)


def _safe_name(value: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
