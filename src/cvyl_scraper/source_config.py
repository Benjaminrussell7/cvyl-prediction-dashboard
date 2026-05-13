from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_DISCOVERED_SOURCES_CSV = "data/processed/discovered_sources.csv"
DEFAULT_DISCOVERED_SOURCES_YML = "config/discovered_sources.yml"
DEFAULT_DISCOVERED_SEASON = 2026
DEFAULT_DISCOVERED_DIVISION = "U12 Boys"


def generate_discovered_sources_config(
    input_path: str | Path = DEFAULT_DISCOVERED_SOURCES_CSV,
    output_path: str | Path = DEFAULT_DISCOVERED_SOURCES_YML,
    *,
    season: int = DEFAULT_DISCOVERED_SEASON,
    division: str = DEFAULT_DISCOVERED_DIVISION,
) -> Path:
    discovered = pd.read_csv(input_path)
    payload = discovered_sources_to_config(discovered, season=season, division=division)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return path


def discovered_sources_to_config(
    discovered_sources: pd.DataFrame,
    *,
    season: int = DEFAULT_DISCOVERED_SEASON,
    division: str = DEFAULT_DISCOVERED_DIVISION,
) -> dict[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for _, row in discovered_sources.iterrows():
        team_name = _required_text(row, "team_name")
        team_games_url = _required_text(row, "team_games_url")
        rows.append(
            {
                "name": safe_source_name(team_name),
                "url": team_games_url,
                "season": season,
                "division": division,
            }
        )
    return {"sources": rows}


def safe_source_name(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("Source name cannot be empty.")
    return normalized


def _required_text(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"Discovered source row must include '{column}'.")
    return str(value).strip()
