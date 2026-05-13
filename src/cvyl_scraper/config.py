from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cvyl_scraper.models import Source


def load_sources(path: str | Path) -> list[Source]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = yaml.safe_load(file) or {}

    rows = payload.get("sources", [])
    if not isinstance(rows, list):
        raise ValueError("Config must contain a 'sources' list.")

    sources: list[Source] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Source #{index} must be a mapping.")
        if not row.get("name") or not row.get("url"):
            raise ValueError(f"Source #{index} must include 'name' and 'url'.")

        sources.append(
            Source(
                name=str(row["name"]),
                url=str(row["url"]),
                season=_optional_int(row.get("season")),
                division=_optional_str(row.get("division")),
            )
        )

    return sources


def load_team_aliases(path: str | Path) -> dict[str, str]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = yaml.safe_load(file) or {}

    rows = payload.get("aliases", {})
    if not isinstance(rows, dict):
        raise ValueError("Team aliases config must contain an 'aliases' mapping.")

    aliases: dict[str, str] = {}
    for raw_name, canonical_name in rows.items():
        if raw_name in (None, "") or canonical_name in (None, ""):
            raise ValueError("Team alias entries must include non-empty source and target names.")
        aliases[str(raw_name)] = str(canonical_name)

    return aliases


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
