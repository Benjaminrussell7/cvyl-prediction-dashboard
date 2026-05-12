from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    season: int | None = None
    division: str | None = None
