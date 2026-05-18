from __future__ import annotations

import re
from collections.abc import Iterable


TEAM_NAME_ALIASES = {
    "berkshires": "Berkshire",
    "minnechaug": "Minnechaug 12U",
    "somers red": "Somers 12U Red",
}


def normalize_team_key(team_name: object) -> str:
    return re.sub(r"\s+", " ", str(team_name or "").strip().casefold())


def resolve_team_name(team_name: object, candidates: Iterable[object]) -> str:
    original = str(team_name or "").strip()
    candidate_names = [str(candidate).strip() for candidate in candidates if str(candidate).strip()]
    if original in candidate_names:
        return original

    candidate_by_key = {normalize_team_key(candidate): candidate for candidate in candidate_names}
    normalized = normalize_team_key(original)
    if normalized in candidate_by_key:
        return candidate_by_key[normalized]

    alias = TEAM_NAME_ALIASES.get(normalized)
    if alias is not None and normalize_team_key(alias) in candidate_by_key:
        return candidate_by_key[normalize_team_key(alias)]

    return original
