from __future__ import annotations

import pytest

from cvyl_scraper.config import load_team_aliases


def test_load_team_aliases_reads_explicit_mapping(tmp_path) -> None:
    path = tmp_path / "team_aliases.yml"
    path.write_text(
        """
aliases:
  Granby: Granby 12U
  "Somers Juniors": "Somers 12U"
""",
        encoding="utf-8",
    )

    assert load_team_aliases(path) == {
        "Granby": "Granby 12U",
        "Somers Juniors": "Somers 12U",
    }


def test_load_team_aliases_returns_empty_mapping_for_missing_file(tmp_path) -> None:
    assert load_team_aliases(tmp_path / "missing.yml") == {}


def test_load_team_aliases_requires_aliases_mapping(tmp_path) -> None:
    path = tmp_path / "team_aliases.yml"
    path.write_text("aliases:\n  - Granby\n", encoding="utf-8")

    with pytest.raises(ValueError, match="aliases"):
        load_team_aliases(path)
