from __future__ import annotations

from cvyl_scraper.team_matching import resolve_team_name


def test_resolve_team_name_maps_known_scheduled_variants() -> None:
    candidates = [
        "Berkshire",
        "Minnechaug 12U",
        "Somers 12U Red",
        "Somers 12U White",
    ]

    assert resolve_team_name("Berkshires", candidates) == "Berkshire"
    assert resolve_team_name("Minnechaug", candidates) == "Minnechaug 12U"
    assert resolve_team_name("Somers Red", candidates) == "Somers 12U Red"


def test_resolve_team_name_leaves_unresolved_non_team_labels_unchanged() -> None:
    assert (
        resolve_team_name("Paul Bowers Lacrosse Tournament", ["Minnechaug 12U"])
        == "Paul Bowers Lacrosse Tournament"
    )
