from __future__ import annotations

from pathlib import Path

from cvyl_scraper.club_branding import (
    ClubBranding,
    build_unresolved_branding_review,
    build_club_branding_review,
    discover_logo_for_club,
    enrich_discovered_clubs,
    extract_club_entries_from_html,
    extract_club_website_url,
    extract_logo_url_from_html,
    extract_logo_colors_from_pixels,
    is_girls_specific_club,
    load_club_branding_config,
    normalized_club_base,
    normalized_team_base,
    export_unresolved_branding_review,
    resolve_club_for_team,
    safe_logo_filename,
)


def _clubs() -> list[ClubBranding]:
    return [
        ClubBranding(
            club_name="Granby Youth Lacrosse",
            aliases=("Granby", "Granby 12U", "Granby 12U Blue"),
        ),
        ClubBranding(
            club_name="West Hartford Youth Lacrosse",
            aliases=("West Hartford", "West Hartford 12U Green", "West Hartford 12U Gold"),
        ),
        ClubBranding(
            club_name="Simsbury Youth Lacrosse",
            aliases=("Simsbury", "Simsbury 12U A", "Simsbury 12U Blue"),
        ),
    ]


def test_resolve_club_for_team_variants() -> None:
    clubs = _clubs()

    assert resolve_club_for_team("Granby 12U", clubs).club.club_name == "Granby Youth Lacrosse"
    assert resolve_club_for_team("Granby 12U Blue", clubs).club.club_name == "Granby Youth Lacrosse"
    assert resolve_club_for_team("West Hartford 12U Green", clubs).club.club_name == "West Hartford Youth Lacrosse"
    assert resolve_club_for_team("West Hartford 12U Gold", clubs).club.club_name == "West Hartford Youth Lacrosse"
    assert resolve_club_for_team("Simsbury 12U A", clubs).club.club_name == "Simsbury Youth Lacrosse"
    assert resolve_club_for_team("Simsbury 12U Blue", clubs).club.club_name == "Simsbury Youth Lacrosse"


def test_alias_handling_and_base_name_resolution() -> None:
    clubs = [ClubBranding("Farmington Youth Lacrosse", aliases=("Farmington",))]

    resolution = resolve_club_for_team("Farmington 12U White", clubs)

    assert resolution.club is not None
    assert resolution.club.club_name == "Farmington Youth Lacrosse"


def test_club_base_resolution_handles_discovered_full_club_names() -> None:
    clubs = [
        ClubBranding("Granby Youth Lacrosse", aliases=("Granby Youth Lacrosse",)),
        ClubBranding("Tolland Lacrosse Club", aliases=("Tolland Lacrosse Club",)),
        ClubBranding("Colchester Youth Lacrosse", aliases=("Colchester Youth Lacrosse",)),
        ClubBranding("West Hartford Youth Lacrosse", aliases=("West Hartford Youth Lacrosse",)),
    ]

    assert resolve_club_for_team("Granby 12U Blue", clubs).club.club_name == "Granby Youth Lacrosse"
    assert resolve_club_for_team("Tolland 12U", clubs).club.club_name == "Tolland Lacrosse Club"
    assert resolve_club_for_team("Colchester 12U", clubs).club.club_name == "Colchester Youth Lacrosse"
    assert resolve_club_for_team("West Hartford 12U Green", clubs).club.club_name == "West Hartford Youth Lacrosse"
    assert resolve_club_for_team("Glastonbury 12U Blue", [ClubBranding("Glastonbury Lacrosse Club", aliases=("Glastonbury Lacrosse Club",))]).club.club_name == "Glastonbury Lacrosse Club"


def test_normalized_club_base_preserves_town_and_removes_org_words() -> None:
    assert normalized_club_base("West Hartford Youth Lacrosse") == "west hartford"
    assert normalized_club_base("Tolland Lacrosse Club") == "tolland"
    assert normalized_club_base("Agawam Boy's Youth Lacrosse") == "agawam"
    assert normalized_team_base("Glastonbury 12U Blue") == "glastonbury"
    assert normalized_team_base("Simsbury 12U A") == "simsbury"
    assert normalized_team_base("West Hartford 12U Green") == "west hartford"


def test_girls_specific_clubs_are_filtered_from_boys_resolution() -> None:
    clubs = [
        ClubBranding("Simsbury Girls Youth Lacrosse", aliases=("Simsbury", "Simsbury 12U")),
        ClubBranding("Simsbury Youth Lacrosse", aliases=("Simsbury",)),
    ]

    resolution = resolve_club_for_team("Simsbury 12U Blue", clubs)

    assert resolution.club is not None
    assert resolution.club.club_name == "Simsbury Youth Lacrosse"
    assert is_girls_specific_club("Simsbury Girl’s Lacrosse")


def test_only_girls_specific_match_is_left_unresolved() -> None:
    clubs = [ClubBranding("Granby Girls Youth Lacrosse", aliases=("Granby", "Granby 12U"))]

    resolution = resolve_club_for_team("Granby 12U", clubs)

    assert resolution.club is None
    assert "girls-specific" in resolution.notes


def test_girls_club_does_not_match_through_base_name_fallback() -> None:
    clubs = [ClubBranding("Colchester Girls Youth Lacrosse", aliases=("Colchester Girls Youth Lacrosse",))]

    resolution = resolve_club_for_team("Colchester 12U", clubs)

    assert resolution.club is None
    assert "girls-specific" in resolution.notes


def test_requested_town_variants_resolve_to_expected_clubs() -> None:
    clubs = [
        ClubBranding("Glastonbury Lacrosse Club", aliases=("Glastonbury Lacrosse Club",)),
        ClubBranding("Granby Youth Lacrosse", aliases=("Granby Youth Lacrosse",)),
        ClubBranding("West Hartford Youth Lacrosse", aliases=("West Hartford Youth Lacrosse",)),
        ClubBranding("Simsbury Youth Lacrosse", aliases=("Simsbury Youth Lacrosse",)),
        ClubBranding("Tolland Lacrosse Club", aliases=("Tolland Lacrosse Club",)),
    ]

    assert resolve_club_for_team("Glastonbury 12U Blue", clubs).club.club_name == "Glastonbury Lacrosse Club"
    assert resolve_club_for_team("Granby 12U Blue", clubs).club.club_name == "Granby Youth Lacrosse"
    assert resolve_club_for_team("West Hartford 12U Green", clubs).club.club_name == "West Hartford Youth Lacrosse"
    assert resolve_club_for_team("Simsbury 12U A", clubs).club.club_name == "Simsbury Youth Lacrosse"
    assert resolve_club_for_team("Tolland 12U", clubs).club.club_name == "Tolland Lacrosse Club"


def test_unresolved_branding_review_lists_only_failed_team_matches() -> None:
    clubs = [ClubBranding("Colchester Youth Lacrosse", aliases=("Colchester Youth Lacrosse",))]

    unresolved = build_unresolved_branding_review(["Colchester 12U", "Unknown 12U"], clubs)

    assert unresolved["team"].tolist() == ["Unknown 12U"]
    assert unresolved.iloc[0]["normalized_team_base"] == "unknown"


def test_export_unresolved_branding_review_writes_only_unresolved(tmp_path: Path) -> None:
    clubs = [
        ClubBranding("Glastonbury Lacrosse Club", aliases=("Glastonbury Lacrosse Club",)),
        ClubBranding("West Hartford Youth Lacrosse", aliases=("West Hartford Youth Lacrosse",)),
    ]
    output = tmp_path / "cvyl_unresolved_branding.csv"

    export_unresolved_branding_review(["Glastonbury 12U Blue", "Unknown 12U"], clubs, output)

    content = output.read_text(encoding="utf-8")
    assert "Unknown 12U" in content
    assert "Glastonbury 12U Blue" not in content


def test_safe_logo_filename_is_deterministic() -> None:
    assert safe_logo_filename("West Hartford Youth Lacrosse", "https://cdn.example.com/logo.svg") == (
        "west_hartford_youth_lacrosse.svg"
    )
    assert safe_logo_filename("Granby Youth Lacrosse", "https://cdn.example.com/logo") == (
        "granby_youth_lacrosse.png"
    )


def test_extract_logo_colors_ignores_white_and_transparent_background() -> None:
    pixels = [
        (255, 255, 255, 255),
        (255, 255, 255, 255),
        (0, 0, 0, 0),
        (20, 20, 20, 255),
        (21, 110, 230, 255),
        (24, 112, 232, 255),
        (220, 170, 20, 255),
    ]

    primary, secondary, source, notes = extract_logo_colors_from_pixels(pixels)

    assert primary == "#0060e0"
    assert secondary == "#c0a000"
    assert source == "extracted_logo"
    assert notes == ""


def test_missing_logo_colors_are_graceful() -> None:
    assert extract_logo_colors_from_pixels([(255, 255, 255, 0)]) == (
        "",
        "",
        "",
        "No usable logo colors found.",
    )


def test_extract_club_entries_filters_girls_clubs_and_finds_logo() -> None:
    html = """
    <nav>
      <a href="/club/granby-youth-lacrosse/501"><img src="/logos/granby.png">Granby Youth Lacrosse</a>
      <a href="/club/granby-girls-youth-lacrosse/502"><img src="/logos/granby-girls.png">Granby Girls Youth Lacrosse</a>
      <a href="/club/simsbury-youth-lacrosse/503">Simsbury Youth Lacrosse</a>
    </nav>
    """

    clubs = extract_club_entries_from_html(html, "https://www.cvyl.org/clubs")

    assert [club.club_name for club in clubs] == ["Granby Youth Lacrosse", "Simsbury Youth Lacrosse"]
    assert clubs[0].logo_url == "https://www.cvyl.org/logos/granby.png"
    assert clubs[0].club_url == "https://www.cvyl.org/club/granby-youth-lacrosse/501"


def test_extract_club_website_and_logo_from_club_page() -> None:
    html = """
    <main>
      <a href="https://granbylax.org">Club Website</a>
      <img src="/assets/cvyl-logo.png" alt="CVYL logo">
      <img src="/assets/granby-club-logo.png" alt="Granby Lacrosse logo">
    </main>
    """

    assert extract_club_website_url(html, "https://www.cvyl.org/club/granby/1") == "https://granbylax.org"
    assert extract_logo_url_from_html(html, "https://www.cvyl.org/club/granby/1") == (
        "https://www.cvyl.org/assets/granby-club-logo.png"
    )


def test_discover_logo_for_club_uses_linked_club_website_when_cvyl_has_no_logo() -> None:
    pages = {
        "https://www.cvyl.org/club/granby/1": '<a href="https://granbylax.org">Club Website</a>',
        "https://granbylax.org": '<img src="/logo.png" alt="Granby Lacrosse Club logo">',
    }

    club = discover_logo_for_club(
        ClubBranding("Granby Youth Lacrosse", aliases=("Granby",), club_url="https://www.cvyl.org/club/granby/1"),
        fetcher=lambda url: pages[url],
    )

    assert club.logo_url == "https://granbylax.org/logo.png"
    assert "linked club website" in club.notes


def test_enrich_discovered_clubs_matches_existing_registry_and_filters_girls() -> None:
    existing = [ClubBranding("Granby Youth Lacrosse", aliases=("Granby",))]
    discovered = [
        ClubBranding(
            "Granby Youth Lacrosse",
            aliases=("Granby Youth Lacrosse",),
            club_url="https://www.cvyl.org/club/granby/1",
            logo_url="https://cdn.example.com/granby.png",
        ),
        ClubBranding(
            "Granby Girls Youth Lacrosse",
            aliases=("Granby",),
            club_url="https://www.cvyl.org/club/granby-girls/2",
            logo_url="https://cdn.example.com/girls.png",
        ),
    ]

    enriched = enrich_discovered_clubs(existing, discovered, cache_logos=False)

    assert [club.club_name for club in enriched] == ["Granby Youth Lacrosse"]
    assert enriched[0].logo_url == "https://cdn.example.com/granby.png"


def test_enrich_discovered_clubs_matches_generic_to_boys_specific_club() -> None:
    existing = [ClubBranding("West Hartford Youth Lacrosse", aliases=("West Hartford", "West Hartford 12U Green"))]
    discovered = [
        ClubBranding(
            "West Hartford Boys Youth Lacrosse",
            aliases=("West Hartford Boys Youth Lacrosse",),
            club_url="https://www.cvyl.org/club/west-hartford-boys/49",
            logo_url="https://cdn.example.com/weha-boys.png",
        ),
        ClubBranding(
            "West Hartford Girls Lacrosse",
            aliases=("West Hartford",),
            club_url="https://www.cvyl.org/club/west-hartford-girls/50",
            logo_url="https://cdn.example.com/weha-girls.png",
        ),
    ]

    enriched = enrich_discovered_clubs(existing, discovered, cache_logos=False)

    assert len(enriched) == 1
    assert enriched[0].club_name == "West Hartford Youth Lacrosse"
    assert enriched[0].logo_url == "https://cdn.example.com/weha-boys.png"


def test_build_club_branding_review_excludes_girls_entries() -> None:
    review = build_club_branding_review(
        [
            ClubBranding("Avon Youth Lacrosse", aliases=("Avon",), primary_color="#123456"),
            ClubBranding("Avon Girls Youth Lacrosse", aliases=("Avon",), primary_color="#abcdef"),
        ]
    )

    assert list(review["club_name"]) == ["Avon Youth Lacrosse"]


def test_load_club_branding_config(tmp_path: Path) -> None:
    path = tmp_path / "club_branding.yml"
    path.write_text(
        """
clubs:
  - club_name: West Hartford Youth Lacrosse
    aliases:
      - West Hartford
    logo_url: https://example.com/wh.png
    logo_path: assets/logos/wh.png
    primary_color: "#0055aa"
    secondary_color: "#facc15"
    color_source: manual
    notes: reviewed
""",
        encoding="utf-8",
    )

    clubs = load_club_branding_config(path)

    assert clubs == [
        ClubBranding(
            club_name="West Hartford Youth Lacrosse",
            aliases=("West Hartford",),
            logo_url="https://example.com/wh.png",
            logo_path="assets/logos/wh.png",
            primary_color="#0055aa",
            secondary_color="#facc15",
            color_source="manual",
            notes="reviewed",
        )
    ]
