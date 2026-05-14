from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.branding import (
    BrandingSource,
    build_team_branding,
    extract_team_branding,
    infer_primary_color,
    load_branding_sources,
    team_page_url,
)
from cvyl_scraper.scraping import FetchedPage


def test_extract_team_branding_finds_logo_and_color() -> None:
    html = """
    <html>
      <head><title>West Hartford 12U Green | CVYL</title></head>
      <body>
        <h1>West Hartford 12U Green</h1>
        <img class="team-logo" alt="West Hartford logo" src="/uploads/wh-green.png">
      </body>
    </html>
    """

    branding = extract_team_branding(
        html,
        "https://www.cvyl.org/team/225939/games",
        fallback_team="west_hartford_12u_green",
    )

    assert branding == {
        "team": "West Hartford 12U Green",
        "team_url": "https://www.cvyl.org/team/225939",
        "logo_url": "https://www.cvyl.org/uploads/wh-green.png",
        "primary_color": "#16a34a",
        "primary_color_source": "team_name_token:Green",
    }


def test_extract_team_branding_uses_meta_image_when_no_logo_img() -> None:
    html = """
    <html>
      <head>
        <meta property="og:title" content="Farmington 12U Red - CVYL">
        <meta property="og:image" content="https://cdn.example.com/farmington.png">
      </head>
      <body><img src="/tracking-pixel.gif"></body>
    </html>
    """

    branding = extract_team_branding(html, "https://www.cvyl.org/team/225929")

    assert branding["team"] == "Farmington 12U Red"
    assert branding["logo_url"] == "https://cdn.example.com/farmington.png"
    assert branding["primary_color"] == "#dc2626"


def test_infer_primary_color_only_when_single_clear_token() -> None:
    assert infer_primary_color("Glastonbury 12U Blue") == (
        "#2563eb",
        "team_name_token:Blue",
    )
    assert infer_primary_color("Westfield 12U Black") == (
        "#111827",
        "team_name_token:Black",
    )
    assert infer_primary_color("Town Red Blue") == ("", "")
    assert infer_primary_color("Granby 12U") == ("", "")


def test_load_branding_sources_from_discovered_csv(tmp_path: Path) -> None:
    path = tmp_path / "discovered_sources.csv"
    pd.DataFrame(
        [
            {
                "team_name": "Avon 12U B",
                "division": "Avon 12U B",
                "team_games_url": "https://www.cvyl.org/team/225869/games",
            }
        ]
    ).to_csv(path, index=False)

    sources = load_branding_sources(path)

    assert sources == [
        BrandingSource(
            team="Avon 12U B",
            url="https://www.cvyl.org/team/225869/games",
        )
    ]


def test_build_team_branding_is_deterministic() -> None:
    pages = {
        "https://www.cvyl.org/team/1": """
            <h1>Team Gold</h1>
            <img alt="Team Gold logo" src="/gold.png">
        """,
        "https://www.cvyl.org/team/2": """
            <h1>Team Blue</h1>
            <img alt="Team Blue crest" src="/blue.png">
        """,
    }

    def fake_fetcher(url: str) -> FetchedPage:
        return FetchedPage(url=url, html=pages[url])

    branding = build_team_branding(
        [
            BrandingSource("Team Blue", "https://www.cvyl.org/team/2/games"),
            BrandingSource("Team Gold", "https://www.cvyl.org/team/1/games"),
        ],
        fetcher=fake_fetcher,
    )

    assert list(branding["team"]) == ["Team Blue", "Team Gold"]
    assert list(branding["logo_url"]) == [
        "https://www.cvyl.org/blue.png",
        "https://www.cvyl.org/gold.png",
    ]


def test_team_page_url_strips_games_suffix() -> None:
    assert team_page_url("https://www.cvyl.org/team/225939/games") == "https://www.cvyl.org/team/225939"
    assert team_page_url("https://www.cvyl.org/team/225939") == "https://www.cvyl.org/team/225939"
