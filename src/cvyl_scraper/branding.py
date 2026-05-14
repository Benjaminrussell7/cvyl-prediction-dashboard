from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from cvyl_scraper.config import load_sources
from cvyl_scraper.export import export_csv
from cvyl_scraper.scraping import FetchedPage, fetch_page


TEAM_BRANDING_COLUMNS = [
    "team",
    "team_url",
    "logo_url",
    "primary_color",
    "primary_color_source",
]

TEAM_COLOR_TOKENS = {
    "red": "#dc2626",
    "blue": "#2563eb",
    "gold": "#d97706",
    "green": "#16a34a",
    "black": "#111827",
    "white": "#ffffff",
}


@dataclass(frozen=True)
class BrandingSource:
    team: str
    url: str


def discover_team_branding(
    sources_input: str | Path = "config/discovered_sources.yml",
    output_path: str | Path = "data/processed/cvyl_team_branding.csv",
    *,
    fetcher: Callable[[str], FetchedPage] = fetch_page,
) -> Path:
    branding = build_team_branding(load_branding_sources(sources_input), fetcher=fetcher)
    return export_csv(branding, output_path)


def build_team_branding(
    sources: list[BrandingSource],
    *,
    fetcher: Callable[[str], FetchedPage] = fetch_page,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for source in sources:
        team_url = team_page_url(source.url)
        page = fetcher(team_url)
        rows.append(extract_team_branding(page.html, page.url, fallback_team=source.team))

    if not rows:
        return pd.DataFrame(columns=TEAM_BRANDING_COLUMNS)

    return pd.DataFrame(rows, columns=TEAM_BRANDING_COLUMNS).sort_values(
        ["team", "team_url"],
        ignore_index=True,
    )


def load_branding_sources(path: str | Path) -> list[BrandingSource]:
    source_path = Path(path)
    if source_path.suffix.lower() in {".yml", ".yaml"}:
        return [
            BrandingSource(team=source.name, url=source.url)
            for source in load_sources(source_path)
        ]

    discovered = pd.read_csv(source_path)
    required = {"team_name", "team_games_url"}
    missing = required - set(discovered.columns)
    if missing:
        raise ValueError(f"Branding source CSV missing columns: {sorted(missing)}")

    return [
        BrandingSource(team=str(row["team_name"]).strip(), url=str(row["team_games_url"]).strip())
        for _, row in discovered.iterrows()
        if str(row["team_name"]).strip() and str(row["team_games_url"]).strip()
    ]


def extract_team_branding(html: str, page_url: str, *, fallback_team: str = "") -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    team = extract_team_name(soup, fallback_team=fallback_team)
    primary_color, primary_color_source = infer_primary_color(team)
    return {
        "team": team,
        "team_url": team_page_url(page_url),
        "logo_url": extract_logo_url(soup, page_url) or "",
        "primary_color": primary_color,
        "primary_color_source": primary_color_source,
    }


def extract_team_name(soup: BeautifulSoup, *, fallback_team: str = "") -> str:
    for selector in [
        '[class*="team"][class*="name"]',
        '[class*="team-title"]',
        "h1",
        "h2",
    ]:
        node = soup.select_one(selector)
        if node:
            text = clean_team_name(node.get_text(" ", strip=True))
            if text:
                return text

    for attrs in [
        {"property": "og:title"},
        {"name": "twitter:title"},
    ]:
        node = soup.find("meta", attrs=attrs)
        if node and node.get("content"):
            text = clean_team_name(str(node["content"]))
            if text:
                return text

    title = soup.find("title")
    if title:
        text = clean_team_name(title.get_text(" ", strip=True))
        if text:
            return text

    return fallback_team.strip()


def extract_logo_url(soup: BeautifulSoup, page_url: str) -> str | None:
    for img in sorted(soup.find_all("img", src=True), key=logo_candidate_score, reverse=True):
        if logo_candidate_score(img) <= 0:
            continue
        return urljoin(page_url, str(img["src"]))

    for attrs in [
        {"property": "og:image"},
        {"name": "twitter:image"},
    ]:
        node = soup.find("meta", attrs=attrs)
        if node and node.get("content"):
            return urljoin(page_url, str(node["content"]))
    return None


def logo_candidate_score(img) -> int:
    text = " ".join(
        [
            str(img.get("src", "")),
            str(img.get("alt", "")),
            " ".join(img.get("class", [])),
            str(img.get("id", "")),
        ]
    ).lower()
    if re.search(r"tracking|pixel|spacer|avatar-placeholder", text):
        return -1
    score = 0
    if "logo" in text:
        score += 5
    if "crest" in text:
        score += 4
    if "team" in text:
        score += 3
    if "org" in text or "club" in text:
        score += 1
    return score


def infer_primary_color(team_name: object) -> tuple[str, str]:
    words = re.findall(r"[A-Za-z]+", str(team_name).lower())
    matches = [word for word in words if word in TEAM_COLOR_TOKENS]
    if len(matches) != 1:
        return "", ""
    token = matches[0]
    return TEAM_COLOR_TOKENS[token], f"team_name_token:{token.title()}"


def team_page_url(url: str) -> str:
    return re.sub(r"/games/?(?:[?#].*)?$", "", str(url).strip())


def clean_team_name(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = re.split(r"\s+[|–-]\s+(?:CVYL|Crossbar|Connecticut Valley Youth Lacrosse)\b", text)[0]
    text = re.sub(r"\s+Games$", "", text, flags=re.IGNORECASE)
    return text.strip()
