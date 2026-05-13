from __future__ import annotations

import re
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup


DISCOVERED_SOURCE_COLUMNS = ["team_name", "division", "team_games_url"]


def discover_team_sources(html: str, page_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for link in soup.find_all("a", href=True):
        team_name = _clean_text(link.get_text(" ", strip=True))
        team_url = _team_games_url(str(link["href"]), page_url)
        if not team_name or not team_url or team_url in seen_urls:
            continue

        division = _division_for_link(link)
        if not division or not is_boys_junior_division(division):
            continue

        records.append(
            {
                "team_name": team_name,
                "division": division,
                "team_games_url": team_url,
            }
        )
        seen_urls.add(team_url)

    if not records:
        return pd.DataFrame(columns=DISCOVERED_SOURCE_COLUMNS)

    return pd.DataFrame(records, columns=DISCOVERED_SOURCE_COLUMNS).sort_values(
        by=["division", "team_name", "team_games_url"],
        ignore_index=True,
    )


def is_boys_junior_division(value: object) -> bool:
    text = _normalized_words(value)
    if not text:
        return False
    if re.search(r"\b(girls?|female|womens?)\b", text):
        return False
    if re.search(r"\b(14u|u14|15u|u15|16u|u16|senior|high school|hs)\b", text):
        return False

    has_junior_a_or_b = bool(re.search(r"\bjunior\s+[ab]\b", text))
    has_u12 = bool(re.search(r"\b(12u|u12)\b", text))

    return has_junior_a_or_b or has_u12


def _team_games_url(href: str, page_url: str) -> str | None:
    absolute_url = urljoin(page_url, href)
    if not re.search(r"/team/|/teams/", absolute_url, re.IGNORECASE):
        return None
    if re.search(r"/games(?:[/?#]|$)", absolute_url, re.IGNORECASE):
        return absolute_url
    return absolute_url.rstrip("/") + "/games"


def _division_for_link(link) -> str | None:
    candidates: list[str] = []

    row = link.find_parent("tr")
    if row:
        cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        candidates.extend(cell for cell in cells if cell and cell != _clean_text(link.get_text(" ", strip=True)))

    section = link.find_parent(["section", "article", "div", "li", "tr"])
    if section:
        heading = section.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if heading:
            candidates.append(_clean_text(heading.get_text(" ", strip=True)))

    for parent in link.parents:
        heading = _previous_heading(parent)
        if heading:
            candidates.append(heading)
        aria_label = parent.get("aria-label") if hasattr(parent, "get") else None
        if aria_label:
            candidates.append(_clean_text(str(aria_label)))
        class_text = " ".join(parent.get("class", [])) if hasattr(parent, "get") else ""
        if class_text:
            candidates.append(_clean_text(class_text.replace("-", " ")))

    section_text = _section_text(link)
    if section_text:
        candidates.append(section_text)

    for candidate in candidates:
        if not _looks_like_division(candidate):
            continue
        if is_boys_junior_division(candidate):
            return _clean_text(candidate)
        return None
    return None


def _previous_heading(node) -> str | None:
    for sibling in node.find_all_previous(["h1", "h2", "h3", "h4", "h5", "h6"], limit=1):
        heading = _clean_text(sibling.get_text(" ", strip=True))
        if heading:
            return heading
    return None


def _section_text(link) -> str | None:
    parent = link.find_parent(["section", "article", "div", "li", "tr"])
    if not parent:
        return None
    return _clean_text(parent.get_text(" ", strip=True))


def _normalized_words(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("-", " "))


def _looks_like_division(value: object) -> bool:
    text = _normalized_words(value)
    return bool(
        re.search(
            r"\b(boys?|girls?|male|female|mens?|womens?|junior|12u|u12|14u|u14|senior)\b",
            text,
        )
    )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
