from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from cvyl_scraper.export import export_csv


CLUB_BRANDING_COLUMNS = [
    "club_name",
    "aliases",
    "logo_url",
    "logo_path",
    "primary_color",
    "secondary_color",
    "color_source",
    "notes",
]

GIRLS_PROGRAM_PATTERN = re.compile(
    r"\b(girls?|girl['’]?s|girls lacrosse|girls youth lacrosse)\b",
    flags=re.IGNORECASE,
)
BOYS_PROGRAM_PATTERN = re.compile(r"\b(boys?|boys youth|youth lacrosse|lacrosse club)\b", flags=re.IGNORECASE)
AGE_AND_TEAM_TOKENS = {
    "u12",
    "12u",
    "u10",
    "10u",
    "u14",
    "14u",
    "boys",
    "boy",
    "a",
    "b",
    "blue",
    "gold",
    "green",
    "red",
    "black",
    "white",
}
CLUB_ORGANIZATION_TOKENS = {
    "youth",
    "lacrosse",
    "club",
    "association",
    "program",
    "select",
    "travel",
    "academy",
    "squad",
    "division",
    "team",
    "boys",
    "boy",
    "girls",
    "girl",
    "s",
    "juniors",
    "junior",
}


@dataclass(frozen=True)
class ClubBranding:
    club_name: str
    aliases: tuple[str, ...]
    club_url: str = ""
    logo_url: str = ""
    logo_path: str = ""
    primary_color: str = ""
    secondary_color: str = ""
    color_source: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ClubResolution:
    team_name: str
    club: ClubBranding | None
    notes: str = ""


def export_club_branding_review(
    config_path: str | Path = "config/club_branding.yml",
    output_path: str | Path = "data/processed/cvyl_club_branding.csv",
) -> Path:
    return export_csv(build_club_branding_review(load_club_branding_config(config_path)), output_path)


def discover_club_branding(
    clubs_page_url: str = "https://www.cvyl.org/",
    config_path: str | Path = "config/club_branding.yml",
    output_path: str | Path = "data/processed/cvyl_club_branding.csv",
    *,
    assets_dir: str | Path = "assets/logos",
    cache_logos: bool = True,
    fetcher: Callable[[str], str] | None = None,
    downloader: Callable[[str], bytes] | None = None,
) -> Path:
    fetcher = fetcher or fetch_html
    existing = load_club_branding_config(config_path)
    discovered = extract_club_entries_from_html(fetcher(clubs_page_url), clubs_page_url)
    enriched = enrich_discovered_clubs(
        existing,
        discovered,
        fetcher=fetcher,
        downloader=downloader,
        cache_logos=cache_logos,
        assets_dir=assets_dir,
    )
    return export_csv(build_club_branding_review(enriched), output_path)


def build_club_branding_review(clubs: list[ClubBranding]) -> pd.DataFrame:
    rows = [
        {
            "club_name": club.club_name,
            "aliases": "; ".join(club.aliases),
            "logo_url": club.logo_url,
            "logo_path": club.logo_path,
            "primary_color": club.primary_color,
            "secondary_color": club.secondary_color,
            "color_source": club.color_source,
            "notes": club.notes,
        }
        for club in clubs
        if not is_girls_specific_club(club.club_name)
    ]
    if not rows:
        return pd.DataFrame(columns=CLUB_BRANDING_COLUMNS)
    return pd.DataFrame(rows, columns=CLUB_BRANDING_COLUMNS).sort_values("club_name", ignore_index=True)


def load_club_branding_config(path: str | Path) -> list[ClubBranding]:
    config_path = Path(path)
    if not config_path.exists():
        return []
    with config_path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    entries = payload.get("clubs", payload if isinstance(payload, list) else [])
    clubs: list[ClubBranding] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        club_name = str(entry.get("club_name", "")).strip()
        aliases = tuple(str(alias).strip() for alias in entry.get("aliases", []) or [] if str(alias).strip())
        if not club_name:
            continue
        clubs.append(
            ClubBranding(
                club_name=club_name,
                aliases=aliases,
                club_url=str(entry.get("club_url", "") or "").strip(),
                logo_url=str(entry.get("logo_url", "") or "").strip(),
                logo_path=str(entry.get("logo_path", "") or "").strip(),
                primary_color=str(entry.get("primary_color", "") or "").strip(),
                secondary_color=str(entry.get("secondary_color", "") or "").strip(),
                color_source=str(entry.get("color_source", "") or "").strip(),
                notes=str(entry.get("notes", "") or "").strip(),
            )
        )
    return clubs


def load_club_branding_registry(
    *,
    csv_path: str | Path = "data/processed/cvyl_club_branding.csv",
    config_path: str | Path = "config/club_branding.yml",
) -> list[ClubBranding]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return []
    frame = pd.read_csv(csv_file)
    return club_branding_records_from_frame(frame)


def resolve_club_for_team(team_name: str, clubs: list[ClubBranding]) -> ClubResolution:
    normalized_team = normalize_branding_name(team_name)
    team_base = normalized_team_base(team_name)
    candidates = [club for club in clubs if not is_girls_specific_club(club.club_name)]
    girls_only_matches = [
        club for club in clubs if is_girls_specific_club(club.club_name) and club_matches_team(club, normalized_team, team_base)
    ]

    exact = scored_branding_matches(team_name, candidates)
    if exact:
        return ClubResolution(team_name=team_name, club=exact[0][1], notes=match_notes(exact[0][0]))

    if girls_only_matches:
        return ClubResolution(
            team_name=team_name,
            club=None,
            notes="Only girls-specific club branding matched; left unresolved for U12 Boys.",
        )

    return ClubResolution(team_name=team_name, club=None, notes="No club branding match found.")


def club_branding_records_from_frame(frame: pd.DataFrame) -> list[ClubBranding]:
    if frame.empty or "club_name" not in frame.columns:
        return []
    records: list[ClubBranding] = []
    for _, row in frame.iterrows():
        club_name = clean_optional_text(row.get("club_name"))
        if not club_name:
            continue
        records.append(
            ClubBranding(
                club_name=club_name,
                aliases=split_aliases(clean_optional_text(row.get("aliases"))),
                logo_url=clean_optional_text(row.get("logo_url")),
                logo_path=clean_optional_text(row.get("logo_path")),
                primary_color=clean_optional_text(row.get("primary_color")),
                secondary_color=clean_optional_text(row.get("secondary_color")),
                color_source=clean_optional_text(row.get("color_source")),
                notes=clean_optional_text(row.get("notes")),
            )
        )
    return records


def enrich_discovered_clubs(
    existing: list[ClubBranding],
    discovered: list[ClubBranding],
    *,
    fetcher: Callable[[str], str] | None = None,
    downloader: Callable[[str], bytes] | None = None,
    cache_logos: bool = True,
    assets_dir: str | Path = "assets/logos",
) -> list[ClubBranding]:
    fetcher = fetcher or fetch_html
    discovered_by_name = {normalize_branding_name(club.club_name): club for club in discovered}
    output: list[ClubBranding] = []
    consumed_discovered_names: set[str] = set()
    for club in existing:
        if is_girls_specific_club(club.club_name):
            continue
        matched = discovered_by_name.get(normalize_branding_name(club.club_name))
        if matched is None:
            matched = first_matching_discovered_club(club, discovered)
        if matched is not None:
            consumed_discovered_names.add(normalize_branding_name(matched.club_name))
        merged = merge_club_branding(club, matched)
        if merged.club_url and not merged.logo_url:
            merged = discover_logo_for_club(merged, fetcher=fetcher)
        if cache_logos and merged.logo_url:
            merged = cache_logo(merged, assets_dir=assets_dir, downloader=downloader)
            merged = enrich_colors_from_cached_logo(merged)
        if not merged.logo_url:
            merged = ClubBranding(**{**merged.__dict__, "notes": append_note(merged.notes, "No reliable club logo found.")})
        output.append(merged)

    existing_names = {normalize_branding_name(club.club_name) for club in output}
    for club in discovered:
        normalized_club_name = normalize_branding_name(club.club_name)
        if normalized_club_name in existing_names or normalized_club_name in consumed_discovered_names or is_girls_specific_club(club.club_name):
            continue
        extra = discover_logo_for_club(club, fetcher=fetcher)
        if cache_logos and extra.logo_url:
            extra = cache_logo(extra, assets_dir=assets_dir, downloader=downloader)
            extra = enrich_colors_from_cached_logo(extra)
        output.append(extra)
    return sorted(output, key=lambda club: club.club_name)


def first_matching_discovered_club(club: ClubBranding, discovered: list[ClubBranding]) -> ClubBranding | None:
    scored = scored_branding_matches(club.club_name, discovered)
    return scored[0][1] if scored else None


def merge_club_branding(base: ClubBranding, discovered: ClubBranding | None) -> ClubBranding:
    if discovered is None:
        return base
    aliases = tuple(dict.fromkeys([*base.aliases, *discovered.aliases, discovered.club_name]))
    return ClubBranding(
        club_name=base.club_name,
        aliases=aliases,
        club_url=base.club_url or discovered.club_url,
        logo_url=base.logo_url or discovered.logo_url,
        logo_path=base.logo_path or discovered.logo_path,
        primary_color=base.primary_color or discovered.primary_color,
        secondary_color=base.secondary_color or discovered.secondary_color,
        color_source=base.color_source or discovered.color_source,
        notes=append_note(base.notes, discovered.notes),
    )


def discover_logo_for_club(
    club: ClubBranding,
    *,
    fetcher: Callable[[str], str] | None = None,
) -> ClubBranding:
    if club.logo_url or not club.club_url:
        return club
    fetcher = fetcher or fetch_html
    notes = club.notes
    try:
        club_html = fetcher(club.club_url)
    except Exception as exc:
        return ClubBranding(**{**club.__dict__, "notes": append_note(notes, f"Club page fetch failed: {exc}")})
    logo_url = extract_logo_url_from_html(club_html, club.club_url)
    external_url = extract_club_website_url(club_html, club.club_url)
    if logo_url:
        return ClubBranding(**{**club.__dict__, "logo_url": logo_url, "notes": append_note(notes, "Logo parsed from CVYL club page.")})
    if external_url:
        try:
            external_html = fetcher(external_url)
            logo_url = extract_logo_url_from_html(external_html, external_url)
        except Exception as exc:
            return ClubBranding(**{**club.__dict__, "notes": append_note(notes, f"Linked club website fetch failed: {exc}")})
        if logo_url:
            return ClubBranding(**{**club.__dict__, "logo_url": logo_url, "notes": append_note(notes, "Logo parsed from linked club website.")})
    return ClubBranding(**{**club.__dict__, "notes": append_note(notes, "No reliable club logo found on CVYL club page.")})


def club_matches_team(club: ClubBranding, normalized_team: str, team_base: str) -> bool:
    return branding_match_score(normalized_team, team_base, club) > 0


def prefer_boys_or_generic_club(clubs: list[ClubBranding]) -> ClubBranding:
    boys = [club for club in clubs if is_boys_or_generic_club(club.club_name)]
    if boys:
        return sorted(boys, key=lambda club: (not has_boys_signal(club.club_name), club.club_name))[0]
    return sorted(clubs, key=lambda club: club.club_name)[0]


def is_boys_or_generic_club(name: object) -> bool:
    return not is_girls_specific_club(name)


def has_boys_signal(name: object) -> bool:
    return bool(BOYS_PROGRAM_PATTERN.search(str(name)))


def is_girls_specific_club(name: object) -> bool:
    return bool(GIRLS_PROGRAM_PATTERN.search(str(name)))


def normalize_branding_name(value: object) -> str:
    text = str(value).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_team_base(team_name: object) -> str:
    words = normalize_branding_name(team_name).split()
    kept = [word for word in words if word not in AGE_AND_TEAM_TOKENS and word not in CLUB_ORGANIZATION_TOKENS]
    return " ".join(kept).strip()


def normalized_club_base(club_name: object) -> str:
    words = normalize_branding_name(club_name).split()
    kept = [
        word
        for word in words
        if word not in AGE_AND_TEAM_TOKENS and word not in CLUB_ORGANIZATION_TOKENS
    ]
    return " ".join(kept).strip()


def split_aliases(value: object) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    return tuple(alias.strip() for alias in text.split(";") if alias.strip())


def clean_optional_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def scored_branding_matches(team_name: str, clubs: list[ClubBranding]) -> list[tuple[int, ClubBranding]]:
    normalized_team = normalize_branding_name(team_name)
    team_base = normalized_team_base(team_name)
    scored: list[tuple[int, ClubBranding]] = []
    for club in clubs:
        if is_girls_specific_club(club.club_name):
            continue
        score = branding_match_score(normalized_team, team_base, club)
        if score > 0:
            scored.append((score, club))
    return sorted(scored, key=lambda item: (-item[0], item[1].club_name))


def match_notes(score: int) -> str:
    if score >= 100:
        return "Exact or alias branding match."
    if score >= 90:
        return "Resolved via normalized town-base match."
    if score >= 80:
        return "Resolved via loose town-base fallback."
    return "Resolved via fuzzy branding fallback."


def branding_match_score(normalized_team: str, team_base: str, club: ClubBranding) -> int:
    names = [club.club_name, *club.aliases]
    normalized_names = [normalize_branding_name(name) for name in names if str(name).strip()]
    club_bases = [normalized_club_base(name) for name in names if str(name).strip()]
    best = 0
    for name, base in zip(normalized_names, club_bases, strict=False):
        if not name and not base:
            continue
        if name == normalized_team or name == team_base:
            best = max(best, 100)
        if base and (base == team_base or base == normalized_team):
            best = max(best, 95)
        if name and normalized_team.startswith(f"{name} "):
            best = max(best, 90)
        if base and normalized_team.startswith(f"{base} "):
            best = max(best, 85)
        if base and team_base.startswith(f"{base} "):
            best = max(best, 85)
        if base and club_base_tokens_overlap(base, team_base):
            best = max(best, 80)
    return best


def club_base_tokens_overlap(club_base: str, team_base: str) -> bool:
    club_tokens = set(club_base.split())
    team_tokens = set(team_base.split())
    if not club_tokens or not team_tokens:
        return False
    if club_tokens <= team_tokens or team_tokens <= club_tokens:
        return True
    return bool(club_tokens & team_tokens)


def build_unresolved_branding_review(teams: Iterable[str], clubs: list[ClubBranding]) -> pd.DataFrame:
    rows = []
    for team in sorted({str(team).strip() for team in teams if str(team).strip()}):
        resolution = resolve_club_for_team(team, clubs)
        if resolution.club is None:
            rows.append(
                {
                    "team": team,
                    "normalized_team": normalize_branding_name(team),
                    "normalized_team_base": normalized_team_base(team),
                    "notes": resolution.notes,
                }
            )
    return pd.DataFrame(rows, columns=["team", "normalized_team", "normalized_team_base", "notes"])


def export_unresolved_branding_review(
    teams: Iterable[str],
    clubs: list[ClubBranding],
    output_path: str | Path = "data/processed/cvyl_unresolved_branding.csv",
) -> Path:
    return export_csv(build_unresolved_branding_review(teams, clubs), output_path)


def safe_logo_filename(club_name: str, logo_url: str = "") -> str:
    parsed = urlparse(str(logo_url))
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        suffix = ".png"
    slug = re.sub(r"[^a-z0-9]+", "_", str(club_name).casefold()).strip("_") or "club_logo"
    return f"{slug}{suffix}"


def extract_club_entries_from_html(html: str, page_url: str) -> list[ClubBranding]:
    soup = BeautifulSoup(html, "html.parser")
    entries: dict[str, ClubBranding] = {}
    for link in soup.find_all("a", href=True):
        name = clean_club_name(link.get_text(" ", strip=True))
        href = str(link.get("href", ""))
        if "/club/" not in href or not name or is_girls_specific_club(name):
            continue
        logo_url = extract_nearby_logo_url(link, page_url)
        entries[name] = ClubBranding(
            club_name=name,
            aliases=(name,),
            club_url=urljoin(page_url, href),
            logo_url=logo_url,
            notes="Parsed from CVYL club listing.",
        )
    return sorted(entries.values(), key=lambda club: club.club_name)


def extract_nearby_logo_url(link, page_url: str) -> str:
    for node in [link, link.parent]:
        if node is None:
            continue
        img = node.find("img", src=True) if hasattr(node, "find") else None
        if img and img.get("src"):
            return urljoin(page_url, str(img["src"]))
    return ""


def clean_club_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"\s+-\s+CVYL$", "", text, flags=re.IGNORECASE)
    return text


def fetch_html(url: str) -> str:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def extract_club_website_url(html: str, page_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", href=True):
        text = link.get_text(" ", strip=True).casefold()
        href = str(link.get("href", ""))
        if "club website" in text and href:
            return urljoin(page_url, href)
    return ""


def extract_logo_url_from_html(html: str, page_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    images = sorted(soup.find_all("img", src=True), key=club_logo_candidate_score, reverse=True)
    for image in images:
        if club_logo_candidate_score(image) > 0:
            return urljoin(page_url, str(image["src"]))
    for attrs in [{"property": "og:image"}, {"name": "twitter:image"}]:
        node = soup.find("meta", attrs=attrs)
        if node and node.get("content") and not is_league_logo_url(str(node["content"])):
            return urljoin(page_url, str(node["content"]))
    return ""


def club_logo_candidate_score(image) -> int:
    text = " ".join(
        [
            str(image.get("src", "")),
            str(image.get("alt", "")),
            " ".join(image.get("class", [])),
            str(image.get("id", "")),
        ]
    ).casefold()
    if is_league_logo_url(text) or re.search(r"tracking|pixel|spacer|placeholder|crossbar", text):
        return -10
    score = 0
    if "logo" in text:
        score += 8
    if "crest" in text:
        score += 5
    if "club" in text:
        score += 3
    if "lacrosse" in text or "lax" in text:
        score += 2
    return score


def is_league_logo_url(value: str) -> bool:
    text = str(value).casefold()
    return "cvyl" in text or "organizations/776" in text or "crossbar" in text


def cache_logo(
    club: ClubBranding,
    assets_dir: str | Path = "assets/logos",
    *,
    downloader: Callable[[str], bytes] | None = None,
) -> ClubBranding:
    if not club.logo_url:
        return ClubBranding(**{**club.__dict__, "notes": append_note(club.notes, "No logo URL available.")})
    downloader = downloader or download_logo
    output_dir = Path(assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_logo_filename(club.club_name, club.logo_url)
    try:
        output_path.write_bytes(downloader(club.logo_url))
    except Exception as exc:
        return ClubBranding(**{**club.__dict__, "notes": append_note(club.notes, f"Logo cache failed: {exc}")})
    return ClubBranding(**{**club.__dict__, "logo_path": str(output_path)})


def enrich_colors_from_cached_logo(club: ClubBranding) -> ClubBranding:
    if club.primary_color or not club.logo_path:
        return club
    path = Path(club.logo_path)
    if not path.exists():
        return ClubBranding(**{**club.__dict__, "notes": append_note(club.notes, "Cached logo file missing; color extraction skipped.")})
    if path.suffix.lower() == ".svg":
        return ClubBranding(**{**club.__dict__, "notes": append_note(club.notes, "SVG logo cached; raster color extraction skipped.")})
    primary, secondary, source, notes = extract_logo_colors_from_image_bytes(path.read_bytes())
    return ClubBranding(
        **{
            **club.__dict__,
            "primary_color": primary,
            "secondary_color": secondary,
            "color_source": source,
            "notes": append_note(club.notes, notes),
        }
    )


def download_logo(url: str) -> bytes:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.content


def extract_logo_colors_from_image_bytes(image_bytes: bytes) -> tuple[str, str, str, str]:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return "", "", "", "Pillow is not installed; color extraction skipped."

    with Image.open(BytesIO(image_bytes)) as image:
        rgba = image.convert("RGBA")
        return extract_logo_colors_from_pixels(rgba.getdata())


def extract_logo_colors_from_pixels(pixels: Iterable[tuple[int, int, int, int]]) -> tuple[str, str, str, str]:
    buckets: dict[tuple[int, int, int], int] = {}
    fallback: dict[tuple[int, int, int], int] = {}
    for pixel in pixels:
        red, green, blue, alpha = [int(value) for value in pixel]
        if alpha < 32:
            continue
        bucket = quantize_color(red, green, blue)
        if is_neutral_color(*bucket):
            fallback[bucket] = fallback.get(bucket, 0) + 1
            continue
        buckets[bucket] = buckets.get(bucket, 0) + 1

    ranked = sorted(buckets.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) >= 2:
        return color_to_hex(ranked[0][0]), color_to_hex(ranked[1][0]), "extracted_logo", ""
    if len(ranked) == 1:
        return color_to_hex(ranked[0][0]), "", "extracted_logo", ""

    fallback_ranked = sorted(fallback.items(), key=lambda item: (-item[1], item[0]))
    if fallback_ranked:
        return color_to_hex(fallback_ranked[0][0]), "", "extracted_logo", "Only neutral logo colors were available."
    return "", "", "", "No usable logo colors found."


def quantize_color(red: int, green: int, blue: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(value / 32) * 32)) for value in (red, green, blue))


def is_neutral_color(red: int, green: int, blue: int) -> bool:
    near_white = red >= 224 and green >= 224 and blue >= 224
    near_black = red <= 32 and green <= 32 and blue <= 32
    low_saturation = max(red, green, blue) - min(red, green, blue) <= 18
    return near_white or near_black or low_saturation


def color_to_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def append_note(existing: str, note: str) -> str:
    return "; ".join(part for part in [existing.strip(), note.strip()] if part)
