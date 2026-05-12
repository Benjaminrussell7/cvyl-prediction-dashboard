from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from cvyl_scraper.models import Source


HEADER_ALIASES = {
    "date": "game_date",
    "game date": "game_date",
    "time": "game_time",
    "start time": "game_time",
    "home": "home_team",
    "home team": "home_team",
    "away": "away_team",
    "away team": "away_team",
    "visitor": "away_team",
    "visiting team": "away_team",
    "score": "score",
    "result": "score",
    "home score": "home_score",
    "away score": "away_score",
    "visitor score": "away_score",
}


def parse_schedule_page(html: str, source: Source) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    frames = [_parse_crossbar_game_boxes(soup, source)]
    frames.extend(_parse_table(table) for table in soup.find_all("table"))
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return _empty_raw_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined["source_name"] = source.name
    combined["source_url"] = source.url
    combined["season"] = source.season
    combined["division"] = source.division
    return combined


def _parse_crossbar_game_boxes(soup: BeautifulSoup, source: Source) -> pd.DataFrame:
    team_name = _extract_crossbar_team_name(soup)
    if not team_name:
        return _empty_raw_frame()

    records: list[dict[str, str | int | None]] = []
    for box in soup.select("div.box"):
        date_columns = box.select("div.col-xs-2 h1")
        opponent_heading = box.select_one("h2.nomargin")
        result_heading = box.select_one("div.text-center h3")
        marker = opponent_heading.select_one("span.small") if opponent_heading else None
        if len(date_columns) < 2 or not opponent_heading or not result_heading or not marker:
            continue

        game_date = _parse_crossbar_date(
            _clean_text(date_columns[0].get_text(" ", strip=True)),
            _clean_text(date_columns[1].get_text(" ", strip=True)),
            source.season,
        )
        venue_marker = _normalize_marker(marker.get_text(" ", strip=True))
        opponent = _opponent_name(opponent_heading, marker)
        result_text = _clean_text(result_heading.get_text(" ", strip=True))

        if not game_date or venue_marker not in {"vs.", "@"} or not opponent or not result_text:
            continue

        team_score, opponent_score, game_time = _parse_crossbar_result(result_text)
        if venue_marker == "vs.":
            home_team = team_name
            away_team = opponent
            home_score = team_score
            away_score = opponent_score
        else:
            home_team = opponent
            away_team = team_name
            home_score = opponent_score
            away_score = team_score

        records.append(
            {
                "game_date": game_date,
                "game_time": game_time,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    if not records:
        return _empty_raw_frame()

    frame = pd.DataFrame(records)
    return _standardize_raw_columns(frame)


def _parse_table(table: Any) -> pd.DataFrame:
    rows = table.find_all("tr")
    if not rows:
        return _empty_raw_frame()

    headers = _extract_headers(rows[0])
    body_rows = rows[1:] if headers else rows

    records: list[dict[str, str | None]] = []
    for row in body_rows:
        cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        if not any(cells):
            continue
        records.append(_cells_to_record(cells, headers))

    if not records:
        return _empty_raw_frame()

    frame = pd.DataFrame(records)
    return _standardize_raw_columns(frame)


def _extract_headers(row: Any) -> list[str]:
    cells = row.find_all(["th", "td"])
    if not cells:
        return []

    values = [_normalize_header(cell.get_text(" ", strip=True)) for cell in cells]
    known_headers = sum(1 for value in values if value in HEADER_ALIASES)
    return values if known_headers >= 2 else []


def _cells_to_record(cells: list[str], headers: list[str]) -> dict[str, str | None]:
    if headers and len(headers) == len(cells):
        return {headers[index]: value for index, value in enumerate(cells)}

    return {
        "game_date": _get(cells, 0),
        "game_time": _get(cells, 1),
        "away_team": _get(cells, 2),
        "home_team": _get(cells, 3),
        "score": _get(cells, 4),
    }


def _standardize_raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: HEADER_ALIASES.get(_normalize_header(str(column)), str(column))
        for column in frame.columns
    }
    frame = frame.rename(columns=renamed)

    for column in [
        "game_date",
        "game_time",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "score",
    ]:
        if column not in frame.columns:
            frame[column] = None

    if frame["home_score"].isna().all() and frame["away_score"].isna().all():
        home_scores, away_scores = _split_score_column(frame["score"])
        frame["home_score"] = home_scores
        frame["away_score"] = away_scores

    return frame[
        [
            "game_date",
            "game_time",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ]


def _split_score_column(scores: Iterable[Any]) -> tuple[list[int | None], list[int | None]]:
    home_scores: list[int | None] = []
    away_scores: list[int | None] = []
    for score in scores:
        numbers = [int(value) for value in re.findall(r"\d+", str(score or ""))]
        if len(numbers) >= 2:
            away_scores.append(numbers[0])
            home_scores.append(numbers[1])
        else:
            away_scores.append(None)
            home_scores.append(None)
    return home_scores, away_scores


def _extract_crossbar_team_name(soup: BeautifulSoup) -> str | None:
    title = soup.find("title")
    if title:
        parts = [_clean_text(part) for part in title.get_text(" ", strip=True).split("|")]
        if len(parts) >= 2 and parts[1]:
            return parts[1]

    for heading in reversed(soup.find_all("h1")):
        value = _clean_text(heading.get_text(" ", strip=True))
        if value and not re.fullmatch(r"\d{1,2}|[A-Za-z]{3}", value):
            return value
    return None


def _parse_crossbar_date(month: str, day: str, season: int | None) -> str | None:
    if season is None:
        return None
    try:
        parsed = datetime.strptime(f"{month} {day} {season}", "%b %d %Y")
    except ValueError:
        return None
    return parsed.date().isoformat()


def _normalize_marker(value: str) -> str:
    value = _clean_text(value).lower()
    if value.startswith("vs"):
        return "vs."
    if value.startswith("@"):
        return "@"
    return value


def _opponent_name(opponent_heading: Any, marker: Any) -> str | None:
    marker.extract()
    return _clean_text(opponent_heading.get_text(" ", strip=True)) or None


def _parse_crossbar_result(value: str) -> tuple[int | None, int | None, str | None]:
    score_match = re.search(r"\b(\d+)\s*-\s*(\d+)\b", value)
    if score_match:
        return int(score_match.group(1)), int(score_match.group(2)), None

    time_match = re.search(r"\b\d{1,2}:\d{2}\s*[AP]M\b", value, re.IGNORECASE)
    if time_match:
        return None, None, time_match.group(0).upper()

    return None, None, None


def _empty_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_date",
            "game_time",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "source_name",
            "source_url",
            "season",
            "division",
        ]
    )


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _get(values: list[str], index: int) -> str | None:
    if index >= len(values):
        return None
    return values[index] or None
