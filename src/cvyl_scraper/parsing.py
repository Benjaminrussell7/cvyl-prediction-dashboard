from __future__ import annotations

import re
from collections.abc import Iterable
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
    frames = [_parse_table(table) for table in soup.find_all("table")]
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return _empty_raw_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined["source_name"] = source.name
    combined["source_url"] = source.url
    combined["season"] = source.season
    combined["division"] = source.division
    return combined


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
