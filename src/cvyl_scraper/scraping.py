from __future__ import annotations

from dataclasses import dataclass

import requests


DEFAULT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class FetchedPage:
    url: str
    html: str


def fetch_page(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> FetchedPage:
    headers = {
        "User-Agent": (
            "cvyl-scraper/0.1 "
            "(schedule and score data collection; contact site owner if needed)"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return FetchedPage(url=url, html=response.text)
