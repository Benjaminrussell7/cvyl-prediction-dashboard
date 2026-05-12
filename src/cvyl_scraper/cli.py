from __future__ import annotations

import argparse

import pandas as pd

from cvyl_scraper.cleaning import build_canonical_games, split_by_status
from cvyl_scraper.config import load_sources
from cvyl_scraper.export import export_csv
from cvyl_scraper.parsing import parse_schedule_page
from cvyl_scraper.scraping import fetch_page


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape CVYL lacrosse schedules and scores into cleaned CSV files."
    )
    parser.add_argument("--config", default="config/sources.yml", help="YAML source config path.")
    parser.add_argument(
        "--output",
        default="data/processed/cvyl_games.csv",
        help="Cleaned canonical games CSV output path.",
    )
    parser.add_argument(
        "--completed-output",
        default="data/processed/cvyl_completed_games.csv",
        help="Completed games CSV output path.",
    )
    parser.add_argument(
        "--scheduled-output",
        default="data/processed/cvyl_scheduled_games.csv",
        help="Scheduled games CSV output path.",
    )
    args = parser.parse_args()

    sources = load_sources(args.config)
    raw_frames = []
    for source in sources:
        page = fetch_page(source.url)
        raw_frames.append(parse_schedule_page(page.html, source))

    raw_games = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    games = build_canonical_games(raw_games)
    completed, scheduled = split_by_status(games)

    export_csv(games, args.output)
    export_csv(completed, args.completed_output)
    export_csv(scheduled, args.scheduled_output)

    print(f"Exported {len(games)} games to {args.output}")
    print(f"Exported {len(completed)} completed games to {args.completed_output}")
    print(f"Exported {len(scheduled)} scheduled games to {args.scheduled_output}")


if __name__ == "__main__":
    main()
