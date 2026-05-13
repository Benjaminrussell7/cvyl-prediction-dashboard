from __future__ import annotations

import argparse

import pandas as pd

from cvyl_scraper.cleaning import build_canonical_games, split_by_status
from cvyl_scraper.config import load_sources
from cvyl_scraper.discovery import discover_team_sources
from cvyl_scraper.elo import build_elo_outputs
from cvyl_scraper.export import export_csv
from cvyl_scraper.modeling import build_team_games
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
    parser.add_argument(
        "--team-games-output",
        default="data/processed/cvyl_team_games.csv",
        help="Team-game modeling CSV output path.",
    )
    parser.add_argument(
        "--elo-ratings-output",
        default="data/processed/cvyl_elo_ratings.csv",
        help="Final ELO ratings CSV output path.",
    )
    parser.add_argument(
        "--elo-history-output",
        default="data/processed/cvyl_elo_history.csv",
        help="Per-game ELO history CSV output path.",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=float,
        default=20.0,
        help="K-factor for ELO rating updates.",
    )
    parser.add_argument(
        "--discover-url",
        default=None,
        help="Optional CVYL divisions/teams page URL for team source discovery.",
    )
    parser.add_argument(
        "--discovered-sources-output",
        default="data/processed/discovered_sources.csv",
        help="Discovered team sources CSV output path.",
    )
    args = parser.parse_args()

    if args.discover_url:
        discovery_page = fetch_page(args.discover_url)
        discovered_sources = discover_team_sources(discovery_page.html, discovery_page.url)
        export_csv(discovered_sources, args.discovered_sources_output)
        print(
            f"Exported {len(discovered_sources)} discovered sources "
            f"to {args.discovered_sources_output}"
        )

    sources = load_sources(args.config)
    raw_frames = []
    for source in sources:
        page = fetch_page(source.url)
        raw_frames.append(parse_schedule_page(page.html, source))

    raw_games = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    games = build_canonical_games(raw_games)
    completed, scheduled = split_by_status(games)
    team_games = build_team_games(games)
    elo_ratings, elo_history = build_elo_outputs(team_games, k_factor=args.elo_k_factor)

    export_csv(games, args.output)
    export_csv(completed, args.completed_output)
    export_csv(scheduled, args.scheduled_output)
    export_csv(team_games, args.team_games_output)
    export_csv(elo_ratings, args.elo_ratings_output)
    export_csv(elo_history, args.elo_history_output)

    print(f"Exported {len(games)} games to {args.output}")
    print(f"Exported {len(completed)} completed games to {args.completed_output}")
    print(f"Exported {len(scheduled)} scheduled games to {args.scheduled_output}")
    print(f"Exported {len(team_games)} team-game rows to {args.team_games_output}")
    print(f"Exported {len(elo_ratings)} ELO ratings to {args.elo_ratings_output}")
    print(f"Exported {len(elo_history)} ELO history rows to {args.elo_history_output}")


if __name__ == "__main__":
    main()
