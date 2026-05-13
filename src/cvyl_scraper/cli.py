from __future__ import annotations

import argparse

import pandas as pd

from cvyl_scraper.backtesting import build_backtest_outputs
from cvyl_scraper.cleaning import build_canonical_games, split_by_status
from cvyl_scraper.config import load_sources, load_team_aliases
from cvyl_scraper.discovery import discover_team_sources
from cvyl_scraper.elo import build_elo_outputs
from cvyl_scraper.export import export_csv
from cvyl_scraper.modeling import build_team_games
from cvyl_scraper.parsing import parse_schedule_page
from cvyl_scraper.prediction import format_matchup_prediction, predict_matchup_from_file
from cvyl_scraper.power_v2 import build_power_ratings_v2
from cvyl_scraper.scraping import fetch_page
from cvyl_scraper.sos import build_sos
from cvyl_scraper.source_config import generate_discovered_sources_config
from cvyl_scraper.team_identity import export_team_identity_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape CVYL lacrosse schedules and scores into cleaned CSV files."
    )
    parser.add_argument("--config", default="config/sources.yml", help="YAML source config path.")
    parser.add_argument(
        "--team-aliases",
        default="config/team_aliases.yml",
        help="Optional explicit team alias mapping config path.",
    )
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
        "--backtest-output",
        default="data/processed/cvyl_backtest.csv",
        help="Completed-game backtest CSV output path.",
    )
    parser.add_argument(
        "--backtest-summary-output",
        default="data/processed/cvyl_backtest_summary.csv",
        help="Backtest summary metrics CSV output path.",
    )
    parser.add_argument(
        "--sos-output",
        default="data/processed/cvyl_sos.csv",
        help="Strength of schedule CSV output path.",
    )
    parser.add_argument(
        "--power-ratings-v2-output",
        default="data/processed/cvyl_power_ratings_v2.csv",
        help="Power Ratings v2 CSV output path.",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=float,
        default=20.0,
        help="K-factor for ELO rating updates.",
    )
    parser.add_argument(
        "--elo-recency-min-multiplier",
        type=float,
        default=0.85,
        help="Starting multiplier for early-season ELO updates.",
    )
    parser.add_argument(
        "--elo-recency-growth-games",
        type=float,
        default=20.0,
        help="Number of games controlling how quickly ELO recency weight approaches 1.0.",
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
    parser.add_argument(
        "--generate-discovered-config",
        action="store_true",
        help="Generate config/discovered_sources.yml from discovered_sources.csv and exit.",
    )
    parser.add_argument(
        "--discovered-config-input",
        default="data/processed/discovered_sources.csv",
        help="Discovered sources CSV input path for config generation.",
    )
    parser.add_argument(
        "--discovered-config-output",
        default="config/discovered_sources.yml",
        help="Generated discovered sources YAML output path.",
    )
    parser.add_argument(
        "--team-identity-audit",
        action="store_true",
        help="Generate data/processed/team_identity_audit.csv from canonical games and exit.",
    )
    parser.add_argument(
        "--team-identity-input",
        default="data/processed/cvyl_games.csv",
        help="Canonical games CSV input path for team identity audit.",
    )
    parser.add_argument(
        "--team-identity-output",
        default="data/processed/team_identity_audit.csv",
        help="Team identity audit CSV output path.",
    )
    parser.add_argument("--predict-team-a", default=None, help="First team for ELO matchup prediction.")
    parser.add_argument("--predict-team-b", default=None, help="Second team for ELO matchup prediction.")
    parser.add_argument(
        "--prediction-ratings",
        default="data/processed/cvyl_elo_ratings.csv",
        help="ELO ratings CSV input path for matchup prediction.",
    )
    parser.add_argument(
        "--prediction-team-games",
        default="data/processed/cvyl_team_games.csv",
        help="Completed team-game CSV input path for matchup projection.",
    )
    parser.add_argument(
        "--prediction-sos",
        default="data/processed/cvyl_sos.csv",
        help="Strength of schedule CSV input path for matchup prediction.",
    )
    parser.add_argument(
        "--use-power-v2",
        action="store_true",
        help="Show Power Ratings v2 context in matchup prediction output.",
    )
    parser.add_argument(
        "--prediction-power-v2",
        default="data/processed/cvyl_power_ratings_v2.csv",
        help="Power Ratings v2 CSV input path for matchup prediction context.",
    )
    args = parser.parse_args()

    if args.predict_team_a or args.predict_team_b:
        if not args.predict_team_a or not args.predict_team_b:
            parser.error("--predict-team-a and --predict-team-b must be provided together.")
        prediction = predict_matchup_from_file(
            args.predict_team_a,
            args.predict_team_b,
            args.prediction_ratings,
            args.prediction_team_games,
            args.prediction_sos,
            args.prediction_power_v2 if args.use_power_v2 else None,
        )
        print(format_matchup_prediction(prediction))
        return

    if args.generate_discovered_config:
        generated_path = generate_discovered_sources_config(
            args.discovered_config_input,
            args.discovered_config_output,
        )
        print(f"Exported discovered sources config to {generated_path}")
        return

    if args.team_identity_audit:
        audit_path = export_team_identity_audit(
            args.team_identity_input,
            args.team_identity_output,
        )
        print(f"Exported team identity audit to {audit_path}")
        return

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
    team_aliases = load_team_aliases(args.team_aliases)
    games = build_canonical_games(raw_games, team_aliases=team_aliases)
    completed, scheduled = split_by_status(games)
    team_games = build_team_games(games)
    elo_ratings, elo_history = build_elo_outputs(
        team_games,
        k_factor=args.elo_k_factor,
        recency_min_multiplier=args.elo_recency_min_multiplier,
        recency_growth_games=args.elo_recency_growth_games,
    )
    backtest, backtest_summary = build_backtest_outputs(
        games,
        k_factor=args.elo_k_factor,
        recency_min_multiplier=args.elo_recency_min_multiplier,
        recency_growth_games=args.elo_recency_growth_games,
    )
    sos = build_sos(team_games, elo_ratings)
    power_ratings_v2 = build_power_ratings_v2(team_games)

    export_csv(games, args.output)
    export_csv(completed, args.completed_output)
    export_csv(scheduled, args.scheduled_output)
    export_csv(team_games, args.team_games_output)
    export_csv(elo_ratings, args.elo_ratings_output)
    export_csv(elo_history, args.elo_history_output)
    export_csv(backtest, args.backtest_output)
    export_csv(backtest_summary, args.backtest_summary_output)
    export_csv(sos, args.sos_output)
    export_csv(power_ratings_v2, args.power_ratings_v2_output)

    print(f"Exported {len(games)} games to {args.output}")
    print(f"Exported {len(completed)} completed games to {args.completed_output}")
    print(f"Exported {len(scheduled)} scheduled games to {args.scheduled_output}")
    print(f"Exported {len(team_games)} team-game rows to {args.team_games_output}")
    print(f"Exported {len(elo_ratings)} ELO ratings to {args.elo_ratings_output}")
    print(f"Exported {len(elo_history)} ELO history rows to {args.elo_history_output}")
    print(f"Exported {len(backtest)} backtest predictions to {args.backtest_output}")
    print(f"Exported backtest summary to {args.backtest_summary_output}")
    print(f"Exported {len(sos)} SOS rows to {args.sos_output}")
    print(f"Exported {len(power_ratings_v2)} Power Ratings v2 rows to {args.power_ratings_v2_output}")


if __name__ == "__main__":
    main()
