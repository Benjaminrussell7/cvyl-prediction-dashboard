from __future__ import annotations

import argparse

import pandas as pd

from cvyl_scraper.backtesting import build_backtest_outputs
from cvyl_scraper.branding import discover_team_branding
from cvyl_scraper.calibration import build_power_rating_calibration
from cvyl_scraper.cleaning import build_canonical_games, split_by_status
from cvyl_scraper.config import load_sources, load_team_aliases
from cvyl_scraper.discovery import discover_team_sources
from cvyl_scraper.elo import build_elo_outputs
from cvyl_scraper.export import export_csv
from cvyl_scraper.historical_snapshots import build_historical_snapshots
from cvyl_scraper.model_comparison import build_model_comparison_outputs
from cvyl_scraper.model_comparison_v3 import build_model_comparison_v3_outputs
from cvyl_scraper.modeling import build_team_games
from cvyl_scraper.parsing import parse_schedule_page
from cvyl_scraper.prediction import format_matchup_prediction, predict_matchup_from_file
from cvyl_scraper.probability_calibration import build_model_comparison_v4_calibrated_outputs
from cvyl_scraper.power_v2 import build_power_ratings_v2
from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency
from cvyl_scraper.scraping import fetch_page
from cvyl_scraper.sos import build_sos
from cvyl_scraper.source_config import generate_discovered_sources_config
from cvyl_scraper.team_identity import export_team_identity_audit
from cvyl_scraper.trends import build_trends


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
        "--power-ratings-v3-recency-output",
        default="data/processed/cvyl_power_ratings_v3_recency.csv",
        help="Experimental recency-weighted Power Ratings v3 CSV output path.",
    )
    parser.add_argument(
        "--model-comparison-output",
        default="data/processed/cvyl_model_comparison.csv",
        help="ELO vs Power Ratings v2 comparison CSV output path.",
    )
    parser.add_argument(
        "--model-comparison-summary-output",
        default="data/processed/cvyl_model_comparison_summary.csv",
        help="ELO vs Power Ratings v2 comparison summary CSV output path.",
    )
    parser.add_argument(
        "--model-comparison-v3-output",
        default="data/processed/cvyl_model_comparison_v3.csv",
        help="ELO vs Power v2 vs Power v3 recency comparison CSV output path.",
    )
    parser.add_argument(
        "--model-comparison-v3-summary-output",
        default="data/processed/cvyl_model_comparison_v3_summary.csv",
        help="ELO vs Power v2 vs Power v3 recency comparison summary CSV output path.",
    )
    parser.add_argument(
        "--power-rating-calibration-output",
        default="data/processed/cvyl_calibration_power_rating.csv",
        help="Power Rating calibration bucket CSV output path.",
    )
    parser.add_argument(
        "--model-comparison-v4-calibrated-output",
        default="data/processed/cvyl_model_comparison_v4_calibrated.csv",
        help="Baseline Power v3 vs calibrated Power v3 comparison CSV output path.",
    )
    parser.add_argument(
        "--model-comparison-v4-calibrated-summary-output",
        default="data/processed/cvyl_model_comparison_v4_calibrated_summary.csv",
        help="Baseline Power v3 vs calibrated Power v3 comparison summary CSV output path.",
    )
    parser.add_argument(
        "--power-rating-v4-calibration-output",
        default="data/processed/cvyl_calibration_power_rating_v4.csv",
        help="Calibrated Power Rating v4 calibration bucket CSV output path.",
    )
    parser.add_argument(
        "--trends-output",
        default="data/processed/cvyl_trends.csv",
        help="Recent team trend and momentum CSV output path.",
    )
    parser.add_argument(
        "--historical-snapshots-output",
        default="data/processed/cvyl_historical_snapshots.csv",
        help="Weekly league historical snapshots CSV output path.",
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
    parser.add_argument(
        "--discover-branding",
        action="store_true",
        help="Discover team logo URLs and branding metadata from discovered team pages and exit.",
    )
    parser.add_argument(
        "--branding-sources-input",
        default="config/discovered_sources.yml",
        help="Discovered sources YAML or CSV input path for branding discovery.",
    )
    parser.add_argument(
        "--branding-output",
        default="data/processed/cvyl_team_branding.csv",
        help="Team branding CSV output path.",
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
            args.prediction_power_v2,
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

    if args.discover_branding:
        branding_path = discover_team_branding(
            args.branding_sources_input,
            args.branding_output,
        )
        print(f"Exported team branding to {branding_path}")
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
    trends = build_trends(team_games)
    historical_snapshots = build_historical_snapshots(team_games)
    power_ratings_v2 = build_power_ratings_v2(team_games)
    power_ratings_v3_recency = build_power_ratings_v3_recency(team_games)
    model_comparison, model_comparison_summary = build_model_comparison_outputs(
        games,
        k_factor=args.elo_k_factor,
        recency_min_multiplier=args.elo_recency_min_multiplier,
        recency_growth_games=args.elo_recency_growth_games,
    )
    model_comparison_v3, model_comparison_v3_summary = build_model_comparison_v3_outputs(
        games,
        k_factor=args.elo_k_factor,
        recency_min_multiplier=args.elo_recency_min_multiplier,
        recency_growth_games=args.elo_recency_growth_games,
    )
    power_rating_calibration = build_power_rating_calibration(model_comparison_v3)
    (
        model_comparison_v4_calibrated,
        model_comparison_v4_calibrated_summary,
        power_rating_v4_calibration,
    ) = build_model_comparison_v4_calibrated_outputs(games)

    export_csv(games, args.output)
    export_csv(completed, args.completed_output)
    export_csv(scheduled, args.scheduled_output)
    export_csv(team_games, args.team_games_output)
    export_csv(elo_ratings, args.elo_ratings_output)
    export_csv(elo_history, args.elo_history_output)
    export_csv(backtest, args.backtest_output)
    export_csv(backtest_summary, args.backtest_summary_output)
    export_csv(sos, args.sos_output)
    export_csv(trends, args.trends_output)
    export_csv(historical_snapshots, args.historical_snapshots_output)
    export_csv(power_ratings_v2, args.power_ratings_v2_output)
    export_csv(power_ratings_v3_recency, args.power_ratings_v3_recency_output)
    export_csv(model_comparison, args.model_comparison_output)
    export_csv(model_comparison_summary, args.model_comparison_summary_output)
    export_csv(model_comparison_v3, args.model_comparison_v3_output)
    export_csv(model_comparison_v3_summary, args.model_comparison_v3_summary_output)
    export_csv(power_rating_calibration, args.power_rating_calibration_output)
    export_csv(model_comparison_v4_calibrated, args.model_comparison_v4_calibrated_output)
    export_csv(
        model_comparison_v4_calibrated_summary,
        args.model_comparison_v4_calibrated_summary_output,
    )
    export_csv(power_rating_v4_calibration, args.power_rating_v4_calibration_output)

    print(f"Exported {len(games)} games to {args.output}")
    print(f"Exported {len(completed)} completed games to {args.completed_output}")
    print(f"Exported {len(scheduled)} scheduled games to {args.scheduled_output}")
    print(f"Exported {len(team_games)} team-game rows to {args.team_games_output}")
    print(f"Exported {len(elo_ratings)} ELO ratings to {args.elo_ratings_output}")
    print(f"Exported {len(elo_history)} ELO history rows to {args.elo_history_output}")
    print(f"Exported {len(backtest)} backtest predictions to {args.backtest_output}")
    print(f"Exported backtest summary to {args.backtest_summary_output}")
    print(f"Exported {len(sos)} SOS rows to {args.sos_output}")
    print(f"Exported {len(trends)} trend rows to {args.trends_output}")
    print(
        f"Exported {len(historical_snapshots)} historical snapshot rows "
        f"to {args.historical_snapshots_output}"
    )
    print(f"Exported {len(power_ratings_v2)} Power Ratings v2 rows to {args.power_ratings_v2_output}")
    print(
        f"Exported {len(power_ratings_v3_recency)} Power Ratings v3 recency rows "
        f"to {args.power_ratings_v3_recency_output}"
    )
    print(f"Exported {len(model_comparison)} model comparison rows to {args.model_comparison_output}")
    print(f"Exported model comparison summary to {args.model_comparison_summary_output}")
    print(
        f"Exported {len(model_comparison_v3)} v3 model comparison rows "
        f"to {args.model_comparison_v3_output}"
    )
    print(f"Exported v3 model comparison summary to {args.model_comparison_v3_summary_output}")
    print(
        f"Exported {len(power_rating_calibration)} Power Rating calibration rows "
        f"to {args.power_rating_calibration_output}"
    )
    print(
        f"Exported {len(model_comparison_v4_calibrated)} calibrated v4 model comparison rows "
        f"to {args.model_comparison_v4_calibrated_output}"
    )
    print(
        "Exported calibrated v4 model comparison summary "
        f"to {args.model_comparison_v4_calibrated_summary_output}"
    )
    print(
        f"Exported {len(power_rating_v4_calibration)} calibrated v4 calibration rows "
        f"to {args.power_rating_v4_calibration_output}"
    )


if __name__ == "__main__":
    main()
