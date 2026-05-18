from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvyl_scraper.backtesting import _actual_winner, _completed_games
from cvyl_scraper.hybrid import DEFAULT_POWER_V2_LOGISTIC_SCALE, power_v2_win_probability
from cvyl_scraper.model_comparison_v3 import _team_game_rows
from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency


DEFAULT_GAMES_CSV = ROOT / "data" / "processed" / "cvyl_games.csv"
OUTPUT_DIR = ROOT / "modeling_lab" / "outputs"
ROLLING_WINDOW_SIZE = 25
DEFAULT_TOTAL_GOALS = 12.0


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    margin_cap: float | None = None
    connectivity_adjustment: bool = False


VARIANTS = [
    Variant("baseline_power_v3", "Baseline Power v3"),
    Variant("cap_6", "Blowout cap 6", margin_cap=6.0),
    Variant("cap_8", "Blowout cap 8", margin_cap=8.0),
    Variant("connectivity", "Connectivity exposure", connectivity_adjustment=True),
    Variant("cap_6_connectivity", "Cap 6 + connectivity", margin_cap=6.0, connectivity_adjustment=True),
    Variant("cap_8_connectivity", "Cap 8 + connectivity", margin_cap=8.0, connectivity_adjustment=True),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(DEFAULT_GAMES_CSV)

    predictions = run_rolling_backtests(games)
    summary = summarize_variants(predictions)
    rolling = summarize_rolling_windows(predictions)
    rankings = build_final_rankings(games)
    interpretation = build_interpretation(summary, rolling, rankings)

    predictions.to_csv(OUTPUT_DIR / "hidden_tiering_predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "hidden_tiering_summary.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "hidden_tiering_rolling_windows.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "hidden_tiering_final_rankings.csv", index=False)
    (OUTPUT_DIR / "hidden_tiering_interpretation.md").write_text(interpretation, encoding="utf-8")

    print(f"Wrote lab outputs to {OUTPUT_DIR}")


def run_rolling_backtests(games: pd.DataFrame) -> pd.DataFrame:
    completed = _completed_games(games)
    prior_team_game_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    for game_number, (_, game) in enumerate(completed.iterrows(), start=1):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        actual_home_result = _home_result(home_score, away_score)
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        total_goals_prediction = _prior_total_goals(prior_team_game_rows)
        exposure = build_top_tier_exposure(prior_team_game_rows)

        for variant in VARIANTS:
            ratings = _variant_ratings(prior_team_game_rows, variant, exposure)
            home_rating = ratings.get(home_team, 0.0)
            away_rating = ratings.get(away_team, 0.0)
            predicted_margin = home_rating - away_rating
            home_probability = power_v2_win_probability(
                predicted_margin,
                scale=DEFAULT_POWER_V2_LOGISTIC_SCALE,
            )
            predicted_winner = home_team if home_probability >= 0.5 else away_team
            actual_winner = _actual_winner(home_team, away_team, home_score, away_score)

            rows.append(
                {
                    "game_number": game_number,
                    "game_id": game["game_id"],
                    "game_date": game["game_date"].date().isoformat(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "home_rating": home_rating,
                    "away_rating": away_rating,
                    "rating_diff": predicted_margin,
                    "home_win_probability": home_probability,
                    "predicted_winner": predicted_winner,
                    "actual_winner": actual_winner,
                    "prediction_correct": predicted_winner == actual_winner,
                    "home_score": home_score,
                    "away_score": away_score,
                    "actual_home_result": actual_home_result,
                    "predicted_margin": predicted_margin,
                    "actual_margin": actual_margin,
                    "predicted_total_goals": total_goals_prediction,
                    "actual_total_goals": actual_total,
                    "home_top_tier_exposure": exposure.get(home_team, 0.0),
                    "away_top_tier_exposure": exposure.get(away_team, 0.0),
                }
            )

        prior_team_game_rows.extend(_team_game_rows(game))

    return pd.DataFrame(rows)


def _variant_ratings(
    prior_team_game_rows: list[dict[str, object]],
    variant: Variant,
    exposure: dict[str, float],
) -> dict[str, float]:
    if not prior_team_game_rows:
        return {}

    kwargs = {}
    if variant.margin_cap is not None:
        kwargs["margin_cap"] = variant.margin_cap
    ratings = build_power_ratings_v3_recency(pd.DataFrame(prior_team_game_rows), **kwargs)
    raw = dict(zip(ratings["team"], ratings["power_rating_v3_recency"], strict=False))
    if not variant.connectivity_adjustment:
        return {team: float(rating) for team, rating in raw.items()}

    adjusted: dict[str, float] = {}
    for team, rating in raw.items():
        team_exposure = exposure.get(team, 0.0)
        multiplier = 0.78 + (0.22 * team_exposure)
        adjusted[team] = float(rating) * multiplier
    return adjusted


def build_top_tier_exposure(prior_team_game_rows: list[dict[str, object]]) -> dict[str, float]:
    if not prior_team_game_rows:
        return {}

    team_games = pd.DataFrame(prior_team_game_rows)
    ratings = build_power_ratings_v3_recency(team_games)
    if ratings.empty:
        return {}

    top_count = max(4, math.ceil(len(ratings) * 0.20))
    top_teams = set(ratings.head(top_count)["team"].astype(str))

    exposure_rows: list[dict[str, object]] = []
    for team, group in team_games.groupby("team", sort=True):
        opponents = group["opponent"].astype(str)
        direct_top_games = int(opponents.isin(top_teams).sum())
        direct_component = min(1.0, direct_top_games / 3.0)
        exposure_rows.append(
            {
                "team": str(team),
                "direct_top_games": direct_top_games,
                "direct_component": direct_component,
                "opponents": list(opponents),
            }
        )

    direct_by_team = {str(row["team"]): float(row["direct_component"]) for row in exposure_rows}
    exposure: dict[str, float] = {}
    for row in exposure_rows:
        opponent_components = [direct_by_team.get(opponent, 0.0) for opponent in row["opponents"]]
        neighborhood_component = (
            sum(opponent_components) / len(opponent_components) if opponent_components else 0.0
        )
        exposure[str(row["team"])] = min(
            1.0,
            (0.75 * float(row["direct_component"])) + (0.25 * neighborhood_component),
        )
    return exposure


def summarize_variants(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in predictions.groupby("variant", sort=False):
        rows.append(_metric_row(group, variant=variant, rolling_window=None))
    summary = pd.DataFrame(rows)
    baseline = summary[summary["variant"] == "baseline_power_v3"].iloc[0]

    summary["delta_winner_accuracy"] = summary["winner_accuracy"] - baseline["winner_accuracy"]
    summary["delta_brier_score"] = summary["brier_score"] - baseline["brier_score"]
    summary["delta_log_loss"] = summary["log_loss"] - baseline["log_loss"]
    summary["delta_margin_mae"] = summary["margin_mae"] - baseline["margin_mae"]
    summary["delta_total_goals_mae"] = summary["total_goals_mae"] - baseline["total_goals_mae"]
    summary["promotion_candidate"] = summary.apply(_promotion_candidate, axis=1)
    return summary


def summarize_rolling_windows(predictions: pd.DataFrame) -> pd.DataFrame:
    windows = []
    max_game = int(predictions["game_number"].max()) if not predictions.empty else 0
    for start in range(1, max_game + 1, ROLLING_WINDOW_SIZE):
        end = min(max_game, start + ROLLING_WINDOW_SIZE - 1)
        window = predictions[
            (predictions["game_number"] >= start) & (predictions["game_number"] <= end)
        ]
        for variant, group in window.groupby("variant", sort=False):
            windows.append(_metric_row(group, variant=variant, rolling_window=f"{start}-{end}"))
    return pd.DataFrame(windows)


def _metric_row(
    group: pd.DataFrame,
    *,
    variant: str,
    rolling_window: str | None,
) -> dict[str, object]:
    probabilities = group["home_win_probability"].clip(1e-6, 1 - 1e-6)
    results = group["actual_home_result"]
    row = {
        "variant": variant,
        "variant_label": group["variant_label"].iloc[0],
        "winner_accuracy": float(group["prediction_correct"].mean()),
        "brier_score": float(((probabilities - results) ** 2).mean()),
        "log_loss": float(
            -(results * probabilities.map(math.log) + (1 - results) * (1 - probabilities).map(math.log)).mean()
        ),
        "margin_mae": float((group["predicted_margin"] - group["actual_margin"]).abs().mean()),
        "total_goals_mae": float(
            (group["predicted_total_goals"] - group["actual_total_goals"]).abs().mean()
        ),
        "games": int(len(group)),
    }
    if rolling_window is not None:
        row["rolling_window"] = rolling_window
    return row


def _promotion_candidate(row: pd.Series) -> bool:
    if row["variant"] == "baseline_power_v3":
        return False
    return bool(
        row["delta_brier_score"] < 0
        and row["delta_log_loss"] < 0
        and row["delta_winner_accuracy"] >= -0.01
        and row["delta_margin_mae"] <= 0
    )


def build_final_rankings(games: pd.DataFrame) -> pd.DataFrame:
    completed = _completed_games(games)
    prior_rows: list[dict[str, object]] = []
    for _, game in completed.iterrows():
        prior_rows.extend(_team_game_rows(game))

    exposure = build_top_tier_exposure(prior_rows)
    rows = []
    for variant in VARIANTS:
        ratings = _variant_ratings(prior_rows, variant, exposure)
        ranked = sorted(ratings.items(), key=lambda item: (-item[1], item[0]))
        for rank, (team, rating) in enumerate(ranked, start=1):
            rows.append(
                {
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "team": team,
                    "rank": rank,
                    "rating": rating,
                    "top_tier_exposure": exposure.get(team, 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_interpretation(
    summary: pd.DataFrame,
    rolling: pd.DataFrame,
    rankings: pd.DataFrame,
) -> str:
    baseline = summary[summary["variant"] == "baseline_power_v3"].iloc[0]
    candidates = summary[summary["promotion_candidate"]]
    improved_brier_log = summary[
        (summary["variant"] != "baseline_power_v3")
        & (summary["delta_brier_score"] < 0)
        & (summary["delta_log_loss"] < 0)
    ]
    worsened = summary[
        (summary["variant"] != "baseline_power_v3")
        & ((summary["delta_brier_score"] > 0) | (summary["delta_log_loss"] > 0))
    ]
    rham_rows = _team_rows(rankings, ["RHAM"])
    west_hartford_rows = _team_rows(rankings, ["West", "Hartford", "Gold"])

    lines = [
        "# Hidden Tiering Power v3 Modeling Lab",
        "",
        "Scope: lab-only rolling backtest using existing canonical games data. Tournament data was not added.",
        "",
        "Connectivity adjustment: for each pregame window, teams are scored by exposure to the prior top 20% of the Power v3 schedule graph. The score combines direct games against top-tier teams with a smaller neighborhood component from opponents' own top-tier exposure, then shrinks rating magnitude for teams with little exposure. This is intentionally different from SOS because it uses top-tier graph exposure rather than average opponent strength.",
        "",
        "## Baseline",
        "",
        (
            f"- Baseline games: {int(baseline['games'])}; winner_accuracy "
            f"{baseline['winner_accuracy']:.3f}; brier_score {baseline['brier_score']:.3f}; "
            f"log_loss {baseline['log_loss']:.3f}; margin_mae {baseline['margin_mae']:.3f}; "
            f"total_goals_mae {baseline['total_goals_mae']:.3f}."
        ),
        "",
        "## Variant Summary",
        "",
        _markdown_table(
            summary[
                [
                    "variant",
                    "winner_accuracy",
                    "brier_score",
                    "log_loss",
                    "margin_mae",
                    "total_goals_mae",
                    "games",
                    "delta_winner_accuracy",
                    "delta_brier_score",
                    "delta_log_loss",
                    "delta_margin_mae",
                    "promotion_candidate",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
    ]

    if improved_brier_log.empty:
        lines.append("- No experimental variant improved both Brier score and log loss versus baseline.")
    else:
        improved_names = ", ".join(improved_brier_log["variant"].tolist())
        lines.append(f"- Improved both Brier score and log loss: {improved_names}.")

    if worsened.empty:
        lines.append("- No variant worsened either Brier score or log loss.")
    else:
        worsened_names = ", ".join(worsened["variant"].tolist())
        lines.append(f"- Worsened at least one probability metric: {worsened_names}.")

    if candidates.empty:
        lines.append("- Promotion recommendation: do not promote any variant under the stated acceptance criteria.")
    else:
        candidate_names = ", ".join(candidates["variant"].tolist())
        lines.append(f"- Promotion recommendation: {candidate_names}.")

    lines.extend(
        [
            "",
            "## Ranking Oddities",
            "",
            _ranking_note("RHAM", rham_rows),
            _ranking_note("West Hartford Gold", west_hartford_rows),
            "",
            "## Rolling Windows",
            "",
            f"- Rolling window size: {ROLLING_WINDOW_SIZE} games.",
            f"- Rolling windows evaluated: {rolling['rolling_window'].nunique() if not rolling.empty else 0}.",
            "- Full window metrics are in `hidden_tiering_rolling_windows.csv`.",
            "",
            "## Files",
            "",
            "- `hidden_tiering_predictions.csv`",
            "- `hidden_tiering_summary.csv`",
            "- `hidden_tiering_rolling_windows.csv`",
            "- `hidden_tiering_final_rankings.csv`",
            "- `hidden_tiering_interpretation.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def _ranking_note(team_label: str, rows: pd.DataFrame) -> str:
    if rows.empty:
        return f"- {team_label}: not present in final ranking output."

    baseline = rows[rows["variant"] == "baseline_power_v3"]
    if baseline.empty:
        return f"- {team_label}: present, but no baseline row found."

    parts = [
        (
            f"{row['variant']} rank {int(row['rank'])}, rating {row['rating']:.2f}, "
            f"top-tier exposure {row['top_tier_exposure']:.2f}"
        )
        for _, row in rows.sort_values("variant").iterrows()
    ]
    return f"- {team_label}: " + "; ".join(parts) + "."


def _team_rows(rankings: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    if rankings.empty:
        return rankings
    mask = pd.Series(True, index=rankings.index)
    for term in terms:
        mask = mask & rankings["team"].str.contains(term, case=False, na=False)
    return rankings[mask]


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_markdown_cell(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _prior_total_goals(prior_team_game_rows: list[dict[str, object]]) -> float:
    if not prior_team_game_rows:
        return DEFAULT_TOTAL_GOALS
    prior = pd.DataFrame(prior_team_game_rows)
    totals = pd.to_numeric(prior["points_for"], errors="coerce") + pd.to_numeric(
        prior["points_against"], errors="coerce"
    )
    return float(totals.dropna().mean())


def _home_result(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if away_score > home_score:
        return 0.0
    return 0.5


if __name__ == "__main__":
    main()
