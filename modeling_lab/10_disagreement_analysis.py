from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUTPUTS, write_output


HYBRID_INPUT = "hybrid_predictions.csv"
CONFIDENCE_INPUT = "confidence_predictions.csv"

DISAGREEMENT_ANALYSIS_OUTPUT = "disagreement_analysis.csv"
DISAGREEMENT_GAMES_OUTPUT = "disagreement_games.csv"
MODEL_DISAGREEMENT_AUDIT_OUTPUT = "model_disagreement_audit.csv"

EPSILON = 1e-6


def read_lab_output(filename: str) -> pd.DataFrame:
    path = OUTPUTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing modeling lab input: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().removeprefix("\ufeff") for column in frame.columns]
    return frame


def log_loss(actual: pd.Series, probability: pd.Series) -> float:
    clipped = probability.astype(float).clip(EPSILON, 1.0 - EPSILON)
    return float((-(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))).mean())


def brier_score(actual: pd.Series, probability: pd.Series) -> float:
    return float(((probability.astype(float) - actual.astype(float)) ** 2).mean())


def actual_home_win(row: pd.Series) -> float:
    if row["actual_winner"] == row["home_team"]:
        return 1.0
    if row["actual_winner"] == row["away_team"]:
        return 0.0
    return 0.5


def predicted_winner_from_probability(row: pd.Series, probability_column: str) -> str:
    return row["home_team"] if float(row[probability_column]) >= 0.5 else row["away_team"]


def disagreement_tier(absolute_gap: float) -> str:
    if absolute_gap >= 0.35:
        return "extreme"
    if absolute_gap >= 0.22:
        return "high"
    if absolute_gap >= 0.10:
        return "medium"
    return "low"


def audit_category(row: pd.Series) -> str:
    if row["power_correct"] and not row["poisson_correct"]:
        return "power_right_poisson_wrong"
    if row["poisson_correct"] and not row["power_correct"]:
        return "poisson_right_power_wrong"
    if not row["power_correct"] and not row["poisson_correct"]:
        return "both_wrong"
    return "both_right"


def confidence_reduction_flag(row: pd.Series) -> bool:
    return bool(
        row["directional_disagreement"]
        and row["absolute_probability_gap"] >= 0.18
        and row["hybrid_confidence_tier"] != "low"
    )


def add_disagreement_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce")
    output["actual_home_win"] = output.apply(actual_home_win, axis=1)
    output["power_favors_home"] = output["power_v3_calibrated_home_probability"].astype(float) >= 0.5
    output["poisson_favors_home"] = output["poisson_home_win_probability"].astype(float) >= 0.5
    output["power_predicted_winner"] = output.apply(
        lambda row: predicted_winner_from_probability(row, "power_v3_calibrated_home_probability"),
        axis=1,
    )
    output["poisson_predicted_winner"] = output.apply(
        lambda row: predicted_winner_from_probability(row, "poisson_home_win_probability"),
        axis=1,
    )
    output["power_poisson_probability_gap"] = (
        output["power_v3_calibrated_home_probability"].astype(float)
        - output["poisson_home_win_probability"].astype(float)
    )
    output["absolute_probability_gap"] = output["power_poisson_probability_gap"].abs()
    output["directional_disagreement"] = output["power_favors_home"] != output["poisson_favors_home"]
    output["strong_directional_disagreement"] = (
        output["directional_disagreement"] & (output["absolute_probability_gap"] >= 0.20)
    )
    output["winner_disagreement"] = output["power_predicted_winner"] != output["poisson_predicted_winner"]
    output["projected_margin_disagreement"] = np.where(
        output["power_favors_home"] == output["poisson_favors_home"],
        "same_direction",
        "opposite_direction",
    )
    output["disagreement_tier"] = output["absolute_probability_gap"].map(disagreement_tier)
    output["power_correct"] = output["power_predicted_winner"] == output["actual_winner"]
    output["poisson_correct"] = output["poisson_predicted_winner"] == output["actual_winner"]
    output["hybrid_correct"] = output["hybrid_predicted_winner"] == output["actual_winner"]
    output["power_brier_score"] = (
        output["power_v3_calibrated_home_probability"].astype(float) - output["actual_home_win"]
    ) ** 2
    output["poisson_brier_score"] = (
        output["poisson_home_win_probability"].astype(float) - output["actual_home_win"]
    ) ** 2
    output["power_log_loss"] = -(
        output["actual_home_win"]
        * np.log(output["power_v3_calibrated_home_probability"].clip(EPSILON, 1.0 - EPSILON))
        + (1.0 - output["actual_home_win"])
        * np.log((1.0 - output["power_v3_calibrated_home_probability"]).clip(EPSILON, 1.0 - EPSILON))
    )
    output["poisson_log_loss"] = -(
        output["actual_home_win"]
        * np.log(output["poisson_home_win_probability"].clip(EPSILON, 1.0 - EPSILON))
        + (1.0 - output["actual_home_win"])
        * np.log((1.0 - output["poisson_home_win_probability"]).clip(EPSILON, 1.0 - EPSILON))
    )
    output["upset"] = output["actual_winner"] != output["hybrid_predicted_winner"]
    output["actual_margin"] = output["home_score"].astype(float) - output["away_score"].astype(float)
    output["actual_total_goals"] = output["home_score"].astype(float) + output["away_score"].astype(float)
    output["actual_close_game"] = output["actual_margin"].abs() <= 2
    output["audit_category"] = output.apply(audit_category, axis=1)
    output["should_reduce_confidence"] = output.apply(confidence_reduction_flag, axis=1)
    return output.sort_values(["game_date", "game_id"], ignore_index=True)


def metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
    groups.extend(
        ("disagreement_tier", str(tier), group)
        for tier, group in frame.groupby("disagreement_tier", observed=True)
    )
    groups.extend(
        ("directional_disagreement", str(flag), group)
        for flag, group in frame.groupby("directional_disagreement", observed=True)
    )
    groups.extend(
        ("audit_category", str(category), group)
        for category, group in frame.groupby("audit_category", observed=True)
    )
    for analysis_type, bucket, group in groups:
        rows.append(
            {
                "analysis_type": analysis_type,
                "bucket": bucket,
                "games": len(group),
                "winner_accuracy": float(group["hybrid_correct"].mean()),
                "power_winner_accuracy": float(group["power_correct"].mean()),
                "poisson_winner_accuracy": float(group["poisson_correct"].mean()),
                "brier_score": brier_score(group["actual_home_win"], group["hybrid_home_win_probability"]),
                "power_brier_score": brier_score(
                    group["actual_home_win"],
                    group["power_v3_calibrated_home_probability"],
                ),
                "poisson_brier_score": brier_score(group["actual_home_win"], group["poisson_home_win_probability"]),
                "log_loss": log_loss(group["actual_home_win"], group["hybrid_home_win_probability"]),
                "margin_mae": float(group["margin_mae"].mean()),
                "total_goals_mae": float(group["total_goals_mae"].mean()),
                "upset_rate": float(group["upset"].mean()),
                "close_game_rate": float(group["actual_close_game"].mean()),
                "average_confidence_score": float(group["hybrid_confidence_score"].mean()),
                "average_probability_gap": float(group["absolute_probability_gap"].mean()),
                "directional_disagreement_rate": float(group["directional_disagreement"].mean()),
                "should_reduce_confidence_rate": float(group["should_reduce_confidence"].mean()),
            }
        )
    return pd.DataFrame(rows)


def game_output(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_winner",
        "power_predicted_winner",
        "poisson_predicted_winner",
        "hybrid_predicted_winner",
        "power_v3_calibrated_home_probability",
        "poisson_home_win_probability",
        "power_poisson_probability_gap",
        "absolute_probability_gap",
        "power_favors_home",
        "poisson_favors_home",
        "directional_disagreement",
        "strong_directional_disagreement",
        "winner_disagreement",
        "projected_margin_disagreement",
        "disagreement_tier",
        "expected_home_goals",
        "expected_away_goals",
        "expected_total_goals",
        "expected_margin",
        "score_mae",
        "margin_mae",
        "total_goals_mae",
        "poisson_close_game_probability",
        "poisson_upset_probability",
        "hybrid_confidence_score",
        "hybrid_confidence_tier",
        "upset_risk",
        "power_correct",
        "poisson_correct",
        "hybrid_correct",
        "audit_category",
        "should_reduce_confidence",
    ]
    return frame[columns].copy()


def audit_output(frame: pd.DataFrame) -> pd.DataFrame:
    categories = [
        "power_right_poisson_wrong",
        "poisson_right_power_wrong",
        "both_wrong",
        "should_reduce_confidence",
    ]
    rows: list[pd.DataFrame] = []
    for category in categories:
        if category == "should_reduce_confidence":
            subset = frame.loc[frame["should_reduce_confidence"]].copy()
        else:
            subset = frame.loc[frame["audit_category"] == category].copy()
        if subset.empty:
            continue
        subset["audit_focus"] = category
        subset = subset.sort_values(
            ["absolute_probability_gap", "margin_mae", "total_goals_mae"],
            ascending=[False, False, False],
        )
        rows.append(subset.head(15))
    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows, ignore_index=True, sort=False)
    columns = [
        "audit_focus",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_winner",
        "power_predicted_winner",
        "poisson_predicted_winner",
        "hybrid_predicted_winner",
        "power_v3_calibrated_home_probability",
        "poisson_home_win_probability",
        "absolute_probability_gap",
        "directional_disagreement",
        "disagreement_tier",
        "expected_margin",
        "actual_margin",
        "margin_mae",
        "total_goals_mae",
        "poisson_close_game_probability",
        "poisson_upset_probability",
        "hybrid_confidence_score",
        "hybrid_confidence_tier",
        "upset_risk",
        "should_reduce_confidence",
    ]
    return output[columns].copy()


def run_disagreement_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hybrid = read_lab_output(HYBRID_INPUT)
    confidence = read_lab_output(CONFIDENCE_INPUT)[
        ["game_id", "team_stability", "model_agreement", "volatility_score", "matchup_consistency"]
    ]
    frame = hybrid.merge(confidence, on="game_id", how="left")
    features = add_disagreement_features(frame)
    return metric_rows(features), game_output(features), audit_output(features)


def main() -> None:
    analysis, games, audit = run_disagreement_analysis()
    analysis_path = write_output(analysis, DISAGREEMENT_ANALYSIS_OUTPUT)
    games_path = write_output(games, DISAGREEMENT_GAMES_OUTPUT)
    audit_path = write_output(audit, MODEL_DISAGREEMENT_AUDIT_OUTPUT)
    print(f"Wrote {analysis_path}")
    print(f"Wrote {games_path}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
