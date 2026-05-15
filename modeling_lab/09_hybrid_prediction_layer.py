from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUTPUTS, write_output


BASELINE_INPUT = "baseline_predictions.csv"
CONFIDENCE_INPUT = "confidence_predictions.csv"
POISSON_PREDICTIONS_INPUT = "poisson_predictions_calibrated.csv"
POISSON_SIMULATION_INPUT = "poisson_simulation_summary_calibrated.csv"

HYBRID_PREDICTIONS_OUTPUT = "hybrid_predictions.csv"
HYBRID_COMPARISON_OUTPUT = "hybrid_model_comparison.csv"
HYBRID_EXPLANATIONS_OUTPUT = "hybrid_explanation_components.csv"

EPSILON = 1e-6
MATERIAL_ACCURACY_GAIN = 0.03
MATERIAL_BRIER_GAIN = 0.025


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


def actual_home_win_probability(row: pd.Series) -> float:
    if row["actual_winner"] == row["home_team"]:
        return 1.0
    if row["actual_winner"] == row["away_team"]:
        return 0.0
    return 0.5


def predicted_winner(row: pd.Series, probability_column: str) -> str:
    return row["home_team"] if float(row[probability_column]) >= 0.5 else row["away_team"]


def confidence_tier(score: float, agreement: str, upset_risk: str) -> str:
    adjusted = float(score)
    if agreement == "split":
        adjusted -= 10.0
    if upset_risk in {"elevated", "high"}:
        adjusted -= 8.0
    if adjusted >= 78:
        return "high"
    if adjusted >= 62:
        return "medium"
    return "low"


def edge_label(probability: float) -> str:
    favorite_probability = max(float(probability), 1.0 - float(probability))
    if favorite_probability >= 0.75:
        return "strong favorite"
    if favorite_probability >= 0.65:
        return "solid favorite"
    if favorite_probability >= 0.55:
        return "slight edge"
    return "toss-up"


def probability_agreement(power_probability: float, poisson_probability: float) -> str:
    power_side = "home" if power_probability >= 0.5 else "away"
    poisson_side = "home" if poisson_probability >= 0.5 else "away"
    if power_side != poisson_side:
        return "split"
    gap = abs(float(power_probability) - float(poisson_probability))
    if gap <= 0.08:
        return "strong"
    if gap <= 0.16:
        return "moderate"
    return "directional"


def model_metrics(
    frame: pd.DataFrame,
    model_name: str,
    probability_column: str,
    predicted_winner_column: str,
    calibration_column: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
    if calibration_column:
        groups.extend(
            ("confidence_tier", str(tier), group)
            for tier, group in frame.groupby(calibration_column, observed=True)
        )
    for calibration_type, bucket, group in groups:
        actual_home = group["actual_home_win"]
        probability = group[probability_column]
        rows.append(
            {
                "model": model_name,
                "calibration_type": calibration_type,
                "bucket": bucket,
                "games": len(group),
                "winner_accuracy": float((group[predicted_winner_column] == group["actual_winner"]).mean()),
                "brier_score": brier_score(actual_home, probability),
                "log_loss": log_loss(actual_home, probability),
                "score_mae": float(group["score_mae"].mean()) if "score_mae" in group else np.nan,
                "margin_mae": float(group["margin_mae"].mean()) if "margin_mae" in group else np.nan,
                "total_goals_mae": float(group["total_goals_mae"].mean())
                if "total_goals_mae" in group
                else np.nan,
                "average_confidence_score": float(group["hybrid_confidence_score"].mean())
                if "hybrid_confidence_score" in group
                else np.nan,
            }
        )
    return rows


def load_hybrid_inputs() -> pd.DataFrame:
    baseline = read_lab_output(BASELINE_INPUT)
    confidence = read_lab_output(CONFIDENCE_INPUT)
    poisson_predictions = read_lab_output(POISSON_PREDICTIONS_INPUT)
    poisson_simulation = read_lab_output(POISSON_SIMULATION_INPUT)

    columns = ["game_id", "game_date", "home_team", "away_team", "actual_winner"]
    power = baseline[
        columns
        + [
            "power_v3_calibrated_home_probability",
            "power_v3_calibrated_predicted_winner",
            "elo_home_probability",
        ]
    ].copy()
    reliability = confidence[
        [
            "game_id",
            "confidence_score",
            "confidence_tier",
            "upset_risk",
            "model_agreement",
            "volatility_score",
            "strength_differential_score",
            "matchup_consistency",
        ]
    ].copy()
    scores = poisson_predictions[
        [
            "game_id",
            "home_score",
            "away_score",
            "expected_home_goals",
            "expected_away_goals",
            "expected_total_goals",
            "expected_margin",
            "score_mae",
            "margin_mae",
            "total_goals_mae",
        ]
    ].copy()
    simulation = poisson_simulation[
        [
            "game_id",
            "home_win_probability",
            "predicted_winner",
            "most_likely_score",
            "upset_probability",
            "close_game_probability",
        ]
    ].rename(
        columns={
            "home_win_probability": "poisson_home_win_probability",
            "predicted_winner": "poisson_predicted_winner",
            "upset_probability": "poisson_upset_probability",
            "close_game_probability": "poisson_close_game_probability",
        }
    )

    frame = power.merge(reliability, on="game_id", how="inner")
    frame = frame.merge(scores, on="game_id", how="inner")
    frame = frame.merge(simulation, on="game_id", how="inner")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame.sort_values(["game_date", "game_id"], ignore_index=True)


def should_poisson_override(frame: pd.DataFrame) -> bool:
    actual = frame["actual_home_win"]
    power_accuracy = (frame["power_v3_calibrated_predicted_winner"] == frame["actual_winner"]).mean()
    poisson_accuracy = (frame["poisson_predicted_winner"] == frame["actual_winner"]).mean()
    power_brier = brier_score(actual, frame["power_v3_calibrated_home_probability"])
    poisson_brier = brier_score(actual, frame["poisson_home_win_probability"])
    return bool(
        poisson_accuracy >= power_accuracy + MATERIAL_ACCURACY_GAIN
        and poisson_brier <= power_brier - MATERIAL_BRIER_GAIN
    )


def build_hybrid_predictions() -> pd.DataFrame:
    frame = load_hybrid_inputs()
    frame["actual_home_win"] = frame.apply(actual_home_win_probability, axis=1)
    poisson_materially_better = should_poisson_override(frame)

    # Power v3 calibrated remains the winner-probability anchor unless Poisson
    # clears a deliberately high validation bar. Current lab outputs use Poisson
    # for score distribution and reliability context, not winner override.
    if poisson_materially_better:
        frame["hybrid_home_win_probability"] = (
            0.85 * frame["poisson_home_win_probability"]
            + 0.15 * frame["power_v3_calibrated_home_probability"]
        )
        frame["winner_probability_source"] = "poisson_validated_override"
    else:
        frame["hybrid_home_win_probability"] = frame["power_v3_calibrated_home_probability"]
        frame["winner_probability_source"] = "power_v3_calibrated"

    frame["hybrid_predicted_winner"] = frame.apply(
        lambda row: predicted_winner(row, "hybrid_home_win_probability"),
        axis=1,
    )
    frame["poisson_power_agreement"] = frame.apply(
        lambda row: probability_agreement(
            row["power_v3_calibrated_home_probability"],
            row["poisson_home_win_probability"],
        ),
        axis=1,
    )
    frame["hybrid_confidence_tier"] = frame.apply(
        lambda row: confidence_tier(
            row["confidence_score"],
            row["poisson_power_agreement"],
            row["upset_risk"],
        ),
        axis=1,
    )
    agreement_bonus = frame["poisson_power_agreement"].map(
        {"strong": 8.0, "moderate": 4.0, "directional": 0.0, "split": -12.0}
    )
    risk_penalty = frame["upset_risk"].map({"low": 0.0, "moderate": -3.0, "elevated": -8.0, "high": -14.0})
    frame["hybrid_confidence_score"] = (
        frame["confidence_score"].astype(float) + agreement_bonus.fillna(0.0) + risk_penalty.fillna(0.0)
    ).clip(0.0, 100.0)
    frame["hybrid_edge_label"] = frame["hybrid_home_win_probability"].map(edge_label)
    frame["hybrid_winner_correct"] = frame["hybrid_predicted_winner"] == frame["actual_winner"]
    frame["hybrid_brier_score"] = (
        frame["hybrid_home_win_probability"].astype(float) - frame["actual_home_win"].astype(float)
    ) ** 2
    frame["hybrid_log_loss"] = -(
        frame["actual_home_win"] * np.log(frame["hybrid_home_win_probability"].clip(EPSILON, 1.0 - EPSILON))
        + (1.0 - frame["actual_home_win"])
        * np.log((1.0 - frame["hybrid_home_win_probability"]).clip(EPSILON, 1.0 - EPSILON))
    )
    return frame


def hybrid_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_winner",
        "hybrid_predicted_winner",
        "hybrid_home_win_probability",
        "power_v3_calibrated_home_probability",
        "poisson_home_win_probability",
        "winner_probability_source",
        "expected_home_goals",
        "expected_away_goals",
        "expected_total_goals",
        "expected_margin",
        "most_likely_score",
        "poisson_close_game_probability",
        "poisson_upset_probability",
        "confidence_score",
        "hybrid_confidence_score",
        "confidence_tier",
        "hybrid_confidence_tier",
        "upset_risk",
        "poisson_power_agreement",
        "hybrid_edge_label",
        "hybrid_winner_correct",
        "hybrid_brier_score",
        "hybrid_log_loss",
        "score_mae",
        "margin_mae",
        "total_goals_mae",
    ]
    return frame[columns].copy()


def build_model_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    comparison_frame = frame.copy()
    comparison_frame["power_v3_predicted_winner"] = comparison_frame.apply(
        lambda row: predicted_winner(row, "power_v3_calibrated_home_probability"),
        axis=1,
    )
    comparison_frame["poisson_predicted_winner_from_probability"] = comparison_frame.apply(
        lambda row: predicted_winner(row, "poisson_home_win_probability"),
        axis=1,
    )
    rows = []
    rows.extend(
        model_metrics(
            comparison_frame,
            "power_v3_calibrated",
            "power_v3_calibrated_home_probability",
            "power_v3_predicted_winner",
            "hybrid_confidence_tier",
        )
    )
    rows.extend(
        model_metrics(
            comparison_frame,
            "calibrated_poisson",
            "poisson_home_win_probability",
            "poisson_predicted_winner_from_probability",
            "hybrid_confidence_tier",
        )
    )
    rows.extend(
        model_metrics(
            comparison_frame,
            "hybrid",
            "hybrid_home_win_probability",
            "hybrid_predicted_winner",
            "hybrid_confidence_tier",
        )
    )
    return pd.DataFrame(rows)


def explanation_components(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        [
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "hybrid_predicted_winner",
            "hybrid_edge_label",
            "hybrid_confidence_tier",
            "upset_risk",
            "poisson_power_agreement",
            "expected_home_goals",
            "expected_away_goals",
            "expected_total_goals",
            "expected_margin",
            "most_likely_score",
            "poisson_close_game_probability",
            "poisson_upset_probability",
            "model_agreement",
            "volatility_score",
            "strength_differential_score",
            "matchup_consistency",
        ]
    ].copy()
    output["score_projection_component"] = output.apply(
        lambda row: f"{row['home_team']} {row['expected_home_goals']:.1f}, "
        f"{row['away_team']} {row['expected_away_goals']:.1f}",
        axis=1,
    )
    output["confidence_component"] = output.apply(
        lambda row: f"{row['hybrid_confidence_tier']} confidence; "
        f"{row['upset_risk']} upset risk; {row['poisson_power_agreement']} model agreement",
        axis=1,
    )
    output["distribution_component"] = output.apply(
        lambda row: f"{row['poisson_close_game_probability']:.1%} close-game probability; "
        f"{row['poisson_upset_probability']:.1%} simulation upset probability",
        axis=1,
    )
    return output


def run_hybrid_prediction_layer() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = build_hybrid_predictions()
    return (
        hybrid_output_columns(predictions),
        build_model_comparison(predictions),
        explanation_components(predictions),
    )


def main() -> None:
    predictions, comparison, explanations = run_hybrid_prediction_layer()
    predictions_path = write_output(predictions, HYBRID_PREDICTIONS_OUTPUT)
    comparison_path = write_output(comparison, HYBRID_COMPARISON_OUTPUT)
    explanations_path = write_output(explanations, HYBRID_EXPLANATIONS_OUTPUT)
    print(f"Wrote {predictions_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {explanations_path}")


if __name__ == "__main__":
    main()
