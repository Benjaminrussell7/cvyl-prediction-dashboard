from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import OUTPUTS, write_output


MODEL_COMPARISON_INPUT = "model_comparison.csv"
POISSON_COMPARISON_INPUT = "poisson_calibration_comparison.csv"
HYBRID_COMPARISON_INPUT = "hybrid_model_comparison.csv"
DISAGREEMENT_INPUT = "disagreement_analysis.csv"
ARCHETYPE_SUMMARY_INPUT = "archetype_performance_summary.csv"

SUMMARY_OUTPUT = "modeling_lab_summary.csv"
READINESS_OUTPUT = "production_readiness_recommendations.csv"
KEY_FINDINGS_OUTPUT = "modeling_lab_key_findings.md"


def read_lab_output(filename: str) -> pd.DataFrame:
    path = OUTPUTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing modeling lab input: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().removeprefix("\ufeff") for column in frame.columns]
    return frame


def metric(row: pd.Series, column: str) -> float:
    return float(row[column])


def overall(frame: pd.DataFrame, **filters: str) -> pd.DataFrame:
    output = frame.copy()
    for column, value in filters.items():
        output = output.loc[output[column] == value]
    if output.empty:
        raise ValueError(f"Missing summary row for filters: {filters}")
    return output


def first_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("Expected a non-empty dataframe")
    return frame.iloc[0]


def pct(value: float) -> str:
    return f"{value:.1%}"


def number(value: float) -> str:
    return f"{value:.3f}"


def build_decision_summary(
    model_comparison: pd.DataFrame,
    poisson_comparison: pd.DataFrame,
    hybrid_comparison: pd.DataFrame,
    disagreement: pd.DataFrame,
    archetypes: pd.DataFrame,
) -> pd.DataFrame:
    model_overall = model_comparison.sort_values(
        ["accuracy", "brier_score", "log_loss"],
        ascending=[False, True, True],
    )
    best_winner = first_row(model_overall)
    best_calibration = first_row(model_comparison.sort_values(["brier_score", "log_loss"]))
    best_score = first_row(
        overall(poisson_comparison, calibration_type="overall", bucket="all").sort_values(
            ["score_mae", "total_goals_mae", "margin_mae"]
        )
    )
    hybrid_overall = first_row(overall(hybrid_comparison, model="hybrid", calibration_type="overall", bucket="all"))
    power_hybrid = first_row(
        overall(hybrid_comparison, model="power_v3_calibrated", calibration_type="overall", bucket="all")
    )
    disagreement_overall = first_row(overall(disagreement, analysis_type="overall", bucket="all"))
    reliable_archetype = first_row(
        overall(archetypes, analysis_type="archetype")
        .loc[lambda frame: frame["games"] >= 5]
        .sort_values(["winner_accuracy", "brier_score"], ascending=[False, True])
    )
    volatile_archetype = first_row(
        overall(archetypes, analysis_type="archetype")
        .loc[lambda frame: frame["games"] >= 5]
        .sort_values(["margin_mae", "upset_rate"], ascending=[False, False])
    )
    upset_archetype = first_row(
        overall(archetypes, analysis_type="archetype")
        .loc[lambda frame: frame["games"] >= 5]
        .sort_values(["upset_rate", "winner_accuracy"], ascending=[False, True])
    )

    rows = [
        {
            "decision_area": "winner_prediction",
            "best_current_choice": best_winner["model"],
            "status": "production_ready",
            "evidence": (
                f"Highest chronological test accuracy at {pct(metric(best_winner, 'accuracy'))}; "
                f"Brier {number(metric(best_winner, 'brier_score'))}."
            ),
            "recommendation": "Keep calibrated Power v3 as the primary winner probability model.",
        },
        {
            "decision_area": "probability_calibration",
            "best_current_choice": best_calibration["model"],
            "status": "production_ready",
            "evidence": (
                f"Lowest Brier score in model comparison at {number(metric(best_calibration, 'brier_score'))}; "
                f"log loss {number(metric(best_calibration, 'log_loss'))}."
            ),
            "recommendation": "Use calibrated Power v3 probabilities for displayed win probabilities.",
        },
        {
            "decision_area": "score_projection",
            "best_current_choice": best_score["model"],
            "status": "experimental",
            "evidence": (
                f"Calibrated Poisson score MAE {number(metric(best_score, 'score_mae'))}; "
                f"total goals MAE {number(metric(best_score, 'total_goals_mae'))}."
            ),
            "recommendation": "Use calibrated Poisson as score context only, not as the winner engine.",
        },
        {
            "decision_area": "confidence_uncertainty",
            "best_current_choice": "hybrid confidence + disagreement analysis",
            "status": "experimental_but_useful",
            "evidence": (
                f"Hybrid confidence tiers segment outcomes; overall hybrid accuracy "
                f"{pct(metric(hybrid_overall, 'winner_accuracy'))}. "
                f"Directional disagreement rate is {pct(metric(disagreement_overall, 'directional_disagreement_rate'))}."
            ),
            "recommendation": "Use confidence and disagreement as warning labels, not as automatic model overrides.",
        },
        {
            "decision_area": "reliable_archetypes",
            "best_current_choice": reliable_archetype["archetype"],
            "status": "production_candidate",
            "evidence": (
                f"{reliable_archetype['archetype']} accuracy {pct(metric(reliable_archetype, 'winner_accuracy'))} "
                f"across {int(reliable_archetype['games'])} games."
            ),
            "recommendation": "Use reliable archetypes to explain when model confidence is more trustworthy.",
        },
        {
            "decision_area": "volatile_archetypes",
            "best_current_choice": volatile_archetype["archetype"],
            "status": "warning_signal",
            "evidence": (
                f"{volatile_archetype['archetype']} margin MAE {number(metric(volatile_archetype, 'margin_mae'))}; "
                f"upset rate {pct(metric(volatile_archetype, 'upset_rate'))}."
            ),
            "recommendation": "Show volatile archetypes as caution flags, not strong picks.",
        },
        {
            "decision_area": "upset_archetypes",
            "best_current_choice": upset_archetype["archetype"],
            "status": "warning_signal",
            "evidence": (
                f"{upset_archetype['archetype']} upset rate {pct(metric(upset_archetype, 'upset_rate'))}."
            ),
            "recommendation": "Use upset-prone archetypes to lower narrative certainty.",
        },
        {
            "decision_area": "recommended_architecture",
            "best_current_choice": "Power v3 calibrated + Poisson score context + confidence warnings",
            "status": "recommended",
            "evidence": (
                f"Power/hybrid winner accuracy {pct(metric(power_hybrid, 'winner_accuracy'))}; "
                f"Poisson adds score distribution but has weaker standalone winner accuracy."
            ),
            "recommendation": "Promote Power v3 calibrated for probabilities, calibrated Poisson for score projections, and confidence/disagreement/archetype layers for interpretation.",
        },
    ]
    return pd.DataFrame(rows)


def build_readiness_recommendations(
    model_comparison: pd.DataFrame,
    poisson_comparison: pd.DataFrame,
    hybrid_comparison: pd.DataFrame,
    disagreement: pd.DataFrame,
    archetypes: pd.DataFrame,
) -> pd.DataFrame:
    calibrated_poisson = first_row(
        overall(poisson_comparison, model="calibrated_poisson", calibration_type="overall", bucket="all")
    )
    raw_poisson = first_row(overall(poisson_comparison, model="raw_poisson", calibration_type="overall", bucket="all"))
    poisson_hybrid = first_row(
        overall(hybrid_comparison, model="calibrated_poisson", calibration_type="overall", bucket="all")
    )
    high_disagreement = first_row(
        overall(disagreement, analysis_type="disagreement_tier", bucket="high")
    )
    extreme_disagreement = first_row(
        overall(disagreement, analysis_type="disagreement_tier", bucket="extreme")
    )
    misleading = (
        overall(archetypes, analysis_type="archetype")
        .loc[lambda frame: frame["games"] >= 5]
        .sort_values(["misleading_confidence_rate", "average_confidence"], ascending=[False, False])
    )
    misleading_row = first_row(misleading)

    rows = [
        {
            "component": "calibrated_power_v3_probability",
            "readiness": "production_ready",
            "use_now": "yes",
            "recommended_role": "Primary winner probability and displayed model confidence.",
            "rationale": "Best current blend of accuracy, Brier score, and interpretability.",
        },
        {
            "component": "calibrated_poisson_scores",
            "readiness": "experimental",
            "use_now": "limited",
            "recommended_role": "Expected score, total, and simulation distribution context.",
            "rationale": (
                f"Calibration improves score MAE from {number(metric(raw_poisson, 'score_mae'))} "
                f"to {number(metric(calibrated_poisson, 'score_mae'))}, but standalone winner accuracy is only "
                f"{pct(metric(poisson_hybrid, 'winner_accuracy'))}."
            ),
        },
        {
            "component": "hybrid_prediction_layer",
            "readiness": "experimental_but_safe",
            "use_now": "yes_for_analysis",
            "recommended_role": "Decision layer that keeps Power v3 as anchor and attaches Poisson/confidence context.",
            "rationale": "Does not let Poisson override winner probability unless validation materially improves.",
        },
        {
            "component": "confidence_engine",
            "readiness": "experimental_but_useful",
            "use_now": "yes_as_warning_label",
            "recommended_role": "Reliability tier and upset-risk language.",
            "rationale": "Useful for reducing certainty in low-confidence and high-disagreement games.",
        },
        {
            "component": "disagreement_analysis",
            "readiness": "production_candidate",
            "use_now": "yes_as_guardrail",
            "recommended_role": "Warn when Power and Poisson disagree or when confidence should be reduced.",
            "rationale": (
                f"High disagreement accuracy is {pct(metric(high_disagreement, 'winner_accuracy'))}; "
                f"extreme disagreement average confidence is {number(metric(extreme_disagreement, 'average_confidence_score'))}."
            ),
        },
        {
            "component": "archetype_labels",
            "readiness": "production_candidate",
            "use_now": "yes_for_explanations",
            "recommended_role": "Short interpretation labels for reliable, volatile, upset-prone, and toss-up games.",
            "rationale": (
                f"{misleading_row['archetype']} has misleading-confidence rate "
                f"{pct(metric(misleading_row, 'misleading_confidence_rate'))}; use archetypes to temper language."
            ),
        },
        {
            "component": "advanced_ml_baselines",
            "readiness": "not_recommended_yet",
            "use_now": "no",
            "recommended_role": "Research only.",
            "rationale": "Current Ridge/logistic/random-forest baselines do not clearly beat calibrated Power v3.",
        },
        {
            "component": "raw_poisson_probabilities",
            "readiness": "not_recommended",
            "use_now": "no",
            "recommended_role": "Audit baseline only.",
            "rationale": "Raw Poisson produced unrealistic extremes and worse score/winner calibration.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in values.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def build_markdown_report(summary: pd.DataFrame, readiness: pd.DataFrame, archetypes: pd.DataFrame) -> str:
    reliable = (
        overall(archetypes, analysis_type="archetype")
        .loc[lambda frame: frame["games"] >= 5]
        .sort_values(["winner_accuracy", "brier_score"], ascending=[False, True])
        .head(5)
    )
    volatile = (
        overall(archetypes, analysis_type="archetype")
        .loc[lambda frame: frame["games"] >= 5]
        .sort_values(["margin_mae", "upset_rate"], ascending=[False, False])
        .head(5)
    )
    not_ready = readiness.loc[readiness["readiness"].isin(["not_recommended", "not_recommended_yet"])]

    lines = [
        "# Modeling Lab Key Findings",
        "",
        "## Executive Decision",
        "",
        "Use calibrated Power v3 as the production winner-probability engine. Use calibrated Poisson only for expected score and distribution context. Use confidence, disagreement, and archetype layers as interpretation and caution signals.",
        "",
        "## Decision Summary",
        "",
        markdown_table(summary, ["decision_area", "best_current_choice", "status", "recommendation"]),
        "",
        "## Production Readiness",
        "",
        markdown_table(readiness, ["component", "readiness", "use_now", "recommended_role"]),
        "",
        "## Reliable Archetypes",
        "",
        markdown_table(
            reliable,
            ["archetype", "games", "winner_accuracy", "brier_score", "upset_rate", "average_confidence"],
        ),
        "",
        "## Volatile Archetypes",
        "",
        markdown_table(
            volatile,
            ["archetype", "games", "margin_mae", "upset_rate", "close_game_rate", "average_confidence"],
        ),
        "",
        "## Do Not Promote Yet",
        "",
        markdown_table(not_ready, ["component", "readiness", "rationale"]),
        "",
        "## Next Experiments",
        "",
        "- Tune calibrated Poisson score projections with more conservative total-goals and margin calibration.",
        "- Test whether disagreement-aware confidence thresholds improve user-facing warning quality.",
        "- Backtest archetype-specific probability calibration instead of one global calibration curve.",
        "- Add more data before promoting ML baselines; current samples are too small for reliable nonlinear model selection.",
        "- Evaluate whether Poisson score projections improve tournament simulation realism without affecting winner probabilities.",
        "",
    ]
    return "\n".join(lines)


def run_modeling_lab_summary() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    model_comparison = read_lab_output(MODEL_COMPARISON_INPUT)
    poisson_comparison = read_lab_output(POISSON_COMPARISON_INPUT)
    hybrid_comparison = read_lab_output(HYBRID_COMPARISON_INPUT)
    disagreement = read_lab_output(DISAGREEMENT_INPUT)
    archetypes = read_lab_output(ARCHETYPE_SUMMARY_INPUT)

    summary = build_decision_summary(
        model_comparison,
        poisson_comparison,
        hybrid_comparison,
        disagreement,
        archetypes,
    )
    readiness = build_readiness_recommendations(
        model_comparison,
        poisson_comparison,
        hybrid_comparison,
        disagreement,
        archetypes,
    )
    report = build_markdown_report(summary, readiness, archetypes)
    return summary, readiness, report


def write_markdown_report(content: str, filename: str) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / filename
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    summary, readiness, report = run_modeling_lab_summary()
    summary_path = write_output(summary, SUMMARY_OUTPUT)
    readiness_path = write_output(readiness, READINESS_OUTPUT)
    report_path = write_markdown_report(report, KEY_FINDINGS_OUTPUT)
    print(f"Wrote {summary_path}")
    print(f"Wrote {readiness_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
