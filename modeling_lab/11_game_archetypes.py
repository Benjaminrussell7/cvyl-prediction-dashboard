from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUTPUTS, write_output


HYBRID_INPUT = "hybrid_predictions.csv"
DISAGREEMENT_ANALYSIS_INPUT = "disagreement_analysis.csv"
DISAGREEMENT_GAMES_INPUT = "disagreement_games.csv"

GAME_ARCHETYPES_OUTPUT = "game_archetypes.csv"
ARCHETYPE_SUMMARY_OUTPUT = "archetype_performance_summary.csv"
ARCHETYPE_EXAMPLES_OUTPUT = "archetype_examples.csv"

EPSILON = 1e-6


ARCHETYPES = [
    "stable_favorite",
    "volatile_favorite",
    "toss_up",
    "deceptive_toss_up",
    "offensive_shootout",
    "defensive_grinder",
    "upset_prone",
    "high_variance",
    "fake_blowout",
    "consensus_pick",
]


def read_lab_output(filename: str) -> pd.DataFrame:
    path = OUTPUTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing modeling lab input: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().removeprefix("\ufeff") for column in frame.columns]
    return frame


def actual_home_win(row: pd.Series) -> float:
    if row["actual_winner"] == row["home_team"]:
        return 1.0
    if row["actual_winner"] == row["away_team"]:
        return 0.0
    return 0.5


def brier_score(actual: pd.Series, probability: pd.Series) -> float:
    return float(((probability.astype(float) - actual.astype(float)) ** 2).mean())


def log_loss(actual: pd.Series, probability: pd.Series) -> float:
    clipped = probability.astype(float).clip(EPSILON, 1.0 - EPSILON)
    return float((-(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))).mean())


def calibration_gap(actual: pd.Series, probability: pd.Series) -> float:
    favorite_probability = np.where(probability.astype(float) >= 0.5, probability, 1.0 - probability)
    favorite_won = np.where(probability.astype(float) >= 0.5, actual, 1.0 - actual)
    return float(pd.Series(favorite_probability).mean() - pd.Series(favorite_won).mean())


def favorite_probability(row: pd.Series) -> float:
    return max(float(row["hybrid_home_win_probability"]), 1.0 - float(row["hybrid_home_win_probability"]))


def classify_archetypes(row: pd.Series) -> list[str]:
    favorite_prob = float(row["favorite_probability"])
    expected_total = float(row["expected_total_goals"])
    expected_margin = abs(float(row["expected_margin"]))
    close_prob = float(row["poisson_close_game_probability"])
    upset_prob = float(row["poisson_upset_probability"])
    confidence = float(row["hybrid_confidence_score"])
    gap = float(row["absolute_probability_gap"])
    directional_disagreement = bool(row["directional_disagreement"])
    tier = str(row["disagreement_tier"])

    labels: list[str] = []
    if favorite_prob >= 0.64 and confidence >= 62 and not directional_disagreement and tier in {"low", "medium"}:
        labels.append("stable_favorite")
    if favorite_prob >= 0.62 and (directional_disagreement or tier in {"high", "extreme"}):
        labels.append("volatile_favorite")
    if favorite_prob < 0.55 and close_prob >= 0.34:
        labels.append("toss_up")
    if favorite_prob < 0.58 and (gap >= 0.18 or directional_disagreement):
        labels.append("deceptive_toss_up")
    if expected_total >= 14.0:
        labels.append("offensive_shootout")
    if expected_total <= 9.0 and close_prob >= 0.28:
        labels.append("defensive_grinder")
    if upset_prob >= 0.25 or str(row["upset_risk"]) in {"elevated", "high"}:
        labels.append("upset_prone")
    if tier in {"high", "extreme"} or gap >= 0.22 or confidence < 45:
        labels.append("high_variance")
    if expected_margin >= 6.0 and (close_prob >= 0.20 or directional_disagreement or favorite_prob < 0.62):
        labels.append("fake_blowout")
    if (
        not directional_disagreement
        and favorite_prob >= 0.62
        and gap <= 0.12
        and confidence >= 58
    ):
        labels.append("consensus_pick")
    if not labels:
        labels.append("toss_up" if favorite_prob < 0.58 else "volatile_favorite")
    return labels


def primary_archetype(labels: list[str]) -> str:
    priority = [
        "deceptive_toss_up",
        "fake_blowout",
        "high_variance",
        "upset_prone",
        "volatile_favorite",
        "offensive_shootout",
        "defensive_grinder",
        "stable_favorite",
        "consensus_pick",
        "toss_up",
    ]
    for archetype in priority:
        if archetype in labels:
            return archetype
    return labels[0]


def load_archetype_frame() -> pd.DataFrame:
    # Load all requested inputs. The summary input is retained as a contract check
    # for this standalone layer even though game-level classification uses rows.
    analysis = read_lab_output(DISAGREEMENT_ANALYSIS_INPUT)
    if analysis.empty:
        raise ValueError("disagreement_analysis.csv is empty")
    hybrid = read_lab_output(HYBRID_INPUT)
    disagreement = read_lab_output(DISAGREEMENT_GAMES_INPUT)
    keep = [
        "game_id",
        "power_poisson_probability_gap",
        "absolute_probability_gap",
        "power_favors_home",
        "poisson_favors_home",
        "directional_disagreement",
        "strong_directional_disagreement",
        "winner_disagreement",
        "projected_margin_disagreement",
        "disagreement_tier",
        "power_correct",
        "poisson_correct",
        "audit_category",
        "should_reduce_confidence",
    ]
    frame = hybrid.merge(disagreement[keep], on="game_id", how="inner")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["actual_home_win"] = frame.apply(actual_home_win, axis=1)
    frame["favorite_probability"] = frame.apply(favorite_probability, axis=1)
    frame["favorite_won"] = frame["hybrid_winner_correct"].astype(bool)
    frame["actual_margin"] = frame["home_score"].astype(float) - frame["away_score"].astype(float)
    frame["actual_total_goals"] = frame["home_score"].astype(float) + frame["away_score"].astype(float)
    frame["actual_close_game"] = frame["actual_margin"].abs() <= 2
    frame["actual_upset"] = ~frame["hybrid_winner_correct"].astype(bool)
    labels = frame.apply(classify_archetypes, axis=1)
    frame["archetypes"] = labels.map(lambda values: "|".join(values))
    frame["primary_archetype"] = labels.map(primary_archetype)
    for archetype in ARCHETYPES:
        frame[f"is_{archetype}"] = labels.map(lambda values, target=archetype: target in values)
    return frame.sort_values(["game_date", "game_id"], ignore_index=True)


def archetype_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["directional_disagreement"]:
        reasons.append("Power and Poisson favor opposite teams")
    if row["disagreement_tier"] in {"high", "extreme"}:
        reasons.append(f"{row['disagreement_tier']} probability disagreement")
    if row["favorite_probability"] < 0.55:
        reasons.append("winner probability is near 50/50")
    if row["expected_total_goals"] >= 14:
        reasons.append("high projected scoring environment")
    if row["expected_total_goals"] <= 9:
        reasons.append("low projected scoring environment")
    if row["poisson_upset_probability"] >= 0.25:
        reasons.append("Poisson simulations show notable upset risk")
    if row["poisson_close_game_probability"] >= 0.34:
        reasons.append("close-game probability is elevated")
    if row["hybrid_confidence_score"] >= 70:
        reasons.append("confidence score is strong")
    if not reasons:
        reasons.append("balanced confidence and scoring signals")
    return "; ".join(reasons)


def game_archetypes_output(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["archetype_reason"] = output.apply(archetype_reason, axis=1)
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
        "favorite_probability",
        "expected_home_goals",
        "expected_away_goals",
        "expected_total_goals",
        "expected_margin",
        "poisson_close_game_probability",
        "poisson_upset_probability",
        "hybrid_confidence_score",
        "hybrid_confidence_tier",
        "directional_disagreement",
        "absolute_probability_gap",
        "disagreement_tier",
        "primary_archetype",
        "archetypes",
        "archetype_reason",
        "hybrid_winner_correct",
        "score_mae",
        "margin_mae",
        "total_goals_mae",
        "actual_upset",
        "actual_close_game",
    ]
    columns.extend([f"is_{archetype}" for archetype in ARCHETYPES])
    return output[columns].copy()


def summary_row(frame: pd.DataFrame, analysis_type: str, archetype: str) -> dict[str, object]:
    if frame.empty:
        return {
            "analysis_type": analysis_type,
            "archetype": archetype,
            "games": 0,
            "winner_accuracy": np.nan,
            "margin_mae": np.nan,
            "total_goals_mae": np.nan,
            "upset_rate": np.nan,
            "close_game_rate": np.nan,
            "average_confidence": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
            "calibration_gap": np.nan,
            "misleading_confidence_rate": np.nan,
        }
    return {
        "analysis_type": analysis_type,
        "archetype": archetype,
        "games": len(frame),
        "winner_accuracy": float(frame["hybrid_winner_correct"].mean()),
        "margin_mae": float(frame["margin_mae"].mean()),
        "total_goals_mae": float(frame["total_goals_mae"].mean()),
        "upset_rate": float(frame["actual_upset"].mean()),
        "close_game_rate": float(frame["actual_close_game"].mean()),
        "average_confidence": float(frame["hybrid_confidence_score"].mean()),
        "brier_score": brier_score(frame["actual_home_win"], frame["hybrid_home_win_probability"]),
        "log_loss": log_loss(frame["actual_home_win"], frame["hybrid_home_win_probability"]),
        "calibration_gap": calibration_gap(frame["actual_home_win"], frame["hybrid_home_win_probability"]),
        "misleading_confidence_rate": float(
            ((frame["hybrid_confidence_score"] >= 65) & (~frame["hybrid_winner_correct"].astype(bool))).mean()
        ),
    }


def archetype_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [summary_row(frame, "overall", "all")]
    for archetype in ARCHETYPES:
        rows.append(summary_row(frame.loc[frame[f"is_{archetype}"]], "archetype", archetype))
    for archetype, group in frame.groupby("primary_archetype", observed=True):
        rows.append(summary_row(group, "primary_archetype", str(archetype)))
    summary = pd.DataFrame(rows)

    archetype_rows = summary.loc[summary["analysis_type"] == "archetype"].copy()
    reliable = archetype_rows.loc[archetype_rows["games"] >= 3].sort_values(
        ["winner_accuracy", "brier_score"],
        ascending=[False, True],
    )
    volatile = archetype_rows.loc[archetype_rows["games"] >= 3].sort_values(
        ["margin_mae", "upset_rate"],
        ascending=[False, False],
    )
    upset = archetype_rows.loc[archetype_rows["games"] >= 3].sort_values(
        ["upset_rate", "winner_accuracy"],
        ascending=[False, True],
    )
    misleading = archetype_rows.loc[archetype_rows["games"] >= 3].sort_values(
        ["misleading_confidence_rate", "average_confidence"],
        ascending=[False, False],
    )
    tags = []
    for label, data in [
        ("most_reliable", reliable.head(3)),
        ("most_volatile", volatile.head(3)),
        ("strongest_upset_frequency", upset.head(3)),
        ("confidence_misleading", misleading.head(3)),
    ]:
        tagged = data.copy()
        tagged["analysis_type"] = label
        tags.append(tagged)
    return pd.concat([summary, *tags], ignore_index=True, sort=False)


def archetype_examples(frame: pd.DataFrame) -> pd.DataFrame:
    examples: list[pd.DataFrame] = []
    for archetype in ARCHETYPES:
        subset = frame.loc[frame[f"is_{archetype}"]].copy()
        if subset.empty:
            continue
        subset["example_archetype"] = archetype
        subset["example_score"] = (
            subset["absolute_probability_gap"].astype(float)
            + subset["poisson_upset_probability"].astype(float)
            + subset["margin_mae"].astype(float) / 20.0
            + subset["total_goals_mae"].astype(float) / 20.0
        )
        if archetype in {"stable_favorite", "consensus_pick"}:
            subset["example_score"] = (
                subset["hybrid_confidence_score"].astype(float) / 100.0
                + subset["favorite_probability"].astype(float)
                - subset["absolute_probability_gap"].astype(float)
            )
        subset = subset.sort_values("example_score", ascending=False).head(5)
        examples.append(subset)
    if not examples:
        return pd.DataFrame()
    output = pd.concat(examples, ignore_index=True, sort=False)
    output["archetype_reason"] = output.apply(archetype_reason, axis=1)
    columns = [
        "example_archetype",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_winner",
        "hybrid_predicted_winner",
        "hybrid_home_win_probability",
        "favorite_probability",
        "expected_total_goals",
        "expected_margin",
        "poisson_close_game_probability",
        "poisson_upset_probability",
        "hybrid_confidence_score",
        "directional_disagreement",
        "absolute_probability_gap",
        "disagreement_tier",
        "primary_archetype",
        "archetypes",
        "archetype_reason",
        "hybrid_winner_correct",
        "margin_mae",
        "total_goals_mae",
        "actual_upset",
        "actual_close_game",
    ]
    return output[columns].copy()


def run_game_archetypes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_archetype_frame()
    return game_archetypes_output(frame), archetype_summary(frame), archetype_examples(frame)


def main() -> None:
    games, summary, examples = run_game_archetypes()
    games_path = write_output(games, GAME_ARCHETYPES_OUTPUT)
    summary_path = write_output(summary, ARCHETYPE_SUMMARY_OUTPUT)
    examples_path = write_output(examples, ARCHETYPE_EXAMPLES_OUTPUT)
    print(f"Wrote {games_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {examples_path}")


if __name__ == "__main__":
    main()
