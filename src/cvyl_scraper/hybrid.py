from __future__ import annotations

from math import exp


DEFAULT_ELO_WEIGHT = 0.35
DEFAULT_POWER_V2_WEIGHT = 0.65
DEFAULT_ELO_DIFF_NORMALIZER = 400.0
DEFAULT_POWER_V2_DIFF_NORMALIZER = 4.0
DEFAULT_POWER_V2_LOGISTIC_SCALE = 4.0


def power_v2_win_probability(
    rating_difference: float,
    *,
    scale: float = DEFAULT_POWER_V2_LOGISTIC_SCALE,
) -> float:
    if scale <= 0:
        raise ValueError("scale must be greater than 0.")
    return logistic(rating_difference / scale)


def hybrid_win_probability(
    elo_difference: float,
    power_v2_difference: float,
    *,
    elo_weight: float = DEFAULT_ELO_WEIGHT,
    power_v2_weight: float = DEFAULT_POWER_V2_WEIGHT,
    elo_diff_normalizer: float = DEFAULT_ELO_DIFF_NORMALIZER,
    power_v2_diff_normalizer: float = DEFAULT_POWER_V2_DIFF_NORMALIZER,
) -> float:
    if elo_diff_normalizer <= 0:
        raise ValueError("elo_diff_normalizer must be greater than 0.")
    if power_v2_diff_normalizer <= 0:
        raise ValueError("power_v2_diff_normalizer must be greater than 0.")

    combined_edge = hybrid_model_edge(
        elo_difference,
        power_v2_difference,
        elo_weight=elo_weight,
        power_v2_weight=power_v2_weight,
        elo_diff_normalizer=elo_diff_normalizer,
        power_v2_diff_normalizer=power_v2_diff_normalizer,
    )
    return logistic(combined_edge)


def hybrid_model_edge(
    elo_difference: float,
    power_v2_difference: float,
    *,
    elo_weight: float = DEFAULT_ELO_WEIGHT,
    power_v2_weight: float = DEFAULT_POWER_V2_WEIGHT,
    elo_diff_normalizer: float = DEFAULT_ELO_DIFF_NORMALIZER,
    power_v2_diff_normalizer: float = DEFAULT_POWER_V2_DIFF_NORMALIZER,
) -> float:
    total_weight = elo_weight + power_v2_weight
    if total_weight <= 0:
        raise ValueError("combined model weights must be greater than 0.")

    normalized_elo = elo_difference / elo_diff_normalizer
    normalized_power = power_v2_difference / power_v2_diff_normalizer
    return ((elo_weight * normalized_elo) + (power_v2_weight * normalized_power)) / total_weight


def logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))
