from __future__ import annotations

import pandas as pd

from cvyl_scraper.calibration import build_power_rating_calibration


def test_power_rating_calibration_creates_expected_buckets() -> None:
    calibration = build_power_rating_calibration(
        pd.DataFrame(
            [
                _comparison_row(0.50, True),
                _comparison_row(0.55, True),
                _comparison_row(0.60, False),
                _comparison_row(0.65, True),
                _comparison_row(0.70, False),
                _comparison_row(0.75, True),
                _comparison_row(0.90, False),
            ]
        )
    )

    assert list(calibration["bucket"]) == [
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75%+",
    ]
    assert list(calibration["games"]) == [1, 1, 1, 1, 1, 2]


def test_power_rating_calibration_calculates_bucket_metrics() -> None:
    calibration = build_power_rating_calibration(
        pd.DataFrame(
            [
                _comparison_row(0.56, True),
                _comparison_row(0.58, False),
                _comparison_row(0.59, True),
            ]
        )
    )
    bucket = calibration[calibration["bucket"] == "55-60%"].iloc[0]

    assert bucket["games"] == 3
    assert round(bucket["average_predicted_probability"], 4) == round((0.56 + 0.58 + 0.59) / 3, 4)
    assert round(bucket["actual_win_rate"], 4) == round(2 / 3, 4)
    assert round(bucket["calibration_gap"], 4) == round((2 / 3) - ((0.56 + 0.58 + 0.59) / 3), 4)


def _comparison_row(probability: float, correct: bool) -> dict[str, object]:
    return {
        "power_v3_recency_win_probability": probability,
        "power_v3_recency_correct": correct,
    }
