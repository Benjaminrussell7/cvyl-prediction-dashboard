# Probability And Spread Calibration Lab

Scope: lab-only post-processing experiment over saved opponent_adjusted_performance rolling predictions. Baseline Power v3, the opponent-adjusted rating engine, and the current conversion logic were preserved; no tournament data, UI, or production prediction code was used.

Approach: variants adjust only the translation layer from rating differential to probability and implied spread. Logistic-scale variants sharpen probabilities by reducing the logistic denominator. Margin-scale variants multiply the implied spread while leaving winner side unchanged. Combined variants do both. The prequential bucket variant recalibrates probabilities from earlier games only, with shrinkage toward the current probability.

## Best Variant

- Best by Brier/log-loss/ECE sort: combined_2_0_1_5 (Brier 0.1854, log_loss 0.5631, ECE 0.0923, margin_mae 4.7234, accuracy 0.7083).
- Current conversion reference: Brier 0.2052, log_loss 0.6111, ECE 0.1397, margin_mae 4.8687, accuracy 0.7083.

## Summary

| variant | winner_accuracy | brier_score | log_loss | margin_mae | mean_favorite_probability | favorite_accuracy | mean_implied_spread | mean_actual_favorite_margin | mean_spread_error | expected_calibration_error | delta_brier_score | delta_log_loss | delta_margin_mae | delta_ece | promotion_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_2_0_1_5 | 0.7083 | 0.1854 | 0.5631 | 4.7234 | 0.6266 | 0.7083 | 1.7041 | 3.0486 | -1.3445 | 0.0923 | -0.0198 | -0.0480 | -0.1453 | -0.0474 | True |
| logistic_scale_2 | 0.7083 | 0.1854 | 0.5631 | 4.8687 | 0.6266 | 0.7083 | 1.1361 | 3.0486 | -1.9125 | 0.0923 | -0.0198 | -0.0480 | 0.0000 | -0.0474 | True |
| combined_2_5_1_5 | 0.7083 | 0.1914 | 0.5789 | 4.7234 | 0.6050 | 0.7083 | 1.7041 | 3.0486 | -1.3445 | 0.1033 | -0.0137 | -0.0323 | -0.1453 | -0.0363 | True |
| combined_2_5_1_25 | 0.7083 | 0.1914 | 0.5789 | 4.7851 | 0.6050 | 0.7083 | 1.4201 | 3.0486 | -1.6285 | 0.1033 | -0.0137 | -0.0323 | -0.0836 | -0.0363 | True |
| logistic_scale_2_5 | 0.7083 | 0.1914 | 0.5789 | 4.8687 | 0.6050 | 0.7083 | 1.1361 | 3.0486 | -1.9125 | 0.1033 | -0.0137 | -0.0323 | 0.0000 | -0.0363 | True |
| logistic_scale_3 | 0.7083 | 0.1968 | 0.5918 | 4.8687 | 0.5894 | 0.7083 | 1.1361 | 3.0486 | -1.9125 | 0.1189 | -0.0084 | -0.0193 | 0.0000 | -0.0208 | True |
| prequential_bucket_calibration | 0.7083 | 0.1983 | 0.5962 | 4.8687 | 0.5906 | 0.7083 | 1.1361 | 3.0486 | -1.9125 | 0.1177 | -0.0069 | -0.0149 | 0.0000 | -0.0219 | True |
| margin_scale_1_5 | 0.7083 | 0.2052 | 0.6111 | 4.7234 | 0.5687 | 0.7083 | 1.7041 | 3.0486 | -1.3445 | 0.1397 | 0.0000 | 0.0000 | -0.1453 | 0.0000 | False |
| margin_scale_1_25 | 0.7083 | 0.2052 | 0.6111 | 4.7851 | 0.5687 | 0.7083 | 1.4201 | 3.0486 | -1.6285 | 0.1397 | 0.0000 | 0.0000 | -0.0836 | 0.0000 | False |
| current_conversion | 0.7083 | 0.2052 | 0.6111 | 4.8687 | 0.5687 | 0.7083 | 1.1361 | 3.0486 | -1.9125 | 0.1397 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |

## Interpretation

- Probabilities became less compressed: mean favorite probability changed from 0.569 to 0.627, while favorite accuracy was 0.708.
- Spreads changed from 1.14 implied goals to 1.70; actual favorite margin was 3.05. Mean spread error moved from -1.91 to -1.34.
- Large favorites under the best variant: 22 games, accuracy 0.955, mean spread error -1.91; current had 2 large-favorite games.
- Current conversion: worst bucket was 60-65% with predicted 0.624, actual 1.000, error +0.376 over 16 games.
- Best variant (combined_2_0_1_5): worst bucket was 70-75% with predicted 0.727, actual 1.000, error +0.273 over 11 games.
- Recommendation: promote combined_2_0_1_5 as the leading Power v5 calibration candidate.

## Output Files

- `calibrated_matchup_predictions.csv`
- `calibrated_matchup_summary.csv`
- `calibrated_matchup_probability_buckets.csv`
- `calibrated_matchup_interpretation.md`
