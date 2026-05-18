# Hidden Tiering Power v3 Modeling Lab

Scope: lab-only rolling backtest using existing canonical games data. Tournament data was not added.

Connectivity adjustment: for each pregame window, teams are scored by exposure to the prior top 20% of the Power v3 schedule graph. The score combines direct games against top-tier teams with a smaller neighborhood component from opponents' own top-tier exposure, then shrinks rating magnitude for teams with little exposure. This is intentionally different from SOS because it uses top-tier graph exposure rather than average opponent strength.

## Baseline

- Baseline games: 144; winner_accuracy 0.708; brier_score 0.207; log_loss 0.614; margin_mae 4.901; total_goals_mae 3.628.

## Variant Summary

| variant | winner_accuracy | brier_score | log_loss | margin_mae | total_goals_mae | games | delta_winner_accuracy | delta_brier_score | delta_log_loss | delta_margin_mae | promotion_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_power_v3 | 0.7083 | 0.2067 | 0.6143 | 4.9013 | 3.6277 | 144 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| cap_6 | 0.7083 | 0.2123 | 0.6267 | 4.9743 | 3.6277 | 144 | 0.0000 | 0.0056 | 0.0123 | 0.0729 | False |
| cap_8 | 0.7083 | 0.2067 | 0.6143 | 4.9013 | 3.6277 | 144 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| connectivity | 0.7014 | 0.2113 | 0.6243 | 4.9449 | 3.6277 | 144 | -0.0069 | 0.0046 | 0.0100 | 0.0435 | False |
| cap_6_connectivity | 0.7083 | 0.2164 | 0.6353 | 5.0168 | 3.6277 | 144 | 0.0000 | 0.0097 | 0.0210 | 0.1154 | False |
| cap_8_connectivity | 0.7014 | 0.2113 | 0.6243 | 4.9449 | 3.6277 | 144 | -0.0069 | 0.0046 | 0.0100 | 0.0435 | False |

## Interpretation

- No experimental variant improved both Brier score and log loss versus baseline.
- Worsened at least one probability metric: cap_6, connectivity, cap_6_connectivity, cap_8_connectivity.
- Promotion recommendation: do not promote any variant under the stated acceptance criteria.

## Ranking Oddities

- RHAM: baseline_power_v3 rank 3, rating 3.01, top-tier exposure 0.17; cap_6 rank 3, rating 2.36, top-tier exposure 0.17; cap_6_connectivity rank 4, rating 1.93, top-tier exposure 0.17; cap_8 rank 3, rating 3.01, top-tier exposure 0.17; cap_8_connectivity rank 4, rating 2.46, top-tier exposure 0.17; connectivity rank 4, rating 2.46, top-tier exposure 0.17.
- West Hartford Gold: baseline_power_v3 rank 1, rating 3.29, top-tier exposure 0.42; cap_6 rank 1, rating 2.55, top-tier exposure 0.42; cap_6_connectivity rank 3, rating 2.22, top-tier exposure 0.42; cap_8 rank 1, rating 3.29, top-tier exposure 0.42; cap_8_connectivity rank 2, rating 2.87, top-tier exposure 0.42; connectivity rank 2, rating 2.87, top-tier exposure 0.42.

## Rolling Windows

- Rolling window size: 25 games.
- Rolling windows evaluated: 6.
- Full window metrics are in `hidden_tiering_rolling_windows.csv`.

## Files

- `hidden_tiering_predictions.csv`
- `hidden_tiering_summary.csv`
- `hidden_tiering_rolling_windows.csv`
- `hidden_tiering_final_rankings.csv`
- `hidden_tiering_interpretation.md`
