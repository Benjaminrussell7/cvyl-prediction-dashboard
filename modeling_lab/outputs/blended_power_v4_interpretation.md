# Blended Power v4 Modeling Lab

Scope: lab-only rolling backtest using existing canonical games data. No Streamlit or production prediction code was modified.

Variant definition: baseline Power v3 and the opponent-adjusted performance model are preserved exactly. Blended variants are linear combinations of those two pregame team ratings at 90/10, 80/20, 70/30, and 60/40 weights.

## Overall Metrics

| variant | winner_accuracy | brier_score | log_loss | margin_mae | total_goals_mae | games | delta_winner_accuracy | delta_brier_score | delta_log_loss | delta_margin_mae | delta_brier_vs_opponent_adjusted | delta_log_loss_vs_opponent_adjusted | delta_margin_mae_vs_opponent_adjusted | promotion_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_power_v3 | 0.7083 | 0.2067 | 0.6143 | 4.9013 | 3.6277 | 144 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0015 | 0.0032 | 0.0326 | False |
| opponent_adjusted_performance | 0.7083 | 0.2052 | 0.6111 | 4.8687 | 3.6277 | 144 | 0.0000 | -0.0015 | -0.0032 | -0.0326 | 0.0000 | 0.0000 | 0.0000 | False |
| blend_90_10 | 0.7083 | 0.2065 | 0.6140 | 4.8981 | 3.6277 | 144 | 0.0000 | -0.0002 | -0.0003 | -0.0033 | 0.0014 | 0.0029 | 0.0294 | True |
| blend_80_20 | 0.7153 | 0.2064 | 0.6137 | 4.8948 | 3.6277 | 144 | 0.0069 | -0.0003 | -0.0007 | -0.0065 | 0.0012 | 0.0026 | 0.0261 | True |
| blend_70_30 | 0.7153 | 0.2062 | 0.6133 | 4.8915 | 3.6277 | 144 | 0.0069 | -0.0005 | -0.0010 | -0.0098 | 0.0011 | 0.0022 | 0.0228 | True |
| blend_60_40 | 0.7153 | 0.2060 | 0.6130 | 4.8883 | 3.6277 | 144 | 0.0069 | -0.0006 | -0.0013 | -0.0130 | 0.0009 | 0.0019 | 0.0196 | True |

## Interpretation

- Best overall by Brier/log-loss/margin sort: opponent_adjusted_performance (Brier 0.2052, log_loss 0.6111, margin_mae 4.8687, accuracy 0.7083).
- Best blended variant: blend_60_40 (Brier 0.2060, log_loss 0.6130, margin_mae 4.8883, accuracy 0.7153).
- Pure opponent-adjusted reference: Brier 0.2052, log_loss 0.6111, margin_mae 4.8687.
- No blended variant improved Brier, log loss, and margin MAE versus pure opponent-adjusted.
- Recommendation: do not make a blended variant the leading Power v4 candidate; blend_60_40 is the best blend versus baseline, but pure opponent-adjusted remains stronger on Brier, log loss, and margin MAE.

## Ranking Movement

- RHAM: baseline_power_v3 rank 3, rating 3.01; opponent_adjusted_performance rank 4, rating 2.89; blend_90_10 rank 3, rating 3.00; blend_80_20 rank 3, rating 2.99; blend_70_30 rank 3, rating 2.97; blend_60_40 rank 3, rating 2.96.
- Westfield: baseline_power_v3 rank 4, rating 2.87; opponent_adjusted_performance rank 3, rating 3.07; blend_90_10 rank 4, rating 2.89; blend_80_20 rank 4, rating 2.91; blend_70_30 rank 4, rating 2.93; blend_60_40 rank 4, rating 2.95.
- Avon: baseline_power_v3 rank 10, rating 1.44; opponent_adjusted_performance rank 9, rating 1.58; blend_90_10 rank 10, rating 1.46; blend_80_20 rank 10, rating 1.47; blend_70_30 rank 10, rating 1.49; blend_60_40 rank 10, rating 1.50.
- Suffield: baseline_power_v3 rank 5, rating 2.07; opponent_adjusted_performance rank 5, rating 2.45; blend_90_10 rank 5, rating 2.11; blend_80_20 rank 5, rating 2.15; blend_70_30 rank 5, rating 2.18; blend_60_40 rank 5, rating 2.22.
- Somers: baseline_power_v3 rank 7, rating 1.97; opponent_adjusted_performance rank 6, rating 2.02; blend_90_10 rank 7, rating 1.97; blend_80_20 rank 7, rating 1.98; blend_70_30 rank 6, rating 1.98; blend_60_40 rank 6, rating 1.99.
- West Hartford Green: baseline_power_v3 rank 8, rating 1.86; opponent_adjusted_performance rank 8, rating 1.75; blend_90_10 rank 8, rating 1.85; blend_80_20 rank 8, rating 1.84; blend_70_30 rank 8, rating 1.83; blend_60_40 rank 8, rating 1.82.

## Rolling Windows

- Rolling window size: 25 games.
- Rolling windows evaluated: 6.
- Full window metrics are in `blended_power_v4_rolling_windows.csv`.

## Files

- `blended_power_v4_predictions.csv`
- `blended_power_v4_summary.csv`
- `blended_power_v4_rolling_windows.csv`
- `blended_power_v4_final_rankings.csv`
- `blended_power_v4_interpretation.md`
