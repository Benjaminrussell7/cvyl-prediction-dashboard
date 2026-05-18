# Recency-Weighted Opponent-Adjusted Power v4 Lab

Scope: lab-only rolling backtest using existing canonical games data. Tournament data was not used, and no Streamlit or production prediction code was modified.

Approach: baseline Power v3 and the existing opponent-adjusted model are preserved exactly. The recency variants reuse the same opponent-adjusted per-game performance values, then apply a team-specific exponential decay by team-game age before averaging. Mild uses a 6-game half-life, moderate uses 3 games, and aggressive uses 1.5 games. The newest game for each team has full decay weight; older games decay by half every half-life interval.

## Overall Metrics

| variant | winner_accuracy | brier_score | log_loss | margin_mae | total_goals_mae | games | delta_winner_accuracy | delta_brier_score | delta_log_loss | delta_margin_mae | delta_accuracy_vs_opponent_adjusted | delta_brier_vs_opponent_adjusted | delta_log_loss_vs_opponent_adjusted | delta_margin_mae_vs_opponent_adjusted | promotion_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_power_v3 | 0.7083 | 0.2067 | 0.6143 | 4.9013 | 3.6277 | 144 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0015 | 0.0032 | 0.0326 | False |
| opponent_adjusted_performance | 0.7083 | 0.2052 | 0.6111 | 4.8687 | 3.6277 | 144 | 0.0000 | -0.0015 | -0.0032 | -0.0326 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| recency_mild | 0.6944 | 0.2062 | 0.6132 | 4.8956 | 3.6277 | 144 | -0.0139 | -0.0005 | -0.0011 | -0.0057 | -0.0139 | 0.0010 | 0.0021 | 0.0269 | False |
| recency_moderate | 0.6875 | 0.2074 | 0.6156 | 4.9245 | 3.6277 | 144 | -0.0208 | 0.0007 | 0.0013 | 0.0232 | -0.0208 | 0.0023 | 0.0045 | 0.0558 | False |
| recency_aggressive | 0.6667 | 0.2101 | 0.6210 | 4.9820 | 3.6277 | 144 | -0.0417 | 0.0034 | 0.0067 | 0.0807 | -0.0417 | 0.0050 | 0.0099 | 0.1133 | False |

## Interpretation

- Best overall by Brier/log-loss/margin sort: opponent_adjusted_performance (Brier 0.2052, log_loss 0.6111, margin_mae 4.8687, accuracy 0.7083).
- Existing opponent-adjusted reference: Brier 0.2052, log_loss 0.6111, margin_mae 4.8687, accuracy 0.7083.
- Best recency variant: recency_mild (Brier 0.2062, log_loss 0.6132, margin_mae 4.8956, accuracy 0.6944).
- No recency-weighted variant improved Brier, log loss, or margin MAE versus the existing opponent-adjusted model.
- Instability/overreaction: aggressive weighting shows overreaction risk, with 13 teams moving at least five ranking spots.
- Recommendation: do not make a recency-weighted variant the leading Power v4 candidate.

## Ranking Movement

- RHAM: baseline_power_v3 rank 3, rating 3.01; opponent_adjusted_performance rank 4, rating 2.89; recency_mild rank 4, rating 2.79; recency_moderate rank 3, rating 2.72; recency_aggressive rank 4, rating 2.69.
- Westfield: baseline_power_v3 rank 4, rating 2.87; opponent_adjusted_performance rank 3, rating 3.07; recency_mild rank 3, rating 2.88; recency_moderate rank 4, rating 2.70; recency_aggressive rank 5, rating 2.40.
- Avon: baseline_power_v3 rank 10, rating 1.44; opponent_adjusted_performance rank 9, rating 1.58; recency_mild rank 10, rating 1.59; recency_moderate rank 10, rating 1.60; recency_aggressive rank 8, rating 1.64.
- Suffield: baseline_power_v3 rank 5, rating 2.07; opponent_adjusted_performance rank 5, rating 2.45; recency_mild rank 6, rating 2.13; recency_moderate rank 8, rating 1.85; recency_aggressive rank 10, rating 1.44.
- Somers: baseline_power_v3 rank 7, rating 1.97; opponent_adjusted_performance rank 6, rating 2.02; recency_mild rank 5, rating 2.30; recency_moderate rank 5, rating 2.57; recency_aggressive rank 2, rating 3.07.
- West Hartford Green: baseline_power_v3 rank 8, rating 1.86; opponent_adjusted_performance rank 8, rating 1.75; recency_mild rank 9, rating 1.67; recency_moderate rank 9, rating 1.62; recency_aggressive rank 9, rating 1.55.

## Material Movers

- Largest aggressive-recency rank changes versus opponent-adjusted: Burlington 12U +15; Westfield 12U Black -11; Minnechaug 12U White +9; Berlin 12U A -7; Bristol 12U +6; Belchertown 12U -6; Longmeadow 12U Black -6; Granby 12U -5; Suffield 12U White +5; South Windsor 12U -5.

## Rolling Windows

- Rolling window size: 25 games.
- Rolling windows evaluated: 6.
- Full window metrics are in `recency_opponent_adjusted_rolling_windows.csv`.

## Files

- `recency_opponent_adjusted_predictions.csv`
- `recency_opponent_adjusted_summary.csv`
- `recency_opponent_adjusted_rolling_windows.csv`
- `recency_opponent_adjusted_final_rankings.csv`
- `recency_opponent_adjusted_interpretation.md`
