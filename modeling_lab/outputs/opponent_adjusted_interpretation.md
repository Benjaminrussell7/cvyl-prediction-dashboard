# Opponent-Adjusted Performance Power v3 Lab

Scope: lab-only rolling backtest using existing canonical games data. No Streamlit or production prediction code was modified.

Variant definition: each pregame window first builds the exact baseline Power v3 ratings. The opponent-adjusted variant evaluates every prior team-game by capped actual margin, expected margin for an average team against that opponent based on the opponent's prior Power v3 rating, and performance above or below that expectation. Team ratings are the recency-weighted average of those performance values with the same shrinkage behavior used by the baseline.

## Overall Metrics

| variant | winner_accuracy | brier_score | log_loss | margin_mae | total_goals_mae | games | delta_winner_accuracy | delta_brier_score | delta_log_loss | delta_margin_mae | promotion_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_power_v3 | 0.7083 | 0.2067 | 0.6143 | 4.9013 | 3.6277 | 144 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| opponent_adjusted_performance | 0.7083 | 0.2052 | 0.6111 | 4.8687 | 3.6277 | 144 | 0.0000 | -0.0015 | -0.0032 | -0.0326 | True |

## Interpretation

- Baseline: winner_accuracy 0.708, Brier 0.207, log_loss 0.614, margin_mae 4.901.
- Opponent-adjusted: winner_accuracy 0.708 (+0.000), Brier 0.205 (-0.002), log_loss 0.611 (-0.003), margin_mae 4.869 (-0.033).
- Promotion recommendation: promote the opponent-adjusted variant under the stated criteria.

## Ranking Oddities

- RHAM: baseline rank 3, rating 3.01; opponent-adjusted rank 4, rating 2.89; rank change +1.
- Westfield: baseline rank 4, rating 2.87; opponent-adjusted rank 3, rating 3.07; rank change -1.
- Avon: baseline rank 10, rating 1.44; opponent-adjusted rank 9, rating 1.58; rank change -1.
- Suffield: baseline rank 5, rating 2.07; opponent-adjusted rank 5, rating 2.45; rank change +0.
- Somers: baseline rank 7, rating 1.97; opponent-adjusted rank 6, rating 2.02; rank change -1.
- West Hartford Green: baseline rank 8, rating 1.86; opponent-adjusted rank 8, rating 1.75; rank change +0.

## Rolling Windows

- Rolling window size: 25 games.
- Rolling windows evaluated: 6.
- Full window metrics are in `opponent_adjusted_rolling_windows.csv`.

## Files

- `opponent_adjusted_predictions.csv`
- `opponent_adjusted_performance_games.csv`
- `opponent_adjusted_summary.csv`
- `opponent_adjusted_rolling_windows.csv`
- `opponent_adjusted_final_rankings.csv`
- `opponent_adjusted_interpretation.md`

Promotion candidates: opponent_adjusted_performance.
