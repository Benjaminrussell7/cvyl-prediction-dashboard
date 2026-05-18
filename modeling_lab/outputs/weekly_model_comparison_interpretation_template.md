# Weekly Shadow Model Comparison Template

Scope: governance-only comparison between baseline Power v3 and the opponent-adjusted Power v4 candidate. This is not a production promotion.

## Snapshot Summary

- Matchups compared: 238
- Predicted winner disagreements: 5
- Mean favorite probability delta, Power v4 minus baseline: +0.002
- Mean implied spread delta, Power v4 minus baseline: +0.04

## Where Models Differ Most

- 2026-05-18 Ellington 12U vs Granby 12U: baseline Granby 12U 50.1%, Power v4 Ellington 12U 51.8%, spread delta +0.29
- 2026-05-02 Simsbury 12U A vs Granby 12U: baseline Granby 12U 50.8%, Power v4 Simsbury 12U A 51.3%, spread delta +0.08
- 2026-04-14 South Hadley 12U vs East Longmeadow 12U A: baseline South Hadley 12U 50.8%, Power v4 East Longmeadow 12U A 50.1%, spread delta -0.10
- 2026-04-13 Minnechaug 12U vs Westfield 12U Red: baseline Westfield 12U Red 50.8%, Power v4 Minnechaug 12U 50.1%, spread delta -0.10
- 2026-04-30 West Springfield 12U vs East Longmeadow 12U A: baseline West Springfield 12U 50.6%, Power v4 East Longmeadow 12U A 50.2%, spread delta -0.07
- 2026-04-17 Agawam 12U vs Minnechaug 12U: baseline Minnechaug 12U 53.9%, Power v4 Minnechaug 12U 57.6%, spread delta +0.60
- 2026-05-09 Granby 12U vs Tolland 12U: baseline Tolland 12U 50.5%, Power v4 Tolland 12U 53.5%, spread delta +0.48
- 2026-05-09 Tolland 12U vs Granby 12U: baseline Tolland 12U 50.5%, Power v4 Tolland 12U 53.5%, spread delta +0.48
- 2026-04-29 Granby 12U vs South Windsor 12U: baseline South Windsor 12U 53.7%, Power v4 South Windsor 12U 56.0%, spread delta +0.36
- 2026-04-25 West Hartford 12U Gold vs Newington 12U: baseline West Hartford 12U Gold 57.7%, Power v4 West Hartford 12U Gold 60.4%, spread delta +0.45

## Directional Read

- Does Power v4 appear directionally stronger this week? [Fill in after reviewing correctness, ranking deltas, and known team context.]
- Are disagreements concentrated around isolated schedules, volatile teams, or specific divisions? [Fill in.]

## Suspicious Predictions

- Review rows with `predicted_winner_disagreement`, `upset_disagreement_flag`, `large_probability_disagreement`, or `large_spread_disagreement` in `weekly_biggest_disagreements.csv`.
- Note any predictions where Power v4 moves strongly against recent observable results.

## Calibration Observations

- Check whether Power v4 probabilities are more or less compressed than baseline in the current week.
- Check whether spread deltas are directionally plausible or too aggressive.

## Largest Ranking Moves

- Ellington 12U: baseline rank 43, Power v4 rank 39, delta -4
- Northampton 12U: baseline rank 40, Power v4 rank 44, delta +4
- Glastonbury 12U White: baseline rank 30, Power v4 rank 27, delta -3
- Simsbury 12U A: baseline rank 48, Power v4 rank 45, delta -3
- Wethersfield 12U A: baseline rank 19, Power v4 rank 16, delta -3
- Berkshire: baseline rank 17, Power v4 rank 19, delta +2
- Farmington 12U Red: baseline rank 16, Power v4 rank 18, delta +2
- Simsbury 12U Blue: baseline rank 44, Power v4 rank 42, delta -2
- Southington 12U Blue: baseline rank 13, Power v4 rank 11, delta -2
- Agawam 12U: baseline rank 36, Power v4 rank 37, delta +1

## Outputs

- `weekly_model_comparison.csv`
- `weekly_biggest_disagreements.csv`
- `weekly_ranking_changes.csv`
