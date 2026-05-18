# Matchup Calibration Analysis

Scope: lab-only analytical evaluation using saved rolling predictions for baseline Power v3 and opponent_adjusted_performance. No tournament data, UI changes, or production prediction changes were used.

## Probability Calibration

- Opponent-adjusted overall favorite accuracy was 0.708 with mean favorite probability 0.569; mean absolute bucket calibration error was 0.217.
- Baseline comparison: favorite accuracy 0.708, mean favorite probability 0.566.
- Calibration flags: underconfident in 55-60%, 60-65%, 65-70%, 70-75%, 75-80%, 80-85%.

## Spread Realism

- Average implied favorite spread was 1.14; average actual favorite margin was 3.05; mean spread error was -1.91 goals.
- Large favorites, implied spread 4+ goals, went 5-0; mean spread error was -5.67 goals.
- Favorite inflation is not broad across spread buckets.

## Upsets And Difficult Matchups

- Biggest predicted upsets: 2026-05-08 Belchertown 12U over Minnechaug 12U White (69% favorite, actual favorite margin -4); 2026-04-25 Avon 12U over Southington 12U Blue (60% favorite, actual favorite margin -5); 2026-04-28 Agawam 12U over East Longmeadow 12U A (60% favorite, actual favorite margin -4); 2026-05-07 Longmeadow 12U Black over Agawam 12U (59% favorite, actual favorite margin -10); 2026-05-14 Bristol 12U over Meriden 12U (59% favorite, actual favorite margin +0).
- Biggest actual upsets: 2026-04-10 Westfield 12U Red over East Longmeadow 12U A (50% favorite, actual favorite margin -14); 2026-04-20 Suffield 12U over Somers 12U White (55% favorite, actual favorite margin -13); 2026-05-07 Longmeadow 12U Black over Agawam 12U (59% favorite, actual favorite margin -10); 2026-05-09 Wolcott 12U over Burlington 12U (52% favorite, actual favorite margin -10); 2026-05-09 Granby 12U over Tolland 12U (52% favorite, actual favorite margin -10).
- Hardest teams to model: Somers 12U White (18.96); Longmeadow 12U Black (17.28); Minnechaug 12U (16.84); Farmington 12U Black (15.96); Ellington 12U (15.81); Granby 12U (15.21); South Windsor 12U (15.06); Agawam 12U (14.79).
- Most volatile weekly ratings: Ellington 12U (1.64); West Hartford 12U Gold (1.29); Agawam 12U (1.18); Southington 12U White (1.03); Somers 12U White (1.02); Colchester 12U (0.99); Glastonbury 12U Blue (0.97); Southington 12U Blue (0.94).
- Largest average prediction errors: Suffield Juniors 1 Blue (12.25); Tolland 12U (10.06); Somers 12U White (8.49); Minnechaug 12U (8.19); West Hartford 12U Gold (7.67); Ellington 12U (7.53); Longmeadow 12U Black (7.24); South Hadley 12U (7.05).

## Recommendation

- Production readiness: use the model cautiously; calibration needs a probability scaling layer before high-confidence matchup probabilities are surfaced strongly.
- Future modeling directions: calibrate probability scale, consider a spread-to-probability calibration layer, and add team-level uncertainty so volatile teams get less extreme probabilities.

## Output Files

- `matchup_calibration_summary.csv`
- `matchup_calibration_team_volatility.csv`
- `matchup_calibration_interpretation.md`
