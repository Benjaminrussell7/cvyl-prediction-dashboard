# Modeling Lab

Purpose:
Experimental modeling work separated from the production prediction pipeline.

This directory is intended for:
- controlled modeling experiments
- rolling backtests
- alternative ranking systems
- feature engineering research
- calibration testing
- exploratory predictive ideas

Production code and UI should not depend directly on experiments in this folder until a variant is formally promoted.

## Experiments

### Hidden Tiering Experiment

Goal:
Evaluate whether hidden league segmentation and weak schedule graph connectivity were causing inflated rankings for isolated teams without using tournament data.

Motivation:
Some teams appeared overrated despite weaker competitive ecosystems, while battle-tested teams may have appeared underrated due to stronger schedules and ecosystem overlap.

Approach:
Tested:
- blowout margin caps
- schedule connectivity adjustments
- combined variants

Connectivity was intentionally designed differently from SOS by measuring exposure to the strongest portion of the schedule graph.

Result:
The connectivity adjustment improved perceived ranking realism for some teams, including RHAM and Westfield, but worsened predictive backtest performance.

Baseline Power v3 remained superior on:
- Brier score
- log loss
- margin MAE

Decision:
Do not promote any hidden tiering variants into production at this time.

Key takeaway:
The hidden connectivity phenomenon appears real, but the current implementation over-corrected and reduced predictive calibration.
