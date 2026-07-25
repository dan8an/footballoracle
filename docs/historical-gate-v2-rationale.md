# Historical production gate v2 proposal

Gate v2 is separate from `historical_gate_v1.json`; v1 and all runs made under it remain immutable and reproducible. V2 retains the same leakage-safe 2022–2024 repeated chronological folds and treats 2025–2026 material as exposed retrospective evidence, not a pristine confirmation holdout.

## What the metrics mean

Signed mean error is the average prediction minus observation. Positive and negative match errors cancel, so it measures directional aggregate calibration rather than typical per-match accuracy. MAE averages error magnitudes and does not permit cancellation. Poisson deviance measures the likelihood quality of the full goal-count mean and penalizes badly assigned scoring rates. Multiclass Brier score measures squared error across home/draw/away probabilities, while multiclass log loss particularly penalizes assigning low probability to the observed outcome.

A model can improve MAE and Poisson deviance while signed bias moves slightly away from zero: it may reduce many large per-match errors while the remaining residuals are mildly asymmetric, or the validation years may have different scoring environments. Result probabilities can likewise improve even when a small directional goal residual remains.

## Why v1 bias vetoes deserve audit

V1 required candidate absolute home bias at or below 0.10 even though the active comparator is around 0.134. That is an absolute aspirational target, not an incumbent-relative non-inferiority comparison. It can veto a candidate that improves every per-match goal error and proper score because of a small signed-mean difference. Bias should remain visible, but promotion should ask whether the candidate materially degrades scoring predictions relative to production.

## V2 bias policy

V2 requires Poisson-deviance improvement and non-inferior home, away, total, and goal-difference MAE. It then compares absolute candidate bias with absolute production bias on identical matches. Side bias may regress by at most 0.05 goals and total bias by at most 0.075. Paired date-block bootstrap upper bounds must exclude increases of 0.08 per side and 0.10 total. More than one fold cannot regress by over 0.10. Broad absolute limits of 0.35 per side and 0.50 total catch obviously broken models.

These values reflect roughly one-goal side MAE, match-to-match count variance, observed temporal fold variability, and existing gate materiality conventions. They are not near-zero calibration demands and were recorded before final v4.4 status evaluation.

## What remains unchanged

Integrity is mandatory. Brier, log loss, fold stability, paired bootstrap harm limits, class-specific Brier, ECE, support-aware calibration, confidence behavior, Poisson deviance, and all goal MAEs remain central. Accuracy is descriptive only. A historical pass is not presented as an untouched 2026 confirmation result, and prospective World Cup evidence remains a separate scorecard.
