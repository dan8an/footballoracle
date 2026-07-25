# Gate-v2 total-goal absolute-bias bootstrap audit v1

This is diagnostic research only. Historical gates v1 and v2, their plans, thresholds, readiness artifacts, and ledgers remain unchanged. V4.4 remains failed under gate v2 and is not promoted.

## Why the interval is wide

The statistic `|candidate mean residual| - |production mean residual|` folds positive and negative bias onto one scale. Absolute value has a cusp at zero: its slope changes from -1 to +1. Production bias is positive while v4.4 bias is negative, so a resample that moves either estimate through zero changes the statistic's local direction even if the underlying signed candidate-minus-production shift changes smoothly.

Across 2,000 deterministic paired date-block samples, production crossed zero 11.7% of the time, v4.4 crossed zero 3.3%, and the estimates had opposite signs 85.0% of the time. The absolute-bias-difference distribution had mean 0.0448, median 0.0549, standard deviation 0.1430, and 95% interval [-0.2450, 0.2642]. It was above zero in 62.1%, above 0.05 in 51.3%, above 0.10 in 39.25%, and above 0.20 in 17.9% of samples.

The result mainly expresses uncertainty over which aggregate mean is closer to zero. It is not a direct estimate of typical match-level predictive harm.

## Alternative diagnostics

- V4.4 signed total bias is -0.1538 with clustered interval [-0.3228, 0.0115]. This measures aggregate directional calibration and remains cancellation-sensitive.
- Candidate-minus-production signed bias is -0.2512 with interval approximately [-0.2955, -0.2065]. It detects a stable downward level shift but does not say which level is better calibrated.
- Mean absolute date-block bias improves from 1.0343 to 1.0271. This avoids cross-date cancellation but gives small dates equal influence.
- RMS date-block bias improves from 1.3644 to 1.3470. It emphasizes large date-level failures and is outlier-sensitive.
- Total-goal MAE improves by 0.0301 goals per match and directly measures typical error.
- Poisson deviance improves by 0.0501 and measures likelihood quality.
- Calibration regression changes from an unstable production intercept/slope of -5.494/2.955 to 0.389/0.906 for v4.4. The production slope is sensitive to its narrow predicted-total range, so this is diagnostic rather than a standalone verdict.
- Predicted-total reliability and 0–1/2/3/4/5+ goal-rate tables expose local and distributional failures that an aggregate mean can hide.

## Stability

V4.4 total bias is -0.009 in 2022, -0.342 in 2023, and -0.182 in 2024. The negative aggregate is therefore not uniform: 2023 is the strongest chronological contributor. Competition segments vary in sign and magnitude. Continental/other competitive matches (-0.363) and friendlies (-0.336) drive underprediction, while qualification (+0.116) and Nations League (+0.099) overpredict. World Cup rows are nearly unbiased (-0.001).

Underprediction occurs in both neutral (-0.183) and non-neutral (-0.133) matches. It is strongest in candidate predicted totals of 2.0–2.5 (-0.272); below 2.0 is slightly positive (+0.044), 2.5–3.0 is -0.148, and 3+ is +0.108. This is time- and segment-dependent rather than a single broad constant offset.

Several largest date-level harms are single-match blocks. Upper-tail resamples are most enriched for 2024-03-26, 2022-06-02, and 2023-03-28, showing that unequal date-block sizes and unusual scoring dates materially influence the percentile tail.

## Recommendation

Retain gate v2's recorded failure unchanged. For a future independently designed gate v3 study, demote the bootstrap of absolute aggregate bias to monitoring or supplement it rather than use it as a sole veto. Preserve signed and absolute bias reporting, catastrophic limits, fold checks, total-goal MAE, Poisson deviance, and goal-distribution calibration. Study a pre-registered combination of block-level absolute calibration, likelihood, typical error, temporal stability, and goal-band reliability across multiple frozen good and bad models before hashing a new gate. Do not derive thresholds from v4.4.

Full samples, quantiles, block influence, segment tables, reliability data, and goal-band rates are stored in `data/evaluation/gate_v2_total_goal_bias_audit_v1.json`.

## Shadow scheduler handoff

The external scheduler must generate both versions in one invocation and then run:

```text
python scripts/persist_scheduled_shadow_batch.py /path/to/same-run-batch.json --production-version elo-context-v4.2.1 --shadow-version elo-context-v4.4-opponent-adjusted-xg-experimental
python scripts/update_prospective_scorecard.py --results /path/to/completed-results.json
```

Repository-local validation without ledger mutation:

```text
python scripts/persist_scheduled_shadow_batch.py data/evaluation/fixtures/wc26_shadow_batch_dry_run.json --dry-run
```

The deployment scheduler is external to this repository, so its job definition cannot be edited here. The persistence command rejects timestamp mismatches, missing or wrong model pairs, post-kickoff snapshots, conflicting retries, and hash mutations.
