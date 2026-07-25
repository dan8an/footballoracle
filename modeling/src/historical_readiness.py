"""Locked, leakage-safe historical production-gate calculations."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from modeling.src.evaluation.reliability import adaptive_reliability, calibration_summary, fixed_width_reliability

CLASSES = ("home", "draw", "away")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_probabilities(probabilities: list[tuple[float, float, float]]) -> bool:
    return all(all(math.isfinite(v) and v >= 0 for v in p) and abs(sum(p) - 1) <= 1e-9 for p in probabilities)


def paired_metrics(rows, predictions) -> dict[str, Any]:
    n = len(rows)
    probs = [p[1] for p in predictions]
    xgs = [p[0] for p in predictions]
    outcomes = [r.backtest.outcome for r in rows]
    actual_goals = [(r.backtest.home_score, r.backtest.away_score) for r in rows]
    class_brier = {CLASSES[c]: sum((p[c] - (y == c)) ** 2 for p, y in zip(probs, outcomes)) / n for c in range(3)}
    poisson_terms = []
    for actual, expected in ((a, e) for scores, xg in zip(actual_goals, xgs) for a, e in zip(scores, xg)):
        poisson_terms.append(2 * (actual * math.log(actual / expected) - (actual - expected)) if actual else 2 * expected)
    fixed = fixed_width_reliability(probs, outcomes)
    calibration = calibration_summary(fixed, n, minimum_support=30)
    bucket_defs = (("below_40", 0, .4), ("40_50", .4, .5), ("50_60", .5, .6), ("60_70", .6, .7), ("70_80", .7, .8), ("above_80", .8, 1.0000001))
    buckets = {}
    for label, lower, upper in bucket_defs:
        selected = [(max(p), int(max(range(3), key=lambda c: p[c]) == y)) for p, y in zip(probs, outcomes) if lower <= max(p) < upper]
        buckets[label] = {"count": len(selected), "mean_predicted_favorite_probability": sum(x for x, _ in selected) / len(selected) if selected else None, "observed_favorite_win_rate": sum(y for _, y in selected) / len(selected) if selected else None, "absolute_calibration_gap": abs(sum(x for x, _ in selected) / len(selected) - sum(y for _, y in selected) / len(selected)) if selected else None, "error_rate": 1 - sum(y for _, y in selected) / len(selected) if selected else None}
    mean_actual_probability = sum(p[y] for p, y in zip(probs, outcomes)) / n
    return {
        "matches": n,
        "multiclass_brier": sum(sum((p[c] - (y == c)) ** 2 for c in range(3)) for p, y in zip(probs, outcomes)) / n,
        "log_loss": sum(-math.log(max(1e-15, p[y])) for p, y in zip(probs, outcomes)) / n,
        "poisson_deviance_per_team": sum(poisson_terms) / (2 * n),
        "accuracy": sum(max(range(3), key=lambda c: p[c]) == y for p, y in zip(probs, outcomes)) / n,
        "class_brier": class_brier,
        "mean_probability_actual_class": mean_actual_probability,
        "predicted_class_frequencies": {CLASSES[c]: sum(p[c] for p in probs) / n for c in range(3)},
        "actual_class_frequencies": {CLASSES[c]: sum(y == c for y in outcomes) / n for c in range(3)},
        "home_goal_mae": sum(abs(a[0] - x[0]) for a, x in zip(actual_goals, xgs)) / n,
        "away_goal_mae": sum(abs(a[1] - x[1]) for a, x in zip(actual_goals, xgs)) / n,
        "total_goal_mae": sum(abs(sum(a) - sum(x)) for a, x in zip(actual_goals, xgs)) / n,
        "goal_difference_mae": sum(abs((a[0] - a[1]) - (x[0] - x[1])) for a, x in zip(actual_goals, xgs)) / n,
        "home_goal_bias": sum(x[0] - a[0] for a, x in zip(actual_goals, xgs)) / n,
        "away_goal_bias": sum(x[1] - a[1] for a, x in zip(actual_goals, xgs)) / n,
        "total_goal_bias": sum(sum(x) - sum(a) for a, x in zip(actual_goals, xgs)) / n,
        "fixed_width_reliability": fixed,
        "adaptive_reliability": adaptive_reliability(probs, outcomes, min(30, max(2, n // 3))),
        "calibration": calibration,
        "confidence_buckets": buckets,
    }


def date_block_bootstrap(rows, current_predictions, candidate_predictions, samples: int, seed: int) -> dict[str, Any]:
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        blocks[row.played_on.isoformat()].append(index)
    dates = sorted(blocks)
    rng = random.Random(seed)
    keys = ("multiclass_brier", "log_loss", "poisson_deviance_per_team")
    distributions = {key: [] for key in keys}
    for _ in range(samples):
        indices = [i for _date in range(len(dates)) for i in blocks[rng.choice(dates)]]
        sampled_rows = [rows[i] for i in indices]
        current = paired_metrics(sampled_rows, [current_predictions[i] for i in indices])
        candidate = paired_metrics(sampled_rows, [candidate_predictions[i] for i in indices])
        for key in keys:
            distributions[key].append(candidate[key] - current[key])
    def percentile(values, q):
        ordered = sorted(values); position = (len(ordered) - 1) * q; lower = int(position); fraction = position - lower
        return ordered[lower] if lower + 1 == len(ordered) else ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction
    return {key: {"lower": percentile(values, .025), "upper": percentile(values, .975), "includes_zero": percentile(values, .025) <= 0 <= percentile(values, .975)} for key, values in distributions.items()}


def condition(name: str, measured: Any, threshold: Any, passed: bool, explanation: str, required: bool = True) -> dict[str, Any]:
    return {"name": name, "measured_value": measured, "threshold": threshold, "passed": bool(passed), "required": required, "explanation": explanation}
