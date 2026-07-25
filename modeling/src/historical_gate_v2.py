"""Incumbent-relative expected-goal conditions for historical gate v2."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from modeling.src.historical_readiness import condition


BIAS_KEYS = ("home_goal_bias", "away_goal_bias", "total_goal_bias")


def absolute_bias_difference(candidate: dict, production: dict, key: str) -> float:
    """Candidate minus production absolute signed bias; negative is improvement."""
    return abs(candidate[key]) - abs(production[key])


def bias_noninferiority_conditions(production: dict, candidate: dict, thresholds: dict) -> list[dict[str, Any]]:
    mapping = {
        "home_goal_bias": ("absolute_home_bias_regression_max", "catastrophic_home_bias_absolute_max"),
        "away_goal_bias": ("absolute_away_bias_regression_max", "catastrophic_away_bias_absolute_max"),
        "total_goal_bias": ("absolute_total_bias_regression_max", "catastrophic_total_bias_absolute_max"),
    }
    output = []
    for key, (relative_name, catastrophic_name) in mapping.items():
        difference = absolute_bias_difference(candidate, production, key)
        relative = thresholds[relative_name]["value"]
        catastrophic = thresholds[catastrophic_name]["value"]
        label = key.removesuffix("_goal_bias") if key != "total_goal_bias" else "total"
        output.append(condition(f"absolute_{label}_bias_relative_noninferiority", difference, f"<= {relative}", difference <= relative, "Absolute candidate bias minus absolute incumbent bias."))
        output.append(condition(f"{label}_bias_catastrophic_safety", abs(candidate[key]), f"<= {catastrophic}", abs(candidate[key]) <= catastrophic, "Broad absolute protection against an obviously broken scoring mean."))
    return output


def mae_noninferiority_conditions(production: dict, candidate: dict, thresholds: dict) -> list[dict[str, Any]]:
    names = ("home_goal_mae", "away_goal_mae", "total_goal_mae", "goal_difference_mae")
    output = []
    for key in names:
        delta = candidate[key] - production[key]
        threshold = thresholds[f"{key}_difference_max"]["value"]
        output.append(condition(f"{key}_noninferiority", delta, f"<= {threshold}", delta <= threshold, "Candidate-minus-production MAE."))
    return output


def fold_bias_stability(production_folds: list[dict], candidate_folds: list[dict], thresholds: dict) -> list[dict[str, Any]]:
    material = thresholds["fold_absolute_bias_material_regression"]["value"]
    maximum = thresholds["maximum_consistently_harmful_bias_folds"]["value"]
    output = []
    for key in BIAS_KEYS:
        deltas = [absolute_bias_difference(c, p, key) for p, c in zip(production_folds, candidate_folds)]
        count = sum(delta > material for delta in deltas)
        output.append(condition(f"{key}_fold_stability", {"differences": deltas, "materially_harmful_folds": count}, f"harm > {material} in <= {maximum} folds", count <= maximum, "Prevents a consistently concentrated chronological bias failure."))
    return output


def paired_date_block_bias_bootstrap(rows, production_predictions, candidate_predictions, *, samples: int, seed: int) -> dict[str, dict[str, float | bool]]:
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows): blocks[row.played_on.isoformat()].append(index)
    dates = sorted(blocks); rng = random.Random(seed); distributions = {key: [] for key in BIAS_KEYS}
    for _ in range(samples):
        indices = [index for _ in dates for index in blocks[rng.choice(dates)]]
        actual_home = sum(rows[i].backtest.home_score for i in indices) / len(indices)
        actual_away = sum(rows[i].backtest.away_score for i in indices) / len(indices)
        values = {}
        for label, predictions in (("production", production_predictions), ("candidate", candidate_predictions)):
            predicted_home = sum(predictions[i][0][0] for i in indices) / len(indices)
            predicted_away = sum(predictions[i][0][1] for i in indices) / len(indices)
            values[label] = {"home_goal_bias": predicted_home-actual_home, "away_goal_bias": predicted_away-actual_away, "total_goal_bias": predicted_home+predicted_away-actual_home-actual_away}
        for key in BIAS_KEYS: distributions[key].append(abs(values["candidate"][key]) - abs(values["production"][key]))
    def percentile(values, q):
        ordered = sorted(values); position = (len(ordered)-1)*q; lower = int(position); fraction = position-lower
        return ordered[lower] if lower+1 == len(ordered) else ordered[lower]*(1-fraction)+ordered[lower+1]*fraction
    return {key: {"lower": percentile(values, .025), "upper": percentile(values, .975), "includes_zero": percentile(values, .025) <= 0 <= percentile(values, .975)} for key, values in distributions.items()}


def bias_bootstrap_conditions(intervals: dict, thresholds: dict) -> list[dict[str, Any]]:
    mapping = {"home_goal_bias": "absolute_home_bias_regression_max", "away_goal_bias": "absolute_away_bias_regression_max", "total_goal_bias": "absolute_total_bias_regression_max"}
    output = []
    for key, threshold_name in mapping.items():
        margin = thresholds[threshold_name]["bootstrap_material_harm"]
        output.append(condition(f"bootstrap_{key}_excludes_material_harm", intervals[key]["upper"], f"< {margin}", intervals[key]["upper"] < margin, "Paired date-block interval for absolute-bias regression; zero may remain inside."))
    return output
