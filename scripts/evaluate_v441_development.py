#!/usr/bin/env python3
"""Nested chronological development and diagnostics for v4.4.1."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from modeling.src.evaluation.artifacts import write_json_atomic
from modeling.src.expected_goals_v44 import PoissonGoalModel, fit_poisson
from modeling.src.expected_goals_v441 import VERSION, SideSpecificGoalModel, fit_poisson_with_penalties, reject_direct_bias_correction
from modeling.src.historical_readiness import paired_metrics
from scripts.evaluate_v44 import FEATURES, FeatureMatch, _competition_multipliers, _fit_glm, _glm_features, _metrics, _preserve_draw_layer, build_feature_matches, predict_candidate

OUT = ROOT / "data/evaluation/elo_context_v441_development.json"
PARAMETERS = ROOT / "data/evaluation/elo_context_v441_parameters.json"
CANDIDATES = ("v44_shared", "separate_intercepts", "separate_vectors", "regularized_home_advantage", "separate_intercepts_environment")
EXTENDED_FEATURES = FEATURES + ("listed_home_side",)
ENV_FEATURES = EXTENDED_FEATURES + ("prior_scoring_environment",)


def environment_by_row(rows: list[FeatureMatch]) -> dict[int, float]:
    result = {}; prior: list[FeatureMatch] = []
    by_date: dict[date, list[FeatureMatch]] = defaultdict(list)
    for row in rows: by_date[row.played_on].append(row)
    for played_on in sorted(by_date):
        recent = [r for r in prior if r.played_on >= played_on - timedelta(days=730)]
        mean = statistics.mean((r.backtest.home_score + r.backtest.away_score) / 2 for r in recent) if recent else 1.35
        value = math.log(max(.5, mean) / 1.35)
        for row in by_date[played_on]: result[id(row)] = value
        prior.extend(by_date[played_on])
    return result


def features(row: FeatureMatch, home_side: bool, architecture: str, environments: dict[int, float]) -> tuple[float, ...]:
    base = _glm_features(row, home_side)
    if architecture == "v44_shared" or architecture == "regularized_home_advantage" or architecture == "separate_vectors": return base
    extended = base + (float(home_side),)
    return extended + (environments[id(row)],) if architecture.endswith("environment") else extended


def fit_candidate(rows: list[FeatureMatch], architecture: str, environments: dict[int, float]) -> SideSpecificGoalModel:
    if architecture == "v44_shared": return SideSpecificGoalModel(architecture, shared=_fit_glm(rows))
    names = ENV_FEATURES if architecture.endswith("environment") else EXTENDED_FEATURES if architecture == "separate_intercepts" else FEATURES
    if architecture == "separate_vectors":
        home_x = [features(r, True, architecture, environments) for r in rows]; away_x = [features(r, False, architecture, environments) for r in rows]
        home = fit_poisson(home_x, [r.backtest.home_score for r in rows], names, ridge=8.0)
        away = fit_poisson(away_x, [r.backtest.away_score for r in rows], names, ridge=8.0)
        return SideSpecificGoalModel(architecture, home=home, away=away)
    stacked = [features(r, side, architecture, environments) for r in rows for side in (True, False)]
    goals = [goal for r in rows for goal in (r.backtest.home_score, r.backtest.away_score)]
    if architecture == "regularized_home_advantage":
        penalties = tuple(0.0 if i == 0 else 8.0 if name == "home_non_neutral" else 1.0 for i, name in enumerate(names))
        model = fit_poisson_with_penalties(stacked, goals, names, penalties)
    else:
        model = fit_poisson(stacked, goals, names, ridge=4.0 if architecture.endswith("environment") else 1.0)
    return SideSpecificGoalModel(architecture, shared=model)


def predictions(rows, model, environments):
    output = []
    for row in rows:
        xg = (model.predict(features(row, True, model.architecture, environments), home_side=True), model.predict(features(row, False, model.architecture, environments), home_side=False))
        output.append((xg, _preserve_draw_layer(row.backtest, xg)))
    return output


def bias_segments(rows, predicted):
    def summarize(indices):
        if not indices: return None
        values = [predicted[i][0][0] - rows[i].backtest.home_score for i in indices]
        return {"matches": len(indices), "home_goal_bias": statistics.mean(values), "home_goal_mae": statistics.mean(abs(v) for v in values)}
    current_xg = [predict_candidate(r, "current_v421", {}, None)[0] for r in rows]
    segments: dict[str, dict] = {}
    definitions = {
        "fold_year": lambda r: str(r.played_on.year),
        "neutral": lambda r: "neutral" if r.backtest.neutral else "non_neutral",
        "competition": lambda r: r.category,
        "home_favorite": lambda r: "away_favorite" if current_xg[rows.index(r)][0] < current_xg[rows.index(r)][1] else "home_edge_under_0_25" if current_xg[rows.index(r)][0]-current_xg[rows.index(r)][1] < .25 else "home_edge_0_25_0_75" if current_xg[rows.index(r)][0]-current_xg[rows.index(r)][1] < .75 else "home_edge_0_75_plus",
        "elo_gap": lambda r: "away_100_plus" if math.log(current_xg[rows.index(r)][0]/current_xg[rows.index(r)][1])*400 < -100 else "within_100" if abs(math.log(current_xg[rows.index(r)][0]/current_xg[rows.index(r)][1])*400) < 100 else "home_100_250" if math.log(current_xg[rows.index(r)][0]/current_xg[rows.index(r)][1])*400 < 250 else "home_250_plus",
        "expected_home_goals": lambda r: "below_1" if predicted[rows.index(r)][0][0] < 1 else "1_1_5" if predicted[rows.index(r)][0][0] < 1.5 else "1_5_2" if predicted[rows.index(r)][0][0] < 2 else "2_plus",
        "team_sample": lambda r: "low" if abs(r.home["long_attack"]) < .05 else "established",
    }
    for dimension, classifier in definitions.items():
        labels = [classifier(r) for r in rows]
        segments[dimension] = {label: summarize([i for i, value in enumerate(labels) if value == label]) for label in sorted(set(labels))}
    return segments


def neutral_audit(rows):
    raw = ROOT / "data/raw/international_results.csv"
    missing = 0
    for line in raw.read_text().splitlines()[1:]:
        value = line.rsplit(",", 1)[-1].strip().upper()
        missing += value not in {"TRUE", "FALSE"}
    return {"source": str(raw.relative_to(ROOT)), "missing_or_unknown_values": missing, "world_cup_marked_non_neutral": sum(r.category == "world_cup" and not r.backtest.neutral for r in rows), "world_cup_marked_neutral": sum(r.category == "world_cup" and r.backtest.neutral for r in rows), "rule": "Use the source neutral boolean exactly; no venue-country data is available and no competition-derived status is invented.", "training_inference_consistency": True}


def run():
    rows = build_feature_matches(); environments = environment_by_row(rows)
    inner_folds = []
    for year in (2020, 2021):
        training = [r for r in rows if r.played_on.year < year]; validation = [r for r in rows if r.played_on.year == year]
        fold = {"year": year, "match_ids": [f"{r.played_on}::{r.backtest.home_team_id}::{r.backtest.away_team_id}" for r in validation], "candidates": {}}
        for name in CANDIDATES:
            model = fit_candidate(training, name, environments); pred = predictions(validation, model, environments)
            fold["candidates"][name] = {"metrics": paired_metrics(validation, pred), "coefficients": model.to_dict()}
        inner_folds.append(fold)
    aggregates = {}
    for name in CANDIDATES:
        ms = [f["candidates"][name]["metrics"] for f in inner_folds]
        aggregates[name] = {key: statistics.mean(m[key] for m in ms) for key in ("multiclass_brier", "log_loss", "poisson_deviance_per_team", "home_goal_bias", "away_goal_bias", "total_goal_bias", "home_goal_mae", "away_goal_mae", "total_goal_mae", "goal_difference_mae")}
    base = aggregates["v44_shared"]
    eligible = [name for name in CANDIDATES if abs(aggregates[name]["home_goal_bias"]) < abs(base["home_goal_bias"]) and aggregates[name]["multiclass_brier"] <= base["multiclass_brier"] + .01 and aggregates[name]["log_loss"] <= base["log_loss"] + .02 and abs(aggregates[name]["away_goal_bias"]) <= .10]
    selected = min(eligible, key=lambda n: (abs(aggregates[n]["home_goal_bias"]), aggregates[n]["poisson_deviance_per_team"])) if eligible else "v44_shared"
    final = fit_candidate(rows, selected, environments); reject_direct_bias_correction(final.to_dict()); final_predictions = predictions(rows, final, environments)
    parameters = {**final.to_dict(), "fitted_through": "2024-12-31", "selection_protocol": "predefined candidates selected on 2020-2021 inner chronological folds only", "direct_bias_addback": False, "dataset_sha256": hashlib.sha256((ROOT / "data/raw/international_results.csv").read_bytes()).hexdigest()}
    write_json_atomic(PARAMETERS, parameters)
    report = {"artifact_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "model_version": VERSION, "candidate_set_predefined": list(CANDIDATES), "selection_folds": [2020, 2021], "inner_folds": inner_folds, "aggregate_inner_metrics": aggregates, "selection_eligibility": eligible, "selected_candidate": selected, "neutral_audit": neutral_audit(rows), "selected_diagnostics_all_development_rows": {"metrics": _metrics(rows, final_predictions), "bias_segments": bias_segments(rows, final_predictions)}, "path_audit": {"draw_calibration_feeds_back_into_xg": False, "dixon_coles_feeds_back_into_xg": False, "confidence_thresholds_feed_back_into_xg": False, "clipping_bounds": [.2, 4.5], "goal_difference_amplification_used_by_selected_candidate": False, "competition_multiplier_used_by_glm": False, "competition_effect": "fitted indicator coefficients", "regularization": final.to_dict()}}
    write_json_atomic(OUT, report); return report


if __name__ == "__main__":
    result = run(); print(json.dumps({"selected": result["selected_candidate"], "output": str(OUT.relative_to(ROOT))}, indent=2))
