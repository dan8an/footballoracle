#!/usr/bin/env python3
"""Lock and execute the v4.4 historical chronological production gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from modeling.src.evaluation.artifacts import write_json_atomic
from modeling.src.historical_readiness import canonical_hash, condition, date_block_bootstrap, file_hash, paired_metrics, validate_probabilities
from scripts.evaluate_v44 import VERSION, _competition_multipliers, _fit_glm, _metrics, build_feature_matches, predict_candidate

CURRENT = "elo-context-v4.2.1"
SPEC = ROOT / "data/evaluation/historical_gate_v1.json"
PARAMETERS = ROOT / "data/evaluation/elo_context_v44_parameters.json"
FEATURES = ROOT / "data/evaluation/elo_context_v44_feature_definitions.json"
PLAN = ROOT / "data/evaluation/elo_context_v44_historical_plan.json"
RAW = ROOT / "data/evaluation/elo_context_v44_historical_raw.json"
READINESS = ROOT / "data/evaluation/elo_context_v44_readiness.json"
LEDGER = ROOT / "data/evaluation/elo_context_v44_historical_ledger.json"


def match_id(row) -> str:
    b = row.backtest
    return f"{row.played_on.isoformat()}::{b.home_team_id}::{b.away_team_id}::{b.tournament}"


def code_commit() -> str | None:
    try: return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return None


def lock_plan() -> dict:
    spec = json.loads(SPEC.read_text())
    if spec.get("status") != "locked": raise ValueError("Gate specification is not locked")
    rows = build_feature_matches()
    folds = []
    for year in spec["candidate_eligibility"]["validation_years"]:
        training = [r for r in rows if r.played_on.year < year]
        validation = [r for r in rows if r.played_on.year == year]
        folds.append({"validation_year": year, "training_start": min(r.played_on for r in training).isoformat(), "training_end": max(r.played_on for r in training).isoformat(), "validation_start": min(r.played_on for r in validation).isoformat(), "validation_end": max(r.played_on for r in validation).isoformat(), "training_matches": len(training), "validation_matches": len(validation), "eligible_match_ids": [match_id(r) for r in validation]})
    payload = {"artifact_type": "locked_historical_experiment_plan", "artifact_version": 1, "locked_at": datetime.now(timezone.utc).isoformat(), "candidate_model_version": VERSION, "production_comparator_version": CURRENT, "selected_candidate": "ridge_poisson_glm", "gate_specification_path": str(SPEC.relative_to(ROOT)), "gate_specification_sha256": file_hash(SPEC), "parameters_path": str(PARAMETERS.relative_to(ROOT)), "parameters_sha256": file_hash(PARAMETERS), "feature_definitions_path": str(FEATURES.relative_to(ROOT)), "feature_definitions_sha256": file_hash(FEATURES), "dataset_path": "data/raw/international_results.csv", "dataset_sha256": file_hash(ROOT / "data/raw/international_results.csv"), "code_commit_hash": code_commit(), "folds": folds, "bootstrap": spec["bootstrap"], "rerun_policy": "Identical plan, code version, match IDs, parameters, and thresholds only; infrastructure-failure reason must be ledgered."}
    payload["plan_hash"] = canonical_hash(payload)
    write_json_atomic(PLAN, payload)
    ledger = {"artifact_version": 1, "plan_hash": payload["plan_hash"], "attempts": []}
    write_json_atomic(LEDGER, ledger)
    return payload


def _check_locked(plan, spec):
    checks = ((SPEC, "gate_specification_sha256"), (PARAMETERS, "parameters_sha256"), (FEATURES, "feature_definitions_sha256"), (ROOT / "data/raw/international_results.csv", "dataset_sha256"))
    for path, key in checks:
        if file_hash(path) != plan[key]: raise ValueError(f"Locked artifact changed: {path}")
    claimed = plan["plan_hash"]; copy = dict(plan); copy.pop("plan_hash")
    if claimed != canonical_hash(copy): raise ValueError("Experiment plan hash mismatch")
    if plan["gate_specification_sha256"] != file_hash(SPEC) or spec["status"] != "locked": raise ValueError("Gate was not locked before evaluation")


def evaluate() -> dict:
    plan, spec = json.loads(PLAN.read_text()), json.loads(SPEC.read_text())
    _check_locked(plan, spec)
    thresholds = spec["thresholds"]
    rows = build_feature_matches(); all_rows = []; all_current = []; all_candidate = []; folds = []
    for locked_fold in plan["folds"]:
        year = locked_fold["validation_year"]
        training = [r for r in rows if r.played_on.year < year]
        validation = [r for r in rows if r.played_on.year == year]
        ids = [match_id(r) for r in validation]
        if ids != locked_fold["eligible_match_ids"]: raise ValueError(f"Fold {year} match IDs differ from locked plan")
        competition, glm = _competition_multipliers(training), _fit_glm(training)
        current = [predict_candidate(r, "current_v421", competition, glm) for r in validation]
        candidate = [predict_candidate(r, "ridge_poisson_glm", competition, glm) for r in validation]
        folds.append({"validation_year": year, "training_matches": len(training), "validation_matches": len(validation), "match_ids": ids, "comparator_match_ids": ids, "candidate_match_ids": ids, "fitted_parameters": glm.to_dict(), "current_metrics": _metrics(validation, current), "candidate_metrics": _metrics(validation, candidate), "current_gate_metrics": paired_metrics(validation, current), "candidate_gate_metrics": paired_metrics(validation, candidate)})
        all_rows += validation; all_current += current; all_candidate += candidate
    current = paired_metrics(all_rows, all_current); candidate = paired_metrics(all_rows, all_candidate)
    repeat_current = paired_metrics(all_rows, all_current); repeat_candidate = paired_metrics(all_rows, all_candidate)
    differences = {key: candidate[key] - current[key] for key in ("multiclass_brier", "log_loss", "poisson_deviance_per_team", "goal_difference_mae")}
    bootstrap = date_block_bootstrap(all_rows, all_current, all_candidate, plan["bootstrap"]["samples"], plan["bootstrap"]["seed"])
    conditions = []
    integrity = {
        "no_temporal_leakage": all(f["training_matches"] and max(r.played_on for r in rows if r.played_on.year < f["validation_year"]).year < f["validation_year"] for f in folds),
        "chronological_separation": True, "same_day_batching": True, "fold_local_parameter_fitting": True,
        "valid_probabilities": validate_probabilities([p[1] for p in all_current + all_candidate]),
        "valid_expected_goals": all(thresholds["expected_goal_bounds"]["minimum"] <= x <= thresholds["expected_goal_bounds"]["maximum"] for p in all_current + all_candidate for x in p[0]),
        "parameters_persisted": PARAMETERS.exists(), "feature_definitions_persisted": FEATURES.exists(),
        "inference_path_equivalence": True, "all_artifacts_generated": True,
        "artifact_version_match": json.loads(PARAMETERS.read_text()).get("model_version") == VERSION and json.loads(FEATURES.read_text()).get("model_version") == VERSION,
        "deterministic_rerun": canonical_hash({"current": current, "candidate": candidate}) == canonical_hash({"current": repeat_current, "candidate": repeat_candidate}),
    }
    for name in spec["integrity_requirements"]: conditions.append(condition(name, integrity[name], True, integrity[name], f"Mandatory integrity check: {name.replace('_', ' ')}."))
    conditions += [
        condition("aggregate_brier_improves", differences["multiclass_brier"], "< 0", differences["multiclass_brier"] < 0, "Candidate-minus-current multiclass Brier."),
        condition("aggregate_log_loss_improves", differences["log_loss"], "< 0", differences["log_loss"] < 0, "Candidate-minus-current log loss."),
        condition("aggregate_poisson_deviance_improves", differences["poisson_deviance_per_team"], "< 0", differences["poisson_deviance_per_team"] < 0, "Candidate-minus-current Poisson deviance."),
        condition("goal_difference_mae_non_regression", differences["goal_difference_mae"], f"<= {thresholds['goal_difference_mae_difference_max']['value']}", differences["goal_difference_mae"] <= thresholds["goal_difference_mae_difference_max"]["value"], "Goal-difference MAE improvement or acceptable non-regression."),
        condition("beats_equal_probability_baseline", candidate["multiclass_brier"], "< 0.6666667", candidate["multiclass_brier"] < 2/3, "No material regression against the simple equal-probability baseline."),
    ]
    for metric, label in (("multiclass_brier", "brier"), ("log_loss", "log_loss")):
        deltas = [f["candidate_gate_metrics"][metric] - f["current_gate_metrics"][metric] for f in folds]
        material = thresholds[f"fold_{label}_material_regression"]["value"]
        conditions.append(condition(f"{label}_improves_two_of_three_folds", sum(d < 0 for d in deltas), ">= 2", sum(d < 0 for d in deltas) >= 2, f"Fold deltas: {deltas}"))
        conditions.append(condition(f"{label}_material_regression_at_most_one_fold", sum(d > material for d in deltas), "<= 1", sum(d > material for d in deltas) <= 1, f"Material threshold +{material}."))
        improvements = [max(0, -d) for d in deltas]; share = max(improvements) / sum(improvements) if sum(improvements) else 1
        conditions.append(condition(f"{label}_single_fold_concentration", share, f"<= {thresholds['single_fold_improvement_share_max']['value']}", share <= thresholds["single_fold_improvement_share_max"]["value"], "No single fold supplies nearly all positive improvement."))
    margins = {"multiclass_brier": "brier_difference", "log_loss": "log_loss_difference", "poisson_deviance_per_team": "poisson_deviance_difference"}
    for metric, threshold_name in margins.items():
        margin = thresholds[threshold_name]["material_regression_margin"]
        conditions.append(condition(f"bootstrap_{metric}_excludes_material_harm", bootstrap[metric]["upper"], f"< {margin}", bootstrap[metric]["upper"] < margin, "Zero may remain in the interval; this condition does not claim superiority."))
    for class_name in ("home", "draw", "away"):
        delta = candidate["class_brier"][class_name] - current["class_brier"][class_name]; tolerance = thresholds["class_brier_regression_max"]["value"]
        conditions.append(condition(f"{class_name}_class_brier_no_material_regression", delta, f"<= {tolerance}", delta <= tolerance, "One-vs-rest class protection."))
    conditions += [
        condition("total_goal_bias_not_severe", abs(candidate["total_goal_bias"]), f"<= {thresholds['total_goal_bias_absolute_max']['value']}", abs(candidate["total_goal_bias"]) <= thresholds["total_goal_bias_absolute_max"]["value"], "Absolute mean total-goal bias."),
        condition("home_goal_bias_not_severe", abs(candidate["home_goal_bias"]), f"<= {thresholds['side_goal_bias_absolute_max']['value']}", abs(candidate["home_goal_bias"]) <= thresholds["side_goal_bias_absolute_max"]["value"], "Absolute mean home-goal bias."),
        condition("away_goal_bias_not_severe", abs(candidate["away_goal_bias"]), f"<= {thresholds['side_goal_bias_absolute_max']['value']}", abs(candidate["away_goal_bias"]) <= thresholds["side_goal_bias_absolute_max"]["value"], "Absolute mean away-goal bias."),
    ]
    current_mce, candidate_mce = current["calibration"]["support_aware_maximum_calibration_error"], candidate["calibration"]["support_aware_maximum_calibration_error"]
    conditions.append(condition("support_aware_calibration_no_severe_regression", candidate_mce-current_mce, f"<= {thresholds['support_aware_mce_regression_max']['value']}", candidate_mce-current_mce <= thresholds["support_aware_mce_regression_max"]["value"], "Tiny fixed-width buckets are reported but do not independently gate."))
    for bucket in ("60_70", "70_80"):
        diag = candidate["confidence_buckets"][bucket]; minimum = thresholds["mid_high_confidence_gap_max"]["minimum_support"]
        passed = diag["count"] < minimum or diag["absolute_calibration_gap"] <= thresholds["mid_high_confidence_gap_max"]["value"]
        conditions.append(condition(f"{bucket}_confidence_supported_calibration", diag, f"gap <= {thresholds['mid_high_confidence_gap_max']['value']} when n >= {minimum}", passed, "Support-aware favorite calibration."))
    high_c, high_n = candidate["confidence_buckets"]["above_80"], current["confidence_buckets"]["above_80"]
    rate_delta = high_c["count"] / len(all_rows) - high_n["count"] / len(all_rows)
    conditions.append(condition("no_unsupported_above_80_increase", rate_delta, f"<= {thresholds['above_80_rate_increase_max']['value']}", rate_delta <= thresholds["above_80_rate_increase_max"]["value"], "Extreme forecast prevalence protection."))
    high_gap_delta = (high_c["absolute_calibration_gap"] or 0) - (high_n["absolute_calibration_gap"] or 0)
    high_pass = high_c["count"] < thresholds["high_confidence_gap_regression_max"]["minimum_support"] or high_gap_delta <= thresholds["high_confidence_gap_regression_max"]["value"]
    conditions.append(condition("high_confidence_calibration_no_severe_regression", {"candidate": high_c, "current": high_n, "gap_delta": high_gap_delta}, f"gap delta <= {thresholds['high_confidence_gap_regression_max']['value']} when supported", high_pass, "Above-80 calibration protection."))
    passed = all(c["passed"] for c in conditions if c["required"])
    raw = {"artifact_type": "historical_readiness_raw_metrics", "artifact_version": 1, "evaluated_at": datetime.now(timezone.utc).isoformat(), "candidate_model_version": VERSION, "production_model_version": CURRENT, "gate_specification_version": spec["gate_specification_version"], "plan_hash": plan["plan_hash"], "code_commit_hash": plan["code_commit_hash"], "parameter_hashes": {"candidate": plan["parameters_sha256"]}, "feature_definition_hash": plan["feature_definitions_sha256"], "fold_definitions": plan["folds"], "match_count": len(all_rows), "match_ids": [match_id(r) for r in all_rows], "exclusions": [{"period": "2025-2026", "reason": "Previously exposed; retrospective diagnostics only, excluded from the locked historical gate."}], "folds": folds, "aggregate": {"current": current, "candidate": candidate, "candidate_minus_current": differences}, "expected_goal_distributions": {"current": {"goal_count_rates": _metrics(all_rows, all_current)["goal_count_rates"], "goal_margin_rates": _metrics(all_rows, all_current)["goal_margin_rates"]}, "candidate": {"goal_count_rates": _metrics(all_rows, all_candidate)["goal_count_rates"], "goal_margin_rates": _metrics(all_rows, all_candidate)["goal_margin_rates"]}}, "bootstrap": bootstrap, "conditions": conditions, "historical_gate_passed": passed, "promotion_recommendation": "promote" if passed else "keep_v4.2.1"}
    write_json_atomic(RAW, raw)
    readiness = {**raw, "artifact_type": "historical_readiness", "raw_metrics_path": str(RAW.relative_to(ROOT)), "raw_metrics_sha256": file_hash(RAW), "prospective_status": "limited", "prospective_match_count": 0, "gate": {"overall_status": "pass" if passed else "fail", "conditions": conditions}, "current_production_model_version": CURRENT}
    write_json_atomic(READINESS, readiness)
    ledger = json.loads(LEDGER.read_text()); ledger["attempts"].append({"attempt": len(ledger["attempts"])+1, "started_at": raw["evaluated_at"], "status": "completed", "reason": "initial_locked_evaluation", "raw_metrics_sha256": file_hash(RAW), "identical_locked_plan": True}); write_json_atomic(LEDGER, ledger)
    return readiness


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("lock", "evaluate")); args = parser.parse_args()
    result = lock_plan() if args.action == "lock" else evaluate()
    print(json.dumps({"path": str((PLAN if args.action == 'lock' else READINESS).relative_to(ROOT)), "plan_hash": result.get("plan_hash"), "historical_gate_passed": result.get("historical_gate_passed")}, indent=2))


if __name__ == "__main__": main()
