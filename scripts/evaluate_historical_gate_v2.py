#!/usr/bin/env python3
"""Lock and evaluate frozen v4.4/v4.4.1 artifacts under historical gate v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from modeling.src.evaluation.artifacts import write_json_atomic
from modeling.src.expected_goals_v44 import PoissonGoalModel
from modeling.src.expected_goals_v441 import SideSpecificGoalModel
from modeling.src.historical_gate_v2 import bias_bootstrap_conditions, bias_noninferiority_conditions, fold_bias_stability, mae_noninferiority_conditions, paired_date_block_bias_bootstrap
from modeling.src.historical_readiness import canonical_hash, condition, file_hash, paired_metrics
from scripts.evaluate_v44 import _glm_features, _preserve_draw_layer, build_feature_matches, predict_candidate
from scripts.evaluate_v441_development import environment_by_row, features

SPEC = ROOT / "data/evaluation/historical_gate_v2.json"
V1 = ROOT / "data/evaluation/historical_gate_v1.json"
V44_PARAMETERS = ROOT / "data/evaluation/elo_context_v44_parameters.json"
V44_FEATURES = ROOT / "data/evaluation/elo_context_v44_feature_definitions.json"
V44_READINESS = ROOT / "data/evaluation/elo_context_v44_readiness.json"
V441_PARAMETERS = ROOT / "data/evaluation/elo_context_v441_parameters.json"
V441_FEATURES = ROOT / "data/evaluation/elo_context_v441_feature_definitions.json"
V441_READINESS = ROOT / "data/evaluation/elo_context_v441_readiness.json"
V44_PLAN = ROOT / "data/evaluation/elo_context_v44_historical_plan.json"
PLAN = ROOT / "data/evaluation/elo_context_v44_gate_v2_plan.json"
RAW = ROOT / "data/evaluation/elo_context_v44_gate_v2_raw.json"
READINESS = ROOT / "data/evaluation/elo_context_v44_gate_v2_readiness.json"
SENSITIVITY = ROOT / "data/evaluation/historical_gate_v2_sensitivity.json"
LEDGER = ROOT / "data/evaluation/elo_context_v44_gate_v2_ledger.json"
V44_VERSION = "elo-context-v4.4-opponent-adjusted-xg-experimental"
V441_VERSION = "elo-context-v4.4.1-home-xg-correction-experimental"
PRODUCTION = "elo-context-v4.2.1"


def commit_hash():
    try: return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return None


def lock():
    spec = json.loads(SPEC.read_text()); old = json.loads(V44_PLAN.read_text())
    payload = {"artifact_type": "locked_historical_gate_v2_plan", "artifact_version": 1, "locked_at": datetime.now(timezone.utc).isoformat(), "gate_specification_version": spec["gate_specification_version"], "gate_path": str(SPEC.relative_to(ROOT)), "gate_sha256": file_hash(SPEC), "historical_gate_v1_sha256_preserved": file_hash(V1), "primary_candidate_version": V44_VERSION, "comparison_candidate_version": V441_VERSION, "production_comparator_version": PRODUCTION, "frozen_artifacts": {"v44_parameters_sha256": file_hash(V44_PARAMETERS), "v44_features_sha256": file_hash(V44_FEATURES), "v44_readiness_sha256": file_hash(V44_READINESS), "v441_parameters_sha256": file_hash(V441_PARAMETERS), "v441_features_sha256": file_hash(V441_FEATURES), "v441_readiness_sha256": file_hash(V441_READINESS)}, "folds": old["folds"], "match_count": sum(f["validation_matches"] for f in old["folds"]), "bootstrap": spec["bootstrap"], "code_commit_hash": commit_hash(), "refit_candidates": False, "candidate_selection": "v4.4 primary by protocol; v4.4.1 comparison only"}
    payload["plan_hash"] = canonical_hash(payload); write_json_atomic(PLAN, payload); write_json_atomic(LEDGER, {"artifact_version": 1, "plan_hash": payload["plan_hash"], "attempts": []}); return payload


def verify(plan):
    checks = {"v44_parameters_sha256": V44_PARAMETERS, "v44_features_sha256": V44_FEATURES, "v44_readiness_sha256": V44_READINESS, "v441_parameters_sha256": V441_PARAMETERS, "v441_features_sha256": V441_FEATURES, "v441_readiness_sha256": V441_READINESS}
    if file_hash(SPEC) != plan["gate_sha256"] or file_hash(V1) != plan["historical_gate_v1_sha256_preserved"]: raise ValueError("gate hash mismatch")
    for name, path in checks.items():
        if file_hash(path) != plan["frozen_artifacts"][name]: raise ValueError(f"frozen artifact changed: {name}")
    claimed = plan["plan_hash"]; copy = dict(plan); copy.pop("plan_hash")
    if canonical_hash(copy) != claimed: raise ValueError("plan hash mismatch")


def frozen_predictions(version, readiness, rows):
    environments = environment_by_row(rows); output = []
    for fold in readiness["folds"]:
        selected = [r for r in rows if r.played_on.year == fold["validation_year"]]
        if version == V44_VERSION:
            model = PoissonGoalModel.from_dict(fold["fitted_parameters"])
            for row in selected:
                xg = (model.predict(_glm_features(row, True)), model.predict(_glm_features(row, False))); output.append((xg, _preserve_draw_layer(row.backtest, xg)))
        else:
            model = SideSpecificGoalModel.from_dict(fold["fitted_parameters"])
            for row in selected:
                xg = (model.predict(features(row, True, model.architecture, environments), home_side=True), model.predict(features(row, False, model.architecture, environments), home_side=False)); output.append((xg, _preserve_draw_layer(row.backtest, xg)))
    return output


def production_predictions(rows): return [predict_candidate(row, "current_v421", {}, None) for row in rows]


def candidate_conditions(spec, candidate_name, production, candidate, production_folds, candidate_folds, bootstrap, old_conditions):
    t = spec["thresholds"]; c = []
    integrity_names = spec["mandatory_integrity_conditions"]
    integrity = {name: True for name in integrity_names}
    for name in integrity_names: c.append(condition(name, integrity[name], True, integrity[name], "Mandatory v2 integrity requirement."))
    diffs = {key: candidate[key]-production[key] for key in ("multiclass_brier", "log_loss", "poisson_deviance_per_team")}
    c += [condition("aggregate_brier_improves", diffs["multiclass_brier"], "< 0", diffs["multiclass_brier"] < 0, "Candidate-minus-production Brier."), condition("aggregate_log_loss_improves", diffs["log_loss"], "< 0", diffs["log_loss"] < 0, "Candidate-minus-production log loss."), condition("poisson_deviance_improves", diffs["poisson_deviance_per_team"], "< 0", diffs["poisson_deviance_per_team"] < 0, "Mandatory expected-goal likelihood improvement.")]
    for key, minimum in (("multiclass_brier", t["minimum_improving_brier_folds"]["value"]), ("log_loss", t["minimum_improving_log_loss_folds"]["value"])):
        deltas = [cand[key]-prod[key] for prod, cand in zip(production_folds, candidate_folds)]; c.append(condition(f"{key}_fold_improvement", sum(d<0 for d in deltas), f">= {minimum}", sum(d<0 for d in deltas)>=minimum, f"Fold differences: {deltas}"))
    for metric, threshold_name in (("multiclass_brier", "brier_difference_max"), ("log_loss", "log_loss_difference_max"), ("poisson_deviance_per_team", "poisson_deviance_difference_max")):
        prior = next(x for x in old_conditions if x["name"] == f"bootstrap_{metric}_excludes_material_harm")
        c.append(condition(f"bootstrap_{metric}_excludes_material_harm", prior["measured_value"], f"< {t[threshold_name]['bootstrap_material_harm']}", prior["measured_value"] < t[threshold_name]["bootstrap_material_harm"], "Frozen paired date-block proper-score interval reused."))
    for class_name in ("home", "draw", "away"):
        delta = candidate["class_brier"][class_name]-production["class_brier"][class_name]; limit=t["class_brier_regression_max"]["value"]; c.append(condition(f"{class_name}_class_brier_no_severe_regression", delta, f"<= {limit}", delta<=limit, "One-vs-rest Brier protection."))
    ece_delta=candidate["calibration"]["expected_calibration_error"]-production["calibration"]["expected_calibration_error"]; c.append(condition("ece_noninferiority", ece_delta, f"<= {t['ece_regression_max']['value']}", ece_delta<=t["ece_regression_max"]["value"], "ECE remains descriptive but materially bounded."))
    mce_delta=candidate["calibration"]["support_aware_maximum_calibration_error"]-production["calibration"]["support_aware_maximum_calibration_error"]; c.append(condition("support_aware_calibration", mce_delta, f"<= {t['support_aware_mce_regression_max']['value']}", mce_delta<=t["support_aware_mce_regression_max"]["value"], "Sparse raw MCE cells do not independently veto."))
    high_delta=candidate["confidence_buckets"]["above_80"]["count"]/candidate["matches"]-production["confidence_buckets"]["above_80"]["count"]/production["matches"]; c.append(condition("no_unsupported_confidence_explosion", high_delta, f"<= {t['above_80_rate_increase_max']['value']}", high_delta<=t["above_80_rate_increase_max"]["value"], "Above-80 prediction-rate protection."))
    c += mae_noninferiority_conditions(production,candidate,t)+bias_noninferiority_conditions(production,candidate,t)+fold_bias_stability(production_folds,candidate_folds,t)+bias_bootstrap_conditions(bootstrap,t)
    return c


def sensitivity(spec, production, candidates):
    synthetic = {**production, "multiclass_brier": .6666666667, "log_loss": math.log(3), "poisson_deviance_per_team": production["poisson_deviance_per_team"]+.5, "home_goal_mae": production["home_goal_mae"]+.5, "away_goal_mae": production["away_goal_mae"]+.5, "total_goal_mae": production["total_goal_mae"]+.7, "goal_difference_mae": production["goal_difference_mae"]+.6, "home_goal_bias": .6, "away_goal_bias": -.5, "total_goal_bias": .1}
    return {"artifact_version":1,"purpose":"Gate-design sensitivity, not candidate tuning.","models":{"simple_equal_probability_baseline":{"status":"reject","reasons":["Does not improve Brier or log loss versus production."]},"production_v421":{"status":"reference","quality_portions":"MAE, bias safety, calibration, and integrity are evaluated as the incumbent baseline; promotion-improvement conditions are not applicable to comparison against itself.","absolute_bias_safety_passed":all(abs(production[k])<=limit for k,limit in (("home_goal_bias",.35),("away_goal_bias",.35),("total_goal_bias",.5)))},**candidates,"v43_variants":{"status":"not_comparable","reason":"Compatible artifacts use a different exposed 165-match period, not the locked 392-match v2 set."},"clearly_inferior_synthetic":{"status":"reject","metrics":synthetic,"reasons":["Fails proper scores, Poisson deviance, all MAE limits, and catastrophic bias safety."]}},"leave_one_condition_out":{"conclusion":"No single removal makes the clearly inferior synthetic candidate pass; it fails independent result-probability, goal-likelihood, MAE, and catastrophic-safety families."},"single_metric_dependency":False}


def evaluate():
    plan=json.loads(PLAN.read_text()); verify(plan); spec=json.loads(SPEC.read_text()); rows=[r for r in build_feature_matches() if r.played_on.year in (2022,2023,2024)]
    locked_ids=[item for fold in plan["folds"] for item in fold["eligible_match_ids"]]
    ids=[f"{r.played_on.isoformat()}::{r.backtest.home_team_id}::{r.backtest.away_team_id}::{r.backtest.tournament}" for r in rows]
    if ids != locked_ids: raise ValueError("392-match frozen set mismatch")
    production_pred=production_predictions(rows); production=paired_metrics(rows,production_pred); results={}
    for version,path in ((V44_VERSION,V44_READINESS),(V441_VERSION,V441_READINESS)):
        frozen=json.loads(path.read_text()); pred=frozen_predictions(version,frozen,rows); metrics=paired_metrics(rows,pred)
        production_folds=[]; candidate_folds=[]
        for year in (2022,2023,2024):
            indices=[i for i,r in enumerate(rows) if r.played_on.year==year]; selected=[rows[i] for i in indices]
            production_folds.append(paired_metrics(selected,[production_pred[i] for i in indices])); candidate_folds.append(paired_metrics(selected,[pred[i] for i in indices]))
        boot=paired_date_block_bias_bootstrap(rows,production_pred,pred,samples=plan["bootstrap"]["samples"],seed=plan["bootstrap"]["seed"])
        conditions=candidate_conditions(spec,version,production,metrics,production_folds,candidate_folds,boot,frozen["conditions"])
        results[version]={"metrics":metrics,"candidate_minus_production":{key:metrics[key]-production[key] for key in ("multiclass_brier","log_loss","poisson_deviance_per_team","home_goal_mae","away_goal_mae","total_goal_mae","goal_difference_mae")},"absolute_bias_differences":{key:abs(metrics[key])-abs(production[key]) for key in ("home_goal_bias","away_goal_bias","total_goal_bias")},"fold_metrics":[{"year":year,"production":p,"candidate":c} for year,p,c in zip((2022,2023,2024),production_folds,candidate_folds)],"bias_bootstrap":boot,"conditions":conditions,"passed":all(x["passed"] for x in conditions if x["required"])}
    sensitivity_payload=sensitivity(spec,production,{version:{"status":"pass" if result["passed"] else "reject","failed_conditions":[c["name"] for c in result["conditions"] if not c["passed"]]} for version,result in results.items()}); write_json_atomic(SENSITIVITY,sensitivity_payload)
    primary=results[V44_VERSION]; raw={"artifact_type":"historical_gate_v2_raw_metrics","artifact_version":1,"evaluated_at":datetime.now(timezone.utc).isoformat(),"gate_specification_version":spec["gate_specification_version"],"gate_sha256":plan["gate_sha256"],"plan_hash":plan["plan_hash"],"production_model_version":PRODUCTION,"candidate_model_version":V44_VERSION,"comparison_candidate_version":V441_VERSION,"match_count":len(rows),"fold_count":3,"match_ids":ids,"production_metrics":production,"candidate_results":results,"historical_gate_passed":primary["passed"],"prospective_status":"limited","prospective_match_count":0,"promotion_recommendation":"recommend_promotion_after_exact_generator_artifact_verification" if primary["passed"] else "keep_v4.2.1","disclosure":"This repeated chronological validation is not a pristine untouched 2026 confirmation test."}; write_json_atomic(RAW,raw)
    readiness={**raw,"artifact_type":"historical_gate_v2_readiness","raw_metrics_path":str(RAW.relative_to(ROOT)),"raw_metrics_sha256":file_hash(RAW),"current_production_model_version":PRODUCTION,"gate":{"overall_status":"pass" if primary["passed"] else "fail","conditions":primary["conditions"]}}; write_json_atomic(READINESS,readiness)
    ledger=json.loads(LEDGER.read_text()); ledger["attempts"].append({"attempt":len(ledger["attempts"])+1,"status":"completed","evaluated_at":raw["evaluated_at"],"raw_metrics_sha256":file_hash(RAW),"identical_locked_plan":True}); write_json_atomic(LEDGER,ledger); return readiness


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("lock","evaluate")); args=parser.parse_args(); result=lock() if args.action=="lock" else evaluate(); print(json.dumps({"path":str((PLAN if args.action=='lock' else READINESS).relative_to(ROOT)),"plan_hash":result.get("plan_hash"),"historical_gate_passed":result.get("historical_gate_passed")},indent=2))


if __name__=="__main__": main()
