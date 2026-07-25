#!/usr/bin/env python3
"""Diagnostic audit of the sole failed gate-v2 total-bias bootstrap condition."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from modeling.src.evaluation.artifacts import write_json_atomic
from modeling.src.total_goal_bias_audit import block_error_diagnostics, calibration_line, full_date_block_bootstrap, poisson_goal_bands, reliability
from scripts.evaluate_historical_gate_v2 import V44_READINESS, V44_VERSION, build_feature_matches, frozen_predictions, production_predictions

OUT=ROOT/"data/evaluation/gate_v2_total_goal_bias_audit_v1.json"
PROTECTED=("historical_gate_v1.json","historical_gate_v2.json","elo_context_v44_gate_v2_plan.json","elo_context_v44_gate_v2_readiness.json","elo_context_v44_gate_v2_raw.json","elo_context_v44_gate_v2_ledger.json")


def hash_file(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def segment(rows, production, candidate, classifier):
    labels=[classifier(row,i) for i,row in enumerate(rows)]; output={}
    for label in sorted(set(labels)):
        indices=[i for i,value in enumerate(labels) if value==label]; actual=statistics.mean(rows[i].backtest.home_score+rows[i].backtest.away_score for i in indices)
        output[label]={"matches":len(indices),"actual_total_goals":actual,"production_predicted_total":statistics.mean(sum(production[i][0]) for i in indices),"candidate_predicted_total":statistics.mean(sum(candidate[i][0]) for i in indices),"production_bias":statistics.mean(sum(production[i][0]) for i in indices)-actual,"candidate_bias":statistics.mean(sum(candidate[i][0]) for i in indices)-actual}
    return output


def run():
    evaluation=ROOT/"data/evaluation"; before={name:hash_file(evaluation/name) for name in PROTECTED}
    rows=[r for r in build_feature_matches() if r.played_on.year in (2022,2023,2024)]; frozen=json.loads(V44_READINESS.read_text()); production=production_predictions(rows); candidate=frozen_predictions(V44_VERSION,frozen,rows)
    bootstrap=full_date_block_bootstrap(rows,production,candidate,samples=2000,seed=2440421); actual=[r.backtest.home_score+r.backtest.away_score for r in rows]; production_total=[sum(p[0]) for p in production]; candidate_total=[sum(p[0]) for p in candidate]
    blocks=block_error_diagnostics(rows,production,candidate)
    current_metrics=frozen["aggregate"]["current"]; candidate_metrics=frozen["aggregate"]["candidate"]
    alternative={
      "candidate_signed_bias_clustered_interval":{"point":statistics.mean(p-a for p,a in zip(candidate_total,actual)),"interval":[bootstrap["candidate_signed_bias_summary"]["quantiles"]["0.025"],bootstrap["candidate_signed_bias_summary"]["quantiles"]["0.975"]],"detects":"directional aggregate calibration","cancellation_sensitive":True,"duplicates_existing_gate":False},
      "candidate_minus_production_signed_bias":{"point":statistics.mean(c-p for c,p in zip(candidate_total,production_total)),"bootstrap_summary":__import__("modeling.src.total_goal_bias_audit",fromlist=["distribution_summary"]).distribution_summary([r["candidate_minus_production_signed_bias"] for r in bootstrap["full_distribution"]]),"detects":"systematic mean shift between models","cancellation_sensitive":False,"duplicates_existing_gate":False},
      "absolute_aggregate_bias_difference":{"point":abs(statistics.mean(c-a for c,a in zip(candidate_total,actual)))-abs(statistics.mean(p-a for p,a in zip(production_total,actual))),"summary":bootstrap["absolute_bias_difference_summary"],"detects":"which aggregate mean is closer to zero","cancellation_sensitive":True,"non_differentiable_at_zero":True,"duplicates_existing_gate":True},
      "mean_absolute_block_bias":blocks["mean_absolute_block_bias"]|{"detects":"typical date-level calibration error","cancellation_sensitive":False,"small_block_sensitive":True},
      "root_mean_square_block_bias":blocks["root_mean_square_block_bias"]|{"detects":"large date-level calibration failures","cancellation_sensitive":False,"outlier_sensitive":True},
      "total_goal_mae_difference":{"value":candidate_metrics["total_goal_mae"]-current_metrics["total_goal_mae"],"detects":"typical per-match total-goal error","duplicates_existing_gate":True},
      "poisson_deviance_difference":{"value":candidate_metrics["poisson_deviance_per_team"]-current_metrics["poisson_deviance_per_team"],"detects":"goal-count likelihood quality","duplicates_existing_gate":True},
      "calibration_line":{"production":calibration_line(production_total,actual),"candidate":calibration_line(candidate_total,actual),"detects":"global level and scaling calibration","small_range_sensitive":True},
      "predicted_total_reliability":{"production":reliability(production_total,actual),"candidate":reliability(candidate_total,actual),"detects":"local calibration by forecast scoring environment"},
      "goal_count_bands":{"production":poisson_goal_bands(production_total,actual),"candidate":poisson_goal_bands(candidate_total,actual),"detects":"distributional calibration at operational goal bands"}
    }
    current_xg=[p[0] for p in production]
    segments={
      "fold":segment(rows,production,candidate,lambda r,i:str(r.played_on.year)),
      "calendar_year":segment(rows,production,candidate,lambda r,i:str(r.played_on.year)),
      "competition":segment(rows,production,candidate,lambda r,i:r.category),
      "neutral":segment(rows,production,candidate,lambda r,i:"neutral" if r.backtest.neutral else "non_neutral"),
      "predicted_total":segment(rows,production,candidate,lambda r,i:"below_2" if sum(candidate[i][0])<2 else "2_2_5" if sum(candidate[i][0])<2.5 else "2_5_3" if sum(candidate[i][0])<3 else "3_plus"),
      "elo_gap":segment(rows,production,candidate,lambda r,i:"within_100" if abs(math.log(current_xg[i][0]/current_xg[i][1])*400)<100 else "100_250" if abs(math.log(current_xg[i][0]/current_xg[i][1])*400)<250 else "250_plus"),
      "favorite_confidence":segment(rows,production,candidate,lambda r,i:"below_40" if max(candidate[i][1])<.4 else "40_50" if max(candidate[i][1])<.5 else "50_60" if max(candidate[i][1])<.6 else "60_70" if max(candidate[i][1])<.7 else "70_80" if max(candidate[i][1])<.8 else "above_80")
    }
    after={name:hash_file(evaluation/name) for name in PROTECTED}
    payload={"artifact_type":"gate_v2_failed_statistic_diagnostic","artifact_version":"1.0.0","scope":"diagnostic research only; does not alter or rerun gate v2","protected_artifact_hashes_before":before,"protected_artifact_hashes_after":after,"protected_artifacts_unchanged":before==after,"gate_v2_result_remains":"fail","production_model_version":"elo-context-v4.2.1","candidate_model_version":V44_VERSION,"match_count":len(rows),"mathematical_interpretation":{"statistic":"abs(candidate mean residual) - abs(production mean residual)","absolute_value_effect":"Folds negative and positive signed means onto the same scale, discarding direction.","non_differentiability":"The transformation has a cusp at zero; small resample changes can reverse the local slope.","opposite_sign_effect":"Because production is positive and candidate negative, resampled zero crossings change which residual magnitude shrinks or grows and widen a non-Gaussian comparison.","predictive_harm_interpretation":"Primarily ranks closeness of two aggregate means to zero; it is not a direct estimate of typical per-match loss."},"bootstrap":bootstrap,"block_diagnostics":blocks,"alternative_diagnostics":alternative,"segments":segments,"recommendation":{"existing_gate_v2":"Retain its recorded failure unchanged.","future_gate_design":"Demote absolute aggregate-bias bootstrap to monitoring or supplement it; study block-level absolute calibration, MAE, deviance, fold stability, catastrophic limits, and goal-band reliability in an independently specified gate v3.","gate_v3_validation_principle":"Pre-register on domain scale, test sensitivity across multiple frozen good and bad models, validate temporal stability, and do not evaluate promotion until the new proposal is hashed."}}
    write_json_atomic(OUT,payload); return payload


if __name__=="__main__":
    result=run(); print(json.dumps({"output":str(OUT.relative_to(ROOT)),"unchanged":result["protected_artifacts_unchanged"]},indent=2))
