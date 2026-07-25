import hashlib
import json
from pathlib import Path

from modeling.src.historical_gate_v2 import absolute_bias_difference, bias_noninferiority_conditions, fold_bias_stability, mae_noninferiority_conditions

ROOT = Path(__file__).resolve().parents[2]


def spec(): return json.loads((ROOT / "data/evaluation/historical_gate_v2.json").read_text())


def test_gate_v1_is_immutable_and_v2_is_separate():
    old_plan=json.loads((ROOT / "data/evaluation/elo_context_v44_historical_plan.json").read_text())
    assert hashlib.sha256((ROOT / "data/evaluation/historical_gate_v1.json").read_bytes()).hexdigest()==old_plan["gate_specification_sha256"]
    assert spec()["gate_specification_version"]=="historical-chronological-v2.0.0"
    assert spec()["difference_convention"].startswith("candidate minus production")


def test_incumbent_relative_bias_and_catastrophic_safety():
    production={"home_goal_bias":-.13,"away_goal_bias":.02,"total_goal_bias":-.11}
    candidate={"home_goal_bias":-.15,"away_goal_bias":.03,"total_goal_bias":-.12}
    assert absolute_bias_difference(candidate,production,"home_goal_bias")==.01999999999999999
    assert all(c["passed"] for c in bias_noninferiority_conditions(production,candidate,spec()["thresholds"]))
    broken={**candidate,"home_goal_bias":.5}
    failed=[c["name"] for c in bias_noninferiority_conditions(production,broken,spec()["thresholds"]) if not c["passed"]]
    assert "home_bias_catastrophic_safety" in failed


def test_expected_goal_mae_noninferiority_and_poisson_is_mandatory_in_spec():
    production={"home_goal_mae":1.0,"away_goal_mae":.9,"total_goal_mae":1.3,"goal_difference_mae":1.2}
    candidate={"home_goal_mae":1.01,"away_goal_mae":.89,"total_goal_mae":1.32,"goal_difference_mae":1.19}
    assert all(c["passed"] for c in mae_noninferiority_conditions(production,candidate,spec()["thresholds"]))
    candidate["home_goal_mae"]=1.03
    assert not next(c for c in mae_noninferiority_conditions(production,candidate,spec()["thresholds"]) if c["name"]=="home_goal_mae_noninferiority")["passed"]
    assert spec()["thresholds"]["poisson_deviance_difference_max"]["value"]==0.0


def test_fold_level_bias_stability_rejects_repeated_harm():
    production=[{"home_goal_bias":.1,"away_goal_bias":.1,"total_goal_bias":.2}]*3
    candidate=[{"home_goal_bias":.25,"away_goal_bias":.1,"total_goal_bias":.35}]*2+[{"home_goal_bias":.1,"away_goal_bias":.1,"total_goal_bias":.2}]
    conditions=fold_bias_stability(production,candidate,spec()["thresholds"])
    assert not next(c for c in conditions if c["name"]=="home_goal_bias_fold_stability")["passed"]


def test_v2_spec_is_locked_before_evaluation_and_has_bias_bootstrap_rules():
    payload=spec()
    assert payload["status"]=="locked_proposal"
    assert payload["created_before_candidate_evaluation"] is True
    assert payload["bootstrap"]["method"]=="paired_date_block_bootstrap"
    assert payload["thresholds"]["absolute_home_bias_regression_max"]["bootstrap_material_harm"]==.08


def test_clearly_inferior_sensitivity_candidate_is_rejected():
    path=ROOT / "data/evaluation/historical_gate_v2_sensitivity.json"
    if path.exists():
        payload=json.loads(path.read_text())
        assert payload["models"]["clearly_inferior_synthetic"]["status"]=="reject"
        assert payload["single_metric_dependency"] is False


def test_frozen_v44_artifact_hash_is_reused_when_plan_exists():
    path=ROOT / "data/evaluation/elo_context_v44_gate_v2_plan.json"
    if path.exists():
        plan=json.loads(path.read_text())
        actual=hashlib.sha256((ROOT / "data/evaluation/elo_context_v44_parameters.json").read_bytes()).hexdigest()
        assert plan["frozen_artifacts"]["v44_parameters_sha256"]==actual


def test_website_distinguishes_historical_and_prospective_evidence():
    source=(ROOT / "apps/web/src/pages/Methodology.tsx").read_text()
    assert "Historical production gate:" in source
    assert "Prospective 2026 scorecard:" in source
    assert "These are separate statuses" in source
