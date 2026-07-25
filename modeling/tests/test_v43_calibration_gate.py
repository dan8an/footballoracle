import json
import hashlib
import math
from pathlib import Path

import pytest

from modeling.src.calibration import (
    MultinomialCalibrator,
    TemperatureCalibrator,
    fit_multinomial,
    fit_temperature,
    temperature_log_loss,
)
from modeling.src.dixon_coles import score_matrix, tau
from modeling.src.evaluation.reliability import (
    adaptive_reliability,
    calibration_summary,
    fixed_width_reliability,
)
from modeling.src.readiness import load_readiness


def test_dixon_coles_targets_only_low_scores_before_normalization():
    home_xg, away_xg, rho = 1.4, 1.1, -0.08
    assert tau(0, 0, home_xg, away_xg, rho) > 1
    assert tau(1, 1, home_xg, away_xg, rho) > 1
    assert tau(1, 0, home_xg, away_xg, rho) < 1
    assert tau(0, 1, home_xg, away_xg, rho) < 1
    assert tau(2, 1, home_xg, away_xg, rho) == 1


def test_dixon_coles_matrix_is_nonnegative_and_normalized():
    matrix = score_matrix(1.3, 0.9, -0.05)
    assert all(value >= 0 for row in matrix for value in row)
    assert sum(map(sum, matrix)) == pytest.approx(1.0)


def test_dixon_coles_rejects_invalid_rho():
    with pytest.raises(ValueError):
        score_matrix(1.3, 0.9, 2.0)


def test_temperature_calibration_is_deterministic_and_normalized():
    predictions = [(0.7, 0.2, 0.1), (0.2, 0.5, 0.3), (0.1, 0.2, 0.7)]
    outcomes = [0, 1, 2]
    first = fit_temperature(predictions, outcomes, "candidate")
    second = fit_temperature(predictions, outcomes, "candidate")
    assert first == second
    transformed = first.transform((0.8, 0.15, 0.05))
    assert all(math.isfinite(value) and value >= 0 for value in transformed)
    assert sum(transformed) == pytest.approx(1.0)
    optimum = temperature_log_loss(predictions, outcomes, first.temperature)
    assert optimum <= temperature_log_loss(predictions, outcomes, first.temperature * 1.01)
    if first.temperature > 0.25:
        assert optimum <= temperature_log_loss(predictions, outcomes, first.temperature * 0.99)


def test_reliability_retains_exact_class_bucket_and_support():
    predictions = [(0.75, 0.15, 0.10), (0.72, 0.18, 0.10), (0.2, 0.5, 0.3)]
    outcomes = [0, 2, 1]
    rows = fixed_width_reliability(predictions, outcomes)
    bucket = next(row for row in rows if row["class"] == "home" and row["lower"] == 0.7)
    assert bucket["count"] == 2
    assert bucket["mean_probability"] == pytest.approx(0.735)
    assert bucket["observed_rate"] == pytest.approx(0.5)
    summary = calibration_summary(rows, 3, minimum_support=2)
    assert summary["maximum_calibration_error_bucket"]["count"] >= 1
    assert summary["support_aware_maximum_calibration_error"] is not None


def test_adaptive_reliability_has_documented_minimum_support():
    predictions = [(0.2 + index / 100, 0.3, 0.5 - index / 100) for index in range(40)]
    rows = adaptive_reliability(predictions, [index % 3 for index in range(40)], 20)
    assert len(rows) == 6
    assert all(row["count"] == 20 for row in rows)


def test_multinomial_calibration_is_deterministic_normalized_and_loadable():
    predictions = [(0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)] * 4
    outcomes = [0, 1, 2] * 4
    first = fit_multinomial(predictions, outcomes, "v", 1.0, iterations=100)
    second = fit_multinomial(predictions, outcomes, "v", 1.0, iterations=100)
    assert first == second
    transformed = first.transform((0.6, 0.25, 0.15))
    assert sum(transformed) == pytest.approx(1.0)
    assert all(value >= 0 and math.isfinite(value) for value in transformed)
    assert MultinomialCalibrator.from_dict(first.to_dict(), "v") == first


def test_multinomial_identity_is_exact_fallback():
    calibrator = MultinomialCalibrator(
        ((0, 1, 0), (0, 0, 1), (0, 0, 0)), 1.0, "v", identity=True
    )
    assert calibrator.transform((0.6, 0.3, 0.1)) == pytest.approx((0.6, 0.3, 0.1))


def test_calibrator_rejects_mismatched_artifact():
    with pytest.raises(ValueError):
        TemperatureCalibrator.from_dict(
            {"temperature": 1.0, "model_version": "old"}, "new"
        )


def test_readiness_fails_safe_for_missing_stale_and_failed_artifacts(tmp_path: Path):
    missing = load_readiness(tmp_path / "missing.json", "v1")
    assert not missing["ready"]
    raw = tmp_path / "raw.json"; raw.write_text("{}")
    payload = {
        "candidate_model_version": "v2",
        "current_production_model_version": "v1",
        "historical_gate_passed": True,
        "prospective_status": "limited",
        "plan_hash": "locked-plan",
        "raw_metrics_path": str(raw),
        "raw_metrics_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "promotion_recommendation": "promote",
        "gate": {"overall_status": "pass", "conditions": []},
    }
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(payload))
    assert not load_readiness(path, "v1")["ready"]
    payload["promotion_recommendation"] = "keep_current_experimental_candidate"
    payload["historical_gate_passed"] = False
    payload["gate"] = {
        "overall_status": "fail",
        "conditions": [{"passed": False, "explanation": "bootstrap failed"}],
    }
    path.write_text(json.dumps(payload))
    assert load_readiness(path, "v1")["failed_conditions"] == ["bootstrap failed"]


def test_readiness_pass_language_requires_every_condition(tmp_path: Path):
    raw = tmp_path / "raw.json"; raw.write_text("{}")
    payload = {
        "candidate_model_version": "v2",
        "current_production_model_version": "v1",
        "historical_gate_passed": True,
        "prospective_status": "limited",
        "prospective_match_count": 3,
        "plan_hash": "locked-plan",
        "raw_metrics_path": str(raw),
        "raw_metrics_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "promotion_recommendation": "promote",
        "gate": {
            "overall_status": "pass",
            "conditions": [{"passed": True, "explanation": "all good"}],
        },
    }
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(payload))
    result = load_readiness(path, "v2")
    assert result["ready"]
    assert "does not guarantee future performance" in result["message"]
    assert "3 prospective matches" in result["message"]
