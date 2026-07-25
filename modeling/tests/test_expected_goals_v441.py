import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from modeling.src.expected_goals_v441 import VERSION, SideSpecificGoalModel, fit_poisson_with_penalties, reject_direct_bias_correction
from modeling.src.prospective import ProspectiveLedger
from scripts.persist_scheduled_shadow_batch import persist_batch

ROOT = Path(__file__).resolve().parents[2]


def test_separate_home_and_away_models_and_exact_artifact_loading():
    names = ("intercept", "strength")
    home = fit_poisson_with_penalties([(1.0, 0.0), (1.0, 1.0)], [2, 3], names, (0.0, 8.0), iterations=50)
    away = fit_poisson_with_penalties([(1.0, 0.0), (1.0, 1.0)], [0, 1], names, (0.0, 8.0), iterations=50)
    combined = SideSpecificGoalModel("separate_vectors", home=home, away=away)
    loaded = SideSpecificGoalModel.from_dict(combined.to_dict())
    assert loaded.predict((1.0, 0.5), home_side=True) != loaded.predict((1.0, 0.5), home_side=False)
    payload = combined.to_dict(); payload["model_version"] = "wrong"
    with pytest.raises(ValueError, match="version"):
        SideSpecificGoalModel.from_dict(payload)


def test_coefficient_regularization_is_deterministic_and_shrinks_feature():
    rows = [(1.0, 1.0)] * 10; goals = [3] * 10
    weak = fit_poisson_with_penalties(rows, goals, ("intercept", "home"), (0.0, 1.0), iterations=100)
    strong = fit_poisson_with_penalties(rows, goals, ("intercept", "home"), (0.0, 100.0), iterations=100)
    repeat = fit_poisson_with_penalties(rows, goals, ("intercept", "home"), (0.0, 100.0), iterations=100)
    assert strong == repeat
    assert abs(strong.coefficients[1]) <= abs(weak.coefficients[1])


def test_direct_gate_bias_correction_is_prohibited():
    reject_direct_bias_correction({"architecture": "regularized_home_advantage"})
    with pytest.raises(ValueError, match="prohibited"):
        reject_direct_bias_correction({"home_bias_offset": 0.146164})


def test_neutral_source_and_feature_definition_are_explicit():
    rows = (ROOT / "data/raw/international_results.csv").read_text().splitlines()[1:]
    assert all(row.rsplit(",", 1)[-1].strip().upper() in {"TRUE", "FALSE"} for row in rows)
    definition = json.loads((ROOT / "data/evaluation/elo_context_v441_feature_definitions.json").read_text())
    assert "source neutral boolean exactly" in definition["neutral_rule"]
    assert definition["direct_bias_offset"] is False
    assert definition["draw_calibration_changed"] is False


def test_nested_selection_and_fold_bias_reporting_are_persisted():
    report = json.loads((ROOT / "data/evaluation/elo_context_v441_development.json").read_text())
    assert report["selection_folds"] == [2020, 2021]
    assert report["selected_candidate"] in report["candidate_set_predefined"]
    for fold in report["inner_folds"]:
        assert max(fold["match_ids"]) < "2022"
        for candidate in fold["candidates"].values():
            assert "home_goal_bias" in candidate["metrics"]


def test_historical_gate_v1_was_not_changed_from_v44_lock():
    old_plan = json.loads((ROOT / "data/evaluation/elo_context_v44_historical_plan.json").read_text())
    actual = hashlib.sha256((ROOT / "data/evaluation/historical_gate_v1.json").read_bytes()).hexdigest()
    assert actual == old_plan["gate_specification_sha256"]


def _snapshot(match, version, generated, kickoff):
    return {"match_id": match, "provider_id": match, "kickoff": kickoff.isoformat(), "generated_at": generated.isoformat(), "model_version": version, "home_win": .5, "draw": .3, "away_win": .2, "expected_home_goals": 1.4, "expected_away_goals": .8}


def test_scheduled_production_shadow_batch_is_same_time_idempotent_and_immutable(tmp_path):
    kickoff = datetime.now(timezone.utc) + timedelta(hours=2); generated = kickoff - timedelta(hours=1)
    rows = [_snapshot("m", "elo-context-v4.2.1", generated, kickoff), _snapshot("m", VERSION, generated, kickoff)]
    path = tmp_path / "ledger.json"
    first = persist_batch(path, rows); second = persist_batch(path, rows)
    assert first == second and len(ProspectiveLedger(path).load()["predictions"]) == 2
    rows[1]["home_win"], rows[1]["away_win"] = .4, .3
    with pytest.raises(ValueError, match="Immutable"):
        persist_batch(path, rows)
    late = kickoff + timedelta(seconds=1)
    with pytest.raises(ValueError, match="before kickoff"):
        persist_batch(tmp_path / "late.json", [_snapshot("m", "p", late, kickoff), _snapshot("m", "s", late, kickoff)])


def test_probability_path_remains_normalized_in_development_artifact():
    report = json.loads((ROOT / "data/evaluation/elo_context_v441_development.json").read_text())
    assert report["path_audit"]["draw_calibration_feeds_back_into_xg"] is False
    assert report["path_audit"]["dixon_coles_feeds_back_into_xg"] is False
