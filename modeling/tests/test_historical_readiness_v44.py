import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from modeling.src.historical_readiness import date_block_bootstrap
from modeling.src.prospective import ProspectiveLedger, build_scorecard


class Backtest:
    outcome = 0
    home_score = 1
    away_score = 0


class Row:
    def __init__(self, day):
        self.played_on = day
        self.backtest = Backtest()


def test_date_block_bootstrap_is_deterministic():
    from datetime import date
    rows = [Row(date(2022, 1, 1)), Row(date(2022, 1, 1)), Row(date(2022, 1, 2))]
    current = [((1.2, .8), (.5, .25, .25))] * 3
    candidate = [((1.1, .7), (.55, .25, .2))] * 3
    assert date_block_bootstrap(rows, current, candidate, 25, 7) == date_block_bootstrap(rows, current, candidate, 25, 7)


def snapshot(model="v1"):
    kickoff = datetime.now(timezone.utc) + timedelta(hours=2)
    return {"match_id": "m1", "provider_id": "p1", "kickoff": kickoff.isoformat(), "generated_at": (kickoff - timedelta(hours=1)).isoformat(), "model_version": model, "home_win": .5, "draw": .3, "away_win": .2, "expected_home_goals": 1.4, "expected_away_goals": .8}


def test_prospective_snapshot_is_pre_kickoff_immutable_and_idempotent(tmp_path: Path):
    ledger = ProspectiveLedger(tmp_path / "ledger.json"); row = snapshot()
    first = ledger.persist(row)
    assert ledger.persist(row) == first
    changed = {**row, "home_win": .4, "away_win": .3}
    with pytest.raises(ValueError, match="Immutable"):
        ledger.persist(changed)
    after = snapshot("v2"); after["generated_at"] = after["kickoff"]
    with pytest.raises(ValueError, match="before kickoff"):
        ledger.persist(after)


def test_shadow_versions_and_paired_scorecard_are_separate(tmp_path: Path):
    ledger = ProspectiveLedger(tmp_path / "ledger.json")
    ledger.persist(snapshot("production")); ledger.persist(snapshot("shadow"))
    scorecard = build_scorecard(ledger.load()["predictions"], {"m1": {"outcome": 0, "home_goals": 2, "away_goals": 0, "stage": "group"}})
    assert set(scorecard["models"]) == {"production", "shadow"}
    assert scorecard["paired_comparisons"][0]["paired_matches"] == 1
    assert scorecard["prospective_status"] == "limited"


def test_after_kickoff_rows_are_rejected_from_scorecard():
    row = snapshot(); row["generated_at"] = row["kickoff"]
    assert build_scorecard([row], {"m1": {"outcome": 0}})["models"] == {}


def test_locked_plan_contains_equal_candidate_and_comparator_match_sets():
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "data/evaluation/elo_context_v44_historical_plan.json").read_text())
    readiness = json.loads((root / "data/evaluation/elo_context_v44_readiness.json").read_text())
    assert plan["plan_hash"] == readiness["plan_hash"]
    assert all(f["candidate_match_ids"] == f["comparator_match_ids"] for f in readiness["folds"])
    assert readiness["historical_gate_passed"] is not readiness.get("prospective_status")
    assert all("threshold" in c and "measured_value" in c and "passed" in c for c in readiness["conditions"])
