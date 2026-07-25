import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from modeling.src.total_goal_bias_audit import calibration_line, full_date_block_bootstrap, reliability
from scripts.persist_scheduled_shadow_batch import persist_batch

ROOT=Path(__file__).resolve().parents[2]


class Backtest:
    def __init__(self,total): self.home_score=total; self.away_score=0


class Row:
    def __init__(self,day,total): self.played_on=day; self.backtest=Backtest(total)


def predictions(totals): return [((total,0.2),(.5,.3,.2)) for total in totals]


def test_absolute_bias_bootstrap_around_zero_and_opposite_signs_is_deterministic():
    rows=[Row(date(2024,1,1),2),Row(date(2024,1,2),4),Row(date(2024,1,3),1)]
    production=predictions([1.0,5.0,.5]); candidate=predictions([3.0,3.0,1.5])
    first=full_date_block_bootstrap(rows,production,candidate,samples=100,seed=9)
    second=full_date_block_bootstrap(rows,production,candidate,samples=100,seed=9)
    assert first==second
    assert len(first["full_distribution"])==100
    assert first["production_crosses_zero_frequency"]>0
    assert first["candidate_crosses_zero_frequency"]>0
    assert first["opposite_sign_frequency"]>0


def test_block_influence_reporting_is_present_and_sorted():
    rows=[Row(date(2024,1,1),0),Row(date(2024,1,2),5),Row(date(2024,1,2),4)]
    result=full_date_block_bootstrap(rows,predictions([1,2,2]),predictions([2,3,3]),samples=50,seed=2)
    influence=result["upper_tail_block_enrichment"]
    assert {row["date"] for row in influence}=={"2024-01-01","2024-01-02"}
    assert all("upper_tail_enrichment" in row and "matches_in_block" in row for row in influence)


def test_calibration_intercept_slope_and_total_reliability():
    line=calibration_line([1,2,3],[2,4,6])
    assert line["intercept"]==pytest.approx(0)
    assert line["slope"]==pytest.approx(2)
    buckets=reliability([1.2,1.8,2.2,3.8],[1,2,2,5])
    assert sum(bucket["count"] for bucket in buckets)==4
    assert any(bucket["signed_gap"] is not None for bucket in buckets)


def test_audit_persisted_full_distribution_and_protected_gate_hashes():
    audit=json.loads((ROOT/"data/evaluation/gate_v2_total_goal_bias_audit_v1.json").read_text())
    assert len(audit["bootstrap"]["full_distribution"])==2000
    assert audit["protected_artifacts_unchanged"] is True
    for name,digest in audit["protected_artifact_hashes_after"].items():
        actual=hashlib.sha256((ROOT/"data/evaluation"/name).read_bytes()).hexdigest()
        assert actual==digest
    assert audit["gate_v2_result_remains"]=="fail"


def test_shadow_batch_exact_versions_and_idempotency(tmp_path):
    rows=json.loads((ROOT/"data/evaluation/fixtures/wc26_shadow_batch_dry_run.json").read_text())
    versions={"elo-context-v4.2.1","elo-context-v4.4-opponent-adjusted-xg-experimental"}; path=tmp_path/"ledger.json"
    assert persist_batch(path,rows,versions)==persist_batch(path,rows,versions)
    wrong=[dict(row) for row in rows]; wrong[1]["model_version"]="wrong"
    with pytest.raises(ValueError,match="exact production and shadow"):
        persist_batch(tmp_path/"wrong.json",wrong,versions)
