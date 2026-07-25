#!/usr/bin/env python3
"""Build the public scorecard strictly from immutable pre-kickoff snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from modeling.src.evaluation.artifacts import write_json_atomic
from modeling.src.prospective import ProspectiveLedger, build_scorecard

LEDGER = ROOT / "data/evaluation/wc26_prospective_predictions.json"
SCORECARD = ROOT / "data/evaluation/wc26_prospective_scorecard.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, help="JSON object keyed by stable match ID")
    args = parser.parse_args()
    results = json.loads(args.results.read_text()) if args.results else {}
    ledger = ProspectiveLedger(LEDGER).load()
    scorecard = build_scorecard(ledger["predictions"], results)
    scorecard.update({"artifact_version": 1, "ledger_path": str(LEDGER.relative_to(ROOT)), "eligibility_rule": "persisted generation timestamp strictly before official kickoff; no historical backfills", "retrospective_2025_2026_note": "Previously exposed predictions and backtests are retrospective only, not pristine confirmation evidence."})
    scorecard["tracked_model_versions"] = ["elo-context-v4.2.1", "elo-context-v4.4-opponent-adjusted-xg-experimental"]
    scorecard["prospective_prediction_count"] = sum(model["eligible_prospective_predictions"] for model in scorecard["models"].values())
    scorecard["paired_production_candidate_count"] = sum(item["paired_matches"] for item in scorecard["paired_comparisons"])
    scorecard["generation_timestamp_range"] = None if not scorecard["models"] else [min(model["date_range"][0] for model in scorecard["models"].values()), max(model["date_range"][1] for model in scorecard["models"].values())]
    write_json_atomic(SCORECARD, scorecard)
    print(SCORECARD)


if __name__ == "__main__": main()
