#!/usr/bin/env python3
"""Lock and run v4.4.1 through the unchanged historical_gate_v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import scripts.evaluate_historical_gate_v44 as gate
from modeling.src.expected_goals_v441 import VERSION
from scripts.evaluate_v44 import _preserve_draw_layer, predict_candidate as v44_prediction
from scripts.evaluate_v441_development import environment_by_row, features, fit_candidate

ROWS = gate.build_feature_matches()
ENVIRONMENTS = environment_by_row(ROWS)
PARAMETER_PAYLOAD = json.loads((ROOT / "data/evaluation/elo_context_v441_parameters.json").read_text())
SELECTED = PARAMETER_PAYLOAD["architecture"]


def fitted(rows):
    return fit_candidate(rows, SELECTED, ENVIRONMENTS)


def predicted(row, name, competition, model):
    if name == "current_v421":
        return v44_prediction(row, name, competition, None)
    xg = (
        model.predict(features(row, True, model.architecture, ENVIRONMENTS), home_side=True),
        model.predict(features(row, False, model.architecture, ENVIRONMENTS), home_side=False),
    )
    return xg, _preserve_draw_layer(row.backtest, xg)


gate.VERSION = VERSION
gate.PARAMETERS = ROOT / "data/evaluation/elo_context_v441_parameters.json"
gate.FEATURES = ROOT / "data/evaluation/elo_context_v441_feature_definitions.json"
gate.PLAN = ROOT / "data/evaluation/elo_context_v441_historical_plan.json"
gate.RAW = ROOT / "data/evaluation/elo_context_v441_historical_raw.json"
gate.READINESS = ROOT / "data/evaluation/elo_context_v441_readiness.json"
gate.LEDGER = ROOT / "data/evaluation/elo_context_v441_historical_ledger.json"
gate.build_feature_matches = lambda: ROWS
gate._fit_glm = fitted
gate.predict_candidate = predicted


if __name__ == "__main__":
    gate.main()
