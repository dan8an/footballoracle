"""Immutable, model-version-aware prospective prediction ledger and scorecard."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modeling.src.evaluation.artifacts import write_json_atomic


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def prediction_hash(snapshot: dict[str, Any]) -> str:
    immutable = {key: snapshot[key] for key in ("match_id", "provider_id", "kickoff", "generated_at", "model_version", "home_win", "draw", "away_win", "expected_home_goals", "expected_away_goals") if key in snapshot}
    return hashlib.sha256(json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ProspectiveLedger:
    def __init__(self, path: Path): self.path = path
    def load(self) -> dict[str, Any]:
        if not self.path.exists(): return {"artifact_version": 1, "predictions": []}
        payload = json.loads(self.path.read_text())
        if not isinstance(payload.get("predictions"), list): raise ValueError("Malformed prospective ledger")
        for row in payload["predictions"]:
            if row.get("content_hash") != prediction_hash(row): raise ValueError("Prospective snapshot hash mismatch")
        return payload
    def persist(self, snapshot: dict[str, Any]) -> str:
        required = {"match_id", "kickoff", "generated_at", "model_version", "home_win", "draw", "away_win"}
        if not required.issubset(snapshot): raise ValueError("Prospective snapshot missing required fields")
        if _timestamp(snapshot["generated_at"]) >= _timestamp(snapshot["kickoff"]): raise ValueError("Prediction was not generated before kickoff")
        probabilities = [snapshot[k] for k in ("home_win", "draw", "away_win")]
        if not all(math.isfinite(x) and x >= 0 for x in probabilities) or abs(sum(probabilities) - 1) > 1e-9: raise ValueError("Invalid probabilities")
        payload = self.load(); key = (str(snapshot["match_id"]), snapshot["model_version"])
        existing = next((row for row in payload["predictions"] if (str(row["match_id"]), row["model_version"]) == key), None)
        candidate = dict(snapshot); candidate["content_hash"] = prediction_hash(candidate)
        if existing:
            if existing["content_hash"] != candidate["content_hash"]: raise ValueError("Immutable prospective snapshot already exists")
            return existing["content_hash"]
        payload["predictions"].append(candidate); write_json_atomic(self.path, payload); return candidate["content_hash"]


def build_scorecard(predictions: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = [p for p in predictions if _timestamp(p["generated_at"]) < _timestamp(p["kickoff"]) and str(p["match_id"]) in results]
    by_version: dict[str, list[dict[str, Any]]] = {}
    for p in eligible: by_version.setdefault(p["model_version"], []).append(p)
    output = {}
    for version, rows in by_version.items():
        outcomes = [int(results[str(p["match_id"])]["outcome"]) for p in rows]; probs = [(p["home_win"], p["draw"], p["away_win"]) for p in rows]; n = len(rows)
        goal_rows = [(p, results[str(p["match_id"])]) for p in rows if p.get("expected_home_goals") is not None and p.get("expected_away_goals") is not None]
        output[version] = {"eligible_prospective_predictions": n, "date_range": [min(p["kickoff"] for p in rows), max(p["kickoff"] for p in rows)], "stages": sorted({results[str(p["match_id"])].get("stage", "unknown") for p in rows}), "brier": sum(sum((q[c] - (y == c)) ** 2 for c in range(3)) for q, y in zip(probs, outcomes)) / n, "log_loss": sum(-math.log(max(1e-15, q[y])) for q, y in zip(probs, outcomes)) / n, "accuracy": sum(max(range(3), key=lambda c: q[c]) == y for q, y in zip(probs, outcomes)) / n, "class_brier": {name: sum((q[c] - (y == c)) ** 2 for q, y in zip(probs, outcomes)) / n for c, name in enumerate(("home", "draw", "away"))}, "mean_probability_actual_class": sum(q[y] for q, y in zip(probs, outcomes)) / n, "predicted_class_frequencies": {name: sum(q[c] for q in probs) / n for c, name in enumerate(("home", "draw", "away"))}, "observed_class_frequencies": {name: sum(y == c for y in outcomes) / n for c, name in enumerate(("home", "draw", "away"))}, "expected_goal_metrics": _goal_metrics(goal_rows), "sample_size_warning": "Prospective tournament evidence is still limited; calibration conclusions are unstable." if n < 50 else None}
    versions = sorted(by_version); paired = []
    for i, left in enumerate(versions):
        for right in versions[i + 1:]:
            common = sorted({str(p["match_id"]) for p in by_version[left]} & {str(p["match_id"]) for p in by_version[right]})
            paired.append({"model_versions": [left, right], "paired_matches": len(common), "match_ids": common})
    return {"prospective_status": "limited" if len(eligible) < 50 else "accumulating", "models": output, "paired_comparisons": paired}


def _goal_metrics(rows):
    if not rows: return {"matches": 0}
    n = len(rows)
    return {"matches": n, "home_goal_mae": sum(abs(p["expected_home_goals"] - r["home_goals"]) for p, r in rows) / n, "away_goal_mae": sum(abs(p["expected_away_goals"] - r["away_goals"]) for p, r in rows) / n, "total_goal_mae": sum(abs(p["expected_home_goals"] + p["expected_away_goals"] - r["home_goals"] - r["away_goals"]) for p, r in rows) / n, "goal_difference_mae": sum(abs(p["expected_home_goals"] - p["expected_away_goals"] - r["home_goals"] + r["away_goals"]) for p, r in rows) / n}
