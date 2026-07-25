"""Fail-safe loading of the versioned production-readiness artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"candidate_model_version", "current_production_model_version", "historical_gate_passed", "prospective_status", "gate", "promotion_recommendation", "plan_hash", "raw_metrics_path", "raw_metrics_sha256"}


def load_readiness(path: Path, active_model_version: str) -> dict[str, Any]:
    fallback = {"status": "fail", "ready": False, "historical_gate_passed": False, "prospective_status": "unavailable", "prospective_match_count": 0, "production_model_version": active_model_version, "candidate_model_version": None, "message": "A valid, matching historical-readiness artifact is unavailable.", "failed_conditions": ["A valid, matching readiness artifact is unavailable."]}
    try:
        payload = json.loads(path.read_text())
        if not REQUIRED_FIELDS.issubset(payload):
            return fallback
        expected_active = payload["candidate_model_version"] if payload.get("promotion_recommendation") == "promote" else payload["current_production_model_version"]
        if expected_active != active_model_version:
            return fallback
        raw_path = path.parents[2] / payload["raw_metrics_path"] if not Path(payload["raw_metrics_path"]).is_absolute() else Path(payload["raw_metrics_path"])
        if not raw_path.exists() or hashlib.sha256(raw_path.read_bytes()).hexdigest() != payload["raw_metrics_sha256"]:
            return fallback
        conditions = payload["gate"]["conditions"]
        failed = [condition["explanation"] for condition in conditions if condition.get("required", True) and not condition.get("passed")]
        passed = payload["historical_gate_passed"] is True and payload["gate"].get("overall_status") == "pass" and not failed
        if not passed:
            return {"status": "fail", "ready": False, "historical_gate_passed": False, "historical_gate_version": payload.get("gate_specification_version"), "historical_match_count": payload.get("match_count"), "historical_fold_count": payload.get("fold_count", len(payload.get("fold_definitions", []))), "prospective_status": payload["prospective_status"], "prospective_match_count": payload.get("prospective_match_count", 0), "production_model_version": active_model_version, "candidate_model_version": payload["candidate_model_version"], "historical_disclosure": payload.get("disclosure"), "message": "This candidate has not passed the project’s historical promotion gate.", "failed_conditions": failed or ["The overall historical gate did not pass."]}
        count = payload.get("prospective_match_count", 0)
        version = payload.get("gate_specification_version", "historical gate")
        return {"status": "pass", "ready": True, "historical_gate_passed": True, "historical_gate_version": version, "historical_match_count": payload.get("match_count"), "historical_fold_count": payload.get("fold_count", len(payload.get("fold_definitions", []))), "prospective_status": payload["prospective_status"], "prospective_match_count": count, "production_model_version": active_model_version, "candidate_model_version": payload["candidate_model_version"], "historical_disclosure": payload.get("disclosure"), "message": f"This release passed the project’s locked leakage-safe historical chronological-validation gate {version}. This was not a pristine untouched 2026 holdout. Prospective World Cup performance is tracked separately and currently includes {count} prospective matches. Prospective tournament evidence is still limited; historical validation does not guarantee future performance.", "failed_conditions": []}
    except (OSError, ValueError, TypeError, KeyError):
        return fallback
