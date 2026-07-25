"""Narrow v4.4.1 home-scoring candidate models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .expected_goals_v44 import PoissonGoalModel

VERSION = "elo-context-v4.4.1-home-xg-correction-experimental"


def fit_poisson_with_penalties(
    features: list[tuple[float, ...]],
    goals: list[int],
    feature_names: tuple[str, ...],
    penalties: tuple[float, ...],
    *,
    iterations: int = 4000,
    learning_rate: float = 0.01,
) -> PoissonGoalModel:
    """Deterministic ridge-Poisson fit with predefined per-coefficient penalties."""
    if not features or len(features) != len(goals) or len(feature_names) != len(penalties):
        raise ValueError("invalid fitting inputs")
    coefficients = [math.log(max(0.2, sum(goals) / len(goals)))] + [0.0] * (len(feature_names) - 1)
    for iteration in range(iterations):
        gradient = [0.0] * len(coefficients)
        for row, goal in zip(features, goals):
            linear = max(-5.0, min(3.0, sum(c * x for c, x in zip(coefficients, row))))
            expected = min(10.0, math.exp(linear))
            for index, value in enumerate(row):
                gradient[index] += (expected - goal) * value / len(goals)
        rate = learning_rate / math.sqrt(1 + iteration / 500)
        for index in range(len(coefficients)):
            penalty = penalties[index] * coefficients[index] / len(goals)
            coefficients[index] -= rate * (gradient[index] + penalty)
    return PoissonGoalModel(tuple(coefficients), feature_names, max(penalties))


@dataclass(frozen=True)
class SideSpecificGoalModel:
    architecture: str
    shared: PoissonGoalModel | None = None
    home: PoissonGoalModel | None = None
    away: PoissonGoalModel | None = None

    def predict(self, features: tuple[float, ...], *, home_side: bool) -> float:
        model = self.home if home_side else self.away
        if model is None:
            model = self.shared
        if model is None:
            raise ValueError("missing fitted side model")
        return model.predict(features)

    def to_dict(self) -> dict:
        return {
            "model_version": VERSION,
            "architecture": self.architecture,
            "shared": self.shared.to_dict() if self.shared else None,
            "home": self.home.to_dict() if self.home else None,
            "away": self.away.to_dict() if self.away else None,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SideSpecificGoalModel":
        if payload.get("model_version") != VERSION:
            raise ValueError("model-version mismatch")
        return cls(
            payload["architecture"],
            PoissonGoalModel.from_dict(payload["shared"]) if payload.get("shared") else None,
            PoissonGoalModel.from_dict(payload["home"]) if payload.get("home") else None,
            PoissonGoalModel.from_dict(payload["away"]) if payload.get("away") else None,
        )


def reject_direct_bias_correction(payload: dict) -> None:
    banned = {"home_bias_offset", "gate_bias_correction", "observed_bias_addback"}
    if banned & payload.keys():
        raise ValueError("direct historical gate-bias correction is prohibited")
