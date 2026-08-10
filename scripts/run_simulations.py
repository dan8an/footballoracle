#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import JSON, MetaData, Table, inspect, select, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.src.data import build_fixtures, load_teams, validate_tournament
from modeling.src.historical_snapshots import (
    SNAPSHOTS,
    SNAPSHOT_ACTIVE_TEAM_COUNTS,
    get_snapshot,
    resolve_snapshot_cutoff,
)
from scripts.database import create_database_engine
from scripts.generate_predictions import (
    MODEL_VERSION,
    PredictionRepository,
    calculate_prediction,
    canonical_prior_elo,
)

DEFAULT_SIMULATIONS = 50_000
DEFAULT_SEED = 2026
SIMULATION_CONFIG_VERSION = "historical-snapshot-v1"
STAGES = (
    "group_stage_exit",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
)
COMPLETED_MATCH_STATUSES = {"completed", "finished", "ft", "aet", "pen"}
GROUP_STAGE_ALIASES = {"group", "group stage", "group_stage", "first round"}
KNOCKOUT_STAGES = (
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
)
KNOCKOUT_STAGE_LIMITS = {
    "round_of_32": 16,
    "round_of_16": 8,
    "quarterfinal": 4,
    "semifinal": 2,
    "final": 1,
}
KNOCKOUT_MATCH_NUMBER_RANGES = {
    "round_of_32": range(73, 89),
    "round_of_16": range(89, 97),
    "quarterfinal": range(97, 101),
    "semifinal": range(101, 103),
    "final": range(104, 105),
}
GROUP_WINDOW_START = datetime(2026, 6, 11, tzinfo=timezone.utc)
GROUP_WINDOW_END = datetime(2026, 6, 28, tzinfo=timezone.utc)
KNOCKOUT_WINDOW_START = datetime(2026, 6, 28, tzinfo=timezone.utc)
KNOCKOUT_WINDOW_END = datetime(2026, 7, 20, tzinfo=timezone.utc)


@dataclass(frozen=True)
class MatchState:
    id: str
    stage: str
    home_team_id: str
    away_team_id: str
    completed: bool
    home_score: int | None = None
    away_score: int | None = None
    home_penalty_score: int | None = None
    away_penalty_score: int | None = None
    match_number: int | None = None
    kickoff: datetime | None = None


def load_environment() -> dict[str, str]:
    """Load server-side env files without replacing exported values."""
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / "backend" / ".env", override=False)
    return dict(os.environ)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical 2026 World Cup Monte Carlo simulation."
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=f"Number of tournaments to simulate (default: {DEFAULT_SIMULATIONS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic random seed (default: {DEFAULT_SEED}).",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--snapshot",
        choices=[snapshot.key for snapshot in SNAPSHOTS],
        help="Generate one cutoff-conditioned historical snapshot.",
    )
    selection.add_argument(
        "--all-snapshots",
        action="store_true",
        help="Generate all canonical historical snapshots in chronological order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit cutoff and input coverage without simulating or writing.",
    )
    return parser.parse_args()


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _timestamp_floor(value: Any) -> datetime:
    return _parse_timestamp(value) or datetime.min.replace(tzinfo=timezone.utc)


def _has_score(row: dict[str, Any]) -> bool:
    return row.get("home_score") is not None and row.get("away_score") is not None


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = (
        _json_value(row.get("provider_payload"))
        or _json_value(row.get("raw_payload"))
        or _json_value(row.get("raw"))
        or {}
    )
    return payload if isinstance(payload, dict) else {}


def _is_world_cup_2026_provider_row(row: dict[str, Any]) -> bool:
    payload = _row_payload(row)
    league = payload.get("league") if isinstance(payload.get("league"), dict) else {}
    return (
        str(row.get("provider_name") or "").lower() == "api_football"
        and str(league.get("id") or "") == "1"
        and str(league.get("season") or "") == "2026"
    )


def _official_match_number(row: dict[str, Any], stage: str) -> int | None:
    number = _integer(row.get("match_number") or row.get("number"))
    if stage == "group":
        return number if number is not None and 1 <= number <= 72 else None
    if number is not None and number in KNOCKOUT_MATCH_NUMBER_RANGES.get(stage, ()):
        return number
    return None


def _official_group_id(row: dict[str, Any]) -> str | None:
    for value in (row.get("id"), row.get("canonical_match_id"), row.get("match_id")):
        raw = str(value or "")
        if raw.startswith("WC26-"):
            number = _integer(raw.removeprefix("WC26-"))
            if number is not None and 1 <= number <= 72:
                return f"WC26-{number:03d}"
    number = _official_match_number(row, "group")
    return f"WC26-{number:03d}" if number is not None else None


def _logical_row_id(row: dict[str, Any]) -> str:
    return str(
        row.get("id")
        or row.get("api_football_fixture_id")
        or row.get("provider_fixture_id")
        or row.get("canonical_match_id")
        or row.get("match_id")
        or "unknown"
    )


def _exclusion_reason(row: dict[str, Any], reason: str) -> str:
    return f"{_logical_row_id(row)}:{reason}"


def _row_rank(row: dict[str, Any], number: int | None) -> tuple[Any, ...]:
    return (
        _is_completed_match(row),
        _has_score(row),
        number is not None,
        _timestamp_floor(row.get("updated_at") or row.get("created_at")),
    )


def _knockout_logical_key(
    row: dict[str, Any],
    stage: str,
    number: int | None,
    kickoff: datetime,
    home_id: str,
    away_id: str,
) -> tuple[Any, ...]:
    if number is not None:
        return ("number", stage, number)
    provider_fixture_id = row.get("api_football_fixture_id") or row.get("provider_fixture_id")
    if provider_fixture_id is not None:
        return ("provider", str(provider_fixture_id))
    return ("teams", stage, kickoff.isoformat(), home_id, away_id)


def _stage_from_value(value: Any) -> str:
    raw = str(value or "").strip().lower()
    normalized = raw.replace("-", " ").replace("_", " ")
    if "round of 32" in raw or "round_of_32" in raw:
        return "round_of_32"
    if "round of 16" in raw or "round_of_16" in raw:
        return "round_of_16"
    if "quarter" in raw:
        return "quarterfinal"
    if "semi" in raw:
        return "semifinal"
    if "third" in raw or "3rd" in raw:
        return "third_place"
    if "final" in raw:
        return "final"
    if normalized in GROUP_STAGE_ALIASES or "group stage" in normalized:
        return "group"
    return "group" if "group" in normalized else raw


def _is_completed_match(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").strip().lower()
    return bool(row.get("completed")) or status in COMPLETED_MATCH_STATUSES or _has_score(row)


def _penalty_scores(row: dict[str, Any]) -> tuple[int | None, int | None]:
    home = _integer(row.get("home_penalty_score"))
    away = _integer(row.get("away_penalty_score"))
    if home is not None or away is not None:
        return home, away
    payload = _json_value(row.get("provider_payload")) or _json_value(row.get("raw")) or {}
    if isinstance(payload, dict):
        penalty = (
            payload.get("score", {}).get("penalty", {})
            if isinstance(payload.get("score"), dict)
            else {}
        )
        if isinstance(penalty, dict):
            return _integer(penalty.get("home")), _integer(penalty.get("away"))
    return None, None


def _completed_winner(match: MatchState) -> str:
    if match.home_score is None or match.away_score is None:
        raise ValueError(f"Completed fixture {match.id} is missing a final score")
    if match.home_score > match.away_score:
        return match.home_team_id
    if match.away_score > match.home_score:
        return match.away_team_id
    if (
        match.home_penalty_score is not None
        and match.away_penalty_score is not None
        and match.home_penalty_score != match.away_penalty_score
    ):
        return (
            match.home_team_id
            if match.home_penalty_score > match.away_penalty_score
            else match.away_team_id
        )
    raise ValueError(
        f"Completed knockout fixture {match.id} is tied without penalty winner data"
    )


def _cumulative_weights(weights: list[float]) -> list[float]:
    cumulative = 0.0
    values = []
    for weight in weights:
        cumulative += weight
        values.append(cumulative)
    return values


def _score_grid(prediction: dict[str, Any]) -> list[tuple[int, int, float]]:
    payload = _json_value(prediction.get("score_probabilities"))
    if isinstance(payload, list) and payload:
        scores = [
            (
                int(row["home_goals"]),
                int(row["away_goals"]),
                _number(row["probability"]),
            )
            for row in payload
        ]
        if sum(score[2] for score in scores) > 0:
            return scores

    home_xg = max(0.05, _number(prediction.get("home_xg"), 1.35))
    away_xg = max(0.05, _number(prediction.get("away_xg"), 1.15))
    scores = []
    for home_goals in range(7):
        for away_goals in range(7):
            probability = (
                math.exp(-home_xg)
                * home_xg**home_goals
                / math.factorial(home_goals)
                * math.exp(-away_xg)
                * away_xg**away_goals
                / math.factorial(away_goals)
            )
            scores.append((home_goals, away_goals, probability))
    return scores


def sample_group_score(
    sampler: tuple[list[tuple[int, int]], list[float]],
    rng: random.Random,
) -> tuple[int, int]:
    scores, cumulative = sampler
    index = bisect.bisect_left(cumulative, rng.random() * cumulative[-1])
    return scores[min(index, len(scores) - 1)]


def compile_score_sampler(
    prediction: dict[str, Any],
) -> tuple[list[tuple[int, int]], list[float]]:
    grid = _score_grid(prediction)
    return (
        [(home_goals, away_goals) for home_goals, away_goals, _ in grid],
        _cumulative_weights([probability for _, _, probability in grid]),
    )


def rank_group(
    team_ids: list[str],
    results: list[tuple[str, str, int, int]],
) -> list[dict[str, Any]]:
    rows = {
        team_id: {
            "team_id": team_id,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
        }
        for team_id in team_ids
    }
    for home_id, away_id, home_goals, away_goals in results:
        home = rows[home_id]
        away = rows[away_id]
        home["goals_for"] += home_goals
        home["goals_against"] += away_goals
        away["goals_for"] += away_goals
        away["goals_against"] += home_goals
        if home_goals > away_goals:
            home["points"] += 3
        elif away_goals > home_goals:
            away["points"] += 3
        else:
            home["points"] += 1
            away["points"] += 1
    for row in rows.values():
        row["goal_difference"] = row["goals_for"] - row["goals_against"]

    # Requested FIFA order, followed by canonical team ID as a stable fallback.
    return sorted(
        rows.values(),
        key=lambda row: (
            -row["points"],
            -row["goal_difference"],
            -row["goals_for"],
            row["team_id"],
        ),
    )


def _qualification_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        -row["points"],
        -row["goal_difference"],
        -row["goals_for"],
        row["team_id"],
    )


def build_round_of_32(
    group_tables: dict[str, list[dict[str, Any]]],
    team_groups: dict[str, str],
) -> list[tuple[str, str]]:
    winners = [group_tables[group][0] for group in "ABCDEFGHIJKL"]
    runners = [group_tables[group][1] for group in "ABCDEFGHIJKL"]
    best_thirds = sorted(
        (group_tables[group][2] for group in "ABCDEFGHIJKL"),
        key=_qualification_key,
    )[:8]

    # The 48-team format needs eight third-place qualifiers in addition to the
    # top two from each group. Seed 12 winners plus the four best runners
    # against the remaining runners and best thirds, avoiding group rematches.
    ranked_runners = sorted(runners, key=_qualification_key)
    seeded = winners + ranked_runners[:4]
    unseeded = ranked_runners[4:] + best_thirds
    pairings = []
    for seeded_team in seeded:
        opponent_index = next(
            (
                index
                for index, opponent in enumerate(unseeded)
                if team_groups[opponent["team_id"]]
                != team_groups[seeded_team["team_id"]]
            ),
            0,
        )
        opponent = unseeded.pop(opponent_index)
        pairings.append((seeded_team["team_id"], opponent["team_id"]))
    return pairings


KnockoutPrediction = Callable[[str, str], dict[str, Any]]


def build_knockout_prediction_provider(
    team_ratings: dict[str, dict[str, Any]],
    shot_volume_ratings: dict[str, float] | None = None,
) -> KnockoutPrediction:
    shot_volume_ratings = shot_volume_ratings or {}
    team_names = {team.id: team.name for team in load_teams()}
    cache: dict[tuple[str, str], dict[str, Any]] = {}

    def prediction(home_id: str, away_id: str) -> dict[str, Any]:
        key = (home_id, away_id)
        if key not in cache:
            cache[key] = calculate_prediction(
                team_ratings[home_id],
                team_ratings[away_id],
                home_team_name=team_names[home_id],
                away_team_name=team_names[away_id],
                home_shot_volume_rating=shot_volume_ratings.get(home_id),
                away_shot_volume_rating=shot_volume_ratings.get(away_id),
            )
        return cache[key]

    return prediction


def knockout_winner(
    home_id: str,
    away_id: str,
    prediction: dict[str, Any],
    rng: random.Random,
) -> str:
    home_regulation = _number(prediction.get("home_win_probability"))
    draw_probability = _number(prediction.get("draw_probability"))
    away_regulation = _number(prediction.get("away_win_probability"))
    total = home_regulation + draw_probability + away_regulation
    if total <= 0:
        raise ValueError(f"Invalid knockout probabilities for {home_id}-{away_id}")
    home_regulation /= total
    draw_probability /= total
    away_regulation /= total
    decisive_total = home_regulation + away_regulation
    decisive_home = (
        home_regulation / decisive_total if decisive_total > 0 else 0.5
    )
    draw_threshold = home_regulation + draw_probability
    roll = rng.random()
    if roll < home_regulation:
        return home_id
    if roll >= draw_threshold:
        return away_id

    # On a regulation draw, 40% are resolved in extra time using the underlying
    # strength edge. The remaining 60% go to penalties, where that edge is
    # compressed toward 50%.
    if rng.random() < 0.40:
        return home_id if rng.random() < decisive_home else away_id
    penalty_home = 0.5 + 0.35 * (decisive_home - 0.5)
    return home_id if rng.random() < penalty_home else away_id


def simulate_tournaments(
    predictions: dict[str, dict[str, Any]],
    num_simulations: int,
    seed: int,
    knockout_prediction: KnockoutPrediction,
    match_states: list[MatchState] | None = None,
) -> list[dict[str, Any]]:
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")
    teams = load_teams()
    fixtures = build_fixtures(teams)
    validate_tournament(teams, fixtures)
    group_fixtures = [fixture for fixture in fixtures if fixture.stage == "group"]
    match_states = match_states or []
    completed_groups = {
        match.id: match
        for match in match_states
        if match.stage == "group" and match.completed
    }
    missing = [
        fixture
        for fixture in group_fixtures
        if fixture.id not in completed_groups and fixture.id not in predictions
    ]
    if missing:
        teams_by_id = {team.id: team.name for team in teams}
        detected = sorted(completed_groups)
        raise ValueError(
            f"Latest prediction run is missing {len(missing)} group fixtures: "
            + ", ".join(
                (
                    f"{fixture.id} "
                    f"({teams_by_id[fixture.home_team_id]} vs "
                    f"{teams_by_id[fixture.away_team_id]})"
                )
                for fixture in missing
            )
            + f". Official completed group fixtures detected={len(detected)}/72"
            + (
                f" examples={', '.join(detected[:8])}"
                if detected
                else " examples=none"
            )
            + ". Check simulation logs for excluded group-row reasons."
        )

    team_groups = {team.id: team.group for team in teams}
    score_samplers = {
        fixture.id: compile_score_sampler(predictions[fixture.id])
        for fixture in group_fixtures
        if fixture.id not in completed_groups
    }
    known_knockouts = {
        stage: sorted(
            (match for match in match_states if match.stage == stage),
            key=lambda match: (
                match.match_number is None,
                match.match_number or 999,
                match.kickoff or datetime.max.replace(tzinfo=timezone.utc),
                match.id,
            ),
        )
        for stage in KNOCKOUT_STAGES
    }
    counts = {stage: Counter() for stage in STAGES}
    rng = random.Random(seed)
    simulation_logger = logging.getLogger("run_simulations")

    def resolve_knockout_stage(
        stage: str,
        pairings: list[tuple[str, str]],
    ) -> list[str]:
        winners = []
        known_by_pair = {
            (match.home_team_id, match.away_team_id): match
            for match in known_knockouts[stage]
        }
        for home_id, away_id in pairings:
            known = known_by_pair.get((home_id, away_id))
            if known and known.completed:
                winners.append(_completed_winner(known))
                continue
            prediction_id = known.id if known else None
            if known and prediction_id not in predictions:
                raise ValueError(
                    "Latest prediction run is missing future knockout fixture "
                    f"{known.id} ({known.home_team_id} vs {known.away_team_id})"
                )
            prediction = (
                predictions[prediction_id]
                if prediction_id is not None and prediction_id in predictions
                else knockout_prediction(home_id, away_id)
            )
            winners.append(knockout_winner(home_id, away_id, prediction, rng))
        return winners

    def bracket_pairings(
        stage: str,
        child_winners: list[str],
    ) -> list[tuple[str, str]]:
        """Fill official round slots from fixed fixtures and child-slot edges."""
        known = known_knockouts[stage]
        expected_matches = KNOCKOUT_STAGE_LIMITS[stage]
        completed_ids = [match.id for match in known if match.completed]
        upcoming_ids = [match.id for match in known if not match.completed]
        if len(child_winners) % 2:
            raise ValueError(
                "Invalid knockout bracket state: "
                f"stage={stage} current_team_count={len(child_winners)} "
                f"current_teams={child_winners} completed_fixtures={completed_ids} "
                f"upcoming_fixtures={upcoming_ids}; dynamic child-winner list is odd"
            )
        if len(child_winners) != expected_matches * 2:
            raise ValueError(
                "Invalid knockout bracket state: "
                f"stage={stage} current_team_count={len(child_winners)} "
                f"expected_team_count={expected_matches * 2} "
                f"current_teams={child_winners} completed_fixtures={completed_ids} "
                f"upcoming_fixtures={upcoming_ids}"
            )

        # Slot n consumes child winner slots 2n and 2n+1. This is the official
        # advancement edge; list adjacency is never inferred from a partial
        # collection of current-stage fixtures.
        dynamic = [
            (child_winners[slot * 2], child_winners[slot * 2 + 1])
            for slot in range(expected_matches)
        ]
        if len(known) == expected_matches:
            return [(match.home_team_id, match.away_team_id) for match in known]

        first_number = KNOCKOUT_MATCH_NUMBER_RANGES[stage].start
        fixed_by_slot: dict[int, MatchState] = {}
        for match in known:
            slot = (
                match.match_number - first_number
                if match.match_number is not None
                and match.match_number in KNOCKOUT_MATCH_NUMBER_RANGES[stage]
                else None
            )
            if slot is None:
                matching_slots = [
                    index for index, pairing in enumerate(dynamic)
                    if {match.home_team_id, match.away_team_id} == set(pairing)
                ]
                if len(matching_slots) != 1:
                    raise ValueError(
                        "Invalid knockout bracket state: cannot place known fixture "
                        f"stage={stage} fixture_id={match.id} "
                        f"teams={[match.home_team_id, match.away_team_id]} "
                        f"candidate_slots={matching_slots} current_teams={child_winners}"
                    )
                slot = matching_slots[0]
            if slot in fixed_by_slot:
                raise ValueError(
                    "Invalid knockout bracket state: duplicate fixed slot "
                    f"stage={stage} slot={slot} fixture_ids="
                    f"{[fixed_by_slot[slot].id, match.id]}"
                )
            expected_pair = dynamic[slot]
            if {match.home_team_id, match.away_team_id} != set(expected_pair):
                raise ValueError(
                    "Invalid knockout bracket state: fixed fixture does not match "
                    f"official advancement edge stage={stage} slot={slot} "
                    f"fixture_id={match.id} fixture_teams="
                    f"{[match.home_team_id, match.away_team_id]} "
                    f"child_winners={list(expected_pair)}"
                )
            fixed_by_slot[slot] = match

        return [
            (fixed_by_slot[slot].home_team_id, fixed_by_slot[slot].away_team_id)
            if slot in fixed_by_slot else dynamic[slot]
            for slot in range(expected_matches)
        ]

    for simulation_index in range(num_simulations):
        results_by_group: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
        for fixture in group_fixtures:
            completed = completed_groups.get(fixture.id)
            if completed is not None:
                home_goals = completed.home_score
                away_goals = completed.away_score
                if home_goals is None or away_goals is None:
                    raise ValueError(f"Completed group fixture {fixture.id} is missing a score")
            else:
                home_goals, away_goals = sample_group_score(
                    score_samplers[fixture.id],
                    rng,
                )
            results_by_group[fixture.group].append(
                (
                    fixture.home_team_id,
                    fixture.away_team_id,
                    home_goals,
                    away_goals,
                )
            )

        group_tables = {
            group: rank_group(
                [team.id for team in teams if team.group == group],
                results_by_group[group],
            )
            for group in "ABCDEFGHIJKL"
        }
        round_of_32_matches = known_knockouts["round_of_32"]
        pairings = (
            [(match.home_team_id, match.away_team_id) for match in round_of_32_matches]
            if round_of_32_matches
            else build_round_of_32(group_tables, team_groups)
        )
        round_of_32_teams = {team_id for pairing in pairings for team_id in pairing}
        counts["round_of_32"].update(round_of_32_teams)
        counts["group_stage_exit"].update(
            team.id for team in teams if team.id not in round_of_32_teams
        )

        current = resolve_knockout_stage("round_of_32", pairings)
        for stage in ("round_of_16", "quarterfinal", "semifinal", "final"):
            stage_matches = known_knockouts[stage]
            if simulation_index == 0:
                simulation_logger.info(
                    "[simulation] Bracket stage=%s current_team_count=%d "
                    "current_teams=%s completed_fixtures=%s upcoming_fixtures=%s",
                    stage,
                    len(current),
                    current,
                    [match.id for match in stage_matches if match.completed],
                    [match.id for match in stage_matches if not match.completed],
                )
            pairings = bracket_pairings(stage, current)
            stage_teams = {team_id for pairing in pairings for team_id in pairing}
            counts[stage].update(stage_teams)
            current = resolve_knockout_stage(stage, pairings)
        champion = current[0]
        counts["champion"].update([champion])

    return [
        {
            "team_id": team.id,
            **{
                f"{stage}_probability": counts[stage][team.id] / num_simulations
                for stage in STAGES
            },
        }
        for team in teams
    ]


class SimulationRepository:
    def __init__(self, engine: Engine, logger: logging.Logger | None = None) -> None:
        self.engine = engine
        self.logger = logger or logging.getLogger(__name__)
        self.schema = None if engine.dialect.name == "sqlite" else "public"
        self.metadata = MetaData()
        self.tables: dict[str, Table] = {}

    def _table(self, name: str) -> Table:
        if name not in self.tables:
            self.tables[name] = Table(
                name,
                self.metadata,
                schema=self.schema,
                autoload_with=self.engine,
            )
        return self.tables[name]

    def assert_schema(self) -> None:
        inspector = inspect(self.engine)
        required = {
            "matches",
            "model_runs",
            "predictions",
            "simulation_runs",
            "team_simulation_results",
        }
        existing = set(inspector.get_table_names(schema=self.schema))
        missing = required - existing
        if missing:
            raise RuntimeError(
                f"Simulation pipeline tables are missing: {sorted(missing)}. Apply "
                "supabase/migrations/202606100005_tournament_simulation.sql first."
            )

    def assert_snapshot_schema(self) -> None:
        columns = {
            column["name"]
            for column in inspect(self.engine).get_columns(
                "simulation_runs", schema=self.schema
            )
        }
        required = {
            "snapshot_key",
            "cutoff_at",
            "reconstruction_mode",
            "simulation_config_version",
            "status",
            "provenance",
        }
        if missing := required - columns:
            raise RuntimeError(
                "Historical snapshot columns are missing: "
                f"{sorted(missing)}. Apply "
                "supabase/migrations/202607250001_historical_simulation_snapshots.sql."
            )

    def load_schedule_rows(self) -> list[dict[str, Any]]:
        matches = self._table("matches")
        with self.engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(select(matches)).mappings()]
        official = []
        for row in rows:
            stage = _stage_from_value(row.get("stage") or row.get("tournament_stage"))
            kickoff = _parse_timestamp(row.get("kickoff") or row.get("match_date"))
            if kickoff is None:
                continue
            if stage == "group":
                if not (GROUP_WINDOW_START <= kickoff < GROUP_WINDOW_END):
                    continue
                if _official_group_id(row) is None and not _is_world_cup_2026_provider_row(row):
                    continue
            elif stage in KNOCKOUT_STAGES:
                if not (KNOCKOUT_WINDOW_START <= kickoff < KNOCKOUT_WINDOW_END):
                    continue
                if (
                    _official_match_number(row, stage) is None
                    and not _is_world_cup_2026_provider_row(row)
                ):
                    continue
            else:
                continue
            official.append(row)
        return official

    def load_match_states(
        self,
        database_team_ids: dict[str, Any],
        cutoff_at: datetime | None = None,
        snapshot_stage: str | None = None,
    ) -> list[MatchState]:
        matches = self._table("matches")
        database_to_canonical_team = {
            str(database_id): team_id
            for team_id, database_id in database_team_ids.items()
            if database_id is not None
        }
        fixtures = build_fixtures()
        fixture_ids = {fixture.id for fixture in fixtures}
        fixtures_by_number = {fixture.number: fixture for fixture in fixtures}
        fixtures_by_key = {
            (
                fixture.kickoff,
                fixture.home_team_id,
                fixture.away_team_id,
            ): fixture
            for fixture in fixtures
        }
        with self.engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(select(matches)).mappings()]

        states = self._official_group_states(
            rows,
            database_to_canonical_team,
            fixtures_by_number,
            fixtures_by_key,
            fixture_ids,
            cutoff_at,
        )
        if snapshot_stage != "group":
            stage_limit = (
                KNOCKOUT_STAGES.index(snapshot_stage)
                if snapshot_stage in KNOCKOUT_STAGES
                else len(KNOCKOUT_STAGES) - 1
            )
            states.extend(
                match
                for match in self._official_knockout_states(
                    rows, database_to_canonical_team, cutoff_at
                )
                if KNOCKOUT_STAGES.index(match.stage) <= stage_limit
            )
        completed_groups = sum(
            1 for match in states if match.stage == "group" and match.completed
        )
        completed_knockouts = sum(
            1 for match in states if match.stage in KNOCKOUT_STAGES and match.completed
        )
        upcoming_knockouts = sum(
            1 for match in states if match.stage in KNOCKOUT_STAGES and not match.completed
        )
        self.logger.info("[simulation] Raw match rows loaded=%d", len(rows))
        self.logger.info(
            "[simulation] Filtered official tournament state: "
            "group_completed=%d knockout_completed=%d knockout_upcoming=%d",
            completed_groups,
            completed_knockouts,
            upcoming_knockouts,
        )
        return states

    def _official_group_states(
        self,
        rows: list[dict[str, Any]],
        database_to_canonical_team: dict[str, str],
        fixtures_by_number: dict[int, Any],
        fixtures_by_key: dict[tuple[datetime, str, str], Any],
        fixture_ids: set[str],
        cutoff_at: datetime | None,
    ) -> list[MatchState]:
        selected: dict[str, tuple[dict[str, Any], Any]] = {}
        selected_numbers: dict[str, int | None] = {}
        excluded: list[str] = []

        for row in rows:
            stage = _stage_from_value(row.get("stage") or row.get("tournament_stage"))
            if stage != "group":
                excluded.append(_exclusion_reason(row, "non_group_stage"))
                continue
            home_id = database_to_canonical_team.get(str(row.get("home_team_id")))
            away_id = database_to_canonical_team.get(str(row.get("away_team_id")))
            if home_id is None or away_id is None:
                excluded.append(_exclusion_reason(row, "unmapped_team"))
                continue
            kickoff = _parse_timestamp(row.get("kickoff") or row.get("match_date"))
            if kickoff is None or not (GROUP_WINDOW_START <= kickoff < GROUP_WINDOW_END):
                excluded.append(_exclusion_reason(row, "outside_group_window"))
                continue
            group_id = _official_group_id(row)
            number = _official_match_number(row, "group")
            fixture = None
            if group_id is not None and group_id in fixture_ids:
                fixture = next(item for item in build_fixtures() if item.id == group_id)
            if fixture is None and number is not None:
                fixture = fixtures_by_number.get(number)
            if fixture is None:
                fixture = fixtures_by_key.get((kickoff, home_id, away_id))
            if fixture is None:
                excluded.append(_exclusion_reason(row, "not_official_group_fixture"))
                continue
            if (
                fixture.home_team_id != home_id
                or fixture.away_team_id != away_id
            ):
                excluded.append(_exclusion_reason(row, "group_team_mismatch"))
                continue
            existing = selected.get(fixture.id)
            if existing is None:
                selected[fixture.id] = (row, fixture)
                selected_numbers[fixture.id] = number
                continue
            if _row_rank(row, number) > _row_rank(existing[0], selected_numbers[fixture.id]):
                excluded.append(_exclusion_reason(existing[0], "duplicate_lower_rank"))
                selected[fixture.id] = (row, fixture)
                selected_numbers[fixture.id] = number
            else:
                excluded.append(_exclusion_reason(row, "duplicate_lower_rank"))

        states = []
        for row, fixture in selected.values():
            states.append(
                self._match_state_from_row(
                    row,
                    fixture.id,
                    "group",
                    fixture.home_team_id,
                    fixture.away_team_id,
                    fixture.number,
                    fixture.kickoff,
                    cutoff_at,
                )
            )
        self._log_exclusions("group", excluded)
        return sorted(states, key=lambda match: match.match_number or 0)

    def _official_knockout_states(
        self,
        rows: list[dict[str, Any]],
        database_to_canonical_team: dict[str, str],
        cutoff_at: datetime | None,
    ) -> list[MatchState]:
        selected: dict[tuple[Any, ...], dict[str, Any]] = {}
        selected_numbers: dict[tuple[Any, ...], int | None] = {}
        selected_kickoffs: dict[tuple[Any, ...], datetime] = {}
        selected_home_ids: dict[tuple[Any, ...], str] = {}
        selected_away_ids: dict[tuple[Any, ...], str] = {}
        excluded: list[str] = []

        for index, row in enumerate(rows, start=1):
            stage = _stage_from_value(row.get("stage") or row.get("tournament_stage"))
            if stage not in KNOCKOUT_STAGE_LIMITS:
                if stage != "group":
                    excluded.append(_exclusion_reason(row, "non_knockout_stage"))
                continue
            home_id = database_to_canonical_team.get(str(row.get("home_team_id")))
            away_id = database_to_canonical_team.get(str(row.get("away_team_id")))
            if home_id is None or away_id is None:
                excluded.append(_exclusion_reason(row, "unmapped_team"))
                continue
            kickoff = _parse_timestamp(row.get("kickoff") or row.get("match_date"))
            if kickoff is None or not (KNOCKOUT_WINDOW_START <= kickoff < KNOCKOUT_WINDOW_END):
                excluded.append(_exclusion_reason(row, "outside_knockout_window"))
                continue
            number = _official_match_number(row, stage)
            if number is None and not _is_world_cup_2026_provider_row(row):
                excluded.append(_exclusion_reason(row, "unofficial_provider_row"))
                continue
            key = _knockout_logical_key(row, stage, number, kickoff, home_id, away_id)
            existing = selected.get(key)
            if existing is None:
                selected[key] = {**row, "_knockout_stage": stage, "_knockout_index": index}
                selected_numbers[key] = number
                selected_kickoffs[key] = kickoff
                selected_home_ids[key] = home_id
                selected_away_ids[key] = away_id
                continue
            if _row_rank(row, number) > _row_rank(existing, selected_numbers[key]):
                excluded.append(_exclusion_reason(existing, "duplicate_lower_rank"))
                selected[key] = {**row, "_knockout_stage": stage, "_knockout_index": index}
                selected_numbers[key] = number
                selected_kickoffs[key] = kickoff
                selected_home_ids[key] = home_id
                selected_away_ids[key] = away_id
            else:
                excluded.append(_exclusion_reason(row, "duplicate_lower_rank"))

        states = []
        for stage in KNOCKOUT_STAGE_LIMITS:
            stage_items = [
                (
                    row,
                    selected_numbers[key],
                    selected_kickoffs[key],
                    selected_home_ids[key],
                    selected_away_ids[key],
                )
                for key, row in selected.items()
                if row["_knockout_stage"] == stage
            ]
            ordered = sorted(
                stage_items,
                key=lambda item: (
                    item[1] is None,
                    item[1] if item[1] is not None else 999,
                    item[2],
                ),
            )
            for row, number, kickoff, home_id, away_id in ordered[
                : KNOCKOUT_STAGE_LIMITS[stage]
            ]:
                match_id = str(
                    row.get("id")
                    or row.get("api_football_fixture_id")
                    or row.get("provider_fixture_id")
                    or row.get("canonical_match_id")
                    or f"provider-knockout-{row['_knockout_index']}"
                )
                states.append(
                    self._match_state_from_row(
                        row,
                        match_id,
                        stage,
                        home_id,
                        away_id,
                        number,
                        kickoff,
                        cutoff_at,
                    )
                )
        self._log_exclusions("knockout", excluded)
        return states

    @staticmethod
    def _match_state_from_row(
        row: dict[str, Any],
        match_id: str,
        stage: str,
        home_id: str,
        away_id: str,
        match_number: int | None,
        kickoff: datetime | None,
        cutoff_at: datetime | None = None,
    ) -> MatchState:
        home_penalty_score, away_penalty_score = _penalty_scores(row)
        return MatchState(
            id=match_id,
            stage=stage,
            home_team_id=home_id,
            away_team_id=away_id,
            completed=(
                _is_completed_match(row)
                and (
                    cutoff_at is None
                    or (kickoff is not None and kickoff < cutoff_at)
                )
            ),
            home_score=_integer(row.get("home_score")),
            away_score=_integer(row.get("away_score")),
            home_penalty_score=home_penalty_score,
            away_penalty_score=away_penalty_score,
            match_number=match_number,
            kickoff=kickoff,
        )

    def _log_exclusions(self, label: str, excluded: list[str]) -> None:
        if not excluded:
            return
        self.logger.info(
            "[simulation] Excluded %d unofficial/duplicate %s rows; examples=%s",
            len(excluded),
            label,
            ", ".join(excluded[:8]),
        )

    def load_latest_predictions(
        self,
    ) -> tuple[Any, str, dict[str, dict[str, Any]]] | None:
        predictions = self._table("predictions")
        with self.engine.connect() as connection:
            latest = connection.execute(
                select(
                    predictions.c.model_run_id,
                    predictions.c.model_version,
                    predictions.c.prediction_timestamp,
                )
                .where(predictions.c.model_run_id.is_not(None))
                .order_by(predictions.c.prediction_timestamp.desc())
                .limit(1)
            ).mappings().one_or_none()
            if latest is None:
                return None
            rows = [
                dict(row)
                for row in connection.execute(
                    select(predictions).where(
                        predictions.c.model_run_id == latest["model_run_id"]
                    )
                ).mappings()
            ]
        canonical = {
            row["canonical_match_id"]: row
            for row in rows
            if row.get("canonical_match_id")
        }
        return (
            latest["model_run_id"],
            latest["model_version"] or "poisson-ratings-v1",
            canonical,
        )

    def load_historical_predictions(
        self,
        cutoff_at: datetime,
        match_states: list[MatchState],
        model_version: str = MODEL_VERSION,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Select the newest authentic prediction strictly before cutoff/kickoff."""
        predictions = self._table("predictions")
        model_runs = self._table("model_runs")
        with self.engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(select(predictions)).mappings()]
            run_rows = [
                dict(row)
                for row in connection.execute(select(model_runs)).mappings()
            ]
        run_by_id = {
            str(row["id"]): row for row in run_rows if row.get("id") is not None
        }

        fixture_kickoffs = {
            fixture.id: fixture.kickoff for fixture in build_fixtures()
        }
        state_by_identity: dict[str, MatchState] = {}
        for state in match_states:
            for identity in (state.id, state.match_number):
                if identity is not None:
                    state_by_identity[str(identity)] = state
            if state.match_number is not None:
                state_by_identity[f"WC26-{state.match_number:03d}"] = state

        eligible: dict[str, tuple[datetime, dict[str, Any]]] = {}
        rejected_after_cutoff: list[str] = []
        rejected_after_kickoff: list[str] = []
        for row in rows:
            run = run_by_id.get(str(row.get("model_run_id")))
            if (
                run is not None
                and "status" in run
                and str(run.get("status") or "completed") != "completed"
            ):
                continue
            effective_model_version = row.get("model_version") or (
                run.get("model_version") if run is not None else None
            )
            if str(effective_model_version or "") != model_version:
                continue
            if str(row.get("generation_mode") or "standard") == "historical_backfill":
                # Backfills are useful provenance, but they were not generated at
                # the historical instant. Snapshot fallback rebuilds their inputs
                # explicitly instead of pretending they were published earlier.
                continue
            generated_at = _parse_timestamp(
                row.get("prediction_timestamp") or row.get("created_at")
            )
            if generated_at is None:
                continue
            identities = [
                str(value)
                for value in (
                    row.get("canonical_match_id"),
                    row.get("match_id"),
                    row.get("provider_fixture_id"),
                )
                if value is not None
            ]
            canonical_id = next(
                (identity for identity in identities if identity in fixture_kickoffs),
                None,
            )
            state = next(
                (state_by_identity[identity] for identity in identities if identity in state_by_identity),
                None,
            )
            kickoff = (
                state.kickoff
                if state is not None and state.kickoff is not None
                else fixture_kickoffs.get(canonical_id)
                if canonical_id is not None
                else None
            )
            if generated_at >= cutoff_at:
                rejected_after_cutoff.extend(identities[:1])
                continue
            if kickoff is not None and generated_at >= kickoff:
                rejected_after_kickoff.extend(identities[:1])
                continue
            logical_id = canonical_id or (state.id if state is not None else None)
            if logical_id is None:
                continue
            current = eligible.get(logical_id)
            if current is None or generated_at > current[0]:
                eligible[logical_id] = (generated_at, row)

        selected = {match_id: row for match_id, (_timestamp, row) in eligible.items()}
        return selected, {
            "eligible_prediction_count": len(selected),
            "prediction_model_version": model_version,
            "latest_eligible_prediction_at": (
                max(timestamp for timestamp, _row in eligible.values()).isoformat()
                if eligible else None
            ),
            "rejected_after_cutoff": sorted(set(rejected_after_cutoff)),
            "rejected_at_or_after_kickoff": sorted(set(rejected_after_kickoff)),
            "model_run_ids": sorted(
                {
                    str(row["model_run_id"])
                    for row in selected.values()
                    if row.get("model_run_id") is not None
                }
            ),
        }

    def load_historical_prediction_provider(
        self,
        cutoff_at: datetime,
        database_team_ids: dict[str, Any],
    ) -> tuple[KnockoutPrediction, dict[str, Any]]:
        """Build leakage-safe matchup probabilities from raw rows before cutoff.

        The mutable current rating tables are intentionally not queried. When
        timestamped raw stats are unavailable (notably in small local fixtures),
        the production model's canonical rank prior is the deterministic fallback.
        """
        teams = load_teams()
        team_ratings = {
            team.id: {
                "team_id": team.id,
                "elo_rating": canonical_prior_elo(team.rank),
                "attack_rating": 50.0,
                "defense_rating": 50.0,
                "form_rating": 50.0,
                "matches_played": 0,
                "_rating_source": "canonical_rank_prior",
            }
            for team in teams
        }
        shot_volume: dict[str, float] = {}
        provenance: dict[str, Any] = {
            "historical_rating_source": "canonical_rank_prior",
            "historical_team_stat_count": 0,
            "historical_player_stat_count": 0,
            "historical_completed_match_count": 0,
            "maximum_source_timestamp": None,
        }
        existing = set(inspect(self.engine).get_table_names(schema=self.schema))
        required = {
            "teams",
            "matches",
            "team_match_stats",
            "player_match_stats",
        }
        if required.issubset(existing):
            from scripts.backfill_historical_knockout_predictions import (
                HistoricalBackfillRepository,
                build_historical_state,
            )

            historical_repository = HistoricalBackfillRepository(self.engine)
            teams_rows = historical_repository.rows("teams")
            state = build_historical_state(
                teams=teams_rows,
                matches=historical_repository.rows("matches"),
                team_stats=historical_repository.load_stats("team_match_stats"),
                player_stats=historical_repository.load_stats("player_match_stats"),
                cutoff=cutoff_at,
                target_match_id="__historical_snapshot__",
            )
            for canonical_id, database_id in database_team_ids.items():
                if database_id in state.team_ratings:
                    team_ratings[canonical_id] = {
                        **state.team_ratings[database_id],
                        "team_id": canonical_id,
                    }
                if database_id in state.shot_volume_ratings:
                    shot_volume[canonical_id] = state.shot_volume_ratings[database_id]
            provenance = {
                "historical_rating_source": "raw_stats_cutoff_rebuild",
                "historical_team_stat_count": state.team_stat_count,
                "historical_player_stat_count": state.player_stat_count,
                "historical_completed_match_count": state.completed_match_count,
                "maximum_source_timestamp": (
                    state.maximum_source_timestamp.isoformat()
                    if state.maximum_source_timestamp is not None else None
                ),
            }
        return build_knockout_prediction_provider(team_ratings, shot_volume), provenance

    def store_results(
        self,
        model_run_id: Any,
        model_version: str,
        num_simulations: int,
        seed: int,
        results: list[dict[str, Any]],
        *,
        snapshot_key: str | None = None,
        cutoff_at: datetime | None = None,
        reconstruction_mode: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Any:
        runs = self._table("simulation_runs")
        result_table = self._table("team_simulation_results")
        run_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        run_values = self._compatible_values(
            runs,
            {
                "id": run_id,
                "model_run_id": model_run_id,
                "model_version": model_version,
                "num_simulations": num_simulations,
                "iterations": num_simulations,
                "random_seed": seed,
                "snapshot_key": snapshot_key,
                "cutoff_at": cutoff_at.isoformat() if cutoff_at else None,
                "reconstruction_mode": reconstruction_mode,
                "simulation_config_version": (
                    SIMULATION_CONFIG_VERSION if snapshot_key else None
                ),
                "status": "completed",
                "provenance": provenance or {},
                "created_at": now,
            },
        )
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                connection.execute(
                    text("select pg_advisory_xact_lock(hashtext('tournament-simulation'))")
                )
            if snapshot_key and "snapshot_key" in runs.c:
                existing_ids = list(
                    connection.execute(
                        select(runs.c.id).where(
                            runs.c.snapshot_key == snapshot_key,
                            runs.c.model_version == model_version,
                            runs.c.simulation_config_version
                            == SIMULATION_CONFIG_VERSION,
                        )
                    ).scalars()
                )
                if existing_ids:
                    connection.execute(
                        result_table.delete().where(
                            result_table.c.simulation_run_id.in_(existing_ids)
                        )
                    )
                    connection.execute(runs.delete().where(runs.c.id.in_(existing_ids)))
            connection.execute(runs.insert().values(**run_values))
            connection.execute(
                result_table.insert(),
                [
                    self._compatible_values(
                        result_table,
                        {
                            "simulation_run_id": run_id,
                            **result,
                            "created_at": now,
                        },
                    )
                    for result in results
                ],
            )
        return run_id

    @staticmethod
    def _compatible_values(table: Table, values: dict[str, Any]) -> dict[str, Any]:
        compatible = {}
        for key, value in values.items():
            if key not in table.c:
                continue
            if isinstance(value, (dict, list)) and not isinstance(table.c[key].type, JSON):
                value = json.dumps(value)
            compatible[key] = value
        return compatible


def run_historical_snapshot(
    repository: SimulationRepository,
    prediction_repository: PredictionRepository,
    snapshot_key: str,
    *,
    num_simulations: int,
    seed: int,
    dry_run: bool,
    logger: logging.Logger,
) -> Any | None:
    snapshot = get_snapshot(snapshot_key)
    cutoff_at, cutoff_source = resolve_snapshot_cutoff(
        snapshot, repository.load_schedule_rows()
    )
    database_team_ids = prediction_repository.load_database_team_ids()
    match_states = repository.load_match_states(
        database_team_ids,
        cutoff_at=cutoff_at,
        snapshot_stage=snapshot.stage,
    )
    if snapshot.stage in KNOCKOUT_STAGE_LIMITS:
        current_stage = [
            match for match in match_states if match.stage == snapshot.stage
        ]
        expected = KNOCKOUT_STAGE_LIMITS[snapshot.stage]
        if len(current_stage) != expected:
            raise ValueError(
                f"Snapshot {snapshot.key} requires {expected} official "
                f"{snapshot.stage} fixtures with known participants; found "
                f"{len(current_stage)}. Refusing to infer them from final rows "
                "or an incomplete bracket."
            )

    predictions, prediction_audit = repository.load_historical_predictions(
        cutoff_at, match_states, MODEL_VERSION
    )
    historical_provider, rating_audit = repository.load_historical_prediction_provider(
        cutoff_at, database_team_ids
    )
    fixtures_by_id = {fixture.id: fixture for fixture in build_fixtures()}
    unresolved = [
        match for match in match_states
        if not match.completed
        and (
            match.stage == "group"
            or match.stage == snapshot.stage
        )
    ]
    if snapshot.stage == "group":
        present_group_ids = {match.id for match in unresolved if match.stage == "group"}
        unresolved.extend(
            MatchState(
                id=fixture.id,
                stage="group",
                home_team_id=fixture.home_team_id,
                away_team_id=fixture.away_team_id,
                completed=False,
                match_number=fixture.number,
                kickoff=fixture.kickoff,
            )
            for fixture in fixtures_by_id.values()
            if fixture.id not in present_group_ids
        )
    fallback_matches = []
    for match in unresolved:
        if match.id in predictions:
            continue
        predictions[match.id] = historical_provider(
            match.home_team_id, match.away_team_id
        )
        fallback_matches.append(match.id)

    future_dynamic_rounds = snapshot.stage != "final"
    reconstruction_mode = (
        "retrospective_reconstruction"
        if fallback_matches or future_dynamic_rounds
        else "historical_prediction_snapshot"
    )
    completed_groups = sum(
        1 for match in match_states if match.stage == "group" and match.completed
    )
    completed_knockouts = sum(
        1 for match in match_states
        if match.stage in KNOCKOUT_STAGES and match.completed
    )
    active_team_ids = (
        sorted(team.id for team in load_teams())
        if snapshot.stage == "group"
        else sorted(
            {
                team_id
                for match in match_states
                if match.stage == snapshot.stage
                for team_id in (match.home_team_id, match.away_team_id)
            }
        )
    )
    expected_active_count = SNAPSHOT_ACTIVE_TEAM_COUNTS[snapshot.key]
    if len(active_team_ids) != expected_active_count:
        raise ValueError(
            f"Snapshot {snapshot.key} requires {expected_active_count} active "
            f"teams; reconstructed {len(active_team_ids)}"
        )
    provenance = {
        "snapshot_key": snapshot.key,
        "snapshot_stage": snapshot.stage,
        "active_team_ids": active_team_ids,
        "cutoff_source": cutoff_source,
        "completed_group_matches": completed_groups,
        "completed_knockout_matches": completed_knockouts,
        "unresolved_stored_fixture_count": len(unresolved),
        "retrospective_prediction_match_ids": sorted(fallback_matches),
        "future_dynamic_matchups_reconstructed": future_dynamic_rounds,
        **prediction_audit,
        **rating_audit,
    }
    logger.info(
        "[simulation] SNAPSHOT key=%s cutoff=%s cutoff_source=%s model=%s "
        "mode=%s completed_group=%d completed_knockout=%d "
        "eligible_predictions=%d retrospective_predictions=%d",
        snapshot.key,
        cutoff_at.isoformat(),
        cutoff_source,
        MODEL_VERSION,
        reconstruction_mode,
        completed_groups,
        completed_knockouts,
        prediction_audit["eligible_prediction_count"],
        len(fallback_matches),
    )
    if dry_run:
        logger.info(
            "[simulation] DRY RUN: no simulation executed and no rows written; "
            "retrospective_match_ids=%s",
            ", ".join(sorted(fallback_matches)) or "none",
        )
        return None

    results = simulate_tournaments(
        predictions,
        num_simulations,
        seed,
        historical_provider,
        match_states,
    )
    return repository.store_results(
        None,
        MODEL_VERSION,
        num_simulations,
        seed,
        results,
        snapshot_key=snapshot.key,
        cutoff_at=cutoff_at,
        reconstruction_mode=reconstruction_mode,
        provenance=provenance,
    )


def _run_current_simulation(
    repository: SimulationRepository,
    prediction_repository: PredictionRepository,
    *,
    num_simulations: int,
    seed: int,
    logger: logging.Logger,
) -> Any | None:
    latest = repository.load_latest_predictions()
    if latest is None:
        logger.info("[simulation] SUCCESS: no prediction run available")
        return None
    model_run_id, model_version, predictions = latest
    database_team_ids = prediction_repository.load_database_team_ids()
    match_states = repository.load_match_states(database_team_ids)
    completed_groups = sum(
        1 for match in match_states if match.stage == "group" and match.completed
    )
    completed_knockouts = sum(
        1 for match in match_states if match.stage in KNOCKOUT_STAGES and match.completed
    )
    upcoming_knockouts = sum(
        1 for match in match_states if match.stage in KNOCKOUT_STAGES and not match.completed
    )
    logger.info(
        "[simulation] Loaded tournament state: completed_group=%d "
        "completed_knockout=%d upcoming_knockout=%d",
        completed_groups,
        completed_knockouts,
        upcoming_knockouts,
    )
    team_ratings = prediction_repository.load_current_team_ratings(
        database_team_ids
    )
    shot_volume_ratings = (
        prediction_repository.load_current_shot_volume_ratings(
            database_team_ids
        )
    )
    knockout_prediction = build_knockout_prediction_provider(
        team_ratings,
        shot_volume_ratings,
    )
    logger.info("[simulation] Running %d tournaments", num_simulations)
    results = simulate_tournaments(
        predictions,
        num_simulations,
        seed,
        knockout_prediction,
        match_states,
    )
    return repository.store_results(
        model_run_id,
        model_version,
        num_simulations,
        seed,
        results,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("run_simulations")
    if args.simulations < 1:
        logger.error("--simulations must be at least 1")
        return 2
    if args.dry_run and not (args.snapshot or args.all_snapshots):
        logger.error("--dry-run requires --snapshot or --all-snapshots")
        return 2

    database_url = load_environment().get("DATABASE_URL")
    if not database_url:
        logger.error("[simulation] FAILED: DATABASE_URL is required")
        return 2

    try:
        engine = create_database_engine(database_url)
    except Exception:
        logger.exception("[simulation] FAILED: could not initialize database")
        return 1

    try:
        logger.info("[simulation] START")
        repository = SimulationRepository(engine)
        repository.assert_schema()
        prediction_repository = PredictionRepository(engine)
        if args.snapshot or args.all_snapshots:
            repository.assert_snapshot_schema()
            keys = (
                [snapshot.key for snapshot in SNAPSHOTS]
                if args.all_snapshots
                else [args.snapshot]
            )
            for key in keys:
                run_id = run_historical_snapshot(
                    repository,
                    prediction_repository,
                    key,
                    num_simulations=args.simulations,
                    seed=args.seed,
                    dry_run=args.dry_run,
                    logger=logger,
                )
                if run_id is not None:
                    logger.info("[simulation] SNAPSHOT SUCCESS key=%s run=%s", key, run_id)
            logger.info("[simulation] SUCCESS snapshots=%s", ",".join(keys))
            return 0
        run_id = _run_current_simulation(
            repository,
            prediction_repository,
            num_simulations=args.simulations,
            seed=args.seed,
            logger=logger,
        )
        if run_id is not None:
            logger.info("[simulation] SUCCESS run=%s", run_id)
        return 0
    except Exception:
        logger.exception("[simulation] FAILED: unexpected simulation error")
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
