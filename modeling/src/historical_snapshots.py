"""Canonical historical tournament snapshot definitions and cutoff resolution."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class SnapshotDefinition:
    key: str
    label: str
    stage: str
    fallback_cutoff_at: datetime
    sort_order: int
    description: str


# Audited against the official FIFA schedule. Production resolution prefers the
# earliest matching kickoff stored in `matches`; these values are one centralized
# fallback for empty/local databases.
SNAPSHOTS = (
    SnapshotDefinition(
        "pre_tournament",
        "Before Tournament",
        "group",
        datetime(2026, 6, 11, 19, tzinfo=timezone.utc),
        1,
        "The forecast immediately before the opening group-stage match.",
    ),
    SnapshotDefinition(
        "pre_round_of_32",
        "Before Round of 32",
        "round_of_32",
        datetime(2026, 6, 28, 19, tzinfo=timezone.utc),
        2,
        "The forecast after the group stage and before the Round of 32 began.",
    ),
    SnapshotDefinition(
        "pre_round_of_16",
        "Before Round of 16",
        "round_of_16",
        datetime(2026, 7, 4, 17, tzinfo=timezone.utc),
        3,
        "The forecast after the Round of 32 and before the Round of 16 began.",
    ),
    SnapshotDefinition(
        "pre_quarterfinals",
        "Before Quarterfinals",
        "quarterfinal",
        datetime(2026, 7, 9, 20, tzinfo=timezone.utc),
        4,
        "The forecast after the Round of 16 and before the quarterfinals began.",
    ),
    SnapshotDefinition(
        "pre_semifinals",
        "Before Semifinals",
        "semifinal",
        datetime(2026, 7, 14, 19, tzinfo=timezone.utc),
        5,
        "The forecast after the quarterfinals and before the semifinals began.",
    ),
    SnapshotDefinition(
        "pre_final",
        "Before Final",
        "final",
        datetime(2026, 7, 19, 19, tzinfo=timezone.utc),
        6,
        "The forecast after the semifinals and before the final began.",
    ),
)
SNAPSHOTS_BY_KEY = {snapshot.key: snapshot for snapshot in SNAPSHOTS}
SNAPSHOT_ACTIVE_TEAM_COUNTS = {
    "pre_tournament": 48,
    "pre_round_of_32": 32,
    "pre_round_of_16": 16,
    "pre_quarterfinals": 8,
    "pre_semifinals": 4,
    "pre_final": 2,
}


def parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_stage(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    if "round of 32" in raw:
        return "round_of_32"
    if "round of 16" in raw:
        return "round_of_16"
    if "quarter" in raw:
        return "quarterfinal"
    if "semi" in raw:
        return "semifinal"
    if "third" in raw or "3rd" in raw:
        return "third_place"
    if "final" in raw:
        return "final"
    return "group" if "group" in raw or raw == "first round" else raw


def get_snapshot(key: str) -> SnapshotDefinition:
    try:
        return SNAPSHOTS_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown snapshot {key!r}; expected one of {', '.join(SNAPSHOTS_BY_KEY)}"
        ) from exc


def resolve_snapshot_cutoff(
    snapshot: SnapshotDefinition,
    schedule_rows: Iterable[dict[str, Any]],
) -> tuple[datetime, str]:
    """Resolve the exclusive UTC cutoff from stored schedule rows or fallback."""
    kickoffs = []
    tournament_start = datetime(2026, 6, 11, tzinfo=timezone.utc)
    tournament_end = datetime(2026, 7, 20, tzinfo=timezone.utc)
    for row in schedule_rows:
        stage = normalize_stage(row.get("stage") or row.get("tournament_stage"))
        if stage != snapshot.stage:
            continue
        kickoff = parse_utc(row.get("kickoff") or row.get("match_date"))
        if kickoff is not None and tournament_start <= kickoff < tournament_end:
            kickoffs.append(kickoff)
    if kickoffs:
        return min(kickoffs), "database_schedule"
    return snapshot.fallback_cutoff_at, "audited_fallback"


def snapshot_payload(
    snapshot: SnapshotDefinition,
    cutoff_at: datetime,
    *,
    cutoff_source: str | None = None,
    available: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "key": snapshot.key,
        "label": snapshot.label,
        "stage": snapshot.stage,
        "cutoff_at": cutoff_at.astimezone(timezone.utc).isoformat(),
        "sort_order": snapshot.sort_order,
        "description": snapshot.description,
    }
    if cutoff_source is not None:
        payload["cutoff_source"] = cutoff_source
    if available is not None:
        payload["available"] = available
    return payload
