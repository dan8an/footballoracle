from datetime import datetime, timezone

import pytest

from modeling.src.historical_snapshots import (
    SNAPSHOTS,
    get_snapshot,
    resolve_snapshot_cutoff,
)


def test_snapshot_definitions_are_stable_ordered_and_utc():
    assert [snapshot.key for snapshot in SNAPSHOTS] == [
        "pre_tournament",
        "pre_round_of_32",
        "pre_round_of_16",
        "pre_quarterfinals",
        "pre_semifinals",
        "pre_final",
    ]
    assert [snapshot.sort_order for snapshot in SNAPSHOTS] == list(range(1, 7))
    assert all(snapshot.fallback_cutoff_at.tzinfo is timezone.utc for snapshot in SNAPSHOTS)


def test_cutoff_prefers_earliest_database_stage_kickoff():
    snapshot = get_snapshot("pre_quarterfinals")
    cutoff, source = resolve_snapshot_cutoff(
        snapshot,
        [
            {"stage": "Quarter-finals", "kickoff": "2026-07-10T20:00:00Z"},
            {"stage": "quarterfinal", "kickoff": "2026-07-09T20:00:00Z"},
        ],
    )
    assert cutoff == datetime(2026, 7, 9, 20, tzinfo=timezone.utc)
    assert source == "database_schedule"


def test_unknown_snapshot_fails_without_interpolating_sql():
    with pytest.raises(ValueError, match="Unknown snapshot"):
        get_snapshot("pre_final' or 1=1")
