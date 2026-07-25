# Historical simulation snapshots

## Why this exists

The legacy simulation command reads the current tournament database state. Once
the tournament is complete, every result and knockout participant is known, so
a new current-state run correctly collapses to 100% for the champion and 0% for
eliminated teams. That output is useful as a completed-state check, but not as a
forecast.

Historical snapshots are separate persisted runs conditioned on an exclusive
UTC cutoff. They continue to use the production model
`elo-context-v4.2.1`; this feature does not promote or alter a model.

## Canonical snapshots and cutoffs

`modeling/src/historical_snapshots.py` is the canonical configuration used by
generation, API serialization, and tests.

| Key | Label | Cutoff rule |
| --- | --- | --- |
| `pre_tournament` | Before Tournament | Earliest group-stage kickoff |
| `pre_round_of_32` | Before Round of 32 | Earliest Round-of-32 kickoff |
| `pre_round_of_16` | Before Round of 16 | Earliest Round-of-16 kickoff |
| `pre_quarterfinals` | Before Quarterfinals | Earliest quarterfinal kickoff |
| `pre_semifinals` | Before Semifinals | Earliest semifinal kickoff |
| `pre_final` | Before Final | Final kickoff |

The resolver takes the earliest official 2026 tournament kickoff for the stage
from `matches`. If no stage row is present, the same module contains one
audited official-schedule fallback per snapshot.
All datetimes are timezone-aware UTC. The cutoff is exclusive.

## Historical-integrity rules

- A stored final score is fixed only when its kickoff is before the cutoff.
- A match at or after the cutoff is unresolved, even if its row now contains a
  final score.
- Knockout rows are loaded only through the selected snapshot stage. Final
  participant assignments therefore cannot enter a Round-of-16 snapshot.
- For post-group snapshots, all official fixtures in the selected stage must be
  present with known participants. Generation fails instead of inferring an
  incomplete historical bracket.
- An authentic stored prediction is eligible only when its model version is
  `elo-context-v4.2.1` and its generation timestamp is strictly before both the
  snapshot cutoff and the match kickoff. The newest eligible row wins.
- A later `historical_backfill` row is not represented as an authentic
  pre-cutoff publication. Missing stored predictions and hypothetical future
  pairings are recalculated from cutoff-filtered inputs.
- Dynamic later-round participants come only from winners generated along the
  bracket's child-slot edges. Populated future match rows are excluded.
- Pre-cutoff penalty and extra-time winners use the existing completed-knockout
  resolution logic.

## Reconstruction modes

- `historical_exact`: all point-in-time inputs are immutable historical
  snapshots. The current repository cannot establish this for these runs, so
  the generator never assigns it automatically.
- `historical_prediction_snapshot`: every remaining matchup uses an authentic
  stored prediction generated before the cutoff, with no dynamic reconstructed
  round required. This can apply to a fully covered pre-final snapshot.
- `retrospective_reconstruction`: at least one matchup is recalculated or a
  future dynamic round requires recalculation.

Current `team_ratings` and `team_chance_quality_ratings` rows are mutable
current-state values and are never read by historical generation. Instead, the
existing historical backfill logic rebuilds ratings from matches and stat rows
whose event and availability timestamps precede the cutoff. If a local/test
database lacks those raw tables, the production model's existing conservative
FIFA-rank prior is the deterministic, leakage-safe fallback. The run provenance
records which source was used, counts of source rows, the maximum source
timestamp, stored-prediction coverage, and every fixture that required
retrospective prediction.

This is the strongest leakage-safe reconstruction supported by the stored data,
but it is not claimed as exact historical replay: the rating tables were not
persisted as immutable snapshots with every historical production run, and
older prediction history may have been removed before the append-only July
migration.

## Storage and retry behavior

Migration
`supabase/migrations/202607250001_historical_simulation_snapshots.sql` extends
`simulation_runs` with:

- `snapshot_key`
- `cutoff_at`
- `reconstruction_mode`
- `simulation_config_version`
- `status`
- `provenance`

Existing rows keep `snapshot_key = null`. Snapshot identity is
`snapshot_key + model_version + simulation_config_version`. A retry replaces
the prior run and all team rows inside one database transaction under the
existing simulation advisory lock. A failed insert rolls back both run and team
rows.

## Commands

Apply the migration, then audit before writing:

```bash
python scripts/run_simulations.py --snapshot pre_tournament --dry-run
python scripts/run_simulations.py --snapshot pre_round_of_16 --dry-run
```

Generate one or all persisted snapshots:

```bash
python scripts/run_simulations.py --snapshot pre_tournament
python scripts/run_simulations.py --snapshot pre_round_of_32
python scripts/run_simulations.py --all-snapshots
```

Use `--simulations` and `--seed` for a bounded verification run:

```bash
python scripts/run_simulations.py \
  --snapshot pre_round_of_16 \
  --simulations 1000 \
  --seed 2026
```

Dry-run prints the resolved cutoff and source, completed-match counts, model
version, reconstruction mode, eligible prediction count, and retrospective
fixture IDs. It never simulates or writes.

## API and frontend

- `GET /v1/simulations/snapshots` lists all definitions in chronological order
  and marks stored availability.
- `GET /v1/simulations?snapshot=pre_round_of_16` returns only that snapshot's
  newest successful run, metadata, provenance, and team probabilities.
- Unknown keys return 422; valid keys without a run return a structured 404.
- `GET /v1/simulations/latest` retains its legacy meaning: the newest
  current-state run (`snapshot_key is null`). Historical backfills cannot
  displace it.

The web explorer uses the snapshot endpoints and stores the selection in
`/simulations?snapshot=...`. It does not use the `latest` endpoint. Invalid URL
values fall back to `pre_tournament`; switching keys clears the prior display
while the selected run loads.

## Production verification checklist

1. Apply the migration.
2. Run every snapshot with `--dry-run`.
3. Confirm the cutoff source is `database_schedule` and matches the earliest
   official stage kickoff.
4. Confirm completed counts match the tournament state immediately before the
   stage.
5. Inspect retrospective fixture IDs and `maximum_source_timestamp`; no source
   timestamp may reach the cutoff.
6. Confirm selected-stage fixture counts are 16, 8, 4, 2, and 1 respectively.
7. Run a small seeded snapshot twice and verify identical probabilities and one
   stored identity.
8. Generate the production simulation count, query both new endpoints, and
   inspect the web selector before exposing it publicly.

No production backfill is performed by implementation or tests.
