# v4.4.1 shadow deployment handoff

The repository now validates and atomically persists a same-timestamp production/shadow JSON batch with:

```text
python scripts/persist_scheduled_shadow_batch.py /path/to/generated-batch.json
python scripts/update_prospective_scorecard.py --results /path/to/completed-results.json
```

Each unplayed match must have one `elo-context-v4.2.1` snapshot and one explicitly versioned shadow snapshot generated in the same scheduler invocation. The batch command rejects different generation timestamps, missing model pairs, after-kickoff rows, invalid probabilities, conflicting retries, and content-hash mutations.

The deployment scheduler is not present in this repository and production database access was deliberately not used. Deployment must add these commands immediately after its existing production and shadow inference calls. It must not synthesize shadow predictions for completed matches. v4.4.1 failed the historical gate, so it remains shadow-only.
