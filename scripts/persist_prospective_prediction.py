#!/usr/bin/env python3
"""Idempotently persist one production or shadow prediction before kickoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from modeling.src.prospective import ProspectiveLedger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path, help="JSON prediction snapshot; model_version distinguishes production and shadow rows")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    digest = ProspectiveLedger(ROOT / "data/evaluation/wc26_prospective_predictions.json").persist(snapshot)
    print(digest)


if __name__ == "__main__": main()
