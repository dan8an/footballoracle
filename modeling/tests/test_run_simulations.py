import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from modeling.src.data import build_fixtures, load_teams
from scripts.generate_predictions import (
    MODEL_VERSION,
    calculate_prediction,
    canonical_prior_elo,
)
from scripts.database import create_database_engine
from scripts.run_simulations import (
    MatchState,
    SimulationRepository,
    build_knockout_prediction_provider,
    build_round_of_32,
    knockout_winner,
    rank_group,
    simulate_tournaments,
)

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = """
create table model_runs (
  id text primary key,
  model_version text,
  status text
);
create table predictions (
  id text primary key,
  match_id text,
  provider_fixture_id integer,
  canonical_match_id text,
  model_run_id text,
  model_version text,
  prediction_timestamp text,
  created_at text,
  generation_mode text,
  home_xg real,
  away_xg real,
  home_win_probability real,
  draw_probability real,
  away_win_probability real,
  score_probabilities text
);
create table teams (
  id text primary key,
  name text not null
);
create table matches (
  id text primary key,
  match_number integer,
  stage text,
  tournament_stage text,
  kickoff text,
  match_date text,
  home_team_id text,
  away_team_id text,
  home_score integer,
  away_score integer,
  completed boolean,
  status text,
  provider_name text,
  provider_payload text,
  provider_fixture_id integer,
  api_football_fixture_id integer,
  updated_at text,
  created_at text
);
create table team_ratings (
  team_id text not null,
  model_run_id text,
  rated_at text,
  updated_at text,
  elo_rating real,
  attack_rating real,
  defense_rating real,
  form_rating real,
  matches_played integer
);
create table player_ratings (
  player_id text primary key,
  team_id text,
  model_run_id text,
  rated_at text,
  overall_rating real
);
create table team_chance_quality_ratings (
  team_id text not null,
  rated_at text,
  model_version text,
  shot_volume_rating real
);
create table simulation_runs (
  id text primary key,
  model_run_id text,
  model_version text,
  num_simulations integer,
  random_seed integer,
  snapshot_key text,
  cutoff_at text,
  reconstruction_mode text,
  simulation_config_version text,
  status text,
  provenance text,
  created_at text
);
create table team_simulation_results (
  simulation_run_id text not null,
  team_id text not null,
  group_stage_exit_probability real not null,
  round_of_32_probability real not null,
  round_of_16_probability real not null,
  quarterfinal_probability real not null,
  semifinal_probability real not null,
  final_probability real not null,
  champion_probability real not null,
  created_at text,
  primary key (simulation_run_id, team_id)
);
"""


def canonical_predictions():
    teams = {team.id: team for team in load_teams()}
    predictions = {}
    for fixture in build_fixtures(list(teams.values())):
        if fixture.stage != "group":
            continue
        home = teams[fixture.home_team_id]
        away = teams[fixture.away_team_id]
        prediction = calculate_prediction(
            {
                "elo_rating": home.elo,
                "attack_rating": 50,
                "defense_rating": 50,
                "form_rating": 50,
                "matches_played": 0,
            },
            {
                "elo_rating": away.elo,
                "attack_rating": 50,
                "defense_rating": 50,
                "form_rating": 50,
                "matches_played": 0,
            },
        )
        predictions[fixture.id] = prediction
    return predictions


def canonical_knockout_prediction():
    ratings = {
        team.id: {
            "elo_rating": canonical_prior_elo(team.rank),
            "attack_rating": 50,
            "defense_rating": 50,
            "form_rating": 50,
            "matches_played": 0,
        }
        for team in load_teams()
    }
    return build_knockout_prediction_provider(ratings)


def certain_home_win_prediction():
    return {
        "home_win_probability": 1.0,
        "draw_probability": 0.0,
        "away_win_probability": 0.0,
        "home_xg": 2.0,
        "away_xg": 0.2,
        "score_probabilities": [
            {"home_goals": 1, "away_goals": 0, "probability": 1.0},
        ],
    }


def world_cup_provider_payload():
    return json.dumps({"league": {"id": 1, "season": 2026}})


def completed_group_states(home_score=0, away_score=0):
    return [
        MatchState(
            id=fixture.id,
            stage="group",
            home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id,
            completed=True,
            home_score=home_score,
            away_score=away_score,
            match_number=fixture.number,
            kickoff=fixture.kickoff,
        )
        for fixture in build_fixtures(load_teams())
    ]


def july_9_knockout_state():
    teams = load_teams()
    team_groups = {team.id: team.group for team in teams}
    results_by_group = defaultdict(list)
    for fixture in build_fixtures(teams):
        results_by_group[fixture.group].append(
            (fixture.home_team_id, fixture.away_team_id, 0, 0)
        )
    group_tables = {
        group: rank_group(
            [team.id for team in teams if team.group == group],
            results_by_group[group],
        )
        for group in "ABCDEFGHIJKL"
    }
    round_of_32 = build_round_of_32(group_tables, team_groups)
    r32_winners = [home_id for home_id, _away_id in round_of_32]
    round_of_16 = [
        (r32_winners[index], r32_winners[index + 1])
        for index in range(0, len(r32_winners), 2)
    ]
    r16_winners = [home_id for home_id, _away_id in round_of_16]
    quarterfinals = [
        (r16_winners[index], r16_winners[index + 1])
        for index in range(0, len(r16_winners), 2)
    ]
    states = completed_group_states()
    states.extend(
        MatchState(
            id=f"r32-{index}",
            stage="round_of_32",
            home_team_id=home_id,
            away_team_id=away_id,
            completed=True,
            home_score=1,
            away_score=0,
            match_number=72 + index,
        )
        for index, (home_id, away_id) in enumerate(round_of_32, start=1)
    )
    states.extend(
        MatchState(
            id=f"r16-{index}",
            stage="round_of_16",
            home_team_id=home_id,
            away_team_id=away_id,
            completed=True,
            home_score=1,
            away_score=0,
            match_number=88 + index,
        )
        for index, (home_id, away_id) in enumerate(round_of_16, start=1)
    )
    states.extend(
        MatchState(
            id=f"qf-{index}",
            stage="quarterfinal",
            home_team_id=home_id,
            away_team_id=away_id,
            completed=False,
            match_number=96 + index,
        )
        for index, (home_id, away_id) in enumerate(quarterfinals, start=1)
    )
    return states, quarterfinals


class SimulationCalculationTests(unittest.TestCase):
    def test_group_tiebreaker_is_deterministic(self):
        table = rank_group(
            ["AAA", "BBB", "CCC", "DDD"],
            [
                ("AAA", "BBB", 0, 0),
                ("CCC", "DDD", 0, 0),
                ("AAA", "CCC", 0, 0),
                ("BBB", "DDD", 0, 0),
                ("AAA", "DDD", 0, 0),
                ("BBB", "CCC", 0, 0),
            ],
        )
        self.assertEqual([row["team_id"] for row in table], ["AAA", "BBB", "CCC", "DDD"])

    def test_simulation_is_reproducible_and_stage_totals_are_correct(self):
        predictions = canonical_predictions()

        first = simulate_tournaments(
            predictions, 20, 7, canonical_knockout_prediction()
        )
        second = simulate_tournaments(
            predictions, 20, 7, canonical_knockout_prediction()
        )

        self.assertEqual(first, second)
        expected_totals = {
            "group_stage_exit_probability": 16,
            "round_of_32_probability": 32,
            "round_of_16_probability": 16,
            "quarterfinal_probability": 8,
            "semifinal_probability": 4,
            "final_probability": 2,
            "champion_probability": 1,
        }
        for field, expected in expected_totals.items():
            self.assertAlmostEqual(sum(row[field] for row in first), expected)

    def test_completed_group_stage_allows_only_upcoming_knockout_predictions(self):
        match_states, quarterfinals = july_9_knockout_state()
        predictions = {
            f"qf-{index}": certain_home_win_prediction()
            for index in range(1, 5)
        }

        results = {
            row["team_id"]: row
            for row in simulate_tournaments(
                predictions,
                20,
                7,
                canonical_knockout_prediction(),
                match_states,
            )
        }

        quarterfinalists = {team_id for pairing in quarterfinals for team_id in pairing}
        self.assertTrue(quarterfinalists)
        for team_id in quarterfinalists:
            self.assertEqual(results[team_id]["quarterfinal_probability"], 1.0)
        self.assertAlmostEqual(
            sum(row["quarterfinal_probability"] for row in results.values()),
            8,
        )

    def test_completed_knockout_fixture_is_fixed_not_resimulated(self):
        match_states = completed_group_states()
        match_states.append(
            MatchState(
                id="final-actual",
                stage="final",
                home_team_id="MEX",
                away_team_id="CAN",
                completed=True,
                home_score=1,
                away_score=1,
                home_penalty_score=5,
                away_penalty_score=4,
                match_number=104,
            )
        )

        results = {
            row["team_id"]: row
            for row in simulate_tournaments(
                {},
                20,
                7,
                canonical_knockout_prediction(),
                match_states,
            )
        }

        self.assertEqual(results["MEX"]["final_probability"], 1.0)
        self.assertEqual(results["CAN"]["final_probability"], 1.0)
        self.assertEqual(results["MEX"]["champion_probability"], 1.0)
        self.assertEqual(results["CAN"]["champion_probability"], 0.0)

    def test_mixed_quarterfinal_and_semifinal_state_uses_bracket_slots(self):
        match_states, quarterfinals = july_9_knockout_state()
        match_states = [
            state for state in match_states if state.stage != "quarterfinal"
        ]
        for index, (home_id, away_id) in enumerate(quarterfinals, start=1):
            match_states.append(
                MatchState(
                    id=f"qf-{index}",
                    stage="quarterfinal",
                    home_team_id=home_id,
                    away_team_id=away_id,
                    completed=index <= 2,
                    home_score=1 if index <= 2 else None,
                    away_score=0 if index <= 2 else None,
                    match_number=96 + index,
                )
            )
        known_semifinal = (quarterfinals[0][0], quarterfinals[1][0])
        match_states.append(
            MatchState(
                id="sf-1",
                stage="semifinal",
                home_team_id=known_semifinal[0],
                away_team_id=known_semifinal[1],
                completed=False,
                match_number=101,
            )
        )
        predictions = {
            "qf-3": certain_home_win_prediction(),
            "qf-4": certain_home_win_prediction(),
            "sf-1": certain_home_win_prediction(),
        }

        results = {
            row["team_id"]: row
            for row in simulate_tournaments(
                predictions,
                20,
                7,
                lambda _home, _away: certain_home_win_prediction(),
                match_states,
            )
        }

        # Completed QFs are fixed, the known SF occupies slot 101, and slot 102
        # is built only from winners of QF99 and QF100.
        self.assertEqual(results[quarterfinals[0][0]]["semifinal_probability"], 1.0)
        self.assertEqual(results[quarterfinals[0][1]]["semifinal_probability"], 0.0)
        self.assertEqual(results[quarterfinals[1][0]]["semifinal_probability"], 1.0)
        self.assertEqual(results[quarterfinals[1][1]]["semifinal_probability"], 0.0)
        self.assertAlmostEqual(
            sum(row["semifinal_probability"] for row in results.values()), 4.0
        )
        self.assertAlmostEqual(
            sum(row["final_probability"] for row in results.values()), 2.0
        )
        self.assertAlmostEqual(
            sum(row["champion_probability"] for row in results.values()), 1.0
        )

    def test_odd_dynamic_round_raises_detailed_bracket_error(self):
        full_state, _quarterfinals = july_9_knockout_state()
        match_states = [
            state for state in full_state
            if state.stage == "group"
            or (state.stage == "round_of_32" and state.match_number != 88)
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"Invalid knockout bracket state: stage=round_of_16 "
            r"current_team_count=15.*dynamic child-winner list is odd",
        ):
            simulate_tournaments(
                {}, 1, 7, canonical_knockout_prediction(), match_states
            )

    def test_incomplete_predictions_report_exact_missing_fixtures(self):
        predictions = canonical_predictions()
        del predictions["WC26-001"]
        del predictions["WC26-002"]

        with self.assertRaisesRegex(
            ValueError,
            (
                r"missing 2 group fixtures: "
                r"WC26-001 \(Mexico vs South Africa\), "
                r"WC26-002 \(South Korea vs Czechia\)"
            ),
        ):
            simulate_tournaments(
                predictions,
                1,
                7,
                canonical_knockout_prediction(),
            )

    def test_pre_tournament_still_requires_all_group_predictions(self):
        results = simulate_tournaments(
            canonical_predictions(),
            5,
            7,
            canonical_knockout_prediction(),
            [],
        )

        self.assertAlmostEqual(
            sum(row["round_of_32_probability"] for row in results),
            32,
        )

    def test_knockout_winner_uses_pair_specific_regulation_probabilities(self):
        prediction = {
            "home_win_probability": 1.0,
            "draw_probability": 0.0,
            "away_win_probability": 0.0,
        }
        winners = {
            knockout_winner("AAA", "BBB", prediction, random.Random(seed))
            for seed in range(20)
        }

        self.assertEqual(winners, {"AAA"})

    def test_knockout_provider_computes_v4_for_the_actual_matchup(self):
        ratings = {
            "CZE": {
                "elo_rating": 1535,
                "attack_rating": 48,
                "defense_rating": 52,
            },
            "ESP": {
                "elo_rating": 1640,
                "attack_rating": 70,
                "defense_rating": 74,
            },
        }
        provider = build_knockout_prediction_provider(ratings)

        actual = provider("CZE", "ESP")
        expected = calculate_prediction(
            ratings["CZE"],
            ratings["ESP"],
            home_team_name="Czechia",
            away_team_name="Spain",
        )

        self.assertEqual(actual, expected)
        self.assertLess(
            actual["home_win_probability"],
            provider("ESP", "CZE")["home_win_probability"],
        )

    def test_knockout_draws_are_resolved(self):
        prediction = {
            "home_win_probability": 0.0,
            "draw_probability": 1.0,
            "away_win_probability": 0.0,
        }
        winners = [
            knockout_winner("AAA", "BBB", prediction, random.Random(seed))
            for seed in range(100)
        ]

        self.assertEqual(set(winners), {"AAA", "BBB"})

    def test_single_saturated_feature_does_not_create_tournament_favorite(self):
        teams = load_teams()
        ratings = {
            team.id: {
                "elo_rating": 1500,
                "attack_rating": 50,
                "defense_rating": 50,
            }
            for team in teams
        }
        shot_volume = {team.id: 50.0 for team in teams}
        shot_volume["NOR"] = 100.0
        predictions = {}
        for fixture in build_fixtures(teams):
            if fixture.stage != "group":
                continue
            predictions[fixture.id] = calculate_prediction(
                ratings[fixture.home_team_id],
                ratings[fixture.away_team_id],
                home_shot_volume_rating=shot_volume[fixture.home_team_id],
                away_shot_volume_rating=shot_volume[fixture.away_team_id],
            )

        results = {
            row["team_id"]: row
            for row in simulate_tournaments(
                predictions,
                5_000,
                7,
                build_knockout_prediction_provider(
                    ratings,
                    shot_volume_ratings=shot_volume,
                ),
            )
        }

        self.assertLess(results["NOR"]["champion_probability"], 0.05)


class SimulationScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "simulations.sqlite3"
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(SCHEMA)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_script(self, extra_args=None):
        return subprocess.run(
            [
                sys.executable,
                "scripts/run_simulations.py",
                "--simulations",
                "10",
                "--seed",
                "11",
                *(extra_args or []),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DATABASE_URL": f"sqlite:///{self.database_path}",
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def insert_teams_and_ratings(self):
        with sqlite3.connect(self.database_path) as connection:
            teams = load_teams()
            connection.executemany(
                "insert into teams (id, name) values (?, ?)",
                [(team.id, team.name) for team in teams],
            )
            connection.executemany(
                """
                insert into team_ratings (
                  team_id, model_run_id, rated_at, updated_at, elo_rating,
                  attack_rating, defense_rating, form_rating, matches_played
                ) values (?, null, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        team.id,
                        "2026-06-10",
                        "2026-06-10",
                        canonical_prior_elo(team.rank),
                        50,
                        50,
                        50,
                        0,
                    )
                    for team in teams
                ],
            )

    def insert_prediction_rows(
        self,
        predictions: dict[str, dict],
        model_run_id: str = "run-1",
    ):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "insert into model_runs (id, model_version) values (?, ?)",
                (model_run_id, MODEL_VERSION),
            )
            rows = []
            for fixture_id, prediction in predictions.items():
                rows.append(
                    (
                        fixture_id,
                        fixture_id,
                        model_run_id,
                        MODEL_VERSION,
                        "2026-06-10T12:00:00+00:00",
                        prediction["home_xg"],
                        prediction["away_xg"],
                        prediction["home_win_probability"],
                        prediction["draw_probability"],
                        prediction["away_win_probability"],
                        json.dumps(prediction["score_probabilities"]),
                    )
                )
            connection.executemany(
                """
                insert into predictions (
                  id, canonical_match_id, model_run_id, model_version,
                  prediction_timestamp, home_xg, away_xg,
                  home_win_probability, draw_probability,
                  away_win_probability, score_probabilities
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def insert_predictions(self):
        self.insert_teams_and_ratings()
        self.insert_prediction_rows(canonical_predictions())

    def repository(self):
        engine = create_database_engine(f"sqlite:///{self.database_path}")
        self.addCleanup(engine.dispose)
        return SimulationRepository(engine)

    def database_team_ids(self):
        return {team.id: team.id for team in load_teams()}

    def insert_match_rows(self, rows):
        columns = [
            "id",
            "match_number",
            "stage",
            "tournament_stage",
            "kickoff",
            "match_date",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
            "completed",
            "status",
            "provider_name",
            "provider_payload",
            "provider_fixture_id",
            "api_football_fixture_id",
            "updated_at",
            "created_at",
        ]
        with sqlite3.connect(self.database_path) as connection:
            connection.executemany(
                f"""
                insert into matches ({', '.join(columns)})
                values ({', '.join('?' for _ in columns)})
                """,
                [tuple(row.get(column) for column in columns) for row in rows],
            )

    def test_loader_normalizes_group_stage_variants(self):
        fixtures = build_fixtures(load_teams())[:4]
        labels = ["group", "Group", "group_stage", "First Round"]
        self.insert_match_rows(
            [
                {
                    "id": fixture.id,
                    "match_number": fixture.number,
                    "tournament_stage": label,
                    "kickoff": fixture.kickoff.isoformat(),
                    "home_team_id": fixture.home_team_id,
                    "away_team_id": fixture.away_team_id,
                    "home_score": 1,
                    "away_score": 0,
                    "status": "FT",
                }
                for fixture, label in zip(fixtures, labels)
            ]
        )

        states = self.repository().load_match_states(self.database_team_ids())

        completed_groups = [
            state for state in states if state.stage == "group" and state.completed
        ]
        self.assertEqual(
            [state.id for state in completed_groups],
            [fixture.id for fixture in fixtures],
        )

    def test_duplicate_knockout_rows_do_not_inflate_completed_count(self):
        self.insert_match_rows(
            [
                {
                    "id": "official-73",
                    "match_number": 73,
                    "tournament_stage": "Round of 32",
                    "kickoff": "2026-06-28T17:00:00+00:00",
                    "home_team_id": "MEX",
                    "away_team_id": "CAN",
                    "home_score": 1,
                    "away_score": 0,
                    "status": "FT",
                    "updated_at": "2026-06-28T20:00:00+00:00",
                },
                {
                    "id": "duplicate-73",
                    "match_number": 73,
                    "tournament_stage": "Round of 32",
                    "kickoff": "2026-06-28T17:00:00+00:00",
                    "home_team_id": "MEX",
                    "away_team_id": "CAN",
                    "home_score": 1,
                    "away_score": 0,
                    "status": "FT",
                    "updated_at": "2026-06-28T19:00:00+00:00",
                },
                {
                    "id": "historical-final",
                    "match_number": 104,
                    "tournament_stage": "Final",
                    "kickoff": "2022-12-18T15:00:00+00:00",
                    "home_team_id": "ARG",
                    "away_team_id": "FRA",
                    "home_score": 3,
                    "away_score": 3,
                    "status": "PEN",
                },
            ]
        )

        states = self.repository().load_match_states(self.database_team_ids())

        completed_knockouts = [
            state
            for state in states
            if state.stage in {"round_of_32", "final"} and state.completed
        ]
        self.assertEqual(len(completed_knockouts), 1)
        self.assertEqual(completed_knockouts[0].id, "official-73")

    def test_completed_scored_knockout_row_wins_over_scheduled_duplicate(self):
        self.insert_match_rows(
            [
                {
                    "id": "scheduled-qf",
                    "match_number": 97,
                    "tournament_stage": "Quarter-finals",
                    "kickoff": "2026-07-09T20:00:00+00:00",
                    "home_team_id": "MEX",
                    "away_team_id": "CAN",
                    "status": "NS",
                    "updated_at": "2026-07-09T18:00:00+00:00",
                },
                {
                    "id": "completed-qf",
                    "match_number": 97,
                    "tournament_stage": "Quarter-finals",
                    "kickoff": "2026-07-09T20:00:00+00:00",
                    "home_team_id": "MEX",
                    "away_team_id": "CAN",
                    "home_score": 2,
                    "away_score": 0,
                    "status": "FT",
                    "updated_at": "2026-07-09T17:00:00+00:00",
                },
            ]
        )

        states = self.repository().load_match_states(self.database_team_ids())

        [quarterfinal] = [state for state in states if state.stage == "quarterfinal"]
        self.assertEqual(quarterfinal.id, "completed-qf")
        self.assertTrue(quarterfinal.completed)
        self.assertEqual((quarterfinal.home_score, quarterfinal.away_score), (2, 0))

    def test_snapshot_cutoff_controls_completed_results_and_future_participants(self):
        self.insert_match_rows(
            [
                {
                    "id": "r32-before",
                    "match_number": 73,
                    "tournament_stage": "Round of 32",
                    "kickoff": "2026-06-28T19:00:00+00:00",
                    "home_team_id": "MEX",
                    "away_team_id": "CAN",
                    "home_score": 1,
                    "away_score": 0,
                    "status": "FT",
                },
                {
                    "id": "r16-after",
                    "match_number": 89,
                    "tournament_stage": "Round of 16",
                    "kickoff": "2026-07-04T21:00:00+00:00",
                    "home_team_id": "MEX",
                    "away_team_id": "USA",
                    "home_score": 2,
                    "away_score": 0,
                    "status": "FT",
                },
                {
                    "id": "final-future-participants",
                    "match_number": 104,
                    "tournament_stage": "Final",
                    "kickoff": "2026-07-19T19:00:00+00:00",
                    "home_team_id": "MEX",
                    "away_team_id": "ARG",
                    "home_score": 0,
                    "away_score": 1,
                    "status": "FT",
                },
            ]
        )
        cutoff = datetime(2026, 7, 4, 17, tzinfo=timezone.utc)

        states = self.repository().load_match_states(
            self.database_team_ids(),
            cutoff_at=cutoff,
            snapshot_stage="round_of_16",
        )

        self.assertTrue(next(row for row in states if row.id == "r32-before").completed)
        self.assertFalse(next(row for row in states if row.id == "r16-after").completed)
        self.assertNotIn("final-future-participants", {row.id for row in states})

    def test_historical_prediction_selection_is_before_cutoff_and_kickoff(self):
        self.insert_teams_and_ratings()
        prediction = certain_home_win_prediction()
        self.insert_prediction_rows(
            {"WC26-001": prediction}, model_run_id="eligible-run"
        )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "insert into model_runs (id, model_version, status) values (?, ?, ?)",
                ("failed-run", MODEL_VERSION, "failed"),
            )
            connection.execute(
                """
                insert into predictions (
                  id, canonical_match_id, model_run_id, model_version,
                  prediction_timestamp, home_xg, away_xg,
                  home_win_probability, draw_probability,
                  away_win_probability, score_probabilities
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "after-cutoff",
                    "WC26-002",
                    "eligible-run",
                    MODEL_VERSION,
                    "2026-06-11T19:30:00+00:00",
                    9,
                    0,
                    1,
                    0,
                    0,
                    "[]",
                ),
            )
            connection.execute(
                """
                insert into predictions (
                  id, canonical_match_id, model_run_id, model_version,
                  prediction_timestamp, home_xg, away_xg,
                  home_win_probability, draw_probability,
                  away_win_probability, score_probabilities
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "failed-run-prediction",
                    "WC26-003",
                    "failed-run",
                    MODEL_VERSION,
                    "2026-06-11T12:00:00+00:00",
                    9,
                    0,
                    1,
                    0,
                    0,
                    "[]",
                ),
            )
            connection.execute(
                """
                insert into predictions (
                  id, canonical_match_id, model_run_id, model_version,
                  prediction_timestamp, home_xg, away_xg,
                  home_win_probability, draw_probability,
                  away_win_probability, score_probabilities
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "after-kickoff",
                    "WC26-001",
                    "eligible-run",
                    MODEL_VERSION,
                    "2026-06-11T18:30:00+00:00",
                    9,
                    0,
                    1,
                    0,
                    0,
                    "[]",
                ),
            )
        cutoff = datetime(2026, 6, 11, 19, tzinfo=timezone.utc)

        selected, audit = self.repository().load_historical_predictions(
            cutoff, [], MODEL_VERSION
        )

        self.assertEqual(selected["WC26-001"]["id"], "WC26-001")
        self.assertIn("WC26-002", audit["rejected_after_cutoff"])
        self.assertIn("WC26-001", audit["rejected_at_or_after_kickoff"])
        self.assertNotIn("WC26-002", selected)
        self.assertNotIn("WC26-003", selected)

    def test_snapshot_storage_is_idempotent_and_transactional(self):
        repository = self.repository()
        result = {
            "team_id": "MEX",
            **{f"{stage}_probability": 0.25 for stage in (
                "group_stage_exit",
                "round_of_32",
                "round_of_16",
                "quarterfinal",
                "semifinal",
                "final",
                "champion",
            )},
        }
        cutoff = datetime(2026, 6, 11, 19, tzinfo=timezone.utc)
        first = repository.store_results(
            None,
            MODEL_VERSION,
            10,
            7,
            [result],
            snapshot_key="pre_tournament",
            cutoff_at=cutoff,
            reconstruction_mode="retrospective_reconstruction",
            provenance={"test": True},
        )
        second = repository.store_results(
            None,
            MODEL_VERSION,
            10,
            7,
            [result],
            snapshot_key="pre_tournament",
            cutoff_at=cutoff,
            reconstruction_mode="retrospective_reconstruction",
            provenance={"test": True},
        )
        self.assertNotEqual(first, second)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "select count(*) from simulation_runs where snapshot_key = ?",
                    ("pre_tournament",),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "select count(*) from team_simulation_results"
                ).fetchone()[0],
                1,
            )

        with self.assertRaises(Exception):
            repository.store_results(
                None,
                MODEL_VERSION,
                10,
                7,
                [{"team_id": "USA"}],
                snapshot_key="pre_final",
                cutoff_at=datetime(2026, 7, 19, 19, tzinfo=timezone.utc),
                reconstruction_mode="retrospective_reconstruction",
            )
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "select count(*) from simulation_runs where snapshot_key = ?",
                    ("pre_final",),
                ).fetchone()[0],
                0,
            )

    def test_script_runs_with_completed_group_stage_and_four_knockout_predictions(self):
        self.insert_teams_and_ratings()
        fixtures = build_fixtures(load_teams())
        self.insert_match_rows(
            [
                {
                    "id": fixture.id,
                    "match_number": fixture.number,
                    "tournament_stage": "Group Stage",
                    "kickoff": fixture.kickoff.isoformat(),
                    "home_team_id": fixture.home_team_id,
                    "away_team_id": fixture.away_team_id,
                    "home_score": 0,
                    "away_score": 0,
                    "status": "FT",
                }
                for fixture in fixtures
            ]
            + [
                {
                    "id": f"qf-{index}",
                    "match_number": 96 + index,
                    "tournament_stage": "Quarter-finals",
                    "kickoff": f"2026-07-{8 + index:02d}T20:00:00+00:00",
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "status": "NS",
                    "provider_name": "api_football",
                    "provider_payload": world_cup_provider_payload(),
                    "api_football_fixture_id": 90000 + index,
                }
                for index, (home_id, away_id) in enumerate(
                    [("MEX", "CAN"), ("USA", "BRA"), ("ARG", "FRA"), ("ESP", "GER")],
                    start=1,
                )
            ]
        )
        self.insert_prediction_rows(
            {
                f"qf-{index}": certain_home_win_prediction()
                for index in range(1, 5)
            }
        )

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("group_completed=72", result.stderr)
        self.assertIn("knockout_upcoming=4", result.stderr)

    def test_script_stores_one_result_per_team(self):
        self.insert_predictions()

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[simulation] SUCCESS", result.stderr)
        with sqlite3.connect(self.database_path) as connection:
            run = connection.execute(
                "select model_version, num_simulations from simulation_runs"
            ).fetchone()
            result_count = connection.execute(
                "select count(*) from team_simulation_results"
            ).fetchone()[0]
        self.assertEqual(run, (MODEL_VERSION, 10))
        self.assertEqual(result_count, 48)

    def test_no_predictions_exits_successfully(self):
        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no prediction run available", result.stderr)

    def test_snapshot_dry_run_audits_without_writing(self):
        self.insert_teams_and_ratings()

        result = self.run_script(["--snapshot", "pre_tournament", "--dry-run"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("model=elo-context-v4.2.1", result.stderr)
        self.assertIn("mode=retrospective_reconstruction", result.stderr)
        self.assertIn("DRY RUN: no simulation executed and no rows written", result.stderr)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute("select count(*) from simulation_runs").fetchone()[0],
                0,
            )


if __name__ == "__main__":
    unittest.main()
