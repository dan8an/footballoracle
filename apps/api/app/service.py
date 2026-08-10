import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Engine, MetaData, Table, inspect, select, text

from modeling.src.data import ROOT, build_fixtures, load_teams, load_venues, validate_tournament
from modeling.src.features.context import ContextRepository
from modeling.src.flags import flag_for_team
from modeling.src.historical_snapshots import (
    SNAPSHOTS,
    SNAPSHOT_ACTIVE_TEAM_COUNTS,
    get_snapshot,
    parse_utc,
    resolve_snapshot_cutoff,
    snapshot_payload,
)
from modeling.src.poisson import predict_match
from modeling.src.simulation import (
    PUBLISHED_SIMULATION_ITERATIONS,
    prediction_dict,
    simulate_tournament,
)
from modeling.src.team_profiles import (
    form_summary,
    key_players,
    load_squad_metadata,
    load_squad_players,
    player_payload,
    recent_results,
    team_analysis,
)
from scripts.database import create_database_engine
from scripts.generate_predictions import (
    PredictionRepository,
    load_team_aliases,
)

STATIC_MODEL_VERSION = "context-0.2.0"
MODEL_VERSION = STATIC_MODEL_VERSION
LOGGER = logging.getLogger(__name__)
KNOCKOUT_STAGE_LIMITS = {
    "round_of_32": 16,
    "round_of_16": 8,
    "quarterfinal": 4,
    "semifinal": 2,
    "final": 1,
    "third_place": 1,
}
KNOCKOUT_MATCH_NUMBER_RANGES = {
    "round_of_32": range(73, 89),
    "round_of_16": range(89, 97),
    "quarterfinal": range(97, 101),
    "semifinal": range(101, 103),
    "third_place": range(103, 104),
    "final": range(104, 105),
}
COMPLETED_MATCH_STATUSES = {"completed", "finished", "ft", "aet", "pen"}
KNOCKOUT_WINDOW_START = datetime(2026, 6, 28, tzinfo=timezone.utc)
KNOCKOUT_WINDOW_END = datetime(2026, 7, 20, tzinfo=timezone.utc)
SNAPSHOT_ADVANCEMENT_FIELDS = {
    "pre_round_of_32": "round_of_32",
    "pre_round_of_16": "round_of_16",
    "pre_quarterfinals": "quarterfinal",
    "pre_semifinals": "semifinal",
    "pre_final": "final",
}


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _integer_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class DatabasePredictionSource:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @classmethod
    def from_environment(cls) -> "DatabasePredictionSource | None":
        load_dotenv(ROOT / ".env", override=False)
        load_dotenv(ROOT / "backend" / ".env", override=False)
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            LOGGER.warning(
                "DATABASE_URL is unavailable; serving static prediction fallback"
            )
            return None
        try:
            return cls(create_database_engine(database_url))
        except Exception:
            LOGGER.warning(
                "DATABASE_URL could not initialize; serving static prediction "
                "fallback",
                exc_info=True,
            )
            return None

    def load_latest(self) -> dict[str, Any] | None:
        table_prefix = "public." if self.engine.dialect.name == "postgresql" else ""
        schema = "public" if self.engine.dialect.name == "postgresql" else None
        prediction_columns = {
            column["name"]
            for column in inspect(self.engine).get_columns("predictions", schema=schema)
        }
        standard_prediction_filter = (
            "and coalesce(p.generation_mode, 'standard') <> 'historical_backfill'"
            if "generation_mode" in prediction_columns else ""
        )
        with self.engine.connect() as connection:
            latest = connection.execute(
                text(
                    f"""
                    select
                      mr.id as model_run_id,
                      mr.model_version,
                      mr.generated_at,
                      mr.data_cutoff
                    from {table_prefix}model_runs mr
                    where coalesce(mr.status, 'completed') = 'completed'
                      and exists (
                        select 1
                        from {table_prefix}predictions p
                        where p.model_run_id = mr.id
                        {standard_prediction_filter}
                      )
                    order by mr.generated_at desc nulls last, mr.id desc
                    limit 1
                    """
                )
            ).mappings().one_or_none()
            if latest is None:
                return None
            rows = [
                dict(row)
                for row in connection.execute(
                    text(
                        f"""
                        select *
                        from {table_prefix}predictions
                        where model_run_id = :model_run_id
                        """
                    ),
                    {"model_run_id": latest["model_run_id"]},
                ).mappings()
            ]
            history = [
                dict(row)
                for row in connection.execute(
                    text(
                        f"""
                        select p.*,
                               mr.model_version as run_model_version,
                               mr.generated_at as run_generated_at,
                               mr.data_cutoff as run_data_cutoff
                        from {table_prefix}predictions p
                        join {table_prefix}model_runs mr on mr.id = p.model_run_id
                        where coalesce(mr.status, 'completed') = 'completed'
                        order by coalesce(p.prediction_timestamp,
                                          mr.generated_at) desc
                        """
                    )
                ).mappings()
            ]
        return {
            "model_run_id": latest["model_run_id"],
            "model_version": latest["model_version"],
            "generated_at": latest["generated_at"],
            "data_cutoff": latest.get("data_cutoff")
            or latest["generated_at"],
            "predictions": self._prediction_lookup(rows),
            "prediction_history": history,
        }

    @staticmethod
    def _prediction_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        predictions = {}
        for row in rows:
            for key in (
                row.get("canonical_match_id"),
                row.get("match_id"),
                row.get("provider_fixture_id"),
            ):
                if key is not None:
                    predictions[str(key)] = row
        return predictions


class DatabaseSimulationSource:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load_latest(self) -> dict[str, Any] | None:
        table_prefix = "public." if self.engine.dialect.name == "postgresql" else ""
        schema = "public" if self.engine.dialect.name == "postgresql" else None
        columns = {
            column["name"]
            for column in inspect(self.engine).get_columns(
                "simulation_runs", schema=schema
            )
        }
        current_run_filter = (
            "and sr.snapshot_key is null" if "snapshot_key" in columns else ""
        )
        with self.engine.connect() as connection:
            latest = connection.execute(
                text(
                    f"""
                    select sr.*
                    from {table_prefix}simulation_runs sr
                    where exists (
                      select 1
                      from {table_prefix}team_simulation_results tsr
                      where tsr.simulation_run_id = sr.id
                    )
                    {current_run_filter}
                    order by sr.created_at desc, sr.id desc
                    limit 1
                    """
                )
            ).mappings().one_or_none()
            if latest is None:
                return None
            rows = [
                dict(row)
                for row in connection.execute(
                    text(
                        f"""
                        select *
                        from {table_prefix}team_simulation_results
                        where simulation_run_id = :simulation_run_id
                        """
                    ),
                    {"simulation_run_id": latest["id"]},
                ).mappings()
            ]
        return {
            "run": dict(latest),
            "results": rows,
            "model_inputs": self._load_model_inputs(),
        }

    def load_snapshot(self, snapshot_key: str) -> dict[str, Any] | None:
        get_snapshot(snapshot_key)
        schema = "public" if self.engine.dialect.name == "postgresql" else None
        columns = {
            column["name"]
            for column in inspect(self.engine).get_columns(
                "simulation_runs", schema=schema
            )
        }
        if "snapshot_key" not in columns:
            return None
        table_prefix = "public." if self.engine.dialect.name == "postgresql" else ""
        status_filter = (
            "and coalesce(sr.status, 'completed') = 'completed'"
            if "status" in columns else ""
        )
        with self.engine.connect() as connection:
            run = connection.execute(
                text(
                    f"""
                    select sr.*
                    from {table_prefix}simulation_runs sr
                    where sr.snapshot_key = :snapshot_key
                      {status_filter}
                      and exists (
                        select 1
                        from {table_prefix}team_simulation_results tsr
                        where tsr.simulation_run_id = sr.id
                      )
                    order by sr.created_at desc, sr.id desc
                    limit 1
                    """
                ),
                {"snapshot_key": snapshot_key},
            ).mappings().one_or_none()
            if run is None:
                return None
            rows = [
                dict(row)
                for row in connection.execute(
                    text(
                        f"""
                        select *
                        from {table_prefix}team_simulation_results
                        where simulation_run_id = :simulation_run_id
                        """
                    ),
                    {"simulation_run_id": run["id"]},
                ).mappings()
            ]
        return {"run": dict(run), "results": rows, "model_inputs": {}}

    def list_snapshots(self) -> list[dict[str, Any]]:
        schema = "public" if self.engine.dialect.name == "postgresql" else None
        inspector = inspect(self.engine)
        run_columns = {
            column["name"]
            for column in inspector.get_columns("simulation_runs", schema=schema)
        }
        table_prefix = "public." if self.engine.dialect.name == "postgresql" else ""
        with self.engine.connect() as connection:
            schedule_rows = [
                dict(row)
                for row in connection.execute(
                    text(f"select * from {table_prefix}matches")
                ).mappings()
            ]
            available_keys: set[str] = set()
            if "snapshot_key" in run_columns:
                status_filter = (
                    "and coalesce(sr.status, 'completed') = 'completed'"
                    if "status" in run_columns else ""
                )
                available_keys = {
                    str(value)
                    for value in connection.execute(
                        text(
                            f"""
                            select distinct sr.snapshot_key
                            from {table_prefix}simulation_runs sr
                            where sr.snapshot_key is not null
                              {status_filter}
                              and exists (
                                select 1
                                from {table_prefix}team_simulation_results tsr
                                where tsr.simulation_run_id = sr.id
                              )
                            """
                        )
                    ).scalars()
                }
        payload = []
        for snapshot in SNAPSHOTS:
            cutoff_at, cutoff_source = resolve_snapshot_cutoff(
                snapshot, schedule_rows
            )
            payload.append(
                snapshot_payload(
                    snapshot,
                    cutoff_at,
                    cutoff_source=cutoff_source,
                    available=snapshot.key in available_keys,
                )
            )
        return payload

    def _load_model_inputs(self) -> dict[str, dict[str, Any]]:
        try:
            repository = PredictionRepository(self.engine)
            database_team_ids = repository.load_database_team_ids()
            ratings = repository.load_current_team_ratings(database_team_ids)
            shot_volume = repository.load_current_shot_volume_details(
                database_team_ids
            )
        except Exception:
            LOGGER.warning(
                "Simulation rating transparency metadata is unavailable",
                exc_info=True,
            )
            return {}

        ranked_team_ids = sorted(
            ratings,
            key=lambda team_id: float(ratings[team_id]["elo_rating"]),
            reverse=True,
        )
        elo_ranks = {
            team_id: rank
            for rank, team_id in enumerate(ranked_team_ids, start=1)
        }
        return {
            team_id: {
                "elo_rating": float(rating["elo_rating"]),
                "elo_rank": elo_ranks[team_id],
                "attack_rating": float(rating["attack_rating"]),
                "defense_rating": float(rating["defense_rating"]),
                "shot_volume_rating": (
                    float(shot_volume[team_id]["shot_volume_rating"])
                    if team_id in shot_volume
                    else None
                ),
                "rating_source": rating["_rating_source"],
                "rating_matches": int(rating.get("matches_played") or 0),
                "shot_volume_sample_matches": (
                    shot_volume.get(team_id, {}).get("sample_matches")
                ),
            }
            for team_id, rating in ratings.items()
        }


class DatabaseMatchResultSource:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load(self) -> list[dict[str, Any]]:
        schema = None if self.engine.dialect.name == "sqlite" else "public"
        inspector = inspect(self.engine)
        if "matches" not in inspector.get_table_names(schema=schema):
            return []

        metadata = MetaData()
        matches = Table("matches", metadata, schema=schema, autoload_with=self.engine)
        teams = (
            Table("teams", metadata, schema=schema, autoload_with=self.engine)
            if "teams" in inspector.get_table_names(schema=schema)
            else None
        )
        with self.engine.connect() as connection:
            match_rows = [
                dict(row)
                for row in connection.execute(select(matches)).mappings()
            ]
            team_names = (
                {
                    row["id"]: row["name"]
                    for row in connection.execute(
                        select(teams.c.id, teams.c.name)
                    ).mappings()
                }
                if teams is not None
                else {}
            )

        def display_name(value: Any, team_id: Any) -> str | None:
            text_value = str(value or "").strip()
            if text_value and not text_value.isdigit():
                return text_value
            return team_names.get(team_id)

        return [
            {
                **row,
                "home_team_name": display_name(
                    row.get("home_team"), row.get("home_team_id")
                ),
                "away_team_name": display_name(
                    row.get("away_team"), row.get("away_team_id")
                ),
            }
            for row in match_rows
        ]


class PredictionService:
    def __init__(
        self,
        prediction_source: DatabasePredictionSource | None = None,
        simulation_source: DatabaseSimulationSource | None = None,
        match_result_source: DatabaseMatchResultSource | None = None,
        prediction_cache_seconds: float | None = None,
    ) -> None:
        self.generated_at = datetime.now(timezone.utc).isoformat()
        self.data_cutoff = self.generated_at
        self.teams = load_teams()
        self.teams_by_id = {team.id: team for team in self.teams}
        self.contexts = ContextRepository()
        all_fixtures = build_fixtures(self.teams)
        validate_tournament(self.teams, all_fixtures)
        self.fixtures = [match for match in all_fixtures if match.stage == "group"]
        self.venues = load_venues()
        self.squad_players = load_squad_players()
        self.squad_metadata = load_squad_metadata()
        self.static_predictions = {
            match.id: predict_match(
                self.teams_by_id[match.home_team_id or ""],
                self.teams_by_id[match.away_team_id or ""],
                match.id,
                context=self.contexts.for_match(
                    match.home_team_id or "",
                    match.away_team_id or "",
                    datetime.fromisoformat(self.data_cutoff),
                ),
            )
            for match in self.fixtures
            if match.stage == "group"
        }
        self.predictions = self.static_predictions
        self.prediction_source = (
            prediction_source
            if prediction_source is not None
            else DatabasePredictionSource.from_environment()
        )
        self.simulation_source = simulation_source
        if self.simulation_source is None and self.prediction_source is not None:
            engine = getattr(self.prediction_source, "engine", None)
            if engine is not None:
                self.simulation_source = DatabaseSimulationSource(engine)
        self.match_result_source = match_result_source
        if self.match_result_source is None and self.prediction_source is not None:
            engine = getattr(self.prediction_source, "engine", None)
            if engine is not None:
                self.match_result_source = DatabaseMatchResultSource(engine)
        aliases = load_team_aliases()
        alias_lookup = {}
        for team in self.teams:
            for name in (team.name, *aliases[team.id]):
                alias_lookup[self._normalize_name(name)] = team.id
        self.team_alias_lookup = alias_lookup
        try:
            configured_cache_seconds = float(
                os.getenv("PREDICTION_READ_CACHE_SECONDS", "30")
            )
        except ValueError:
            configured_cache_seconds = 30.0
            LOGGER.warning(
                "PREDICTION_READ_CACHE_SECONDS is invalid; using 30 seconds"
            )
        self.prediction_cache_seconds = max(
            0.0,
            prediction_cache_seconds
            if prediction_cache_seconds is not None
            else configured_cache_seconds,
        )
        self._database_prediction_run: dict[str, Any] | None = None
        self._database_prediction_checked_at = 0.0
        self._simulation: dict | None = None
        self._simulation_generated_at = self.generated_at
        self._simulation_data_cutoff = self.data_cutoff

    @staticmethod
    def _normalize_name(value: Any) -> str:
        return "".join(
            character
            for character in str(value or "").lower()
            if character.isalnum()
        )

    def current_match_rows(self) -> list[dict[str, Any]]:
        if self.match_result_source is None:
            return []
        try:
            return self.match_result_source.load()
        except Exception:
            LOGGER.warning("Database match results are unavailable", exc_info=True)
            return []

    def current_match_results(
        self,
        rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        rows = rows if rows is not None else self.current_match_rows()

        fixtures_by_teams: dict[
            tuple[str, str],
            list[Any],
        ] = {}
        for fixture in self.fixtures:
            if fixture.home_team_id and fixture.away_team_id:
                fixtures_by_teams.setdefault(
                    (fixture.home_team_id, fixture.away_team_id),
                    [],
                ).append(fixture)
        results = {}
        for row in rows:
            raw_match_id = row.get("canonical_match_id") or row.get("match_id") or row.get("id")
            direct_fixture = next(
                (
                    fixture
                    for fixture in self.fixtures
                    if str(raw_match_id) == fixture.id
                    or str(row.get("match_number") or row.get("number") or "")
                    == str(fixture.number)
                ),
                None,
            )
            played_at = row.get("match_date") or row.get("kickoff")
            if played_at is None and direct_fixture is None:
                continue
            if played_at is not None:
                try:
                    played_at = datetime.fromisoformat(
                        str(played_at).replace("Z", "+00:00")
                    )
                except ValueError:
                    continue
                if played_at.tzinfo is None:
                    played_at = played_at.replace(tzinfo=timezone.utc)
            home_id = self.team_alias_lookup.get(
                self._normalize_name(row.get("home_team_name"))
            )
            away_id = self.team_alias_lookup.get(
                self._normalize_name(row.get("away_team_name"))
            )
            fixture = direct_fixture
            if fixture is None:
                candidates = fixtures_by_teams.get((home_id, away_id), [])
                if not candidates:
                    continue
                fixture = min(
                    candidates,
                    key=lambda candidate: abs(
                        (candidate.kickoff - played_at).total_seconds()
                    ),
                )
                if abs((fixture.kickoff - played_at).total_seconds()) > 36 * 60 * 60:
                    continue
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            status = str(row.get("status") or "").strip()
            if not status and row.get("completed"):
                status = "completed"
            results[fixture.id] = {
                "status": status or "scheduled",
                "home_score": (
                    int(home_score) if home_score is not None else None
                ),
                "away_score": (
                    int(away_score) if away_score is not None else None
                ),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "winner_team_id": self.team_alias_lookup.get(
                    self._normalize_name(
                        row.get("winner_team_name") or row.get("winner")
                    )
                ),
            }
        return results

    @staticmethod
    def _stage_from_row(row: dict[str, Any]) -> str:
        raw = str(row.get("stage") or row.get("tournament_stage") or "").lower()
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
        return "group" if "group" in raw else str(row.get("stage") or "group")

    @staticmethod
    def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
        payload = _json_value(
            row.get("provider_payload") or row.get("raw_payload") or row.get("raw"),
            {},
        )
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _official_match_number(row: dict[str, Any], stage: str) -> int | None:
        number = _integer_value(row.get("match_number") or row.get("number"))
        if number is not None and number in KNOCKOUT_MATCH_NUMBER_RANGES.get(stage, ()):
            return number
        return None

    @staticmethod
    def _is_world_cup_2026_provider_row(row: dict[str, Any]) -> bool:
        payload = PredictionService._row_payload(row)
        league = payload.get("league") if isinstance(payload.get("league"), dict) else {}
        return (
            str(row.get("provider_name") or "").lower() == "api_football"
            and str(league.get("id") or "") == "1"
            and str(league.get("season") or "") == "2026"
        )

    @staticmethod
    def _knockout_status(row: dict[str, Any]) -> str:
        status = str(row.get("status") or "").strip()
        if not status and row.get("completed"):
            status = "completed"
        return status or "scheduled"

    @staticmethod
    def _has_score(row: dict[str, Any]) -> bool:
        return row.get("home_score") is not None and row.get("away_score") is not None

    @staticmethod
    def _is_completed_row(row: dict[str, Any]) -> bool:
        return (
            PredictionService._knockout_status(row).lower() in COMPLETED_MATCH_STATUSES
            or bool(row.get("completed"))
            or PredictionService._has_score(row)
        )

    @staticmethod
    def _updated_at(row: dict[str, Any]) -> datetime:
        value = row.get("updated_at") or row.get("created_at")
        if value is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return datetime.min.replace(tzinfo=timezone.utc)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
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

    @staticmethod
    def _knockout_row_rank(row: dict[str, Any], number: int | None) -> tuple[Any, ...]:
        return (
            PredictionService._is_completed_row(row),
            PredictionService._has_score(row),
            number is not None,
            PredictionService._updated_at(row),
        )

    def _official_knockout_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        selected: dict[tuple[Any, ...], dict[str, Any]] = {}
        selected_numbers: dict[tuple[Any, ...], int | None] = {}
        selected_kickoffs: dict[tuple[Any, ...], datetime] = {}
        selected_home_ids: dict[tuple[Any, ...], str] = {}
        selected_away_ids: dict[tuple[Any, ...], str] = {}

        for index, row in enumerate(rows, start=1):
            stage = self._stage_from_row(row)
            if stage not in KNOCKOUT_STAGE_LIMITS:
                continue
            home_id = self._row_team_id(row, "home")
            away_id = self._row_team_id(row, "away")
            if home_id not in self.teams_by_id or away_id not in self.teams_by_id:
                continue
            kickoff = row.get("kickoff") or row.get("match_date")
            if kickoff is None:
                continue
            try:
                kickoff_dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
            except ValueError:
                continue
            if kickoff_dt.tzinfo is None:
                kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
            if not (KNOCKOUT_WINDOW_START <= kickoff_dt < KNOCKOUT_WINDOW_END):
                continue
            number = self._official_match_number(row, stage)
            if number is None and not self._is_world_cup_2026_provider_row(row):
                continue
            key = self._knockout_logical_key(row, stage, number, kickoff_dt, home_id, away_id)
            existing = selected.get(key)
            if existing is None or self._knockout_row_rank(row, number) > self._knockout_row_rank(
                existing,
                selected_numbers[key],
            ):
                selected[key] = {**row, "_knockout_stage": stage, "_knockout_index": index}
                selected_numbers[key] = number
                selected_kickoffs[key] = kickoff_dt
                selected_home_ids[key] = home_id
                selected_away_ids[key] = away_id

        rows_by_stage: dict[str, list[tuple[dict[str, Any], int | None, datetime, str, str]]] = {
            stage: [] for stage in KNOCKOUT_STAGE_LIMITS
        }
        for key, row in selected.items():
            rows_by_stage[row["_knockout_stage"]].append(
                (
                    row,
                    selected_numbers[key],
                    selected_kickoffs[key],
                    selected_home_ids[key],
                    selected_away_ids[key],
                )
            )

        official_rows: list[dict[str, Any]] = []
        for stage, stage_rows in rows_by_stage.items():
            ordered = sorted(
                stage_rows,
                key=lambda item: (
                    item[1] is None,
                    item[1] if item[1] is not None else 999,
                    item[2],
                ),
            )
            for row, number, kickoff, home_id, away_id in ordered[: KNOCKOUT_STAGE_LIMITS[stage]]:
                official_rows.append(
                    {
                        **row,
                        "_knockout_stage": stage,
                        "_official_match_number": number,
                        "_kickoff_dt": kickoff,
                        "_home_id": home_id,
                        "_away_id": away_id,
                    }
                )
        return sorted(
            official_rows,
            key=lambda row: (
                row["_official_match_number"] is None,
                row["_official_match_number"] or 999,
                row["_kickoff_dt"],
            ),
        )

    def _row_team_id(self, row: dict[str, Any], side: str) -> str | None:
        direct = row.get(f"{side}_team_id")
        if direct in self.teams_by_id:
            return str(direct)
        return self.team_alias_lookup.get(
            self._normalize_name(row.get(f"{side}_team_name") or row.get(f"{side}_team"))
        )

    def team_payload(self, team_id: str) -> dict:
        team = self.teams_by_id[team_id]
        return {
            **asdict(team),
            "elo": round(team.elo, 1),
            "flag": flag_for_team(team_id),
        }

    def _latest_database_prediction_run(
        self,
        force: bool = False,
    ) -> dict[str, Any] | None:
        if self.prediction_source is None:
            return None
        now = time.monotonic()
        if (
            not force
            and now - self._database_prediction_checked_at
            < self.prediction_cache_seconds
        ):
            return self._database_prediction_run
        self._database_prediction_checked_at = now
        try:
            run = self.prediction_source.load_latest()
        except Exception:
            self._database_prediction_run = None
            LOGGER.warning(
                "Latest database predictions are unavailable; serving static "
                "prediction fallback",
                exc_info=True,
            )
            return None
        if run is None or not run["predictions"]:
            self._database_prediction_run = None
            LOGGER.warning(
                "No database prediction run is available; serving static "
                "prediction fallback"
            )
            return None
        self._database_prediction_run = run
        LOGGER.info(
            "Serving latest database predictions: run=%s model=%s rows=%d",
            run["model_run_id"],
            run["model_version"],
            len(run["predictions"]),
        )
        return run

    def current_prediction_run(self, force: bool = False) -> dict[str, Any]:
        database_run = self._latest_database_prediction_run(force=force)
        if database_run is not None:
            return database_run
        return {
            "model_run_id": None,
            "model_version": STATIC_MODEL_VERSION,
            "generated_at": self.generated_at,
            "data_cutoff": self.data_cutoff,
            "predictions": self.static_predictions,
            "source": "fallback_static",
        }

    def prediction_payload(
        self,
        match_id: str,
        prediction_run: dict[str, Any] | None = None,
        database_prediction_override: dict[str, Any] | None = None,
        allow_run_lookup: bool = True,
    ) -> dict | None:
        prediction_run = prediction_run or self.current_prediction_run()
        database_prediction = database_prediction_override or (
            prediction_run["predictions"].get(match_id)
            if allow_run_lookup and prediction_run["model_run_id"] is not None
            else None
        )
        prediction = self.static_predictions.get(match_id)
        if not prediction and database_prediction is None:
            return None
        if prediction:
            payload = {
                **prediction_dict(prediction),
                "model_version": STATIC_MODEL_VERSION,
                "generated_at": self.generated_at,
                "data_cutoff": self.data_cutoff,
                "source": "fallback_static",
            }
        else:
            payload = {
                "match_id": match_id,
                "home_team_id": database_prediction.get("home_team_id", ""),
                "away_team_id": database_prediction.get("away_team_id", ""),
                "home_xg": float(database_prediction.get("home_xg") or 0),
                "away_xg": float(database_prediction.get("away_xg") or 0),
                "probabilities": {
                    "home_win": float(
                        database_prediction.get("home_win_probability")
                        or database_prediction.get("home_win")
                        or 0
                    ),
                    "draw": float(
                        database_prediction.get("draw_probability")
                        or database_prediction.get("draw")
                        or 0
                    ),
                    "away_win": float(
                        database_prediction.get("away_win_probability")
                        or database_prediction.get("away_win")
                        or 0
                    ),
                },
                "top_scores": [],
                "confidence": database_prediction.get("confidence_tier")
                or "High uncertainty",
                "key_factors": [],
                "context": {
                    "home_form_elo": 0,
                    "away_form_elo": 0,
                    "home_h2h_elo": 0,
                    "away_h2h_elo": 0,
                    "home_availability_elo": 0,
                    "away_availability_elo": 0,
                    "historical_matches_home": 0,
                    "historical_matches_away": 0,
                    "h2h_matches": 0,
                    "availability_reports": 0,
                    "data_cutoff": None,
                },
                "model_version": STATIC_MODEL_VERSION,
                "generated_at": self.generated_at,
                "data_cutoff": self.data_cutoff,
                "source": "database_latest",
            }
        if database_prediction is None:
            return payload

        home = float(
            database_prediction.get("final_home_probability")
            if database_prediction.get("final_home_probability") is not None
            else database_prediction["home_win_probability"]
        )
        draw = float(
            database_prediction.get("final_draw_probability")
            if database_prediction.get("final_draw_probability") is not None
            else database_prediction["draw_probability"]
        )
        away = float(
            database_prediction.get("final_away_probability")
            if database_prediction.get("final_away_probability") is not None
            else database_prediction["away_win_probability"]
        )
        payload.update(
            {
                "probabilities": {
                    "home_win": home,
                    "draw": draw,
                    "away_win": away,
                },
                "elo_base_home_probability": database_prediction.get(
                    "elo_base_home_probability"
                ),
                "elo_base_draw_probability": database_prediction.get(
                    "elo_base_draw_probability"
                ),
                "elo_base_away_probability": database_prediction.get(
                    "elo_base_away_probability"
                ),
                "attack_defense_adjustment": database_prediction.get(
                    "attack_defense_adjustment"
                ),
                "draw_calibration_adjustment": database_prediction.get(
                    "draw_calibration_adjustment"
                ),
                "context_adjustment_total": database_prediction.get(
                    "context_adjustment_total"
                ),
                "final_home_probability": home,
                "final_draw_probability": draw,
                "final_away_probability": away,
                "confidence_score": database_prediction.get("confidence_score"),
                "confidence_tier": database_prediction.get("confidence_tier"),
                "confidence_explanation": database_prediction.get(
                    "confidence_explanation"
                ),
                "confidence": database_prediction.get("confidence_tier")
                or payload["confidence"],
                "top_factors": _json_value(
                    database_prediction.get("top_factors"),
                    [],
                ),
                "model_version": database_prediction.get("model_version")
                or database_prediction.get("run_model_version")
                or prediction_run["model_version"],
                "generated_at": str(
                    database_prediction.get("prediction_timestamp")
                    or database_prediction.get("created_at")
                    or database_prediction.get("run_generated_at")
                    or prediction_run["generated_at"]
                ),
                "data_cutoff": str(
                    database_prediction.get("data_cutoff")
                    or database_prediction.get("run_data_cutoff")
                    or database_prediction.get("prediction_timestamp")
                    or prediction_run["data_cutoff"]
                ),
                "source": "database_latest",
                "generation_mode": database_prediction.get("generation_mode")
                or "standard",
                "historical_cutoff": (
                    str(database_prediction.get("historical_cutoff"))
                    if database_prediction.get("historical_cutoff") is not None
                    else None
                ),
                "backfilled_at": (
                    str(database_prediction.get("backfilled_at"))
                    if database_prediction.get("backfilled_at") is not None
                    else None
                ),
            }
        )
        return payload

    @staticmethod
    def _prediction_timestamp(row: dict[str, Any]) -> datetime | None:
        value = (
            row.get("prediction_timestamp") or row.get("created_at")
            or row.get("run_generated_at")
        )
        if value is None:
            return None
        try:
            parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _historical_cutoff(row: dict[str, Any]) -> datetime | None:
        value = row.get("historical_cutoff")
        if value is None:
            return None
        try:
            parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _historical_knockout_prediction(
        self,
        row: dict[str, Any],
        match_id: str,
        kickoff: datetime,
        home_id: str,
        away_id: str,
        prediction_run: dict[str, Any],
    ) -> dict[str, Any] | None:
        history = list(prediction_run.get("prediction_history", []))
        history.extend(
            {
                **prediction,
                "run_model_version": prediction_run.get("model_version"),
                "run_generated_at": prediction_run.get("generated_at"),
                "run_data_cutoff": prediction_run.get("data_cutoff"),
            }
            for prediction in prediction_run.get("predictions", {}).values()
            if isinstance(prediction, dict)
        )
        current_provider = row.get("api_football_fixture_id") or row.get("provider_fixture_id")
        current_canonical = row.get("canonical_match_id")
        number = row.get("_official_match_number")
        official_id = f"WC26-{number:03d}" if number is not None else None
        candidates: list[tuple[int, int, datetime, dict[str, Any], str]] = []
        for prediction in history:
            timestamp = self._prediction_timestamp(prediction)
            backfilled = prediction.get("generation_mode") == "historical_backfill"
            cutoff = self._historical_cutoff(prediction)
            if timestamp is None or (
                not backfilled and timestamp >= kickoff
            ) or (
                backfilled and (cutoff is None or cutoff >= kickoff)
            ):
                continue
            prediction_match = prediction.get("match_id")
            prediction_provider = prediction.get("provider_fixture_id") or prediction.get("api_football_fixture_id")
            prediction_canonical = prediction.get("canonical_match_id")
            prediction_number = _integer_value(prediction.get("match_number") or prediction.get("number"))
            priority = None
            reason = ""
            if str(prediction_match or "") == match_id:
                priority, reason = 0, "exact_match_id"
            elif current_provider is not None and str(prediction_provider or "") == str(current_provider):
                priority, reason = 1, "provider_fixture_id"
            elif (
                current_canonical is not None
                and str(prediction_canonical or "") == str(current_canonical)
            ) or (
                official_id is not None
                and str(prediction_canonical or "") == official_id
            ) or (number is not None and prediction_number == number):
                priority, reason = 2, "canonical_knockout_identity"
            else:
                prediction_home = self._row_team_id(prediction, "home")
                prediction_away = self._row_team_id(prediction, "away")
                prediction_kickoff = prediction.get("kickoff") or prediction.get("match_date")
                try:
                    prediction_kickoff_dt = (
                        datetime.fromisoformat(str(prediction_kickoff).replace("Z", "+00:00"))
                        if prediction_kickoff is not None else None
                    )
                except ValueError:
                    prediction_kickoff_dt = None
                if (
                    prediction_home == home_id and prediction_away == away_id
                    and prediction_kickoff_dt is not None
                    and abs((prediction_kickoff_dt - kickoff).total_seconds()) <= 3 * 3600
                ):
                    priority, reason = 3, "teams_and_kickoff"
            if priority is not None:
                # Authentic predictions always beat backfills for the same
                # fixture, regardless of which identity field matched.
                candidates.append((int(backfilled), priority, timestamp, prediction, reason))
        if not candidates:
            return None
        _backfilled, priority, _timestamp, prediction, reason = min(
            candidates, key=lambda item: (item[0], item[1], -item[2].timestamp())
        )
        LOGGER.info(
            "Resolved knockout prediction: reason=%s matches.id=%s "
            "prediction.match_id=%s api_fixture=%s prediction_provider_fixture=%s "
            "canonical=%s prediction_canonical=%s match_number=%s teams=%s_vs_%s",
            reason, match_id, prediction.get("match_id"), current_provider,
            prediction.get("provider_fixture_id"), current_canonical,
            prediction.get("canonical_match_id"), number, home_id, away_id,
        )
        return prediction

    def database_match_payloads(
        self,
        rows: list[dict[str, Any]] | None = None,
        prediction_run: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        rows = rows if rows is not None else self.current_match_rows()
        prediction_run = prediction_run or self.current_prediction_run()
        payloads = []
        seen_ids = {fixture.id for fixture in self.fixtures}
        for row in self._official_knockout_rows(rows):
            stage = row["_knockout_stage"]
            home_id = row["_home_id"]
            away_id = row["_away_id"]
            raw_id = (
                row.get("id")
                or row.get("api_football_fixture_id")
                or row.get("provider_fixture_id")
                or row.get("canonical_match_id")
                or row.get("match_id")
            )
            if raw_id is None:
                raw_id = f"provider-knockout-{stage}-{row['_official_match_number'] or row['_knockout_index']}"
            match_id = str(raw_id)
            if match_id in seen_ids:
                continue
            seen_ids.add(match_id)
            kickoff_dt = row["_kickoff_dt"]
            status = self._knockout_status(row)
            number = row["_official_match_number"]
            database_prediction = self._historical_knockout_prediction(
                row, match_id, kickoff_dt, home_id, away_id, prediction_run
            )
            payloads.append(
                {
                    "id": match_id,
                    "number": number if number is not None else 0,
                    "stage": stage,
                    "kickoff": kickoff_dt.isoformat(),
                    "venue_id": row.get("venue_id") or "TBD",
                    "group": None,
                    "home_team": self.team_payload(home_id),
                    "away_team": self.team_payload(away_id),
                    "home_slot": None,
                    "away_slot": None,
                    "status": status or "scheduled",
                    "home_score": (
                        int(row["home_score"])
                        if row.get("home_score") is not None
                        else None
                    ),
                    "away_score": (
                        int(row["away_score"])
                        if row.get("away_score") is not None
                        else None
                    ),
                    "prediction": self.prediction_payload(
                        match_id, prediction_run, database_prediction,
                        allow_run_lookup=False,
                    ),
                }
            )
            if payloads[-1]["prediction"] is None:
                LOGGER.info(
                    "Showing real knockout match without prediction: id=%s stage=%s status=%s",
                    match_id,
                    stage,
                    payloads[-1]["status"],
                )
        return sorted(payloads, key=lambda match: (match["kickoff"], match["number"]))

    def match_payload(
        self,
        match_id: str,
        prediction_run: dict[str, Any] | None = None,
        match_results: dict[str, dict[str, Any]] | None = None,
    ) -> dict:
        match = next(match for match in self.fixtures if match.id == match_id)
        match_results = match_results or {}
        result = match_results.get(match.id, {})
        return {
            "id": match.id,
            "number": match.number,
            "stage": match.stage,
            "kickoff": match.kickoff.isoformat(),
            "venue_id": match.venue_id,
            "group": match.group,
            "home_team": self.team_payload(match.home_team_id),
            "away_team": self.team_payload(match.away_team_id),
            "home_slot": None,
            "away_slot": None,
            "status": result.get("status", "scheduled"),
            "home_score": result.get("home_score"),
            "away_score": result.get("away_score"),
            "prediction": self.prediction_payload(match.id, prediction_run),
        }

    def latest_predictions_payload(self, force: bool = False) -> dict[str, Any]:
        prediction_run = self.current_prediction_run(force=force)
        prediction_ids = (
            prediction_run["predictions"]
            if prediction_run["model_run_id"] is not None
            else self.static_predictions
        )
        return {
            "model_version": prediction_run["model_version"],
            "generated_at": str(prediction_run["generated_at"]),
            "data_cutoff": str(prediction_run["data_cutoff"]),
            "source": (
                "database_latest"
                if prediction_run["model_run_id"] is not None
                else "fallback_static"
            ),
            "predictions": [
                self.prediction_payload(match_id, prediction_run)
                for match_id in prediction_ids
            ],
        }

    def simulation_snapshots(self) -> list[dict[str, Any]]:
        if self.simulation_source is not None and hasattr(
            self.simulation_source, "list_snapshots"
        ):
            return self.simulation_source.list_snapshots()
        return [
            snapshot_payload(
                snapshot,
                snapshot.fallback_cutoff_at,
                cutoff_source="audited_fallback",
                available=False,
            )
            for snapshot in SNAPSHOTS
        ]

    def snapshot_simulation(self, snapshot_key: str) -> dict[str, Any] | None:
        snapshot = get_snapshot(snapshot_key)
        if self.simulation_source is None or not hasattr(
            self.simulation_source, "load_snapshot"
        ):
            return None
        database_simulation = self.simulation_source.load_snapshot(snapshot_key)
        if database_simulation is None:
            return None
        payload = self._database_simulation_payload(database_simulation)
        run = database_simulation["run"]
        cutoff = parse_utc(run.get("cutoff_at")) or snapshot.fallback_cutoff_at
        payload["snapshot"] = snapshot_payload(
            snapshot,
            cutoff,
        )
        payload["reconstruction_mode"] = (
            run.get("reconstruction_mode") or "retrospective_reconstruction"
        )
        payload["provenance"] = _json_value(run.get("provenance"), {})
        payload["source"] = "database_snapshot"
        active_team_ids = self._snapshot_active_team_ids(
            snapshot,
            database_simulation,
            payload["teams"],
        )
        payload["teams"] = [
            {**team, "is_active_at_snapshot": True}
            for team in payload["teams"]
            if team["team_id"] in active_team_ids
        ]
        return payload

    def _snapshot_active_team_ids(
        self,
        snapshot: Any,
        database_simulation: dict[str, Any],
        serialized_teams: list[dict[str, Any]],
    ) -> set[str]:
        """Resolve historical eligibility without relying on title probability."""
        available_team_ids = {team["team_id"] for team in serialized_teams}
        expected_count = SNAPSHOT_ACTIVE_TEAM_COUNTS[snapshot.key]
        if snapshot.key == "pre_tournament":
            return available_team_ids

        run = database_simulation["run"]
        provenance = _json_value(run.get("provenance"), {})
        provenance_ids = {
            str(team_id)
            for team_id in (
                provenance.get("active_team_ids", [])
                if isinstance(provenance, dict)
                else []
            )
            if str(team_id) in available_team_ids
        }
        if len(provenance_ids) == expected_count:
            return provenance_ids

        participant_ids = {
            team_id
            for row in self._official_knockout_rows(self.current_match_rows())
            if row["_knockout_stage"] == snapshot.stage
            for team_id in (row["_home_id"], row["_away_id"])
            if team_id in available_team_ids
        }
        if len(participant_ids) == expected_count:
            return participant_ids

        probability_field = SNAPSHOT_ADVANCEMENT_FIELDS[snapshot.key]
        probability_ids = {
            team["team_id"]
            for team in serialized_teams
            if float(team.get(probability_field) or 0) > 0
        }
        if len(probability_ids) != expected_count:
            LOGGER.warning(
                "Snapshot %s eligibility fallback returned %d teams; expected %d",
                snapshot.key,
                len(probability_ids),
                expected_count,
            )
        return probability_ids

    def _database_simulation_payload(
        self, database_simulation: dict[str, Any]
    ) -> dict[str, Any]:
        run = database_simulation["run"]
        model_inputs = database_simulation.get("model_inputs", {})
        generated_at = run.get("generated_at") or run.get("created_at")
        iterations = int(run.get("num_simulations") or run.get("iterations") or 0)
        teams = []
        for row in database_simulation["results"]:
            team_id = str(row["team_id"])
            team = self.teams_by_id.get(team_id)
            if team is None:
                LOGGER.warning(
                    "Skipping simulation result for unknown team %s", team_id
                )
                continue
            teams.append(
                {
                    "team_id": team_id,
                    "team_name": team.name,
                    "flag": flag_for_team(team_id),
                    "group": team.group,
                    "group_stage_exit": float(
                        row.get("group_stage_exit_probability")
                        or row.get("group_stage_exit")
                        or 0
                    ),
                    "round_of_32": float(
                        row.get("round_of_32_probability")
                        or row.get("round_of_32")
                        or 0
                    ),
                    "round_of_16": float(
                        row.get("round_of_16_probability")
                        or row.get("round_of_16")
                        or 0
                    ),
                    "quarterfinal": float(
                        row.get("quarterfinal_probability")
                        or row.get("quarterfinal")
                        or 0
                    ),
                    "semifinal": float(
                        row.get("semifinal_probability")
                        or row.get("semifinal")
                        or 0
                    ),
                    "final": float(
                        row.get("final_probability") or row.get("final") or 0
                    ),
                    "champion": float(
                        row.get("champion_probability") or row.get("champion") or 0
                    ),
                    "model_inputs": model_inputs.get(team_id),
                }
            )
        return {
            "iterations": iterations,
            "seed": int(run.get("random_seed") or run.get("seed") or 2026),
            "model_version": run.get("model_version") or "unknown",
            "generated_at": str(generated_at),
            "created_at": str(run.get("created_at") or generated_at),
            "data_cutoff": str(
                run.get("cutoff_at") or run.get("data_cutoff") or generated_at
            ),
            "source": "database_latest",
            "monte_carlo_precision": {
                "worst_case_standard_error": (
                    (0.25 / iterations) ** 0.5 if iterations else 0.0
                ),
                "worst_case_95_margin": (
                    1.96 * (0.25 / iterations) ** 0.5 if iterations else 0.0
                ),
            },
            "teams": teams,
        }

    def latest_simulation(self) -> dict:
        if self.simulation_source is not None:
            try:
                database_simulation = self.simulation_source.load_latest()
            except Exception:
                LOGGER.warning(
                    "Latest database simulation is unavailable; serving static "
                    "simulation fallback",
                    exc_info=True,
                )
            else:
                if database_simulation is not None:
                    return self._database_simulation_payload(database_simulation)

        if self._simulation is None:
            snapshot_path = ROOT / "data" / "generated" / "latest.json"
            if snapshot_path.exists():
                snapshot = json.loads(snapshot_path.read_text())
                simulation = snapshot.get("simulation", {})
                if (
                    snapshot.get("model_version") == MODEL_VERSION
                    and simulation.get("iterations") == PUBLISHED_SIMULATION_ITERATIONS
                ):
                    self._simulation = simulation
                    self._simulation_generated_at = snapshot["generated_at"]
                    self._simulation_data_cutoff = snapshot["data_cutoff"]
            if self._simulation is None:
                self._simulation = simulate_tournament(
                    iterations=PUBLISHED_SIMULATION_ITERATIONS,
                    seed=2026,
                    context_repository=self.contexts,
                    cutoff=datetime.fromisoformat(self.data_cutoff),
                )
        return {
            **self._simulation,
            "model_version": MODEL_VERSION,
            "generated_at": self._simulation_generated_at,
            "created_at": self._simulation_generated_at,
            "data_cutoff": self._simulation_data_cutoff,
            "source": "fallback_static",
        }

    def team_profile_payload(self, team_id: str) -> dict:
        team = self.teams_by_id[team_id]
        team_names = {item.id: item.name for item in self.teams}
        cutoff = datetime.fromisoformat(self.data_cutoff)
        recent = recent_results(
            team_id,
            self.contexts.results,
            team_names,
            cutoff,
        )
        players = key_players(team_id, self.squad_players)
        probability = next(
            row
            for row in self.latest_simulation()["teams"]
            if row["team_id"] == team_id
        )
        matches = [
            self.match_payload(match.id, self.current_prediction_run())
            for match in self.fixtures
            if team_id in (match.home_team_id, match.away_team_id)
        ]
        group_path = []
        for match in matches:
            prediction = match["prediction"]
            if prediction is None:
                continue
            is_home = match["home_team"]["id"] == team_id
            opponent = match["away_team"] if is_home else match["home_team"]
            group_path.append(
                {
                    "match_id": match["id"],
                    "opponent_id": opponent["id"],
                    "opponent_name": opponent["name"],
                    "kickoff": match["kickoff"],
                    "team_win_probability": prediction["probabilities"][
                        "home_win" if is_home else "away_win"
                    ],
                    "draw_probability": prediction["probabilities"]["draw"],
                    "opponent_win_probability": prediction["probabilities"][
                        "away_win" if is_home else "home_win"
                    ],
                }
            )
        analysis = team_analysis(
            team,
            probability,
            recent,
            players,
            group_path,
            "experimental",
        )
        return {
            **self.team_payload(team_id),
            "matches": matches,
            "group_path": group_path,
            "tournament_probability": probability,
            "recent_results": recent,
            "form_summary": form_summary(recent),
            "key_players": player_payload(players),
            "analysis": analysis,
            "player_data_source": self.squad_metadata,
            "results_data_cutoff": self.data_cutoff,
        }


service = PredictionService()
