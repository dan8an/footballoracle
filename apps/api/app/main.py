import os
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import MatchResponse, SimulationRequest, TeamResponse
from .service import STATIC_MODEL_VERSION, service
from modeling.src.simulation import simulate_tournament
from modeling.src.readiness import load_readiness

READINESS_PATH = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "elo_context_v44_gate_v2_readiness.json"
PROSPECTIVE_SCORECARD_PATH = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "wc26_prospective_scorecard.json"

# Local development, production, and optional deployment-specific frontend origins.
allowed_origins = [
    "https://footballoracle.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
if web_origin := os.getenv("WEB_ORIGIN"):
    if web_origin not in allowed_origins:
        allowed_origins.append(web_origin)

app = FastAPI(
    title="2026 World Cup Prediction API",
    version="0.1.0",
    description="Educational match and tournament probabilities. Not betting advice.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": service.current_prediction_run()["model_version"],
    }


@app.get("/v1/tournament")
def tournament() -> dict:
    return {
        "id": "FIFA-WC-2026",
        "name": "2026 FIFA World Cup",
        "starts_on": "2026-06-11",
        "ends_on": "2026-07-19",
        "team_count": len(service.teams),
        "match_count": len(service.fixtures),
        "groups": list("ABCDEFGHIJKL"),
        "venues": service.venues,
    }


@app.get("/api/teams", response_model=list[TeamResponse], include_in_schema=False)
@app.get("/v1/teams", response_model=list[TeamResponse])
def teams() -> list[dict]:
    return [service.team_payload(team.id) for team in service.teams]


@app.get("/v1/teams/{team_id}")
def team(team_id: str) -> dict:
    if team_id not in service.teams_by_id:
        raise HTTPException(status_code=404, detail="Team not found")
    return service.team_profile_payload(team_id)


@app.get("/api/matches", response_model=list[MatchResponse], include_in_schema=False)
@app.get("/v1/matches", response_model=list[MatchResponse])
def matches(
    stage: str | None = Query(default=None),
    group: str | None = Query(default=None, min_length=1, max_length=1),
    team_id: str | None = Query(default=None),
) -> list[dict]:
    rows = service.current_match_rows()
    fixtures = service.fixtures
    if stage:
        fixtures = [match for match in fixtures if match.stage == stage]
    if group:
        fixtures = [match for match in fixtures if match.group == group.upper()]
    prediction_run = service.current_prediction_run()
    match_results = service.current_match_results(rows)
    payload = [
        service.match_payload(match.id, prediction_run, match_results)
        for match in fixtures
    ]
    if not group:
        payload.extend(service.database_match_payloads(rows, prediction_run))
    if stage:
        payload = [match for match in payload if match["stage"] == stage]
    if team_id:
        payload = [
            match
            for match in payload
            if team_id
            in (
                (match["home_team"] or {}).get("id"),
                (match["away_team"] or {}).get("id"),
            )
        ]
    return sorted(payload, key=lambda match: (match["kickoff"], match["number"]))


@app.get(
    "/api/matches/{match_id}",
    response_model=MatchResponse,
    include_in_schema=False,
)
@app.get("/v1/matches/{match_id}", response_model=MatchResponse)
def match(match_id: str) -> dict:
    rows = service.current_match_rows()
    if any(fixture.id == match_id for fixture in service.fixtures):
        return service.match_payload(
            match_id,
            match_results=service.current_match_results(rows),
        )
    prediction_run = service.current_prediction_run()
    database_match = next(
        (
            item
            for item in service.database_match_payloads(rows, prediction_run)
            if item["id"] == match_id
        ),
        None,
    )
    if database_match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return database_match


@app.get("/v1/predictions/latest")
def predictions() -> dict:
    return service.latest_predictions_payload()


@app.get("/v1/simulations/snapshots")
def simulation_snapshots() -> list[dict]:
    return service.simulation_snapshots()


@app.get("/v1/simulations")
def historical_simulation(snapshot: str = Query(...)) -> dict:
    try:
        payload = service.snapshot_simulation(snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "snapshot_unavailable",
                "snapshot": snapshot,
                "message": "This historical simulation snapshot has not been generated.",
            },
        )
    return payload


@app.get("/api/simulations/latest", include_in_schema=False)
@app.get("/v1/simulations/latest")
def latest_simulation() -> dict:
    return service.latest_simulation()


@app.post("/v1/simulations/custom")
def custom_simulation(request: SimulationRequest) -> dict:
    return {
        **simulate_tournament(
            iterations=request.iterations,
            seed=request.seed,
            context_repository=service.contexts,
            cutoff=datetime.fromisoformat(service.data_cutoff),
        ),
        "model_version": STATIC_MODEL_VERSION,
        "generated_at": service.generated_at,
        "data_cutoff": service.data_cutoff,
    }


@app.get("/v1/model/versions/current")
def model_version() -> dict:
    current_version = service.current_prediction_run()["model_version"]
    readiness = load_readiness(READINESS_PATH, current_version)
    return {
        "name": "Context-adjusted Poisson model",
        "semantic_version": current_version,
        "feature_schema_version": "1",
        "training_cutoff": None,
        "status": "production_calibrated" if readiness["ready"] else "experimental",
        "readiness": readiness,
    }


@app.get("/v1/model/performance")
def model_performance() -> dict:
    report_path = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "latest.json"
    if not report_path.exists():
        return {
            "status": "not_backtested",
            "message": "Run python -m modeling.src.evaluation.backtest.",
            "metrics": {},
        }
    report = json.loads(report_path.read_text())
    readiness = load_readiness(
        READINESS_PATH, service.current_prediction_run()["model_version"]
    )
    return {
        "status": "evaluated",
        "message": readiness["message"],
        "readiness": readiness,
        **report,
    }


@app.get("/v1/model/prospective-scorecard")
def prospective_scorecard() -> dict:
    try:
        payload = json.loads(PROSPECTIVE_SCORECARD_PATH.read_text())
        if not isinstance(payload.get("models"), dict) or "prospective_status" not in payload:
            raise ValueError("malformed scorecard")
        return payload
    except (OSError, ValueError, TypeError):
        return {"prospective_status": "unavailable", "models": {}, "paired_comparisons": [], "message": "A valid prospective tournament scorecard is unavailable."}


@app.get("/v1/data/status")
def data_status() -> dict:
    latest_result = (
        max(result.played_on for result in service.contexts.results).isoformat()
        if service.contexts.results
        else None
    )
    latest_report = (
        max(report.published_at for report in service.contexts.reports).isoformat()
        if service.contexts.reports
        else None
    )
    return {
        "historical_results": len(service.contexts.results),
        "latest_historical_result": latest_result,
        "availability_reports": len(service.contexts.reports),
        "latest_availability_report": latest_report,
        "squad_selections": len(service.contexts.squads),
        "latest_squad_selection": (
            max(selection.published_at for selection in service.contexts.squads).isoformat()
            if service.contexts.squads
            else None
        ),
        "data_cutoff": service.data_cutoff,
    }
