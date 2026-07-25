import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

export function Methodology() {
  const performance = useQuery({
    queryKey: ["model-performance"],
    queryFn: api.performance,
  });
  const report = performance.data;

  return (
    <section className="prose">
      <span className="eyebrow">Model card</span>
      <h1>Transparent by design</h1>
      <p className="lead">
        {report?.readiness.message ??
          "A valid historical-readiness artifact is unavailable."}
      </p>
      {report && (
        <div className={`notice ${report.readiness.historical_gate_passed ? "" : "warning"}`}>
          <strong>Historical production gate:</strong>{" "}
          {report.readiness.historical_gate_passed ? "passed" : "not passed"}.{" "}
          {report.readiness.historical_gate_version &&
            `${report.readiness.historical_gate_version} · ${report.readiness.historical_fold_count ?? 0} folds · ${report.readiness.historical_match_count ?? 0} matches. `}
          <strong>Prospective 2026 scorecard:</strong>{" "}
          {report.readiness.prospective_match_count} eligible pre-kickoff predictions
          ({report.readiness.prospective_status}). These are separate statuses.
          {report.readiness.historical_disclosure && ` ${report.readiness.historical_disclosure}`}
        </div>
      )}
      <h2>How a match is modeled</h2>
      <p>
        FIFA ranking positions are transformed into an Elo-equivalent strength
        prior. The rating gap, neutral-site scoring environment, and host status
        produce expected goals for each team. Independent Poisson distributions
        generate the score matrix and result probabilities.
      </p>
      <p>
        Fixture ordering does not create home advantage. Only Mexico, Canada,
        and the United States receive a host-country adjustment, including when
        they are listed as the away team. The current adjustment is a
        conservative 50 Elo-equivalent points.
      </p>
      <h2>Context data</h2>
      <p>
        Recent results use four years of time-decayed international matches,
        with competitive fixtures weighted above friendlies. Head-to-head
        history uses an eight-year window but is heavily shrunk and capped so a
        handful of old meetings cannot dominate the forecast.
      </p>
      <p>
        Squad and injury adjustments only use reports with a source URL,
        publication time, effective window, confidence, and player-importance
        estimate. A newer report for the same player replaces an older one.
      </p>
      <h2>Chronological evaluation</h2>
      {report ? (
        <>
          <div className="metric-grid">
            {(["equal", "elo", "context"] as const).map((model) => (
              <article className="metric-card" key={model}>
                <span>{model === "equal" ? "Equal probability" : model}</span>
                <strong>{report.aggregate[model].log_loss.toFixed(3)}</strong>
                <small>Log loss · {report.aggregate[model].matches} matches</small>
                <small>
                  Brier {report.aggregate[model].brier_score.toFixed(3)} · ECE{" "}
                  {report.aggregate[model].expected_calibration_error.toFixed(3)}
                </small>
              </article>
            ))}
          </div>
          <div className={`notice ${report.promotion_gate.status === "fail" ? "warning" : ""}`}>
            {report.message} Evaluation covers {report.protocol.start_year} through{" "}
            {report.protocol.end_date}; same-day results are updated only after
            all predictions for that date.
            {!report.readiness.ready && report.readiness.failed_conditions.length > 0
              ? ` Main blocker: ${report.readiness.failed_conditions[0]}`
              : ""}
          </div>
        </>
      ) : (
        <p>Loading the latest evaluation report...</p>
      )}
      <h2>How the tournament is modeled</h2>
      <p>
        Every group score is sampled, tables are ranked by points, goal
        difference, goals scored, fair play, and deterministic lots. The top two
        teams plus the eight best third-place teams enter knockout simulation.
      </p>
      <h2>Known limitations</h2>
      <ul>
        <li>Availability is only as current as the sourced report feed.</li>
        <li>No projected lineups, travel, rest, or weather inputs yet.</li>
        <li>Rank-derived ratings are a launch prior, not a historical Elo feed.</li>
        <li>Official Annex C third-place pairing mapping is not imported yet.</li>
        <li>Context features have not yet beaten walk-forward Elo on aggregate.</li>
        <li>No player props or market comparison is shown.</li>
      </ul>
      <div className="notice">
        Predictions are educational analytics. The app does not recommend bets,
        stake sizes, parlays, or wagering actions.
      </div>
    </section>
  );
}
