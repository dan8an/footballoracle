import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Loading, percent } from "../components";
import { rankSimulationTeams } from "../simulation-display";
import type { SimulationSnapshot, SimulationTeam } from "../types";

const DEFAULT_SNAPSHOT: SimulationSnapshot["key"] = "pre_tournament";

const reconstructionLabels: Record<
  NonNullable<import("../types").Simulation["reconstruction_mode"]>,
  string
> = {
  historical_exact: "Exact point-in-time inputs",
  historical_prediction_snapshot: "Archived pre-stage predictions",
  retrospective_reconstruction: "Cutoff-safe historical reconstruction",
};

const probabilityColumns: {
  key: keyof Pick<
    SimulationTeam,
    | "round_of_32"
    | "round_of_16"
    | "quarterfinal"
    | "semifinal"
    | "final"
    | "champion"
  >;
  label: string;
}[] = [
  { key: "round_of_32", label: "Round of 32" },
  { key: "round_of_16", label: "Round of 16" },
  { key: "quarterfinal", label: "Quarterfinal" },
  { key: "semifinal", label: "Semifinal" },
  { key: "final", label: "Final" },
  { key: "champion", label: "Champion" },
];

function formatCutoff(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(new Date(value));
}

export function Simulator() {
  const [searchParams, setSearchParams] = useSearchParams();
  const snapshotsQuery = useQuery({
    queryKey: ["simulation-snapshots"],
    queryFn: api.simulationSnapshots,
  });
  const snapshots = [...(snapshotsQuery.data ?? [])].sort(
    (left, right) => left.sort_order - right.sort_order,
  );
  const requestedSnapshot = searchParams.get("snapshot");
  const selectedSnapshot =
    snapshots.find((snapshot) => snapshot.key === requestedSnapshot) ??
    snapshots.find((snapshot) => snapshot.key === DEFAULT_SNAPSHOT) ??
    snapshots[0];

  useEffect(() => {
    if (!selectedSnapshot || requestedSnapshot === selectedSnapshot.key) return;
    setSearchParams({ snapshot: selectedSnapshot.key }, { replace: true });
  }, [requestedSnapshot, selectedSnapshot, setSearchParams]);

  const simulationQuery = useQuery({
    queryKey: ["historical-simulation", selectedSnapshot?.key],
    queryFn: () => api.historicalSimulation(selectedSnapshot!.key),
    enabled: Boolean(selectedSnapshot && selectedSnapshot.available !== false),
  });

  if (snapshotsQuery.isLoading) {
    return <Loading label="Loading historical snapshots" />;
  }
  if (snapshotsQuery.isError || !selectedSnapshot) {
    return (
      <section className="empty-state" role="alert">
        <h1>Historical forecasts are unavailable</h1>
        <p>The snapshot catalog could not be loaded. Please try again shortly.</p>
      </section>
    );
  }

  const isLoadingSelection =
    simulationQuery.isLoading ||
    (simulationQuery.isFetching && !simulationQuery.data);
  const data = simulationQuery.data;
  const rankedTeams = data ? rankSimulationTeams(data.teams) : [];
  const visibleColumns = probabilityColumns.filter(({ key }) =>
    rankedTeams.some((team) => typeof team[key] === "number"),
  );

  return (
    <section>
      <div className="page-heading">
        <span className="eyebrow">Historical tournament forecast</span>
        <h1>Tournament probabilities</h1>
        <p>
          These probabilities show what the model believed before the selected
          stage began. Results after that cutoff are not included.
        </p>
      </div>

      <div className="snapshot-controls">
        <label htmlFor="simulation-snapshot">Tournament probabilities</label>
        <select
          id="simulation-snapshot"
          value={selectedSnapshot.key}
          onChange={(event) =>
            setSearchParams({ snapshot: event.target.value })
          }
        >
          {snapshots.map((snapshot) => (
            <option key={snapshot.key} value={snapshot.key}>
              {snapshot.label}
            </option>
          ))}
        </select>
        <p>{selectedSnapshot.description}</p>
      </div>

      {simulationQuery.isError || (!isLoadingSelection && !data) ? (
        <div className="empty-state" role="status">
          <h2>{selectedSnapshot.label} forecast unavailable</h2>
          <p>
            This cutoff-conditioned simulation has not been generated yet. Run
            the historical snapshot backfill, then refresh this page.
          </p>
        </div>
      ) : isLoadingSelection ? (
        <Loading label={`Loading ${selectedSnapshot.label} forecast`} />
      ) : data ? (
        <>
          <dl className="snapshot-metadata">
            <div>
              <dt>Cutoff</dt>
              <dd>{formatCutoff(data.snapshot?.cutoff_at ?? data.data_cutoff)}</dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd>{data.model_version}</dd>
            </div>
            <div>
              <dt>Simulations</dt>
              <dd>{data.iterations.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Historical basis</dt>
              <dd>
                {data.reconstruction_mode
                  ? reconstructionLabels[data.reconstruction_mode]
                  : "Archived historical simulation"}
              </dd>
            </div>
          </dl>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Team</th>
                  {visibleColumns.map((column) => (
                    <th key={column.key}>{column.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rankedTeams.map((team) => (
                  <tr key={team.team_id}>
                    <td>
                      <strong>
                        {team.flag ? `${team.flag} ` : ""}
                        {team.team_name}
                      </strong>
                    </td>
                    {visibleColumns.map((column) => (
                      <td key={column.key}>
                        {column.key === "champion" ? (
                          <b>{percent(team[column.key])}</b>
                        ) : (
                          percent(team[column.key])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </section>
  );
}
