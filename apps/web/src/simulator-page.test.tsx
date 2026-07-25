import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderToString } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { Simulator } from "./pages/Simulator";
import type { Simulation, SimulationSnapshot } from "./types";

const snapshots: SimulationSnapshot[] = [
  ["pre_tournament", "Before Tournament"],
  ["pre_round_of_32", "Before Round of 32"],
  ["pre_round_of_16", "Before Round of 16"],
  ["pre_quarterfinals", "Before Quarterfinals"],
  ["pre_semifinals", "Before Semifinals"],
  ["pre_final", "Before Final"],
].map(([key, label], index) => ({
  key: key as SimulationSnapshot["key"],
  label,
  stage: key,
  cutoff_at: `2026-07-${String(index + 1).padStart(2, "0")}T17:00:00Z`,
  sort_order: index + 1,
  description: `${label} description`,
  available: key !== "pre_semifinals",
}));

function simulation(
  key: SimulationSnapshot["key"],
  teamName = "Spain",
): Simulation {
  return {
    iterations: 50_000,
    seed: 2026,
    model_version: "elo-context-v4.2.1",
    generated_at: "2026-07-20T00:00:00Z",
    created_at: "2026-07-20T00:00:00Z",
    data_cutoff: "2026-07-04T17:00:00Z",
    source: "database_snapshot",
    snapshot: snapshots.find((snapshot) => snapshot.key === key),
    reconstruction_mode: "retrospective_reconstruction",
    monte_carlo_precision: {
      worst_case_standard_error: 0.002,
      worst_case_95_margin: 0.004,
    },
    teams: [
      {
        team_id: "ESP",
        team_name: teamName,
        round_of_32: 0.8,
        round_of_16: 0.6,
        quarterfinal: 0.4,
        semifinal: 0.3,
        final: 0.2,
        champion: 0.187,
      },
    ],
  };
}

function renderPage(
  entry: string,
  simulations: Partial<Record<SimulationSnapshot["key"], Simulation>> = {},
  errorKey?: SimulationSnapshot["key"],
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { staleTime: Infinity, retry: false, refetchOnMount: false },
    },
  });
  queryClient.setQueryData(["simulation-snapshots"], snapshots);
  for (const [key, value] of Object.entries(simulations)) {
    queryClient.setQueryData(["historical-simulation", key], value);
  }
  if (errorKey) {
    const query = queryClient.getQueryCache().build<Simulation, Error>(
      queryClient,
      {
        queryKey: ["historical-simulation", errorKey],
        queryFn: () => Promise.reject(new Error("not generated")),
      },
    );
    query.setState({
      ...query.state,
      error: new Error("not generated"),
      errorUpdatedAt: Date.now(),
      fetchStatus: "idle",
      status: "error",
    });
  }
  return renderToString(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[entry]}>
        <Simulator />
      </MemoryRouter>
    </QueryClientProvider>,
  ).replaceAll("<!-- -->", "");
}

describe("Historical simulation explorer", () => {
  it("renders selector options in canonical order with an accessible label", () => {
    const html = renderPage("/simulations", {
      pre_tournament: simulation("pre_tournament"),
    });

    expect(html).toContain('for="simulation-snapshot"');
    expect(html).toContain('id="simulation-snapshot"');
    let previous = -1;
    for (const snapshot of snapshots) {
      const position = html.indexOf(snapshot.label);
      expect(position).toBeGreaterThan(previous);
      previous = position;
    }
  });

  it("defaults invalid URLs to Before Tournament and renders metadata", () => {
    const html = renderPage("/simulations?snapshot=invalid", {
      pre_tournament: simulation("pre_tournament"),
    });

    expect(html).toContain(
      '<option value="pre_tournament" selected="">Before Tournament</option>',
    );
    expect(html).toContain("elo-context-v4.2.1");
    expect(html).toContain("50,000");
    expect(html).toContain("Cutoff-safe historical reconstruction");
  });

  it("uses the URL-selected snapshot and its probabilities", () => {
    const html = renderPage("/simulations?snapshot=pre_round_of_16", {
      pre_round_of_16: simulation("pre_round_of_16", "Spain at R16"),
    });

    expect(html).toContain("Spain at R16");
    expect(html).toContain("19%");
    expect(html).toContain(
      '<option value="pre_round_of_16" selected="">Before Round of 16</option>',
    );
  });

  it("shows loading without presenting the previous snapshot as the new one", () => {
    const html = renderPage("/simulations?snapshot=pre_final", {
      pre_tournament: simulation("pre_tournament", "Old snapshot team"),
    });

    expect(html).toContain("Loading Before Final forecast");
    expect(html).not.toContain("Old snapshot team");
  });

  it("shows an informative unavailable state", () => {
    const html = renderPage(
      "/simulations?snapshot=pre_semifinals",
      {},
      "pre_semifinals",
    );

    expect(html).toContain("Before Semifinals forecast unavailable");
    expect(html).toContain("has not been generated yet");
  });
});
