export type Team = {
  id: string;
  name: string;
  group: string;
  position: number;
  rank: number;
  host: boolean;
  elo: number;
  flag: string;
  confederation?: string;
};

export type TeamProfile = Team & {
  matches: Match[];
  tournament_probability: SimulationTeam;
  group_path: {
    match_id: string;
    opponent_id: string;
    opponent_name: string;
    kickoff: string;
    team_win_probability: number;
    draw_probability: number;
    opponent_win_probability: number;
  }[];
  recent_results: {
    played_on: string;
    opponent_id: string;
    opponent_name: string;
    goals_for: number;
    goals_against: number;
    outcome: "W" | "D" | "L";
    tournament: string;
    neutral: boolean;
  }[];
  form_summary: {
    matches: number;
    wins: number;
    draws: number;
    losses: number;
    goals_for: number;
    goals_against: number;
    goal_difference: number;
    points_per_match: number;
  };
  key_players: {
    name: string;
    position: string;
    club: string;
    caps: number;
    goals: number;
    age: number | null;
    why_key: string;
  }[];
  analysis: {
    headline: string;
    overview: string;
    form: string;
    path: string;
    personnel: string;
    objectives: string[];
    method: string;
  };
  player_data_source: {
    source_url?: string;
    source_name?: string;
    retrieved_at?: string;
    player_count?: number;
    team_count?: number;
    note?: string;
  };
  results_data_cutoff: string;
};

export type ScoreProbability = {
  home: number;
  away: number;
  probability: number;
};

export type Prediction = {
  match_id: string;
  home_team_id: string;
  away_team_id: string;
  home_xg: number;
  away_xg: number;
  probabilities: {
    home_win: number;
    draw: number;
    away_win: number;
  };
  elo_base_home_probability?: number | null;
  elo_base_draw_probability?: number | null;
  elo_base_away_probability?: number | null;
  attack_defense_adjustment?: number | null;
  draw_calibration_adjustment?: number | null;
  context_adjustment_total?: number | null;
  final_home_probability?: number;
  final_draw_probability?: number;
  final_away_probability?: number;
  confidence_score?: number | null;
  confidence_tier?: "Very High" | "High" | "Medium" | "Low" | null;
  confidence_explanation?: string | null;
  top_factors?: {
    factor: string;
    team: string;
    impact: string;
  }[];
  top_scores: ScoreProbability[];
  confidence: string;
  key_factors: string[];
  context: {
    home_form_elo: number;
    away_form_elo: number;
    home_h2h_elo: number;
    away_h2h_elo: number;
    home_availability_elo: number;
    away_availability_elo: number;
    historical_matches_home: number;
    historical_matches_away: number;
    h2h_matches: number;
    availability_reports: number;
    data_cutoff: string | null;
  };
  model_version: string;
  generated_at: string;
  data_cutoff: string;
  generation_mode?: "standard" | "historical_backfill";
  historical_cutoff?: string | null;
  backfilled_at?: string | null;
};

export type Match = {
  id: string;
  number: number;
  stage: string;
  kickoff: string;
  venue_id: string;
  group: string | null;
  home_team: Team | null;
  away_team: Team | null;
  home_slot: string | null;
  away_slot: string | null;
  status: string;
  home_score: number | null;
  away_score: number | null;
  prediction: Prediction | null;
};

export type SimulationTeam = {
  team_id: string;
  team_name: string;
  is_active_at_snapshot?: boolean;
  flag?: string;
  group?: string;
  confederation?: string;
  round_of_32: number;
  round_of_16: number;
  quarterfinal: number;
  semifinal: number;
  final: number;
  champion: number;
  model_inputs?: {
    elo_rating: number;
    elo_rank: number;
    attack_rating: number;
    defense_rating: number;
    shot_volume_rating: number | null;
    rating_source: "database_current" | "canonical_rank_prior" | string;
    rating_matches: number;
    shot_volume_sample_matches: number | null;
  } | null;
};

export type Simulation = {
  iterations: number;
  seed: number;
  model_version: string;
  generated_at: string;
  created_at: string;
  data_cutoff: string;
  source: "database_latest" | "database_snapshot" | "fallback_static";
  snapshot?: SimulationSnapshot;
  reconstruction_mode?:
    | "historical_exact"
    | "historical_prediction_snapshot"
    | "retrospective_reconstruction";
  provenance?: Record<string, unknown>;
  monte_carlo_precision: {
    worst_case_standard_error: number;
    worst_case_95_margin: number;
  };
  teams: SimulationTeam[];
};

export type SimulationSnapshot = {
  key:
    | "pre_tournament"
    | "pre_round_of_32"
    | "pre_round_of_16"
    | "pre_quarterfinals"
    | "pre_semifinals"
    | "pre_final";
  label: string;
  stage: string;
  cutoff_at: string;
  sort_order: number;
  description: string;
  cutoff_source?: "database_schedule" | "audited_fallback";
  available?: boolean;
};

export type ModelMetrics = {
  matches: number;
  log_loss: number;
  brier_score: number;
  ranked_probability_score: number;
  accuracy: number;
  expected_calibration_error: number;
  calibration_bins: {
    lower: number;
    upper: number;
    count: number;
    mean_probability: number;
    observed_rate: number;
  }[];
};

export type ModelPerformance = {
  status: string;
  message: string;
  readiness: {
    status: "pass" | "fail";
    ready: boolean;
    message: string;
    failed_conditions: string[];
    historical_gate_passed: boolean;
    historical_gate_version?: string;
    historical_match_count?: number;
    historical_fold_count?: number;
    historical_disclosure?: string;
    prospective_status: string;
    prospective_match_count: number;
    production_model_version: string;
    candidate_model_version: string | null;
  };
  generated_at: string;
  protocol: {
    start_year: number;
    end_date: string;
    minimum_prior_matches_per_team: number;
    same_day_updates: string;
    rank_data: string;
    availability_data: string;
  };
  aggregate: Record<"equal" | "elo" | "context", ModelMetrics>;
  years: Record<string, Record<"equal" | "elo" | "context", ModelMetrics>>;
  promotion_gate: {
    context_beats_elo_on_log_loss_and_brier: boolean;
    status: "pass" | "fail";
  };
};
