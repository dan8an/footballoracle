-- Persisted, cutoff-conditioned tournament simulation snapshots.
-- Existing current-state simulation rows remain valid with snapshot_key null.

alter table public.simulation_runs
  add column if not exists snapshot_key text,
  add column if not exists cutoff_at timestamptz,
  add column if not exists reconstruction_mode text,
  add column if not exists simulation_config_version text,
  add column if not exists status text not null default 'completed',
  add column if not exists provenance jsonb not null default '{}'::jsonb;

alter table public.simulation_runs
  drop constraint if exists simulation_runs_snapshot_key_check;
alter table public.simulation_runs
  add constraint simulation_runs_snapshot_key_check check (
    snapshot_key is null or snapshot_key in (
      'pre_tournament',
      'pre_round_of_32',
      'pre_round_of_16',
      'pre_quarterfinals',
      'pre_semifinals',
      'pre_final'
    )
  );

alter table public.simulation_runs
  drop constraint if exists simulation_runs_reconstruction_mode_check;
alter table public.simulation_runs
  add constraint simulation_runs_reconstruction_mode_check check (
    reconstruction_mode is null or reconstruction_mode in (
      'historical_exact',
      'historical_prediction_snapshot',
      'retrospective_reconstruction'
    )
  );

create unique index if not exists simulation_runs_snapshot_identity_uidx
  on public.simulation_runs (
    snapshot_key,
    model_version,
    simulation_config_version
  )
  where snapshot_key is not null;

create index if not exists simulation_runs_snapshot_available_idx
  on public.simulation_runs (snapshot_key, status, created_at desc)
  where snapshot_key is not null;

comment on column public.simulation_runs.cutoff_at is
  'Exclusive UTC cutoff: results and prediction inputs at or after it are ineligible.';
comment on column public.simulation_runs.reconstruction_mode is
  'Historical integrity level; never infer historical_exact from a retrospective rebuild.';
comment on column public.simulation_runs.provenance is
  'Input coverage, cutoff source, fallback usage, and other reproducibility metadata.';
