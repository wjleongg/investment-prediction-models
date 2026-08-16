-- =====================================================================
-- STATISTICAL ARBITRAGE PLATFORM — DATABASE CONTRACT
-- Target: Supabase / PostgreSQL 15+
--
-- This schema is the single contract between:
--   * the Python trading engine  (writer)
--   * the Streamlit frontend     (reader, plus config + commands)
--
-- Design rules encoded here:
--   1. No ticker is ever hard-coded. Pairs are rows in `pairs`;
--      every downstream table references pair_id.
--   2. `live_state` is a materialised snapshot, one row per pair,
--      UPSERTed by the engine. The Overview page reads it in ONE query.
--   3. Engine liveness is proven by `engine_heartbeat.heartbeat_at`,
--      never inferred from a stored status value.
--   4. Control commands carry a full lifecycle, not a boolean.
--   5. All statistical estimation is engine-side. The frontend reads
--      persisted results only.
--
-- Written for a fresh database. To rebuild, run teardown.sql first.
-- =====================================================================


-- =====================================================================
-- SECTION 1 — ENUMERATED TYPES
-- =====================================================================

create type system_status as enum (
  'STARTING', 'RUNNING', 'PAUSED', 'STOPPING', 'STOPPED', 'ERROR', 'KILLED'
);

create type data_source as enum (
  'LIVE_IBKR', 'SIMULATED_FEED', 'HISTORICAL_DATA', 'DISCONNECTED'
);

create type market_status as enum (
  'PRE_MARKET', 'MARKET_OPEN', 'AFTER_HOURS', 'MARKET_CLOSED', 'UNKNOWN'
);

create type signal_type as enum (
  'LONG_SPREAD', 'SHORT_SPREAD', 'EXIT', 'FLAT', 'NO_SIGNAL'
);

create type relationship_health as enum (
  'VALID', 'DEGRADED', 'INVALID'
);

create type position_side as enum (
  'LONG', 'SHORT', 'FLAT'
);

create type order_side as enum (
  'BUY', 'SELL'
);

create type order_type as enum (
  'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
);

create type order_status as enum (
  'PENDING', 'SUBMITTED', 'ACKNOWLEDGED', 'PARTIALLY_FILLED',
  'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED'
);

create type trade_direction as enum (
  'LONG_SPREAD', 'SHORT_SPREAD'
);

create type trade_status as enum (
  'OPEN', 'CLOSING', 'CLOSED', 'FAILED'
);

create type exit_reason as enum (
  'MEAN_REVERSION', 'STOP_LOSS', 'MAX_HOLDING_PERIOD',
  'RELATIONSHIP_INVALID', 'MANUAL', 'KILL_SWITCH', 'END_OF_DAY'
);

create type log_level as enum (
  'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
);

create type log_category as enum (
  'MARKET_DATA', 'MODEL', 'SIGNAL', 'ORDER', 'RISK', 'SYSTEM', 'CONTROL'
);

create type command_type as enum (
  'START', 'PAUSE', 'RESUME', 'RESTART', 'STOP',
  'CANCEL_ALL_ORDERS', 'FLATTEN_ALL_POSITIONS', 'KILL_SWITCH',
  'APPLY_CONFIG'
);

-- REQUESTED  : frontend wrote the row
-- RECEIVED   : engine has read it
-- ACKNOWLEDGED: engine validated it and intends to act
-- EXECUTED   : engine completed the action
-- Anything still REQUESTED past `expires_at` must be shown as UNCONFIRMED.
create type command_status as enum (
  'REQUESTED', 'RECEIVED', 'ACKNOWLEDGED', 'EXECUTED', 'FAILED', 'EXPIRED'
);

create type connection_state as enum (
  'CONNECTED', 'DISCONNECTED', 'RECONNECTING', 'ERROR', 'UNKNOWN'
);

create type config_status as enum (
  'CURRENT', 'PENDING', 'ARCHIVED', 'REJECTED'
);

create type cointegration_test as enum (
  'ENGLE_GRANGER', 'ADF', 'PHILLIPS_PERRON', 'JOHANSEN', 'KPSS'
);

create type price_field as enum (
  'LAST', 'BID', 'ASK', 'MID', 'CLOSE'
);


-- =====================================================================
-- SECTION 2 — PAIR REGISTRY
-- The only place tickers are named. Everything else uses pair_id.
-- =====================================================================

create table pairs (
  id               bigserial primary key,
  leg1_ticker      text        not null,
  leg2_ticker      text        not null,
  label            text        generated always as (leg1_ticker || ' / ' || leg2_ticker) stored,
  is_active        boolean     not null default false,
  created_at       timestamptz not null default now(),
  notes            text,
  constraint pairs_distinct_legs check (leg1_ticker <> leg2_ticker),
  constraint pairs_unique_legs   unique (leg1_ticker, leg2_ticker)
);

comment on table pairs is
  'Ticker pairs under management. SPY/IVV is seeded as a row, never hard-coded.';


-- =====================================================================
-- SECTION 3 — CONFIGURATION (versioned, with pending/current separation)
-- =====================================================================

create table strategy_config (
  id                          bigserial primary key,
  version                     integer      not null,
  status                      config_status not null default 'PENDING',
  pair_id                     bigint       not null references pairs(id),

  -- Lookbacks (in bars unless stated)
  historical_lookback         integer      not null,
  zscore_lookback             integer      not null,
  correlation_lookback        integer      not null,
  cointegration_lookback      integer      not null,

  -- Signal thresholds (absolute z-score values)
  entry_threshold             double precision not null,
  exit_threshold              double precision not null,
  stop_loss_threshold         double precision not null,
  max_holding_period_seconds  integer,

  -- Relationship validity gates
  min_correlation             double precision not null,
  max_cointegration_pvalue    double precision not null,

  -- Position / risk
  capital_allocation          numeric(20,2) not null,
  max_position_size           numeric(20,2) not null,
  max_pair_exposure           numeric(20,2) not null,
  max_leverage                double precision not null,
  max_simultaneous_positions  integer       not null,

  -- Kill-switch behaviour
  kill_switch_flattens        boolean      not null default true,

  -- Provenance
  created_at                  timestamptz  not null default now(),
  created_by                  text,
  applied_at                  timestamptz,
  applied_by                  text,
  requires_restart            boolean      not null default true,
  rejection_reason            text,

  constraint config_version_unique   unique (version),
  constraint config_thresholds_order check (exit_threshold < entry_threshold
                                            and entry_threshold < stop_loss_threshold),
  constraint config_pvalue_range     check (max_cointegration_pvalue > 0
                                            and max_cointegration_pvalue < 1),
  constraint config_correlation_range check (min_correlation >= -1 and min_correlation <= 1)
);

-- At most one CURRENT and one PENDING config at any time.
create unique index config_one_current on strategy_config (status)
  where status = 'CURRENT';
create unique index config_one_pending on strategy_config (status)
  where status = 'PENDING';

comment on table strategy_config is
  'Versioned config. Frontend writes PENDING rows only; the engine promotes to CURRENT.';


-- =====================================================================
-- SECTION 4 — ENGINE LIVENESS AND INFRASTRUCTURE HEALTH
-- =====================================================================

create table engine_heartbeat (
  engine_id        text         primary key,
  status           system_status not null,
  data_source      data_source   not null,
  market_status    market_status not null default 'UNKNOWN',
  heartbeat_at     timestamptz   not null,
  started_at       timestamptz   not null,
  engine_version   text,
  host             text,
  active_config_version integer,
  detail           text
);

comment on column engine_heartbeat.heartbeat_at is
  'Freshness of THIS column is the only valid proof of liveness. The frontend must compare it to now() and show ENGINE STALE past the agreed threshold, regardless of what the status column says.';

create table connection_status (
  component        text primary key,   -- ENGINE | BROKER | MARKET_DATA | DATABASE | ANALYTICS
  state            connection_state not null default 'UNKNOWN',
  last_ok_at       timestamptz,
  updated_at       timestamptz not null default now(),
  detail           text
);

create table system_metrics (
  engine_id             text primary key references engine_heartbeat(engine_id) on delete cascade,
  as_of                 timestamptz not null default now(),
  session_started_at    timestamptz,
  market_data_events    bigint not null default 0,
  model_calculations    bigint not null default 0,
  signals_generated     bigint not null default 0,
  orders_submitted      bigint not null default 0,
  orders_filled         bigint not null default 0,
  state_writes          bigint not null default 0,
  error_count           bigint not null default 0,
  warning_count         bigint not null default 0,
  last_market_data_at   timestamptz,
  last_model_calc_at    timestamptz,
  last_state_write_at   timestamptz,
  last_order_event_at   timestamptz
);


-- =====================================================================
-- SECTION 5 — LIVE STATE SNAPSHOT
-- One row per pair. UPSERTed by the engine on every model cycle.
-- The Overview page reads this (via v_overview) in a single round trip.
-- =====================================================================

create table live_state (
  pair_id                 bigint primary key references pairs(id) on delete cascade,
  as_of                   timestamptz not null,

  -- Market data
  leg1_price              numeric(18,6),
  leg2_price              numeric(18,6),
  leg1_bid                numeric(18,6),
  leg1_ask                numeric(18,6),
  leg2_bid                numeric(18,6),
  leg2_ask                numeric(18,6),
  leg1_mid                numeric(18,6),
  leg2_mid                numeric(18,6),
  last_market_data_at     timestamptz,

  -- Model state
  spread                  double precision,
  hedge_ratio             double precision,
  zscore                  double precision,
  spread_mean             double precision,
  spread_std              double precision,
  spread_volatility       double precision,
  correlation             double precision,
  rolling_correlation     double precision,
  cointegration_stat      double precision,
  cointegration_pvalue    double precision,
  half_life               double precision,
  health                  relationship_health not null default 'VALID',
  health_reason           text,
  last_model_update_at    timestamptz,

  -- Signal state
  current_signal          signal_type not null default 'NO_SIGNAL',
  signal_since            timestamptz,
  signal_explanation      text,
  current_position        position_side not null default 'FLAT',
  target_position         position_side not null default 'FLAT',

  -- Active thresholds (denormalised from config so charts can plot them
  -- without a join, and so historical rows keep the thresholds in force
  -- at the time)
  entry_threshold         double precision,
  exit_threshold          double precision,
  stop_loss_threshold     double precision,

  -- Portfolio aggregates
  total_pnl               numeric(20,4) not null default 0,
  daily_pnl               numeric(20,4) not null default 0,
  realized_pnl            numeric(20,4) not null default 0,
  unrealized_pnl          numeric(20,4) not null default 0,
  total_return_pct        double precision not null default 0,
  daily_return_pct        double precision not null default 0,
  current_exposure        numeric(20,4) not null default 0,
  capital_utilisation     double precision not null default 0,

  config_version          integer,
  updated_at              timestamptz not null default now()
);

comment on table live_state is
  'Materialised current state. Engine UPSERTs; frontend never writes here.';


-- =====================================================================
-- SECTION 6 — TIME SERIES
-- =====================================================================

create table market_data (
  id           bigserial primary key,
  pair_id      bigint      not null references pairs(id) on delete cascade,
  ticker       text        not null,
  ts           timestamptz not null,
  price        numeric(18,6) not null,
  bid          numeric(18,6),
  ask          numeric(18,6),
  volume       bigint,
  field        price_field not null default 'LAST',
  source       data_source not null
);

create index market_data_ticker_ts on market_data (ticker, ts desc);
create index market_data_pair_ts   on market_data (pair_id, ts desc);

-- Drives the Overview spread and z-score charts, and the Research page.
-- Thresholds are stored per row so a threshold change is visible on the chart.
create table model_state_history (
  id                    bigserial primary key,
  pair_id               bigint      not null references pairs(id) on delete cascade,
  ts                    timestamptz not null,
  leg1_price            numeric(18,6),
  leg2_price            numeric(18,6),
  spread                double precision not null,
  zscore                double precision not null,
  hedge_ratio           double precision,
  spread_mean           double precision,
  spread_std            double precision,
  correlation           double precision,
  cointegration_pvalue  double precision,
  half_life             double precision,
  entry_threshold       double precision,
  exit_threshold        double precision,
  stop_loss_threshold   double precision,
  health                relationship_health,
  signal                signal_type,
  config_version        integer
);

create index model_state_pair_ts on model_state_history (pair_id, ts desc);


-- =====================================================================
-- SECTION 7 — SIGNALS, ORDERS, FILLS, POSITIONS, TRADES
-- =====================================================================

create table signals (
  id               bigserial primary key,
  pair_id          bigint      not null references pairs(id) on delete cascade,
  ts               timestamptz not null,
  signal           signal_type not null,
  previous_signal  signal_type,
  zscore           double precision,
  hedge_ratio      double precision,
  health           relationship_health,
  explanation      text,          -- plain-English text rendered on the Strategy page
  acted_upon       boolean not null default false,
  suppressed_reason text,         -- populated when a signal was generated but not traded
  config_version   integer
);

create index signals_pair_ts on signals (pair_id, ts desc);

create table trades (
  id                     bigserial primary key,
  pair_id                bigint         not null references pairs(id),
  direction              trade_direction not null,
  status                 trade_status    not null default 'OPEN',

  entry_time             timestamptz    not null,
  exit_time              timestamptz,

  leg1_entry_price       numeric(18,6),
  leg2_entry_price       numeric(18,6),
  leg1_exit_price        numeric(18,6),
  leg2_exit_price        numeric(18,6),
  leg1_quantity          numeric(18,4),
  leg2_quantity          numeric(18,4),

  entry_zscore           double precision,
  exit_zscore            double precision,
  entry_hedge_ratio      double precision,
  exit_hedge_ratio       double precision,

  -- Full model snapshot at each boundary, for the Trade Detail panel
  entry_model_state      jsonb,
  exit_model_state       jsonb,

  gross_pnl              numeric(20,4),
  fees                   numeric(20,4) not null default 0,
  net_pnl                numeric(20,4),
  return_pct             double precision,
  holding_period_seconds integer,
  exit_reason            exit_reason,
  config_version         integer,

  constraint trades_exit_after_entry check (exit_time is null or exit_time >= entry_time)
);

create index trades_pair_entry on trades (pair_id, entry_time desc);
create index trades_status     on trades (status);

create table orders (
  id                bigserial primary key,
  broker_order_id   text,
  pair_id           bigint      not null references pairs(id),
  trade_id          bigint      references trades(id) on delete set null,
  ticker            text        not null,
  side              order_side  not null,
  quantity          numeric(18,4) not null,
  order_type        order_type  not null default 'MARKET',
  limit_price       numeric(18,6),
  status            order_status not null default 'PENDING',
  submitted_at      timestamptz not null default now(),
  acknowledged_at   timestamptz,
  filled_at         timestamptz,
  cancelled_at      timestamptz,
  filled_quantity   numeric(18,4) not null default 0,
  avg_fill_price    numeric(18,6),
  error_message     text
);

create index orders_pair_submitted on orders (pair_id, submitted_at desc);
create index orders_trade          on orders (trade_id);

create table fills (
  id             bigserial primary key,
  order_id       bigint      not null references orders(id) on delete cascade,
  broker_exec_id text,
  pair_id        bigint      not null references pairs(id),
  ticker         text        not null,
  side           order_side  not null,
  quantity       numeric(18,4) not null,
  price          numeric(18,6) not null,
  commission     numeric(20,4) not null default 0,
  ts             timestamptz not null
);

create index fills_ts       on fills (ts desc);
create index fills_order    on fills (order_id);

create table positions (
  id                bigserial primary key,
  pair_id           bigint      not null references pairs(id) on delete cascade,
  trade_id          bigint      references trades(id) on delete set null,
  ticker            text        not null,
  side              position_side not null,
  quantity          numeric(18,4) not null,
  avg_entry_price   numeric(18,6) not null,
  current_price     numeric(18,6),
  market_value      numeric(20,4),
  unrealized_pnl    numeric(20,4),
  opened_at         timestamptz not null,
  updated_at        timestamptz not null default now(),
  constraint positions_one_per_ticker unique (pair_id, ticker)
);


-- =====================================================================
-- SECTION 8 — RESEARCH DIAGNOSTICS (engine-computed, frontend read-only)
-- =====================================================================

create table cointegration_results (
  id               bigserial primary key,
  pair_id          bigint      not null references pairs(id) on delete cascade,
  as_of            timestamptz not null,
  test             cointegration_test not null,
  lookback_bars    integer     not null,
  statistic        double precision,
  pvalue           double precision,
  critical_values  jsonb,         -- {"1%": -3.96, "5%": -3.41, "10%": -3.13}
  passed           boolean,
  interpretation   text
);

create index cointegration_pair_asof on cointegration_results (pair_id, as_of desc);

create table rolling_diagnostics (
  id                            bigserial primary key,
  pair_id                       bigint      not null references pairs(id) on delete cascade,
  ts                            timestamptz not null,
  window_bars                   integer     not null,
  rolling_correlation           double precision,
  rolling_hedge_ratio           double precision,
  rolling_cointegration_stat    double precision,
  rolling_cointegration_pvalue  double precision,
  rolling_spread_volatility     double precision,
  half_life                     double precision,
  hurst_exponent                double precision,
  relationship_valid            boolean
);

create index rolling_diag_pair_ts on rolling_diagnostics (pair_id, ts desc);


-- =====================================================================
-- SECTION 9 — PERFORMANCE (engine/analytics-computed, frontend read-only)
-- =====================================================================

create table equity_curve (
  id                     bigserial primary key,
  pair_id                bigint      not null references pairs(id) on delete cascade,
  ts                     timestamptz not null,
  equity                 numeric(20,4) not null,
  cumulative_pnl         numeric(20,4) not null,
  cumulative_return_pct  double precision,
  drawdown_pct           double precision,
  -- Benchmarks kept pair-agnostic: these track the pair's own legs
  leg1_cumulative_return_pct double precision,
  leg2_cumulative_return_pct double precision,
  constraint equity_curve_unique_point unique (pair_id, ts)
);

create index equity_curve_pair_ts on equity_curve (pair_id, ts desc);

create table performance_metrics (
  id                          bigserial primary key,
  pair_id                     bigint      not null references pairs(id) on delete cascade,
  as_of                       timestamptz not null,
  period_label                text        not null default 'ALL',  -- ALL | YTD | 2026-08 | 2026-W33
  period_start                timestamptz,
  period_end                  timestamptz,

  total_return_pct            double precision,
  annualised_return_pct       double precision,
  sharpe_ratio                double precision,
  sortino_ratio               double precision,
  max_drawdown_pct            double precision,
  max_drawdown_duration_days  double precision,
  current_drawdown_pct        double precision,
  win_rate                    double precision,
  profit_factor               double precision,
  num_trades                  integer,
  avg_trade_pnl               numeric(20,4),
  avg_holding_period_seconds  integer,
  best_trade_pnl              numeric(20,4),
  worst_trade_pnl             numeric(20,4),
  total_pnl                   numeric(20,4),

  -- Return distribution moments
  return_mean                 double precision,
  return_median               double precision,
  return_std                  double precision,
  return_skew                 double precision,
  return_kurtosis             double precision,

  constraint performance_unique_period unique (pair_id, period_label)
);


-- =====================================================================
-- SECTION 10 — LOGS
-- =====================================================================

create table system_logs (
  id         bigserial primary key,
  ts         timestamptz not null default now(),
  engine_id  text,
  pair_id    bigint references pairs(id) on delete set null,
  level      log_level    not null,
  category   log_category not null,
  message    text         not null,
  details    jsonb
);

create index system_logs_ts       on system_logs (ts desc);
create index system_logs_level    on system_logs (level, ts desc);
create index system_logs_category on system_logs (category, ts desc);


-- =====================================================================
-- SECTION 11 — CONTROL COMMANDS
-- Written by the frontend, consumed and acknowledged by the engine.
-- This is the AUDIT + FALLBACK path. The primary kill path is the
-- engine's authenticated HTTPS control endpoint.
-- =====================================================================

create table commands (
  id               uuid primary key default gen_random_uuid(),
  command          command_type   not null,
  status           command_status not null default 'REQUESTED',
  pair_id          bigint         references pairs(id) on delete set null,
  payload          jsonb,

  requested_at     timestamptz not null default now(),
  requested_by     text,
  received_at      timestamptz,
  acknowledged_at  timestamptz,
  executed_at      timestamptz,
  expires_at       timestamptz not null default (now() + interval '30 seconds'),

  engine_id        text,
  result           jsonb,
  error_message    text,
  source           text not null default 'STREAMLIT'   -- STREAMLIT | HTTP_ENDPOINT | CLI
);

create index commands_status_requested on commands (status, requested_at desc);

comment on table commands is
  'Command lifecycle: REQUESTED -> RECEIVED -> ACKNOWLEDGED -> EXECUTED. The frontend must display UNCONFIRMED for any row still REQUESTED past expires_at. Never report success merely because the insert succeeded.';


-- =====================================================================
-- SECTION 12 — READ VIEWS
-- Keep query logic in the database, not scattered through page code.
-- =====================================================================

-- One-round-trip payload for the Overview header and KPI cards.
create view v_overview as
select
  ls.*,
  p.leg1_ticker,
  p.leg2_ticker,
  p.label                      as pair_label,
  eh.engine_id,
  eh.status                    as engine_status,
  eh.data_source,
  eh.market_status,
  eh.heartbeat_at,
  extract(epoch from (now() - eh.heartbeat_at)) as heartbeat_age_seconds,
  cfg.version                  as current_config_version
from live_state ls
join pairs p            on p.id = ls.pair_id
left join engine_heartbeat eh on eh.active_config_version is not null
left join strategy_config cfg on cfg.status = 'CURRENT';

-- Trades with tickers resolved, so the Trades page never joins manually.
create view v_trades as
select
  t.*,
  p.leg1_ticker,
  p.leg2_ticker,
  p.label as pair_label,
  case when t.net_pnl > 0 then true
       when t.net_pnl < 0 then false
       else null end as is_profitable
from trades t
join pairs p on p.id = t.pair_id;

-- Latest cointegration test result per test type.
create view v_latest_cointegration as
select distinct on (pair_id, test)
  pair_id, test, as_of, lookback_bars, statistic, pvalue,
  critical_values, passed, interpretation
from cointegration_results
order by pair_id, test, as_of desc;


-- =====================================================================
-- SECTION 13 — ROW LEVEL SECURITY
-- Streamlit Cloud is a public URL. Reads go through a restricted role;
-- writes (config + commands) require an authenticated session.
-- =====================================================================

alter table strategy_config enable row level security;
alter table commands        enable row level security;
alter table live_state      enable row level security;
alter table trades          enable row level security;
alter table system_logs     enable row level security;

-- Read-only for authenticated app users.
create policy read_live_state on live_state
  for select to authenticated using (true);
create policy read_trades on trades
  for select to authenticated using (true);
create policy read_logs on system_logs
  for select to authenticated using (true);
create policy read_config on strategy_config
  for select to authenticated using (true);
create policy read_commands on commands
  for select to authenticated using (true);

-- The frontend may propose config and issue commands, nothing more.
create policy insert_pending_config on strategy_config
  for insert to authenticated with check (status = 'PENDING');
create policy insert_commands on commands
  for insert to authenticated with check (status = 'REQUESTED');

-- The engine connects with the service role, which bypasses RLS.


-- =====================================================================
-- SECTION 14 — SEED DATA
-- SPY/IVV enters the system as a row, exactly like any future pair.
-- =====================================================================

insert into pairs (leg1_ticker, leg2_ticker, is_active, notes)
values ('SPY', 'IVV', true, 'Initial proof-of-concept pair.');

insert into strategy_config (
  version, status, pair_id,
  historical_lookback, zscore_lookback, correlation_lookback, cointegration_lookback,
  entry_threshold, exit_threshold, stop_loss_threshold, max_holding_period_seconds,
  min_correlation, max_cointegration_pvalue,
  capital_allocation, max_position_size, max_pair_exposure,
  max_leverage, max_simultaneous_positions,
  kill_switch_flattens, created_by, applied_at, requires_restart
)
select
  1, 'CURRENT', id,
  2520, 60, 252, 252,
  2.0, 0.5, 3.5, 86400,
  0.80, 0.05,
  100000.00, 25000.00, 50000.00,
  1.0, 1,
  true, 'seed', now(), false
from pairs where leg1_ticker = 'SPY' and leg2_ticker = 'IVV';

insert into connection_status (component, state) values
  ('ENGINE',      'UNKNOWN'),
  ('BROKER',      'UNKNOWN'),
  ('MARKET_DATA', 'UNKNOWN'),
  ('DATABASE',    'UNKNOWN'),
  ('ANALYTICS',   'UNKNOWN');
