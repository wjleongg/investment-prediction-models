-- =====================================================================
-- TEARDOWN — drops everything created by schema.sql.
-- Development use only. Never run against a database holding real
-- trade history you intend to keep.
-- =====================================================================

drop view if exists v_latest_cointegration cascade;
drop view if exists v_trades cascade;
drop view if exists v_overview cascade;

drop table if exists commands cascade;
drop table if exists system_logs cascade;
drop table if exists performance_metrics cascade;
drop table if exists equity_curve cascade;
drop table if exists rolling_diagnostics cascade;
drop table if exists cointegration_results cascade;
drop table if exists positions cascade;
drop table if exists fills cascade;
drop table if exists orders cascade;
drop table if exists trades cascade;
drop table if exists signals cascade;
drop table if exists model_state_history cascade;
drop table if exists market_data cascade;
drop table if exists live_state cascade;
drop table if exists system_metrics cascade;
drop table if exists connection_status cascade;
drop table if exists engine_heartbeat cascade;
drop table if exists strategy_config cascade;
drop table if exists pairs cascade;

drop type if exists price_field cascade;
drop type if exists cointegration_test cascade;
drop type if exists config_status cascade;
drop type if exists connection_state cascade;
drop type if exists command_status cascade;
drop type if exists command_type cascade;
drop type if exists log_category cascade;
drop type if exists log_level cascade;
drop type if exists exit_reason cascade;
drop type if exists trade_status cascade;
drop type if exists trade_direction cascade;
drop type if exists order_status cascade;
drop type if exists order_type cascade;
drop type if exists order_side cascade;
drop type if exists position_side cascade;
drop type if exists relationship_health cascade;
drop type if exists signal_type cascade;
drop type if exists market_status cascade;
drop type if exists data_source cascade;
drop type if exists system_status cascade;
