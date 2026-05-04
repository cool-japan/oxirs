# OxiRS TSDB - TODO

*Version: 0.3.0 | Last Updated: May 3, 2026*

## Status: Production Ready

**oxirs-tsdb** provides time-series database capabilities for IoT-scale workloads with RDF integration.

### Features

#### Storage Engine
- **Gorilla Compression** - XOR-based compression with 40:1 ratio target
- **Delta-of-Delta Timestamps** - Variable-length encoding (<2 bits per timestamp)
- **Time Chunks** - 2-hour duration chunks with metadata
- **Columnar Storage** - LRU cache, atomic writes, binary chunk format
- **Indexing** - SeriesIndex with time-based lookup and crash recovery

#### Query Engine
- **Range Queries** - Efficient chunk scanning with lazy decompression
- **Aggregations** - AVG, MIN, MAX, SUM, COUNT with incremental processing
- **Window Functions** - Moving averages, moving min/max, rolling std dev
- **Resampling** - Time bucketing with configurable aggregation
- **Interpolation** - Linear, forward fill, backward fill, spline

#### Write Path
- **Write-Ahead Log** - Crash recovery with fsync control
- **Write Buffer** - 100K points default with flush triggers
- **Compactor** - Background merging with compression optimization
- **Retention Enforcer** - Automatic downsampling and expiration

#### RDF Integration
- **Hybrid Store Adapter** - Transparent RDF/TSDB routing
- **RDF Bridge** - Confidence-based auto-detection
- **Subject-Series Mapping** - Bidirectional lookups

#### SPARQL Extensions
- **Temporal Functions** - ts:window, ts:resample, ts:interpolate
- **Query Router** - Automatic backend routing

### Test Coverage
- **1127 tests passing** with comprehensive coverage
- Integration, SPARQL, storage, query, and write path tests

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ Gorilla compression, delta-of-delta, columnar storage, WAL, hybrid RDF/TSDB, 128 tests

### v0.2.3 - Current Release (March 16, 2026)
- ✅ 1M+ writes/sec sustained performance
- ✅ Memory usage profiling and optimization
- ✅ Advanced compression (adaptive, dictionary, RLE)
- ✅ Anomaly detection algorithms
- ✅ Holt-Winters forecasting support
- ✅ Apache Arrow integration (zero-copy export)
- ✅ Parquet export for analytics
- ✅ Prometheus remote write API
- ✅ Rollup engine, downsampler, gap filler, retention policy
- ✅ Anomaly detector, series metadata, compression codec
- ✅ 1127 tests passing

### v0.3.0 - Planned (Q2 2026)
- [x] GPU-accelerated aggregations (implemented 2026-04-28)
  - **Goal:** GPU dispatch for SUM/MIN/MAX/AVG/COUNT and rolling-window reductions over chunked columnar data.
  - **Design:** Feature gate `gpu` (default off). Free-function columnar API: `gpu_sum` dispatches to `scirs2_core::gpu` when `gpu` feature is on; `sum_column`, `min_column`, `max_column`, `avg_column`, `count_column`, `rolling_sum`, `rolling_avg` always available with CPU fallback. `GpuAggError` enum for typed errors.
  - **Files:** `src/analytics/gpu_aggregations.rs` (extended), `Cargo.toml` (gpu feature), `tests/gpu_agg.rs` (new)
  - **Tests:** 24 integration tests in `tests/gpu_agg.rs` — all pass (1217 total)
- [x] Kalman filter integration (implemented 2026-04-17)
  - **Goal:** Integrate existing KalmanFilter/AdaptiveKalmanFilter into the Forecaster trait with state persistence, online parameter adaptation, and SPARQL temporal function bindings
  - **Design:** KalmanForecaster struct implementing Forecaster: forecast(history, horizon) applies Kalman smoothing then projects state forward; confidence bands from innovation covariance P_k; KalmanHoltWinters ensemble blending Kalman state with Holt-Winters; serializable KalmanState { x, p, q, r } for persistence; SPARQL kalman_smooth(series_id, q, r) temporal function returning smoothed iterator
  - **Files:** src/analytics/forecasting.rs, src/analytics/kalman.rs, src/sparql/temporal.rs
  - **Tests:** Synthetic noisy sine wave: Kalman smoothed MAE < raw MAE; state round-trip serialization; SPARQL binding integration test; online adaptation on non-stationary series
  - **Risk:** Divergence on non-stationary series; adaptive Q noise tuning mitigates
- [x] DuckDB integration (implemented 2026-04-30)
  - **Goal:** Two-way DuckDB bridge feature-gated behind `duckdb` (default off — duckdb-rs has C deps per COOLJAPAN policy). User explicitly approved reopen 2026-04-30.
  - **Design:** Feature gate `duckdb`. Optional dep `duckdb` (1.10502 latest crates.io) with `bundled`, `vtab-arrow`, `appender-arrow`. Export: TSDB chunk → DuckDB table via Arrow `RecordBatch` appender (zero-copy through workspace-aligned `arrow = 58`). Import: DuckDB SQL → Arrow `RecordBatch` → TSDB `TimeChunk`. CLI helper `oxirs tsdb duck-db <chunk> <sql>` to inspect TSDB chunks via DuckDB SQL.
  - **Files:** `src/duckdb_bridge.rs` (feature-gated, new), `Cargo.toml` (feature flags + optional deps), `tests/duckdb_bridge.rs` (new), `tools/oxirs/src/tools/tsdb_duckdb.rs` (new CLI subcommand)
  - **Tests:** 12 unit tests on DuckDB round-trip + 12 integration tests gated on `duckdb` feature; 4 CLI tests gated on `tsdb-duckdb`. All pass with zero clippy warnings on both feature configurations.
  - **Risk:** duckdb-rs C deps. Mitigation: feature-gated, default off; CI without `duckdb` stays pure-Rust.
- [x] Raft replication for high availability (implemented 2026-04-28)
  - **Goal:** Wire TSDB WAL records as Raft log entries through the Raft cluster; leader-only writes, followers serve reads; snapshot-based log compaction.
  - **Design:** `TsdbRaftOp` (Insert/Delete/Truncate) with JSON wire format; `WalReplicator` bridges WAL→Raft via `mpsc::Sender<Vec<u8>>`; `TsdbStateMachine` applies ops with strict sequential index enforcement; `TsdbRaftSnapshot` / `SnapshotStore` for log compaction and install/restore.
  - **Files:** `src/replication/wal_replicator.rs` (new), `src/replication/snapshot.rs` (new), `src/replication/mod.rs` (extended), `src/lib.rs` (re-exports), `tests/raft_ha.rs` (new)
  - **Tests:** 33 integration tests in `tests/raft_ha.rs` — all pass (1217 total)
- [x] Multi-region deployment (implemented 2026-04-30)
  - **Goal:** Active-active multi-region geometry with per-region Raft groups + global write routing.
  - **Design:** Per-region `ReplicationGroup` (Raft) replicates within region. Global routing layer in `src/multi_region/`: `routing.rs` picks a home region per write (subject prefix + tenant rules + failover chain); `health_probe.rs` runs heartbeat/timeout-based unreachable detection; `replication.rs` runs async cross-region fanout with last-writer-wins conflict resolution keyed on `(timestamp_ms, region_id)`. Self-contained — no oxirs-cluster dep — but mirrors oxirs-cluster's W2-S5 active-active geo design.
  - **Files:** `src/multi_region/{routing,health_probe,replication,mod}.rs` (new), `tests/multi_region.rs` (new)
  - **Prerequisites:** Raft from round 1 W2-S7 (already shipped)
  - **Tests:** 40 unit tests + 20 integration tests in `tests/multi_region.rs` (3-region simulator with kill/recover cycle, routing rule priority, LWW resolution, failover chain). All pass.
  - **Risk:** cross-region split-brain. Mitigation: per-region Raft groups isolated; cross-region async with deterministic LWW conflict resolution.

## Documentation

- 9 working examples included:
  - compression_demo.rs - Gorilla + delta-of-delta encoding
  - query_demo.rs - Range queries and aggregations
  - window_functions.rs - Moving averages and window operations
  - resampling_demo.rs - Time bucketing and downsampling
  - hybrid_storage.rs - Basic hybrid RDF + TSDB
  - benchmark_demo.rs - Performance measurement
  - hybrid_store_demo.rs - Advanced hybrid routing
  - auto_detection_demo.rs - Intelligent auto-detection
  - columnar_storage_demo.rs - Disk-backed storage

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS TSDB v0.2.3 - Time-series database for IoT-scale workloads*
