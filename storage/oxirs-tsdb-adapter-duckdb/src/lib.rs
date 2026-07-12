//! Quarantined DuckDB analytics-bridge adapter for `oxirs-tsdb`.
//!
//! # Why this crate exists
//!
//! Under the COOLJAPAN **Pure Rust Policy v2** (purity is measured on the full
//! `--all-features` dependency closure), a *published* crate must not drag in
//! C FFI. The DuckDB bridge uses the [`duckdb`] crate, which pulls in
//! `libduckdb-sys` — the C FFI binding to an embedded `libduckdb` that is
//! *bundled and built from source*. To keep the published `oxirs-tsdb` surface
//! 100% Pure Rust while *preserving* the real DuckDB capability, the live
//! `duckdb`-backed bridge has been **quarantined** into this crate.
//!
//! This crate is `publish = false`: it never ships to crates.io, so its C FFI
//! dependency never appears in the published Pure-Rust surface. Binaries that
//! actually want to inspect TSDB chunks through embedded DuckDB SQL depend on
//! this crate directly.
//!
//! # Relationship to `oxirs-tsdb`
//!
//! The bridge consumes `oxirs-tsdb`'s public [`oxirs_tsdb::TimeChunk`] /
//! [`oxirs_tsdb::DataPoint`] types and reports errors through
//! [`oxirs_tsdb::TsdbError`] / [`oxirs_tsdb::TsdbResult`] — the very same types
//! the published crate exposes — so callers drive it through an identical type
//! surface, just from this adapter crate. The free-function API and option
//! structs are re-exported here under the same friendly aliases the published
//! crate previously exposed behind its (now removed) `duckdb` feature.
//!
//! [`duckdb`]: https://docs.rs/duckdb

/// DuckDB ↔ TSDB chunk bridge: Arrow `RecordBatch` transport between TSDB
/// time chunks and an embedded DuckDB connection.
pub mod duckdb_bridge;

pub use duckdb_bridge::{
    chunk_to_record_batch, open_in_memory as duckdb_open_in_memory, points_to_record_batch,
    read_into_chunk, record_batches_to_points, register_tsdb_chunk, run_sql as duckdb_run_sql,
    ReadOptions as DuckDbReadOptions, RegisterOptions as DuckDbRegisterOptions, SERIES_COL, TS_COL,
    VAL_COL,
};
