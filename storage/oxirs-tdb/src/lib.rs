//! # OxiRS TDB - Apache Jena TDB/TDB2 Compatible Storage Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--beta.1-blue)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Beta Release (v0.1.0-beta.1)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! High-performance RDF triple store with B+Tree indexes, ACID transactions,
//! and Apache Jena TDB/TDB2 API compatibility.
//!
//! ## Features
//!
//! - **B+Tree Storage** - Efficient range queries and sequential scans
//! - **Triple Indexes** - SPO, POS, OSP for optimal query performance
//! - **ACID Transactions** - Write-Ahead Logging (WAL) with 2PL concurrency control
//! - **Dictionary Encoding** - Compress IRIs and literals to 8-byte NodeIDs
//! - **Buffer Pool** - LRU caching for hot pages
//! - **100M+ Triples** - Scalable to large datasets
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxirs_tdb::TdbStore;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new TDB store
//! let mut store = TdbStore::open("/path/to/data")?;
//!
//! // Insert triples
//! // store.insert(...)?;
//!
//! // Query triples
//! // let results = store.query(...)?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         TdbStore (High-level API)       │
//! └─────────────────┬───────────────────────┘
//!                   │
//!       ┌───────────┴───────────┐
//!       │                       │
//! ┌─────▼─────┐         ┌──────▼──────┐
//! │  Indexes  │         │ Dictionary  │
//! │ SPO/POS/  │         │   Encoding  │
//! │    OSP    │         │             │
//! └─────┬─────┘         └──────┬──────┘
//!       │                      │
//!       └───────────┬──────────┘
//!                   │
//!            ┌──────▼──────┐
//!            │   B+Tree    │
//!            └──────┬──────┘
//!                   │
//!       ┌───────────┴───────────┐
//!       │                       │
//! ┌─────▼─────┐         ┌──────▼──────┐
//! │  Buffer   │         │     WAL     │
//! │   Pool    │         │  (Logging)  │
//! └─────┬─────┘         └──────┬──────┘
//!       │                      │
//!       └───────────┬──────────┘
//!                   │
//!            ┌──────▼──────┐
//!            │ File Manager│
//!            │   (mmap)    │
//!            └─────────────┘
//! ```
//!
//! ## See Also
//!
//! - [`oxirs-core`](https://docs.rs/oxirs-core) - RDF data model
//! - [`oxirs-arq`](https://docs.rs/oxirs-arq) - SPARQL query engine

#![doc(html_root_url = "https://docs.rs/oxirs-tdb/0.1.0-beta.1")]
#![warn(missing_docs)]
#![allow(dead_code)] // Allow during development
#![allow(unused_imports)] // Allow during development
#![allow(unused_variables)] // Allow during development

// Core modules
pub mod error;

// Storage layer
pub mod storage;

// B+Tree implementation
pub mod btree;

// Index structures
pub mod index;

// Dictionary encoding
pub mod dictionary;

// Transaction management
pub mod transaction;

// Compression and optimization
pub mod compression;

// High-level store API
pub mod store;

// Production hardening features
pub mod production;

// Backup and restore utilities
pub mod backup;

// Crash recovery and corruption detection
pub mod recovery;

// Query hint support
pub mod query_hints;

// Query result caching
pub mod query_cache;

// Cost-based query optimization
// TODO(v0.1.0-rc.1): Complete StatisticsSnapshot API integration for full functionality
pub mod query_optimizer;

// Statistics collection for cost-based optimization
pub mod statistics;

// Query monitoring (timeout enforcement and slow query logging)
pub mod query_monitor;

// Advanced diagnostic tools
pub mod diagnostics;

// High-performance operations using SciRS2-Core
pub mod performance;

// RDF-star support for quoted triples
pub mod rdf_star;

// Connection pooling for multi-client access
pub mod connection_pool;

// Query resource quotas for per-query resource limiting
pub mod query_resource_quota;

// Materialized views for query acceleration
pub mod materialized_views;

// WAL archiving for point-in-time recovery
pub mod wal_archive;

// Re-export commonly used types
pub use connection_pool::{ConnectionPool, ConnectionPoolConfig, ConnectionPoolStatsSnapshot};
pub use error::{Result, TdbError};
pub use materialized_views::{
    MaterializedView, MaterializedViewConfig, MaterializedViewManager,
    MaterializedViewManagerStats, RefreshStrategy, ViewInfo,
};
pub use query_resource_quota::{
    QueryQuotaStats, QueryResourceQuotaConfig, QueryResourceQuotaManager, QueryResourceTracker,
    QueryResourceUsage,
};
pub use store::{TdbConfig, TdbStats, TdbStore};
pub use wal_archive::{
    WalArchiveConfig, WalArchiveMetadata, WalArchiver, WalArchiverStatsSnapshot,
};

/// TDB storage engine version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default page size (4KB)
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// Default buffer pool size (1000 pages = 4MB)
pub const DEFAULT_BUFFER_POOL_SIZE: usize = 1000;
