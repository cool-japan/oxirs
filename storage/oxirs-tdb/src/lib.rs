//! # OxiRS TDB - Apache Jena TDB/TDB2 Compatible Storage Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.2.4-blue)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Production Release (v0.2.4)
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

#![doc(html_root_url = "https://docs.rs/oxirs-tdb/0.2.4")]
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
pub mod btree_index;

// Index structures
pub mod index;

// Dictionary encoding
pub mod dictionary;

// Transaction management
pub mod transaction;

// Consensus algorithms for distributed coordination
pub mod consensus;

// Distributed coordination and replication
pub mod distributed;

// Data loading utilities
pub mod loader;

// Compression and optimization
pub mod compression;

// High-level store API
pub mod store;

// Production hardening features
pub mod production;

// Backup and restore utilities
pub mod backup;

// Backup encryption for data at rest
pub mod backup_encryption;

// Online backup without downtime
pub mod online_backup;

// Crash recovery and corruption detection
pub mod recovery;

// Query hint support
pub mod query_hints;

// Query result caching
pub mod query_cache;

// Cost-based query optimization with StatisticsSnapshot integration
pub mod query_optimizer;

// Join order optimization for query planning
pub mod query_join_optimizer;

// Adaptive query execution with runtime plan adjustment
pub mod adaptive_execution;

// Statistics collection for cost-based optimization
pub mod statistics;

// Slow query logging and analysis for production monitoring
pub mod slow_query_log;

// Query timeout enforcement for production safety
pub mod query_timeout;

// Query monitoring (timeout enforcement and slow query logging)
pub mod query_monitor;

// Advanced diagnostic tools
pub mod diagnostics;

// Advanced diagnostic tools (monitoring and analysis)
pub mod advanced_diagnostics;

// High-performance operations using SciRS2-Core
pub mod performance;

// Bulk operations for high-throughput processing
pub mod bulk_operations;

// RDF-star support for quoted triples
pub mod rdf_star;

// Connection pooling for multi-client access
pub mod connection_pool;

// Connection pool optimizer (advanced pool management)
pub mod connection_pool_optimizer;

// Query resource quotas for per-query resource limiting
pub mod query_resource_quota;

// Materialized views for query acceleration
pub mod materialized_views;

// WAL archiving for point-in-time recovery
pub mod wal_archive;

// WAL shipping for continuous archiving
pub mod wal_shipping;

// Cloud storage integration (S3, GCS, Azure)
pub mod cloud_storage;

// Database operations and management
pub mod database_ops;

// Observability and monitoring
pub mod observability;

// Full-text search over RDF literals (Tantivy-backed)
pub mod text_search;

// Bloom filter for probabilistic triple existence checks
pub mod bloom_filter;

// Re-export commonly used types
pub use backup_encryption::{BackupEncryption, EncryptedData, EncryptionConfig};
pub use bulk_operations::{
    BulkStats, BulkTripleProcessor, ParallelPipelineBuilder, StreamingTripleIterator,
};
pub use compression::{
    DeltaStoreConfig, DeltaStoreStats, EncodedDeltaLog, TripleDelta, TripleDeltaStore,
};
pub use connection_pool::{ConnectionPool, ConnectionPoolConfig, ConnectionPoolStatsSnapshot};
pub use database_ops::{
    CompactionStats, DatabaseMetadata, DatabaseOps, DatabaseStatus, RepairReport,
};
pub use error::{Result, TdbError};
pub use index::{BTreeTripleIndex, EncodedTriple, TripleIndexSet, TripleOrdering};
pub use loader::{BulkLoadStats, BulkLoader, BulkLoaderConfig, BulkLoaderFactory};
pub use loader::{
    NodeDictionary, ParallelBulkLoadConfig, ParallelBulkLoadStats, ParallelBulkLoader, RawTriple,
    RdfNode, TripleSource, VecTripleSource,
};
pub use materialized_views::{
    MaterializedView, MaterializedViewConfig, MaterializedViewManager,
    MaterializedViewManagerStats, RefreshStrategy, ViewInfo,
};
pub use observability::{
    HealthCheck, HealthCheckResult, HealthCheckResults, HealthStatus, MetricSnapshot,
    ObservabilityConfig, ObservabilityManager, TraceSpan, TraceSpanId,
};
pub use online_backup::{
    OnlineBackupManager, OnlineBackupStats, Snapshot, SnapshotConfig, SnapshotId, SnapshotStatus,
};
pub use query_resource_quota::{
    QueryQuotaStats, QueryResourceQuotaConfig, QueryResourceQuotaManager, QueryResourceTracker,
    QueryResourceUsage,
};
pub use store::{
    CompressionAlgorithm, ReplicationMode, StoreParams, StoreParamsBuilder, StorePresets,
    TdbConfig, TdbStats, TdbStore,
};
pub use wal_archive::{
    WalArchiveConfig, WalArchiveMetadata, WalArchiver, WalArchiverStatsSnapshot,
};
pub use wal_shipping::{
    ShippingConfig, ShippingDestination, ShippingRecord, ShippingStats, ShippingStatus, WalShipper,
};

// Temporal RDF: version control and time-travel queries (v0.3.0)
pub mod temporal;

pub use temporal::{
    ChangeEntry, ChangeLog, ChangeLogStats, ChangeOperation, TemporalDiff, TemporalSnapshot,
    TemporalVersionStore, TimeTravelQuery, TimeTravelQueryBuilder, TripleKey, TriplePattern,
    VersionedTriple,
};

/// TDB storage engine version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default page size (4KB)
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// Default buffer pool size (1000 pages = 4MB)
pub const DEFAULT_BUFFER_POOL_SIZE: usize = 1000;

// MVCC Transaction Manager: snapshot isolation and SSI (v0.3.0)
pub mod mvcc;

pub use mvcc::deadlock::{DeadlockDetector, DeadlockScanResult};
pub use mvcc::{
    IsolationLevel, MvccError, MvccManager, MvccStats, TransactionContext, TransactionState, TxId,
    VersionedEntry, VersionedKey, TX_ID_COMMITTED,
};

// Multi-tenant isolation (v1.0.0 LTS)
pub mod tenant;

pub use tenant::{
    TenantAuditEvent, TenantAuditLog, TenantConfig, TenantError, TenantHandle, TenantId,
    TenantIsolationLayer, TenantRegistry, TenantResult, TenantStats, TenantStore,
};

// TDB2-compatible data format (v1.1.0)
pub mod tdb2;

pub use tdb2::{NodeTable, RdfTerm, Tdb2Database, Tdb2Format, TripleIndex};

// Backup engine: triple-level backup/restore with format + compression (v0.2.0)
pub mod backup_engine;

pub use backup_engine::{
    BackupCompression, BackupDelta, BackupEngine, BackupFormat, BackupManifest, IncrementalBackup,
};

// Advanced SLA monitoring and enforcement (v0.2.0)
pub mod sla;

pub use sla::{SlaComplianceReport, SlaMonitor, SlaViolation, StorageSla, ViolationType};

// Dataset import/export formats: CSV-RDF, JSON-LD, N-Quads (v0.2.0)
pub mod formats;

pub use formats::{CsvRdfMapper, JsonLdImporter, NQuadsExporter, NQuadsImporter, RdfColumnType};

// WAL compaction and point-in-time recovery (v1.1.0)
pub mod wal_compaction;

// LSM-tree compaction strategy with bloom filters (v1.1.0)
pub mod lsm_compaction;

// v1.2.0 Compressed prefix table for IRI storage
pub mod prefix_table;

// v1.2.0 Triple store cardinality estimator for query optimization
pub mod index_statistics;

// v1.5.0 Write-ahead journal for crash recovery
pub mod journal;

/// Write buffer for batching triple insertions (v1.6.0).
pub mod triple_buffer;

/// Background compaction task scheduler (v1.7.0).
pub mod compaction_scheduler;

/// Atomic batch write operations with WAL integration (v1.8.0).
pub mod write_batch;

/// Database checkpoint management for crash recovery and WAL truncation (v1.9.0).
pub mod checkpoint_manager;

/// Database vacuuming and compaction: dead-space detection, fragmentation analysis, online vacuum (v2.0.0).
pub mod vacuum;

/// B-tree index rebuild from raw triple data: SPO/POS/OSP/GSPO/GPOS/GOSP orders,
/// sorted IndexEntry generation, rebuild statistics, and sortedness verification (v1.1.0 round 14)
pub mod index_rebuilder;

/// Triple pattern cache: memoize repeated pattern lookups using LRU eviction (v1.1.0 round 15)
pub mod triple_cache;

/// Bloom filter index for fast triple existence checks: FNV double-hashing,
/// capacity-tuned bit array, TripleBloomIndex wrapper (v1.1.0 round 16)
pub mod bloom_index;

pub use wal_compaction::{
    BaseSnapshot, CompactionConfig, CompactionStats as WalCompactionStats, PitrConfig, PitrEngine,
    RecoveryPlan, WalCompactor, WalRecord, WalRecordType, WalSegment,
};
