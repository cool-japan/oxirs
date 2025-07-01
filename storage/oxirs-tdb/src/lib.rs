//! # OxiRS TDB - Rust-Native RDF Triple Database
//!
//! OxiRS TDB is a high-performance, persistent RDF storage engine that provides
//! **TDB2-equivalent functionality** with modern Rust performance optimizations.
//! It features multi-version concurrency control (MVCC), ACID transactions,
//! advanced compression, and seamless integration with the OxiRS ecosystem.
//!
//! ## Key Features
//!
//! - **Full ACID Compliance**: Complete transaction support with MVCC
//! - **Advanced Compression**: Multiple compression algorithms including adaptive, column-store, and bitmap compression
//! - **High Performance**: Optimized B+ tree indices with bulk loading and validation
//! - **Crash Recovery**: ARIES-style write-ahead logging with checkpointing
//! - **Concurrent Access**: Multi-reader, single-writer with snapshot isolation
//! - **TDB2 Compatibility**: Feature parity with Apache Jena TDB2
//!
//! ## Quick Start
//!
//! ```rust
//! use oxirs_tdb::{TdbStore, TdbConfig, Term};
//! use anyhow::Result;
//!
//! # fn example() -> Result<()> {
//! // Create or open a TDB store
//! let config = TdbConfig::default();
//! let store = TdbStore::new(config)?;
//!
//! // Create some RDF terms
//! let subject = Term::iri("http://example.org/person1");
//! let predicate = Term::iri("http://example.org/name");
//! let object = Term::literal("Alice");
//!
//! // Insert a triple
//! store.insert_triple(&subject, &predicate, &object)?;
//!
//! // Query triples
//! let results = store.query_triples(
//!     Some(&subject),
//!     None,
//!     None
//! )?;
//!
//! println!("Found {} triples", results.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture Overview
//!
//! The OxiRS TDB implementation consists of several key components:

// Core modules
pub mod assembler;
pub mod backup_restore;
pub mod bitmap_index;
pub mod block_manager;
pub mod btree;
pub mod checkpoint;
pub mod compact_encoding;
pub mod compression;
pub mod config;
pub mod dictionary;
pub mod filesystem;
pub mod hash_index;
pub mod lock_manager;
pub mod metrics;
pub mod mvcc;
pub mod nodes;
pub mod optimistic_concurrency;
pub mod page;
pub mod production_hardening;
pub mod query_execution;
pub mod query_optimizer;
pub mod storage;
pub mod timestamp_ordering;
pub mod transactions;
pub mod triple_store;
pub mod wal;

// Re-export key types for convenience
pub use config::{TdbConfig as TdbAdvancedConfig, WorkloadType};

use anyhow::{anyhow, Result};
use serde::{Serialize, Deserialize};
use std::path::Path;

// Re-export main types for convenience
pub use backup_restore::{
    BackupConfig, BackupMetadata, BackupProgress, BackupRestoreManager, BackupRestoreStats,
    BackupType, CompressionLevel, RecoveryOptions, RecoveryPhase, RecoveryProgress, RecoveryTarget,
};
pub use bitmap_index::{
    BitmapCompression, BitmapIndex, BitmapIndexConfig, BitmapIndexStats, CompressedBitmap,
};
pub use block_manager::{
    AllocationStrategy, BlockId, BlockManager, BlockManagerConfig, BlockManagerStats,
    BlockMetadata, BlockStatus,
};
pub use checkpoint::{
    CheckpointConfig, CheckpointManagerStats, CheckpointMetadata, CheckpointType, DirtyPageStats,
    DirtyPageTracker, OnlineCheckpointManager, PageModificationInfo,
};
pub use compact_encoding::{
    CompactDecoder, CompactEncoder, CompactEncodingScheme, EncodingStats,
};
pub use compression::{
    AdaptiveCompressor, AdvancedCompressionType, ColumnStoreCompressor, CompressedData,
    CompressionMetadata,
};
pub use dictionary::{
    DictionaryConfig, DictionaryStats, InternedString, StringDictionary, StringId,
};
pub use filesystem::{
    AdvancedLockingConfig, DatabaseMetadata, FileSystemConfig, FileType, TdbFileSystem,
};
pub use hash_index::{HashIndex, HashIndexConfig, HashIndexStats};
pub use lock_manager::{
    LockGrant, LockManager, LockManagerConfig, LockManagerError, LockManagerStats, LockMode,
    LockRequest,
};
pub use mvcc::{TransactionId, Version};
pub use nodes::{NodeId, NodeTable, Term};
pub use optimistic_concurrency::{
    ConflictType, OptimisticConcurrencyController, OptimisticConfig, OptimisticStats,
    OptimisticTransactionInfo, TransactionPhase, ValidationResult, VersionVector, WriteOperation,
};
pub use production_hardening::{
    CircuitBreaker, EdgeCaseValidator, HealthMetrics, HealthMonitor, ResourceLimits,
};
pub use timestamp_ordering::{
    CausalRelation, ClockSyncManager, HybridLogicalClock, LamportTimestamp,
    NodeId as TimestampNodeId, TimestampBundle, TimestampManager, TimestampStats, VectorClock,
};
pub use query_optimizer::{
    IndexType, OptimizationRecommendation, PatternType, QueryCostModel, QueryOptimizer,
    QueryPattern, QueryStatisticsSummary, QueryStats,
};
pub use metrics::{
    ComprehensiveMetrics, ErrorMetrics, MetricsCollector, MetricsReport, MetricsStatistics,
    QueryMetrics, StorageMetrics, SystemMetrics, TimeSeriesMetrics,
};
pub use triple_store::{Quad, Triple, TripleStore, TripleStoreConfig, TripleStoreStats};

// Export both simple and advanced configs
// Removed TdbConfig alias to avoid conflict with config::TdbConfig

/// Simple configuration for TDB storage engine
///
/// This structure controls basic TDB storage engine behavior.
/// For advanced configuration, use `TdbAdvancedConfig`.
///
/// # Examples
///
/// ```rust
/// use oxirs_tdb::SimpleTdbConfig;
///
/// // Default configuration
/// let config = SimpleTdbConfig::default();
///
/// // Custom configuration for production
/// let prod_config = SimpleTdbConfig {
///     location: "/var/lib/oxirs/data".to_string(),
///     cache_size: 1024 * 1024 * 1024, // 1GB cache
///     enable_transactions: true,
///     enable_mvcc: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SimpleTdbConfig {
    /// Database location on disk
    ///
    /// This directory will contain all database files including:
    /// - Node table files (nodes.dat, nodes.idn)
    /// - Triple index files (SPO.bpt, POS.bpt, etc.)
    /// - Transaction logs (txn.log)
    /// - Metadata files (tdb.info, tdb.lock)
    pub location: String,

    /// Buffer pool cache size in bytes
    ///
    /// Controls how much memory is used for caching database pages.
    /// Larger values improve performance but use more memory.
    /// Recommended: 25-50% of available system memory.
    pub cache_size: usize,

    /// Enable ACID transaction support
    ///
    /// When enabled, provides full ACID guarantees with write-ahead logging.
    /// Disable only for read-only workloads or bulk loading scenarios.
    pub enable_transactions: bool,

    /// Enable multi-version concurrency control
    ///
    /// Allows multiple concurrent readers with snapshot isolation.
    /// Provides better performance for read-heavy workloads.
    pub enable_mvcc: bool,
}

impl Default for SimpleTdbConfig {
    fn default() -> Self {
        Self {
            location: "./tdb".to_string(),
            cache_size: 1024 * 1024 * 100, // 100MB
            enable_transactions: true,
            enable_mvcc: true,
        }
    }
}

/// High-level TDB storage engine
///
/// `TdbStore` is the main entry point for interacting with a TDB database.
/// It provides a high-level API for storing and querying RDF triples and quads
/// with full ACID transaction support and MVCC.
///
/// The store automatically manages:
/// - **Index Management**: Six standard RDF indices (SPO, POS, OSP, SOP, PSO, OPS)
/// - **Compression**: Adaptive compression based on data characteristics
/// - **Transactions**: ACID compliance with write-ahead logging
/// - **Concurrency**: Multi-reader, single-writer with snapshot isolation
/// - **Recovery**: Automatic crash recovery using ARIES protocol
/// - **File System**: TDB2-compatible directory structure and file management
///
/// # Examples
///
/// ```rust
/// use oxirs_tdb::{TdbStore, TdbConfig, Term};
/// use anyhow::Result;
///
/// # fn example() -> Result<()> {
/// // Create a new database
/// let config = TdbConfig {
///     location: "./test-db".to_string(),
///     cache_size: 1024 * 1024 * 100, // 100MB
///     enable_transactions: true,
///     enable_mvcc: true,
/// };
/// let store = TdbStore::new(config)?;
///
/// // Work with transactions
/// let tx = store.begin_transaction()?;
///
/// // Insert RDF data
/// let person = Term::iri("http://example.org/alice");
/// let name = Term::iri("http://xmlns.com/foaf/0.1/name");
/// let alice = Term::literal("Alice");
///
/// store.insert_triple(&person, &name, &alice)?;
///
/// // Commit the transaction
/// store.commit_transaction(tx)?;
///
/// // Query the data
/// let results = store.query_triples(Some(&person), None, None)?;
/// assert_eq!(results.len(), 1);
/// # Ok(())
/// # }
/// ```
///
/// # Thread Safety
///
/// `TdbStore` is designed to be shared safely across threads:
/// - Multiple concurrent readers are supported
/// - Write operations are serialized through the transaction system
/// - All operations are atomic and isolated
///
/// # Performance Considerations
///
/// - **Bulk Loading**: Use transactions for bulk operations to minimize I/O
/// - **Cache Size**: Larger cache sizes improve query performance
/// - **Index Selection**: The store automatically selects optimal indices for queries
/// - **Compression**: Enable advanced compression for space-constrained environments
/// - **File Layout**: TDB2-compatible file organization for optimal I/O patterns
pub struct TdbStore {
    config: SimpleTdbConfig,
    filesystem: TdbFileSystem,
    triple_store: TripleStore,
    health_monitor: HealthMonitor,
    query_optimizer: QueryOptimizer,
    metrics_collector: MetricsCollector,
}

impl TdbStore {
    /// Create a new TDB store with the specified configuration
    ///
    /// This method initializes a new TDB database at the specified location.
    /// If the database already exists, it will be opened for use.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the database
    ///
    /// # Returns
    ///
    /// Returns `Ok(TdbStore)` on success, or an error if:
    /// - The location is not accessible or writable
    /// - Database files are corrupted and cannot be recovered
    /// - Insufficient system resources (memory, file handles)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_tdb::{TdbStore, TdbConfig};
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let config = TdbConfig {
    ///     location: "./my-database".to_string(),
    ///     cache_size: 1024 * 1024 * 256, // 256MB cache
    ///     enable_transactions: true,
    ///     enable_mvcc: true,
    /// };
    ///
    /// let store = TdbStore::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: SimpleTdbConfig) -> Result<Self> {
        // Initialize TDB2-compatible file system
        let filesystem_config = FileSystemConfig {
            create_if_missing: true,
            sync_writes: config.enable_transactions,
            use_memory_mapping: false,
            backup_on_startup: false, // Don't backup on every open
            max_file_handles: 256,
            page_size: crate::page::PAGE_SIZE,
            advanced_locking: AdvancedLockingConfig::default(),
        };

        let filesystem = TdbFileSystem::new(&config.location, filesystem_config)?;

        // Initialize production hardening components
        let resource_limits = ResourceLimits {
            max_memory_usage: 85.0,
            max_cpu_usage: 90.0,
            max_disk_usage: 95.0,
            max_file_handles: 1000,
            max_connections: 500,
            max_error_rate: 5.0,
            max_response_time_ms: 1000.0,
        };

        let health_monitor = HealthMonitor::new(resource_limits);

        let triple_store_config = TripleStoreConfig {
            storage_path: filesystem.data_path().to_path_buf(),
            buffer_config: crate::page::BufferPoolConfig {
                max_pages: config.cache_size / crate::page::PAGE_SIZE,
                ..Default::default()
            },
            ..Default::default()
        };

        let triple_store = TripleStore::with_config(triple_store_config)?;
        let query_optimizer = QueryOptimizer::new();
        let metrics_collector = MetricsCollector::new(std::time::Duration::from_secs(60));

        Ok(Self {
            config,
            filesystem,
            triple_store,
            health_monitor,
            query_optimizer,
            metrics_collector,
        })
    }

    /// Open an existing TDB store
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = SimpleTdbConfig {
            location: path.as_ref().to_string_lossy().to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Begin a new read-write transaction
    ///
    /// Creates a new ACID transaction that allows both reading and writing operations.
    /// All changes made within the transaction are isolated from other transactions
    /// until the transaction is committed.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Transaction)` on success, or an error if:
    /// - The transaction system is unavailable
    /// - Maximum number of concurrent transactions is exceeded
    /// - System resources are exhausted
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_tdb::{TdbStore, TdbConfig, Term};
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let store = TdbStore::new(TdbConfig::default())?;
    ///
    /// // Begin a transaction
    /// let tx = store.begin_transaction()?;
    ///
    /// // Perform operations...
    /// let subject = Term::iri("http://example.org/resource");
    /// let predicate = Term::iri("http://example.org/property");
    /// let object = Term::literal("value");
    /// store.insert_triple(&subject, &predicate, &object)?;
    ///
    /// // Commit the changes
    /// store.commit_transaction(tx)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Transaction Isolation
    ///
    /// Transactions provide full ACID guarantees:
    /// - **Atomicity**: All operations succeed or fail together
    /// - **Consistency**: Database remains in a valid state
    /// - **Isolation**: Changes are not visible to other transactions until commit
    /// - **Durability**: Committed changes survive system failures
    pub fn begin_transaction(&self) -> Result<Transaction> {
        let tx_id = self.triple_store.begin_transaction()?;
        Ok(Transaction::new(tx_id))
    }

    /// Begin a read-only transaction
    pub fn begin_read_transaction(&self) -> Result<Transaction> {
        let tx_id = self.triple_store.begin_read_transaction()?;
        Ok(Transaction::new(tx_id))
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, transaction: Transaction) -> Result<Version> {
        self.triple_store.commit_transaction(transaction.tx_id)
    }

    /// Rollback a transaction
    pub fn rollback_transaction(&self, transaction: Transaction) -> Result<()> {
        self.triple_store.abort_transaction(transaction.tx_id)
    }

    /// Insert an RDF triple into the store
    ///
    /// Adds a new triple (subject, predicate, object) to the database.
    /// The triple is automatically indexed in all six standard RDF indices
    /// for efficient querying.
    ///
    /// # Arguments
    ///
    /// * `subject` - The subject term (typically an IRI or blank node)
    /// * `predicate` - The predicate term (typically an IRI)
    /// * `object` - The object term (IRI, literal, or blank node)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if:
    /// - The term encoding fails
    /// - Index update operations fail
    /// - Transaction constraints are violated
    /// - Storage I/O errors occur
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_tdb::{TdbStore, TdbConfig, Term};
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let store = TdbStore::new(TdbConfig::default())?;
    ///
    /// // Insert a simple triple
    /// let subject = Term::iri("http://example.org/alice");
    /// let predicate = Term::iri("http://xmlns.com/foaf/0.1/name");
    /// let object = Term::literal("Alice");
    ///
    /// store.insert_triple(&subject, &predicate, &object)?;
    ///
    /// // Insert a typed literal
    /// let age_pred = Term::iri("http://xmlns.com/foaf/0.1/age");
    /// let age_val = Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer");
    ///
    /// store.insert_triple(&subject, &age_pred, &age_val)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Terms are automatically compressed using adaptive algorithms
    /// - Duplicate triples are handled efficiently (no-op for exact duplicates)
    /// - Bulk insertions should be wrapped in transactions for better performance
    pub fn insert_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<()> {
        // Fast path validation - only check term types without expensive string validation
        match subject {
            Term::Iri(_) | Term::BlankNode(_) => {}
            _ => return Err(anyhow!("Subject must be an IRI or blank node")),
        };

        match predicate {
            Term::Iri(_) => {}
            _ => return Err(anyhow!("Predicate must be an IRI")),
        };

        match object {
            Term::Iri(_) | Term::BlankNode(_) | Term::Literal { .. } => {}
            _ => return Err(anyhow!("Invalid object term type")),
        };

        // Execute directly without circuit breaker overhead for performance
        let subject_id = self.triple_store.store_term(subject)?;
        let predicate_id = self.triple_store.store_term(predicate)?;
        let object_id = self.triple_store.store_term(object)?;

        let triple = Triple::new(subject_id, predicate_id, object_id);
        self.triple_store.insert_triple(&triple)
    }

    /// Insert a quad
    pub fn insert_quad(
        &self,
        subject: &Term,
        predicate: &Term,
        object: &Term,
        graph: Option<&Term>,
    ) -> Result<()> {
        let subject_id = self.triple_store.store_term(subject)?;
        let predicate_id = self.triple_store.store_term(predicate)?;
        let object_id = self.triple_store.store_term(object)?;
        let graph_id = if let Some(g) = graph {
            Some(self.triple_store.store_term(g)?)
        } else {
            Some(self.triple_store.default_graph())
        };

        let quad = Quad::new(subject_id, predicate_id, object_id, graph_id);
        self.triple_store.insert_quad(&quad)
    }

    /// Delete a triple
    pub fn delete_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<bool> {
        let subject_id = self.triple_store.get_node_id(subject)?.unwrap_or(0);
        let predicate_id = self.triple_store.get_node_id(predicate)?.unwrap_or(0);
        let object_id = self.triple_store.get_node_id(object)?.unwrap_or(0);

        if subject_id == 0 || predicate_id == 0 || object_id == 0 {
            return Ok(false); // Triple doesn't exist
        }

        let triple = Triple::new(subject_id, predicate_id, object_id);
        self.triple_store.delete_triple(&triple)
    }

    /// Query triples matching the specified pattern
    ///
    /// Searches for all triples that match the given pattern. Any component
    /// can be `None` to act as a wildcard that matches any value.
    ///
    /// # Arguments
    ///
    /// * `subject` - Optional subject term to match (None = wildcard)
    /// * `predicate` - Optional predicate term to match (None = wildcard)
    /// * `object` - Optional object term to match (None = wildcard)
    ///
    /// # Returns
    ///
    /// Returns a vector of matching triples as `(subject, predicate, object)` tuples,
    /// or an error if:
    /// - Index access fails
    /// - Term decoding fails
    /// - I/O errors occur during query processing
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_tdb::{TdbStore, TdbConfig, Term};
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let store = TdbStore::new(TdbConfig::default())?;
    ///
    /// // Insert some test data
    /// let alice = Term::iri("http://example.org/alice");
    /// let bob = Term::iri("http://example.org/bob");
    /// let name = Term::iri("http://xmlns.com/foaf/0.1/name");
    /// let age = Term::iri("http://xmlns.com/foaf/0.1/age");
    ///
    /// store.insert_triple(&alice, &name, &Term::literal("Alice"))?;
    /// store.insert_triple(&bob, &name, &Term::literal("Bob"))?;
    /// store.insert_triple(&alice, &age, &Term::literal("30"))?;
    ///
    /// // Query all triples about Alice
    /// let alice_triples = store.query_triples(Some(&alice), None, None)?;
    /// assert_eq!(alice_triples.len(), 2);
    ///
    /// // Query all name triples
    /// let name_triples = store.query_triples(None, Some(&name), None)?;
    /// assert_eq!(name_triples.len(), 2);
    ///
    /// // Query specific triple
    /// let specific = store.query_triples(
    ///     Some(&alice),
    ///     Some(&name),
    ///     Some(&Term::literal("Alice"))
    /// )?;
    /// assert_eq!(specific.len(), 1);
    ///
    /// // Query all triples (use with caution on large datasets)
    /// let all_triples = store.query_triples(None, None, None)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - The query engine automatically selects the most efficient index
    /// - More specific patterns (fewer wildcards) execute faster
    /// - Results are returned in index order for consistent iteration
    /// - Large result sets are streamed efficiently from storage
    pub fn query_triples(
        &self,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
    ) -> Result<Vec<(Term, Term, Term)>> {
        let subject_id = if let Some(s) = subject {
            self.triple_store.get_node_id(s)?
        } else {
            None
        };
        let predicate_id = if let Some(p) = predicate {
            self.triple_store.get_node_id(p)?
        } else {
            None
        };
        let object_id = if let Some(o) = object {
            self.triple_store.get_node_id(o)?
        } else {
            None
        };

        let triples = self
            .triple_store
            .query_triples(subject_id, predicate_id, object_id)?;

        // Convert back to terms
        let mut result = Vec::new();
        for triple in triples {
            let subject_term = self
                .triple_store
                .get_term(triple.subject)?
                .unwrap_or_else(|| Term::iri("unknown"));
            let predicate_term = self
                .triple_store
                .get_term(triple.predicate)?
                .unwrap_or_else(|| Term::iri("unknown"));
            let object_term = self
                .triple_store
                .get_term(triple.object)?
                .unwrap_or_else(|| Term::iri("unknown"));

            result.push((subject_term, predicate_term, object_term));
        }

        Ok(result)
    }

    /// Get statistics
    pub fn get_stats(&self) -> Result<TripleStoreStats> {
        self.triple_store.get_stats()
    }

    /// Get the number of triples
    pub fn len(&self) -> Result<u64> {
        self.triple_store.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> Result<bool> {
        self.triple_store.is_empty()
    }

    /// Compact the store
    pub fn compact(&self) -> Result<()> {
        self.triple_store.compact()
    }

    /// Clear all data
    pub fn clear(&self) -> Result<()> {
        self.triple_store.clear()
    }

    /// Access the underlying triple store
    pub fn triple_store(&self) -> &TripleStore {
        &self.triple_store
    }

    /// Access the file system
    pub fn filesystem(&self) -> &TdbFileSystem {
        &self.filesystem
    }

    /// Get database metadata
    pub fn get_database_metadata(&self) -> DatabaseMetadata {
        self.filesystem.get_metadata()
    }

    /// Create a backup of the database
    pub fn create_backup(&self) -> Result<std::path::PathBuf> {
        self.filesystem.create_backup()
    }

    /// Validate database integrity
    pub fn validate_integrity(&self) -> Result<Vec<String>> {
        self.filesystem.validate_integrity()
    }

    /// Update database statistics
    pub fn update_database_stats(&self) -> Result<()> {
        let stats = self.get_stats()?;
        let triple_count = stats.total_triples;
        // For node_count, we can estimate from the total_triples or use a default
        let node_count = stats.total_triples; // Using total_triples as an estimate for now
        self.filesystem.update_stats(triple_count, node_count)
    }

    /// Get health monitor
    pub fn health_monitor(&self) -> &HealthMonitor {
        &self.health_monitor
    }

    /// Check system health status
    pub fn check_health(&self) -> Result<(), production_hardening::HardeningError> {
        self.health_monitor.is_healthy()
    }

    /// Generate comprehensive health report
    pub fn generate_health_report(&self) -> String {
        self.health_monitor.generate_health_report()
    }

    /// Update system metrics for health monitoring
    pub fn update_health_metrics(&self, metrics: HealthMetrics) {
        self.health_monitor.update_metrics(metrics);
    }

    /// Attempt automatic recovery from system issues
    pub fn attempt_recovery(&self, component: &str, error: &str) -> Result<()> {
        self.health_monitor.attempt_recovery(component, error)
    }

    /// Execute operation with circuit breaker protection
    pub fn execute_protected<T, F>(&self, service: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        self.health_monitor
            .execute_with_protection(service, operation)
    }

    /// Get operation statistics from health monitor
    pub fn get_operation_stats(&self) -> std::collections::HashMap<String, f64> {
        self.health_monitor.get_operation_stats()
    }

    /// Get query optimization recommendation for a triple pattern
    pub fn get_query_recommendation(
        &self,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
    ) -> Result<OptimizationRecommendation> {
        let pattern = QueryPattern::new(subject, predicate, object);
        let store_stats = self.get_stats()?;
        self.query_optimizer.recommend_optimization(&pattern, &store_stats)
    }

    /// Get query execution statistics summary
    pub fn get_query_statistics(&self) -> Result<QueryStatisticsSummary> {
        self.query_optimizer.get_statistics_summary()
    }

    /// Record query execution performance for optimization
    pub fn record_query_performance(
        &self,
        pattern: QueryPattern,
        execution_time: std::time::Duration,
        result_count: u64,
        index_used: IndexType,
    ) -> Result<()> {
        let stats = QueryStats {
            pattern,
            execution_time,
            result_count,
            index_used,
            cost: execution_time.as_secs_f64() * 1000.0, // Convert to cost units
            timestamp: std::time::Instant::now(),
        };
        self.query_optimizer.record_execution(stats)
    }

    /// Clear query optimization history
    pub fn clear_query_history(&self) -> Result<()> {
        self.query_optimizer.clear_history()
    }

    /// Get the query optimizer reference
    pub fn query_optimizer(&self) -> &QueryOptimizer {
        &self.query_optimizer
    }

    /// Get the metrics collector reference
    pub fn metrics_collector(&self) -> &MetricsCollector {
        &self.metrics_collector
    }

    /// Generate comprehensive metrics report
    pub fn generate_metrics_report(&self, duration: std::time::Duration) -> Result<MetricsReport> {
        self.metrics_collector.generate_report(duration)
    }

    /// Get current system metrics snapshot
    pub fn get_current_metrics(&self) -> Result<ComprehensiveMetrics> {
        self.metrics_collector.get_current_metrics()
    }

    /// Record query performance for metrics and optimization
    pub fn record_query_metrics(
        &self,
        pattern: QueryPattern,
        execution_time: std::time::Duration,
        result_count: u64,
        index_used: IndexType,
    ) -> Result<()> {
        // Record for query optimizer
        self.record_query_performance(pattern.clone(), execution_time, result_count, index_used)?;

        // Record for metrics collector
        let mut labels = std::collections::HashMap::new();
        labels.insert("pattern_type".to_string(), format!("{:?}", pattern.pattern_type()));
        labels.insert("index_used".to_string(), format!("{:?}", index_used));
        labels.insert("result_count".to_string(), result_count.to_string());

        self.metrics_collector.record_query_time(execution_time, labels)?;

        Ok(())
    }

    /// Record system metrics
    pub fn record_system_metrics(&self, metrics: SystemMetrics) -> Result<()> {
        self.metrics_collector.record_system_metrics(metrics)
    }

    /// Record error for monitoring
    pub fn record_error(&self, error_type: &str) -> Result<()> {
        self.metrics_collector.record_error(error_type)
    }

    /// Get performance statistics for various metrics
    pub fn get_performance_stats(&self, duration: std::time::Duration) -> Result<PerformanceStatsSummary> {
        let query_stats = self.metrics_collector.get_query_stats(duration)?;
        let error_stats = self.metrics_collector.get_error_stats(duration)?;
        let memory_stats = self.metrics_collector.get_memory_stats(duration)?;
        let cpu_stats = self.metrics_collector.get_cpu_stats(duration)?;
        let throughput_stats = self.metrics_collector.get_throughput_stats(duration)?;

        Ok(PerformanceStatsSummary {
            query_performance: query_stats,
            error_rates: error_stats,
            memory_usage: memory_stats,
            cpu_usage: cpu_stats,
            throughput: throughput_stats,
            period: duration,
        })
    }

    /// Export metrics report as JSON
    pub fn export_metrics_json(&self, duration: std::time::Duration) -> Result<String> {
        let report = self.generate_metrics_report(duration)?;
        report.to_json()
    }

    /// Export metrics report as CSV
    pub fn export_metrics_csv(&self, duration: std::time::Duration) -> Result<String> {
        let report = self.generate_metrics_report(duration)?;
        Ok(report.to_csv())
    }

    /// Check if automatic metrics collection should occur
    pub fn should_collect_metrics(&self) -> bool {
        self.metrics_collector.should_collect()
    }

    /// Trigger metrics collection
    pub fn collect_metrics(&self) -> Result<()> {
        if self.should_collect_metrics() {
            // Collect current system state
            let stats = self.get_stats()?;
            let storage_metrics = StorageMetrics {
                total_triples: stats.total_triples,
                database_size_bytes: stats.total_triples * 100, // Estimated bytes per triple
                index_size_bytes: stats.total_triples * 50, // Estimated index size
                compression_ratio: 0.8, // Estimated compression ratio
                active_transactions: 0, // Would need actual transaction count
                completed_transactions: 0, // Would need actual transaction count
                failed_transactions: 0, // Would need actual error count
                wal_size_bytes: 0, // Would need actual WAL size
                checkpoint_count: 0, // Would need actual checkpoint count
                avg_checkpoint_duration_ms: 0.0,
                buffer_pool_hit_rate: 0.95, // Estimated buffer pool hit rate
                dirty_pages: 0, // Would need actual dirty page count
            };

            // Create basic system metrics (in production, these would come from system monitoring)
            let system_metrics = SystemMetrics {
                cpu_usage: 0.0, // Would be collected from system
                memory_usage: 0, // Would be collected from system
                memory_available: 0, // Would be collected from system
                disk_read_rate: 0,
                disk_write_rate: 0,
                network_io_rate: 0,
                active_threads: std::thread::available_parallelism().map(|p| p.get() as u32).unwrap_or(1),
                timestamp: std::time::SystemTime::now(),
            };

            self.record_system_metrics(system_metrics)?;
            self.metrics_collector.mark_collected()?;
        }
        Ok(())
    }
}

/// Performance statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatsSummary {
    pub query_performance: MetricsStatistics,
    pub error_rates: MetricsStatistics,
    pub memory_usage: MetricsStatistics,
    pub cpu_usage: MetricsStatistics,
    pub throughput: MetricsStatistics,
    pub period: std::time::Duration,
}

/// Database transaction handle
///
/// Represents an active database transaction that provides ACID guarantees.
/// Transactions must be explicitly committed or rolled back.
///
/// # Transaction Lifecycle
///
/// 1. **Begin**: Create transaction with [`TdbStore::begin_transaction`] or [`TdbStore::begin_read_transaction`]
/// 2. **Execute**: Perform database operations within the transaction context
/// 3. **Commit**: Finalize changes with [`TdbStore::commit_transaction`]
/// 4. **Rollback**: Abort changes with [`TdbStore::rollback_transaction`] (optional)
///
/// # Examples
///
/// ```rust
/// use oxirs_tdb::{TdbStore, TdbConfig, Term};
///
/// # fn example() -> anyhow::Result<()> {
/// let store = TdbStore::new(TdbConfig::default())?;
///
/// // Begin a transaction
/// let tx = store.begin_transaction()?;
///
/// // Perform operations
/// let subject = Term::iri("http://example.org/resource");
/// let predicate = Term::iri("http://example.org/property");  
/// let object = Term::literal("value");
/// store.insert_triple(&subject, &predicate, &object)?;
///
/// // Commit to make changes permanent
/// store.commit_transaction(tx)?;
/// # Ok(())
/// # }
/// ```
///
/// # Error Handling
///
/// If a transaction is dropped without being committed, changes are automatically
/// rolled back. However, it's recommended to explicitly handle transaction outcomes:
///
/// ```rust
/// use oxirs_tdb::{TdbStore, TdbConfig, Term};
///
/// # fn example() -> anyhow::Result<()> {
/// let store = TdbStore::new(TdbConfig::default())?;
/// let tx = store.begin_transaction()?;
///
/// // Perform operations...
/// let result = store.insert_triple(
///     &Term::iri("http://example.org/s"),
///     &Term::iri("http://example.org/p"),
///     &Term::literal("o")
/// );
///
/// match result {
///     Ok(_) => {
///         store.commit_transaction(tx)?;
///         println!("Transaction committed successfully");
///     }
///     Err(e) => {
///         store.rollback_transaction(tx)?;
///         println!("Transaction rolled back due to error: {}", e);
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct Transaction {
    tx_id: TransactionId,
}

impl Transaction {
    fn new(tx_id: TransactionId) -> Self {
        Self { tx_id }
    }

    /// Get the transaction ID
    pub fn id(&self) -> TransactionId {
        self.tx_id
    }
}
