//! Store configuration, statistics types, and metric structures for oxirs-tdb.

use std::path::Path;
use std::path::PathBuf;

/// TDB Store configuration
#[derive(Debug, Clone)]
pub struct TdbConfig {
    /// Directory for storage files
    pub data_dir: PathBuf,
    /// Buffer pool size (number of pages)
    pub buffer_pool_size: usize,
    /// Enable compression
    pub enable_compression: bool,
    /// Enable bloom filters
    pub enable_bloom_filters: bool,
    /// Bloom filter false positive rate
    pub bloom_filter_fpr: f64,
    /// Enable query result caching
    pub enable_query_cache: bool,
    /// Enable statistics collection
    pub enable_statistics: bool,
    /// Enable query monitoring
    pub enable_query_monitoring: bool,
    /// Enable spatial indexing (GeoSPARQL)
    pub enable_spatial_indexing: bool,
    /// Enable named-graph (quad) indexes (GSPO/GPOS/GOSP).
    ///
    /// When enabled (the default), the store maintains quad indexes so the
    /// quad API (`insert_quad`/`scan_quads`/…) can store and query named
    /// graphs. The default (unnamed) graph is always served by the triple
    /// indexes regardless of this flag. Mirrors
    /// [`StoreParams::enable_quad_indexes`](crate::store::StoreParams).
    pub enable_quad_indexes: bool,
    /// Enable write-ahead logging (WAL) with crash-recovery replay (F3).
    ///
    /// When enabled (the default), every mutating operation (triple/quad insert
    /// and delete, single and bulk) is bracketed in a WAL transaction
    /// (`Begin -> DataOp -> Commit`) so that committed writes survive a crash
    /// *before the next* [`sync`](crate::store::TdbStore::sync): on reopen the
    /// committed operations are replayed on top of the last checkpoint. When
    /// disabled, durability is checkpoint-only (data survives only a successful
    /// `sync()`), matching the pre-F3 behaviour. Mirrors
    /// [`StoreParams::enable_wal`](crate::store::StoreParams) so a `StoreParams`
    /// caller can thread it through.
    pub enable_wal: bool,
    /// Whether each *single* mutating operation fsyncs the WAL at commit.
    ///
    /// The default is `false`: single-operation commits append to the WAL
    /// without forcing an fsync, and durability is amortized to the next
    /// [`sync`](crate::store::TdbStore::sync)/checkpoint (or process exit via
    /// `Drop`). Buffered WAL appends already survive a process crash that is not
    /// a hardware power loss (the bytes reach the OS on `write`), which is what
    /// the crash-recovery contract simulates. Set to `true` for
    /// power-loss-durable single-write semantics at the cost of an fsync per
    /// operation. Bulk operations are always a single WAL transaction and never
    /// fsync per element regardless of this flag.
    pub wal_sync_on_commit: bool,
}

impl TdbConfig {
    /// Create default configuration
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            buffer_pool_size: 1000,
            enable_compression: true,
            enable_bloom_filters: true,
            bloom_filter_fpr: 0.01,
            enable_query_cache: true,
            enable_statistics: true,
            enable_query_monitoring: true,
            enable_spatial_indexing: true,
            enable_quad_indexes: true,
            enable_wal: true,
            wal_sync_on_commit: false,
        }
    }

    /// Enable/disable query result caching
    pub fn with_query_cache(mut self, enable: bool) -> Self {
        self.enable_query_cache = enable;
        self
    }

    /// Enable/disable statistics collection
    pub fn with_statistics(mut self, enable: bool) -> Self {
        self.enable_statistics = enable;
        self
    }

    /// Enable/disable query monitoring
    pub fn with_query_monitoring(mut self, enable: bool) -> Self {
        self.enable_query_monitoring = enable;
        self
    }

    /// Set buffer pool size
    pub fn with_buffer_pool_size(mut self, size: usize) -> Self {
        self.buffer_pool_size = size;
        self
    }

    /// Enable/disable compression
    pub fn with_compression(mut self, enable: bool) -> Self {
        self.enable_compression = enable;
        self
    }

    /// Enable/disable bloom filters
    pub fn with_bloom_filters(mut self, enable: bool) -> Self {
        self.enable_bloom_filters = enable;
        self
    }

    /// Enable/disable spatial indexing
    pub fn with_spatial_indexing(mut self, enable: bool) -> Self {
        self.enable_spatial_indexing = enable;
        self
    }

    /// Enable/disable named-graph (quad) indexes.
    pub fn with_quad_indexes(mut self, enable: bool) -> Self {
        self.enable_quad_indexes = enable;
        self
    }

    /// Enable/disable write-ahead logging with crash-recovery replay.
    pub fn with_wal(mut self, enable: bool) -> Self {
        self.enable_wal = enable;
        self
    }

    /// Set whether each single mutating operation fsyncs the WAL at commit.
    pub fn with_wal_sync_on_commit(mut self, enable: bool) -> Self {
        self.wal_sync_on_commit = enable;
        self
    }
}

/// Type alias for query results with statistics
pub type QueryResultWithStats = (
    Vec<(
        crate::dictionary::Term,
        crate::dictionary::Term,
        crate::dictionary::Term,
    )>,
    crate::query_hints::QueryStats,
);

/// TDB Store statistics (basic)
#[derive(Debug, Clone)]
pub struct TdbStats {
    /// Number of triples
    pub triple_count: usize,
    /// Dictionary size (number of unique terms)
    pub dictionary_size: usize,
    /// Bloom filter statistics
    pub bloom_filter_stats: Option<crate::compression::BloomFilterStats>,
    /// Compression statistics
    pub compression_stats: Option<crate::compression::PrefixCompressionStats>,
}

/// Enhanced TDB Store statistics with comprehensive metrics
#[derive(Debug)]
pub struct TdbEnhancedStats {
    /// Basic triple and dictionary statistics
    pub basic: TdbStats,
    /// Buffer pool performance metrics
    pub buffer_pool: crate::storage::buffer_pool::BufferPoolStats,
    /// Storage metrics
    pub storage: StorageMetrics,
    /// Transaction metrics
    pub transaction: TransactionMetrics,
    /// Index metrics
    pub index: IndexMetrics,
}

/// Storage-level metrics
#[derive(Debug, Clone, Copy)]
pub struct StorageMetrics {
    /// Total storage size in bytes (estimated)
    pub total_size_bytes: u64,
    /// Number of pages allocated
    pub pages_allocated: usize,
    /// Page size in bytes
    pub page_size: usize,
    /// Estimated memory usage in bytes
    pub memory_usage_bytes: usize,
}

impl StorageMetrics {
    /// Calculate storage efficiency (used space / allocated space)
    pub fn efficiency(&self) -> f64 {
        if self.pages_allocated == 0 {
            0.0
        } else {
            let allocated = (self.pages_allocated * self.page_size) as f64;
            if allocated == 0.0 {
                0.0
            } else {
                self.total_size_bytes as f64 / allocated
            }
        }
    }

    /// Calculate fragmentation percentage
    pub fn fragmentation(&self) -> f64 {
        (1.0 - self.efficiency()) * 100.0
    }
}

/// Transaction-level metrics
#[derive(Debug, Clone, Copy)]
pub struct TransactionMetrics {
    /// Number of currently active transactions
    pub active_transactions: usize,
    /// Whether WAL is enabled
    pub wal_enabled: bool,
    /// Estimated WAL size in bytes
    pub wal_size_bytes: u64,
}

/// Index-level metrics
#[derive(Debug, Clone, Copy)]
pub struct IndexMetrics {
    /// Number of entries in SPO index (estimated)
    pub spo_entries: usize,
    /// Number of entries in POS index (estimated)
    pub pos_entries: usize,
    /// Number of entries in OSP index (estimated)
    pub osp_entries: usize,
    /// Whether indexes are consistent
    pub indexes_consistent: bool,
}

impl IndexMetrics {
    /// Total index entries across all indexes
    pub fn total_entries(&self) -> usize {
        self.spo_entries + self.pos_entries + self.osp_entries
    }

    /// Average entries per index
    pub fn avg_entries_per_index(&self) -> f64 {
        self.total_entries() as f64 / 3.0
    }
}
