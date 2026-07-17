//! Store configuration, statistics types, and metric structures for oxirs-tdb.

use crate::error::{Result, TdbError};
use crate::storage::page::PAGE_SIZE;
use crate::store::store_params::{CompressionAlgorithm, StoreParams};
use std::path::Path;
use std::path::PathBuf;

/// TDB Store configuration.
///
/// This is the concrete, engine-level configuration the store is built from. The
/// broader, presettable [`StoreParams`](crate::store::StoreParams) is converted
/// into it via [`TdbConfig::from_store_params`], which is how
/// [`TdbStore::open_with_params`](crate::store::TdbStore::open_with_params)
/// threads every honored parameter (buffer-pool size, bloom sizing, cache size,
/// statistics sampling, query-monitor thresholds, compression, spatial fan-out,
/// WAL, direct I/O) down to the subsystems that consume them.
#[derive(Debug, Clone)]
pub struct TdbConfig {
    /// Directory for storage files
    pub data_dir: PathBuf,
    /// Buffer pool size (number of pages)
    pub buffer_pool_size: usize,
    /// Enable compression
    pub enable_compression: bool,
    /// Block/page compression algorithm recorded for the compression subsystem.
    ///
    /// Mirrors [`StoreParams::compression_algorithm`](crate::store::StoreParams).
    /// `enable_compression` (which gates the URI prefix compressor) is derived
    /// from this being non-[`None`](CompressionAlgorithm::None) in
    /// [`from_store_params`](Self::from_store_params).
    pub compression_algorithm: CompressionAlgorithm,
    /// Compression level for the selected algorithm. Mirrors
    /// [`StoreParams::compression_level`](crate::store::StoreParams).
    pub compression_level: u32,
    /// Enable bloom filters
    pub enable_bloom_filters: bool,
    /// Bloom filter false positive rate
    pub bloom_filter_fpr: f64,
    /// Expected element count each per-index bloom filter is sized for.
    ///
    /// Threaded to [`BloomFilter::new`](crate::compression::BloomFilter) at open
    /// (replacing a previously hardcoded capacity). Mirrors
    /// [`StoreParams::bloom_filter_size_per_index`](crate::store::StoreParams).
    pub bloom_filter_size_per_index: usize,
    /// Enable query result caching
    pub enable_query_cache: bool,
    /// Maximum number of cached query results. Threaded to the query cache's
    /// `max_entries`. Mirrors
    /// [`StoreParams::query_cache_size`](crate::store::StoreParams).
    pub query_cache_size: usize,
    /// Enable statistics collection
    pub enable_statistics: bool,
    /// Statistics sampling rate (0.0–1.0). Threaded to the statistics collector;
    /// a rate below 1.0 enables sampling. Mirrors
    /// [`StoreParams::statistics_sample_rate`](crate::store::StoreParams).
    pub statistics_sample_rate: f64,
    /// Enable query monitoring
    pub enable_query_monitoring: bool,
    /// Slow-query logging threshold in milliseconds. Threaded to the query
    /// monitor. Mirrors
    /// [`StoreParams::slow_query_threshold_ms`](crate::store::StoreParams).
    pub slow_query_threshold_ms: u64,
    /// Query timeout in milliseconds. Threaded to the query monitor. Mirrors
    /// [`StoreParams::query_timeout_ms`](crate::store::StoreParams).
    pub query_timeout_ms: u64,
    /// Enable spatial indexing (GeoSPARQL)
    pub enable_spatial_indexing: bool,
    /// Advisory maximum entries per spatial-index node (R*-tree fan-out).
    /// Threaded to [`SpatialIndex::with_max_entries`](crate::index::SpatialIndex).
    /// Mirrors
    /// [`StoreParams::spatial_index_max_entries`](crate::store::StoreParams).
    pub spatial_index_max_entries: usize,
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
    /// WAL in-memory buffer size hint in bytes. Recorded from
    /// [`StoreParams::wal_buffer_size`](crate::store::StoreParams) so a later
    /// WAL revision can consume it; the current WAL buffers entries in memory and
    /// does not yet act on this hint.
    pub wal_buffer_size: usize,
    /// Opt-in OS page-cache-bypassing direct I/O for large sequential scans
    /// ([`DirectIOFile`](crate::storage::direct_io::DirectIOFile)). Default
    /// `false`; the default path stays pure-std. Not wired into the page store
    /// (alignment hazards). Mirrors
    /// [`StoreParams::enable_direct_io`](crate::store::StoreParams).
    pub enable_direct_io: bool,
}

impl TdbConfig {
    /// Create default configuration
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            buffer_pool_size: 1000,
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Lz4,
            compression_level: 3,
            enable_bloom_filters: true,
            bloom_filter_fpr: 0.01,
            bloom_filter_size_per_index: 100_000,
            enable_query_cache: true,
            query_cache_size: 1000,
            enable_statistics: true,
            statistics_sample_rate: 1.0,
            enable_query_monitoring: true,
            slow_query_threshold_ms: 100,
            query_timeout_ms: 30_000,
            enable_spatial_indexing: true,
            spatial_index_max_entries: 1000,
            enable_quad_indexes: true,
            enable_wal: true,
            wal_sync_on_commit: false,
            wal_buffer_size: 1024 * 1024,
            enable_direct_io: false,
        }
    }

    /// Build an engine [`TdbConfig`] from the presettable
    /// [`StoreParams`](crate::store::StoreParams), honoring every parameter the
    /// storage engine actually consumes.
    ///
    /// The params are validated first. `page_size` is treated specially: the
    /// engine's on-disk page size is a compile-time constant
    /// ([`PAGE_SIZE`](crate::storage::page::PAGE_SIZE)) baked into every page
    /// layout and the superblock, so it cannot be reconfigured per store; a
    /// `StoreParams` requesting a different `page_size` is rejected loudly rather
    /// than silently ignored.
    pub fn from_store_params(params: &StoreParams) -> Result<Self> {
        params.validate()?;

        if params.page_size != PAGE_SIZE {
            return Err(TdbError::InvalidConfiguration(format!(
                "page_size must equal the compile-time engine page size {PAGE_SIZE} (the page \
                 layout and superblock are fixed at that size); got {}",
                params.page_size
            )));
        }

        Ok(Self {
            data_dir: params.data_dir.clone(),
            buffer_pool_size: params.buffer_pool_size,
            // The URI prefix compressor is engaged whenever a real (non-None)
            // block compression algorithm is selected, mirroring StoreParams'
            // own `enable_compression = algorithm != None` derivation.
            enable_compression: params.enable_compression
                && params.compression_algorithm != CompressionAlgorithm::None,
            compression_algorithm: params.compression_algorithm,
            compression_level: params.compression_level,
            enable_bloom_filters: params.enable_bloom_filters,
            bloom_filter_fpr: params.bloom_filter_fpr,
            bloom_filter_size_per_index: params.bloom_filter_size_per_index,
            enable_query_cache: params.enable_query_cache,
            query_cache_size: params.query_cache_size,
            enable_statistics: params.enable_statistics,
            statistics_sample_rate: params.statistics_sample_rate,
            enable_query_monitoring: params.enable_query_monitoring,
            slow_query_threshold_ms: params.slow_query_threshold_ms,
            query_timeout_ms: params.query_timeout_ms,
            enable_spatial_indexing: params.enable_spatial_indexing,
            spatial_index_max_entries: params.spatial_index_max_entries,
            enable_quad_indexes: params.enable_quad_indexes,
            enable_wal: params.enable_wal,
            // StoreParams has no per-commit-fsync knob; keep the amortized F3
            // default (durability via the next checkpoint / Drop).
            wal_sync_on_commit: false,
            wal_buffer_size: params.wal_buffer_size,
            enable_direct_io: params.enable_direct_io,
        })
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
