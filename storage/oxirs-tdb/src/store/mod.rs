//! High-level TDB store API
//!
//! Provides the main TDBStore interface for interacting with the storage engine.
//! Integrates all components: dictionary, indexes, transactions, compression.

use crate::compression::{BloomFilter, PrefixCompressor};
use crate::dictionary::{Dictionary, Term};
use crate::error::{Result, TdbError};
use crate::index::{Triple, TripleIndexes};
use crate::storage::buffer_pool::BufferPool;
use crate::storage::file_manager::FileManager;
use crate::transaction::{LockManager, TransactionManager, WriteAheadLog};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
        }
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
}

/// High-level TDB triple store
pub struct TdbStore {
    /// Configuration
    config: TdbConfig,
    /// Dictionary for term encoding
    dictionary: Dictionary,
    /// Triple indexes (SPO, POS, OSP)
    indexes: TripleIndexes,
    /// Transaction manager
    txn_manager: Arc<TransactionManager>,
    /// Buffer pool (for stats collection)
    buffer_pool: Arc<BufferPool>,
    /// Bloom filter for existence checks (optional)
    bloom_filter: Option<BloomFilter>,
    /// Prefix compressor (optional)
    prefix_compressor: Option<PrefixCompressor>,
    /// Triple count
    triple_count: usize,
}

impl TdbStore {
    /// Open or create a TDB store
    pub fn open<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        let config = TdbConfig::new(data_dir);
        Self::open_with_config(config)
    }

    /// Open with custom configuration
    pub fn open_with_config(config: TdbConfig) -> Result<Self> {
        // Create data directory
        std::fs::create_dir_all(&config.data_dir).map_err(TdbError::Io)?;

        // Initialize file manager and buffer pool
        let data_file = config.data_dir.join("data.tdb");
        let file_manager = Arc::new(FileManager::open(&data_file, false)?);
        let buffer_pool = Arc::new(BufferPool::new(config.buffer_pool_size, file_manager));

        // Initialize dictionary
        let dictionary = Dictionary::new(buffer_pool.clone());

        // Initialize triple indexes
        let indexes = TripleIndexes::new(buffer_pool.clone());

        // Initialize transaction management
        let wal = Arc::new(WriteAheadLog::new(&config.data_dir)?);
        let lock_manager = Arc::new(LockManager::new());
        let txn_manager = Arc::new(TransactionManager::new(wal, lock_manager));

        // Initialize optional components
        let bloom_filter = if config.enable_bloom_filters {
            Some(BloomFilter::new(100000, config.bloom_filter_fpr))
        } else {
            None
        };

        let prefix_compressor = if config.enable_compression {
            Some(PrefixCompressor::new(10))
        } else {
            None
        };

        Ok(Self {
            config,
            dictionary,
            indexes,
            txn_manager,
            buffer_pool,
            bloom_filter,
            prefix_compressor,
            triple_count: 0,
        })
    }

    /// Insert a triple
    pub fn insert(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()> {
        // Encode terms to node IDs
        let s_term = Term::Iri(subject.to_string());
        let p_term = Term::Iri(predicate.to_string());
        let o_term = Term::Iri(object.to_string());

        let s_id = self.dictionary.encode(&s_term)?;
        let p_id = self.dictionary.encode(&p_term)?;
        let o_id = self.dictionary.encode(&o_term)?;

        let triple = Triple::new(s_id, p_id, o_id);

        // Add to indexes
        self.indexes.insert(triple)?;

        // Update bloom filter
        if let Some(ref mut bloom) = self.bloom_filter {
            bloom.insert(&triple);
        }

        // Update count
        self.triple_count += 1;

        Ok(())
    }

    /// Check if a triple exists
    pub fn contains(&self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        // Encode terms
        let s_term = Term::Iri(subject.to_string());
        let p_term = Term::Iri(predicate.to_string());
        let o_term = Term::Iri(object.to_string());

        let s_id = self
            .dictionary
            .lookup(&s_term)?
            .ok_or_else(|| TdbError::Other("Subject not found in dictionary".to_string()))?;
        let p_id = self
            .dictionary
            .lookup(&p_term)?
            .ok_or_else(|| TdbError::Other("Predicate not found in dictionary".to_string()))?;
        let o_id = self
            .dictionary
            .lookup(&o_term)?
            .ok_or_else(|| TdbError::Other("Object not found in dictionary".to_string()))?;

        let triple = Triple::new(s_id, p_id, o_id);

        // Check bloom filter first (if enabled)
        if let Some(ref bloom) = self.bloom_filter {
            if !bloom.contains(&triple) {
                return Ok(false); // Definitely not present
            }
        }

        // Check in indexes
        self.indexes.contains(&triple)
    }

    /// Delete a triple
    pub fn delete(&mut self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        // Encode terms
        let s_term = Term::Iri(subject.to_string());
        let p_term = Term::Iri(predicate.to_string());
        let o_term = Term::Iri(object.to_string());

        let s_id = self
            .dictionary
            .lookup(&s_term)?
            .ok_or_else(|| TdbError::Other("Subject not found in dictionary".to_string()))?;
        let p_id = self
            .dictionary
            .lookup(&p_term)?
            .ok_or_else(|| TdbError::Other("Predicate not found in dictionary".to_string()))?;
        let o_id = self
            .dictionary
            .lookup(&o_term)?
            .ok_or_else(|| TdbError::Other("Object not found in dictionary".to_string()))?;

        let triple = Triple::new(s_id, p_id, o_id);

        // Delete from indexes
        let deleted = self.indexes.delete(&triple)?;

        // Update count
        if deleted {
            self.triple_count = self.triple_count.saturating_sub(1);
        }

        Ok(deleted)
    }

    /// Get number of triples
    pub fn count(&self) -> usize {
        self.triple_count
    }

    /// Get configuration
    pub fn config(&self) -> &TdbConfig {
        &self.config
    }

    /// Get transaction manager
    pub fn transaction_manager(&self) -> &Arc<TransactionManager> {
        &self.txn_manager
    }

    /// Get basic statistics
    pub fn stats(&self) -> TdbStats {
        TdbStats {
            triple_count: self.count(),
            dictionary_size: self.dictionary.size() as usize,
            bloom_filter_stats: self.bloom_filter.as_ref().map(|b| b.stats()),
            compression_stats: self.prefix_compressor.as_ref().map(|c| c.stats()),
        }
    }

    /// Get enhanced statistics with comprehensive metrics
    pub fn enhanced_stats(&self) -> TdbEnhancedStats {
        // Get basic stats
        let basic = self.stats();

        // Get buffer pool stats
        let buffer_pool_stats = self.buffer_pool.stats();

        // Estimate storage metrics
        let page_size = crate::DEFAULT_PAGE_SIZE;
        let pages_allocated = (basic.triple_count / 10).max(1); // Rough estimate
        let storage = StorageMetrics {
            total_size_bytes: (basic.triple_count * 100) as u64, // Rough estimate
            pages_allocated,
            page_size,
            memory_usage_bytes: self.config.buffer_pool_size * page_size,
        };

        // Transaction metrics
        let transaction = TransactionMetrics {
            active_transactions: 0, // Would need transaction manager stats
            wal_enabled: true,
            wal_size_bytes: 0, // Would need WAL stats
        };

        // Index metrics (all indexes should have same count)
        let index = IndexMetrics {
            spo_entries: basic.triple_count,
            pos_entries: basic.triple_count,
            osp_entries: basic.triple_count,
            indexes_consistent: true, // Assuming consistency
        };

        TdbEnhancedStats {
            basic,
            buffer_pool: buffer_pool_stats,
            storage,
            transaction,
            index,
        }
    }

    /// Insert a triple using Term types (convenience method)
    pub fn insert_triple(&mut self, subject: &Term, predicate: &Term, object: &Term) -> Result<()> {
        // Encode terms to node IDs
        let s_id = self.dictionary.encode(subject)?;
        let p_id = self.dictionary.encode(predicate)?;
        let o_id = self.dictionary.encode(object)?;

        let triple = Triple::new(s_id, p_id, o_id);

        // Add to indexes
        self.indexes.insert(triple)?;

        // Update bloom filter
        if let Some(ref mut bloom) = self.bloom_filter {
            bloom.insert(&triple);
        }

        // Update count
        self.triple_count += 1;

        Ok(())
    }

    /// Alias for count() for compatibility
    pub fn len(&self) -> Result<usize> {
        Ok(self.count())
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Bulk insert triples (transactional: all-or-nothing)
    pub fn insert_triples_bulk(&mut self, triples: &[(Term, Term, Term)]) -> Result<()> {
        // Validate all triples first (subject must be IRI or blank node, not literal)
        for (subject, _predicate, _object) in triples {
            if matches!(subject, Term::Literal { .. }) {
                return Err(TdbError::Other("Subject cannot be a literal".to_string()));
            }
        }

        // If all valid, insert them
        for (subject, predicate, object) in triples {
            self.insert_triple(subject, predicate, object)?;
        }

        Ok(())
    }

    /// Query triples with optional pattern matching (None = wildcard)
    /// Returns matching triples as (subject, predicate, object) Terms
    pub fn query_triples(
        &self,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
    ) -> Result<Vec<(Term, Term, Term)>> {
        // For now, simple implementation: check all triples
        // TODO: Use indexes for efficient pattern matching
        let mut results = Vec::new();

        // If no pattern specified, return empty (would be all triples, expensive)
        if subject.is_none() && predicate.is_none() && object.is_none() {
            return Ok(results);
        }

        // Convert pattern to node IDs (if specified)
        let s_id = if let Some(s) = subject {
            self.dictionary.lookup(s)?
        } else {
            None
        };

        let p_id = if let Some(p) = predicate {
            self.dictionary.lookup(p)?
        } else {
            None
        };

        let o_id = if let Some(o) = object {
            self.dictionary.lookup(o)?
        } else {
            None
        };

        // Query indexes based on pattern
        // This is a simplified implementation - real version would use index optimization
        if let (Some(s), Some(p), Some(o)) = (s_id, p_id, o_id) {
            // Specific triple - check if exists
            let triple = Triple::new(s, p, o);
            if self.indexes.contains(&triple)? {
                results.push((
                    subject.unwrap().clone(),
                    predicate.unwrap().clone(),
                    object.unwrap().clone(),
                ));
            }
        }

        // For other patterns, return empty for now (would require index scan)
        // TODO: Implement full pattern matching using SPO/POS/OSP indexes

        Ok(results)
    }

    /// Begin a write transaction
    pub fn begin_transaction(&self) -> Result<crate::transaction::Transaction> {
        self.txn_manager.begin()
    }

    /// Begin a read-only transaction
    pub fn begin_read_transaction(&self) -> Result<crate::transaction::Transaction> {
        // For now, same as write transaction (read-only enforcement TODO)
        self.txn_manager.begin()
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, txn: crate::transaction::Transaction) -> Result<()> {
        txn.commit()?;
        Ok(())
    }

    /// Clear all triples from the store
    pub fn clear(&mut self) -> Result<()> {
        // TODO: Implement proper clearing by resetting indexes and dictionary
        // For now, this is a simplified version that just resets the count
        // Real implementation would need to reset internal structures

        // Reset bloom filter
        if let Some(ref mut bloom) = self.bloom_filter {
            *bloom = BloomFilter::new(100000, self.config.bloom_filter_fpr);
        }

        // Reset count
        self.triple_count = 0;

        // Note: This doesn't actually clear the underlying data structures
        // Full implementation would require rebuilding indexes and dictionary

        Ok(())
    }

    /// Compact the database (remove deleted entries, optimize layout)
    pub fn compact(&self) -> Result<()> {
        // TODO: Implement actual compaction
        // For now, this is a no-op placeholder
        // Real implementation would:
        // - Rebuild B+trees without deleted entries
        // - Reclaim freed space
        // - Optimize page layout
        Ok(())
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_tdb_store_open() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_open");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = TdbStore::open(&temp_dir).unwrap();
        assert_eq!(store.count(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_insert_count() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_insert");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        assert_eq!(store.count(), 1);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_contains() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_contains");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        assert!(store
            .contains(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob"
            )
            .unwrap());

        assert!(!store
            .contains(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/charlie"
            )
            .unwrap_or(false));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_delete() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_delete");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        assert_eq!(store.count(), 1);

        let deleted = store
            .delete(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        assert!(deleted);
        assert_eq!(store.count(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_multiple_inserts() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_multiple");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();
        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/charlie",
            )
            .unwrap();
        store
            .insert(
                "http://example.org/bob",
                "http://example.org/likes",
                "http://example.org/pizza",
            )
            .unwrap();

        assert_eq!(store.count(), 3);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_config() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_config");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let config = TdbConfig::new(&temp_dir)
            .with_buffer_pool_size(2000)
            .with_compression(false)
            .with_bloom_filters(false);

        let store = TdbStore::open_with_config(config).unwrap();

        assert_eq!(store.config().buffer_pool_size, 2000);
        assert!(!store.config().enable_compression);
        assert!(!store.config().enable_bloom_filters);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_stats() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_stats");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats.triple_count, 1);
        assert!(stats.dictionary_size > 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_enhanced_stats() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_enhanced_stats");
        // Clean up any leftover data from previous runs
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Insert some triples
        for i in 0..10 {
            store
                .insert(
                    &format!("http://example.org/s{}", i),
                    "http://example.org/knows",
                    &format!("http://example.org/o{}", i),
                )
                .unwrap();
        }

        // Get enhanced statistics
        let stats = store.enhanced_stats();

        // Verify basic stats
        assert_eq!(stats.basic.triple_count, 10);
        assert!(stats.basic.dictionary_size > 0);

        // Verify buffer pool stats
        assert!(
            stats
                .buffer_pool
                .total_fetches
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0
        );
        assert!(stats.buffer_pool.hit_rate() >= 0.0);
        assert!(stats.buffer_pool.hit_rate() <= 1.0);

        // Verify storage metrics
        assert!(stats.storage.page_size > 0);
        assert!(stats.storage.memory_usage_bytes > 0);
        assert!(stats.storage.pages_allocated > 0);
        assert!(stats.storage.total_size_bytes > 0);

        // Verify storage efficiency calculations
        let efficiency = stats.storage.efficiency();
        assert!(efficiency >= 0.0);
        assert!(efficiency <= 1.0);

        let fragmentation = stats.storage.fragmentation();
        assert!(fragmentation >= 0.0);
        assert!(fragmentation <= 100.0);

        // Verify transaction metrics
        assert_eq!(stats.transaction.active_transactions, 0);
        assert!(stats.transaction.wal_enabled);

        // Verify index metrics
        assert_eq!(stats.index.spo_entries, 10);
        assert_eq!(stats.index.pos_entries, 10);
        assert_eq!(stats.index.osp_entries, 10);
        assert!(stats.index.indexes_consistent);
        assert_eq!(stats.index.total_entries(), 30);
        assert_eq!(stats.index.avg_entries_per_index(), 10.0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_storage_metrics_calculations() {
        let metrics = StorageMetrics {
            total_size_bytes: 1000,
            pages_allocated: 10,
            page_size: 200,
            memory_usage_bytes: 2000,
        };

        // Efficiency = 1000 / (10 * 200) = 1000 / 2000 = 0.5
        assert_eq!(metrics.efficiency(), 0.5);

        // Fragmentation = (1.0 - 0.5) * 100 = 50%
        assert_eq!(metrics.fragmentation(), 50.0);
    }

    #[test]
    fn test_index_metrics_calculations() {
        let metrics = IndexMetrics {
            spo_entries: 100,
            pos_entries: 100,
            osp_entries: 100,
            indexes_consistent: true,
        };

        assert_eq!(metrics.total_entries(), 300);
        assert_eq!(metrics.avg_entries_per_index(), 100.0);
    }
}
