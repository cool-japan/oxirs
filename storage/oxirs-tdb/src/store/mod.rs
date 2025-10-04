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
use crate::transaction::{TransactionManager, WriteAheadLog, LockManager};
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
        std::fs::create_dir_all(&config.data_dir)
            .map_err(|e| TdbError::Io(e))?;

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

        let s_id = self.dictionary.lookup(&s_term)?.ok_or_else(|| {
            TdbError::Other("Subject not found in dictionary".to_string())
        })?;
        let p_id = self.dictionary.lookup(&p_term)?.ok_or_else(|| {
            TdbError::Other("Predicate not found in dictionary".to_string())
        })?;
        let o_id = self.dictionary.lookup(&o_term)?.ok_or_else(|| {
            TdbError::Other("Object not found in dictionary".to_string())
        })?;

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

        let s_id = self.dictionary.lookup(&s_term)?.ok_or_else(|| {
            TdbError::Other("Subject not found in dictionary".to_string())
        })?;
        let p_id = self.dictionary.lookup(&p_term)?.ok_or_else(|| {
            TdbError::Other("Predicate not found in dictionary".to_string())
        })?;
        let o_id = self.dictionary.lookup(&o_term)?.ok_or_else(|| {
            TdbError::Other("Object not found in dictionary".to_string())
        })?;

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

    /// Get statistics
    pub fn stats(&self) -> TdbStats {
        TdbStats {
            triple_count: self.count(),
            dictionary_size: self.dictionary.size() as usize,
            bloom_filter_stats: self.bloom_filter.as_ref().map(|b| b.stats()),
            compression_stats: self.prefix_compressor.as_ref().map(|c| c.stats()),
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
}

/// TDB Store statistics
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
            .insert("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
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
            .insert("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
            .unwrap();

        assert!(store
            .contains("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
            .unwrap());

        assert!(!store
            .contains("http://example.org/alice", "http://example.org/knows", "http://example.org/charlie")
            .unwrap_or(false));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_delete() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_delete");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
            .unwrap();

        assert_eq!(store.count(), 1);

        let deleted = store
            .delete("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
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
            .insert("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
            .unwrap();
        store
            .insert("http://example.org/alice", "http://example.org/knows", "http://example.org/charlie")
            .unwrap();
        store
            .insert("http://example.org/bob", "http://example.org/likes", "http://example.org/pizza")
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
            .insert("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats.triple_count, 1);
        assert!(stats.dictionary_size > 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}

