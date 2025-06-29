//! Tiered storage engine with intelligent data placement
//!
//! This module implements a multi-tier storage system that automatically
//! moves data between tiers based on access patterns and age.

use super::*;
use crate::model::{Triple, TriplePattern};
use crate::OxirsError;
use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Access tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AccessInfo {
    /// Last access time
    last_access: SystemTime,
    /// Total access count
    access_count: u64,
    /// Current storage tier
    tier: StorageTier,
    /// Size in bytes
    size_bytes: usize,
}

/// Storage tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum StorageTier {
    Hot,
    Warm,
    Cold,
    Archive,
}

/// Triple with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredTriple {
    triple: Triple,
    metadata: TripleMetadata,
}

/// Triple metadata for tiering decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TripleMetadata {
    /// Creation timestamp
    created_at: SystemTime,
    /// Last modified timestamp
    modified_at: SystemTime,
    /// Access information
    access_info: AccessInfo,
    /// Content hash for deduplication
    content_hash: u64,
    /// Compression info
    compression: Option<CompressionInfo>,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressionInfo {
    /// Original size
    original_size: usize,
    /// Compressed size
    compressed_size: usize,
    /// Compression algorithm
    algorithm: String,
}

/// Tiered storage engine
pub struct TieredStorageEngine {
    /// Configuration
    config: StorageConfig,
    /// Hot tier - in-memory cache
    hot_tier: Arc<Mutex<LruCache<u64, StoredTriple>>>,
    /// Warm tier - SSD storage
    warm_tier: Arc<RwLock<WarmTier>>,
    /// Cold tier - HDD storage
    cold_tier: Arc<RwLock<ColdTier>>,
    /// Archive tier - long-term storage
    archive_tier: Arc<RwLock<ArchiveTier>>,
    /// Triple index for fast lookups
    index: Arc<DashMap<u64, AccessInfo>>,
    /// Statistics
    stats: Arc<Statistics>,
    /// Background task handle
    background_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Warm tier implementation
struct WarmTier {
    path: PathBuf,
    storage: rocksdb::DB,
    access_tracker: HashMap<u64, u64>,
}

/// Cold tier implementation
struct ColdTier {
    path: PathBuf,
    storage: rocksdb::DB,
    compression: compression::Compressor,
}

/// Archive tier implementation
struct ArchiveTier {
    backend: ArchiveBackend,
    index: HashMap<u64, ArchiveLocation>,
}

/// Archive location information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArchiveLocation {
    /// Archive file path
    file_path: String,
    /// Offset in file
    offset: u64,
    /// Size in bytes
    size: usize,
    /// Checksum
    checksum: u64,
}

/// Statistics tracker
struct Statistics {
    total_triples: AtomicU64,
    hot_count: AtomicU64,
    warm_count: AtomicU64,
    cold_count: AtomicU64,
    archive_count: AtomicU64,
    total_size: AtomicU64,
    hot_hits: AtomicU64,
    warm_hits: AtomicU64,
    cold_hits: AtomicU64,
    total_queries: AtomicU64,
}

impl TieredStorageEngine {
    /// Create a new tiered storage engine
    pub async fn new(config: StorageConfig) -> Result<Arc<dyn StorageEngine>, OxirsError> {
        // Initialize hot tier
        let hot_capacity = config.tiers.hot_tier.max_size_mb * 1024 * 1024 / 1000; // Approximate
        let hot_tier = Arc::new(Mutex::new(LruCache::new(
            std::num::NonZeroUsize::new(hot_capacity)
                .unwrap_or(std::num::NonZeroUsize::new(10000).unwrap()),
        )));

        // Initialize warm tier
        let warm_path = PathBuf::from(&config.tiers.warm_tier.path);
        std::fs::create_dir_all(&warm_path)?;
        let mut warm_opts = rocksdb::Options::default();
        warm_opts.create_if_missing(true);
        let warm_storage = rocksdb::DB::open(&warm_opts, warm_path.join("data"))?;
        let warm_tier = Arc::new(RwLock::new(WarmTier {
            path: warm_path,
            storage: warm_storage,
            access_tracker: HashMap::new(),
        }));

        // Initialize cold tier
        let cold_path = PathBuf::from(&config.tiers.cold_tier.path);
        std::fs::create_dir_all(&cold_path)?;
        let mut cold_opts = rocksdb::Options::default();
        cold_opts.create_if_missing(true);
        let cold_storage = rocksdb::DB::open(&cold_opts, cold_path.join("data"))?;
        let cold_tier = Arc::new(RwLock::new(ColdTier {
            path: cold_path,
            storage: cold_storage,
            compression: compression::Compressor::new(config.compression.clone()),
        }));

        // Initialize archive tier
        let archive_tier = Arc::new(RwLock::new(ArchiveTier {
            backend: config.tiers.archive_tier.backend.clone(),
            index: HashMap::new(),
        }));

        // Initialize index and statistics
        let index = Arc::new(DashMap::new());
        let stats = Arc::new(Statistics {
            total_triples: AtomicU64::new(0),
            hot_count: AtomicU64::new(0),
            warm_count: AtomicU64::new(0),
            cold_count: AtomicU64::new(0),
            archive_count: AtomicU64::new(0),
            total_size: AtomicU64::new(0),
            hot_hits: AtomicU64::new(0),
            warm_hits: AtomicU64::new(0),
            cold_hits: AtomicU64::new(0),
            total_queries: AtomicU64::new(0),
        });

        let mut engine = TieredStorageEngine {
            config,
            hot_tier,
            warm_tier,
            cold_tier,
            archive_tier,
            index,
            stats,
            background_handle: None,
        };

        // Start background tier management
        engine.start_background_tasks();

        Ok(Arc::new(engine))
    }

    /// Start background tasks for tier management
    fn start_background_tasks(&mut self) {
        let hot_tier = self.hot_tier.clone();
        let warm_tier = self.warm_tier.clone();
        let cold_tier = self.cold_tier.clone();
        let archive_tier = self.archive_tier.clone();
        let index = self.index.clone();
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                // Perform tier management
                if let Err(e) = Self::manage_tiers(
                    &hot_tier,
                    &warm_tier,
                    &cold_tier,
                    &archive_tier,
                    &index,
                    &config,
                )
                .await
                {
                    tracing::error!("Tier management error: {}", e);
                }
            }
        });

        self.background_handle = Some(handle);
    }

    /// Manage data movement between tiers
    async fn manage_tiers(
        hot_tier: &Arc<Mutex<LruCache<u64, StoredTriple>>>,
        warm_tier: &Arc<RwLock<WarmTier>>,
        cold_tier: &Arc<RwLock<ColdTier>>,
        archive_tier: &Arc<RwLock<ArchiveTier>>,
        index: &Arc<DashMap<u64, AccessInfo>>,
        config: &StorageConfig,
    ) -> Result<(), OxirsError> {
        let now = SystemTime::now();

        // Check warm tier for promotion/demotion
        {
            let mut warm = warm_tier.write().await;
            let mut to_promote = Vec::new();
            let mut to_demote = Vec::new();

            for (hash, access_count) in &warm.access_tracker {
                if *access_count >= config.tiers.warm_tier.promotion_threshold as u64 {
                    to_promote.push(*hash);
                } else if let Some(info) = index.get(hash) {
                    let days_since_access = now
                        .duration_since(info.last_access)
                        .unwrap_or(Duration::ZERO)
                        .as_secs()
                        / 86400;

                    if days_since_access >= config.tiers.warm_tier.demotion_threshold_days as u64 {
                        to_demote.push(*hash);
                    }
                }
            }

            // Promote to hot tier
            for hash in to_promote {
                if let Ok(Some(data)) = warm.storage.get(hash.to_be_bytes()) {
                    if let Ok(triple) = bincode::deserialize::<StoredTriple>(&data) {
                        hot_tier.lock().put(hash, triple);
                        warm.storage.delete(hash.to_be_bytes())?;
                        warm.access_tracker.remove(&hash);

                        if let Some(mut info) = index.get_mut(&hash) {
                            info.tier = StorageTier::Hot;
                        }
                    }
                }
            }

            // Demote to cold tier
            let mut cold = cold_tier.write().await;
            for hash in to_demote {
                if let Ok(Some(data)) = warm.storage.get(hash.to_be_bytes()) {
                    // Compress before storing in cold tier
                    let compressed = cold.compression.compress(&data)?;
                    cold.storage.put(hash.to_be_bytes(), compressed)?;
                    warm.storage.delete(hash.to_be_bytes())?;
                    warm.access_tracker.remove(&hash);

                    if let Some(mut info) = index.get_mut(&hash) {
                        info.tier = StorageTier::Cold;
                    }
                }
            }
        }

        // Check cold tier for archival
        {
            let cold = cold_tier.read().await;
            let mut to_archive = Vec::new();

            for entry in index.iter() {
                let (hash, info) = entry.pair();
                if info.tier == StorageTier::Cold {
                    let days_since_access = now
                        .duration_since(info.last_access)
                        .unwrap_or(Duration::ZERO)
                        .as_secs()
                        / 86400;

                    if days_since_access >= config.tiers.cold_tier.archive_threshold_days as u64 {
                        to_archive.push(*hash);
                    }
                }
            }

            // Move to archive
            if !to_archive.is_empty() {
                let mut archive = archive_tier.write().await;
                // Archive implementation would batch multiple triples into archive files
                // For now, we'll skip the actual archival process
            }
        }

        Ok(())
    }

    /// Calculate hash for a triple
    fn hash_triple(triple: &Triple) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        triple.hash(&mut hasher);
        hasher.finish()
    }

    /// Get the appropriate tier for a new triple based on hints
    fn determine_initial_tier(&self, triple: &Triple) -> StorageTier {
        // For now, all new data goes to warm tier
        // In a real implementation, we might analyze the triple's predicate
        // or other characteristics to make smarter placement decisions
        StorageTier::Warm
    }
}

#[async_trait::async_trait]
impl StorageEngine for TieredStorageEngine {
    async fn init(&mut self, config: StorageConfig) -> Result<(), OxirsError> {
        self.config = config;
        Ok(())
    }

    async fn store_triple(&self, triple: &Triple) -> Result<(), OxirsError> {
        let hash = Self::hash_triple(triple);
        let now = SystemTime::now();

        // Check if triple already exists
        if self.index.contains_key(&hash) {
            return Ok(());
        }

        // Create stored triple with metadata
        let stored = StoredTriple {
            triple: triple.clone(),
            metadata: TripleMetadata {
                created_at: now,
                modified_at: now,
                access_info: AccessInfo {
                    last_access: now,
                    access_count: 0,
                    tier: self.determine_initial_tier(triple),
                    size_bytes: bincode::serialized_size(triple)? as usize,
                },
                content_hash: hash,
                compression: None,
            },
        };

        // Store in appropriate tier
        match stored.metadata.access_info.tier {
            StorageTier::Hot => {
                self.hot_tier.lock().put(hash, stored.clone());
                self.stats.hot_count.fetch_add(1, Ordering::Relaxed);
            }
            StorageTier::Warm => {
                let data = bincode::serialize(&stored)?;
                self.warm_tier
                    .write()
                    .await
                    .storage
                    .put(hash.to_be_bytes(), data)?;
                self.stats.warm_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => unreachable!("New triples should not go directly to cold/archive"),
        }

        // Update index
        self.index.insert(hash, stored.metadata.access_info.clone());
        self.stats.total_triples.fetch_add(1, Ordering::Relaxed);
        self.stats.total_size.fetch_add(
            stored.metadata.access_info.size_bytes as u64,
            Ordering::Relaxed,
        );

        Ok(())
    }

    async fn store_triples(&self, triples: &[Triple]) -> Result<(), OxirsError> {
        // Use parallel processing for batch inserts
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            triples
                .par_iter()
                .try_for_each(|triple| futures::executor::block_on(self.store_triple(triple)))
        }
        #[cfg(not(feature = "parallel"))]
        {
            for triple in triples {
                self.store_triple(triple).await?;
            }
            Ok(())
        }
    }

    async fn query_triples(&self, pattern: &TriplePattern) -> Result<Vec<Triple>, OxirsError> {
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);
        let mut results = Vec::new();

        // Search hot tier first
        {
            let hot = self.hot_tier.lock();
            for (_, stored) in hot.iter() {
                if pattern.matches(&stored.triple) {
                    results.push(stored.triple.clone());
                    self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // If not enough results, search warm tier
        if results.is_empty() {
            let warm = self.warm_tier.read().await;
            // Iterate through warm tier storage
            let iter = warm.storage.iterator(rocksdb::IteratorMode::Start);
            for item in iter {
                if let Ok((key, value)) = item {
                    if let Ok(stored) = bincode::deserialize::<StoredTriple>(&value) {
                        if pattern.matches(&stored.triple) {
                            results.push(stored.triple.clone());
                            self.stats.warm_hits.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        }

        // Update access info for queried triples
        let now = SystemTime::now();
        for triple in &results {
            let hash = Self::hash_triple(triple);
            if let Some(mut info) = self.index.get_mut(&hash) {
                info.last_access = now;
                info.access_count += 1;

                // Track access in warm tier
                if info.tier == StorageTier::Warm {
                    if let Ok(mut warm) = self.warm_tier.try_write() {
                        *warm.access_tracker.entry(hash).or_insert(0) += 1;
                    }
                }
            }
        }

        Ok(results)
    }

    async fn delete_triples(&self, pattern: &TriplePattern) -> Result<usize, OxirsError> {
        let triples = self.query_triples(pattern).await?;
        let count = triples.len();

        for triple in triples {
            let hash = Self::hash_triple(&triple);

            // Remove from index
            if let Some((_, info)) = self.index.remove(&hash) {
                // Remove from appropriate tier
                match info.tier {
                    StorageTier::Hot => {
                        self.hot_tier.lock().pop(&hash);
                        self.stats.hot_count.fetch_sub(1, Ordering::Relaxed);
                    }
                    StorageTier::Warm => {
                        self.warm_tier
                            .write()
                            .await
                            .storage
                            .delete(hash.to_be_bytes())?;
                        self.stats.warm_count.fetch_sub(1, Ordering::Relaxed);
                    }
                    StorageTier::Cold => {
                        self.cold_tier
                            .write()
                            .await
                            .storage
                            .delete(hash.to_be_bytes())?;
                        self.stats.cold_count.fetch_sub(1, Ordering::Relaxed);
                    }
                    StorageTier::Archive => {
                        // Archive deletion is more complex and might not be allowed
                        if !self.config.tiers.archive_tier.immutable {
                            // Implement archive deletion
                        }
                    }
                }

                self.stats.total_triples.fetch_sub(1, Ordering::Relaxed);
                self.stats
                    .total_size
                    .fetch_sub(info.size_bytes as u64, Ordering::Relaxed);
            }
        }

        Ok(count)
    }

    async fn stats(&self) -> Result<StorageStats, OxirsError> {
        let total_queries = self.stats.total_queries.load(Ordering::Relaxed);
        let hot_hits = self.stats.hot_hits.load(Ordering::Relaxed);
        let warm_hits = self.stats.warm_hits.load(Ordering::Relaxed);
        let cold_hits = self.stats.cold_hits.load(Ordering::Relaxed);
        let total_hits = hot_hits + warm_hits + cold_hits;

        Ok(StorageStats {
            total_triples: self.stats.total_triples.load(Ordering::Relaxed),
            total_size_bytes: self.stats.total_size.load(Ordering::Relaxed),
            tier_stats: TierStats {
                hot: TierStat {
                    triple_count: self.stats.hot_count.load(Ordering::Relaxed),
                    size_bytes: 0, // Calculate from hot tier
                    hit_rate: if total_queries > 0 {
                        (hot_hits as f64 / total_queries as f64) * 100.0
                    } else {
                        0.0
                    },
                    avg_access_time_us: 1, // Sub-microsecond for memory
                },
                warm: TierStat {
                    triple_count: self.stats.warm_count.load(Ordering::Relaxed),
                    size_bytes: 0, // Calculate from warm tier
                    hit_rate: if total_queries > 0 {
                        (warm_hits as f64 / total_queries as f64) * 100.0
                    } else {
                        0.0
                    },
                    avg_access_time_us: 100, // ~100μs for SSD
                },
                cold: TierStat {
                    triple_count: self.stats.cold_count.load(Ordering::Relaxed),
                    size_bytes: 0, // Calculate from cold tier
                    hit_rate: if total_queries > 0 {
                        (cold_hits as f64 / total_queries as f64) * 100.0
                    } else {
                        0.0
                    },
                    avg_access_time_us: 10000, // ~10ms for HDD
                },
                archive: TierStat {
                    triple_count: self.stats.archive_count.load(Ordering::Relaxed),
                    size_bytes: 0,               // Calculate from archive
                    hit_rate: 0.0,               // Archive is rarely accessed
                    avg_access_time_us: 1000000, // ~1s for archive retrieval
                },
            },
            compression_ratio: 1.5, // Placeholder
            query_metrics: QueryMetrics {
                avg_query_time_ms: 0.1,                            // Placeholder
                p99_query_time_ms: 1.0,                            // Placeholder
                qps: if total_queries > 0 { 1000.0 } else { 0.0 }, // Placeholder
                cache_hit_rate: if total_queries > 0 {
                    (total_hits as f64 / total_queries as f64) * 100.0
                } else {
                    0.0
                },
            },
        })
    }

    async fn optimize(&self) -> Result<(), OxirsError> {
        // Trigger compaction on RocksDB instances
        self.warm_tier
            .read()
            .await
            .storage
            .compact_range(None::<&[u8]>, None::<&[u8]>);
        self.cold_tier
            .read()
            .await
            .storage
            .compact_range(None::<&[u8]>, None::<&[u8]>);

        // Force tier rebalancing
        Self::manage_tiers(
            &self.hot_tier,
            &self.warm_tier,
            &self.cold_tier,
            &self.archive_tier,
            &self.index,
            &self.config,
        )
        .await?;

        Ok(())
    }

    async fn backup(&self, path: &Path) -> Result<(), OxirsError> {
        // Create backup directory
        std::fs::create_dir_all(path)?;

        // Backup metadata
        let metadata = BackupMetadata {
            version: 1,
            created_at: SystemTime::now(),
            total_triples: self.stats.total_triples.load(Ordering::Relaxed),
            config: self.config.clone(),
        };

        let metadata_path = path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(metadata_path, metadata_json)?;

        // Backup each tier
        // Hot tier
        let hot_backup = path.join("hot.bin");
        let hot_data: Vec<_> = self
            .hot_tier
            .lock()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        let hot_bytes = bincode::serialize(&hot_data)?;
        std::fs::write(hot_backup, hot_bytes)?;

        // Warm and cold tiers - backup by iterating and saving
        let warm_backup = path.join("warm.bin");
        let warm_data: Vec<(Vec<u8>, Vec<u8>)> = {
            let warm = self.warm_tier.read().await;
            let mut data = Vec::new();
            let iter = warm.storage.iterator(rocksdb::IteratorMode::Start);
            for item in iter {
                if let Ok((key, value)) = item {
                    data.push((key.to_vec(), value.to_vec()));
                }
            }
            data
        };
        let warm_bytes = bincode::serialize(&warm_data)?;
        std::fs::write(warm_backup, warm_bytes)?;

        let cold_backup = path.join("cold.bin");
        let cold_data: Vec<(Vec<u8>, Vec<u8>)> = {
            let cold = self.cold_tier.read().await;
            let mut data = Vec::new();
            let iter = cold.storage.iterator(rocksdb::IteratorMode::Start);
            for item in iter {
                if let Ok((key, value)) = item {
                    data.push((key.to_vec(), value.to_vec()));
                }
            }
            data
        };
        let cold_bytes = bincode::serialize(&cold_data)?;
        std::fs::write(cold_backup, cold_bytes)?;

        // Index backup
        let index_backup = path.join("index.bin");
        let index_data: HashMap<_, _> = self
            .index
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();
        let index_bytes = bincode::serialize(&index_data)?;
        std::fs::write(index_backup, index_bytes)?;

        Ok(())
    }

    async fn restore(&self, path: &Path) -> Result<(), OxirsError> {
        // Read metadata
        let metadata_path = path.join("metadata.json");
        let metadata_json = std::fs::read_to_string(metadata_path)?;
        let metadata: BackupMetadata = serde_json::from_str(&metadata_json)?;

        // Restore hot tier
        let hot_backup = path.join("hot.bin");
        if hot_backup.exists() {
            let hot_bytes = std::fs::read(hot_backup)?;
            let hot_data: Vec<(u64, StoredTriple)> = bincode::deserialize(&hot_bytes)?;

            let mut hot = self.hot_tier.lock();
            hot.clear();
            for (k, v) in hot_data {
                hot.put(k, v);
            }
        }

        // Restore warm tier
        let warm_backup = path.join("warm.bin");
        if warm_backup.exists() {
            let warm_bytes = std::fs::read(warm_backup)?;
            let warm_data: Vec<(Vec<u8>, Vec<u8>)> = bincode::deserialize(&warm_bytes)?;

            let warm = self.warm_tier.write().await;
            for (key, value) in warm_data {
                warm.storage.put(&key, &value)?;
            }
        }

        // Restore cold tier
        let cold_backup = path.join("cold.bin");
        if cold_backup.exists() {
            let cold_bytes = std::fs::read(cold_backup)?;
            let cold_data: Vec<(Vec<u8>, Vec<u8>)> = bincode::deserialize(&cold_bytes)?;

            let cold = self.cold_tier.write().await;
            for (key, value) in cold_data {
                cold.storage.put(&key, &value)?;
            }
        }

        // Restore index
        let index_backup = path.join("index.bin");
        if index_backup.exists() {
            let index_bytes = std::fs::read(index_backup)?;
            let index_data: HashMap<u64, AccessInfo> = bincode::deserialize(&index_bytes)?;

            self.index.clear();
            for (k, v) in index_data {
                self.index.insert(k, v);
            }
        }

        // Update statistics
        self.stats
            .total_triples
            .store(metadata.total_triples, Ordering::Relaxed);

        Ok(())
    }
}

/// Backup metadata
#[derive(Debug, Serialize, Deserialize)]
struct BackupMetadata {
    version: u32,
    created_at: SystemTime,
    total_triples: u64,
    config: StorageConfig,
}

// Placeholder compression module
mod compression {
    use super::*;

    pub struct Compressor {
        compression_type: CompressionType,
    }

    impl Compressor {
        pub fn new(compression_type: CompressionType) -> Self {
            Compressor { compression_type }
        }

        pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, OxirsError> {
            // Placeholder - would use actual compression libraries
            Ok(data.to_vec())
        }

        pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, OxirsError> {
            // Placeholder - would use actual compression libraries
            Ok(data.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};
    use crate::storage::ArchiveBackend;

    #[tokio::test]
    async fn test_tiered_storage() {
        let test_dir = format!(
            "/tmp/oxirs_tiered_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        let mut config = StorageConfig::default();
        config.tiers.warm_tier.path = format!("{}/warm", test_dir);
        config.tiers.cold_tier.path = format!("{}/cold", test_dir);
        config.tiers.archive_tier.backend = ArchiveBackend::Local(format!("{}/archive", test_dir));

        let engine = TieredStorageEngine::new(config).await.unwrap();

        // Create test triple
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = crate::model::Object::Literal(Literal::new("test"));
        let triple = Triple::new(subject, predicate, object);

        // Store triple
        engine.store_triple(&triple).await.unwrap();

        // Query triple
        let pattern = TriplePattern::new(None, None, None);
        let results = engine.query_triples(&pattern).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], triple);

        // Check stats
        let stats = engine.stats().await.unwrap();
        assert_eq!(stats.total_triples, 1);
    }
}
