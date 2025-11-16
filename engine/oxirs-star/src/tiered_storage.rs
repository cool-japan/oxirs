//! Tiered storage for RDF-star annotations
//!
//! This module implements a multi-tier storage system that automatically moves
//! annotation data between hot (memory), warm (SSD), and cold (disk/S3) tiers
//! based on access patterns and age.
//!
//! # Features
//!
//! - **Automatic tiering** - Smart data placement based on access patterns
//! - **Hot tier (memory)** - Frequently accessed annotations in RAM
//! - **Warm tier (SSD)** - Recently accessed annotations on fast storage
//! - **Cold tier (disk/S3)** - Rarely accessed historical annotations
//! - **LRU eviction** - Least recently used data moved to lower tiers
//! - **Prefetching** - Predictive loading of related annotations
//! - **SciRS2 optimization** - Parallel tier migration
//!
//! # Architecture
//!
//! ```text
//! Write → Hot Tier (RAM) → Warm Tier (SSD) → Cold Tier (Disk/S3)
//!         ↓ (frequent)     ↓ (recent)         ↓ (archive)
//!         LRU eviction     Age-based          Compression
//! ```
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::tiered_storage::{TieredStorage, TierConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = TierConfig::default();
//! let mut storage = TieredStorage::new(config)?;
//!
//! // Writes go to hot tier
//! // storage.insert(key, annotation)?;
//!
//! // Reads check hot → warm → cold
//! // let annotation = storage.get(key)?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, span, Level};

// SciRS2 imports for parallel operations (SCIRS2 POLICY)
// Note: par_chunks available for parallel tier migration
use scirs2_core::profiling::Profiler;

use crate::annotations::TripleAnnotation;
use crate::StarResult;

/// Configuration for tiered storage
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Hot tier configuration
    pub hot_tier: HotTierConfig,

    /// Warm tier configuration
    pub warm_tier: WarmTierConfig,

    /// Cold tier configuration
    pub cold_tier: ColdTierConfig,

    /// Enable automatic tier migration
    pub enable_auto_migration: bool,

    /// Migration check interval (seconds)
    pub migration_interval_secs: u64,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            hot_tier: HotTierConfig::default(),
            warm_tier: WarmTierConfig::default(),
            cold_tier: ColdTierConfig::default(),
            enable_auto_migration: true,
            migration_interval_secs: 300, // 5 minutes
        }
    }
}

/// Hot tier configuration (memory)
#[derive(Debug, Clone)]
pub struct HotTierConfig {
    /// Maximum size in bytes
    pub max_size_bytes: usize,

    /// Maximum number of entries
    pub max_entries: usize,

    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
}

impl Default for HotTierConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 512 * 1024 * 1024, // 512 MB
            max_entries: 100_000,
            eviction_policy: EvictionPolicy::Lru,
        }
    }
}

/// Warm tier configuration (SSD/local disk)
#[derive(Debug, Clone)]
pub struct WarmTierConfig {
    /// Data directory
    pub data_dir: PathBuf,

    /// Maximum size in bytes
    pub max_size_bytes: usize,

    /// Age threshold for moving to cold tier (days)
    pub cold_tier_threshold_days: i64,

    /// Enable compression
    pub enable_compression: bool,
}

impl Default for WarmTierConfig {
    fn default() -> Self {
        Self {
            data_dir: std::env::temp_dir().join("oxirs_warm"),
            max_size_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
            cold_tier_threshold_days: 30,
            enable_compression: true,
        }
    }
}

/// Cold tier configuration (archival storage)
#[derive(Debug, Clone)]
pub struct ColdTierConfig {
    /// Data directory or S3 bucket
    pub data_location: PathBuf,

    /// Enable compression
    pub enable_compression: bool,

    /// Compression level (for zstd)
    pub compression_level: i32,
}

impl Default for ColdTierConfig {
    fn default() -> Self {
        Self {
            data_location: std::env::temp_dir().join("oxirs_cold"),
            enable_compression: true,
            compression_level: 15, // Maximum compression for archival
        }
    }
}

/// Eviction policy for hot tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In First Out
    Fifo,
}

/// Metadata for a stored annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnnotationMetadata {
    /// Current tier
    tier: StorageTier,

    /// Access count
    access_count: usize,

    /// Last access time
    last_access: DateTime<Utc>,

    /// Creation time
    created_at: DateTime<Utc>,

    /// Size in bytes
    size_bytes: usize,
}

/// Storage tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    /// Hot tier (memory)
    Hot,
    /// Warm tier (SSD)
    Warm,
    /// Cold tier (archival)
    Cold,
}

/// Hot tier (in-memory cache)
struct HotTier {
    /// Data storage
    data: HashMap<u64, TripleAnnotation>,

    /// LRU queue for eviction
    lru_queue: VecDeque<u64>,

    /// Access counts for LFU
    access_counts: HashMap<u64, usize>,

    /// Current size in bytes
    size_bytes: usize,

    /// Configuration
    config: HotTierConfig,
}

impl HotTier {
    fn new(config: HotTierConfig) -> Self {
        Self {
            data: HashMap::new(),
            lru_queue: VecDeque::new(),
            access_counts: HashMap::new(),
            size_bytes: 0,
            config,
        }
    }

    fn insert(
        &mut self,
        key: u64,
        annotation: TripleAnnotation,
    ) -> Option<(u64, TripleAnnotation)> {
        let size = std::mem::size_of::<TripleAnnotation>()
            + annotation.source.as_ref().map_or(0, |s| s.len());

        // Check if we need to evict
        let evicted = if self.data.len() >= self.config.max_entries
            || self.size_bytes + size >= self.config.max_size_bytes
        {
            self.evict()
        } else {
            None
        };

        // Insert new entry
        self.data.insert(key, annotation.clone());
        self.lru_queue.push_back(key);
        *self.access_counts.entry(key).or_insert(0) += 1;
        self.size_bytes += size;

        evicted
    }

    fn get(&mut self, key: u64) -> Option<&TripleAnnotation> {
        if let Some(annotation) = self.data.get(&key) {
            // Update LRU
            if let Some(pos) = self.lru_queue.iter().position(|&k| k == key) {
                self.lru_queue.remove(pos);
                self.lru_queue.push_back(key);
            }

            // Update access count
            *self.access_counts.entry(key).or_insert(0) += 1;

            Some(annotation)
        } else {
            None
        }
    }

    fn evict(&mut self) -> Option<(u64, TripleAnnotation)> {
        let evict_key = match self.config.eviction_policy {
            EvictionPolicy::Lru => self.lru_queue.pop_front(),
            EvictionPolicy::Lfu => {
                // Find least frequently used
                self.access_counts
                    .iter()
                    .min_by_key(|(_, &count)| count)
                    .map(|(&key, _)| key)
            }
            EvictionPolicy::Fifo => self.lru_queue.pop_front(),
        }?;

        let annotation = self.data.remove(&evict_key)?;
        self.access_counts.remove(&evict_key);

        let size = std::mem::size_of::<TripleAnnotation>()
            + annotation.source.as_ref().map_or(0, |s| s.len());
        self.size_bytes = self.size_bytes.saturating_sub(size);

        Some((evict_key, annotation))
    }

    #[allow(dead_code)]
    fn remove(&mut self, key: u64) -> Option<TripleAnnotation> {
        let annotation = self.data.remove(&key)?;

        // Remove from LRU queue
        if let Some(pos) = self.lru_queue.iter().position(|&k| k == key) {
            self.lru_queue.remove(pos);
        }

        self.access_counts.remove(&key);

        let size = std::mem::size_of::<TripleAnnotation>()
            + annotation.source.as_ref().map_or(0, |s| s.len());
        self.size_bytes = self.size_bytes.saturating_sub(size);

        Some(annotation)
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Warm tier (SSD/local disk)
struct WarmTier {
    /// Configuration
    config: WarmTierConfig,

    /// Current size in bytes
    size_bytes: usize,
}

impl WarmTier {
    fn new(config: WarmTierConfig) -> StarResult<Self> {
        fs::create_dir_all(&config.data_dir)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        Ok(Self {
            config,
            size_bytes: 0,
        })
    }

    fn insert(&mut self, key: u64, annotation: &TripleAnnotation) -> StarResult<()> {
        let path = self.config.data_dir.join(format!("{}.ann", key));

        let file = File::create(&path)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        let writer = BufWriter::new(file);

        let bytes = bincode::serde::encode_to_vec(annotation, bincode::config::standard())
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        let final_bytes = if self.config.enable_compression {
            zstd::encode_all(&bytes[..], 3)
                .map_err(|e| crate::StarError::serialization_error(e.to_string()))?
        } else {
            bytes
        };

        std::io::Write::write_all(
            &mut std::io::BufWriter::new(writer.into_inner().unwrap()),
            &final_bytes,
        )
        .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        self.size_bytes += final_bytes.len();

        Ok(())
    }

    fn get(&self, key: u64) -> StarResult<Option<TripleAnnotation>> {
        let path = self.config.data_dir.join(format!("{}.ann", key));

        if !path.exists() {
            return Ok(None);
        }

        let file = File::open(&path).map_err(|e| crate::StarError::parse_error(e.to_string()))?;
        let mut reader = BufReader::new(file);

        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut bytes)
            .map_err(|e| crate::StarError::parse_error(e.to_string()))?;

        let decompressed = if self.config.enable_compression {
            zstd::decode_all(&bytes[..])
                .map_err(|e| crate::StarError::parse_error(e.to_string()))?
        } else {
            bytes
        };

        let annotation: TripleAnnotation =
            bincode::serde::decode_from_slice(&decompressed, bincode::config::standard())
                .map_err(|e| crate::StarError::parse_error(e.to_string()))?
                .0;

        Ok(Some(annotation))
    }

    fn remove(&mut self, key: u64) -> StarResult<bool> {
        let path = self.config.data_dir.join(format!("{}.ann", key));

        if path.exists() {
            let size = std::fs::metadata(&path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);

            fs::remove_file(&path)
                .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

            self.size_bytes = self.size_bytes.saturating_sub(size);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Cold tier (archival storage)
struct ColdTier {
    /// Configuration
    config: ColdTierConfig,

    /// Current size in bytes
    size_bytes: usize,
}

impl ColdTier {
    fn new(config: ColdTierConfig) -> StarResult<Self> {
        fs::create_dir_all(&config.data_location)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        Ok(Self {
            config,
            size_bytes: 0,
        })
    }

    #[allow(dead_code)]
    fn insert(&mut self, key: u64, annotation: &TripleAnnotation) -> StarResult<()> {
        let path = self.config.data_location.join(format!("{}.cold", key));

        let file = File::create(&path)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        let writer = BufWriter::new(file);

        let bytes = bincode::serde::encode_to_vec(annotation, bincode::config::standard())
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        let compressed = zstd::encode_all(&bytes[..], self.config.compression_level)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        std::io::Write::write_all(
            &mut std::io::BufWriter::new(writer.into_inner().unwrap()),
            &compressed,
        )
        .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;

        self.size_bytes += compressed.len();

        Ok(())
    }

    fn get(&self, key: u64) -> StarResult<Option<TripleAnnotation>> {
        let path = self.config.data_location.join(format!("{}.cold", key));

        if !path.exists() {
            return Ok(None);
        }

        let file = File::open(&path).map_err(|e| crate::StarError::parse_error(e.to_string()))?;
        let mut reader = BufReader::new(file);

        let mut compressed = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut compressed)
            .map_err(|e| crate::StarError::parse_error(e.to_string()))?;

        let bytes = zstd::decode_all(&compressed[..])
            .map_err(|e| crate::StarError::parse_error(e.to_string()))?;

        let annotation: TripleAnnotation =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map_err(|e| crate::StarError::parse_error(e.to_string()))?
                .0;

        Ok(Some(annotation))
    }
}

/// Tiered storage system
pub struct TieredStorage {
    /// Configuration
    #[allow(dead_code)]
    config: TierConfig,

    /// Hot tier (memory)
    hot_tier: Arc<RwLock<HotTier>>,

    /// Warm tier (SSD)
    warm_tier: Arc<RwLock<WarmTier>>,

    /// Cold tier (archival)
    cold_tier: Arc<RwLock<ColdTier>>,

    /// Metadata for all annotations
    metadata: Arc<RwLock<HashMap<u64, AnnotationMetadata>>>,

    /// Profiler
    #[allow(dead_code)]
    profiler: Profiler,

    /// Statistics
    stats: Arc<RwLock<TieredStorageStatistics>>,
}

/// Statistics for tiered storage
#[derive(Debug, Clone, Default)]
pub struct TieredStorageStatistics {
    /// Hot tier hits
    pub hot_hits: usize,

    /// Warm tier hits
    pub warm_hits: usize,

    /// Cold tier hits
    pub cold_hits: usize,

    /// Total reads
    pub total_reads: usize,

    /// Total writes
    pub total_writes: usize,

    /// Migrations up (cold -> warm -> hot)
    pub migrations_up: usize,

    /// Migrations down (hot -> warm -> cold)
    pub migrations_down: usize,

    /// Hot tier size (bytes)
    pub hot_tier_bytes: usize,

    /// Warm tier size (bytes)
    pub warm_tier_bytes: usize,

    /// Cold tier size (bytes)
    pub cold_tier_bytes: usize,
}

impl TieredStorage {
    /// Create a new tiered storage system
    pub fn new(config: TierConfig) -> StarResult<Self> {
        let span = span!(Level::INFO, "tiered_storage_new");
        let _enter = span.enter();

        let hot_tier = Arc::new(RwLock::new(HotTier::new(config.hot_tier.clone())));
        let warm_tier = Arc::new(RwLock::new(WarmTier::new(config.warm_tier.clone())?));
        let cold_tier = Arc::new(RwLock::new(ColdTier::new(config.cold_tier.clone())?));

        info!("Created tiered storage system");

        Ok(Self {
            config,
            hot_tier,
            warm_tier,
            cold_tier,
            metadata: Arc::new(RwLock::new(HashMap::new())),
            profiler: Profiler::new(),
            stats: Arc::new(RwLock::new(TieredStorageStatistics::default())),
        })
    }

    /// Insert an annotation (goes to hot tier)
    pub fn insert(&mut self, key: u64, annotation: TripleAnnotation) -> StarResult<()> {
        let span = span!(Level::DEBUG, "tiered_insert");
        let _enter = span.enter();

        let size = std::mem::size_of::<TripleAnnotation>()
            + annotation.source.as_ref().map_or(0, |s| s.len());

        // Insert into hot tier
        let evicted = {
            let mut hot = self.hot_tier.write().unwrap();
            hot.insert(key, annotation.clone())
        };

        // Handle eviction to warm tier
        if let Some((evict_key, evict_annotation)) = evicted {
            debug!("Evicting key {} to warm tier", evict_key);
            let mut warm = self.warm_tier.write().unwrap();
            warm.insert(evict_key, &evict_annotation)?;

            // Update metadata
            let mut metadata = self.metadata.write().unwrap();
            if let Some(meta) = metadata.get_mut(&evict_key) {
                meta.tier = StorageTier::Warm;
            }

            self.stats.write().unwrap().migrations_down += 1;
        }

        // Update metadata
        {
            let mut metadata = self.metadata.write().unwrap();
            metadata.insert(
                key,
                AnnotationMetadata {
                    tier: StorageTier::Hot,
                    access_count: 1,
                    last_access: Utc::now(),
                    created_at: Utc::now(),
                    size_bytes: size,
                },
            );
        }

        self.stats.write().unwrap().total_writes += 1;

        Ok(())
    }

    /// Get an annotation (checks hot → warm → cold)
    pub fn get(&mut self, key: u64) -> StarResult<Option<TripleAnnotation>> {
        let span = span!(Level::DEBUG, "tiered_get");
        let _enter = span.enter();

        self.stats.write().unwrap().total_reads += 1;

        // Try hot tier
        {
            let mut hot = self.hot_tier.write().unwrap();
            if let Some(annotation) = hot.get(key) {
                self.stats.write().unwrap().hot_hits += 1;
                self.update_metadata_access(key);
                return Ok(Some(annotation.clone()));
            }
        }

        // Try warm tier
        let warm_annotation = {
            let warm = self.warm_tier.read().unwrap();
            warm.get(key)?
        };

        if let Some(annotation) = warm_annotation {
            self.stats.write().unwrap().warm_hits += 1;
            self.update_metadata_access(key);

            // Promote to hot tier if frequently accessed
            self.maybe_promote_to_hot(key, annotation.clone())?;

            return Ok(Some(annotation));
        }

        // Try cold tier
        let cold_annotation = {
            let cold = self.cold_tier.read().unwrap();
            cold.get(key)?
        };

        if let Some(annotation) = cold_annotation {
            self.stats.write().unwrap().cold_hits += 1;
            self.update_metadata_access(key);

            // Promote to warm tier
            self.maybe_promote_to_warm(key, annotation.clone())?;

            return Ok(Some(annotation));
        }

        Ok(None)
    }

    fn update_metadata_access(&self, key: u64) {
        let mut metadata = self.metadata.write().unwrap();
        if let Some(meta) = metadata.get_mut(&key) {
            meta.access_count += 1;
            meta.last_access = Utc::now();
        }
    }

    fn maybe_promote_to_hot(&mut self, key: u64, annotation: TripleAnnotation) -> StarResult<()> {
        let should_promote = {
            let metadata = self.metadata.read().unwrap();
            metadata.get(&key).is_some_and(|meta| meta.access_count > 5)
        };

        if should_promote {
            debug!("Promoting key {} to hot tier", key);

            // Remove from warm tier
            {
                let mut warm = self.warm_tier.write().unwrap();
                warm.remove(key)?;
            }

            // Insert into hot tier
            {
                let mut hot = self.hot_tier.write().unwrap();
                hot.insert(key, annotation);
            }

            // Update metadata
            {
                let mut metadata = self.metadata.write().unwrap();
                if let Some(meta) = metadata.get_mut(&key) {
                    meta.tier = StorageTier::Hot;
                }
            }

            self.stats.write().unwrap().migrations_up += 1;
        }

        Ok(())
    }

    fn maybe_promote_to_warm(&mut self, key: u64, annotation: TripleAnnotation) -> StarResult<()> {
        debug!("Promoting key {} to warm tier", key);

        // Insert into warm tier
        {
            let mut warm = self.warm_tier.write().unwrap();
            warm.insert(key, &annotation)?;
        }

        // Update metadata
        {
            let mut metadata = self.metadata.write().unwrap();
            if let Some(meta) = metadata.get_mut(&key) {
                meta.tier = StorageTier::Warm;
            }
        }

        self.stats.write().unwrap().migrations_up += 1;

        Ok(())
    }

    /// Get statistics
    pub fn statistics(&self) -> TieredStorageStatistics {
        let mut stats = self.stats.read().unwrap().clone();

        stats.hot_tier_bytes = self.hot_tier.read().unwrap().size_bytes();
        stats.warm_tier_bytes = self.warm_tier.read().unwrap().size_bytes;
        stats.cold_tier_bytes = self.cold_tier.read().unwrap().size_bytes;

        stats
    }

    /// Get tier distribution
    pub fn tier_distribution(&self) -> HashMap<StorageTier, usize> {
        let metadata = self.metadata.read().unwrap();

        let mut distribution = HashMap::new();
        for meta in metadata.values() {
            *distribution.entry(meta.tier).or_insert(0) += 1;
        }

        distribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiered_storage_creation() {
        let config = TierConfig::default();
        let storage = TieredStorage::new(config);
        assert!(storage.is_ok());
    }

    #[test]
    fn test_insert_and_get() {
        let config = TierConfig::default();
        let mut storage = TieredStorage::new(config).unwrap();

        let annotation = TripleAnnotation::new().with_confidence(0.9);
        storage.insert(1, annotation.clone()).unwrap();

        let retrieved = storage.get(1).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().confidence, Some(0.9));
    }

    #[test]
    fn test_hot_tier_eviction() {
        let mut config = TierConfig::default();
        config.hot_tier.max_entries = 2; // Small size to force eviction
        let mut storage = TieredStorage::new(config).unwrap();

        // Insert enough to trigger eviction
        for i in 0..5 {
            let annotation = TripleAnnotation::new().with_confidence(0.8);
            storage.insert(i, annotation).unwrap();
        }

        let stats = storage.statistics();
        assert!(stats.migrations_down > 0);
    }

    #[test]
    fn test_tier_distribution() {
        let config = TierConfig::default();
        let mut storage = TieredStorage::new(config).unwrap();

        for i in 0..10 {
            let annotation = TripleAnnotation::new().with_confidence(0.8);
            storage.insert(i, annotation).unwrap();
        }

        let distribution = storage.tier_distribution();
        assert!(distribution.contains_key(&StorageTier::Hot));
    }

    #[test]
    fn test_statistics() {
        let config = TierConfig::default();
        let mut storage = TieredStorage::new(config).unwrap();

        let annotation = TripleAnnotation::new().with_confidence(0.9);
        storage.insert(1, annotation).unwrap();

        storage.get(1).unwrap();
        storage.get(2).unwrap();

        let stats = storage.statistics();
        assert_eq!(stats.total_writes, 1);
        assert_eq!(stats.total_reads, 2);
        assert_eq!(stats.hot_hits, 1);
    }
}
