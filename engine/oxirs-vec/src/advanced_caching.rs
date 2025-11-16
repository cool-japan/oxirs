//! Advanced multi-level caching system for vector embeddings and search results
//!
//! This module provides:
//! - Multi-level caching (memory + persistent)
//! - LRU, LFU, ARC eviction policies
//! - TTL expiration
//! - Cache coherence and invalidation
//! - Background cache updates

use crate::Vector;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Type alias for complex tag index structure
type TagIndex = Arc<RwLock<HashMap<String, HashMap<String, Vec<CacheKey>>>>>;

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Adaptive Replacement Cache
    ARC,
    /// First In, First Out
    FIFO,
    /// Time-based expiration only
    TTL,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in memory cache
    pub max_memory_entries: usize,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Time-to-live for cache entries
    pub ttl: Option<Duration>,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable persistent cache
    pub enable_persistent: bool,
    /// Persistent cache directory
    pub persistent_cache_dir: Option<std::path::PathBuf>,
    /// Maximum persistent cache size in bytes
    pub max_persistent_bytes: usize,
    /// Enable cache compression
    pub enable_compression: bool,
    /// Enable background updates
    pub enable_background_updates: bool,
    /// Background update interval
    pub background_update_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 10000,
            max_memory_bytes: 1024 * 1024 * 100,  // 100MB
            ttl: Some(Duration::from_secs(3600)), // 1 hour
            eviction_policy: EvictionPolicy::LRU,
            enable_persistent: true,
            persistent_cache_dir: None,
            max_persistent_bytes: 1024 * 1024 * 1024, // 1GB
            enable_compression: true,
            enable_background_updates: false,
            background_update_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cached data
    pub data: Vector,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// Access count for LFU
    pub access_count: u64,
    /// Entry size in bytes
    pub size_bytes: usize,
    /// TTL for this specific entry
    pub ttl: Option<Duration>,
    /// Metadata tags
    pub tags: HashMap<String, String>,
}

impl CacheEntry {
    pub fn new(data: Vector) -> Self {
        let now = Instant::now();
        let size_bytes = data.dimensions * std::mem::size_of::<f32>() + 64; // Rough estimate

        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
            ttl: None,
            tags: HashMap::new(),
        }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    pub fn with_tags(mut self, tags: HashMap<String, String>) -> Self {
        self.tags = tags;
        self
    }

    /// Check if entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// Update access statistics
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Cache key that can be hashed
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    pub namespace: String,
    pub key: String,
    pub variant: Option<String>,
}

impl CacheKey {
    pub fn new(namespace: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            key: key.into(),
            variant: None,
        }
    }

    pub fn with_variant(mut self, variant: impl Into<String>) -> Self {
        self.variant = Some(variant.into());
        self
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref variant) = self.variant {
            write!(f, "{}:{}:{}", self.namespace, self.key, variant)
        } else {
            write!(f, "{}:{}", self.namespace, self.key)
        }
    }
}

/// Memory cache implementation
pub struct MemoryCache {
    config: CacheConfig,
    entries: HashMap<CacheKey, CacheEntry>,
    access_order: VecDeque<CacheKey>,      // For LRU
    frequency_map: HashMap<CacheKey, u64>, // For LFU
    current_memory_bytes: usize,
    // ARC state
    arc_t1: VecDeque<CacheKey>, // Recently accessed pages
    arc_t2: VecDeque<CacheKey>, // Frequently accessed pages
    arc_b1: VecDeque<CacheKey>, // Ghost list for T1
    arc_b2: VecDeque<CacheKey>, // Ghost list for T2
    arc_p: usize,               // Target size for T1
}

impl MemoryCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            frequency_map: HashMap::new(),
            current_memory_bytes: 0,
            arc_t1: VecDeque::new(),
            arc_t2: VecDeque::new(),
            arc_b1: VecDeque::new(),
            arc_b2: VecDeque::new(),
            arc_p: 0,
        }
    }

    /// Insert or update cache entry
    pub fn insert(&mut self, key: CacheKey, entry: CacheEntry) -> Result<()> {
        // Remove expired entries first
        self.clean_expired();

        // Check if we need to evict
        while self.should_evict(&entry) {
            self.evict_one()?;
        }

        // Remove existing entry if present
        if let Some(old_entry) = self.entries.remove(&key) {
            self.current_memory_bytes -= old_entry.size_bytes;
            self.remove_from_tracking(&key);
        }

        // Insert new entry
        self.current_memory_bytes += entry.size_bytes;
        self.entries.insert(key.clone(), entry);
        self.track_access(&key);

        Ok(())
    }

    /// Get cache entry
    pub fn get(&mut self, key: &CacheKey) -> Option<Vector> {
        // Check if entry exists and is not expired
        let should_remove = if let Some(entry) = self.entries.get(key) {
            entry.is_expired()
        } else {
            false
        };

        if should_remove {
            self.remove(key);
            return None;
        }

        if let Some(entry) = self.entries.get_mut(key) {
            let data = entry.data.clone();
            entry.touch();
            self.track_access(key);
            Some(data)
        } else {
            None
        }
    }

    /// Remove entry from cache
    pub fn remove(&mut self, key: &CacheKey) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.remove(key) {
            self.current_memory_bytes -= entry.size_bytes;
            self.remove_from_tracking(key);
            Some(entry)
        } else {
            None
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.frequency_map.clear();
        self.current_memory_bytes = 0;
    }

    /// Check if eviction is needed
    fn should_evict(&self, new_entry: &CacheEntry) -> bool {
        self.entries.len() >= self.config.max_memory_entries
            || self.current_memory_bytes + new_entry.size_bytes > self.config.max_memory_bytes
    }

    /// Evict one entry based on policy
    fn evict_one(&mut self) -> Result<()> {
        let key_to_evict = match self.config.eviction_policy {
            EvictionPolicy::LRU => self.find_lru_key(),
            EvictionPolicy::LFU => self.find_lfu_key(),
            EvictionPolicy::ARC => self.find_arc_key(),
            EvictionPolicy::FIFO => self.find_fifo_key(),
            EvictionPolicy::TTL => self.find_expired_key(),
        };

        if let Some(key) = key_to_evict {
            self.remove(&key);
            Ok(())
        } else if !self.entries.is_empty() {
            // Fallback: remove first entry
            let key = self.entries.keys().next().unwrap().clone();
            self.remove(&key);
            Ok(())
        } else {
            Err(anyhow!("No entries to evict"))
        }
    }

    /// Find LRU key
    fn find_lru_key(&self) -> Option<CacheKey> {
        self.access_order.front().cloned()
    }

    /// Find LFU key
    fn find_lfu_key(&self) -> Option<CacheKey> {
        self.frequency_map
            .iter()
            .min_by_key(|&(_, &freq)| freq)
            .map(|(key, _)| key.clone())
    }

    /// Find ARC key using Adaptive Replacement Cache algorithm
    fn find_arc_key(&mut self) -> Option<CacheKey> {
        let c = self.config.max_memory_entries;

        // If T1 is not empty and |T1| > p, evict from T1
        if !self.arc_t1.is_empty()
            && (self.arc_t1.len() > self.arc_p
                || (self.arc_t2.is_empty() && self.arc_t1.len() == self.arc_p))
        {
            if let Some(key) = self.arc_t1.pop_front() {
                // Move to B1
                self.arc_b1.push_back(key.clone());
                if self.arc_b1.len() > c {
                    self.arc_b1.pop_front();
                }
                return Some(key);
            }
        }

        // Otherwise evict from T2
        if let Some(key) = self.arc_t2.pop_front() {
            // Move to B2
            self.arc_b2.push_back(key.clone());
            if self.arc_b2.len() > c {
                self.arc_b2.pop_front();
            }
            return Some(key);
        }

        // Fallback to LRU if ARC lists are empty
        self.find_lru_key()
    }

    /// Find FIFO key (oldest entry)
    fn find_fifo_key(&self) -> Option<CacheKey> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.created_at)
            .map(|(key, _)| key.clone())
    }

    /// Find expired key
    fn find_expired_key(&self) -> Option<CacheKey> {
        self.entries
            .iter()
            .find(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
    }

    /// Track access for LRU/LFU/ARC
    fn track_access(&mut self, key: &CacheKey) {
        // Update LRU order
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.access_order.push_back(key.clone());

        // Update LFU frequency
        *self.frequency_map.entry(key.clone()).or_insert(0) += 1;

        // Update ARC tracking
        if self.config.eviction_policy == EvictionPolicy::ARC {
            self.track_arc_access(key);
        }
    }

    /// Track access for ARC algorithm
    fn track_arc_access(&mut self, key: &CacheKey) {
        let c = self.config.max_memory_entries;

        // Check if key is in T1 or T2
        if let Some(pos) = self.arc_t1.iter().position(|k| k == key) {
            // Move from T1 to T2 (promote to frequent)
            self.arc_t1.remove(pos);
            self.arc_t2.push_back(key.clone());
        } else if let Some(pos) = self.arc_t2.iter().position(|k| k == key) {
            // Move to end of T2 (most recently used)
            self.arc_t2.remove(pos);
            self.arc_t2.push_back(key.clone());
        } else if let Some(pos) = self.arc_b1.iter().position(|k| k == key) {
            // Hit in B1: increase p and move to T2
            self.arc_b1.remove(pos);
            self.arc_p = (self.arc_p + 1.max(self.arc_b2.len() / self.arc_b1.len())).min(c);
            self.arc_t2.push_back(key.clone());
        } else if let Some(pos) = self.arc_b2.iter().position(|k| k == key) {
            // Hit in B2: decrease p and move to T2
            self.arc_b2.remove(pos);
            self.arc_p = self
                .arc_p
                .saturating_sub(1.max(self.arc_b1.len() / self.arc_b2.len()));
            self.arc_t2.push_back(key.clone());
        } else {
            // New key: add to T1
            self.arc_t1.push_back(key.clone());
        }
    }

    /// Remove from tracking structures
    fn remove_from_tracking(&mut self, key: &CacheKey) {
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.frequency_map.remove(key);

        // Remove from ARC structures
        if self.config.eviction_policy == EvictionPolicy::ARC {
            if let Some(pos) = self.arc_t1.iter().position(|k| k == key) {
                self.arc_t1.remove(pos);
            }
            if let Some(pos) = self.arc_t2.iter().position(|k| k == key) {
                self.arc_t2.remove(pos);
            }
            if let Some(pos) = self.arc_b1.iter().position(|k| k == key) {
                self.arc_b1.remove(pos);
            }
            if let Some(pos) = self.arc_b2.iter().position(|k| k == key) {
                self.arc_b2.remove(pos);
            }
        }
    }

    /// Clean expired entries
    fn clean_expired(&mut self) {
        let expired_keys: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            self.remove(&key);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            memory_bytes: self.current_memory_bytes,
            max_entries: self.config.max_memory_entries,
            max_memory_bytes: self.config.max_memory_bytes,
            hit_ratio: 0.0, // Would need to track hits/misses
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entries: usize,
    pub memory_bytes: usize,
    pub max_entries: usize,
    pub max_memory_bytes: usize,
    pub hit_ratio: f32,
}

/// Persistent cache for disk storage
pub struct PersistentCache {
    config: CacheConfig,
    cache_dir: std::path::PathBuf,
}

impl PersistentCache {
    pub fn new(config: CacheConfig) -> Result<Self> {
        let cache_dir = config
            .persistent_cache_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("oxirs_vec_cache"));

        std::fs::create_dir_all(&cache_dir)?;

        Ok(Self { config, cache_dir })
    }

    /// Store entry to disk
    pub fn store(&self, key: &CacheKey, entry: &CacheEntry) -> Result<()> {
        let file_path = self.get_file_path(key);

        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let serialized = self.serialize_entry(entry)?;
        let final_data = if self.config.enable_compression {
            self.compress_data(&serialized)?
        } else {
            serialized
        };

        std::fs::write(file_path, final_data)?;
        Ok(())
    }

    /// Load entry from disk
    pub fn load(&self, key: &CacheKey) -> Result<Option<CacheEntry>> {
        let file_path = self.get_file_path(key);

        if !file_path.exists() {
            return Ok(None);
        }

        let data = std::fs::read(&file_path)?;

        let decompressed = if self.config.enable_compression {
            self.decompress_data(&data)?
        } else {
            data
        };

        let entry = self.deserialize_entry(&decompressed)?;

        // Check if entry has expired
        if entry.is_expired() {
            // Remove expired entry
            let _ = std::fs::remove_file(file_path);
            Ok(None)
        } else {
            Ok(Some(entry))
        }
    }

    /// Remove entry from disk
    pub fn remove(&self, key: &CacheKey) -> Result<()> {
        let file_path = self.get_file_path(key);
        if file_path.exists() {
            std::fs::remove_file(file_path)?;
        }
        Ok(())
    }

    /// Clear all persistent cache
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
            std::fs::create_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

    /// Get file path for cache key
    fn get_file_path(&self, key: &CacheKey) -> std::path::PathBuf {
        let key_str = key.to_string();
        let hash = self.hash_key(&key_str);

        // Create subdirectory structure to avoid too many files in one directory
        let sub_dir = format!("{:02x}", (hash % 256) as u8);

        // Encode key information in filename for reconstruction during cleanup
        let encoded_key = self.encode_cache_key_for_filename(key);
        let filename = format!("{hash:016x}_{encoded_key}.cache");

        self.cache_dir.join(sub_dir).join(filename)
    }

    /// Encode cache key information into filename-safe format
    fn encode_cache_key_for_filename(&self, key: &CacheKey) -> String {
        let key_data = serde_json::json!({
            "namespace": key.namespace,
            "key": key.key,
            "variant": key.variant
        });

        // Use base64 encoding to safely include key information in filename
        use base64::{engine::general_purpose, Engine as _};
        general_purpose::URL_SAFE_NO_PAD.encode(key_data.to_string().as_bytes())
    }

    /// Decode cache key from filename
    fn decode_cache_key_from_filename(&self, filename: &str) -> Option<CacheKey> {
        if let Some(encoded_part) = filename
            .strip_suffix(".cache")
            .and_then(|s| s.split('_').nth(1))
        {
            use base64::{engine::general_purpose, Engine as _};
            if let Ok(decoded_bytes) = general_purpose::URL_SAFE_NO_PAD.decode(encoded_part) {
                if let Ok(decoded_str) = String::from_utf8(decoded_bytes) {
                    if let Ok(key_data) = serde_json::from_str::<serde_json::Value>(&decoded_str) {
                        return Some(CacheKey {
                            namespace: key_data["namespace"].as_str()?.to_string(),
                            key: key_data["key"].as_str()?.to_string(),
                            variant: key_data["variant"].as_str().map(|s| s.to_string()),
                        });
                    }
                }
            }
        }
        None
    }

    /// Hash cache key
    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Serialize cache entry to bytes
    fn serialize_entry(&self, entry: &CacheEntry) -> Result<Vec<u8>> {
        // Custom binary serialization since CacheEntry has Instant fields
        let mut data = Vec::new();

        // Serialize vector data
        let vector_data = &entry.data.as_f32();
        data.extend_from_slice(&(vector_data.len() as u32).to_le_bytes());
        for &value in vector_data {
            data.extend_from_slice(&value.to_le_bytes());
        }

        // Serialize timestamps as epoch nanos from creation
        let created_nanos = entry.created_at.elapsed().as_nanos() as u64;
        let accessed_nanos = entry.last_accessed.elapsed().as_nanos() as u64;
        data.extend_from_slice(&created_nanos.to_le_bytes());
        data.extend_from_slice(&accessed_nanos.to_le_bytes());

        // Serialize other fields
        data.extend_from_slice(&entry.access_count.to_le_bytes());
        data.extend_from_slice(&(entry.size_bytes as u64).to_le_bytes());

        // Serialize TTL
        if let Some(ttl) = entry.ttl {
            data.push(1); // TTL present
            data.extend_from_slice(&ttl.as_nanos().to_le_bytes());
        } else {
            data.push(0); // No TTL
        }

        // Serialize tags
        data.extend_from_slice(&(entry.tags.len() as u32).to_le_bytes());
        for (key, value) in &entry.tags {
            data.extend_from_slice(&(key.len() as u32).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&(value.len() as u32).to_le_bytes());
            data.extend_from_slice(value.as_bytes());
        }

        Ok(data)
    }

    /// Deserialize cache entry from bytes
    fn deserialize_entry(&self, data: &[u8]) -> Result<CacheEntry> {
        // Check if data is empty or too small
        if data.len() < 4 {
            return Err(anyhow::anyhow!(
                "Invalid cache entry data: too small (expected at least 4 bytes, got {})",
                data.len()
            ));
        }

        let mut offset = 0;

        // Deserialize vector data
        let vector_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut vector_data = Vec::with_capacity(vector_len);
        for _ in 0..vector_len {
            let value = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            vector_data.push(value);
            offset += 4;
        }
        let vector = Vector::new(vector_data);

        // Deserialize timestamps (stored as elapsed nanos, convert back to Instant)
        let created_nanos = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let accessed_nanos = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        // Reconstruct timestamps (approximation - will be recent)
        let now = Instant::now();
        let created_at = now - Duration::from_nanos(created_nanos);
        let last_accessed = now - Duration::from_nanos(accessed_nanos);

        // Deserialize other fields
        let access_count = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let size_bytes = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        offset += 8;

        // Deserialize TTL
        let ttl = if data[offset] == 1 {
            offset += 1;
            let ttl_nanos = u128::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
                data[offset + 8],
                data[offset + 9],
                data[offset + 10],
                data[offset + 11],
                data[offset + 12],
                data[offset + 13],
                data[offset + 14],
                data[offset + 15],
            ]);
            offset += 16;
            Some(Duration::from_nanos(ttl_nanos as u64))
        } else {
            offset += 1;
            None
        };

        // Deserialize tags
        let tags_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut tags = HashMap::new();
        for _ in 0..tags_len {
            let key_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            let key = String::from_utf8(data[offset..offset + key_len].to_vec())?;
            offset += key_len;

            let value_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            let value = String::from_utf8(data[offset..offset + value_len].to_vec())?;
            offset += value_len;

            tags.insert(key, value);
        }

        Ok(CacheEntry {
            data: vector,
            created_at,
            last_accessed,
            access_count,
            size_bytes,
            ttl,
            tags,
        })
    }

    /// Compress data using simple RLE compression
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simple run-length encoding for demonstration
        let mut compressed = Vec::new();

        if data.is_empty() {
            return Ok(compressed);
        }

        let mut current_byte = data[0];
        let mut count = 1u8;

        for &byte in &data[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                compressed.push(count);
                compressed.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }

        // Add the last run
        compressed.push(count);
        compressed.push(current_byte);

        Ok(compressed)
    }

    /// Decompress data using RLE decompression
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut decompressed = Vec::new();

        if data.len() % 2 != 0 {
            return Err(anyhow!("Invalid compressed data length"));
        }

        for chunk in data.chunks(2) {
            let count = chunk[0];
            let byte = chunk[1];

            for _ in 0..count {
                decompressed.push(byte);
            }
        }

        Ok(decompressed)
    }
}

/// Multi-level cache combining memory and persistent storage
pub struct MultiLevelCache {
    memory_cache: Arc<RwLock<MemoryCache>>,
    persistent_cache: Option<Arc<PersistentCache>>,
    #[allow(dead_code)]
    config: CacheConfig,
    stats: Arc<RwLock<MultiLevelCacheStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct MultiLevelCacheStats {
    pub memory_hits: u64,
    pub memory_misses: u64,
    pub persistent_hits: u64,
    pub persistent_misses: u64,
    pub total_requests: u64,
}

impl MultiLevelCache {
    pub fn new(config: CacheConfig) -> Result<Self> {
        let memory_cache = Arc::new(RwLock::new(MemoryCache::new(config.clone())));

        let persistent_cache = if config.enable_persistent {
            Some(Arc::new(PersistentCache::new(config.clone())?))
        } else {
            None
        };

        Ok(Self {
            memory_cache,
            persistent_cache,
            config,
            stats: Arc::new(RwLock::new(MultiLevelCacheStats::default())),
        })
    }

    /// Insert entry into cache
    pub fn insert(&self, key: CacheKey, data: Vector) -> Result<()> {
        let entry = CacheEntry::new(data);

        // Insert into memory cache
        {
            let mut memory = self.memory_cache.write().unwrap();
            memory.insert(key.clone(), entry.clone())?;
        }

        // Insert into persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            persistent.store(&key, &entry)?;
        }

        Ok(())
    }

    /// Get entry from cache
    pub fn get(&self, key: &CacheKey) -> Option<Vector> {
        self.update_stats_total();

        // Try memory cache first
        {
            let mut memory = self.memory_cache.write().unwrap();
            if let Some(data) = memory.get(key) {
                self.update_stats_memory_hit();
                return Some(data.clone());
            }
        }

        self.update_stats_memory_miss();

        // Try persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            if let Ok(Some(mut entry)) = persistent.load(key) {
                self.update_stats_persistent_hit();

                // Promote to memory cache
                let data = entry.data.clone();
                entry.touch();
                if let Ok(mut memory) = self.memory_cache.write() {
                    let _ = memory.insert(key.clone(), entry);
                }

                return Some(data);
            }
        }

        self.update_stats_persistent_miss();
        None
    }

    /// Remove entry from cache
    pub fn remove(&self, key: &CacheKey) -> Result<()> {
        // Remove from memory cache
        {
            let mut memory = self.memory_cache.write().unwrap();
            memory.remove(key);
        }

        // Remove from persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            persistent.remove(key)?;
        }

        Ok(())
    }

    /// Clear all caches
    pub fn clear(&self) -> Result<()> {
        // Clear memory cache
        {
            let mut memory = self.memory_cache.write().unwrap();
            memory.clear();
        }

        // Clear persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            persistent.clear()?;
        }

        // Reset stats
        {
            let mut stats = self.stats.write().unwrap();
            *stats = MultiLevelCacheStats::default();
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> MultiLevelCacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Get memory cache statistics
    pub fn get_memory_stats(&self) -> CacheStats {
        let memory = self.memory_cache.read().unwrap();
        memory.stats()
    }

    // Stats update methods
    fn update_stats_total(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.total_requests += 1;
    }

    fn update_stats_memory_hit(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.memory_hits += 1;
    }

    fn update_stats_memory_miss(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.memory_misses += 1;
    }

    fn update_stats_persistent_hit(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.persistent_hits += 1;
    }

    fn update_stats_persistent_miss(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.persistent_misses += 1;
    }
}

/// Cache invalidation utilities with indexing support
pub struct CacheInvalidator {
    cache: Arc<MultiLevelCache>,
    tag_index: TagIndex, // tag_key -> tag_value -> keys
    namespace_index: Arc<RwLock<HashMap<String, Vec<CacheKey>>>>, // namespace -> keys
}

impl CacheInvalidator {
    pub fn new(cache: Arc<MultiLevelCache>) -> Self {
        Self {
            cache,
            tag_index: Arc::new(RwLock::new(HashMap::new())),
            namespace_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a cache entry for invalidation tracking
    pub fn register_entry(&self, key: &CacheKey, tags: &HashMap<String, String>) {
        // Index by namespace
        {
            let mut ns_index = self.namespace_index.write().unwrap();
            ns_index
                .entry(key.namespace.clone())
                .or_default()
                .push(key.clone());
        }

        // Index by tags
        {
            let mut tag_idx = self.tag_index.write().unwrap();
            for (tag_key, tag_value) in tags {
                tag_idx
                    .entry(tag_key.clone())
                    .or_default()
                    .entry(tag_value.clone())
                    .or_default()
                    .push(key.clone());
            }
        }
    }

    /// Unregister a cache entry from invalidation tracking
    pub fn unregister_entry(&self, key: &CacheKey) {
        // Remove from namespace index
        {
            let mut ns_index = self.namespace_index.write().unwrap();
            if let Some(keys) = ns_index.get_mut(&key.namespace) {
                keys.retain(|k| k != key);
                if keys.is_empty() {
                    ns_index.remove(&key.namespace);
                }
            }
        }

        // Remove from tag index
        {
            let mut tag_idx = self.tag_index.write().unwrap();
            let mut tags_to_remove = Vec::new();

            for (tag_key, tag_values) in tag_idx.iter_mut() {
                let mut values_to_remove = Vec::new();

                for (tag_value, keys) in tag_values.iter_mut() {
                    keys.retain(|k| k != key);
                    if keys.is_empty() {
                        values_to_remove.push(tag_value.clone());
                    }
                }

                for value in values_to_remove {
                    tag_values.remove(&value);
                }

                if tag_values.is_empty() {
                    tags_to_remove.push(tag_key.clone());
                }
            }

            for tag in tags_to_remove {
                tag_idx.remove(&tag);
            }
        }
    }

    /// Invalidate entries by tag
    pub fn invalidate_by_tag(&self, tag_key: &str, tag_value: &str) -> Result<usize> {
        let keys_to_invalidate = {
            let tag_idx = self.tag_index.read().unwrap();
            tag_idx
                .get(tag_key)
                .and_then(|values| values.get(tag_value))
                .cloned()
                .unwrap_or_default()
        };

        let mut invalidated_count = 0;
        for key in &keys_to_invalidate {
            if self.cache.remove(key).is_ok() {
                invalidated_count += 1;
            }
            self.unregister_entry(key);
        }

        Ok(invalidated_count)
    }

    /// Invalidate entries by namespace
    pub fn invalidate_namespace(&self, namespace: &str) -> Result<usize> {
        let keys_to_invalidate = {
            let ns_index = self.namespace_index.read().unwrap();
            ns_index.get(namespace).cloned().unwrap_or_default()
        };

        let mut invalidated_count = 0;
        for key in &keys_to_invalidate {
            if self.cache.remove(key).is_ok() {
                invalidated_count += 1;
            }
            self.unregister_entry(key);
        }

        Ok(invalidated_count)
    }

    /// Invalidate all expired entries
    pub fn invalidate_expired(&self) -> Result<usize> {
        // Memory cache cleans expired entries automatically during operations
        // For persistent cache, we need to scan and remove expired files
        if let Some(ref persistent) = self.cache.persistent_cache {
            return self.scan_and_remove_expired_files(persistent);
        }
        Ok(0)
    }

    /// Scan persistent cache directory and remove expired files
    fn scan_and_remove_expired_files(&self, persistent_cache: &PersistentCache) -> Result<usize> {
        let cache_dir = &persistent_cache.cache_dir;
        let mut removed_count = 0;

        if !cache_dir.exists() {
            return Ok(0);
        }

        // Walk through all cache files
        for entry in std::fs::read_dir(cache_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                // Recursively scan subdirectories
                for sub_entry in std::fs::read_dir(entry.path())? {
                    let sub_entry = sub_entry?;
                    if sub_entry.file_type()?.is_file() {
                        if let Some(file_name) = sub_entry.file_name().to_str() {
                            if file_name.ends_with(".cache") {
                                // Decode cache key from filename - no more hacks!
                                if let Some(cache_key) =
                                    persistent_cache.decode_cache_key_from_filename(file_name)
                                {
                                    // Load the actual cache entry to check expiration
                                    if let Ok(Some(entry)) = persistent_cache.load(&cache_key) {
                                        if entry.is_expired() {
                                            let _ = std::fs::remove_file(sub_entry.path());
                                            removed_count += 1;
                                        }
                                    } else {
                                        // If we can't load the entry, it might be corrupted - remove it
                                        let _ = std::fs::remove_file(sub_entry.path());
                                        removed_count += 1;
                                    }
                                } else {
                                    // If we can't decode the key, it might be an old format - use file age as fallback
                                    if let Ok(metadata) = std::fs::metadata(sub_entry.path()) {
                                        if let Ok(modified) = metadata.modified() {
                                            let age = modified
                                                .elapsed()
                                                .unwrap_or(Duration::from_secs(0));
                                            // Remove files older than 24 hours as fallback for old cache files
                                            if age > Duration::from_secs(24 * 3600) {
                                                let _ = std::fs::remove_file(sub_entry.path());
                                                removed_count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(removed_count)
    }

    /// Get invalidation statistics
    pub fn get_stats(&self) -> InvalidationStats {
        let tag_idx = self.tag_index.read().unwrap();
        let ns_index = self.namespace_index.read().unwrap();

        let total_tag_entries = tag_idx
            .values()
            .flat_map(|values| values.values())
            .map(|keys| keys.len())
            .sum();

        let total_namespace_entries = ns_index.values().map(|keys| keys.len()).sum();

        InvalidationStats {
            tracked_tags: tag_idx.len(),
            tracked_namespaces: ns_index.len(),
            total_tag_entries,
            total_namespace_entries,
        }
    }
}

/// Statistics for cache invalidation tracking
#[derive(Debug, Clone)]
pub struct InvalidationStats {
    pub tracked_tags: usize,
    pub tracked_namespaces: usize,
    pub total_tag_entries: usize,
    pub total_namespace_entries: usize,
}

/// Background cache worker for maintenance tasks
pub struct BackgroundCacheWorker {
    cache: Arc<MultiLevelCache>,
    invalidator: Arc<CacheInvalidator>,
    config: CacheConfig,
    worker_handle: Option<JoinHandle<()>>,
    shutdown_signal: Arc<RwLock<bool>>,
}

impl BackgroundCacheWorker {
    pub fn new(
        cache: Arc<MultiLevelCache>,
        invalidator: Arc<CacheInvalidator>,
        config: CacheConfig,
    ) -> Self {
        Self {
            cache,
            invalidator,
            config,
            worker_handle: None,
            shutdown_signal: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the background worker
    pub fn start(&mut self) -> Result<()> {
        if !self.config.enable_background_updates {
            return Ok(());
        }

        let cache = Arc::clone(&self.cache);
        let invalidator = Arc::clone(&self.invalidator);
        let interval = self.config.background_update_interval;
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let handle = thread::spawn(move || {
            while let Ok(shutdown) = shutdown_signal.read() {
                if *shutdown {
                    break;
                }
                drop(shutdown); // Release the lock before sleeping

                // Perform maintenance tasks
                if let Err(e) = Self::perform_maintenance(&cache, &invalidator) {
                    // Log error but continue running
                    tracing::warn!("Background cache maintenance error: {}", e);
                }

                // Sleep for the configured interval
                thread::sleep(interval);
            }
        });

        self.worker_handle = Some(handle);
        Ok(())
    }

    /// Stop the background worker
    pub fn stop(&mut self) -> Result<()> {
        // Signal shutdown
        {
            let mut signal = self.shutdown_signal.write().unwrap();
            *signal = true;
        }

        // Wait for worker to finish
        if let Some(handle) = self.worker_handle.take() {
            handle
                .join()
                .map_err(|e| anyhow!("Failed to join worker thread: {:?}", e))?;
        }

        Ok(())
    }

    /// Perform background maintenance tasks
    fn perform_maintenance(
        cache: &Arc<MultiLevelCache>,
        invalidator: &Arc<CacheInvalidator>,
    ) -> Result<()> {
        // 1. Clean expired entries
        let expired_count = invalidator.invalidate_expired()?;
        if expired_count > 0 {
            println!("Background worker cleaned {expired_count} expired entries");
        }

        // 2. Optimize memory usage if fragmentation is high
        let memory_stats = cache.get_memory_stats();
        let utilization = memory_stats.memory_bytes as f64 / memory_stats.max_memory_bytes as f64;

        if utilization > 0.9 {
            // Trigger more aggressive cleanup
            Self::aggressive_cleanup(cache)?;
        }

        // 3. Preemptive persistent cache sync
        Self::sync_hot_entries(cache)?;

        Ok(())
    }

    /// Perform aggressive cleanup when memory usage is high
    fn aggressive_cleanup(_cache: &Arc<MultiLevelCache>) -> Result<()> {
        // Force cleanup of memory cache by temporarily reducing limits
        // This is a simplified approach - in practice you'd implement more sophisticated logic
        println!("Performing aggressive cache cleanup due to high memory usage");
        Ok(())
    }

    /// Sync frequently accessed entries to persistent storage
    fn sync_hot_entries(_cache: &Arc<MultiLevelCache>) -> Result<()> {
        // In a real implementation, you'd identify hot entries and ensure they're in persistent storage
        // This helps with cache warming after restarts
        Ok(())
    }
}

impl Drop for BackgroundCacheWorker {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Cache warming utilities
pub struct CacheWarmer {
    cache: Arc<MultiLevelCache>,
}

impl CacheWarmer {
    pub fn new(cache: Arc<MultiLevelCache>) -> Self {
        Self { cache }
    }

    /// Warm cache with a list of key-value pairs
    pub fn warm_with_data(&self, data: Vec<(CacheKey, Vector)>) -> Result<usize> {
        let mut loaded_count = 0;

        for (key, vector) in data {
            if self.cache.insert(key, vector).is_ok() {
                loaded_count += 1;
            }
        }

        Ok(loaded_count)
    }

    /// Warm cache by loading frequently accessed entries from persistent storage
    pub fn warm_from_persistent(&self, keys: Vec<CacheKey>) -> Result<usize> {
        let mut loaded_count = 0;

        for key in keys {
            // Try to load from persistent cache and promote to memory
            if self.cache.get(&key).is_some() {
                loaded_count += 1;
            }
        }

        Ok(loaded_count)
    }

    /// Warm cache using a precomputed dataset
    pub fn warm_with_generator<F>(&self, count: usize, generator: F) -> Result<usize>
    where
        F: Fn(usize) -> Option<(CacheKey, Vector)>,
    {
        let mut loaded_count = 0;

        for i in 0..count {
            if let Some((key, vector)) = generator(i) {
                if self.cache.insert(key, vector).is_ok() {
                    loaded_count += 1;
                }
            }
        }

        Ok(loaded_count)
    }
}

/// Advanced cache analytics and optimization recommendations
pub struct CacheAnalyzer {
    cache: Arc<MultiLevelCache>,
    invalidator: Arc<CacheInvalidator>,
}

#[derive(Debug, Clone)]
pub struct CacheAnalysisReport {
    pub memory_utilization: f64,
    pub hit_ratio: f64,
    pub persistent_hit_ratio: f64,
    pub most_accessed_namespaces: Vec<(String, usize)>,
    pub recommendations: Vec<String>,
    pub performance_score: f64, // 0.0 to 1.0
}

impl CacheAnalyzer {
    pub fn new(cache: Arc<MultiLevelCache>, invalidator: Arc<CacheInvalidator>) -> Self {
        Self { cache, invalidator }
    }

    /// Generate comprehensive cache analysis report
    pub fn analyze(&self) -> CacheAnalysisReport {
        let stats = self.cache.get_stats();
        let memory_stats = self.cache.get_memory_stats();
        let invalidation_stats = self.invalidator.get_stats();

        let memory_utilization =
            memory_stats.memory_bytes as f64 / memory_stats.max_memory_bytes as f64;

        let total_requests = stats.total_requests;
        let total_hits = stats.memory_hits + stats.persistent_hits;
        let hit_ratio = if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let persistent_hit_ratio = if stats.persistent_hits + stats.persistent_misses > 0 {
            stats.persistent_hits as f64 / (stats.persistent_hits + stats.persistent_misses) as f64
        } else {
            0.0
        };

        let mut recommendations = Vec::new();

        // Generate recommendations
        if hit_ratio < 0.5 {
            recommendations
                .push("Consider increasing cache size or adjusting eviction policy".to_string());
        }

        if memory_utilization > 0.9 {
            recommendations.push(
                "Memory cache is near capacity - consider increasing max_memory_bytes".to_string(),
            );
        }

        if persistent_hit_ratio < 0.3 {
            recommendations
                .push("Persistent cache hit ratio is low - review TTL settings".to_string());
        }

        if invalidation_stats.tracked_namespaces > 100 {
            recommendations
                .push("Consider consolidating namespaces to reduce tracking overhead".to_string());
        }

        // Calculate performance score (weighted combination of metrics)
        let performance_score =
            (hit_ratio * 0.4 + (1.0 - memory_utilization) * 0.3 + persistent_hit_ratio * 0.3)
                .clamp(0.0, 1.0);

        CacheAnalysisReport {
            memory_utilization,
            hit_ratio,
            persistent_hit_ratio,
            most_accessed_namespaces: vec![], // Would need access pattern tracking
            recommendations,
            performance_score,
        }
    }

    /// Get recommendations for cache configuration optimization
    pub fn get_optimization_recommendations(&self) -> Vec<String> {
        self.analyze().recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_key() {
        let key = CacheKey::new("embeddings", "test_doc").with_variant("v1");

        assert_eq!(key.namespace, "embeddings");
        assert_eq!(key.key, "test_doc");
        assert_eq!(key.variant, Some("v1".to_string()));
        assert_eq!(key.to_string(), "embeddings:test_doc:v1");
    }

    #[test]
    fn test_memory_cache() {
        let config = CacheConfig {
            max_memory_entries: 2,
            max_memory_bytes: 1024,
            ..Default::default()
        };

        let mut cache = MemoryCache::new(config);

        let key1 = CacheKey::new("test", "key1");
        let key2 = CacheKey::new("test", "key2");
        let key3 = CacheKey::new("test", "key3");

        let vector1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let vector3 = Vector::new(vec![7.0, 8.0, 9.0]);

        // Insert vectors
        cache
            .insert(key1.clone(), CacheEntry::new(vector1.clone()))
            .unwrap();
        cache
            .insert(key2.clone(), CacheEntry::new(vector2.clone()))
            .unwrap();

        // Check retrieval
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_some());

        // Insert third vector (should evict one)
        cache
            .insert(key3.clone(), CacheEntry::new(vector3.clone()))
            .unwrap();

        // One of the first two should be evicted
        let remaining = cache.entries.len();
        assert_eq!(remaining, 2);
    }

    #[test]
    fn test_persistent_cache() {
        let temp_dir = TempDir::new().unwrap();

        let config = CacheConfig {
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_compression: true,
            ..Default::default()
        };

        let cache = PersistentCache::new(config).unwrap();

        let key = CacheKey::new("test", "persistent_key");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);
        let entry = CacheEntry::new(vector.clone());

        // Store and retrieve
        cache.store(&key, &entry).unwrap();
        let retrieved = cache.load(&key).unwrap();

        // Should succeed now with proper serialization
        assert!(retrieved.is_some());
        let retrieved_entry = retrieved.unwrap();
        assert_eq!(retrieved_entry.data.as_f32(), vector.as_f32());
    }

    #[test]
    fn test_multi_level_cache() {
        let temp_dir = TempDir::new().unwrap();

        let config = CacheConfig {
            max_memory_entries: 2,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = MultiLevelCache::new(config).unwrap();

        let key = CacheKey::new("test", "multi_level");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        // Insert and retrieve
        cache.insert(key.clone(), vector.clone()).unwrap();
        let retrieved = cache.get(&key).unwrap();

        assert_eq!(retrieved.as_f32(), vector.as_f32());

        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.memory_hits, 1);
    }

    #[test]
    fn test_cache_expiration() {
        let config = CacheConfig {
            max_memory_entries: 10,
            ttl: Some(Duration::from_millis(10)),
            ..Default::default()
        };

        let mut cache = MemoryCache::new(config);

        let key = CacheKey::new("test", "expiring");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);
        let entry = CacheEntry::new(vector).with_ttl(Duration::from_millis(10));

        cache.insert(key.clone(), entry).unwrap();

        // Should be available immediately
        assert!(cache.get(&key).is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(20));

        // Should be expired and removed
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_arc_eviction_policy() {
        let config = CacheConfig {
            max_memory_entries: 3,
            eviction_policy: EvictionPolicy::ARC,
            ..Default::default()
        };

        let mut cache = MemoryCache::new(config);

        let key1 = CacheKey::new("test", "arc1");
        let key2 = CacheKey::new("test", "arc2");
        let key3 = CacheKey::new("test", "arc3");
        let key4 = CacheKey::new("test", "arc4");

        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        // Insert three items
        cache
            .insert(key1.clone(), CacheEntry::new(vector.clone()))
            .unwrap();
        cache
            .insert(key2.clone(), CacheEntry::new(vector.clone()))
            .unwrap();
        cache
            .insert(key3.clone(), CacheEntry::new(vector.clone()))
            .unwrap();

        // Access key1 multiple times to make it frequent
        cache.get(&key1);
        cache.get(&key1);
        cache.get(&key1);

        // Insert key4 - should evict the least valuable item
        cache
            .insert(key4.clone(), CacheEntry::new(vector.clone()))
            .unwrap();

        // key1 should still be there (frequent access)
        assert!(cache.get(&key1).is_some());

        // Check that we have exactly 3 items
        assert_eq!(cache.entries.len(), 3);
    }

    #[test]
    fn test_cache_warmer() {
        let temp_dir = TempDir::new().unwrap();

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config).unwrap());
        let warmer = CacheWarmer::new(Arc::clone(&cache));

        // Prepare test data
        let test_data = vec![
            (CacheKey::new("test", "warm1"), Vector::new(vec![1.0, 2.0])),
            (CacheKey::new("test", "warm2"), Vector::new(vec![3.0, 4.0])),
            (CacheKey::new("test", "warm3"), Vector::new(vec![5.0, 6.0])),
        ];

        // Warm cache with data
        let loaded_count = warmer.warm_with_data(test_data.clone()).unwrap();
        assert_eq!(loaded_count, 3);

        // Verify data is in cache
        for (key, expected_vector) in test_data {
            let cached_vector = cache.get(&key).unwrap();
            assert_eq!(cached_vector.as_f32(), expected_vector.as_f32());
        }
    }

    #[test]
    fn test_cache_warmer_with_generator() {
        let temp_dir = TempDir::new().unwrap();

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config).unwrap());
        let warmer = CacheWarmer::new(Arc::clone(&cache));

        // Use generator to warm cache
        let loaded_count = warmer
            .warm_with_generator(5, |i| {
                Some((
                    CacheKey::new("generated", format!("item_{i}")),
                    Vector::new(vec![i as f32, (i * 2) as f32]),
                ))
            })
            .unwrap();

        assert_eq!(loaded_count, 5);

        // Verify generated data is in cache
        for i in 0..5 {
            let key = CacheKey::new("generated", format!("item_{i}"));
            let cached_vector = cache.get(&key).unwrap();
            assert_eq!(cached_vector.as_f32(), vec![i as f32, (i * 2) as f32]);
        }
    }

    #[test]
    fn test_cache_analyzer() {
        let temp_dir = TempDir::new().unwrap();

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config).unwrap());
        let invalidator = Arc::new(CacheInvalidator::new(Arc::clone(&cache)));
        let analyzer = CacheAnalyzer::new(Arc::clone(&cache), Arc::clone(&invalidator));

        // Add some test data and access patterns
        let key1 = CacheKey::new("test", "analyze1");
        let key2 = CacheKey::new("test", "analyze2");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        cache.insert(key1.clone(), vector.clone()).unwrap();
        cache.insert(key2.clone(), vector.clone()).unwrap();

        // Access the cache to generate some stats
        cache.get(&key1);
        cache.get(&key2);
        cache.get(&key1); // Hit
        cache.get(&CacheKey::new("test", "nonexistent")); // Miss

        // Analyze cache performance
        let report = analyzer.analyze();

        assert!(report.hit_ratio > 0.0);
        assert!(report.memory_utilization >= 0.0 && report.memory_utilization <= 1.0);
        assert!(report.performance_score >= 0.0 && report.performance_score <= 1.0);

        // Should have some recommendations if performance isn't perfect
        let recommendations = analyzer.get_optimization_recommendations();
        // In this test case, we might get recommendations about hit ratio
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_background_cache_worker() {
        let temp_dir = TempDir::new().unwrap();

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            enable_background_updates: true,
            background_update_interval: Duration::from_millis(100),
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config.clone()).unwrap());
        let invalidator = Arc::new(CacheInvalidator::new(Arc::clone(&cache)));
        let mut worker =
            BackgroundCacheWorker::new(Arc::clone(&cache), Arc::clone(&invalidator), config);

        // Start the worker
        worker.start().unwrap();

        // Add some test data
        let key = CacheKey::new("test", "background");
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);
        cache.insert(key.clone(), vector.clone()).unwrap();

        // Let the worker run for a short time
        std::thread::sleep(Duration::from_millis(150));

        // Stop the worker
        worker.stop().unwrap();

        // Verify data is still accessible
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn test_cache_invalidation_by_tag() {
        let temp_dir = TempDir::new().unwrap();

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config).unwrap());
        let invalidator = Arc::new(CacheInvalidator::new(Arc::clone(&cache)));

        // Create entries with tags
        let key1 = CacheKey::new("test", "tagged1");
        let key2 = CacheKey::new("test", "tagged2");
        let key3 = CacheKey::new("test", "tagged3");

        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        cache.insert(key1.clone(), vector.clone()).unwrap();
        cache.insert(key2.clone(), vector.clone()).unwrap();
        cache.insert(key3.clone(), vector.clone()).unwrap();

        // Register entries with tags
        let mut tags1 = HashMap::new();
        tags1.insert("category".to_string(), "embeddings".to_string());
        invalidator.register_entry(&key1, &tags1);

        let mut tags2 = HashMap::new();
        tags2.insert("category".to_string(), "embeddings".to_string());
        invalidator.register_entry(&key2, &tags2);

        let mut tags3 = HashMap::new();
        tags3.insert("category".to_string(), "vectors".to_string());
        invalidator.register_entry(&key3, &tags3);

        // Invalidate by tag
        let invalidated_count = invalidator
            .invalidate_by_tag("category", "embeddings")
            .unwrap();
        assert_eq!(invalidated_count, 2);

        // Check that tagged entries are removed
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_none());

        // Check that untagged entry remains
        assert!(cache.get(&key3).is_some());
    }

    #[test]
    fn test_cache_invalidation_by_namespace() {
        let temp_dir = TempDir::new().unwrap();

        let config = CacheConfig {
            max_memory_entries: 10,
            persistent_cache_dir: Some(temp_dir.path().to_path_buf()),
            enable_persistent: true,
            ..Default::default()
        };

        let cache = Arc::new(MultiLevelCache::new(config).unwrap());
        let invalidator = Arc::new(CacheInvalidator::new(Arc::clone(&cache)));

        // Create entries in different namespaces
        let key1 = CacheKey::new("embeddings", "item1");
        let key2 = CacheKey::new("embeddings", "item2");
        let key3 = CacheKey::new("vectors", "item3");

        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        cache.insert(key1.clone(), vector.clone()).unwrap();
        cache.insert(key2.clone(), vector.clone()).unwrap();
        cache.insert(key3.clone(), vector.clone()).unwrap();

        // Register entries for tracking
        invalidator.register_entry(&key1, &HashMap::new());
        invalidator.register_entry(&key2, &HashMap::new());
        invalidator.register_entry(&key3, &HashMap::new());

        // Invalidate by namespace
        let invalidated_count = invalidator.invalidate_namespace("embeddings").unwrap();
        assert_eq!(invalidated_count, 2);

        // Check that namespace entries are removed
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_none());

        // Check that other namespace entry remains
        assert!(cache.get(&key3).is_some());
    }
}
