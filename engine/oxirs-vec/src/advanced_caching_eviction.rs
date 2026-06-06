//! Eviction policies and cache storage implementations for the advanced caching system.
//!
//! Contains:
//! - `MemoryCache` — in-process LRU/LFU/ARC/FIFO/TTL cache
//! - `PersistentCache` — disk-backed cache with optional RLE compression

use crate::advanced_caching::{CacheConfig, CacheEntry, CacheKey, CacheStats, EvictionPolicy};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

/// Memory cache implementation
pub struct MemoryCache {
    pub(super) config: CacheConfig,
    pub(super) entries: HashMap<CacheKey, CacheEntry>,
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
    pub fn get(&mut self, key: &CacheKey) -> Option<crate::Vector> {
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
            let key = self
                .entries
                .keys()
                .next()
                .expect("entries should not be empty when at capacity")
                .clone();
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

// ---------------------------------------------------------------------------
// PersistentCache
// ---------------------------------------------------------------------------

/// Persistent cache for disk storage
pub struct PersistentCache {
    pub(super) config: CacheConfig,
    pub(super) cache_dir: std::path::PathBuf,
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
    pub(super) fn get_file_path(&self, key: &CacheKey) -> std::path::PathBuf {
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
    pub(super) fn decode_cache_key_from_filename(&self, filename: &str) -> Option<CacheKey> {
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
    pub(super) fn serialize_entry(&self, entry: &CacheEntry) -> Result<Vec<u8>> {
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
    pub(super) fn deserialize_entry(&self, data: &[u8]) -> Result<CacheEntry> {
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
        let vector = crate::Vector::new(vector_data);

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
    pub(super) fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
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
    pub(super) fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
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
