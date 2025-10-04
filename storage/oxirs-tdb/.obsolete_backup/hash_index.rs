//! # Hash Index Implementation with Linear Hashing
//!
//! High-performance hash index implementation using linear hashing for dynamic
//! table growth and excellent performance characteristics for equality queries.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// Hash function type for flexible hashing strategies
pub type HashFn = fn(&[u8]) -> u64;

/// Linear hash table configuration
#[derive(Debug, Clone)]
pub struct HashIndexConfig {
    /// Initial number of buckets (power of 2)
    pub initial_buckets: usize,
    /// Target load factor for triggering splits
    pub max_load_factor: f64,
    /// Maximum entries per bucket before overflow
    pub max_bucket_size: usize,
    /// Enable overflow bucket chaining
    pub enable_overflow: bool,
    /// Hash function to use
    pub hash_function: HashFn,
}

impl Default for HashIndexConfig {
    fn default() -> Self {
        Self {
            initial_buckets: 256,
            max_load_factor: 0.75,
            max_bucket_size: 32,
            enable_overflow: true,
            hash_function: Self::default_hash,
        }
    }
}

impl HashIndexConfig {
    /// Default FNV-1a hash function
    fn default_hash(data: &[u8]) -> u64 {
        let mut hash = 14695981039346656037u64; // FNV offset basis
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211u64); // FNV prime
        }
        hash
    }

    /// Fast CRC32-based hash function
    pub fn crc32_hash(data: &[u8]) -> u64 {
        let mut hash = 0u32;
        for &byte in data {
            hash ^= byte as u32;
            for _ in 0..8 {
                if hash & 1 == 1 {
                    hash = (hash >> 1) ^ 0xEDB88320;
                } else {
                    hash >>= 1;
                }
            }
        }
        hash as u64
    }

    /// MurmurHash3-inspired fast hash
    pub fn murmur_hash(data: &[u8]) -> u64 {
        let mut hash = 0x9e3779b97f4a7c15u64;
        let mut i = 0;

        while i + 8 <= data.len() {
            let k = u64::from_le_bytes([
                data[i],
                data[i + 1],
                data[i + 2],
                data[i + 3],
                data[i + 4],
                data[i + 5],
                data[i + 6],
                data[i + 7],
            ]);
            hash ^= k.wrapping_mul(0xc6a4a7935bd1e995);
            hash = hash.rotate_left(47);
            hash = hash.wrapping_mul(0x9e3779b97f4a7c15);
            i += 8;
        }

        // Handle remaining bytes
        for (offset, &byte) in data.iter().enumerate().skip(i) {
            hash ^= (byte as u64) << ((offset - i) * 8);
        }

        hash ^ (hash >> 32)
    }
}

/// Hash bucket containing key-value pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashBucket<K, V>
where
    K: Clone + Debug + Hash + Eq,
    V: Clone + Debug,
{
    /// Entries in this bucket
    entries: Vec<(K, V)>,
    /// Overflow bucket pointer
    overflow: Option<Box<HashBucket<K, V>>>,
    /// Bucket statistics
    access_count: u64,
    modification_count: u64,
}

impl<K, V> Default for HashBucket<K, V>
where
    K: Clone + Debug + Hash + Eq,
    V: Clone + Debug,
{
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            overflow: None,
            access_count: 0,
            modification_count: 0,
        }
    }
}

impl<K, V> HashBucket<K, V>
where
    K: Clone + Debug + Hash + Eq,
    V: Clone + Debug,
{
    /// Create a new empty bucket
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key-value pair into the bucket
    pub fn insert(&mut self, key: K, value: V, max_size: usize) -> Result<Option<V>> {
        self.modification_count += 1;

        // Check if key already exists
        for (existing_key, existing_value) in &mut self.entries {
            if *existing_key == key {
                let old_value = existing_value.clone();
                *existing_value = value;
                return Ok(Some(old_value));
            }
        }

        // If bucket has space, add here
        if self.entries.len() < max_size {
            self.entries.push((key, value));
            return Ok(None);
        }

        // Create or use overflow bucket
        if self.overflow.is_none() {
            self.overflow = Some(Box::new(HashBucket::new()));
        }

        self.overflow.as_mut().unwrap().insert(key, value, max_size)
    }

    /// Find a value by key
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.access_count += 1;

        // Search main bucket
        for (existing_key, value) in &self.entries {
            if existing_key == key {
                return Some(value);
            }
        }

        // Search overflow chain
        self.overflow.as_mut()?.get(key)
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.modification_count += 1;

        // Search main bucket
        for i in 0..self.entries.len() {
            if self.entries[i].0 == *key {
                return Some(self.entries.remove(i).1);
            }
        }

        // Search overflow chain
        self.overflow.as_mut()?.remove(key)
    }

    /// Get all entries (including overflow)
    pub fn all_entries(&self) -> Vec<(K, V)> {
        let mut result = self.entries.clone();
        if let Some(overflow) = &self.overflow {
            result.extend(overflow.all_entries());
        }
        result
    }

    /// Get bucket size (including overflow)
    pub fn len(&self) -> usize {
        let mut size = self.entries.len();
        if let Some(overflow) = &self.overflow {
            size += overflow.len();
        }
        size
    }

    /// Check if bucket is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
            && (self.overflow.is_none() || self.overflow.as_ref().unwrap().is_empty())
    }

    /// Get access statistics
    pub fn access_stats(&self) -> (u64, u64) {
        (self.access_count, self.modification_count)
    }
}

/// Hash index statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct HashIndexStats {
    pub total_entries: usize,
    pub bucket_count: usize,
    pub overflow_buckets: usize,
    pub load_factor: f64,
    pub avg_bucket_size: f64,
    pub max_bucket_size: usize,
    pub hash_collisions: u64,
    pub bucket_splits: u64,
    pub access_count: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Linear hash index implementation
pub struct HashIndex<K, V>
where
    K: Clone + Debug + Hash + Eq,
    V: Clone + Debug,
{
    /// Hash buckets array
    buckets: Arc<RwLock<Vec<HashBucket<K, V>>>>,
    /// Configuration
    config: HashIndexConfig,
    /// Current split point for linear hashing
    split_point: usize,
    /// Round number (number of complete splits)
    round: usize,
    /// Statistics
    stats: HashIndexStats,
    /// Next bucket to split
    next_split: usize,
}

impl<K, V> HashIndex<K, V>
where
    K: Clone + Debug + Hash + Eq,
    V: Clone + Debug,
{
    /// Create a new hash index
    pub fn new() -> Self {
        Self::with_config(HashIndexConfig::default())
    }

    /// Create a new hash index with custom configuration
    pub fn with_config(config: HashIndexConfig) -> Self {
        let mut buckets = Vec::new();
        buckets.resize_with(config.initial_buckets, HashBucket::default);

        Self {
            buckets: Arc::new(RwLock::new(buckets)),
            config,
            split_point: 0,
            round: 0,
            stats: HashIndexStats::default(),
            next_split: 0,
        }
    }

    /// Calculate hash for a key
    fn hash_key(&self, key: &K) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash_result = hasher.finish();

        // Apply configured hash function to the bytes
        let key_bytes = format!("{key:?}").as_bytes().to_vec();
        (self.config.hash_function)(&key_bytes) ^ hash_result
    }

    /// Calculate bucket index using linear hashing algorithm
    fn bucket_index(&self, hash: u64) -> usize {
        let _buckets = self.buckets.read().unwrap();
        let initial_size = self.config.initial_buckets * (1 << self.round);
        let index = (hash as usize) % initial_size;

        if index < self.split_point {
            // This bucket has been split, use doubled size
            (hash as usize) % (initial_size * 2)
        } else {
            index
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        let hash = self.hash_key(&key);
        let bucket_idx = self.bucket_index(hash);

        let result = {
            let mut buckets = self.buckets.write().unwrap();

            // Ensure bucket exists
            if bucket_idx >= buckets.len() {
                buckets.resize_with(bucket_idx + 1, HashBucket::default);
                self.stats.bucket_count = buckets.len();
            }

            buckets[bucket_idx].insert(key, value, self.config.max_bucket_size)
        };

        // Update statistics
        if result.is_ok() && result.as_ref().unwrap().is_none() {
            self.stats.total_entries += 1;
        }

        // Check if we need to split
        self.check_and_split()?;

        result
    }

    /// Get a value by key
    pub fn get(&mut self, key: &K) -> Option<V> {
        self.stats.access_count += 1;

        let hash = self.hash_key(key);
        let bucket_idx = self.bucket_index(hash);

        let mut buckets = self.buckets.write().unwrap();
        if bucket_idx < buckets.len() {
            if let Some(value) = buckets[bucket_idx].get(key) {
                self.stats.cache_hits += 1;
                return Some(value.clone());
            }
        }

        self.stats.cache_misses += 1;
        None
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = self.hash_key(key);
        let bucket_idx = self.bucket_index(hash);

        let result = {
            let mut buckets = self.buckets.write().unwrap();
            if bucket_idx < buckets.len() {
                buckets[bucket_idx].remove(key)
            } else {
                None
            }
        };

        if result.is_some() {
            self.stats.total_entries -= 1;
        }

        result
    }

    /// Check if split is needed and perform it
    fn check_and_split(&mut self) -> Result<()> {
        let current_load_factor = self.load_factor();

        if current_load_factor > self.config.max_load_factor {
            self.split_bucket()?;
        }

        Ok(())
    }

    /// Split a bucket using linear hashing
    fn split_bucket(&mut self) -> Result<()> {
        let mut buckets = self.buckets.write().unwrap();
        let old_bucket_count = buckets.len();

        // Resize to accommodate new bucket
        buckets.resize_with(old_bucket_count + 1, HashBucket::default);

        // Split the bucket at next_split index
        let split_idx = self.next_split;
        let new_idx = old_bucket_count;

        // Get all entries from the bucket being split
        let entries = buckets[split_idx].all_entries();

        // Clear the original bucket
        buckets[split_idx] = HashBucket::new();

        // Redistribute entries
        for (key, value) in entries {
            let hash = self.hash_key(&key);
            let new_bucket_idx = if hash as usize % (old_bucket_count * 2) < old_bucket_count {
                split_idx
            } else {
                new_idx
            };

            buckets[new_bucket_idx].insert(key, value, self.config.max_bucket_size)?;
        }

        // Update split tracking
        self.next_split += 1;
        self.split_point += 1;

        // Check if we've completed a round
        if self.next_split >= self.config.initial_buckets * (1 << self.round) {
            self.round += 1;
            self.next_split = 0;
            self.split_point = 0;
        }

        self.stats.bucket_splits += 1;
        self.stats.bucket_count = buckets.len();

        Ok(())
    }

    /// Calculate current load factor
    pub fn load_factor(&self) -> f64 {
        let buckets = self.buckets.read().unwrap();
        if buckets.is_empty() {
            return 0.0;
        }
        self.stats.total_entries as f64 / buckets.len() as f64
    }

    /// Get all key-value pairs
    pub fn all_entries(&self) -> Vec<(K, V)> {
        let buckets = self.buckets.read().unwrap();
        let mut result = Vec::new();

        for bucket in buckets.iter() {
            result.extend(bucket.all_entries());
        }

        result
    }

    /// Get index statistics
    pub fn get_stats(&mut self) -> HashIndexStats {
        let buckets = self.buckets.read().unwrap();

        let mut total_bucket_size = 0;
        let mut max_bucket_size = 0;
        let mut overflow_count = 0;

        for bucket in buckets.iter() {
            let size = bucket.len();
            total_bucket_size += size;
            max_bucket_size = max_bucket_size.max(size);

            if bucket.overflow.is_some() {
                overflow_count += 1;
            }
        }

        self.stats.bucket_count = buckets.len();
        self.stats.overflow_buckets = overflow_count;
        self.stats.load_factor = self.load_factor();
        self.stats.avg_bucket_size = if buckets.is_empty() {
            0.0
        } else {
            total_bucket_size as f64 / buckets.len() as f64
        };
        self.stats.max_bucket_size = max_bucket_size;

        self.stats.clone()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.stats.total_entries == 0
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.stats.total_entries
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        let mut buckets = self.buckets.write().unwrap();
        buckets.clear();
        buckets.resize_with(self.config.initial_buckets, HashBucket::default);

        self.stats = HashIndexStats::default();
        self.split_point = 0;
        self.round = 0;
        self.next_split = 0;
    }

    /// Compact the index (remove empty buckets and optimize)
    pub fn compact(&mut self) -> Result<()> {
        // This would implement compaction logic
        // For now, just update statistics
        let _ = self.get_stats();
        Ok(())
    }

    /// Validate index integrity
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();
        let buckets = self.buckets.read().unwrap();

        if buckets.is_empty() {
            issues.push("Hash index has no buckets".to_string());
        }

        // Validate each bucket
        for (i, bucket) in buckets.iter().enumerate() {
            if bucket.len() > self.config.max_bucket_size * 10 {
                issues.push(format!(
                    "Bucket {} is extremely oversized: {} entries",
                    i,
                    bucket.len()
                ));
            }
        }

        // Check load factor
        if self.load_factor() > 1.0 {
            issues.push(format!("Load factor too high: {:.2}", self.load_factor()));
        }

        Ok(issues)
    }
}

impl<K, V> Default for HashIndex<K, V>
where
    K: Clone + Debug + Hash + Eq,
    V: Clone + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::uninlined_format_args)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_index_basic_operations() {
        let mut index = HashIndex::new();

        // Test insertion
        assert!(index
            .insert("key1".to_string(), "value1".to_string())
            .is_ok());
        assert!(index
            .insert("key2".to_string(), "value2".to_string())
            .is_ok());
        assert_eq!(index.len(), 2);

        // Test retrieval
        assert_eq!(index.get(&"key1".to_string()), Some("value1".to_string()));
        assert_eq!(index.get(&"key2".to_string()), Some("value2".to_string()));
        assert_eq!(index.get(&"nonexistent".to_string()), None);

        // Test update
        assert_eq!(
            index
                .insert("key1".to_string(), "updated_value1".to_string())
                .unwrap(),
            Some("value1".to_string())
        );
        assert_eq!(
            index.get(&"key1".to_string()),
            Some("updated_value1".to_string())
        );

        // Test removal
        assert_eq!(
            index.remove(&"key1".to_string()),
            Some("updated_value1".to_string())
        );
        assert_eq!(index.get(&"key1".to_string()), None);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_linear_hashing_splits() {
        let config = HashIndexConfig {
            initial_buckets: 2,
            max_load_factor: 0.5,
            ..Default::default()
        };
        let mut index = HashIndex::with_config(config);

        // Insert enough items to trigger splits
        for i in 0..10 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            assert!(index.insert(key, value).is_ok());
        }

        let stats = index.get_stats();
        assert!(stats.bucket_splits > 0);
        assert!(stats.bucket_count > 2);
    }

    #[test]
    fn test_hash_functions() {
        let data = b"test data";

        let hash1 = HashIndexConfig::default_hash(data);
        let hash2 = HashIndexConfig::crc32_hash(data);
        let hash3 = HashIndexConfig::murmur_hash(data);

        // Hashes should be different (very unlikely to collide)
        assert_ne!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_ne!(hash2, hash3);

        // Same input should produce same hash
        assert_eq!(hash1, HashIndexConfig::default_hash(data));
        assert_eq!(hash2, HashIndexConfig::crc32_hash(data));
        assert_eq!(hash3, HashIndexConfig::murmur_hash(data));
    }

    #[test]
    fn test_overflow_buckets() {
        let config = HashIndexConfig {
            initial_buckets: 1, // Force everything into one bucket initially
            max_bucket_size: 3,
            max_load_factor: 10.0, // Don't split for this test
            ..Default::default()
        };
        let mut index = HashIndex::with_config(config);

        // Fill beyond bucket capacity to test overflow
        for i in 0..10 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            assert!(index.insert(key, value).is_ok());
        }

        // All items should still be retrievable
        for i in 0..10 {
            let key = format!("key{}", i);
            let expected = format!("value{}", i);
            assert_eq!(index.get(&key), Some(expected));
        }

        let stats = index.get_stats();
        assert!(stats.overflow_buckets > 0);
    }

    #[test]
    fn test_statistics() {
        let mut index = HashIndex::new();

        // Insert some data
        for i in 0..100 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            index.insert(key, value).unwrap();
        }

        let stats = index.get_stats();
        assert_eq!(stats.total_entries, 100);
        assert!(stats.load_factor > 0.0);
        assert!(stats.avg_bucket_size > 0.0);
    }

    #[test]
    fn test_validation() {
        let index = HashIndex::<String, String>::new();
        let issues = index.validate().unwrap();
        assert!(issues.is_empty()); // New index should be valid
    }
}
