//! Cache storage implementations

use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;

use super::types::{CacheKey, CacheSizeInfo, CacheValue, StorageStatistics};

/// Cache storage trait for different storage implementations
pub trait CacheStorage: Send + Sync + std::fmt::Debug {
    /// Store an item in the cache
    fn store(&mut self, key: CacheKey, value: CacheValue, ttl: Option<Duration>) -> Result<()>;

    /// Retrieve an item from the cache
    fn retrieve(&self, key: &CacheKey) -> Option<CacheValue>;

    /// Remove an item from the cache
    fn remove(&mut self, key: &CacheKey) -> bool;

    /// Get cache size information
    fn size_info(&self) -> CacheSizeInfo;

    /// Clear the entire cache
    fn clear(&mut self);

    /// Get storage-specific statistics
    fn statistics(&self) -> StorageStatistics;
}

/// In-memory cache storage
#[derive(Debug)]
pub struct MemoryStorage {
    data: HashMap<CacheKey, CacheValue>,
    max_size: u64,
    current_size: u64,
}

impl MemoryStorage {
    pub fn new(max_size: u64) -> Self {
        Self {
            data: HashMap::new(),
            max_size,
            current_size: 0,
        }
    }
}

impl CacheStorage for MemoryStorage {
    fn store(&mut self, key: CacheKey, value: CacheValue, _ttl: Option<Duration>) -> Result<()> {
        let size = value.metadata.size_bytes;
        if self.current_size + size <= self.max_size {
            self.data.insert(key, value);
            self.current_size += size;
        }
        Ok(())
    }

    fn retrieve(&self, key: &CacheKey) -> Option<CacheValue> {
        self.data.get(key).cloned()
    }

    fn remove(&mut self, key: &CacheKey) -> bool {
        if let Some(value) = self.data.remove(key) {
            self.current_size -= value.metadata.size_bytes;
            true
        } else {
            false
        }
    }

    fn size_info(&self) -> CacheSizeInfo {
        CacheSizeInfo {
            used_bytes: self.current_size,
            available_bytes: self.max_size - self.current_size,
            total_capacity_bytes: self.max_size,
            item_count: self.data.len() as u64,
        }
    }

    fn clear(&mut self) {
        self.data.clear();
        self.current_size = 0;
    }

    fn statistics(&self) -> StorageStatistics {
        StorageStatistics::default()
    }
}

/// Compressed cache storage (wraps MemoryStorage with compression)
#[derive(Debug)]
pub struct CompressedStorage {
    inner: MemoryStorage,
}

impl CompressedStorage {
    pub fn new(max_size: u64) -> Self {
        Self {
            inner: MemoryStorage::new(max_size),
        }
    }
}

impl CacheStorage for CompressedStorage {
    fn store(&mut self, key: CacheKey, value: CacheValue, ttl: Option<Duration>) -> Result<()> {
        // In a real implementation, this would compress the value
        self.inner.store(key, value, ttl)
    }

    fn retrieve(&self, key: &CacheKey) -> Option<CacheValue> {
        // In a real implementation, this would decompress the value
        self.inner.retrieve(key)
    }

    fn remove(&mut self, key: &CacheKey) -> bool {
        self.inner.remove(key)
    }

    fn size_info(&self) -> CacheSizeInfo {
        self.inner.size_info()
    }

    fn clear(&mut self) {
        self.inner.clear()
    }

    fn statistics(&self) -> StorageStatistics {
        self.inner.statistics()
    }
}

/// Persistent cache storage (disk-backed)
#[derive(Debug)]
pub struct PersistentStorage {
    inner: MemoryStorage,
}

impl PersistentStorage {
    pub fn new(max_size: u64) -> Result<Self> {
        Ok(Self {
            inner: MemoryStorage::new(max_size),
        })
    }
}

impl CacheStorage for PersistentStorage {
    fn store(&mut self, key: CacheKey, value: CacheValue, ttl: Option<Duration>) -> Result<()> {
        // In a real implementation, this would persist to disk
        self.inner.store(key, value, ttl)
    }

    fn retrieve(&self, key: &CacheKey) -> Option<CacheValue> {
        // In a real implementation, this would load from disk if not in memory
        self.inner.retrieve(key)
    }

    fn remove(&mut self, key: &CacheKey) -> bool {
        self.inner.remove(key)
    }

    fn size_info(&self) -> CacheSizeInfo {
        self.inner.size_info()
    }

    fn clear(&mut self) {
        self.inner.clear()
    }

    fn statistics(&self) -> StorageStatistics {
        self.inner.statistics()
    }
}
