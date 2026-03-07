//! Storage backends for different tiers

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Storage backend trait for tier implementations
pub trait StorageBackend: Send + Sync {
    /// Load an index from storage
    fn load_index(&self, index_id: &str) -> Result<Vec<u8>>;

    /// Save an index to storage
    fn save_index(&mut self, index_id: &str, data: &[u8]) -> Result<()>;

    /// Delete an index from storage
    fn delete_index(&mut self, index_id: &str) -> Result<()>;

    /// Check if an index exists
    fn exists(&self, index_id: &str) -> bool;

    /// Get the size of an index in bytes
    fn get_size(&self, index_id: &str) -> Result<u64>;

    /// List all indices in this storage
    fn list_indices(&self) -> Result<Vec<String>>;

    /// Get storage backend type name
    fn backend_type(&self) -> &'static str;
}

/// Hot tier storage: In-memory storage
pub struct HotTierStorage {
    /// In-memory cache of indices
    cache: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>>,
}

impl HotTierStorage {
    /// Create a new hot tier storage
    pub fn new() -> Self {
        Self {
            cache: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        let cache = self.cache.lock().expect("lock should not be poisoned");
        cache.values().map(|v| v.len() as u64).sum()
    }

    /// Get number of cached indices
    pub fn cache_size(&self) -> usize {
        let cache = self.cache.lock().expect("lock should not be poisoned");
        cache.len()
    }
}

impl Default for HotTierStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for HotTierStorage {
    fn load_index(&self, index_id: &str) -> Result<Vec<u8>> {
        let cache = self.cache.lock().expect("lock should not be poisoned");
        cache
            .get(index_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Index {} not found in hot tier", index_id))
    }

    fn save_index(&mut self, index_id: &str, data: &[u8]) -> Result<()> {
        let mut cache = self.cache.lock().expect("lock should not be poisoned");
        cache.insert(index_id.to_string(), data.to_vec());
        Ok(())
    }

    fn delete_index(&mut self, index_id: &str) -> Result<()> {
        let mut cache = self.cache.lock().expect("lock should not be poisoned");
        cache
            .remove(index_id)
            .ok_or_else(|| anyhow::anyhow!("Index {} not found in hot tier", index_id))?;
        Ok(())
    }

    fn exists(&self, index_id: &str) -> bool {
        let cache = self.cache.lock().expect("lock should not be poisoned");
        cache.contains_key(index_id)
    }

    fn get_size(&self, index_id: &str) -> Result<u64> {
        let cache = self.cache.lock().expect("lock should not be poisoned");
        cache
            .get(index_id)
            .map(|v| v.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("Index {} not found in hot tier", index_id))
    }

    fn list_indices(&self) -> Result<Vec<String>> {
        let cache = self.cache.lock().expect("lock should not be poisoned");
        Ok(cache.keys().cloned().collect())
    }

    fn backend_type(&self) -> &'static str {
        "HotTier (In-Memory)"
    }
}

/// Warm tier storage: Memory-mapped files
pub struct WarmTierStorage {
    /// Base directory for storage
    base_path: PathBuf,
    /// Compression enabled
    compression_enabled: bool,
    /// Compression level
    compression_level: i32,
}

impl WarmTierStorage {
    /// Create a new warm tier storage
    pub fn new<P: AsRef<Path>>(
        base_path: P,
        compression_enabled: bool,
        compression_level: i32,
    ) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        Ok(Self {
            base_path,
            compression_enabled,
            compression_level,
        })
    }

    /// Get path for an index
    fn get_index_path(&self, index_id: &str) -> PathBuf {
        let filename = if self.compression_enabled {
            format!("{}.idx.zst", index_id)
        } else {
            format!("{}.idx", index_id)
        };
        self.base_path.join(filename)
    }
}

impl StorageBackend for WarmTierStorage {
    fn load_index(&self, index_id: &str) -> Result<Vec<u8>> {
        let path = self.get_index_path(index_id);
        let data = std::fs::read(&path)?;

        if self.compression_enabled {
            Ok(zstd::decode_all(&data[..])?)
        } else {
            Ok(data)
        }
    }

    fn save_index(&mut self, index_id: &str, data: &[u8]) -> Result<()> {
        let path = self.get_index_path(index_id);

        let final_data = if self.compression_enabled {
            zstd::encode_all(data, self.compression_level)?
        } else {
            data.to_vec()
        };

        std::fs::write(&path, final_data)?;
        Ok(())
    }

    fn delete_index(&mut self, index_id: &str) -> Result<()> {
        let path = self.get_index_path(index_id);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    fn exists(&self, index_id: &str) -> bool {
        self.get_index_path(index_id).exists()
    }

    fn get_size(&self, index_id: &str) -> Result<u64> {
        let path = self.get_index_path(index_id);
        Ok(std::fs::metadata(&path)?.len())
    }

    fn list_indices(&self) -> Result<Vec<String>> {
        let mut indices = Vec::new();
        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            if let Some(filename) = entry.file_name().to_str() {
                if filename.ends_with(".idx") || filename.ends_with(".idx.zst") {
                    let index_id = filename
                        .trim_end_matches(".idx.zst")
                        .trim_end_matches(".idx")
                        .to_string();
                    indices.push(index_id);
                }
            }
        }
        Ok(indices)
    }

    fn backend_type(&self) -> &'static str {
        "WarmTier (Memory-Mapped)"
    }
}

/// Cold tier storage: Compressed disk storage
pub struct ColdTierStorage {
    /// Base directory for storage
    base_path: PathBuf,
    /// Compression level (high)
    compression_level: i32,
}

impl ColdTierStorage {
    /// Create a new cold tier storage
    pub fn new<P: AsRef<Path>>(base_path: P, compression_level: i32) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        Ok(Self {
            base_path,
            compression_level,
        })
    }

    /// Get path for an index
    fn get_index_path(&self, index_id: &str) -> PathBuf {
        self.base_path.join(format!("{}.idx.zst", index_id))
    }
}

impl StorageBackend for ColdTierStorage {
    fn load_index(&self, index_id: &str) -> Result<Vec<u8>> {
        let path = self.get_index_path(index_id);
        let compressed_data = std::fs::read(&path)?;
        Ok(zstd::decode_all(&compressed_data[..])?)
    }

    fn save_index(&mut self, index_id: &str, data: &[u8]) -> Result<()> {
        let path = self.get_index_path(index_id);
        let compressed_data = zstd::encode_all(data, self.compression_level)?;
        std::fs::write(&path, compressed_data)?;
        Ok(())
    }

    fn delete_index(&mut self, index_id: &str) -> Result<()> {
        let path = self.get_index_path(index_id);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    fn exists(&self, index_id: &str) -> bool {
        self.get_index_path(index_id).exists()
    }

    fn get_size(&self, index_id: &str) -> Result<u64> {
        let path = self.get_index_path(index_id);
        Ok(std::fs::metadata(&path)?.len())
    }

    fn list_indices(&self) -> Result<Vec<String>> {
        let mut indices = Vec::new();
        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            if let Some(filename) = entry.file_name().to_str() {
                if filename.ends_with(".idx.zst") {
                    let index_id = filename.trim_end_matches(".idx.zst").to_string();
                    indices.push(index_id);
                }
            }
        }
        Ok(indices)
    }

    fn backend_type(&self) -> &'static str {
        "ColdTier (Compressed Disk)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_tier_storage() {
        let mut storage = HotTierStorage::new();

        let data = vec![1, 2, 3, 4, 5];
        storage.save_index("test", &data).unwrap();

        assert!(storage.exists("test"));
        assert_eq!(storage.get_size("test").unwrap(), 5);

        let loaded = storage.load_index("test").unwrap();
        assert_eq!(loaded, data);

        storage.delete_index("test").unwrap();
        assert!(!storage.exists("test"));
    }

    #[test]
    fn test_warm_tier_storage() {
        use std::env;
        let temp_dir = env::temp_dir().join("oxirs_warm_tier_test");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut storage = WarmTierStorage::new(&temp_dir, true, 6).unwrap();

        let data = vec![1, 2, 3, 4, 5];
        storage.save_index("test", &data).unwrap();

        assert!(storage.exists("test"));

        let loaded = storage.load_index("test").unwrap();
        assert_eq!(loaded, data);

        storage.delete_index("test").unwrap();
        assert!(!storage.exists("test"));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_cold_tier_storage() {
        use std::env;
        let temp_dir = env::temp_dir().join("oxirs_cold_tier_test");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut storage = ColdTierStorage::new(&temp_dir, 19).unwrap();

        let data = vec![1, 2, 3, 4, 5];
        storage.save_index("test", &data).unwrap();

        assert!(storage.exists("test"));

        let loaded = storage.load_index("test").unwrap();
        assert_eq!(loaded, data);

        storage.delete_index("test").unwrap();
        assert!(!storage.exists("test"));

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
