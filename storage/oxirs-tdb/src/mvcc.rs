//! # MVCC (Multi-Version Concurrency Control)
//!
//! MVCC implementation for TDB storage to support concurrent access.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Version identifier
pub type Version = u64;

/// Versioned value container
#[derive(Debug, Clone)]
pub struct VersionedValue<T> {
    pub value: T,
    pub version: Version,
    pub timestamp: std::time::SystemTime,
    pub is_deleted: bool,
}

/// MVCC storage for key-value pairs
pub struct MvccStorage<K, V> 
where 
    K: Clone + Eq + std::hash::Hash,
    V: Clone,
{
    data: Arc<RwLock<HashMap<K, Vec<VersionedValue<V>>>>>,
    current_version: Arc<RwLock<Version>>,
}

impl<K, V> MvccStorage<K, V>
where
    K: Clone + Eq + std::hash::Hash,
    V: Clone,
{
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            current_version: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Insert or update a value
    pub fn put(&self, key: K, value: V) -> Result<Version> {
        let mut current_version = self.current_version.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire version lock"))?;
        *current_version += 1;
        let version = *current_version;
        
        let mut data = self.data.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire data lock"))?;
        
        let versioned_value = VersionedValue {
            value,
            version,
            timestamp: std::time::SystemTime::now(),
            is_deleted: false,
        };
        
        data.entry(key)
            .or_insert_with(Vec::new)
            .push(versioned_value);
        
        Ok(version)
    }
    
    /// Get the latest non-deleted value for a key
    pub fn get(&self, key: &K) -> Result<Option<V>> {
        let data = self.data.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire data lock"))?;
        
        if let Some(versions) = data.get(key) {
            // Find the latest non-deleted version
            for versioned_value in versions.iter().rev() {
                if !versioned_value.is_deleted {
                    return Ok(Some(versioned_value.value.clone()));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Get a value as of a specific version
    pub fn get_at_version(&self, key: &K, target_version: Version) -> Result<Option<V>> {
        let data = self.data.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire data lock"))?;
        
        if let Some(versions) = data.get(key) {
            // Find the latest version <= target_version that is not deleted
            for versioned_value in versions.iter().rev() {
                if versioned_value.version <= target_version && !versioned_value.is_deleted {
                    return Ok(Some(versioned_value.value.clone()));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Delete a value (mark as deleted)
    pub fn delete(&self, key: K) -> Result<Version> {
        let mut current_version = self.current_version.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire version lock"))?;
        *current_version += 1;
        let version = *current_version;
        
        let mut data = self.data.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire data lock"))?;
        
        let versioned_value = VersionedValue {
            value: Default::default(), // We don't care about the value for deletions
            version,
            timestamp: std::time::SystemTime::now(),
            is_deleted: true,
        };
        
        data.entry(key)
            .or_insert_with(Vec::new)
            .push(versioned_value);
        
        Ok(version)
    }
    
    /// Get current version
    pub fn current_version(&self) -> Result<Version> {
        let version = self.current_version.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire version lock"))?;
        Ok(*version)
    }
    
    /// Cleanup old versions (garbage collection)
    pub fn cleanup_old_versions(&self, keep_versions: usize) -> Result<()> {
        let mut data = self.data.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire data lock"))?;
        
        for versions in data.values_mut() {
            if versions.len() > keep_versions {
                versions.drain(0..versions.len() - keep_versions);
            }
        }
        
        Ok(())
    }
}

impl<K, V> Default for MvccStorage<K, V>
where
    K: Clone + Eq + std::hash::Hash,
    V: Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Clone for MvccStorage<K, V>
where
    K: Clone + Eq + std::hash::Hash,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            current_version: Arc::clone(&self.current_version),
        }
    }
}