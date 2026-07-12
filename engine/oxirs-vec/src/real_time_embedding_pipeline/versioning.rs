//! Version management for the real-time embedding pipeline

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::real_time_embedding_pipeline::{
    config::VersionControlConfig,
    traits::{HealthStatus, Version, VersionStorage},
};

/// Configuration for version manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionManagerConfig {
    pub max_versions_per_id: usize,
    pub cleanup_interval: Duration,
    pub enable_compression: bool,
    pub max_total_versions: usize,
}

impl Default for VersionManagerConfig {
    fn default() -> Self {
        Self {
            max_versions_per_id: 10,
            cleanup_interval: Duration::from_secs(300),
            enable_compression: false,
            max_total_versions: 100_000,
        }
    }
}

/// Statistics for version management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionManagerStatistics {
    pub total_versions: u64,
    pub total_cleanups: u64,
    pub total_ids_tracked: usize,
    pub is_running: bool,
}

/// In-memory version storage
pub struct InMemoryVersionStorage {
    data: HashMap<String, Vec<Version>>,
    max_versions_per_id: usize,
}

impl InMemoryVersionStorage {
    pub fn new(max_versions_per_id: usize) -> Self {
        Self {
            data: HashMap::new(),
            max_versions_per_id,
        }
    }
}

impl VersionStorage for InMemoryVersionStorage {
    fn store_version(&mut self, id: &str, version: &Version) -> Result<()> {
        let versions = self.data.entry(id.to_string()).or_default();
        versions.push(version.clone());
        if versions.len() > self.max_versions_per_id {
            let excess = versions.len() - self.max_versions_per_id;
            versions.drain(0..excess);
        }
        Ok(())
    }

    fn get_version(&self, id: &str, version_number: u64) -> Result<Option<Version>> {
        Ok(self
            .data
            .get(id)
            .and_then(|vs| vs.iter().find(|v| v.version == version_number).cloned()))
    }

    fn get_all_versions(&self, id: &str) -> Result<Vec<Version>> {
        Ok(self.data.get(id).cloned().unwrap_or_default())
    }

    fn cleanup_old_versions(&mut self, id: &str, keep_count: usize) -> Result<usize> {
        if let Some(versions) = self.data.get_mut(id) {
            let original_len = versions.len();
            if original_len > keep_count {
                let to_remove = original_len - keep_count;
                versions.drain(0..to_remove);
                return Ok(to_remove);
            }
        }
        Ok(0)
    }
}

/// Version manager for tracking embedding versions
pub struct VersionManager {
    config: VersionManagerConfig,
    storage: Arc<RwLock<InMemoryVersionStorage>>,
    is_running: AtomicBool,
    total_versions: AtomicU64,
    total_cleanups: AtomicU64,
}

impl VersionManager {
    pub fn new(version_control_config: VersionControlConfig) -> Result<Self> {
        let config = VersionManagerConfig {
            max_versions_per_id: version_control_config.max_versions,
            enable_compression: false,
            ..Default::default()
        };
        let storage = Arc::new(RwLock::new(InMemoryVersionStorage::new(
            config.max_versions_per_id,
        )));
        Ok(Self {
            config,
            storage,
            is_running: AtomicBool::new(false),
            total_versions: AtomicU64::new(0),
            total_cleanups: AtomicU64::new(0),
        })
    }

    pub async fn start(&self) -> Result<()> {
        self.is_running.store(true, Ordering::Release);
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        self.is_running.store(false, Ordering::Release);
        Ok(())
    }

    pub async fn health_check(&self) -> Result<HealthStatus> {
        if self.is_running.load(Ordering::Acquire) {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Warning {
                message: "Version manager not running".to_string(),
            })
        }
    }

    pub fn track_version(&self, id: &str, version: Version) -> Result<()> {
        let mut storage = self
            .storage
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire version storage lock"))?;
        storage.store_version(id, &version)?;
        self.total_versions.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn get_version(&self, id: &str, version_number: u64) -> Result<Option<Version>> {
        let storage = self
            .storage
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire version storage lock"))?;
        storage.get_version(id, version_number)
    }

    pub fn cleanup_old(&self, id: &str) -> Result<usize> {
        let mut storage = self
            .storage
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire version storage lock"))?;
        let removed = storage.cleanup_old_versions(id, self.config.max_versions_per_id)?;
        if removed > 0 {
            self.total_cleanups.fetch_add(1, Ordering::Relaxed);
        }
        Ok(removed)
    }

    pub fn get_statistics(&self) -> VersionManagerStatistics {
        let total_ids = self.storage.read().map(|s| s.data.len()).unwrap_or(0);
        VersionManagerStatistics {
            total_versions: self.total_versions.load(Ordering::Acquire),
            total_cleanups: self.total_cleanups.load(Ordering::Acquire),
            total_ids_tracked: total_ids,
            is_running: self.is_running.load(Ordering::Acquire),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real_time_embedding_pipeline::config::VersionControlConfig;
    use crate::real_time_embedding_pipeline::traits::Version;
    use crate::Vector;
    use std::time::SystemTime;

    fn make_version(num: u64) -> Version {
        Version {
            version: num,
            vector: Vector::new(vec![1.0, 2.0, 3.0]),
            created_at: SystemTime::now(),
            metadata: HashMap::new(),
            checksum: format!("checksum_{}", num),
        }
    }

    #[tokio::test]
    async fn test_version_manager_start_stop() {
        let config = VersionControlConfig::default();
        let manager = VersionManager::new(config).expect("should create");
        manager.start().await.expect("should start");
        let health = manager.health_check().await.expect("should check");
        assert!(matches!(health, HealthStatus::Healthy));
        manager.stop().await.expect("should stop");
    }

    #[test]
    fn test_track_and_get_version() {
        let config = VersionControlConfig::default();
        let manager = VersionManager::new(config).expect("should create");
        let v = make_version(1);
        manager.track_version("item1", v).expect("should track");
        let result = manager.get_version("item1", 1).expect("should get");
        assert!(result.is_some());
        assert_eq!(result.expect("should have version").version, 1);
    }

    #[test]
    fn test_cleanup_old_versions() {
        let config = VersionControlConfig {
            max_versions: 3,
            ..Default::default()
        };
        let manager = VersionManager::new(config).expect("should create");
        for i in 1..=5 {
            manager
                .track_version("item1", make_version(i))
                .expect("should track");
        }
        let _removed = manager.cleanup_old("item1").expect("should cleanup");
        let stats = manager.get_statistics();
        assert!(stats.total_versions > 0);
    }

    #[test]
    fn test_in_memory_version_storage() {
        let mut storage = InMemoryVersionStorage::new(3);
        for i in 1..=5u64 {
            let v = Version {
                version: i,
                vector: Vector::new(vec![i as f32]),
                created_at: SystemTime::now(),
                metadata: HashMap::new(),
                checksum: format!("c{}", i),
            };
            storage.store_version("key", &v).expect("should store");
        }
        let all = storage.get_all_versions("key").expect("should get all");
        assert!(all.len() <= 3);
    }
}
