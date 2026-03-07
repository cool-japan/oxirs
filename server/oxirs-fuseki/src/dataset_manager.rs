//! Dataset lifecycle management.
//!
//! Provides creation, deletion, backup, restore, state transitions, and
//! statistics for named RDF datasets registered with the Fuseki server.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Storage backend for a dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum StoreType {
    /// Pure in-memory store; data is lost on restart.
    InMemory,
    /// TDB2 on-disk triple store.
    TDB2,
    /// External SPARQL endpoint or custom backend (URI / path).
    External(String),
}

impl StoreType {
    /// Human-readable label used in serialisation / reporting.
    pub fn label(&self) -> &str {
        match self {
            StoreType::InMemory => "mem",
            StoreType::TDB2 => "tdb2",
            StoreType::External(_) => "external",
        }
    }
}

/// Lifecycle state of a registered dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum DatasetState {
    /// Accepting reads and writes.
    Active,
    /// Accepting reads only.
    ReadOnly,
    /// Not accessible (e.g. maintenance mode).
    Offline,
    /// Soft-deleted / archived; not served.
    Archived,
}

impl DatasetState {
    /// Returns `true` if reads are permitted.
    pub fn allows_read(&self) -> bool {
        matches!(self, DatasetState::Active | DatasetState::ReadOnly)
    }

    /// Returns `true` if writes are permitted.
    pub fn allows_write(&self) -> bool {
        matches!(self, DatasetState::Active)
    }
}

/// Configuration record stored per dataset.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Dataset name (must be non-empty, no whitespace).
    pub name: String,
    /// Storage backend type.
    pub store_type: StoreType,
    /// Unix timestamp (ms) when the dataset was created.
    pub created_at: u64,
    /// Whether the dataset was registered as read-only.
    pub read_only: bool,
}

impl DatasetConfig {
    /// Create a new `DatasetConfig`.
    pub fn new(name: impl Into<String>, store_type: StoreType, created_at: u64) -> Self {
        Self {
            name: name.into(),
            store_type,
            created_at,
            read_only: false,
        }
    }

    /// Mark the dataset as read-only.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }
}

/// Information returned after a successful backup.
#[derive(Debug, Clone)]
pub struct BackupInfo {
    /// Filesystem path where the backup was written.
    pub path: String,
    /// Name of the backed-up dataset.
    pub dataset: String,
    /// Unix timestamp (ms) when the backup was created.
    pub timestamp: u64,
    /// Estimated size of the backup in bytes.
    pub size_bytes: u64,
}

/// Aggregate statistics over all registered datasets.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetStats {
    pub total: usize,
    pub active: usize,
    pub read_only: usize,
    pub offline: usize,
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by `DatasetManager` operations.
#[derive(Debug, Clone, PartialEq)]
pub enum DatasetError {
    /// A dataset with that name already exists.
    AlreadyExists,
    /// No dataset with that name was found.
    NotFound,
    /// The operation requires write access but the dataset is read-only.
    ReadOnly,
    /// The name is empty or contains invalid characters.
    InvalidName,
    /// A backup or restore operation failed.
    BackupFailed(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::AlreadyExists => write!(f, "dataset already exists"),
            DatasetError::NotFound => write!(f, "dataset not found"),
            DatasetError::ReadOnly => write!(f, "dataset is read-only"),
            DatasetError::InvalidName => write!(f, "invalid dataset name"),
            DatasetError::BackupFailed(msg) => write!(f, "backup failed: {msg}"),
        }
    }
}

impl std::error::Error for DatasetError {}

// ---------------------------------------------------------------------------
// DatasetManager
// ---------------------------------------------------------------------------

/// Internal entry combining the dataset config and its current state.
#[derive(Debug, Clone)]
struct Entry {
    config: DatasetConfig,
    state: DatasetState,
}

/// In-memory registry of datasets with lifecycle operations.
#[derive(Debug, Default)]
pub struct DatasetManager {
    /// Datasets indexed by name.
    datasets: HashMap<String, Entry>,
    /// Monotonic clock used for timestamps (in tests we inject this).
    now_ms: u64,
}

impl DatasetManager {
    /// Create a new, empty `DatasetManager`.
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            now_ms: 0,
        }
    }

    /// Override the internal timestamp source (useful for deterministic tests).
    pub fn with_clock(mut self, now_ms: u64) -> Self {
        self.now_ms = now_ms;
        self
    }

    /// Advance the internal clock by `delta_ms` milliseconds.
    pub fn advance_clock(&mut self, delta_ms: u64) {
        self.now_ms += delta_ms;
    }

    // -----------------------------------------------------------------------
    // Validation helpers
    // -----------------------------------------------------------------------

    fn validate_name(name: &str) -> Result<(), DatasetError> {
        if name.is_empty() || name.chars().any(|c| c.is_whitespace()) {
            Err(DatasetError::InvalidName)
        } else {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // CRUD
    // -----------------------------------------------------------------------

    /// Register a new dataset.
    ///
    /// Returns `DatasetError::AlreadyExists` if a dataset with the same name
    /// is already registered.
    pub fn create(&mut self, config: DatasetConfig) -> Result<(), DatasetError> {
        Self::validate_name(&config.name)?;
        if self.datasets.contains_key(&config.name) {
            return Err(DatasetError::AlreadyExists);
        }
        let initial_state = if config.read_only {
            DatasetState::ReadOnly
        } else {
            DatasetState::Active
        };
        self.datasets.insert(
            config.name.clone(),
            Entry {
                config,
                state: initial_state,
            },
        );
        Ok(())
    }

    /// Remove a dataset from the registry.
    ///
    /// Returns `DatasetError::NotFound` if the dataset does not exist.
    pub fn delete(&mut self, name: &str) -> Result<(), DatasetError> {
        if self.datasets.remove(name).is_none() {
            Err(DatasetError::NotFound)
        } else {
            Ok(())
        }
    }

    /// Retrieve the config for a named dataset.
    pub fn get(&self, name: &str) -> Option<&DatasetConfig> {
        self.datasets.get(name).map(|e| &e.config)
    }

    /// List all registered datasets in insertion-order (sorted by name for
    /// determinism).
    pub fn list(&self) -> Vec<&DatasetConfig> {
        let mut configs: Vec<&DatasetConfig> = self.datasets.values().map(|e| &e.config).collect();
        configs.sort_by(|a, b| a.name.cmp(&b.name));
        configs
    }

    /// Return the current state of a dataset.
    pub fn state(&self, name: &str) -> Option<&DatasetState> {
        self.datasets.get(name).map(|e| &e.state)
    }

    /// Transition a dataset to a new `DatasetState`.
    pub fn set_state(&mut self, name: &str, state: DatasetState) -> Result<(), DatasetError> {
        let entry = self.datasets.get_mut(name).ok_or(DatasetError::NotFound)?;
        entry.state = state;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Backup / restore
    // -----------------------------------------------------------------------

    /// Simulate a backup of the named dataset to `path`.
    ///
    /// In this in-memory implementation no actual I/O is performed; the
    /// operation always succeeds unless the dataset is not found.
    pub fn backup(&self, name: &str, path: &str) -> Result<BackupInfo, DatasetError> {
        let entry = self.datasets.get(name).ok_or(DatasetError::NotFound)?;

        if path.is_empty() {
            return Err(DatasetError::BackupFailed("empty path".to_string()));
        }

        // Simulate different backup sizes depending on store type.
        let size_bytes: u64 = match &entry.config.store_type {
            StoreType::InMemory => 1024,
            StoreType::TDB2 => 65536,
            StoreType::External(_) => 4096,
        };

        Ok(BackupInfo {
            path: path.to_string(),
            dataset: name.to_string(),
            timestamp: self.now_ms,
            size_bytes,
        })
    }

    /// Simulate restoring a dataset from a backup at `path`.
    ///
    /// The dataset name is derived from the last path segment before any
    /// extension (e.g. `/backups/myds.bak` → `myds`).  The restored dataset
    /// is registered with `StoreType::TDB2` and `DatasetState::Active`.
    pub fn restore(&mut self, path: &str) -> Result<DatasetConfig, DatasetError> {
        if path.is_empty() {
            return Err(DatasetError::BackupFailed("empty path".to_string()));
        }

        // Derive dataset name from path stem.
        let file_name = path.split('/').next_back().unwrap_or(path);
        let stem = file_name.split('.').next().unwrap_or(file_name);

        if stem.is_empty() {
            return Err(DatasetError::BackupFailed(
                "cannot derive dataset name from path".to_string(),
            ));
        }

        let config = DatasetConfig::new(stem, StoreType::TDB2, self.now_ms);
        self.create(config.clone())?;
        Ok(config)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Aggregate statistics over all registered datasets.
    pub fn stats(&self) -> DatasetStats {
        let total = self.datasets.len();
        let active = self
            .datasets
            .values()
            .filter(|e| e.state == DatasetState::Active)
            .count();
        let read_only = self
            .datasets
            .values()
            .filter(|e| e.state == DatasetState::ReadOnly)
            .count();
        let offline = self
            .datasets
            .values()
            .filter(|e| e.state == DatasetState::Offline)
            .count();
        DatasetStats {
            total,
            active,
            read_only,
            offline,
        }
    }

    /// Return the number of registered datasets.
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    /// Return `true` if no datasets are registered.
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ts() -> u64 {
        1_700_000_000_000
    }

    fn mem_config(name: &str) -> DatasetConfig {
        DatasetConfig::new(name, StoreType::InMemory, ts())
    }

    fn tdb2_config(name: &str) -> DatasetConfig {
        DatasetConfig::new(name, StoreType::TDB2, ts())
    }

    fn manager() -> DatasetManager {
        DatasetManager::new().with_clock(ts())
    }

    // -----------------------------------------------------------------------
    // StoreType
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_type_label_mem() {
        assert_eq!(StoreType::InMemory.label(), "mem");
    }

    #[test]
    fn test_store_type_label_tdb2() {
        assert_eq!(StoreType::TDB2.label(), "tdb2");
    }

    #[test]
    fn test_store_type_label_external() {
        assert_eq!(
            StoreType::External("http://x".to_string()).label(),
            "external"
        );
    }

    // -----------------------------------------------------------------------
    // DatasetState
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_active_allows_read_and_write() {
        assert!(DatasetState::Active.allows_read());
        assert!(DatasetState::Active.allows_write());
    }

    #[test]
    fn test_state_readonly_allows_read_not_write() {
        assert!(DatasetState::ReadOnly.allows_read());
        assert!(!DatasetState::ReadOnly.allows_write());
    }

    #[test]
    fn test_state_offline_denies_both() {
        assert!(!DatasetState::Offline.allows_read());
        assert!(!DatasetState::Offline.allows_write());
    }

    #[test]
    fn test_state_archived_denies_both() {
        assert!(!DatasetState::Archived.allows_read());
        assert!(!DatasetState::Archived.allows_write());
    }

    // -----------------------------------------------------------------------
    // DatasetConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_new() {
        let cfg = DatasetConfig::new("ds1", StoreType::InMemory, 1000);
        assert_eq!(cfg.name, "ds1");
        assert_eq!(cfg.store_type, StoreType::InMemory);
        assert_eq!(cfg.created_at, 1000);
        assert!(!cfg.read_only);
    }

    #[test]
    fn test_config_with_read_only() {
        let cfg = DatasetConfig::new("ds1", StoreType::TDB2, 0).with_read_only(true);
        assert!(cfg.read_only);
    }

    // -----------------------------------------------------------------------
    // DatasetManager::create
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_succeeds() {
        let mut mgr = manager();
        assert!(mgr.create(mem_config("alpha")).is_ok());
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_create_duplicate_returns_error() {
        let mut mgr = manager();
        mgr.create(mem_config("alpha")).unwrap();
        let err = mgr.create(mem_config("alpha")).unwrap_err();
        assert_eq!(err, DatasetError::AlreadyExists);
    }

    #[test]
    fn test_create_empty_name_returns_invalid() {
        let mut mgr = manager();
        let err = mgr
            .create(DatasetConfig::new("", StoreType::InMemory, ts()))
            .unwrap_err();
        assert_eq!(err, DatasetError::InvalidName);
    }

    #[test]
    fn test_create_name_with_space_returns_invalid() {
        let mut mgr = manager();
        let err = mgr
            .create(DatasetConfig::new("bad name", StoreType::InMemory, ts()))
            .unwrap_err();
        assert_eq!(err, DatasetError::InvalidName);
    }

    #[test]
    fn test_create_readonly_config_sets_readonly_state() {
        let mut mgr = manager();
        let cfg = DatasetConfig::new("ro", StoreType::TDB2, ts()).with_read_only(true);
        mgr.create(cfg).unwrap();
        assert_eq!(mgr.state("ro"), Some(&DatasetState::ReadOnly));
    }

    // -----------------------------------------------------------------------
    // DatasetManager::delete
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_existing() {
        let mut mgr = manager();
        mgr.create(mem_config("ds")).unwrap();
        assert!(mgr.delete("ds").is_ok());
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_delete_nonexistent_returns_not_found() {
        let mut mgr = manager();
        assert_eq!(mgr.delete("missing").unwrap_err(), DatasetError::NotFound);
    }

    // -----------------------------------------------------------------------
    // DatasetManager::get
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_returns_config() {
        let mut mgr = manager();
        mgr.create(mem_config("ds")).unwrap();
        let cfg = mgr.get("ds").unwrap();
        assert_eq!(cfg.name, "ds");
    }

    #[test]
    fn test_get_missing_returns_none() {
        let mgr = manager();
        assert!(mgr.get("missing").is_none());
    }

    // -----------------------------------------------------------------------
    // DatasetManager::list
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_empty() {
        let mgr = manager();
        assert!(mgr.list().is_empty());
    }

    #[test]
    fn test_list_sorted_by_name() {
        let mut mgr = manager();
        mgr.create(mem_config("zebra")).unwrap();
        mgr.create(mem_config("alpha")).unwrap();
        mgr.create(mem_config("mango")).unwrap();
        let names: Vec<&str> = mgr.list().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mango", "zebra"]);
    }

    #[test]
    fn test_list_count() {
        let mut mgr = manager();
        for i in 0..5 {
            mgr.create(mem_config(&format!("ds{i}"))).unwrap();
        }
        assert_eq!(mgr.list().len(), 5);
    }

    // -----------------------------------------------------------------------
    // DatasetManager::set_state
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_state_active_to_offline() {
        let mut mgr = manager();
        mgr.create(mem_config("ds")).unwrap();
        mgr.set_state("ds", DatasetState::Offline).unwrap();
        assert_eq!(mgr.state("ds"), Some(&DatasetState::Offline));
    }

    #[test]
    fn test_set_state_not_found() {
        let mut mgr = manager();
        let err = mgr.set_state("ghost", DatasetState::Active).unwrap_err();
        assert_eq!(err, DatasetError::NotFound);
    }

    #[test]
    fn test_set_state_cycles() {
        let mut mgr = manager();
        mgr.create(mem_config("ds")).unwrap();
        mgr.set_state("ds", DatasetState::ReadOnly).unwrap();
        mgr.set_state("ds", DatasetState::Archived).unwrap();
        mgr.set_state("ds", DatasetState::Active).unwrap();
        assert_eq!(mgr.state("ds"), Some(&DatasetState::Active));
    }

    // -----------------------------------------------------------------------
    // DatasetManager::backup
    // -----------------------------------------------------------------------

    #[test]
    fn test_backup_success() {
        let mgr = {
            let mut m = manager();
            m.create(tdb2_config("prod")).unwrap();
            m
        };
        let info = mgr.backup("prod", "/tmp/prod.bak").unwrap();
        assert_eq!(info.dataset, "prod");
        assert_eq!(info.path, "/tmp/prod.bak");
        assert_eq!(info.size_bytes, 65536); // TDB2
    }

    #[test]
    fn test_backup_mem_size() {
        let mgr = {
            let mut m = manager();
            m.create(mem_config("mem_ds")).unwrap();
            m
        };
        let info = mgr.backup("mem_ds", "/tmp/mem.bak").unwrap();
        assert_eq!(info.size_bytes, 1024);
    }

    #[test]
    fn test_backup_not_found() {
        let mgr = manager();
        let err = mgr.backup("ghost", "/tmp/x.bak").unwrap_err();
        assert_eq!(err, DatasetError::NotFound);
    }

    #[test]
    fn test_backup_empty_path() {
        let mut mgr = manager();
        mgr.create(mem_config("ds")).unwrap();
        let err = mgr.backup("ds", "").unwrap_err();
        matches!(err, DatasetError::BackupFailed(_));
    }

    #[test]
    fn test_backup_timestamp() {
        let mgr = manager(); // clock = ts()
        let mut m2 = manager();
        m2.create(mem_config("ds")).unwrap();
        let info = m2.backup("ds", "/tmp/x").unwrap();
        assert_eq!(info.timestamp, ts());
    }

    // -----------------------------------------------------------------------
    // DatasetManager::restore
    // -----------------------------------------------------------------------

    #[test]
    fn test_restore_success() {
        let mut mgr = manager();
        let cfg = mgr.restore("/backups/myds.bak").unwrap();
        assert_eq!(cfg.name, "myds");
    }

    #[test]
    fn test_restore_registers_dataset() {
        let mut mgr = manager();
        mgr.restore("/backups/restored.bak").unwrap();
        assert!(mgr.get("restored").is_some());
    }

    #[test]
    fn test_restore_empty_path_fails() {
        let mut mgr = manager();
        let err = mgr.restore("").unwrap_err();
        matches!(err, DatasetError::BackupFailed(_));
    }

    #[test]
    fn test_restore_duplicate_fails() {
        let mut mgr = manager();
        mgr.restore("/backups/ds.bak").unwrap();
        let err = mgr.restore("/backups/ds.bak").unwrap_err();
        assert_eq!(err, DatasetError::AlreadyExists);
    }

    #[test]
    fn test_restore_no_extension() {
        let mut mgr = manager();
        let cfg = mgr.restore("/backups/mydata").unwrap();
        assert_eq!(cfg.name, "mydata");
    }

    // -----------------------------------------------------------------------
    // DatasetManager::stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let mgr = manager();
        let s = mgr.stats();
        assert_eq!(
            s,
            DatasetStats {
                total: 0,
                active: 0,
                read_only: 0,
                offline: 0
            }
        );
    }

    #[test]
    fn test_stats_all_active() {
        let mut mgr = manager();
        mgr.create(mem_config("a")).unwrap();
        mgr.create(mem_config("b")).unwrap();
        let s = mgr.stats();
        assert_eq!(s.total, 2);
        assert_eq!(s.active, 2);
    }

    #[test]
    fn test_stats_mixed_states() {
        let mut mgr = manager();
        mgr.create(mem_config("a")).unwrap();
        mgr.create(mem_config("b")).unwrap();
        mgr.create(mem_config("c")).unwrap();
        mgr.set_state("b", DatasetState::ReadOnly).unwrap();
        mgr.set_state("c", DatasetState::Offline).unwrap();
        let s = mgr.stats();
        assert_eq!(s.total, 3);
        assert_eq!(s.active, 1);
        assert_eq!(s.read_only, 1);
        assert_eq!(s.offline, 1);
    }

    #[test]
    fn test_stats_archived_not_counted_in_named_states() {
        let mut mgr = manager();
        mgr.create(mem_config("a")).unwrap();
        mgr.set_state("a", DatasetState::Archived).unwrap();
        let s = mgr.stats();
        assert_eq!(s.total, 1);
        assert_eq!(s.active, 0);
        assert_eq!(s.read_only, 0);
        assert_eq!(s.offline, 0);
    }

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_empty() {
        let mgr = manager();
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_is_not_empty_after_create() {
        let mut mgr = manager();
        mgr.create(mem_config("ds")).unwrap();
        assert!(!mgr.is_empty());
    }

    #[test]
    fn test_external_store_backup_size() {
        let mut mgr = manager();
        let cfg = DatasetConfig::new(
            "ext",
            StoreType::External("http://remote".to_string()),
            ts(),
        );
        mgr.create(cfg).unwrap();
        let info = mgr.backup("ext", "/tmp/ext.bak").unwrap();
        assert_eq!(info.size_bytes, 4096);
    }

    #[test]
    fn test_dataset_error_display() {
        assert_eq!(
            DatasetError::AlreadyExists.to_string(),
            "dataset already exists"
        );
        assert_eq!(DatasetError::NotFound.to_string(), "dataset not found");
        assert_eq!(DatasetError::ReadOnly.to_string(), "dataset is read-only");
        assert_eq!(
            DatasetError::InvalidName.to_string(),
            "invalid dataset name"
        );
        assert!(DatasetError::BackupFailed("oops".to_string())
            .to_string()
            .contains("oops"));
    }

    #[test]
    fn test_advance_clock() {
        let mut mgr = manager();
        mgr.create(mem_config("ds")).unwrap();
        mgr.advance_clock(1000);
        let info = mgr.backup("ds", "/tmp/x").unwrap();
        assert_eq!(info.timestamp, ts() + 1000);
    }

    #[test]
    fn test_name_with_tab_is_invalid() {
        let mut mgr = manager();
        let err = mgr
            .create(DatasetConfig::new("bad\tname", StoreType::InMemory, ts()))
            .unwrap_err();
        assert_eq!(err, DatasetError::InvalidName);
    }

    #[test]
    fn test_many_datasets() {
        let mut mgr = manager();
        for i in 0..20 {
            mgr.create(mem_config(&format!("dataset_{i:03}"))).unwrap();
        }
        assert_eq!(mgr.len(), 20);
        let stats = mgr.stats();
        assert_eq!(stats.total, 20);
        assert_eq!(stats.active, 20);
    }
}
