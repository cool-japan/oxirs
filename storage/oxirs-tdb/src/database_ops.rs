//! Database operations and management utilities
//!
//! Provides comprehensive database management operations inspired by Apache Jena's DatabaseOps.
//! Includes database lifecycle management, maintenance operations, and administrative tasks.

use crate::error::{Result, TdbError};
use crate::store::{StoreParams, TdbStore};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Database metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetadata {
    /// Database name
    pub name: String,
    /// Database location
    pub location: PathBuf,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modified timestamp
    pub modified_at: SystemTime,
    /// Database version
    pub version: String,
    /// Database size in bytes
    pub size_bytes: u64,
    /// Number of triples
    pub triple_count: u64,
    /// Database status
    pub status: DatabaseStatus,
}

/// Database status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseStatus {
    /// Database is active and available
    Active,
    /// Database is being created
    Creating,
    /// Database is being compacted
    Compacting,
    /// Database is being backed up
    BackingUp,
    /// Database is being repaired
    Repairing,
    /// Database is offline
    Offline,
    /// Database has errors
    Error,
}

/// Database operations manager
pub struct DatabaseOps {
    /// Base directory for all databases
    base_dir: PathBuf,
}

impl DatabaseOps {
    /// Create new database operations manager
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir)?;
        }

        Ok(Self { base_dir })
    }

    /// Create a new database with the given parameters
    pub fn create_database(&self, name: &str, params: StoreParams) -> Result<DatabaseMetadata> {
        let db_path = self.base_dir.join(name);

        // Check if database already exists
        if db_path.exists() {
            return Err(TdbError::InvalidInput(format!(
                "Database '{}' already exists",
                name
            )));
        }

        // Validate database name
        self.validate_database_name(name)?;

        // Create database directory
        std::fs::create_dir_all(&db_path)?;

        // Save store parameters
        let params_file = db_path.join("store_params.json");
        params.save_to_file(&params_file)?;

        // Initialize database
        let _store = TdbStore::open(&db_path)?;

        // Create metadata
        let metadata = DatabaseMetadata {
            name: name.to_string(),
            location: db_path.clone(),
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            version: crate::VERSION.to_string(),
            size_bytes: self.calculate_database_size(&db_path)?,
            triple_count: 0,
            status: DatabaseStatus::Active,
        };

        // Save metadata
        self.save_metadata(&metadata)?;

        log::info!("Created database '{}' at {:?}", name, db_path);

        Ok(metadata)
    }

    /// Delete a database
    pub fn delete_database(&self, name: &str) -> Result<()> {
        let db_path = self.base_dir.join(name);

        if !db_path.exists() {
            return Err(TdbError::InvalidInput(format!(
                "Database '{}' does not exist",
                name
            )));
        }

        // Delete database directory and all contents
        std::fs::remove_dir_all(&db_path)?;

        log::info!("Deleted database '{}'", name);

        Ok(())
    }

    /// List all databases
    pub fn list_databases(&self) -> Result<Vec<DatabaseMetadata>> {
        let mut databases = Vec::new();

        if !self.base_dir.exists() {
            return Ok(databases);
        }

        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Try to load metadata
                if let Ok(metadata) = self.load_metadata(&path) {
                    databases.push(metadata);
                }
            }
        }

        Ok(databases)
    }

    /// Get metadata for a specific database
    pub fn get_metadata(&self, name: &str) -> Result<DatabaseMetadata> {
        let db_path = self.base_dir.join(name);

        if !db_path.exists() {
            return Err(TdbError::InvalidInput(format!(
                "Database '{}' does not exist",
                name
            )));
        }

        self.load_metadata(&db_path)
    }

    /// Compact a database
    pub fn compact_database(&self, name: &str) -> Result<CompactionStats> {
        let db_path = self.base_dir.join(name);

        if !db_path.exists() {
            return Err(TdbError::InvalidInput(format!(
                "Database '{}' does not exist",
                name
            )));
        }

        // Update status
        let mut metadata = self.load_metadata(&db_path)?;
        metadata.status = DatabaseStatus::Compacting;
        self.save_metadata(&metadata)?;

        let start_time = SystemTime::now();
        let size_before = self.calculate_database_size(&db_path)?;

        // Open database and compact
        let mut store = TdbStore::open(&db_path)?;
        store.compact()?;

        let size_after = self.calculate_database_size(&db_path)?;
        let duration = start_time.elapsed().unwrap_or(Duration::from_secs(0));

        // Update metadata
        metadata.status = DatabaseStatus::Active;
        metadata.modified_at = SystemTime::now();
        metadata.size_bytes = size_after;
        self.save_metadata(&metadata)?;

        let stats = CompactionStats {
            size_before,
            size_after,
            space_saved: size_before.saturating_sub(size_after),
            duration_secs: duration.as_secs_f64(),
            compression_ratio: if size_before > 0 {
                size_after as f64 / size_before as f64
            } else {
                1.0
            },
        };

        log::info!(
            "Compacted database '{}': saved {} bytes ({:.1}% reduction)",
            name,
            stats.space_saved,
            (1.0 - stats.compression_ratio) * 100.0
        );

        Ok(stats)
    }

    /// Repair a database (check and fix corruption)
    pub fn repair_database(&self, name: &str) -> Result<RepairReport> {
        let db_path = self.base_dir.join(name);

        if !db_path.exists() {
            return Err(TdbError::InvalidInput(format!(
                "Database '{}' does not exist",
                name
            )));
        }

        // Update status
        let mut metadata = self.load_metadata(&db_path)?;
        metadata.status = DatabaseStatus::Repairing;
        self.save_metadata(&metadata)?;

        let start_time = SystemTime::now();

        // Open database and run diagnostics
        let store = TdbStore::open(&db_path)?;
        let diagnostic_report = store.run_diagnostics(crate::diagnostics::DiagnosticLevel::Deep);

        // Count issues from diagnostic report
        let issues_found =
            diagnostic_report.summary.error_count + diagnostic_report.summary.critical_count;
        let issues_fixed = 0; // Currently no automatic repair implemented

        let duration = start_time.elapsed().unwrap_or(Duration::from_secs(0));

        // Update metadata
        metadata.status = if issues_found == issues_fixed {
            DatabaseStatus::Active
        } else {
            DatabaseStatus::Error
        };
        metadata.modified_at = SystemTime::now();
        self.save_metadata(&metadata)?;

        let report = RepairReport {
            issues_found,
            issues_fixed,
            duration_secs: duration.as_secs_f64(),
            success: issues_found == issues_fixed,
        };

        log::info!(
            "Repaired database '{}': {} issues found, {} fixed",
            name,
            report.issues_found,
            report.issues_fixed
        );

        Ok(report)
    }

    /// Copy a database
    pub fn copy_database(&self, source: &str, destination: &str) -> Result<()> {
        let source_path = self.base_dir.join(source);
        let dest_path = self.base_dir.join(destination);

        if !source_path.exists() {
            return Err(TdbError::InvalidInput(format!(
                "Source database '{}' does not exist",
                source
            )));
        }

        if dest_path.exists() {
            return Err(TdbError::InvalidInput(format!(
                "Destination database '{}' already exists",
                destination
            )));
        }

        // Validate destination name
        self.validate_database_name(destination)?;

        // Copy directory recursively
        self.copy_dir_recursive(&source_path, &dest_path)?;

        // Update metadata for destination
        if let Ok(mut metadata) = self.load_metadata(&dest_path) {
            metadata.name = destination.to_string();
            metadata.location = dest_path.clone();
            metadata.created_at = SystemTime::now();
            self.save_metadata(&metadata)?;
        }

        log::info!("Copied database '{}' to '{}'", source, destination);

        Ok(())
    }

    /// Get database size in bytes
    pub fn get_database_size(&self, name: &str) -> Result<u64> {
        let db_path = self.base_dir.join(name);

        if !db_path.exists() {
            return Err(TdbError::InvalidInput(format!(
                "Database '{}' does not exist",
                name
            )));
        }

        self.calculate_database_size(&db_path)
    }

    /// Validate database name
    fn validate_database_name(&self, name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(TdbError::InvalidInput(
                "Database name cannot be empty".to_string(),
            ));
        }

        // Check for invalid characters
        if name.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) {
            return Err(TdbError::InvalidInput(format!(
                "Database name '{}' contains invalid characters",
                name
            )));
        }

        Ok(())
    }

    /// Calculate total size of database directory
    #[allow(clippy::only_used_in_recursion)]
    fn calculate_database_size(&self, path: &Path) -> Result<u64> {
        let mut total_size = 0u64;

        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();

                if entry_path.is_dir() {
                    total_size += self.calculate_database_size(&entry_path)?;
                } else if entry_path.is_file() {
                    total_size += entry.metadata()?.len();
                }
            }
        }

        Ok(total_size)
    }

    /// Save database metadata
    fn save_metadata(&self, metadata: &DatabaseMetadata) -> Result<()> {
        let metadata_file = metadata.location.join("metadata.json");
        let json = serde_json::to_string_pretty(metadata)
            .map_err(|e| TdbError::Serialization(format!("Failed to serialize metadata: {}", e)))?;
        std::fs::write(metadata_file, json)?;
        Ok(())
    }

    /// Load database metadata
    fn load_metadata(&self, db_path: &Path) -> Result<DatabaseMetadata> {
        let metadata_file = db_path.join("metadata.json");

        if !metadata_file.exists() {
            // Create default metadata if file doesn't exist
            let metadata = DatabaseMetadata {
                name: db_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                location: db_path.to_path_buf(),
                created_at: SystemTime::now(),
                modified_at: SystemTime::now(),
                version: crate::VERSION.to_string(),
                size_bytes: self.calculate_database_size(db_path)?,
                triple_count: 0,
                status: DatabaseStatus::Active,
            };
            self.save_metadata(&metadata)?;
            return Ok(metadata);
        }

        let json = std::fs::read_to_string(metadata_file)?;
        let metadata: DatabaseMetadata = serde_json::from_str(&json)
            .map_err(|e| TdbError::Deserialization(format!("Failed to parse metadata: {}", e)))?;
        Ok(metadata)
    }

    /// Copy directory recursively
    #[allow(clippy::only_used_in_recursion)]
    fn copy_dir_recursive(&self, src: &Path, dst: &Path) -> Result<()> {
        std::fs::create_dir_all(dst)?;

        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());

            if src_path.is_dir() {
                self.copy_dir_recursive(&src_path, &dst_path)?;
            } else {
                std::fs::copy(&src_path, &dst_path)?;
            }
        }

        Ok(())
    }
}

/// Statistics from database compaction
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Database size before compaction
    pub size_before: u64,
    /// Database size after compaction
    pub size_after: u64,
    /// Space saved in bytes
    pub space_saved: u64,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Compression ratio (after/before)
    pub compression_ratio: f64,
}

impl CompactionStats {
    /// Get space savings percentage
    pub fn savings_percentage(&self) -> f64 {
        if self.size_before > 0 {
            (self.space_saved as f64 / self.size_before as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Report from database repair operation
#[derive(Debug, Clone)]
pub struct RepairReport {
    /// Number of issues found
    pub issues_found: usize,
    /// Number of issues fixed
    pub issues_fixed: usize,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Whether repair was successful
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StorePresets;
    use std::env;

    fn create_test_base_dir() -> PathBuf {
        env::temp_dir().join(format!("oxirs_dbops_test_{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn test_create_database() {
        let base_dir = create_test_base_dir();
        let ops = DatabaseOps::new(&base_dir).unwrap();

        let params = StorePresets::minimal(base_dir.join("test_db"))
            .build()
            .unwrap();

        let metadata = ops.create_database("test_db", params).unwrap();

        assert_eq!(metadata.name, "test_db");
        assert_eq!(metadata.status, DatabaseStatus::Active);
    }

    #[test]
    fn test_list_databases() {
        let base_dir = create_test_base_dir();
        let ops = DatabaseOps::new(&base_dir).unwrap();

        let params1 = StorePresets::minimal(base_dir.join("db1")).build().unwrap();
        let params2 = StorePresets::minimal(base_dir.join("db2")).build().unwrap();

        ops.create_database("db1", params1).unwrap();
        ops.create_database("db2", params2).unwrap();

        let databases = ops.list_databases().unwrap();
        assert_eq!(databases.len(), 2);
    }

    #[test]
    fn test_delete_database() {
        let base_dir = create_test_base_dir();
        let ops = DatabaseOps::new(&base_dir).unwrap();

        let params = StorePresets::minimal(base_dir.join("test_db"))
            .build()
            .unwrap();

        ops.create_database("test_db", params).unwrap();
        assert!(ops.get_metadata("test_db").is_ok());

        ops.delete_database("test_db").unwrap();
        assert!(ops.get_metadata("test_db").is_err());
    }

    #[test]
    fn test_get_database_size() {
        let base_dir = create_test_base_dir();
        let ops = DatabaseOps::new(&base_dir).unwrap();

        let params = StorePresets::minimal(base_dir.join("test_db"))
            .build()
            .unwrap();

        ops.create_database("test_db", params).unwrap();

        let size = ops.get_database_size("test_db").unwrap();
        assert!(size > 0);
    }

    #[test]
    fn test_copy_database() {
        let base_dir = create_test_base_dir();
        let ops = DatabaseOps::new(&base_dir).unwrap();

        let params = StorePresets::minimal(base_dir.join("source_db"))
            .build()
            .unwrap();

        ops.create_database("source_db", params).unwrap();
        ops.copy_database("source_db", "dest_db").unwrap();

        assert!(ops.get_metadata("source_db").is_ok());
        assert!(ops.get_metadata("dest_db").is_ok());
    }

    #[test]
    fn test_validate_database_name() {
        let base_dir = create_test_base_dir();
        let ops = DatabaseOps::new(&base_dir).unwrap();

        assert!(ops.validate_database_name("valid_name").is_ok());
        assert!(ops.validate_database_name("").is_err());
        assert!(ops.validate_database_name("invalid/name").is_err());
        assert!(ops.validate_database_name("invalid:name").is_err());
    }

    #[test]
    fn test_compaction_stats() {
        let stats = CompactionStats {
            size_before: 1000,
            size_after: 600,
            space_saved: 400,
            duration_secs: 1.5,
            compression_ratio: 0.6,
        };

        assert_eq!(stats.savings_percentage(), 40.0);
    }

    #[test]
    fn test_repair_report() {
        let report = RepairReport {
            issues_found: 5,
            issues_fixed: 5,
            duration_secs: 2.0,
            success: true,
        };

        assert!(report.success);
        assert_eq!(report.issues_found, 5);
    }
}
