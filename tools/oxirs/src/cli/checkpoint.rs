//! Checkpoint system for resumable import/export operations
//!
//! Provides checkpoint functionality to enable resuming interrupted
//! import and export operations, particularly useful for large RDF datasets.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Checkpoint data for resumable operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Operation type (import or export)
    pub operation: String,

    /// Dataset name
    pub dataset: String,

    /// Source/target file path
    pub file_path: String,

    /// Number of triples/quads processed
    pub processed_count: usize,

    /// Last processed line/offset in file
    pub last_offset: u64,

    /// Timestamp when checkpoint was created
    pub timestamp: String,

    /// Format being used
    pub format: String,

    /// Optional graph URI for imports
    pub graph: Option<String>,

    /// File size for progress tracking
    pub total_size: u64,
}

/// Checkpoint manager for handling resume operations
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let checkpoint_dir = Self::get_checkpoint_dir()?;
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self { checkpoint_dir })
    }

    /// Get the checkpoint directory path
    fn get_checkpoint_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let config_dir = dirs::config_dir()
            .ok_or("Could not determine config directory")?
            .join("oxirs")
            .join("checkpoints");

        Ok(config_dir)
    }

    /// Generate checkpoint filename from operation details
    fn checkpoint_filename(&self, operation: &str, dataset: &str, file_path: &str) -> PathBuf {
        // Create a unique but deterministic filename
        let hash = format!(
            "{:x}",
            md5::compute(format!("{}{}{}", operation, dataset, file_path))
        );
        self.checkpoint_dir.join(format!("{}.checkpoint", hash))
    }

    /// Save a checkpoint
    pub fn save(&self, checkpoint: &Checkpoint) -> Result<(), Box<dyn std::error::Error>> {
        let checkpoint_path = self.checkpoint_filename(
            &checkpoint.operation,
            &checkpoint.dataset,
            &checkpoint.file_path,
        );

        let json = serde_json::to_string_pretty(checkpoint)?;
        fs::write(checkpoint_path, json)?;

        Ok(())
    }

    /// Load a checkpoint if it exists
    pub fn load(
        &self,
        operation: &str,
        dataset: &str,
        file_path: &str,
    ) -> Result<Option<Checkpoint>, Box<dyn std::error::Error>> {
        let checkpoint_path = self.checkpoint_filename(operation, dataset, file_path);

        if !checkpoint_path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&checkpoint_path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&content)?;

        Ok(Some(checkpoint))
    }

    /// Delete a checkpoint
    pub fn delete(
        &self,
        operation: &str,
        dataset: &str,
        file_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let checkpoint_path = self.checkpoint_filename(operation, dataset, file_path);

        if checkpoint_path.exists() {
            fs::remove_file(checkpoint_path)?;
        }

        Ok(())
    }

    /// List all checkpoints
    pub fn list_all(&self) -> Result<Vec<Checkpoint>, Box<dyn std::error::Error>> {
        let mut checkpoints = Vec::new();

        if !self.checkpoint_dir.exists() {
            return Ok(checkpoints);
        }

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("checkpoint") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(checkpoint) = serde_json::from_str::<Checkpoint>(&content) {
                        checkpoints.push(checkpoint);
                    }
                }
            }
        }

        Ok(checkpoints)
    }

    /// Check if a checkpoint exists
    pub fn exists(&self, operation: &str, dataset: &str, file_path: &str) -> bool {
        let checkpoint_path = self.checkpoint_filename(operation, dataset, file_path);
        checkpoint_path.exists()
    }

    /// Get checkpoint progress percentage
    pub fn progress_percentage(&self, checkpoint: &Checkpoint) -> f64 {
        if checkpoint.total_size == 0 {
            return 0.0;
        }

        (checkpoint.last_offset as f64 / checkpoint.total_size as f64) * 100.0
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new().expect("Failed to create checkpoint manager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_manager() -> (CheckpointManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new().unwrap();
        manager.checkpoint_dir = temp_dir.path().to_path_buf();
        fs::create_dir_all(&manager.checkpoint_dir).unwrap();
        (manager, temp_dir)
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        let (manager, _temp_dir) = create_test_manager();

        let checkpoint = Checkpoint {
            operation: "import".to_string(),
            dataset: "testdb".to_string(),
            file_path: "/tmp/test.ttl".to_string(),
            processed_count: 1000,
            last_offset: 5000,
            timestamp: chrono::Local::now().to_rfc3339(),
            format: "turtle".to_string(),
            graph: None,
            total_size: 10000,
        };

        // Save checkpoint
        manager.save(&checkpoint).unwrap();

        // Load checkpoint
        let loaded = manager.load("import", "testdb", "/tmp/test.ttl").unwrap();
        assert!(loaded.is_some());

        let loaded = loaded.unwrap();
        assert_eq!(loaded.processed_count, 1000);
        assert_eq!(loaded.last_offset, 5000);
    }

    #[test]
    fn test_checkpoint_exists() {
        let (manager, _temp_dir) = create_test_manager();

        assert!(!manager.exists("import", "testdb", "/tmp/test.ttl"));

        let checkpoint = Checkpoint {
            operation: "import".to_string(),
            dataset: "testdb".to_string(),
            file_path: "/tmp/test.ttl".to_string(),
            processed_count: 1000,
            last_offset: 5000,
            timestamp: chrono::Local::now().to_rfc3339(),
            format: "turtle".to_string(),
            graph: None,
            total_size: 10000,
        };

        manager.save(&checkpoint).unwrap();
        assert!(manager.exists("import", "testdb", "/tmp/test.ttl"));
    }

    #[test]
    fn test_delete_checkpoint() {
        let (manager, _temp_dir) = create_test_manager();

        let checkpoint = Checkpoint {
            operation: "import".to_string(),
            dataset: "testdb".to_string(),
            file_path: "/tmp/test.ttl".to_string(),
            processed_count: 1000,
            last_offset: 5000,
            timestamp: chrono::Local::now().to_rfc3339(),
            format: "turtle".to_string(),
            graph: None,
            total_size: 10000,
        };

        manager.save(&checkpoint).unwrap();
        assert!(manager.exists("import", "testdb", "/tmp/test.ttl"));

        manager.delete("import", "testdb", "/tmp/test.ttl").unwrap();
        assert!(!manager.exists("import", "testdb", "/tmp/test.ttl"));
    }

    #[test]
    fn test_progress_percentage() {
        let manager = CheckpointManager::new().unwrap();

        let checkpoint = Checkpoint {
            operation: "import".to_string(),
            dataset: "testdb".to_string(),
            file_path: "/tmp/test.ttl".to_string(),
            processed_count: 1000,
            last_offset: 5000,
            timestamp: chrono::Local::now().to_rfc3339(),
            format: "turtle".to_string(),
            graph: None,
            total_size: 10000,
        };

        let progress = manager.progress_percentage(&checkpoint);
        assert_eq!(progress, 50.0);
    }
}
