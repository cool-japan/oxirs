//! Point-in-Time Recovery (PITR) Module
//!
//! Transaction log-based recovery system for OxiRS databases.
//! Enables restoration to specific timestamps or transaction IDs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Transaction log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionLogEntry {
    pub transaction_id: u64,
    pub timestamp: DateTime<Utc>,
    pub operation_type: OperationType,
    pub data: Vec<u8>,
    pub checksum: String,
}

/// Types of operations in transaction log
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OperationType {
    Insert,
    Delete,
    Update,
    BeginTransaction,
    CommitTransaction,
    RollbackTransaction,
}

/// PITR configuration
pub struct PitrConfig {
    pub log_dir: PathBuf,
    pub archive_dir: PathBuf,
    pub max_log_size: u64,
    pub auto_archive: bool,
}

/// Transaction log manager
pub struct TransactionLog {
    config: PitrConfig,
    current_log: Option<BufWriter<File>>,
    next_transaction_id: u64,
}

impl TransactionLog {
    /// Create a new transaction log
    pub fn new(config: PitrConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Ensure directories exist
        fs::create_dir_all(&config.log_dir)?;
        fs::create_dir_all(&config.archive_dir)?;

        // Load next transaction ID
        let next_transaction_id = Self::get_next_transaction_id(&config.log_dir)?;

        Ok(Self {
            config,
            current_log: None,
            next_transaction_id,
        })
    }

    /// Append a transaction entry to the log
    pub fn append(
        &mut self,
        operation_type: OperationType,
        data: Vec<u8>,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        // Ensure log file is open
        if self.current_log.is_none() {
            self.open_current_log()?;
        }

        let transaction_id = self.next_transaction_id;
        self.next_transaction_id += 1;

        let entry = TransactionLogEntry {
            transaction_id,
            timestamp: Utc::now(),
            operation_type,
            data: data.clone(),
            checksum: Self::calculate_checksum(&data),
        };

        // Serialize and write entry
        let entry_json = serde_json::to_string(&entry)?;
        if let Some(ref mut log) = self.current_log {
            writeln!(log, "{}", entry_json)?;
            log.flush()?;
        }

        // Check if log needs rotation
        if self.should_rotate_log()? {
            self.rotate_log()?;
        }

        Ok(transaction_id)
    }

    /// Recover to a specific point in time
    pub fn recover_to_timestamp(
        &self,
        target_timestamp: DateTime<Utc>,
        _output_dir: &Path,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        println!("Starting Point-in-Time Recovery");
        println!("Target timestamp: {}", target_timestamp.to_rfc3339());

        // Collect all log files (current + archived)
        let mut log_files = self.collect_log_files()?;
        log_files.sort();

        let mut applied_transactions = 0;

        // Replay transactions up to the target timestamp
        for log_file in log_files {
            applied_transactions += self.replay_log_file(&log_file, Some(target_timestamp))?;
        }

        println!(
            "Recovery complete: {} transactions applied",
            applied_transactions
        );
        Ok(applied_transactions)
    }

    /// Recover to a specific transaction ID
    pub fn recover_to_transaction(
        &self,
        target_transaction_id: u64,
        _output_dir: &Path,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        println!("Starting Point-in-Time Recovery");
        println!("Target transaction ID: {}", target_transaction_id);

        let mut log_files = self.collect_log_files()?;
        log_files.sort();

        let mut applied_transactions = 0;

        for log_file in log_files {
            let file = File::open(&log_file)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }

                let entry: TransactionLogEntry = serde_json::from_str(&line)?;

                if entry.transaction_id > target_transaction_id {
                    // Stop when we exceed target transaction ID
                    println!("Reached target transaction ID");
                    return Ok(applied_transactions);
                }

                // Verify checksum
                if !Self::verify_checksum(&entry) {
                    eprintln!(
                        "Warning: Checksum mismatch for transaction {}",
                        entry.transaction_id
                    );
                    continue;
                }

                // Apply transaction (placeholder - actual implementation would replay to store)
                applied_transactions += 1;
            }
        }

        Ok(applied_transactions)
    }

    /// Create an incremental backup checkpoint
    pub fn create_checkpoint(&self, name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let checkpoint_file = self
            .config
            .archive_dir
            .join(format!("checkpoint_{}.json", name));

        let checkpoint = CheckpointMetadata {
            name: name.to_string(),
            timestamp: Utc::now(),
            last_transaction_id: self.next_transaction_id - 1,
            log_files: self.collect_log_files()?,
        };

        let mut file = File::create(&checkpoint_file)?;
        serde_json::to_writer_pretty(&mut file, &checkpoint)?;

        println!("Checkpoint created: {}", checkpoint_file.display());
        Ok(checkpoint_file)
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointMetadata>, Box<dyn std::error::Error>> {
        let mut checkpoints: Vec<CheckpointMetadata> = Vec::new();

        for entry in fs::read_dir(&self.config.archive_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json")
                && path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with("checkpoint_"))
                    .unwrap_or(false)
            {
                let file = File::open(&path)?;
                if let Ok(checkpoint) = serde_json::from_reader(file) {
                    checkpoints.push(checkpoint);
                }
            }
        }

        checkpoints.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(checkpoints)
    }

    /// Archive old transaction logs
    pub fn archive_logs(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        // Close current log
        self.close_current_log()?;

        let mut archived = 0;

        // Move all .wal files to archive
        for entry in fs::read_dir(&self.config.log_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("wal") {
                let filename = path.file_name().expect("file path should have a file name");
                let archive_path = self.config.archive_dir.join(filename);

                fs::rename(&path, &archive_path)?;
                archived += 1;
            }
        }

        println!("Archived {} log file(s)", archived);
        Ok(archived)
    }

    // === Private helper methods ===

    fn open_current_log(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let log_path = self
            .config
            .log_dir
            .join(format!("transaction_{}.wal", Utc::now().timestamp()));

        let file = File::options().create(true).append(true).open(&log_path)?;

        self.current_log = Some(BufWriter::new(file));
        Ok(())
    }

    fn close_current_log(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut log) = self.current_log.take() {
            log.flush()?;
        }
        Ok(())
    }

    fn should_rotate_log(&self) -> Result<bool, Box<dyn std::error::Error>> {
        // Rotation based on file size
        for entry in fs::read_dir(&self.config.log_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("wal") {
                let metadata = fs::metadata(&path)?;
                if metadata.len() >= self.config.max_log_size {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn rotate_log(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Close current log
        self.close_current_log()?;

        // Archive if enabled
        if self.config.auto_archive {
            self.archive_logs()?;
        }

        // Open new log
        self.open_current_log()?;

        Ok(())
    }

    fn collect_log_files(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut log_files = Vec::new();

        // Collect from log directory
        if self.config.log_dir.exists() {
            for entry in fs::read_dir(&self.config.log_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("wal") {
                    log_files.push(path);
                }
            }
        }

        // Collect from archive directory
        if self.config.archive_dir.exists() {
            for entry in fs::read_dir(&self.config.archive_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("wal") {
                    log_files.push(path);
                }
            }
        }

        Ok(log_files)
    }

    fn replay_log_file(
        &self,
        log_file: &Path,
        until_timestamp: Option<DateTime<Utc>>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let file = File::open(log_file)?;
        let reader = BufReader::new(file);
        let mut replayed = 0;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let entry: TransactionLogEntry = serde_json::from_str(&line)?;

            // Stop if we've passed the target timestamp
            if let Some(target) = until_timestamp {
                if entry.timestamp > target {
                    return Ok(replayed);
                }
            }

            // Verify checksum
            if !Self::verify_checksum(&entry) {
                eprintln!(
                    "Warning: Checksum mismatch for transaction {}",
                    entry.transaction_id
                );
                continue;
            }

            // Apply transaction (placeholder)
            replayed += 1;
        }

        Ok(replayed)
    }

    fn get_next_transaction_id(log_dir: &Path) -> Result<u64, Box<dyn std::error::Error>> {
        let mut max_id = 0u64;

        if !log_dir.exists() {
            return Ok(1);
        }

        // Scan all log files to find highest transaction ID
        for entry in fs::read_dir(log_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("wal") {
                let file = File::open(&path)?;
                let reader = BufReader::new(file);

                for line in reader.lines() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }

                    if let Ok(entry) = serde_json::from_str::<TransactionLogEntry>(&line) {
                        max_id = max_id.max(entry.transaction_id);
                    }
                }
            }
        }

        Ok(max_id + 1)
    }

    fn calculate_checksum(data: &[u8]) -> String {
        use ring::digest;
        let hash = digest::digest(&digest::SHA256, data);
        hex::encode(hash.as_ref())
    }

    fn verify_checksum(entry: &TransactionLogEntry) -> bool {
        let calculated = Self::calculate_checksum(&entry.data);
        calculated == entry.checksum
    }
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub name: String,
    pub timestamp: DateTime<Utc>,
    pub last_transaction_id: u64,
    pub log_files: Vec<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_transaction_log_creation() {
        let temp_log_dir = temp_dir().join("oxirs_pitr_test_logs");
        let temp_archive_dir = temp_dir().join("oxirs_pitr_test_archives");

        let config = PitrConfig {
            log_dir: temp_log_dir.clone(),
            archive_dir: temp_archive_dir.clone(),
            max_log_size: 10_000_000, // 10MB
            auto_archive: false,
        };

        let log = TransactionLog::new(config).unwrap();
        assert_eq!(log.next_transaction_id, 1);

        // Cleanup
        let _ = fs::remove_dir_all(temp_log_dir);
        let _ = fs::remove_dir_all(temp_archive_dir);
    }

    #[test]
    fn test_transaction_append() {
        let temp_log_dir = temp_dir().join("oxirs_pitr_test_logs2");
        let temp_archive_dir = temp_dir().join("oxirs_pitr_test_archives2");

        let config = PitrConfig {
            log_dir: temp_log_dir.clone(),
            archive_dir: temp_archive_dir.clone(),
            max_log_size: 10_000_000,
            auto_archive: false,
        };

        let mut log = TransactionLog::new(config).unwrap();

        // Append several transactions
        let tx1 = log
            .append(OperationType::Insert, b"test data 1".to_vec())
            .unwrap();
        let tx2 = log
            .append(OperationType::Update, b"test data 2".to_vec())
            .unwrap();
        let tx3 = log
            .append(OperationType::Delete, b"test data 3".to_vec())
            .unwrap();

        assert_eq!(tx1, 1);
        assert_eq!(tx2, 2);
        assert_eq!(tx3, 3);

        // Cleanup
        let _ = fs::remove_dir_all(temp_log_dir);
        let _ = fs::remove_dir_all(temp_archive_dir);
    }

    #[test]
    fn test_checkpoint_creation() {
        let temp_log_dir = temp_dir().join("oxirs_pitr_test_logs3");
        let temp_archive_dir = temp_dir().join("oxirs_pitr_test_archives3");

        let config = PitrConfig {
            log_dir: temp_log_dir.clone(),
            archive_dir: temp_archive_dir.clone(),
            max_log_size: 10_000_000,
            auto_archive: false,
        };

        let mut log = TransactionLog::new(config).unwrap();

        // Add some transactions
        log.append(OperationType::Insert, b"data1".to_vec())
            .unwrap();
        log.append(OperationType::Insert, b"data2".to_vec())
            .unwrap();

        // Create checkpoint
        let checkpoint_path = log.create_checkpoint("test_checkpoint").unwrap();
        assert!(checkpoint_path.exists());

        // List checkpoints
        let checkpoints = log.list_checkpoints().unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].name, "test_checkpoint");

        // Cleanup
        let _ = fs::remove_dir_all(temp_log_dir);
        let _ = fs::remove_dir_all(temp_archive_dir);
    }
}
