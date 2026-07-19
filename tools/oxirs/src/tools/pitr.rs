//! Point-in-Time Recovery (PITR) Module
//!
//! Transaction log-based recovery system for OxiRS databases.
//! Enables restoration to specific timestamps or transaction IDs.

use chrono::{DateTime, Utc};
use oxirs_core::model::Quad;
use oxirs_core::rdf_store::RdfStore;
use oxirs_ttl::formats::nquads::{NQuadsParser, NQuadsSerializer};
use oxirs_ttl::toolkit::{Parser as TtlParser, Serializer as TtlSerializer};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Cursor, Write};
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

    /// Append a transaction entry whose payload is a set of RDF quads,
    /// serialized as N-Quads text so `recover_to_timestamp`/
    /// `recover_to_transaction` can replay them into a real target store.
    /// This is the canonical encoding `data` must use for
    /// `OperationType::Insert`/`Update`/`Delete` entries to be replayable.
    pub fn append_quads(
        &mut self,
        operation_type: OperationType,
        quads: &[Quad],
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let mut buf = Vec::new();
        NQuadsSerializer::new()
            .serialize(quads, &mut buf)
            .map_err(|e| format!("Failed to serialize transaction quads: {e}"))?;
        self.append(operation_type, buf)
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

    /// Recover to a specific point in time.
    ///
    /// Opens (creating if necessary) a real [`RdfStore`] at `output_dir` and
    /// replays every logged transaction up to `target_timestamp` into it:
    /// `Insert`/`Update` entries insert their quads, `Delete` entries remove
    /// theirs, and `BeginTransaction`/`CommitTransaction`/`RollbackTransaction`
    /// markers are no-ops. A transaction whose payload cannot be decoded as
    /// N-Quads text (i.e. was not written via [`Self::append_quads`]) fails
    /// the whole recovery loudly rather than being silently counted as
    /// "applied" — see [`Self::parse_entry_quads`].
    pub fn recover_to_timestamp(
        &self,
        target_timestamp: DateTime<Utc>,
        output_dir: &Path,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        println!("Starting Point-in-Time Recovery");
        println!("Target timestamp: {}", target_timestamp.to_rfc3339());

        // Collect all log files (current + archived)
        let mut log_files = self.collect_log_files()?;
        log_files.sort();

        let mut store = RdfStore::open(output_dir).map_err(|e| {
            format!(
                "Failed to open recovery target '{}': {e}",
                output_dir.display()
            )
        })?;

        let mut applied_transactions = 0;

        // Replay transactions up to the target timestamp
        for log_file in log_files {
            applied_transactions +=
                self.replay_log_file(&log_file, Some(target_timestamp), &mut store)?;
        }

        store
            .flush()
            .map_err(|e| format!("Failed to flush recovered dataset: {e}"))?;

        println!(
            "Recovery complete: {} transactions applied to {}",
            applied_transactions,
            output_dir.display()
        );
        Ok(applied_transactions)
    }

    /// Recover to a specific transaction ID.
    ///
    /// Same real-replay semantics as [`Self::recover_to_timestamp`], but
    /// stops once a transaction ID beyond `target_transaction_id` is seen.
    pub fn recover_to_transaction(
        &self,
        target_transaction_id: u64,
        output_dir: &Path,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        println!("Starting Point-in-Time Recovery");
        println!("Target transaction ID: {}", target_transaction_id);

        let mut log_files = self.collect_log_files()?;
        log_files.sort();

        let mut store = RdfStore::open(output_dir).map_err(|e| {
            format!(
                "Failed to open recovery target '{}': {e}",
                output_dir.display()
            )
        })?;

        let mut applied_transactions = 0;

        'outer: for log_file in log_files {
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
                    break 'outer;
                }

                // Verify checksum
                if !Self::verify_checksum(&entry) {
                    eprintln!(
                        "Warning: Checksum mismatch for transaction {}, skipping",
                        entry.transaction_id
                    );
                    continue;
                }

                // Apply the transaction to the real recovery target store.
                Self::apply_entry_to_store(&mut store, &entry)?;
                applied_transactions += 1;
            }
        }

        store
            .flush()
            .map_err(|e| format!("Failed to flush recovered dataset: {e}"))?;

        println!(
            "Recovery complete: {} transactions applied to {}",
            applied_transactions,
            output_dir.display()
        );
        Ok(applied_transactions)
    }

    /// Decode a WAL entry's `data` payload as N-Quads text (the encoding
    /// written by [`Self::append_quads`]). `Begin`/`Commit`/`Rollback`
    /// markers legitimately carry an empty payload and decode to no quads.
    /// Any other payload that is not valid UTF-8 N-Quads means the
    /// transaction format prevents real replay, so this fails loudly instead
    /// of the caller fabricating a successful "applied" count.
    fn parse_entry_quads(
        entry: &TransactionLogEntry,
    ) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
        if entry.data.is_empty() {
            return Ok(Vec::new());
        }

        let text = std::str::from_utf8(&entry.data).map_err(|e| {
            format!(
                "Transaction {} payload is not valid UTF-8 N-Quads text and cannot be replayed: {e}",
                entry.transaction_id
            )
        })?;

        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        NQuadsParser::new().parse(Cursor::new(text)).map_err(|e| {
            format!(
                "Transaction {} payload could not be parsed as N-Quads and cannot be replayed: {e}",
                entry.transaction_id
            )
            .into()
        })
    }

    /// Apply one transaction log entry to the recovery target store.
    fn apply_entry_to_store(
        store: &mut RdfStore,
        entry: &TransactionLogEntry,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match entry.operation_type {
            OperationType::BeginTransaction
            | OperationType::CommitTransaction
            | OperationType::RollbackTransaction => {
                // Control markers carry no store mutation of their own.
                Ok(())
            }
            OperationType::Insert | OperationType::Update => {
                for quad in Self::parse_entry_quads(entry)? {
                    store.insert_quad(quad).map_err(|e| {
                        format!("Failed to apply transaction {}: {e}", entry.transaction_id)
                    })?;
                }
                Ok(())
            }
            OperationType::Delete => {
                for quad in Self::parse_entry_quads(entry)? {
                    store.remove_quad(&quad).map_err(|e| {
                        format!("Failed to apply transaction {}: {e}", entry.transaction_id)
                    })?;
                }
                Ok(())
            }
        }
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

        checkpoints.sort_by_key(|item| item.timestamp);
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
        store: &mut RdfStore,
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
                    "Warning: Checksum mismatch for transaction {}, skipping",
                    entry.transaction_id
                );
                continue;
            }

            // Apply the transaction to the real recovery target store.
            Self::apply_entry_to_store(store, &entry)?;
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
        // SHA-256 one-shot via OxiCrypto (Pure Rust); hex-encode the 32-byte
        // digest. Produces the same checksum string as the previous `ring`
        // implementation. `hash_fixed` is an inherent method on `Sha256`, so the
        // `Hash` trait does not need to be in scope.
        let hash: [u8; 32] = oxicrypto_hash::Sha256.hash_fixed(data);
        hex::encode(hash)
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

    fn test_quad(subject: &str, predicate: &str, object: &str) -> Quad {
        use oxirs_core::model::{NamedNode, Object};
        Quad::new_default_graph(
            NamedNode::new(subject).expect("subject IRI"),
            NamedNode::new(predicate).expect("predicate IRI"),
            Object::NamedNode(NamedNode::new(object).expect("object IRI")),
        )
    }

    #[test]
    fn test_recover_to_timestamp_applies_real_quads_to_output_store() {
        let temp_log_dir = temp_dir().join("oxirs_pitr_test_recover_ts_logs");
        let temp_archive_dir = temp_dir().join("oxirs_pitr_test_recover_ts_archive");
        let output_dir = temp_dir().join("oxirs_pitr_test_recover_ts_output");
        let _ = fs::remove_dir_all(&temp_log_dir);
        let _ = fs::remove_dir_all(&temp_archive_dir);
        let _ = fs::remove_dir_all(&output_dir);

        let config = PitrConfig {
            log_dir: temp_log_dir.clone(),
            archive_dir: temp_archive_dir.clone(),
            max_log_size: 10_000_000,
            auto_archive: false,
        };
        let mut log = TransactionLog::new(config).unwrap();

        let quad = test_quad(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        );
        log.append_quads(OperationType::Insert, std::slice::from_ref(&quad))
            .unwrap();

        // Recover to a timestamp far in the future so every entry replays.
        let future = Utc::now() + chrono::Duration::hours(1);
        let applied = log.recover_to_timestamp(future, &output_dir).unwrap();
        assert_eq!(
            applied, 1,
            "the single real Insert transaction must be applied"
        );

        // The old implementation only incremented a counter; verify the
        // output directory now holds a real, queryable restored dataset.
        let restored = RdfStore::open(&output_dir).expect("open restored dataset");
        let quads = restored.quads().expect("restored quads");
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].subject().to_string(), "<http://example.org/alice>");

        let _ = fs::remove_dir_all(temp_log_dir);
        let _ = fs::remove_dir_all(temp_archive_dir);
        let _ = fs::remove_dir_all(output_dir);
    }

    #[test]
    fn test_recover_to_transaction_applies_and_deletes_real_quads() {
        let temp_log_dir = temp_dir().join("oxirs_pitr_test_recover_tx_logs");
        let temp_archive_dir = temp_dir().join("oxirs_pitr_test_recover_tx_archive");
        let output_dir = temp_dir().join("oxirs_pitr_test_recover_tx_output");
        let _ = fs::remove_dir_all(&temp_log_dir);
        let _ = fs::remove_dir_all(&temp_archive_dir);
        let _ = fs::remove_dir_all(&output_dir);

        let config = PitrConfig {
            log_dir: temp_log_dir.clone(),
            archive_dir: temp_archive_dir.clone(),
            max_log_size: 10_000_000,
            auto_archive: false,
        };
        let mut log = TransactionLog::new(config).unwrap();

        let q1 = test_quad(
            "http://example.org/s1",
            "http://example.org/p",
            "http://example.org/o1",
        );
        let q2 = test_quad(
            "http://example.org/s2",
            "http://example.org/p",
            "http://example.org/o2",
        );
        let tx1 = log
            .append_quads(OperationType::Insert, std::slice::from_ref(&q1))
            .unwrap();
        let tx2 = log
            .append_quads(OperationType::Insert, std::slice::from_ref(&q2))
            .unwrap();
        let tx3 = log
            .append_quads(OperationType::Delete, std::slice::from_ref(&q1))
            .unwrap();
        assert_eq!((tx1, tx2, tx3), (1, 2, 3));

        // Recover through transaction 3 (both inserts and the delete).
        let applied = log.recover_to_transaction(3, &output_dir).unwrap();
        assert_eq!(applied, 3);

        let restored = RdfStore::open(&output_dir).expect("open restored dataset");
        let quads = restored.quads().expect("restored quads");
        // q1 was inserted then deleted; only q2 must remain — proves real
        // Delete replay, not just a fabricated "applied" counter.
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].subject().to_string(), "<http://example.org/s2>");

        let _ = fs::remove_dir_all(temp_log_dir);
        let _ = fs::remove_dir_all(temp_archive_dir);
        let _ = fs::remove_dir_all(output_dir);
    }

    #[test]
    fn test_recover_to_transaction_stops_at_target_id() {
        let temp_log_dir = temp_dir().join("oxirs_pitr_test_recover_stop_logs");
        let temp_archive_dir = temp_dir().join("oxirs_pitr_test_recover_stop_archive");
        let output_dir = temp_dir().join("oxirs_pitr_test_recover_stop_output");
        let _ = fs::remove_dir_all(&temp_log_dir);
        let _ = fs::remove_dir_all(&temp_archive_dir);
        let _ = fs::remove_dir_all(&output_dir);

        let config = PitrConfig {
            log_dir: temp_log_dir.clone(),
            archive_dir: temp_archive_dir.clone(),
            max_log_size: 10_000_000,
            auto_archive: false,
        };
        let mut log = TransactionLog::new(config).unwrap();

        let q1 = test_quad(
            "http://example.org/s1",
            "http://example.org/p",
            "http://example.org/o1",
        );
        let q2 = test_quad(
            "http://example.org/s2",
            "http://example.org/p",
            "http://example.org/o2",
        );
        log.append_quads(OperationType::Insert, std::slice::from_ref(&q1))
            .unwrap();
        log.append_quads(OperationType::Insert, std::slice::from_ref(&q2))
            .unwrap();

        let applied = log.recover_to_transaction(1, &output_dir).unwrap();
        assert_eq!(applied, 1, "must stop after the target transaction id");

        let restored = RdfStore::open(&output_dir).expect("open restored dataset");
        assert_eq!(restored.quads().unwrap().len(), 1);

        let _ = fs::remove_dir_all(temp_log_dir);
        let _ = fs::remove_dir_all(temp_archive_dir);
        let _ = fs::remove_dir_all(output_dir);
    }

    #[test]
    fn test_recover_fails_loudly_on_non_replayable_payload() {
        let temp_log_dir = temp_dir().join("oxirs_pitr_test_recover_fail_logs");
        let temp_archive_dir = temp_dir().join("oxirs_pitr_test_recover_fail_archive");
        let output_dir = temp_dir().join("oxirs_pitr_test_recover_fail_output");
        let _ = fs::remove_dir_all(&temp_log_dir);
        let _ = fs::remove_dir_all(&temp_archive_dir);
        let _ = fs::remove_dir_all(&output_dir);

        let config = PitrConfig {
            log_dir: temp_log_dir.clone(),
            archive_dir: temp_archive_dir.clone(),
            max_log_size: 10_000_000,
            auto_archive: false,
        };
        let mut log = TransactionLog::new(config).unwrap();

        // An Insert entry whose payload is NOT N-Quads text (e.g. written by
        // some other producer via the raw `append` API) must cause recovery
        // to fail loudly rather than silently reporting fake success.
        log.append(OperationType::Insert, b"this is not n-quads".to_vec())
            .unwrap();

        let future = Utc::now() + chrono::Duration::hours(1);
        let result = log.recover_to_timestamp(future, &output_dir);
        assert!(
            result.is_err(),
            "a transaction format that prevents real replay must fail loudly"
        );

        let _ = fs::remove_dir_all(temp_log_dir);
        let _ = fs::remove_dir_all(temp_archive_dir);
        let _ = fs::remove_dir_all(output_dir);
    }
}
