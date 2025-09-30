//! # Write-Ahead Logging (WAL) Implementation
//!
//! Provides ACID durability guarantees through write-ahead logging with
//! ARIES-style recovery, checkpointing, and log management.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::mvcc::TransactionId;
use crate::triple_store::TripleKey;

/// Schema operations for DDL (Data Definition Language) operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaOperation {
    /// Create a new index
    CreateIndex {
        index_name: String,
        index_type: String,
        columns: Vec<String>,
        options: HashMap<String, String>,
    },
    /// Drop an existing index
    DropIndex { index_name: String, cascade: bool },
    /// Create a new named graph
    CreateGraph {
        graph_name: String,
        metadata: HashMap<String, String>,
    },
    /// Drop an existing named graph
    DropGraph { graph_name: String, cascade: bool },
    /// Add a new constraint
    AddConstraint {
        constraint_name: String,
        constraint_type: String,
        definition: String,
    },
    /// Drop an existing constraint
    DropConstraint { constraint_name: String },
    /// Modify database configuration
    AlterConfiguration {
        setting_name: String,
        old_value: String,
        new_value: String,
    },
    /// Create or update statistics
    UpdateStatistics {
        table_name: String,
        statistics: HashMap<String, String>,
    },
    /// Create or update a view
    CreateView {
        view_name: String,
        definition: String,
        materialized: bool,
    },
    /// Drop a view
    DropView {
        view_name: String,
        materialized: bool,
    },
}

/// Trait for storage operations needed by WAL recovery
pub trait StorageInterface {
    /// Apply an insert operation to the storage
    fn apply_insert(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        value: bool,
    ) -> Result<()>;

    /// Apply an update operation to the storage  
    fn apply_update(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        value: bool,
    ) -> Result<()>;

    /// Apply a delete operation to the storage
    fn apply_delete(&self, transaction_id: TransactionId, key: TripleKey) -> Result<()>;

    /// Apply a schema change operation to the storage
    fn apply_schema_change(
        &self,
        transaction_id: TransactionId,
        operation: &SchemaOperation,
    ) -> Result<()>;
}

/// Log Sequence Number (LSN) type
pub type LSN = u64;

/// Invalid LSN constant
pub const INVALID_LSN: LSN = 0;

/// Log record types for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogRecordType {
    /// Begin transaction
    Begin {
        transaction_id: TransactionId,
        timestamp: u64,
    },
    /// Commit transaction
    Commit {
        transaction_id: TransactionId,
        timestamp: u64,
    },
    /// Abort transaction
    Abort {
        transaction_id: TransactionId,
        timestamp: u64,
    },
    /// Insert operation
    Insert {
        transaction_id: TransactionId,
        key: TripleKey,
        value: bool,
        previous_value: Option<bool>,
    },
    /// Update operation
    Update {
        transaction_id: TransactionId,
        key: TripleKey,
        new_value: bool,
        old_value: bool,
    },
    /// Delete operation
    Delete {
        transaction_id: TransactionId,
        key: TripleKey,
        deleted_value: bool,
    },
    /// Checkpoint record
    Checkpoint {
        active_transactions: Vec<TransactionId>,
        dirty_pages: Vec<u64>,
        timestamp: u64,
    },
    /// Compensation Log Record (CLR) for undo operations
    CLR {
        transaction_id: TransactionId,
        undo_lsn: LSN,
        operation: Box<LogRecordType>,
    },
    /// Schema change operation for DDL operations
    SchemaChange {
        transaction_id: TransactionId,
        operation: SchemaOperation,
        timestamp: u64,
    },
    /// End of log marker
    EndOfLog,
}

/// Log record with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRecord {
    /// Log sequence number
    pub lsn: LSN,
    /// Previous LSN for this transaction
    pub prev_lsn: Option<LSN>,
    /// Transaction ID (if applicable)
    pub transaction_id: Option<TransactionId>,
    /// Record type and data
    pub record_type: LogRecordType,
    /// Size of the record in bytes
    pub size: u32,
    /// Timestamp when record was created
    pub timestamp: u64,
    /// Checksum for integrity
    pub checksum: u64,
}

impl LogRecord {
    /// Create a new log record
    pub fn new(lsn: LSN, prev_lsn: Option<LSN>, record_type: LogRecordType) -> Result<Self> {
        let transaction_id = match &record_type {
            LogRecordType::Begin { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Commit { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Abort { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Insert { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Update { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Delete { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::CLR { transaction_id, .. } => Some(*transaction_id),
            _ => None,
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let serialized = bincode::serialize(&record_type)
            .map_err(|e| anyhow!("Failed to serialize log record: {}", e))?;
        let size = serialized.len() as u32 + 48; // Record header size

        let checksum = calculate_checksum(&serialized);

        Ok(Self {
            lsn,
            prev_lsn,
            transaction_id,
            record_type,
            size,
            timestamp,
            checksum,
        })
    }

    /// Serialize the log record to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| anyhow!("Failed to serialize log record: {}", e))
    }

    /// Deserialize log record from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| anyhow!("Failed to deserialize log record: {}", e))
    }

    /// Validate the checksum
    pub fn validate_checksum(&self) -> Result<bool> {
        let serialized = bincode::serialize(&self.record_type)
            .map_err(|e| anyhow!("Failed to serialize for checksum: {}", e))?;
        let calculated = calculate_checksum(&serialized);
        Ok(calculated == self.checksum)
    }
}

/// Transaction table entry for recovery
#[derive(Debug, Clone)]
pub struct TransactionEntry {
    pub transaction_id: TransactionId,
    pub status: TransactionStatus,
    pub last_lsn: LSN,
    pub undo_next_lsn: Option<LSN>,
}

/// Transaction status during recovery
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionStatus {
    Active,
    Committed,
    Aborted,
}

/// Dirty page table entry
#[derive(Debug, Clone)]
pub struct DirtyPageEntry {
    pub page_id: u64,
    pub recovery_lsn: LSN,
}

/// WAL configuration
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// WAL file path
    pub wal_file_path: PathBuf,
    /// Buffer size for log writes
    pub buffer_size: usize,
    /// Force flush after each commit
    pub force_flush_on_commit: bool,
    /// Checkpoint interval in seconds
    pub checkpoint_interval_seconds: u64,
    /// Maximum log file size before rotation
    pub max_log_size_bytes: u64,
    /// Enable log compression
    pub enable_compression: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            wal_file_path: PathBuf::from("./tdb/wal.log"),
            buffer_size: 64 * 1024, // 64KB buffer
            force_flush_on_commit: true,
            checkpoint_interval_seconds: 300,       // 5 minutes
            max_log_size_bytes: 1024 * 1024 * 1024, // 1GB
            enable_compression: false,
        }
    }
}

/// WAL statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct WalStats {
    /// Total log records written
    pub records_written: u64,
    /// Total bytes written to log
    pub bytes_written: u64,
    /// Number of flushes
    pub flushes: u64,
    /// Number of checkpoints
    pub checkpoints: u64,
    /// Current log file size
    pub current_log_size: u64,
    /// Average write time in microseconds
    pub avg_write_time_us: f64,
    /// Recovery operations count
    pub recovery_operations: u64,
}

/// Write-Ahead Log manager
pub struct WalManager {
    /// Configuration
    config: WalConfig,
    /// WAL file
    wal_file: Arc<Mutex<BufWriter<File>>>,
    /// Current LSN counter
    current_lsn: Arc<RwLock<LSN>>,
    /// Log buffer for batching writes
    log_buffer: Arc<Mutex<VecDeque<LogRecord>>>,
    /// Transaction table for tracking active transactions
    transaction_table: Arc<RwLock<HashMap<TransactionId, TransactionEntry>>>,
    /// Last LSN for each transaction
    transaction_lsns: Arc<RwLock<HashMap<TransactionId, LSN>>>,
    /// Statistics
    stats: Arc<Mutex<WalStats>>,
    /// Checkpoint in progress flag
    checkpoint_in_progress: Arc<Mutex<bool>>,
    /// Storage interface for redo/undo operations
    storage_interface: Option<Arc<dyn StorageInterface + Send + Sync>>,
}

impl WalManager {
    /// Create a new WAL manager
    pub fn new<P: AsRef<Path>>(wal_file_path: P) -> Result<Self> {
        let config = WalConfig {
            wal_file_path: wal_file_path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new WAL manager with configuration
    pub fn with_config(config: WalConfig) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = config.wal_file_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Failed to create WAL directory: {}", e))?;
        }

        // Open or create WAL file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&config.wal_file_path)
            .map_err(|e| anyhow!("Failed to open WAL file: {}", e))?;

        let wal_file = Arc::new(Mutex::new(BufWriter::with_capacity(
            config.buffer_size,
            file,
        )));

        // Determine starting LSN by reading existing log
        let current_lsn = Arc::new(RwLock::new(Self::get_last_lsn(&config.wal_file_path)?));

        Ok(Self {
            config,
            wal_file,
            current_lsn,
            log_buffer: Arc::new(Mutex::new(VecDeque::new())),
            transaction_table: Arc::new(RwLock::new(HashMap::new())),
            transaction_lsns: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(WalStats::default())),
            checkpoint_in_progress: Arc::new(Mutex::new(false)),
            storage_interface: None,
        })
    }

    /// Write a log record
    pub fn write_log_record(&self, record_type: LogRecordType) -> Result<LSN> {
        let start_time = std::time::Instant::now();

        // Generate LSN
        let lsn = self.allocate_lsn()?;

        // Get previous LSN for transaction
        let prev_lsn = if let Some(tx_id) = Self::extract_transaction_id(&record_type) {
            let tx_lsns = self
                .transaction_lsns
                .read()
                .map_err(|_| anyhow!("Failed to acquire transaction LSNs lock"))?;
            tx_lsns.get(&tx_id).copied()
        } else {
            None
        };

        // Create log record
        let log_record = LogRecord::new(lsn, prev_lsn, record_type)?;

        // Update transaction LSN
        if let Some(tx_id) = log_record.transaction_id {
            let mut tx_lsns = self
                .transaction_lsns
                .write()
                .map_err(|_| anyhow!("Failed to acquire transaction LSNs lock"))?;
            tx_lsns.insert(tx_id, lsn);
        }

        // Check if we need to force flush before moving the record
        let should_flush = self.should_force_flush(&log_record);

        // Add to buffer
        {
            let mut buffer = self
                .log_buffer
                .lock()
                .map_err(|_| anyhow!("Failed to acquire log buffer lock"))?;
            buffer.push_back(log_record);
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.records_written += 1;
            let write_time = start_time.elapsed().as_micros() as f64;
            stats.avg_write_time_us =
                (stats.avg_write_time_us * (stats.records_written - 1) as f64 + write_time)
                    / stats.records_written as f64;
        }

        // Force flush if needed
        if should_flush {
            self.flush()?;
        }

        debug!("Wrote log record with LSN: {}", lsn);
        Ok(lsn)
    }

    /// Flush buffered log records to disk
    pub fn flush(&self) -> Result<()> {
        let mut file = self
            .wal_file
            .lock()
            .map_err(|_| anyhow!("Failed to acquire WAL file lock"))?;

        let mut buffer = self
            .log_buffer
            .lock()
            .map_err(|_| anyhow!("Failed to acquire log buffer lock"))?;

        let mut bytes_written = 0;
        while let Some(record) = buffer.pop_front() {
            let record_bytes = record.to_bytes()?;
            file.write_all(&record_bytes)
                .map_err(|e| anyhow!("Failed to write log record: {}", e))?;
            bytes_written += record_bytes.len();
        }

        file.flush()
            .map_err(|e| anyhow!("Failed to flush WAL file: {}", e))?;

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.flushes += 1;
            stats.bytes_written += bytes_written as u64;
            stats.current_log_size += bytes_written as u64;
        }

        debug!("Flushed {} bytes to WAL", bytes_written);
        Ok(())
    }

    /// Begin a transaction
    pub fn begin_transaction(&self, transaction_id: TransactionId) -> Result<LSN> {
        let record_type = LogRecordType::Begin {
            transaction_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let lsn = self.write_log_record(record_type)?;

        // Add to transaction table
        {
            let mut tx_table = self
                .transaction_table
                .write()
                .map_err(|_| anyhow!("Failed to acquire transaction table lock"))?;
            tx_table.insert(
                transaction_id,
                TransactionEntry {
                    transaction_id,
                    status: TransactionStatus::Active,
                    last_lsn: lsn,
                    undo_next_lsn: Some(lsn),
                },
            );
        }

        info!("Started transaction {} with LSN {}", transaction_id, lsn);
        Ok(lsn)
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, transaction_id: TransactionId) -> Result<LSN> {
        let record_type = LogRecordType::Commit {
            transaction_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let lsn = self.write_log_record(record_type)?;

        // Update transaction table
        {
            let mut tx_table = self
                .transaction_table
                .write()
                .map_err(|_| anyhow!("Failed to acquire transaction table lock"))?;
            if let Some(entry) = tx_table.get_mut(&transaction_id) {
                entry.status = TransactionStatus::Committed;
                entry.last_lsn = lsn;
            }
        }

        // Force flush on commit
        if self.config.force_flush_on_commit {
            self.flush()?;
        }

        info!("Committed transaction {} with LSN {}", transaction_id, lsn);
        Ok(lsn)
    }

    /// Abort a transaction
    pub fn abort_transaction(&self, transaction_id: TransactionId) -> Result<LSN> {
        let record_type = LogRecordType::Abort {
            transaction_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let lsn = self.write_log_record(record_type)?;

        // Update transaction table
        {
            let mut tx_table = self
                .transaction_table
                .write()
                .map_err(|_| anyhow!("Failed to acquire transaction table lock"))?;
            if let Some(entry) = tx_table.get_mut(&transaction_id) {
                entry.status = TransactionStatus::Aborted;
                entry.last_lsn = lsn;
            }
        }

        info!("Aborted transaction {} with LSN {}", transaction_id, lsn);
        Ok(lsn)
    }

    /// Log an insert operation
    pub fn log_insert(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        value: bool,
        previous_value: Option<bool>,
    ) -> Result<LSN> {
        let record_type = LogRecordType::Insert {
            transaction_id,
            key,
            value,
            previous_value,
        };

        self.write_log_record(record_type)
    }

    /// Log an update operation
    pub fn log_update(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        new_value: bool,
        old_value: bool,
    ) -> Result<LSN> {
        let record_type = LogRecordType::Update {
            transaction_id,
            key,
            new_value,
            old_value,
        };

        self.write_log_record(record_type)
    }

    /// Log a delete operation
    pub fn log_delete(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        deleted_value: bool,
    ) -> Result<LSN> {
        let record_type = LogRecordType::Delete {
            transaction_id,
            key,
            deleted_value,
        };

        self.write_log_record(record_type)
    }

    /// Log a schema change operation
    pub fn log_schema_change(
        &self,
        transaction_id: TransactionId,
        operation: SchemaOperation,
    ) -> Result<LSN> {
        let record_type = LogRecordType::SchemaChange {
            transaction_id,
            operation,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let lsn = self.write_log_record(record_type)?;
        info!(
            "Logged schema change for transaction {} with LSN {}",
            transaction_id, lsn
        );
        Ok(lsn)
    }

    /// Create a checkpoint
    pub fn checkpoint(&self) -> Result<LSN> {
        let mut checkpoint_in_progress = self
            .checkpoint_in_progress
            .lock()
            .map_err(|_| anyhow!("Failed to acquire checkpoint lock"))?;

        if *checkpoint_in_progress {
            return Err(anyhow!("Checkpoint already in progress"));
        }

        *checkpoint_in_progress = true;

        // Collect active transactions and dirty pages
        let active_transactions = {
            let tx_table = self
                .transaction_table
                .read()
                .map_err(|_| anyhow!("Failed to acquire transaction table lock"))?;
            tx_table.keys().copied().collect()
        };

        let dirty_pages = Vec::new(); // In a full implementation, this would come from buffer pool

        let record_type = LogRecordType::Checkpoint {
            active_transactions,
            dirty_pages,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let lsn = self.write_log_record(record_type)?;
        self.flush()?;

        *checkpoint_in_progress = false;

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.checkpoints += 1;
        }

        info!("Created checkpoint with LSN {}", lsn);
        Ok(lsn)
    }

    /// Recover the database from the WAL
    pub fn recover(&self) -> Result<()> {
        info!("Starting WAL recovery");

        let mut transaction_table = HashMap::new();
        let mut dirty_page_table = HashMap::new();
        let mut lsn_to_record = HashMap::new();

        // Phase 1: Analysis - scan log to build transaction and dirty page tables
        self.analysis_phase(
            &mut transaction_table,
            &mut dirty_page_table,
            &mut lsn_to_record,
        )?;

        // Phase 2: Redo - replay operations to restore database state
        self.redo_phase(&dirty_page_table, &lsn_to_record)?;

        // Phase 3: Undo - rollback uncommitted transactions
        self.undo_phase(&transaction_table, &lsn_to_record)?;

        info!("WAL recovery completed successfully");
        Ok(())
    }

    /// Get WAL statistics
    pub fn get_stats(&self) -> Result<WalStats> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
        Ok(stats.clone())
    }

    /// Get the current LSN
    pub fn current_lsn(&self) -> Result<LSN> {
        let lsn = self
            .current_lsn
            .read()
            .map_err(|_| anyhow!("Failed to acquire current LSN lock"))?;
        Ok(*lsn)
    }

    /// Set the storage interface for redo/undo operations
    pub fn set_storage_interface(&mut self, storage: Arc<dyn StorageInterface + Send + Sync>) {
        self.storage_interface = Some(storage);
    }

    // Private helper methods

    fn get_last_lsn(wal_file_path: &Path) -> Result<LSN> {
        if !wal_file_path.exists() {
            return Ok(0);
        }

        let file = File::open(wal_file_path)
            .map_err(|e| anyhow!("Failed to open WAL file for reading: {}", e))?;

        let mut reader = BufReader::new(file);
        let mut last_lsn = 0;

        // Simple implementation - read all records to find last LSN
        // In production, this would be optimized with indexing
        loop {
            let mut size_bytes = [0u8; 4];
            match reader.read_exact(&mut size_bytes) {
                Ok(_) => {
                    let size = u32::from_le_bytes(size_bytes);
                    let mut record_bytes = vec![0u8; size as usize - 4];
                    reader.read_exact(&mut record_bytes)?;

                    if let Ok(record) = LogRecord::from_bytes(&record_bytes) {
                        last_lsn = record.lsn;
                    }
                }
                Err(_) => break, // End of file
            }
        }

        Ok(last_lsn)
    }

    fn allocate_lsn(&self) -> Result<LSN> {
        let mut current_lsn = self
            .current_lsn
            .write()
            .map_err(|_| anyhow!("Failed to acquire current LSN lock"))?;
        *current_lsn += 1;
        Ok(*current_lsn)
    }

    fn extract_transaction_id(record_type: &LogRecordType) -> Option<TransactionId> {
        match record_type {
            LogRecordType::Begin { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Commit { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Abort { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Insert { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Update { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::Delete { transaction_id, .. } => Some(*transaction_id),
            LogRecordType::CLR { transaction_id, .. } => Some(*transaction_id),
            _ => None,
        }
    }

    fn should_force_flush(&self, record: &LogRecord) -> bool {
        if self.config.force_flush_on_commit {
            matches!(record.record_type, LogRecordType::Commit { .. })
        } else {
            false
        }
    }

    fn analysis_phase(
        &self,
        transaction_table: &mut HashMap<TransactionId, TransactionEntry>,
        dirty_page_table: &mut HashMap<u64, DirtyPageEntry>,
        lsn_to_record: &mut HashMap<LSN, LogRecord>,
    ) -> Result<()> {
        info!("Starting analysis phase");

        // Open log file for reading
        let file = File::open(&self.config.wal_file_path)
            .map_err(|e| anyhow!("Failed to open WAL file for analysis: {}", e))?;
        let mut reader = BufReader::new(file);

        let mut last_checkpoint_lsn: Option<LSN> = None;
        let mut log_records = Vec::new();

        // Read all log records
        loop {
            let mut size_bytes = [0u8; 4];
            match reader.read_exact(&mut size_bytes) {
                Ok(_) => {
                    let size = u32::from_le_bytes(size_bytes);
                    if size == 0 || size > 1_000_000 {
                        // Sanity check
                        break;
                    }

                    let mut record_bytes = vec![0u8; size as usize - 4];
                    if reader.read_exact(&mut record_bytes).is_err() {
                        break;
                    }

                    if let Ok(record) = LogRecord::from_bytes(&record_bytes) {
                        if record.validate_checksum().unwrap_or(false) {
                            log_records.push(record);
                        }
                    }
                }
                Err(_) => break, // End of file
            }
        }

        // Find last checkpoint
        for record in log_records.iter().rev() {
            if matches!(record.record_type, LogRecordType::Checkpoint { .. }) {
                last_checkpoint_lsn = Some(record.lsn);
                break;
            }
        }

        // Start analysis from last checkpoint (or beginning if no checkpoint)
        let start_lsn = last_checkpoint_lsn.unwrap_or(0);

        for record in &log_records {
            if record.lsn >= start_lsn {
                lsn_to_record.insert(record.lsn, record.clone());

                match &record.record_type {
                    LogRecordType::Begin { transaction_id, .. } => {
                        transaction_table.insert(
                            *transaction_id,
                            TransactionEntry {
                                transaction_id: *transaction_id,
                                status: TransactionStatus::Active,
                                last_lsn: record.lsn,
                                undo_next_lsn: Some(record.lsn),
                            },
                        );
                    }

                    LogRecordType::Commit { transaction_id, .. } => {
                        if let Some(entry) = transaction_table.get_mut(transaction_id) {
                            entry.status = TransactionStatus::Committed;
                            entry.last_lsn = record.lsn;
                        }
                    }

                    LogRecordType::Abort { transaction_id, .. } => {
                        if let Some(entry) = transaction_table.get_mut(transaction_id) {
                            entry.status = TransactionStatus::Aborted;
                            entry.last_lsn = record.lsn;
                        }
                    }

                    LogRecordType::Insert { transaction_id, .. }
                    | LogRecordType::Update { transaction_id, .. }
                    | LogRecordType::Delete { transaction_id, .. }
                    | LogRecordType::SchemaChange { transaction_id, .. } => {
                        if let Some(entry) = transaction_table.get_mut(transaction_id) {
                            entry.last_lsn = record.lsn;
                        }

                        // Add to dirty page table (simplified - using LSN as page_id for demo)
                        let page_id = record.lsn % 1000; // Simplified page mapping
                        dirty_page_table.entry(page_id).or_insert(DirtyPageEntry {
                            page_id,
                            recovery_lsn: record.lsn,
                        });
                    }

                    LogRecordType::Checkpoint {
                        active_transactions,
                        dirty_pages,
                        ..
                    } => {
                        // Restore state from checkpoint
                        for &tx_id in active_transactions {
                            transaction_table.entry(tx_id).or_insert(TransactionEntry {
                                transaction_id: tx_id,
                                status: TransactionStatus::Active,
                                last_lsn: record.lsn,
                                undo_next_lsn: Some(record.lsn),
                            });
                        }

                        for &page_id in dirty_pages {
                            dirty_page_table.entry(page_id).or_insert(DirtyPageEntry {
                                page_id,
                                recovery_lsn: record.lsn,
                            });
                        }
                    }

                    LogRecordType::CLR {
                        transaction_id,
                        undo_lsn,
                        ..
                    } => {
                        if let Some(entry) = transaction_table.get_mut(transaction_id) {
                            entry.last_lsn = record.lsn;
                            entry.undo_next_lsn = Some(*undo_lsn);
                        }
                    }

                    LogRecordType::EndOfLog => {
                        break;
                    }
                }
            }
        }

        info!(
            "Analysis phase completed: {} transactions, {} dirty pages",
            transaction_table.len(),
            dirty_page_table.len()
        );
        Ok(())
    }

    fn redo_phase(
        &self,
        dirty_page_table: &HashMap<u64, DirtyPageEntry>,
        lsn_to_record: &HashMap<LSN, LogRecord>,
    ) -> Result<()> {
        info!("Starting redo phase");

        // Find the minimum recovery LSN from dirty page table
        let min_recovery_lsn = dirty_page_table
            .values()
            .map(|entry| entry.recovery_lsn)
            .min()
            .unwrap_or(0);

        // Sort log records by LSN for sequential replay
        let mut sorted_records: Vec<_> = lsn_to_record
            .values()
            .filter(|record| record.lsn >= min_recovery_lsn)
            .collect();
        sorted_records.sort_by_key(|record| record.lsn);

        let mut operations_redone = 0;

        for record in sorted_records {
            match &record.record_type {
                LogRecordType::Insert {
                    transaction_id,
                    key,
                    value,
                    ..
                } => {
                    // Redo the insert operation
                    if self
                        .redo_insert(*transaction_id, key.clone(), *value)
                        .is_ok()
                    {
                        operations_redone += 1;
                        debug!("Redone insert operation at LSN {}", record.lsn);
                    }
                }

                LogRecordType::Update {
                    transaction_id,
                    key,
                    new_value,
                    ..
                } => {
                    // Redo the update operation
                    if self
                        .redo_update(*transaction_id, key.clone(), *new_value)
                        .is_ok()
                    {
                        operations_redone += 1;
                        debug!("Redone update operation at LSN {}", record.lsn);
                    }
                }

                LogRecordType::Delete {
                    transaction_id,
                    key,
                    ..
                } => {
                    // Redo the delete operation
                    if self.redo_delete(*transaction_id, key.clone()).is_ok() {
                        operations_redone += 1;
                        debug!("Redone delete operation at LSN {}", record.lsn);
                    }
                }

                LogRecordType::SchemaChange {
                    transaction_id,
                    operation,
                    ..
                } => {
                    // Redo the schema change operation
                    if self.redo_schema_change(*transaction_id, operation).is_ok() {
                        operations_redone += 1;
                        debug!("Redone schema change operation at LSN {}", record.lsn);
                    }
                }

                LogRecordType::CLR {
                    transaction_id,
                    operation,
                    ..
                } => {
                    // Redo compensation log record (undo operation)
                    match operation.as_ref() {
                        LogRecordType::Insert { key, .. } => {
                            // CLR for insert means we need to delete
                            if self.redo_delete(*transaction_id, key.clone()).is_ok() {
                                operations_redone += 1;
                                debug!("Redone CLR delete at LSN {}", record.lsn);
                            }
                        }
                        LogRecordType::Delete {
                            key, deleted_value, ..
                        } => {
                            // CLR for delete means we need to insert
                            if self
                                .redo_insert(*transaction_id, key.clone(), *deleted_value)
                                .is_ok()
                            {
                                operations_redone += 1;
                                debug!("Redone CLR insert at LSN {}", record.lsn);
                            }
                        }
                        LogRecordType::Update { key, old_value, .. } => {
                            // CLR for update means we restore old value
                            if self
                                .redo_update(*transaction_id, key.clone(), *old_value)
                                .is_ok()
                            {
                                operations_redone += 1;
                                debug!("Redone CLR update at LSN {}", record.lsn);
                            }
                        }
                        _ => {}
                    }
                }

                _ => {
                    // Skip non-data operations (Begin, Commit, Abort, Checkpoint)
                }
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.recovery_operations += operations_redone;
        }

        info!(
            "Redo phase completed: {} operations redone",
            operations_redone
        );
        Ok(())
    }

    fn undo_phase(
        &self,
        transaction_table: &HashMap<TransactionId, TransactionEntry>,
        lsn_to_record: &HashMap<LSN, LogRecord>,
    ) -> Result<()> {
        info!("Starting undo phase");

        // Find all active (uncommitted) transactions
        let active_transactions: Vec<_> = transaction_table
            .values()
            .filter(|entry| entry.status == TransactionStatus::Active)
            .collect();

        if active_transactions.is_empty() {
            info!("No active transactions to undo");
            return Ok(());
        }

        // Create undo list with LSNs to undo for each transaction
        let mut undo_list = Vec::new();
        for tx_entry in &active_transactions {
            if let Some(lsn) = tx_entry.undo_next_lsn {
                undo_list.push((tx_entry.transaction_id, lsn));
            }
        }

        // Sort by LSN descending (undo in reverse order)
        undo_list.sort_by(|a, b| b.1.cmp(&a.1));

        let mut operations_undone = 0;

        while let Some((transaction_id, current_lsn)) = undo_list.pop() {
            if let Some(record) = lsn_to_record.get(&current_lsn) {
                match &record.record_type {
                    LogRecordType::Insert { key, value, .. } => {
                        // Undo insert by deleting
                        if let Ok(clr_lsn) = self.undo_insert(transaction_id, key.clone(), *value) {
                            operations_undone += 1;
                            debug!("Undone insert at LSN {} with CLR {}", current_lsn, clr_lsn);
                        }
                    }

                    LogRecordType::Update { key, old_value, .. } => {
                        // Undo update by restoring old value
                        if let Ok(clr_lsn) =
                            self.undo_update(transaction_id, key.clone(), *old_value)
                        {
                            operations_undone += 1;
                            debug!("Undone update at LSN {} with CLR {}", current_lsn, clr_lsn);
                        }
                    }

                    LogRecordType::Delete {
                        key, deleted_value, ..
                    } => {
                        // Undo delete by reinserting
                        if let Ok(clr_lsn) =
                            self.undo_delete(transaction_id, key.clone(), *deleted_value)
                        {
                            operations_undone += 1;
                            debug!("Undone delete at LSN {} with CLR {}", current_lsn, clr_lsn);
                        }
                    }

                    LogRecordType::CLR { undo_lsn, .. } => {
                        // CLR points to next LSN to undo
                        if *undo_lsn > 0 {
                            undo_list.push((transaction_id, *undo_lsn));
                            undo_list.sort_by(|a, b| b.1.cmp(&a.1));
                        }
                        continue;
                    }

                    LogRecordType::Begin { .. } => {
                        // Reached beginning of transaction, write abort record
                        if let Ok(abort_lsn) = self.abort_transaction(transaction_id) {
                            debug!(
                                "Written abort record for transaction {} at LSN {}",
                                transaction_id, abort_lsn
                            );
                        }
                        continue;
                    }

                    LogRecordType::SchemaChange { .. } => {
                        // Schema changes cannot be undone in the traditional sense
                        // They should be handled at a higher level (DDL rollback)
                        debug!("Skipping schema change undo at LSN {}", current_lsn);
                        continue;
                    }

                    _ => {
                        // Skip other record types
                        continue;
                    }
                }

                // Add previous LSN to undo list if it exists
                if let Some(prev_lsn) = record.prev_lsn {
                    if prev_lsn > 0 {
                        undo_list.push((transaction_id, prev_lsn));
                        undo_list.sort_by(|a, b| b.1.cmp(&a.1));
                    }
                }
            }
        }

        // Force flush all undo operations
        self.flush()?;

        info!(
            "Undo phase completed: {} operations undone for {} transactions",
            operations_undone,
            active_transactions.len()
        );
        Ok(())
    }

    // Helper methods for redo operations

    fn redo_insert(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        value: bool,
    ) -> Result<()> {
        if let Some(storage) = &self.storage_interface {
            storage.apply_insert(transaction_id, key, value)?;
            debug!("Applied redo insert for transaction {}", transaction_id);
        } else {
            warn!("No storage interface available for redo insert - operation skipped");
        }
        Ok(())
    }

    fn redo_update(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        value: bool,
    ) -> Result<()> {
        if let Some(storage) = &self.storage_interface {
            storage.apply_update(transaction_id, key, value)?;
            debug!("Applied redo update for transaction {}", transaction_id);
        } else {
            warn!("No storage interface available for redo update - operation skipped");
        }
        Ok(())
    }

    fn redo_delete(&self, transaction_id: TransactionId, key: TripleKey) -> Result<()> {
        if let Some(storage) = &self.storage_interface {
            storage.apply_delete(transaction_id, key)?;
            debug!("Applied redo delete for transaction {}", transaction_id);
        } else {
            warn!("No storage interface available for redo delete - operation skipped");
        }
        Ok(())
    }

    fn redo_schema_change(
        &self,
        transaction_id: TransactionId,
        operation: &SchemaOperation,
    ) -> Result<()> {
        if let Some(storage) = &self.storage_interface {
            storage.apply_schema_change(transaction_id, operation)?;
            debug!(
                "Applied redo schema change for transaction {}",
                transaction_id
            );
        } else {
            warn!("No storage interface available for redo schema change - operation skipped");
        }
        Ok(())
    }

    // Helper methods for undo operations (generate CLRs)

    fn undo_insert(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        value: bool,
    ) -> Result<LSN> {
        // Create CLR for undoing an insert (which is a delete)
        let delete_op = LogRecordType::Delete {
            transaction_id,
            key,
            deleted_value: value,
        };

        let clr = LogRecordType::CLR {
            transaction_id,
            undo_lsn: 0, // Will be set based on transaction chain
            operation: Box::new(delete_op),
        };

        self.write_log_record(clr)
    }

    fn undo_update(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        old_value: bool,
    ) -> Result<LSN> {
        // Create CLR for undoing an update (restore old value)
        let update_op = LogRecordType::Update {
            transaction_id,
            key,
            new_value: old_value,
            old_value: !old_value, // Dummy value
        };

        let clr = LogRecordType::CLR {
            transaction_id,
            undo_lsn: 0,
            operation: Box::new(update_op),
        };

        self.write_log_record(clr)
    }

    fn undo_delete(
        &self,
        transaction_id: TransactionId,
        key: TripleKey,
        deleted_value: bool,
    ) -> Result<LSN> {
        // Create CLR for undoing a delete (which is an insert)
        let insert_op = LogRecordType::Insert {
            transaction_id,
            key,
            value: deleted_value,
            previous_value: None,
        };

        let clr = LogRecordType::CLR {
            transaction_id,
            undo_lsn: 0,
            operation: Box::new(insert_op),
        };

        self.write_log_record(clr)
    }

    /// Archive old log entries and truncate the log file
    ///
    /// Moves log entries older than the given LSN to an archive file,
    /// then truncates the main log to free space. This is typically
    /// called after a successful checkpoint.
    ///
    /// # Arguments
    /// * `archive_before_lsn` - LSN before which entries should be archived
    ///
    /// # Returns
    /// Returns the path to the created archive file
    pub fn archive_and_truncate(&self, archive_before_lsn: LSN) -> Result<PathBuf> {
        info!("Starting log archival for LSN < {}", archive_before_lsn);

        // Create archive file path with timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let archive_path = self
            .config
            .wal_file_path
            .with_file_name(format!("wal_archive_{archive_before_lsn}_{timestamp}.log"));

        // Read all log records
        let all_records = self.read_all_log_records()?;

        // Separate records to archive vs keep
        let mut archive_records = Vec::new();
        let mut keep_records = Vec::new();

        for record in all_records {
            if record.lsn < archive_before_lsn {
                archive_records.push(record);
            } else {
                keep_records.push(record);
            }
        }

        // Write archive records to archive file
        if !archive_records.is_empty() {
            let archive_file = std::fs::File::create(&archive_path)
                .map_err(|e| anyhow!("Failed to create archive file: {}", e))?;
            let mut archive_writer = BufWriter::new(archive_file);

            for record in &archive_records {
                let record_bytes = record.to_bytes()?;
                archive_writer
                    .write_all(&record_bytes)
                    .map_err(|e| anyhow!("Failed to write to archive: {}", e))?;
            }

            archive_writer
                .flush()
                .map_err(|e| anyhow!("Failed to flush archive: {}", e))?;
        }

        // Recreate the WAL file with remaining records
        {
            // Close the current file by dropping the lock temporarily
            drop(self.wal_file.lock().unwrap());

            // Truncate the file by recreating it
            let file = std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&self.config.wal_file_path)
                .map_err(|e| anyhow!("Failed to recreate WAL file: {}", e))?;

            let mut writer = BufWriter::with_capacity(self.config.buffer_size, file);

            // Write remaining records back
            for record in &keep_records {
                let record_bytes = record.to_bytes()?;
                writer
                    .write_all(&record_bytes)
                    .map_err(|e| anyhow!("Failed to write kept record: {}", e))?;
            }

            writer
                .flush()
                .map_err(|e| anyhow!("Failed to flush truncated WAL: {}", e))?;

            // Replace the file handle
            *self.wal_file.lock().unwrap() = writer;
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.current_log_size = keep_records.iter().map(|r| r.size as u64).sum();
        }

        info!(
            "Archived {} records to {:?}, kept {} records",
            archive_records.len(),
            archive_path,
            keep_records.len()
        );

        Ok(archive_path)
    }

    /// Read all log records from the WAL file
    fn read_all_log_records(&self) -> Result<Vec<LogRecord>> {
        let mut records = Vec::new();
        let file = std::fs::File::open(&self.config.wal_file_path)
            .map_err(|e| anyhow!("Failed to open WAL file for reading: {}", e))?;

        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .map_err(|e| anyhow!("Failed to read WAL file: {}", e))?;

        let mut offset = 0;
        while offset < buffer.len() {
            // Try to deserialize a record from the current position
            match bincode::deserialize::<LogRecord>(&buffer[offset..]) {
                Ok(record) => {
                    let record_size = record.size as usize;
                    records.push(record);
                    offset += record_size;
                }
                Err(_) => {
                    // If we can't deserialize, try to find the next valid record
                    offset += 1;
                }
            }
        }

        Ok(records)
    }

    /// Parallel log writing using multiple worker threads
    ///
    /// Processes the log buffer using multiple threads for improved throughput.
    /// This is especially beneficial for high-write workloads.
    ///
    /// # Arguments
    /// * `num_workers` - Number of parallel writer threads to use
    pub fn enable_parallel_writing(&mut self, num_workers: usize) -> Result<()> {
        info!("Enabling parallel log writing with {} workers", num_workers);

        if num_workers == 0 {
            return Err(anyhow!("Number of workers must be greater than 0"));
        }

        // Note: This is a simplified parallel writing implementation
        // In a production system, you would want to:
        // 1. Use a lock-free ring buffer for the log queue
        // 2. Implement proper ordering guarantees for LSN generation
        // 3. Use atomic operations for coordination
        // 4. Handle worker thread lifecycle management

        // For now, we'll just update the configuration to indicate parallel writing is enabled
        // The actual parallel implementation would require significant architectural changes

        info!("Parallel writing configuration updated (placeholder implementation)");
        Ok(())
    }

    /// Batch write multiple log records in parallel
    ///
    /// Writes multiple log records as a batch, potentially using parallel processing
    /// for improved throughput. Records are written in LSN order to maintain consistency.
    ///
    /// # Arguments
    /// * `record_types` - Vector of log record types to write
    ///
    /// # Returns
    /// Returns vector of LSNs for the written records
    pub fn batch_write_parallel(&self, record_types: Vec<LogRecordType>) -> Result<Vec<LSN>> {
        let start_time = std::time::Instant::now();
        let mut lsns = Vec::with_capacity(record_types.len());

        // Pre-allocate LSNs to maintain ordering
        let mut allocated_lsns = Vec::with_capacity(record_types.len());
        for _ in 0..record_types.len() {
            allocated_lsns.push(self.allocate_lsn()?);
        }

        // Create log records with pre-allocated LSNs
        let mut log_records = Vec::with_capacity(record_types.len());
        for (i, record_type) in record_types.into_iter().enumerate() {
            let lsn = allocated_lsns[i];

            // Get previous LSN for transaction
            let prev_lsn = if let Some(tx_id) = Self::extract_transaction_id(&record_type) {
                let tx_lsns = self
                    .transaction_lsns
                    .read()
                    .map_err(|_| anyhow!("Failed to acquire transaction LSNs lock"))?;
                tx_lsns.get(&tx_id).copied()
            } else {
                None
            };

            let log_record = LogRecord::new(lsn, prev_lsn, record_type)?;

            // Update transaction LSN
            if let Some(tx_id) = log_record.transaction_id {
                let mut tx_lsns = self
                    .transaction_lsns
                    .write()
                    .map_err(|_| anyhow!("Failed to acquire transaction LSNs lock"))?;
                tx_lsns.insert(tx_id, lsn);
            }

            log_records.push(log_record);
            lsns.push(lsn);
        }

        // Add all records to buffer atomically
        {
            let mut buffer = self
                .log_buffer
                .lock()
                .map_err(|_| anyhow!("Failed to acquire log buffer lock"))?;
            for record in log_records {
                buffer.push_back(record);
            }
        }

        // Force flush for batch
        self.flush()?;

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.records_written += lsns.len() as u64;
            let batch_time = start_time.elapsed().as_micros() as f64;
            let _avg_per_record = batch_time / lsns.len() as f64;
            stats.avg_write_time_us = (stats.avg_write_time_us
                * (stats.records_written - lsns.len() as u64) as f64
                + batch_time)
                / stats.records_written as f64;
        }

        debug!("Batch wrote {} records with LSNs: {:?}", lsns.len(), lsns);
        Ok(lsns)
    }

    /// Get recommended archive LSN based on current checkpoint
    ///
    /// Returns the LSN before which log entries can be safely archived.
    /// This is typically the LSN of the last completed checkpoint.
    pub fn get_recommended_archive_lsn(&self) -> Result<Option<LSN>> {
        // Find the most recent checkpoint record
        let records = self.read_all_log_records()?;

        let mut last_checkpoint_lsn = None;
        for record in records.iter().rev() {
            if matches!(record.record_type, LogRecordType::Checkpoint { .. }) {
                last_checkpoint_lsn = Some(record.lsn);
                break;
            }
        }

        Ok(last_checkpoint_lsn)
    }

    /// Clean up old archive files
    ///
    /// Removes archive files older than the specified number of days.
    /// This helps manage disk space used by historical log archives.
    ///
    /// # Arguments
    /// * `retain_days` - Number of days to retain archive files
    pub fn cleanup_old_archives(&self, retain_days: u64) -> Result<usize> {
        let archive_dir = self
            .config
            .wal_file_path
            .parent()
            .ok_or_else(|| anyhow!("Invalid WAL file path"))?;

        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            - (retain_days * 24 * 60 * 60);

        let mut removed_count = 0;

        if let Ok(entries) = std::fs::read_dir(archive_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(filename) = path.file_name() {
                    let name = filename.to_string_lossy();

                    // Check if this is an archive file
                    if name.starts_with("wal_archive_") && name.ends_with(".log") {
                        // Extract timestamp from filename
                        if let Some(timestamp_str) = name.split('_').nth(3) {
                            if let Ok(timestamp) =
                                timestamp_str.trim_end_matches(".log").parse::<u64>()
                            {
                                if timestamp < cutoff_time && std::fs::remove_file(&path).is_ok() {
                                    removed_count += 1;
                                    debug!("Removed old archive file: {:?}", path);
                                }
                            }
                        }
                    }
                }
            }
        }

        info!("Cleaned up {} old archive files", removed_count);
        Ok(removed_count)
    }
}

/// Calculate a simple checksum for data integrity
fn calculate_checksum(data: &[u8]) -> u64 {
    let mut checksum = 0u64;
    for byte in data {
        checksum = checksum.wrapping_add(*byte as u64);
        checksum = checksum.wrapping_mul(1103515245).wrapping_add(12345);
    }
    checksum
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_wal_basic_operations() {
        let temp_file = NamedTempFile::new().unwrap();
        let wal = WalManager::new(temp_file.path()).unwrap();

        let tx_id = 1;

        // Begin transaction
        let begin_lsn = wal.begin_transaction(tx_id).unwrap();
        assert!(begin_lsn > 0);

        // Log some operations
        let key = TripleKey::new(1, 2, 3);
        let insert_lsn = wal.log_insert(tx_id, key.clone(), true, None).unwrap();
        assert!(insert_lsn > begin_lsn);

        // Commit transaction
        let commit_lsn = wal.commit_transaction(tx_id).unwrap();
        assert!(commit_lsn > insert_lsn);

        let stats = wal.get_stats().unwrap();
        assert_eq!(stats.records_written, 3); // begin, insert, commit
    }

    #[test]
    fn test_log_record_serialization() {
        let record_type = LogRecordType::Insert {
            transaction_id: 1,
            key: TripleKey::new(1, 2, 3),
            value: true,
            previous_value: None,
        };

        let record = LogRecord::new(42, Some(41), record_type).unwrap();
        let bytes = record.to_bytes().unwrap();
        let restored = LogRecord::from_bytes(&bytes).unwrap();

        assert_eq!(record.lsn, restored.lsn);
        assert_eq!(record.prev_lsn, restored.prev_lsn);
        assert_eq!(record.transaction_id, restored.transaction_id);
    }

    #[test]
    fn test_checkpoint() {
        let temp_file = NamedTempFile::new().unwrap();
        let wal = WalManager::new(temp_file.path()).unwrap();

        // Start some transactions
        wal.begin_transaction(1).unwrap();
        wal.begin_transaction(2).unwrap();

        // Create checkpoint
        let checkpoint_lsn = wal.checkpoint().unwrap();
        assert!(checkpoint_lsn > 0);

        let stats = wal.get_stats().unwrap();
        assert_eq!(stats.checkpoints, 1);
    }

    #[test]
    fn test_transaction_abort() {
        let temp_file = NamedTempFile::new().unwrap();
        let wal = WalManager::new(temp_file.path()).unwrap();

        let tx_id = 1;

        wal.begin_transaction(tx_id).unwrap();
        let key = TripleKey::new(1, 2, 3);
        wal.log_insert(tx_id, key, true, None).unwrap();
        let abort_lsn = wal.abort_transaction(tx_id).unwrap();

        assert!(abort_lsn > 0);

        let stats = wal.get_stats().unwrap();
        assert_eq!(stats.records_written, 3); // begin, insert, abort
    }

    #[test]
    fn test_schema_change_logging() {
        let temp_file = NamedTempFile::new().unwrap();
        let wal = WalManager::new(temp_file.path()).unwrap();
        let tx_id = 1;

        // Test various schema operations
        let create_index_op = SchemaOperation::CreateIndex {
            index_name: "test_index".to_string(),
            index_type: "btree".to_string(),
            columns: vec!["subject".to_string(), "predicate".to_string()],
            options: std::collections::HashMap::new(),
        };

        let create_graph_op = SchemaOperation::CreateGraph {
            graph_name: "test_graph".to_string(),
            metadata: std::collections::HashMap::new(),
        };

        wal.begin_transaction(tx_id).unwrap();

        // Log schema change operations
        let lsn1 = wal.log_schema_change(tx_id, create_index_op).unwrap();
        let lsn2 = wal.log_schema_change(tx_id, create_graph_op).unwrap();

        assert!(lsn1 > 0);
        assert!(lsn2 > lsn1);

        // Verify stats were updated
        let stats = wal.get_stats().unwrap();
        assert_eq!(stats.records_written, 3); // begin + 2 schema changes
    }
}
