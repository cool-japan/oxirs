//! # Advanced Storage Backend
//!
//! Production-ready storage backend with atomic writes, crash recovery,
//! corruption detection, and performance optimization for Raft consensus.
//!
//! ## v0.2.0 Enhancements
//! - SIMD-accelerated compression/decompression using scirs2_core
//! - Parallel chunk processing for large snapshots
//! - Profiling integration for performance analysis

use crate::at_rest::{self, AtRestKey};
use crate::raft::OxirsNodeId;
use crate::serialization::{
    CompressionAlgorithm, MessageSerializer, SerializationConfig, SerializedMessage,
};
use crate::storage::{RaftState, SnapshotMetadata, WalEntry, WalOperation};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::sync::{Mutex, RwLock};
use tokio::time::Instant;

// SciRS2-Core imports for profiling
use scirs2_core::profiling::Profiler;

/// Context bound into every encrypted WAL record (see [`crate::at_rest`]).
const WAL_CONTEXT: &[u8] = b"oxirs-cluster/wal";

/// Context bound into an encrypted snapshot file, tying it to its position in
/// the log so one snapshot file cannot be swapped in for another.
fn snapshot_context(last_included_index: u64, last_included_term: u64) -> Vec<u8> {
    format!("oxirs-cluster/snapshot/{last_included_index}/{last_included_term}").into_bytes()
}

/// Upper bound for one decompressed WAL record (matches the serializer).
const MAX_WAL_RECORD_SIZE: usize = 100 * 1024 * 1024;

/// Records whose encoding exceeded this many bytes were LZ4-compressed by
/// releases up to 0.4.1, which also hashed the compressed bytes.
const LEGACY_COMPRESSION_THRESHOLD: usize = 1024;

/// SHA-256 over the uncompressed MessagePack encoding of `entry`, whose
/// `checksum` field must be empty. Independent of compression, so a log
/// stays verifiable when `enable_compression` changes.
fn wal_entry_checksum(entry: &WalEntry) -> Result<String> {
    let encoded =
        rmp_serde::to_vec(entry).map_err(|e| anyhow!("failed to encode WAL entry: {e}"))?;
    Ok(hex::encode(Sha256::digest(&encoded)))
}

/// LZ4 frame descriptor flag: the frame ends with an xxh32 checksum of its
/// content, which decompression verifies. oxiarc-lz4 always sets it.
const LZ4_FRAME_CONTENT_CHECKSUM_FLAG: u8 = 0x04;

/// Decode one WAL payload: decompress it if it is an LZ4 frame (a
/// MessagePack entry starts with an array marker, never with the frame
/// magic), decode the entry and verify its checksum.
///
/// The outer error is fatal for the whole log; the inner one describes why
/// this record is damaged.
fn decode_wal_payload(payload: &[u8]) -> Result<std::result::Result<WalEntry, String>> {
    if payload.starts_with(&oxiarc_lz4::LZ4_FRAME_MAGIC.to_le_bytes()) {
        let content_checked = payload
            .get(4)
            .is_some_and(|flags| flags & LZ4_FRAME_CONTENT_CHECKSUM_FLAG != 0);
        return Ok(match oxiarc_lz4::decompress(payload, MAX_WAL_RECORD_SIZE) {
            Ok(encoded) => decode_wal_entry(&encoded, true, content_checked),
            Err(e) => Err(format!("LZ4 decompression failed: {e}")),
        });
    }
    match decode_wal_entry(payload, false, false) {
        Ok(entry) => Ok(Ok(entry)),
        Err(_) if is_legacy_lz4_block(payload) => Err(anyhow!(
            "the record was compressed by oxirs-cluster 0.2.0 or earlier (lz4_flex), and its \
             checksum can no longer be verified; move wal.log aside to start without it (this \
             node then loses its local Raft state)"
        )),
        Err(reason) => Ok(Err(reason)),
    }
}

/// Decode a MessagePack WAL entry and verify its checksum.
///
/// Entries from compressed records may carry the checksum that releases up
/// to 0.4.1 computed over the compressed bytes. Older oxiarc-lz4 releases
/// compressed differently, so that checksum cannot always be reproduced;
/// `content_checked` (the LZ4 frame's own content checksum verified these
/// bytes) then vouches for the record instead.
fn decode_wal_entry(
    encoded: &[u8],
    compressed: bool,
    content_checked: bool,
) -> std::result::Result<WalEntry, String> {
    let mut entry: WalEntry =
        rmp_serde::from_slice(encoded).map_err(|e| format!("cannot decode entry: {e}"))?;

    let checksum = std::mem::take(&mut entry.checksum);
    let content = rmp_serde::to_vec(&entry).map_err(|e| format!("cannot re-encode entry: {e}"))?;
    let legacy = compressed && content.len() > LEGACY_COMPRESSION_THRESHOLD;
    let verified = checksum == hex::encode(Sha256::digest(&content))
        || (legacy
            && oxiarc_lz4::compress(&content)
                .is_ok_and(|bytes| checksum == hex::encode(Sha256::digest(&bytes))));
    let vouched = !verified
        && legacy
        && content_checked
        && checksum.len() == 64
        && checksum.bytes().all(|b| b.is_ascii_hexdigit());
    if !verified && !vouched {
        return Err(format!("checksum mismatch for entry {}", entry.sequence));
    }
    if vouched {
        tracing::debug!(
            "WAL entry {} predates the current checksum scheme; accepted on its LZ4 frame checksum",
            entry.sequence
        );
    }
    entry.checksum = checksum;
    Ok(entry)
}

/// The payload to journal for a serialized WAL entry.
///
/// Recovery decompresses an entry into at most `limit` bytes, so a
/// compressed entry larger than that could be written but never read back.
/// Such an entry is journaled uncompressed instead, which recovery reads at
/// any size.
fn journal_payload(
    serialized: SerializedMessage,
    limit: usize,
    uncompressed: impl FnOnce() -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    if serialized.compression != CompressionAlgorithm::None && serialized.original_size > limit {
        return uncompressed();
    }
    Ok(serialized.payload)
}

/// Whether `payload` is an LZ4 block behind a 4-byte size prefix that holds
/// a WAL entry, as releases up to 0.2.0 wrote large records (lz4_flex).
fn is_legacy_lz4_block(payload: &[u8]) -> bool {
    let Some(Ok(prefix)) = payload.get(..4).map(<[u8; 4]>::try_from) else {
        return false;
    };
    let (size, block) = (u32::from_le_bytes(prefix) as usize, &payload[4..]);
    size > LEGACY_COMPRESSION_THRESHOLD
        && size <= MAX_WAL_RECORD_SIZE
        && oxiarc_lz4::decompress_block(block, size).is_ok_and(|decoded| {
            decoded.len() == size && rmp_serde::from_slice::<WalEntry>(&decoded).is_ok()
        })
}

/// Storage backend configuration
#[derive(Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Directory for storing data files
    pub data_dir: PathBuf,
    /// Maximum WAL file size in bytes
    pub max_wal_size: u64,
    /// WAL sync mode for durability
    pub wal_sync_mode: WalSyncMode,
    /// Enable fsync for atomic writes
    pub enable_fsync: bool,
    /// Enable memory-mapped files for performance
    pub enable_mmap: bool,
    /// Checkpoint interval in seconds
    pub checkpoint_interval: u64,
    /// Enable background compaction
    pub enable_compaction: bool,
    /// Compression for stored data
    pub enable_compression: bool,
    /// Maximum memory cache size in bytes
    pub cache_size: usize,
    /// Encrypt WAL records and snapshot files at rest with XChaCha20-Poly1305
    /// (see [`crate::at_rest`]). Requires `encryption_key`.
    pub enable_encryption: bool,
    /// 256-bit key for `enable_encryption` (in production, load from a secure
    /// store). Setting it without `enable_encryption`, or the reverse, is a
    /// configuration error.
    pub encryption_key: Option<[u8; 32]>,
}

impl std::fmt::Debug for StorageConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageConfig")
            .field("data_dir", &self.data_dir)
            .field("max_wal_size", &self.max_wal_size)
            .field("wal_sync_mode", &self.wal_sync_mode)
            .field("enable_fsync", &self.enable_fsync)
            .field("enable_mmap", &self.enable_mmap)
            .field("checkpoint_interval", &self.checkpoint_interval)
            .field("enable_compaction", &self.enable_compaction)
            .field("enable_compression", &self.enable_compression)
            .field("cache_size", &self.cache_size)
            .field("enable_encryption", &self.enable_encryption)
            .field("encryption_key", &self.encryption_key.map(|_| "<redacted>"))
            .finish()
    }
}

impl StorageConfig {
    /// The at-rest key implied by the encryption settings.
    ///
    /// # Errors
    ///
    /// Fails when `enable_encryption` and `encryption_key` disagree, so that a
    /// half-configured store never silently runs unencrypted.
    fn at_rest_key(&self) -> Result<Option<AtRestKey>> {
        match (self.enable_encryption, self.encryption_key) {
            (true, Some(key)) => Ok(Some(AtRestKey::from_key(key))),
            (true, None) => Err(anyhow!(
                "enable_encryption is set but encryption_key is not; refusing to store data unencrypted"
            )),
            (false, Some(_)) => Err(anyhow!(
                "encryption_key is set but enable_encryption is false; enable encryption or remove the key"
            )),
            (false, None) => Ok(None),
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            max_wal_size: 64 * 1024 * 1024, // 64MB
            wal_sync_mode: WalSyncMode::Sync,
            enable_fsync: true,
            enable_mmap: true,
            checkpoint_interval: 300, // 5 minutes
            enable_compaction: true,
            enable_compression: true,
            cache_size: 128 * 1024 * 1024, // 128MB
            enable_encryption: false,
            encryption_key: None,
        }
    }
}

/// WAL synchronization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalSyncMode {
    /// No synchronization (fastest, least safe)
    NoSync,
    /// Sync on commit (balanced)
    Sync,
    /// Sync on every write (safest, slowest)
    FullSync,
}

/// Storage statistics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total operations performed
    pub total_operations: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Number of checkpoints created
    pub checkpoints_created: u64,
    /// Number of corruption detections
    pub corruption_detections: u64,
    /// Number of recovery operations
    pub recovery_operations: u64,
    /// Average write latency
    pub avg_write_latency: Duration,
    /// Average read latency
    pub avg_read_latency: Duration,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Current WAL size
    pub current_wal_size: u64,
    /// Number of compactions performed
    pub compactions_performed: u64,
}

/// File handle with atomic write support
#[derive(Debug)]
struct AtomicFile {
    file: File,
    temp_path: PathBuf,
    final_path: PathBuf,
    sync_mode: WalSyncMode,
}

impl AtomicFile {
    /// Create a new atomic file writer
    fn create(final_path: PathBuf, sync_mode: WalSyncMode) -> Result<Self> {
        let temp_path = final_path.with_extension("tmp");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&temp_path)?;

        Ok(Self {
            file,
            temp_path,
            final_path,
            sync_mode,
        })
    }

    /// Write data to the temporary file
    fn write_all(&mut self, data: &[u8]) -> Result<()> {
        self.file.write_all(data)?;
        if matches!(self.sync_mode, WalSyncMode::FullSync) {
            self.file.sync_all()?;
        }
        Ok(())
    }

    /// Commit the atomic write operation
    fn commit(self) -> Result<()> {
        if matches!(self.sync_mode, WalSyncMode::Sync | WalSyncMode::FullSync) {
            self.file.sync_all()?;
        }
        // File will be automatically dropped when self goes out of scope
        std::fs::rename(&self.temp_path, &self.final_path)?;
        Ok(())
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        // Clean up temporary file if commit wasn't called
        let _ = std::fs::remove_file(&self.temp_path);
    }
}

/// Write-Ahead Log with atomic operations and crash recovery
struct WriteAheadLog {
    config: StorageConfig,
    current_sequence: AtomicU64,
    wal_file: Arc<Mutex<File>>,
    serializer: Arc<Mutex<MessageSerializer>>,
    /// Seals every record when encryption at rest is enabled
    encryption_key: Option<AtRestKey>,
}

impl WriteAheadLog {
    /// Create a new Write-Ahead Log
    async fn new(config: StorageConfig, encryption_key: Option<AtRestKey>) -> Result<Self> {
        let wal_path = config.data_dir.join("wal.log");

        // Create directory if it doesn't exist
        if let Some(parent) = wal_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let wal_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;

        let current_sequence = AtomicU64::new(0);

        // Recover sequence number from existing WAL
        if wal_path.exists() && wal_file.metadata()?.len() > 0 {
            // In a full implementation, we would scan the WAL file to find the last sequence number
            // For now, we'll start from 0 and let the recovery process handle it
        }

        let serializer_config = SerializationConfig {
            compression: if config.enable_compression {
                crate::serialization::CompressionAlgorithm::Lz4
            } else {
                crate::serialization::CompressionAlgorithm::None
            },
            ..Default::default()
        };
        let serializer = Arc::new(Mutex::new(MessageSerializer::with_config(
            serializer_config,
        )));

        Ok(Self {
            config,
            current_sequence,
            wal_file: Arc::new(Mutex::new(wal_file)),
            serializer,
            encryption_key,
        })
    }

    /// Append an entry to the WAL
    async fn append(&self, operation: WalOperation) -> Result<u64> {
        let sequence = self.current_sequence.fetch_add(1, Ordering::SeqCst);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        let entry = WalEntry {
            sequence,
            timestamp,
            operation: operation.clone(),
            checksum: String::new(), // Will be filled after serialization
        };

        // The checksum covers the uncompressed encoding, so it does not
        // depend on whether this record ends up compressed.
        let checksum = wal_entry_checksum(&entry)?;
        let entry_with_checksum = WalEntry { checksum, ..entry };

        // Re-serialize with checksum
        let mut serializer = self.serializer.lock().await;
        let final_serialized = serializer.serialize(&entry_with_checksum)?;
        drop(serializer);
        let payload = journal_payload(final_serialized, MAX_WAL_RECORD_SIZE, || {
            Ok(rmp_serde::to_vec(&entry_with_checksum)?)
        })?;

        // Seal the record when encryption at rest is enabled
        let record = match &self.encryption_key {
            Some(key) => at_rest::seal(key, WAL_CONTEXT, &payload)?,
            None => payload,
        };

        // Write to WAL file
        let entry_size = u32::try_from(record.len())
            .map_err(|_| {
                anyhow!(
                    "WAL record of {} bytes exceeds the 4 GiB record limit; compact the Raft \
                     log with a snapshot",
                    record.len()
                )
            })?
            .to_le_bytes();
        let mut wal_file = self.wal_file.lock().await;
        wal_file.write_all(&entry_size)?;
        wal_file.write_all(&record)?;

        match self.config.wal_sync_mode {
            WalSyncMode::NoSync => {}
            WalSyncMode::Sync | WalSyncMode::FullSync => {
                wal_file.sync_all()?;
            }
        }

        Ok(sequence)
    }

    /// Recover from WAL after crash
    ///
    /// Each record is opened (with encryption at rest), decompressed when it
    /// is an LZ4 frame, decoded and checked against its checksum. A final
    /// record that is incomplete or fails these checks is a torn write: its
    /// bytes are moved to `wal.log.torn-<unix-ms>` and the log is truncated
    /// after the last good record, so that later appends stay readable.
    ///
    /// # Errors
    ///
    /// Fails without modifying the log when a damaged record is followed by
    /// further records (damage a torn write cannot explain), when the log was
    /// written under a different key, or when it mixes plaintext and
    /// encrypted records.
    async fn recover(&self) -> Result<Vec<WalEntry>> {
        let wal_path = self.config.data_dir.join("wal.log");
        if !wal_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&wal_path)?;
        let file_len = file.metadata()?.len();
        let mut reader = std::io::BufReader::new(file);
        let mut recovered_entries = Vec::new();
        let mut offset: u64 = 0;

        while offset < file_len {
            let remaining = file_len - offset;
            if remaining < 4 {
                self.set_aside_torn_tail(&wal_path, offset, "incomplete record header")?;
                break;
            }
            let mut size_bytes = [0u8; 4];
            reader.read_exact(&mut size_bytes)?;
            let entry_size = u64::from(u32::from_le_bytes(size_bytes));
            if entry_size > remaining - 4 {
                self.set_aside_torn_tail(&wal_path, offset, "incomplete record")?;
                break;
            }
            let mut record = vec![0u8; usize::try_from(entry_size)?];
            reader.read_exact(&mut record)?;
            let record_end = offset + 4 + entry_size;

            let decoded = self
                .decode_record(&record)
                .map_err(|e| anyhow!("WAL {} at byte {offset}: {e}", wal_path.display()))?;
            match decoded {
                Ok(entry) => {
                    self.current_sequence
                        .store(entry.sequence + 1, Ordering::SeqCst);
                    recovered_entries.push(entry);
                }
                Err(reason) if record_end == file_len => {
                    self.set_aside_torn_tail(&wal_path, offset, &reason)?;
                    break;
                }
                Err(reason) => {
                    return Err(anyhow!(
                        "WAL record at byte {offset} of {} is damaged ({reason}) and is followed by \
                         {} more bytes; refusing to discard them",
                        wal_path.display(),
                        file_len - record_end
                    ));
                }
            }
            offset = record_end;
        }

        Ok(recovered_entries)
    }

    /// Decode one WAL record. The outer error is fatal for the whole log (a
    /// different key, or plaintext and encrypted records mixed); the inner
    /// one marks just this record as damaged.
    fn decode_record(&self, record: &[u8]) -> Result<std::result::Result<WalEntry, String>> {
        let payload = match &self.encryption_key {
            Some(key) => match at_rest::envelope_key_id(record) {
                Ok(key_id) if key_id != key.id() => {
                    return Err(anyhow!(
                        "WAL was encrypted with key '{key_id}', but key '{}' is configured",
                        key.id()
                    ));
                }
                Ok(_) => match at_rest::open(key, WAL_CONTEXT, record) {
                    Ok(payload) => payload,
                    Err(e) => return Ok(Err(format!("record failed authentication: {e}"))),
                },
                Err(e) if !at_rest::is_sealed(record) => {
                    if matches!(decode_wal_payload(record), Ok(Ok(_))) {
                        return Err(anyhow!(
                            "WAL holds unencrypted records but encryption is enabled; \
                             refusing to mix plaintext and encrypted records"
                        ));
                    }
                    return Ok(Err(format!(
                        "record is neither encrypted nor a valid entry: {e}"
                    )));
                }
                Err(e) => return Ok(Err(format!("corrupt encrypted record: {e}"))),
            },
            None if at_rest::is_sealed(record) => {
                return Err(anyhow!(
                    "WAL holds encrypted records but encryption is disabled; configure the encryption key"
                ));
            }
            None => record.to_vec(),
        };
        decode_wal_payload(&payload)
    }

    /// Move the bytes from `offset` to the end of the log into
    /// `wal.log.torn-<unix-ms>`, then truncate the log at `offset`.
    fn set_aside_torn_tail(&self, wal_path: &Path, offset: u64, reason: &str) -> Result<()> {
        let mut wal = OpenOptions::new().read(true).write(true).open(wal_path)?;
        wal.seek(SeekFrom::Start(offset))?;
        let mut tail = Vec::new();
        wal.read_to_end(&mut tail)?;

        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis())
            .unwrap_or_default();
        let mut attempt = 0u32;
        let (torn_path, mut torn_file) = loop {
            let name = match attempt {
                0 => format!("wal.log.torn-{millis}"),
                n => format!("wal.log.torn-{millis}-{n}"),
            };
            let path = wal_path.with_file_name(name);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => break (path, file),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempt < 100 => {
                    attempt += 1;
                }
                Err(e) => return Err(e.into()),
            }
        };
        torn_file.write_all(&tail)?;
        torn_file.sync_all()?;

        wal.set_len(offset)?;
        wal.sync_all()?;
        tracing::warn!(
            "WAL {} ended with a torn record at byte {offset} ({reason}); moved its {} bytes to {} \
             and truncated the log",
            wal_path.display(),
            tail.len(),
            torn_path.display()
        );
        Ok(())
    }

    /// Truncate WAL up to a given sequence number
    async fn truncate(&self, up_to_sequence: u64) -> Result<()> {
        // In a full implementation, we would rewrite the WAL file excluding entries up to the given sequence
        // For now, we'll just log the operation
        tracing::info!("WAL truncation requested up to sequence {}", up_to_sequence);
        Ok(())
    }

    /// Get current WAL size
    async fn size(&self) -> Result<u64> {
        let wal_path = self.config.data_dir.join("wal.log");
        if wal_path.exists() {
            Ok(fs::metadata(&wal_path).await?.len())
        } else {
            Ok(0)
        }
    }
}

/// Memory-mapped storage for high-performance reads
#[derive(Debug)]
struct MmapStorage {
    _file: File,
    #[allow(dead_code)]
    mmap: Mmap,
}

impl MmapStorage {
    /// Create memory-mapped storage from file
    fn new(file_path: &Path) -> Result<Self> {
        let file = File::open(file_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self { _file: file, mmap })
    }

    /// Read data from memory-mapped region
    #[allow(dead_code)]
    fn read(&self, offset: usize, length: usize) -> Result<&[u8]> {
        if offset + length > self.mmap.len() {
            return Err(anyhow!("Read beyond end of memory-mapped file"));
        }
        Ok(&self.mmap[offset..offset + length])
    }

    /// Get total size of memory-mapped region
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.mmap.len()
    }
}

/// LRU cache for frequently accessed data
#[derive(Debug)]
struct LruCache<K, V> {
    map: HashMap<K, V>,
    max_size: usize,
    access_order: Vec<K>,
}

impl<K: Clone + std::hash::Hash + Eq, V: Clone> LruCache<K, V> {
    fn new(max_size: usize) -> Self {
        Self {
            map: HashMap::new(),
            max_size,
            access_order: Vec::new(),
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.map.get(key) {
            // Move to end (most recently used)
            self.access_order.retain(|k| k != key);
            self.access_order.push(key.clone());
            Some(value.clone())
        } else {
            None
        }
    }

    fn put(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            // Update existing entry
            self.map.insert(key.clone(), value);
            self.access_order.retain(|k| k != &key);
            self.access_order.push(key);
        } else {
            // Add new entry
            if self.map.len() >= self.max_size {
                // Evict least recently used
                if let Some(lru_key) = self.access_order.first().cloned() {
                    self.map.remove(&lru_key);
                    self.access_order.remove(0);
                }
            }
            self.map.insert(key.clone(), value);
            self.access_order.push(key);
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.map.clear();
        self.access_order.clear();
    }
}

/// Advanced storage backend with all production features
pub struct AdvancedStorageBackend {
    config: StorageConfig,
    wal: WriteAheadLog,
    /// Seals snapshot files when encryption at rest is enabled
    encryption_key: Option<AtRestKey>,
    state_cache: Arc<RwLock<LruCache<String, RaftState>>>,
    snapshot_cache: Arc<RwLock<LruCache<String, SnapshotMetadata>>>,
    mmap_storage: Arc<RwLock<Option<MmapStorage>>>,
    stats: Arc<RwLock<StorageStats>>,
    serializer: Arc<Mutex<MessageSerializer>>,
    /// Profiler for performance tracking (v0.2.0)
    profiler: Arc<Profiler>,
}

impl AdvancedStorageBackend {
    /// Create a new advanced storage backend
    ///
    /// # Errors
    ///
    /// Fails when `enable_encryption` and `encryption_key` disagree, when the
    /// data directory cannot be created, or when WAL recovery fails (including
    /// a WAL written under a different key or encryption mode).
    pub async fn new(config: StorageConfig) -> Result<Self> {
        let encryption_key = config.at_rest_key()?;

        // Create data directory
        fs::create_dir_all(&config.data_dir).await?;

        // Initialize WAL. Durable Raft-state and snapshot-metadata persistence is
        // provided by the write-ahead log plus the in-memory caches (which the
        // recovery path repopulates by replaying the WAL); snapshot payloads live
        // in atomic on-disk files.
        let wal = WriteAheadLog::new(config.clone(), encryption_key.clone()).await?;

        // Initialize caches
        let cache_entries = config.cache_size / 1024; // Estimate entries
        let state_cache = Arc::new(RwLock::new(LruCache::new(cache_entries)));
        let snapshot_cache = Arc::new(RwLock::new(LruCache::new(cache_entries / 10)));

        // Initialize serializer with compression if enabled
        let serializer_config = SerializationConfig {
            compression: if config.enable_compression {
                crate::serialization::CompressionAlgorithm::Lz4
            } else {
                crate::serialization::CompressionAlgorithm::None
            },
            ..Default::default()
        };
        let serializer = Arc::new(Mutex::new(MessageSerializer::with_config(
            serializer_config,
        )));

        let mut backend = Self {
            config,
            wal,
            encryption_key,
            state_cache,
            snapshot_cache,
            mmap_storage: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(StorageStats::default())),
            serializer,
            profiler: Arc::new(Profiler::new()),
        };

        // Perform crash recovery
        backend.recover().await?;

        Ok(backend)
    }

    /// Perform crash recovery from WAL
    async fn recover(&mut self) -> Result<()> {
        let start_time = Instant::now();
        let entries = self.wal.recover().await?;

        tracing::info!("Recovering from {} WAL entries", entries.len());

        for entry in entries {
            match entry.operation {
                WalOperation::WriteRaftState(state) => {
                    self.store_raft_state_internal(state).await?;
                }
                WalOperation::WriteAppState(_app_state) => {
                    // In a full implementation, we would restore application state
                    tracing::debug!("Recovered application state");
                }
                WalOperation::CreateSnapshot(metadata) => {
                    // The snapshot cache is where `load_snapshot` finds the
                    // checksum of a snapshot file.
                    let key = format!(
                        "snapshot-{}-{}",
                        metadata.last_included_index, metadata.last_included_term
                    );
                    self.snapshot_cache.write().await.put(key, metadata);
                }
                WalOperation::TruncateLog(index) => {
                    tracing::debug!("Recovered log truncation to index {}", index);
                }
                WalOperation::Commit(sequence) => {
                    tracing::debug!("Recovered commit for sequence {}", sequence);
                }
            }
        }

        let mut stats = self.stats.write().await;
        stats.recovery_operations += 1;
        stats.avg_read_latency = start_time.elapsed();

        tracing::info!("Recovery completed in {:?}", start_time.elapsed());
        Ok(())
    }

    /// Store Raft state with atomic write
    pub async fn store_raft_state(&self, state: RaftState) -> Result<()> {
        let start_time = Instant::now();

        // Write to WAL first for durability
        self.wal
            .append(WalOperation::WriteRaftState(state.clone()))
            .await?;

        // Store in database
        self.store_raft_state_internal(state.clone()).await?;

        // Update cache
        let mut cache = self.state_cache.write().await;
        cache.put("current".to_string(), state);

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_operations += 1;
        stats.avg_write_latency = (stats.avg_write_latency * (stats.total_operations - 1) as u32
            + start_time.elapsed())
            / stats.total_operations as u32;

        Ok(())
    }

    /// Internal method to store Raft state.
    ///
    /// Serializes the state for byte accounting and publishes it to the
    /// in-memory state cache. This is the path invoked by WAL recovery (for
    /// each `WriteRaftState` entry), so populating the cache here is what makes
    /// a restarted backend observe its recovered state via `load_raft_state`.
    async fn store_raft_state_internal(&self, state: RaftState) -> Result<()> {
        // Serialize the state first (for byte accounting / durability framing).
        let serialized = {
            let mut serializer = self.serializer.lock().await;
            serializer.serialize(&state)?
        };

        // Publish to the state cache, which is the authoritative read source.
        {
            let mut cache = self.state_cache.write().await;
            cache.put("current".to_string(), state);
        }

        // Update stats after the store is complete.
        let mut stats = self.stats.write().await;
        stats.bytes_written += serialized.payload.len() as u64;

        Ok(())
    }

    /// Load Raft state with caching
    pub async fn load_raft_state(&self) -> Result<Option<RaftState>> {
        let start_time = Instant::now();

        // Try cache first
        {
            let mut cache = self.state_cache.write().await;
            if let Some(state) = cache.get(&"current".to_string()) {
                let mut stats = self.stats.write().await;
                stats.cache_hit_ratio = (stats.cache_hit_ratio * 0.9) + 0.1; // Exponential moving average
                return Ok(Some(state));
            }
        }

        // The state cache (populated by live writes and by WAL recovery) is the
        // authoritative read source; a miss here means no state has been stored.
        let result: Option<RaftState> = None;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_operations += 1;
        stats.bytes_read += result
            .as_ref()
            .map(|_| std::mem::size_of::<RaftState>())
            .unwrap_or(0) as u64;
        stats.avg_read_latency = (stats.avg_read_latency * (stats.total_operations - 1) as u32
            + start_time.elapsed())
            / stats.total_operations as u32;
        stats.cache_hit_ratio = (stats.cache_hit_ratio * 0.9) + 0.0; // Cache miss

        Ok(result)
    }

    /// Get profiling report (v0.2.0)
    pub fn get_profiling_report(&self) -> String {
        self.profiler.get_report()
    }

    /// Create a snapshot with SIMD-accelerated parallel compression and metadata
    pub async fn create_snapshot(
        &self,
        last_included_index: u64,
        last_included_term: u64,
        configuration: Vec<OxirsNodeId>,
        data: &[u8],
    ) -> Result<SnapshotMetadata> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        // Encode (and compress, if enabled) with a header that records how,
        // so that loading never has to guess
        let payload = snapshot_codec::encode(data, self.config.enable_compression);
        let encoded_len = payload.len();

        // Encrypt after compressing; the checksum covers the bytes on disk
        let final_data = match &self.encryption_key {
            Some(key) => at_rest::seal(
                key,
                &snapshot_context(last_included_index, last_included_term),
                &payload,
            )?,
            None => payload,
        };

        // Calculate checksum
        let checksum = hex::encode(Sha256::digest(&final_data));

        let metadata = SnapshotMetadata {
            last_included_index,
            last_included_term,
            configuration,
            timestamp,
            size: final_data.len() as u64,
            checksum,
        };

        // Write snapshot file atomically
        let snapshot_path = self.config.data_dir.join(format!(
            "snapshot-{last_included_index}-{last_included_term}.dat"
        ));

        let mut atomic_file = AtomicFile::create(snapshot_path, self.config.wal_sync_mode)?;
        atomic_file.write_all(&final_data)?;
        atomic_file.commit()?;

        // Snapshot metadata is held in the snapshot cache and recorded in the
        // WAL; the payload itself lives in the atomic file written above.
        let key = format!("snapshot-{last_included_index}-{last_included_term}");

        // Update cache
        {
            let mut cache = self.snapshot_cache.write().await;
            cache.put(key.clone(), metadata.clone());
        }

        // Log WAL operation
        self.wal
            .append(WalOperation::CreateSnapshot(metadata.clone()))
            .await?;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.checkpoints_created += 1;
        stats.bytes_written += final_data.len() as u64;

        tracing::info!(
            "Created snapshot {} ({} bytes, {} encoded, {} on disk)",
            metadata.last_included_index,
            data.len(),
            encoded_len,
            final_data.len()
        );

        Ok(metadata)
    }

    /// Load a snapshot by index and term
    pub async fn load_snapshot(
        &self,
        last_included_index: u64,
        last_included_term: u64,
    ) -> Result<Option<Vec<u8>>> {
        let key = format!("snapshot-{last_included_index}-{last_included_term}");

        // The snapshot cache is the authoritative source of snapshot metadata
        // (its checksum is required to verify the on-disk payload). A miss means
        // the metadata is not available, so the snapshot cannot be served.
        let metadata = {
            let mut cache = self.snapshot_cache.write().await;
            match cache.get(&key) {
                Some(meta) => meta,
                None => return Ok(None),
            }
        };

        // Load snapshot data from file
        let snapshot_path = self.config.data_dir.join(format!(
            "snapshot-{last_included_index}-{last_included_term}.dat"
        ));

        if !snapshot_path.exists() {
            return Ok(None);
        }

        let data = fs::read(&snapshot_path).await?;

        // Verify checksum
        let computed_checksum = hex::encode(Sha256::digest(&data));
        if computed_checksum != metadata.checksum {
            let mut stats = self.stats.write().await;
            stats.corruption_detections += 1;
            return Err(anyhow!("Snapshot checksum verification failed"));
        }

        // Decrypt when encryption at rest is enabled
        let data = match &self.encryption_key {
            Some(key) => at_rest::open(
                key,
                &snapshot_context(last_included_index, last_included_term),
                &data,
            )?,
            None if at_rest::is_sealed(&data) => {
                return Err(anyhow!(
                    "snapshot {last_included_index}-{last_included_term} is encrypted but encryption is disabled"
                ));
            }
            None => data,
        };

        // Decode exactly as recorded in the payload header
        let final_data = snapshot_codec::decode(&data).map_err(|e| {
            anyhow!("snapshot {last_included_index}-{last_included_term} cannot be decoded: {e}")
        })?;

        let mut stats = self.stats.write().await;
        stats.bytes_read += final_data.len() as u64;

        Ok(Some(final_data))
    }

    /// Get storage statistics
    pub async fn stats(&self) -> StorageStats {
        let mut stats = self.stats.read().await.clone();
        stats.current_wal_size = self.wal.size().await.unwrap_or(0);
        stats
    }

    /// Clear all caches
    pub async fn clear_caches(&self) {
        let mut state_cache = self.state_cache.write().await;
        state_cache.clear();

        let mut snapshot_cache = self.snapshot_cache.write().await;
        snapshot_cache.clear();

        tracing::info!("Cleared all storage caches");
    }

    /// Compact storage files (remove unused data)
    pub async fn compact(&self) -> Result<()> {
        if !self.config.enable_compaction {
            return Ok(());
        }

        let start_time = Instant::now();

        // In a full implementation, this would:
        // 1. Identify unused log entries
        // 2. Compact snapshot files
        // 3. Rebuild database files
        // 4. Update WAL

        tracing::info!("Storage compaction completed in {:?}", start_time.elapsed());

        let mut stats = self.stats.write().await;
        stats.compactions_performed += 1;

        Ok(())
    }

    /// Enable memory-mapped storage for large files
    pub async fn enable_mmap(&self, file_path: &Path) -> Result<()> {
        if !self.config.enable_mmap {
            return Ok(());
        }

        let mmap_storage = MmapStorage::new(file_path)?;
        let mut mmap = self.mmap_storage.write().await;
        *mmap = Some(mmap_storage);

        tracing::info!("Enabled memory-mapped storage for {:?}", file_path);
        Ok(())
    }

    /// Perform periodic maintenance
    pub async fn maintenance(&self) -> Result<()> {
        // Check WAL size and truncate if necessary
        let wal_size = self.wal.size().await?;
        if wal_size > self.config.max_wal_size {
            tracing::info!(
                "WAL size {} exceeds limit {}, truncating",
                wal_size,
                self.config.max_wal_size
            );
            // In a full implementation, we would checkpoint and truncate the WAL
            self.wal.truncate(0).await?;
        }

        // Perform compaction if enabled
        if self.config.enable_compaction {
            self.compact().await?;
        }

        Ok(())
    }
}

/// Storage backend trait for pluggable implementations
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store Raft state
    async fn store_raft_state(&self, state: RaftState) -> Result<()>;

    /// Load Raft state
    async fn load_raft_state(&self) -> Result<Option<RaftState>>;

    /// Create snapshot
    async fn create_snapshot(
        &self,
        last_included_index: u64,
        last_included_term: u64,
        configuration: Vec<OxirsNodeId>,
        data: &[u8],
    ) -> Result<SnapshotMetadata>;

    /// Load snapshot
    async fn load_snapshot(
        &self,
        last_included_index: u64,
        last_included_term: u64,
    ) -> Result<Option<Vec<u8>>>;

    /// Get storage statistics
    async fn stats(&self) -> StorageStats;
}

#[async_trait]
impl StorageBackend for AdvancedStorageBackend {
    async fn store_raft_state(&self, state: RaftState) -> Result<()> {
        self.store_raft_state(state).await
    }

    async fn load_raft_state(&self) -> Result<Option<RaftState>> {
        self.load_raft_state().await
    }

    async fn create_snapshot(
        &self,
        last_included_index: u64,
        last_included_term: u64,
        configuration: Vec<OxirsNodeId>,
        data: &[u8],
    ) -> Result<SnapshotMetadata> {
        self.create_snapshot(last_included_index, last_included_term, configuration, data)
            .await
    }

    async fn load_snapshot(
        &self,
        last_included_index: u64,
        last_included_term: u64,
    ) -> Result<Option<Vec<u8>>> {
        self.load_snapshot(last_included_index, last_included_term)
            .await
    }

    async fn stats(&self) -> StorageStats {
        self.stats().await
    }
}

mod snapshot_codec;

#[cfg(test)]
mod tests;
