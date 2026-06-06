//! WAL (Write-Ahead Log) operations for PersistentStorage

use super::config::StorageConfig;
use super::types::*;

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use crate::raft::OxirsNodeId;

/// WAL state carried by PersistentStorage (exposed for impl blocks in this module)
#[allow(dead_code)]
pub(super) struct WalContext<'a> {
    pub data_dir: &'a PathBuf,
    pub node_id: OxirsNodeId,
    pub config: &'a StorageConfig,
    pub wal_sequence: &'a Arc<RwLock<u64>>,
    pub wal_writer: &'a Arc<RwLock<Option<BufWriter<File>>>>,
}

/// Initialize Write-Ahead Log
pub(super) async fn init_wal(ctx: WalContext<'_>) -> Result<()> {
    let wal_path = ctx.data_dir.join("wal.log");

    let mut sequence = 0;
    if wal_path.exists() {
        sequence = get_last_wal_sequence(&wal_path).await?;
    }

    let wal_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&wal_path)?;

    let writer = BufWriter::new(wal_file);
    *ctx.wal_writer.write().await = Some(writer);
    *ctx.wal_sequence.write().await = sequence;

    tracing::info!(
        "Initialized WAL for node {} at sequence {}",
        ctx.node_id,
        sequence
    );
    Ok(())
}

/// Get the last sequence number from WAL file
pub(super) async fn get_last_wal_sequence(wal_path: &Path) -> Result<u64> {
    let file = File::open(wal_path)?;
    let mut reader = BufReader::new(file);
    let mut last_sequence = 0;

    loop {
        let mut length_bytes = [0u8; 8];
        match reader.read_exact(&mut length_bytes) {
            Ok(_) => {
                let length = u64::from_le_bytes(length_bytes);

                if length > 100 * 1024 * 1024 {
                    tracing::warn!("WAL entry length too large: {}, skipping", length);
                    break;
                }

                let mut entry_bytes = vec![0u8; length as usize];
                match reader.read_exact(&mut entry_bytes) {
                    Ok(_) => {
                        if let Ok(entry) = oxicode::serde::decode_from_slice(
                            &entry_bytes,
                            oxicode::config::standard(),
                        )
                        .map(|(v, _): (WalEntry, _)| v)
                        {
                            last_sequence = entry.sequence;
                        }
                    }
                    Err(_) => break,
                }
            }
            Err(_) => break,
        }
    }

    Ok(last_sequence)
}

/// Write entry to WAL
pub(super) async fn write_wal_entry(
    config: &StorageConfig,
    wal_sequence: &Arc<RwLock<u64>>,
    wal_writer: &Arc<RwLock<Option<BufWriter<File>>>>,
    operation: WalOperation,
) -> Result<()> {
    if !config.enable_wal {
        return Ok(());
    }

    let mut wal_sequence_guard = wal_sequence.write().await;
    *wal_sequence_guard += 1;
    let sequence = *wal_sequence_guard;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime should be after UNIX_EPOCH")
        .as_secs();

    let op_bytes = oxicode::serde::encode_to_vec(&operation, oxicode::config::standard())?;
    let mut hasher = Sha256::new();
    hasher.update(&op_bytes);
    hasher.update(sequence.to_le_bytes());
    hasher.update(timestamp.to_le_bytes());
    let checksum = hex::encode(hasher.finalize());

    let wal_entry = WalEntry {
        sequence,
        timestamp,
        operation,
        checksum,
    };

    let entry_bytes = oxicode::serde::encode_to_vec(&wal_entry, oxicode::config::standard())?;
    let length = entry_bytes.len() as u64;

    if let Some(ref mut writer) = wal_writer.write().await.as_mut() {
        writer.write_all(&length.to_le_bytes())?;
        writer.write_all(&entry_bytes)?;
        writer.flush()?;

        if config.sync_writes {
            writer.get_mut().sync_all()?;
        }
    }

    Ok(())
}

/// Verify WAL entry checksum
pub(super) fn verify_wal_entry_checksum(entry: &WalEntry) -> Result<bool> {
    let op_bytes = oxicode::serde::encode_to_vec(&entry.operation, oxicode::config::standard())?;
    let mut hasher = Sha256::new();
    hasher.update(&op_bytes);
    hasher.update(entry.sequence.to_le_bytes());
    hasher.update(entry.timestamp.to_le_bytes());
    let computed_checksum = hex::encode(hasher.finalize());
    Ok(computed_checksum == entry.checksum)
}

/// Rotate WAL file when it gets too large
pub(super) async fn rotate_wal(
    config: &StorageConfig,
    data_dir: &PathBuf,
    node_id: OxirsNodeId,
    wal_sequence: &Arc<RwLock<u64>>,
    wal_writer: &Arc<RwLock<Option<BufWriter<File>>>>,
) -> Result<()> {
    if !config.enable_wal {
        return Ok(());
    }

    let wal_path = data_dir.join("wal.log");
    if !wal_path.exists() {
        return Ok(());
    }

    let metadata = std::fs::metadata(&wal_path)?;
    if metadata.len() < 100 * 1024 * 1024 {
        return Ok(());
    }

    *wal_writer.write().await = None;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime should be after UNIX_EPOCH")
        .as_secs();
    let archive_path = data_dir.join(format!("wal-{timestamp}.log"));
    std::fs::rename(&wal_path, &archive_path)?;

    let ctx = WalContext {
        data_dir,
        node_id,
        config,
        wal_sequence,
        wal_writer,
    };
    init_wal(ctx).await?;

    tracing::info!(
        "Rotated WAL for node {}, archived to {:?}",
        node_id,
        archive_path
    );
    Ok(())
}

/// Compact WAL by removing committed entries
pub(super) async fn compact_wal(
    config: &StorageConfig,
    data_dir: &PathBuf,
    node_id: OxirsNodeId,
    wal_sequence: &Arc<RwLock<u64>>,
    wal_writer: &Arc<RwLock<Option<BufWriter<File>>>>,
) -> Result<()> {
    if !config.enable_wal {
        return Ok(());
    }

    let wal_path = data_dir.join("wal.log");
    if !wal_path.exists() {
        return Ok(());
    }

    let file = File::open(&wal_path)?;
    let mut reader = BufReader::new(file);
    let mut entries = Vec::new();
    let mut last_commit_sequence = 0;

    loop {
        let mut length_bytes = [0u8; 8];
        match reader.read_exact(&mut length_bytes) {
            Ok(_) => {
                let length = u64::from_le_bytes(length_bytes);
                let mut entry_bytes = vec![0u8; length as usize];
                reader.read_exact(&mut entry_bytes)?;

                if let Ok(entry) =
                    oxicode::serde::decode_from_slice(&entry_bytes, oxicode::config::standard())
                        .map(|(v, _): (WalEntry, _)| v)
                {
                    if let WalOperation::Commit(seq) = &entry.operation {
                        last_commit_sequence = *seq;
                    }
                    entries.push(entry);
                }
            }
            Err(_) => break,
        }
    }

    let total_entries = entries.len();
    let uncommitted: Vec<_> = entries
        .into_iter()
        .filter(|entry| entry.sequence > last_commit_sequence)
        .collect();

    let temp_wal_path = data_dir.join("wal.log.tmp");
    {
        let temp_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_wal_path)?;

        let mut writer = BufWriter::new(temp_file);

        for entry in &uncommitted {
            let entry_bytes = oxicode::serde::encode_to_vec(entry, oxicode::config::standard())?;
            let length = entry_bytes.len() as u64;
            writer.write_all(&length.to_le_bytes())?;
            writer.write_all(&entry_bytes)?;
        }

        writer.flush()?;
        if config.sync_writes {
            writer.get_mut().sync_all()?;
        }
    }

    std::fs::rename(&temp_wal_path, &wal_path)?;

    *wal_writer.write().await = None;
    let ctx = WalContext {
        data_dir,
        node_id,
        config,
        wal_sequence,
        wal_writer,
    };
    init_wal(ctx).await?;

    tracing::info!(
        "Compacted WAL for node {}, removed {} committed entries, kept {} uncommitted",
        node_id,
        total_entries - uncommitted.len(),
        uncommitted.len()
    );

    Ok(())
}

/// Verify WAL file integrity
pub(super) async fn verify_wal_integrity(wal_path: &Path) -> Result<usize> {
    use anyhow::anyhow;

    let file = File::open(wal_path)?;
    let mut reader = BufReader::new(file);
    let mut valid_entries = 0;

    loop {
        let mut length_bytes = [0u8; 8];
        match reader.read_exact(&mut length_bytes) {
            Ok(_) => {
                let length = u64::from_le_bytes(length_bytes);
                let mut entry_bytes = vec![0u8; length as usize];
                reader.read_exact(&mut entry_bytes)?;

                if let Ok(entry) =
                    oxicode::serde::decode_from_slice(&entry_bytes, oxicode::config::standard())
                        .map(|(v, _): (WalEntry, _)| v)
                {
                    if verify_wal_entry_checksum(&entry)? {
                        valid_entries += 1;
                    } else {
                        return Err(anyhow!(
                            "Invalid checksum for WAL entry at sequence {}",
                            entry.sequence
                        ));
                    }
                } else {
                    return Err(anyhow!("Failed to deserialize WAL entry"));
                }
            }
            Err(_) => break,
        }
    }

    Ok(valid_entries)
}
