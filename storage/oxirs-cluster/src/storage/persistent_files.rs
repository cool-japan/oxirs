//! Crash-safe file writes for PersistentStorage.

use super::config::StorageConfig;
use super::types::ChecksummedData;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Raft state, written as [`ChecksummedData`].
pub(super) const RAFT_STATE_FILE: &str = "raft_state.dat";
/// Application state, written as [`ChecksummedData`].
pub(super) const APP_STATE_FILE: &str = "app_state.dat";

/// Seconds since the Unix epoch (0 if the clock is set before it).
pub(super) fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Replace `dir/name` with `bytes`, so that a crash leaves either the old or
/// the new contents. With `sync`, both the data and the rename are on disk
/// when this returns.
pub(super) fn write_atomic(dir: &Path, name: &str, bytes: &[u8], sync: bool) -> Result<()> {
    let path = dir.join(name);
    let temp = dir.join(format!("{name}.tmp"));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        if sync {
            file.sync_all()?;
        }
    }
    std::fs::rename(&temp, &path)?;
    if sync {
        sync_dir(dir)?;
    }
    Ok(())
}

/// Flush the directory entries of `dir` (renames, new files) to disk.
pub(super) fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    std::fs::File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Write `data` to `dir/name` as [`ChecksummedData`], with a checksum when
/// corruption detection is enabled.
pub(super) fn write_checksummed<T: Serialize>(
    dir: &Path,
    name: &str,
    data: &T,
    config: &StorageConfig,
) -> Result<()> {
    let wrapped = if config.enable_corruption_detection {
        ChecksummedData::new(data)?
    } else {
        ChecksummedData {
            data,
            checksum: String::new(),
            timestamp: unix_secs(),
        }
    };
    let bytes = oxicode::serde::encode_to_vec(&wrapped, oxicode::config::standard())?;
    write_atomic(dir, name, &bytes, config.sync_writes)
}

/// Read a file written by [`write_checksummed`], verifying its checksum when
/// corruption detection is enabled.
pub(super) fn read_checksummed<T>(path: &Path, config: &StorageConfig) -> Result<T>
where
    T: for<'de> Deserialize<'de> + Serialize,
{
    let bytes = std::fs::read(path)?;
    let (wrapped, _): (ChecksummedData<T>, _) =
        oxicode::serde::decode_from_slice(&bytes, oxicode::config::standard())?;
    if config.enable_corruption_detection && !wrapped.verify()? {
        return Err(anyhow!(
            "Checksum verification failed for {}",
            path.display()
        ));
    }
    Ok(wrapped.data)
}
