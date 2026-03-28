//! BackupEngine and IncrementalBackup for OxiRS TDB
//!
//! Provides triple-level backup/restore with format and compression options,
//! SHA-256 checksum verification, and delta-based incremental backups.

use crate::error::{Result, TdbError};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

// ─── Enumerations ────────────────────────────────────────────────────────────

/// Serialization format for backup data
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackupFormat {
    /// N-Quads text format
    NQuads,
    /// Turtle text format
    Turtle,
    /// JSON-LD format
    JsonLd,
    /// Binary (compact length-prefixed) format
    Binary,
}

/// Compression algorithm applied to backup data
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackupCompression {
    /// No compression
    None,
    /// Zstandard compression
    Zstd,
    /// Gzip compression
    Gzip,
}

// ─── BackupManifest ──────────────────────────────────────────────────────────

/// Metadata describing a completed backup
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackupManifest {
    /// Unique identifier for this backup
    pub backup_id: String,
    /// Unix timestamp (milliseconds) when backup was created
    pub created_at_ms: i64,
    /// Name of the dataset that was backed up
    pub dataset_name: String,
    /// Number of triples in the backup
    pub triple_count: usize,
    /// Serialization format used
    pub format: BackupFormat,
    /// Compression algorithm used
    pub compression: BackupCompression,
    /// SHA-256 hex digest of the (possibly compressed) backup data file
    pub checksum: String,
    /// Total size of the backup data file in bytes
    pub size_bytes: usize,
}

// ─── BackupEngine ────────────────────────────────────────────────────────────

/// Engine that creates, lists, restores, verifies and prunes backups.
///
/// Each backup is stored as a pair of files:
/// - `<id>.dat` — the (possibly compressed) triple data
/// - `<id>.json` — the [`BackupManifest`]
pub struct BackupEngine {
    /// Root directory where backup files are written
    backup_dir: PathBuf,
    /// Advisory limit: call [`prune_old`] to enforce
    max_backups: usize,
}

impl BackupEngine {
    /// Create a new `BackupEngine` rooted at `backup_dir`.
    pub fn new(backup_dir: PathBuf, max_backups: usize) -> Self {
        Self {
            backup_dir,
            max_backups,
        }
    }

    /// Create a backup of `triples` and write it to the backup directory.
    pub fn create_backup(
        &self,
        triples: &[(String, String, String)],
        dataset_name: &str,
        format: BackupFormat,
        compression: BackupCompression,
    ) -> Result<BackupManifest> {
        fs::create_dir_all(&self.backup_dir).map_err(TdbError::Io)?;

        let backup_id = Self::new_id();
        let raw = Self::serialize(triples, format)?;
        let data = Self::compress(raw, compression)?;
        let checksum = Self::sha256_hex(&data);
        let size_bytes = data.len();

        let data_path = self.backup_dir.join(format!("{}.dat", backup_id));
        fs::write(&data_path, &data).map_err(TdbError::Io)?;

        let manifest = BackupManifest {
            backup_id: backup_id.clone(),
            created_at_ms: Self::now_ms(),
            dataset_name: dataset_name.to_string(),
            triple_count: triples.len(),
            format,
            compression,
            checksum,
            size_bytes,
        };

        let manifest_path = self.backup_dir.join(format!("{}.json", backup_id));
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| TdbError::Serialization(e.to_string()))?;
        fs::write(&manifest_path, json).map_err(TdbError::Io)?;

        Ok(manifest)
    }

    /// List all manifests in the backup directory, sorted newest-first.
    pub fn list_backups(&self) -> Vec<BackupManifest> {
        let mut manifests = Vec::new();
        let entries = match fs::read_dir(&self.backup_dir) {
            Ok(e) => e,
            Err(_) => return manifests,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(json) = fs::read_to_string(&path) {
                    if let Ok(m) = serde_json::from_str::<BackupManifest>(&json) {
                        manifests.push(m);
                    }
                }
            }
        }
        manifests.sort_by_key(|b| std::cmp::Reverse(b.created_at_ms));
        manifests
    }

    /// Restore triples from the backup identified by `backup_id`.
    pub fn restore(&self, backup_id: &str) -> Result<Vec<(String, String, String)>> {
        let manifest = self.load_manifest(backup_id)?;
        let data_path = self.backup_dir.join(format!("{}.dat", backup_id));
        let data = fs::read(&data_path).map_err(TdbError::Io)?;
        let raw = Self::decompress(data, manifest.compression)?;
        Self::deserialize(&raw, manifest.format)
    }

    /// Verify a backup by re-computing its SHA-256 checksum.
    ///
    /// Returns `true` if the stored checksum matches.
    pub fn verify(&self, backup_id: &str) -> Result<bool> {
        let manifest = self.load_manifest(backup_id)?;
        let data_path = self.backup_dir.join(format!("{}.dat", backup_id));
        let data = fs::read(&data_path).map_err(TdbError::Io)?;
        Ok(Self::sha256_hex(&data) == manifest.checksum)
    }

    /// Delete the backup identified by `backup_id`.
    pub fn delete_backup(&self, backup_id: &str) -> Result<()> {
        let data_path = self.backup_dir.join(format!("{}.dat", backup_id));
        let manifest_path = self.backup_dir.join(format!("{}.json", backup_id));
        if data_path.exists() {
            fs::remove_file(&data_path).map_err(TdbError::Io)?;
        }
        if manifest_path.exists() {
            fs::remove_file(&manifest_path).map_err(TdbError::Io)?;
        }
        Ok(())
    }

    /// Delete the oldest backups, keeping only the `keep_count` most recent.
    ///
    /// Returns the number of backups deleted.
    pub fn prune_old(&self, keep_count: usize) -> usize {
        let manifests = self.list_backups(); // sorted newest-first
        if manifests.len() <= keep_count {
            return 0;
        }
        let mut deleted = 0usize;
        for m in &manifests[keep_count..] {
            if self.delete_backup(&m.backup_id).is_ok() {
                deleted += 1;
            }
        }
        deleted
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn load_manifest(&self, backup_id: &str) -> Result<BackupManifest> {
        let path = self.backup_dir.join(format!("{}.json", backup_id));
        let json = fs::read_to_string(&path).map_err(TdbError::Io)?;
        serde_json::from_str(&json).map_err(|e| TdbError::Deserialization(e.to_string()))
    }

    fn new_id() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        format!("bkp_{}_{}_{}", t.as_millis(), t.subsec_nanos(), seq)
    }

    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    fn sha256_hex(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    fn serialize(triples: &[(String, String, String)], format: BackupFormat) -> Result<Vec<u8>> {
        match format {
            BackupFormat::NQuads | BackupFormat::Turtle => {
                let mut out = String::new();
                for (s, p, o) in triples {
                    out.push_str(&format!("<{}> <{}> <{}> .\n", s, p, o));
                }
                Ok(out.into_bytes())
            }
            BackupFormat::JsonLd => {
                let mut items = Vec::new();
                for (s, p, o) in triples {
                    items.push(serde_json::json!({
                        "@id": s,
                        p: [{ "@id": o }]
                    }));
                }
                serde_json::to_vec(&items).map_err(|e| TdbError::Serialization(e.to_string()))
            }
            BackupFormat::Binary => {
                // Layout: u64-LE count, then each triple as three u32-LE-prefixed UTF-8 strings
                let mut buf = Vec::new();
                buf.extend_from_slice(&(triples.len() as u64).to_le_bytes());
                for (s, p, o) in triples {
                    for part in &[s, p, o] {
                        let bytes = part.as_bytes();
                        buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        buf.extend_from_slice(bytes);
                    }
                }
                Ok(buf)
            }
        }
    }

    fn deserialize(data: &[u8], format: BackupFormat) -> Result<Vec<(String, String, String)>> {
        match format {
            BackupFormat::NQuads | BackupFormat::Turtle => {
                let text = std::str::from_utf8(data)
                    .map_err(|e| TdbError::Deserialization(e.to_string()))?;
                let mut triples = Vec::new();
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    // Parse "<s> <p> <o> ."
                    let parts: Vec<&str> = line.splitn(4, "> <").collect();
                    if parts.len() < 3 {
                        continue;
                    }
                    let s = parts[0].trim_start_matches('<').to_string();
                    let p = parts[1].to_string();
                    let o = parts[2]
                        .trim_end_matches("> .")
                        .trim_end_matches(" .")
                        .trim_end_matches('>')
                        .to_string();
                    triples.push((s, p, o));
                }
                Ok(triples)
            }
            BackupFormat::JsonLd => {
                let items: Vec<serde_json::Value> = serde_json::from_slice(data)
                    .map_err(|e| TdbError::Deserialization(e.to_string()))?;
                let mut triples = Vec::new();
                for item in items {
                    let s = item["@id"].as_str().unwrap_or("").to_string();
                    if let Some(obj) = item.as_object() {
                        for (key, val) in obj {
                            if key == "@id" {
                                continue;
                            }
                            if let Some(arr) = val.as_array() {
                                for v in arr {
                                    let o = v["@id"].as_str().unwrap_or("").to_string();
                                    triples.push((s.clone(), key.clone(), o));
                                }
                            }
                        }
                    }
                }
                Ok(triples)
            }
            BackupFormat::Binary => {
                if data.len() < 8 {
                    return Err(TdbError::Deserialization(
                        "Binary data too short".to_string(),
                    ));
                }
                let count = u64::from_le_bytes(
                    data[..8]
                        .try_into()
                        .map_err(|_| TdbError::Deserialization("count slice".to_string()))?,
                ) as usize;
                let mut pos = 8usize;
                let mut triples = Vec::with_capacity(count);
                for _ in 0..count {
                    let s = read_str_at(data, &mut pos)?;
                    let p = read_str_at(data, &mut pos)?;
                    let o = read_str_at(data, &mut pos)?;
                    triples.push((s, p, o));
                }
                Ok(triples)
            }
        }
    }

    fn compress(data: Vec<u8>, compression: BackupCompression) -> Result<Vec<u8>> {
        match compression {
            BackupCompression::None => Ok(data),
            BackupCompression::Zstd => oxiarc_zstd::encode_all(&data, 3)
                .map_err(|e| TdbError::Other(format!("zstd compress: {}", e))),
            BackupCompression::Gzip => {
                use flate2::write::GzEncoder;
                use flate2::Compression;
                use std::io::Write;
                let mut enc = GzEncoder::new(Vec::new(), Compression::default());
                enc.write_all(&data)
                    .map_err(|e| TdbError::Other(format!("gzip write: {}", e)))?;
                enc.finish()
                    .map_err(|e| TdbError::Other(format!("gzip finish: {}", e)))
            }
        }
    }

    fn decompress(data: Vec<u8>, compression: BackupCompression) -> Result<Vec<u8>> {
        match compression {
            BackupCompression::None => Ok(data),
            BackupCompression::Zstd => oxiarc_zstd::decode_all(&data)
                .map_err(|e| TdbError::Other(format!("zstd decompress: {}", e))),
            BackupCompression::Gzip => {
                use flate2::read::GzDecoder;
                use std::io::Read;
                let mut dec = GzDecoder::new(&data[..]);
                let mut out = Vec::new();
                dec.read_to_end(&mut out)
                    .map_err(|e| TdbError::Other(format!("gzip decode: {}", e)))?;
                Ok(out)
            }
        }
    }
}

/// Read a u32-length-prefixed UTF-8 string from `data` at `pos`, advancing `pos`.
fn read_str_at(data: &[u8], pos: &mut usize) -> Result<String> {
    if *pos + 4 > data.len() {
        return Err(TdbError::Deserialization("EOF reading length".to_string()));
    }
    let len = u32::from_le_bytes(
        data[*pos..*pos + 4]
            .try_into()
            .map_err(|_| TdbError::Deserialization("len slice".to_string()))?,
    ) as usize;
    *pos += 4;
    if *pos + len > data.len() {
        return Err(TdbError::Deserialization("EOF reading string".to_string()));
    }
    let s = std::str::from_utf8(&data[*pos..*pos + len])
        .map_err(|e| TdbError::Deserialization(e.to_string()))?
        .to_string();
    *pos += len;
    Ok(s)
}

// ─── IncrementalBackup ───────────────────────────────────────────────────────

/// A single delta (set of added and removed triples) within an incremental backup.
#[derive(Debug, Clone)]
pub struct BackupDelta {
    /// Unique identifier for this delta
    pub delta_id: String,
    /// Unix timestamp (milliseconds) when the delta was created
    pub created_at_ms: i64,
    /// Triples added in this delta
    pub added: Vec<(String, String, String)>,
    /// Triples removed in this delta
    pub removed: Vec<(String, String, String)>,
}

/// An incremental backup layered on top of a full base backup.
///
/// Deltas are applied in order to reconstruct the current dataset state.
pub struct IncrementalBackup {
    /// The base full-backup manifest
    pub base_manifest: BackupManifest,
    /// Ordered list of deltas applied since the base backup
    pub deltas: Vec<BackupDelta>,
}

impl IncrementalBackup {
    /// Create a new incremental backup anchored to `base`.
    pub fn new(base: BackupManifest) -> Self {
        Self {
            base_manifest: base,
            deltas: Vec::new(),
        }
    }

    /// Append a new delta and return a reference to it.
    pub fn add_delta(
        &mut self,
        added: Vec<(String, String, String)>,
        removed: Vec<(String, String, String)>,
    ) -> &BackupDelta {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let delta = BackupDelta {
            delta_id: format!("delta_{}_{}_{}", t.as_millis(), t.subsec_nanos(), seq),
            created_at_ms: t.as_millis() as i64,
            added,
            removed,
        };
        self.deltas.push(delta);
        self.deltas.last().expect("just pushed")
    }

    /// Reconstruct the full triple set by replaying all deltas in order.
    ///
    /// This applies additions and removals cumulatively across all deltas.
    pub fn reconstruct(&self) -> Vec<(String, String, String)> {
        let mut current: std::collections::HashSet<(String, String, String)> =
            std::collections::HashSet::new();
        for delta in &self.deltas {
            for t in &delta.removed {
                current.remove(t);
            }
            for t in &delta.added {
                current.insert(t.clone());
            }
        }
        current.into_iter().collect()
    }

    /// Return the number of deltas recorded so far.
    pub fn delta_count(&self) -> usize {
        self.deltas.len()
    }

    /// Return the total number of triple operations (adds + removes) across all deltas.
    pub fn total_size(&self) -> usize {
        self.deltas
            .iter()
            .map(|d| d.added.len() + d.removed.len())
            .sum()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples(n: usize) -> Vec<(String, String, String)> {
        (0..n)
            .map(|i| {
                (
                    format!("http://ex.org/s{}", i),
                    format!("http://ex.org/p{}", i),
                    format!("http://ex.org/o{}", i),
                )
            })
            .collect()
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oxirs_tdb_bkpeng_{}", name));
        fs::remove_dir_all(&d).ok();
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn dummy_manifest() -> BackupManifest {
        BackupManifest {
            backup_id: "base_001".to_string(),
            created_at_ms: 0,
            dataset_name: "test".to_string(),
            triple_count: 0,
            format: BackupFormat::NQuads,
            compression: BackupCompression::None,
            checksum: "abc123".to_string(),
            size_bytes: 0,
        }
    }

    // ── BackupEngine ─────────────────────────────────────────────────────────

    #[test]
    fn test_backup_engine_new() {
        let dir = tmp_dir("new");
        let engine = BackupEngine::new(dir.clone(), 5);
        assert_eq!(engine.max_backups, 5);
        assert_eq!(engine.backup_dir, dir);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_create_backup_nquads_none() {
        let dir = tmp_dir("create_nq");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(5);
        let m = engine
            .create_backup(
                &triples,
                "test_ds",
                BackupFormat::NQuads,
                BackupCompression::None,
            )
            .unwrap();
        assert_eq!(m.triple_count, 5);
        assert_eq!(m.format, BackupFormat::NQuads);
        assert_eq!(m.compression, BackupCompression::None);
        assert!(!m.checksum.is_empty());
        assert!(m.size_bytes > 0);
        assert!(!m.backup_id.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_create_backup_binary_zstd() {
        let dir = tmp_dir("create_bin_zstd");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(10);
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::Binary,
                BackupCompression::Zstd,
            )
            .unwrap();
        assert_eq!(m.triple_count, 10);
        assert_eq!(m.format, BackupFormat::Binary);
        assert_eq!(m.compression, BackupCompression::Zstd);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_create_backup_jsonld_gzip() {
        let dir = tmp_dir("create_jsonld_gz");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(3);
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::JsonLd,
                BackupCompression::Gzip,
            )
            .unwrap();
        assert_eq!(m.format, BackupFormat::JsonLd);
        assert_eq!(m.compression, BackupCompression::Gzip);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_create_backup_turtle_none() {
        let dir = tmp_dir("create_turtle");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(2);
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::Turtle,
                BackupCompression::None,
            )
            .unwrap();
        assert_eq!(m.format, BackupFormat::Turtle);
        assert_eq!(m.triple_count, 2);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_restore_nquads() {
        let dir = tmp_dir("restore_nq");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(4);
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::NQuads,
                BackupCompression::None,
            )
            .unwrap();
        let restored = engine.restore(&m.backup_id).unwrap();
        assert_eq!(restored.len(), 4);
        for t in &triples {
            assert!(restored.contains(t), "missing triple {:?}", t);
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_restore_binary_zstd() {
        let dir = tmp_dir("restore_bin_zstd");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(7);
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::Binary,
                BackupCompression::Zstd,
            )
            .unwrap();
        let restored = engine.restore(&m.backup_id).unwrap();
        assert_eq!(restored.len(), 7);
        for t in &triples {
            assert!(restored.contains(t));
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_restore_binary_gzip() {
        let dir = tmp_dir("restore_bin_gz");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(6);
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::Binary,
                BackupCompression::Gzip,
            )
            .unwrap();
        let restored = engine.restore(&m.backup_id).unwrap();
        assert_eq!(restored.len(), 6);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_intact() {
        let dir = tmp_dir("verify_ok");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(3);
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::NQuads,
                BackupCompression::None,
            )
            .unwrap();
        assert!(engine.verify(&m.backup_id).unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_corrupted() {
        let dir = tmp_dir("verify_corrupt");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples = sample_triples(3);
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::NQuads,
                BackupCompression::None,
            )
            .unwrap();
        // Corrupt the data file
        fs::write(dir.join(format!("{}.dat", m.backup_id)), b"corrupted").unwrap();
        assert!(!engine.verify(&m.backup_id).unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_list_backups_empty() {
        let dir = tmp_dir("list_empty");
        let engine = BackupEngine::new(dir.clone(), 10);
        assert!(engine.list_backups().is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_list_backups_multiple_sorted() {
        let dir = tmp_dir("list_multi");
        let engine = BackupEngine::new(dir.clone(), 10);
        let t = sample_triples(2);
        engine
            .create_backup(&t, "ds", BackupFormat::NQuads, BackupCompression::None)
            .unwrap();
        engine
            .create_backup(&t, "ds", BackupFormat::Binary, BackupCompression::None)
            .unwrap();
        let list = engine.list_backups();
        assert_eq!(list.len(), 2);
        assert!(list[0].created_at_ms >= list[1].created_at_ms);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_backup() {
        let dir = tmp_dir("delete");
        let engine = BackupEngine::new(dir.clone(), 10);
        let t = sample_triples(2);
        let m = engine
            .create_backup(&t, "ds", BackupFormat::NQuads, BackupCompression::None)
            .unwrap();
        engine.delete_backup(&m.backup_id).unwrap();
        assert!(engine.list_backups().is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_nonexistent_is_ok() {
        let dir = tmp_dir("delete_ne");
        let engine = BackupEngine::new(dir.clone(), 10);
        // Should not return error for non-existent backup
        engine.delete_backup("no_such_id").unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_prune_old_removes_oldest() {
        let dir = tmp_dir("prune");
        let engine = BackupEngine::new(dir.clone(), 10);
        let t = sample_triples(1);
        for _ in 0..5 {
            engine
                .create_backup(&t, "ds", BackupFormat::NQuads, BackupCompression::None)
                .unwrap();
        }
        assert_eq!(engine.list_backups().len(), 5);
        let removed = engine.prune_old(3);
        assert_eq!(removed, 2);
        assert_eq!(engine.list_backups().len(), 3);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_prune_no_removal_needed() {
        let dir = tmp_dir("prune_noop");
        let engine = BackupEngine::new(dir.clone(), 10);
        let t = sample_triples(1);
        engine
            .create_backup(&t, "ds", BackupFormat::NQuads, BackupCompression::None)
            .unwrap();
        let removed = engine.prune_old(5);
        assert_eq!(removed, 0);
        assert_eq!(engine.list_backups().len(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_empty_triples_backup_restore() {
        let dir = tmp_dir("empty_triples");
        let engine = BackupEngine::new(dir.clone(), 10);
        let triples: Vec<(String, String, String)> = vec![];
        let m = engine
            .create_backup(
                &triples,
                "ds",
                BackupFormat::NQuads,
                BackupCompression::None,
            )
            .unwrap();
        assert_eq!(m.triple_count, 0);
        let restored = engine.restore(&m.backup_id).unwrap();
        assert!(restored.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_dataset_name_preserved_in_manifest() {
        let dir = tmp_dir("ds_name");
        let engine = BackupEngine::new(dir.clone(), 10);
        let t = sample_triples(1);
        let m = engine
            .create_backup(
                &t,
                "my_dataset",
                BackupFormat::Binary,
                BackupCompression::None,
            )
            .unwrap();
        assert_eq!(m.dataset_name, "my_dataset");
        let list = engine.list_backups();
        assert_eq!(list[0].dataset_name, "my_dataset");
        fs::remove_dir_all(&dir).ok();
    }

    // ── IncrementalBackup ────────────────────────────────────────────────────

    #[test]
    fn test_incremental_new() {
        let ib = IncrementalBackup::new(dummy_manifest());
        assert_eq!(ib.base_manifest.backup_id, "base_001");
        assert_eq!(ib.delta_count(), 0);
        assert_eq!(ib.total_size(), 0);
    }

    #[test]
    fn test_add_delta_returns_ref() {
        let mut ib = IncrementalBackup::new(dummy_manifest());
        let delta = ib.add_delta(
            vec![("s".to_string(), "p".to_string(), "o".to_string())],
            vec![],
        );
        assert!(!delta.delta_id.is_empty());
        assert_eq!(delta.added.len(), 1);
        assert_eq!(delta.removed.len(), 0);
    }

    #[test]
    fn test_delta_count_multiple() {
        let mut ib = IncrementalBackup::new(dummy_manifest());
        ib.add_delta(vec![], vec![]);
        ib.add_delta(vec![], vec![]);
        assert_eq!(ib.delta_count(), 2);
    }

    #[test]
    fn test_total_size_sums_adds_and_removes() {
        let mut ib = IncrementalBackup::new(dummy_manifest());
        // Delta 1: 1 add + 1 remove = 2 operations
        ib.add_delta(
            vec![("s1".to_string(), "p".to_string(), "o1".to_string())],
            vec![("s2".to_string(), "p".to_string(), "o2".to_string())],
        );
        // Delta 2: 2 adds + 0 removes = 2 operations
        ib.add_delta(
            vec![
                ("s3".to_string(), "p".to_string(), "o3".to_string()),
                ("s4".to_string(), "p".to_string(), "o4".to_string()),
            ],
            vec![],
        );
        // Total: 2 + 2 = 4
        assert_eq!(ib.total_size(), 4);
    }

    #[test]
    fn test_reconstruct_add_only() {
        let mut ib = IncrementalBackup::new(dummy_manifest());
        let t1 = ("a".to_string(), "b".to_string(), "c".to_string());
        let t2 = ("d".to_string(), "e".to_string(), "f".to_string());
        ib.add_delta(vec![t1.clone(), t2.clone()], vec![]);
        let result = ib.reconstruct();
        assert_eq!(result.len(), 2);
        assert!(result.contains(&t1));
        assert!(result.contains(&t2));
    }

    #[test]
    fn test_reconstruct_add_then_remove() {
        let mut ib = IncrementalBackup::new(dummy_manifest());
        let t1 = ("s1".to_string(), "p".to_string(), "o1".to_string());
        let t2 = ("s2".to_string(), "p".to_string(), "o2".to_string());
        ib.add_delta(vec![t1.clone(), t2.clone()], vec![]);
        ib.add_delta(vec![], vec![t1.clone()]);
        let result = ib.reconstruct();
        assert_eq!(result.len(), 1);
        assert!(result.contains(&t2));
        assert!(!result.contains(&t1));
    }

    #[test]
    fn test_reconstruct_empty() {
        let ib = IncrementalBackup::new(dummy_manifest());
        assert!(ib.reconstruct().is_empty());
    }

    #[test]
    fn test_delta_timestamps_are_positive() {
        let mut ib = IncrementalBackup::new(dummy_manifest());
        let delta = ib.add_delta(vec![], vec![]);
        assert!(delta.created_at_ms > 0);
    }
}
