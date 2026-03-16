//! HNSW Index Serialization / Persistence
//!
//! This module provides binary serialization and file-based persistence for
//! Hierarchical Navigable Small World (HNSW) index snapshots.
//!
//! # Format
//!
//! The binary format is a custom little-endian encoding (no bincode):
//!
//! ```text
//! Magic:     4 bytes  = b"HNSW"
//! Version:   2 bytes  = u16 LE
//! CRC32:     4 bytes  = u32 LE (checksum of all following bytes)
//! meta_len:  4 bytes  = u32 LE (length of serialized HnswMeta)
//! meta_data: N bytes
//! layers:    zero or more layer records:
//!   layer_level: 4 bytes = u32 LE
//!   node_count:  4 bytes = u32 LE
//!   nodes:       node_count × LayerNode records:
//!     node_id:        8 bytes = u64 LE
//!     neighbor_count: 4 bytes = u32 LE
//!     neighbors:      neighbor_count × 8 bytes (u64 LE each)
//! ```
//!
//! # Atomic file writes
//!
//! `save_to_file` writes to a `.tmp` file first, then renames into place —
//! ensuring that the destination file is never partially written.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

// ─── CRC32 implementation ─────────────────────────────────────────────────────
//
// Simple table-driven CRC32 (ISO 3309 / IEEE 802.3 polynomial 0xEDB88320).

const CRC32_POLY: u32 = 0xEDB8_8320;

/// Build the 256-entry CRC32 look-up table.
fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0u32..256 {
        let mut c = i;
        for _ in 0..8 {
            if c & 1 != 0 {
                c = CRC32_POLY ^ (c >> 1);
            } else {
                c >>= 1;
            }
        }
        table[i as usize] = c;
    }
    table
}

/// Compute the CRC32 checksum of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let table = build_crc32_table();
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = table[idx] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

// ─── Domain types ─────────────────────────────────────────────────────────────

/// HNSW index metadata / configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswMeta {
    /// Maximum layer index (0-based; layer 0 is the base layer)
    pub max_layer: usize,
    /// Entry-point node ID (None if the index is empty)
    pub entry_point: Option<u64>,
    /// `ef_construction` parameter
    pub ef_construction: usize,
    /// `M` parameter (number of bidirectional links per node)
    pub m: usize,
    /// Total number of distinct nodes across all layers
    pub node_count: usize,
}

/// A single node in one HNSW layer: its ID and its neighbour list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerNode {
    /// Unique node identifier
    pub id: u64,
    /// Neighbours (up to `M` per layer)
    pub neighbors: Vec<u64>,
}

/// One layer of the HNSW graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswLayer {
    /// Layer level (0 = base)
    pub level: usize,
    /// Nodes in this layer
    pub nodes: Vec<LayerNode>,
}

/// A complete snapshot of an HNSW index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswSnapshot {
    /// Index metadata
    pub meta: HnswMeta,
    /// Layer data (ordered by level)
    pub layers: Vec<HnswLayer>,
}

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Error from HNSW persistence operations.
#[derive(Debug)]
pub enum PersistError {
    /// I/O failure (wrapped)
    Io(io::Error),
    /// Magic bytes mismatch
    BadMagic,
    /// Unsupported format version
    UnsupportedVersion(u16),
    /// CRC32 checksum mismatch
    ChecksumMismatch,
    /// Data is truncated or malformed
    Malformed(String),
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::BadMagic => write!(f, "bad magic bytes (expected 'HNSW')"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::ChecksumMismatch => write!(f, "CRC32 checksum mismatch"),
            Self::Malformed(s) => write!(f, "malformed data: {s}"),
        }
    }
}

impl std::error::Error for PersistError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for PersistError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ─── Format constants ─────────────────────────────────────────────────────────

const MAGIC: &[u8; 4] = b"HNSW";
const FORMAT_VERSION: u16 = 1;

// ─── Low-level read / write helpers ──────────────────────────────────────────

fn write_u16_le(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64_le(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn read_u16_le(data: &[u8], pos: &mut usize) -> Result<u16, PersistError> {
    if *pos + 2 > data.len() {
        return Err(PersistError::Malformed("unexpected end (u16)".into()));
    }
    let v = u16::from_le_bytes(data[*pos..*pos + 2].try_into().expect("slice of 2"));
    *pos += 2;
    Ok(v)
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, PersistError> {
    if *pos + 4 > data.len() {
        return Err(PersistError::Malformed("unexpected end (u32)".into()));
    }
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().expect("slice of 4"));
    *pos += 4;
    Ok(v)
}

fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64, PersistError> {
    if *pos + 8 > data.len() {
        return Err(PersistError::Malformed("unexpected end (u64)".into()));
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().expect("slice of 8"));
    *pos += 8;
    Ok(v)
}

// ─── Meta serialization ────────────────────────────────────────────────────────

fn serialize_meta(meta: &HnswMeta) -> Vec<u8> {
    let mut buf = Vec::new();
    write_u64_le(&mut buf, meta.max_layer as u64);
    match meta.entry_point {
        Some(ep) => {
            write_u32_le(&mut buf, 1);
            write_u64_le(&mut buf, ep);
        }
        None => {
            write_u32_le(&mut buf, 0);
            write_u64_le(&mut buf, 0);
        }
    }
    write_u64_le(&mut buf, meta.ef_construction as u64);
    write_u64_le(&mut buf, meta.m as u64);
    write_u64_le(&mut buf, meta.node_count as u64);
    buf
}

fn deserialize_meta(data: &[u8], pos: &mut usize) -> Result<HnswMeta, PersistError> {
    let max_layer = read_u64_le(data, pos)? as usize;
    let has_ep = read_u32_le(data, pos)?;
    let ep_raw = read_u64_le(data, pos)?;
    let entry_point = if has_ep == 1 { Some(ep_raw) } else { None };
    let ef_construction = read_u64_le(data, pos)? as usize;
    let m = read_u64_le(data, pos)? as usize;
    let node_count = read_u64_le(data, pos)? as usize;
    Ok(HnswMeta {
        max_layer,
        entry_point,
        ef_construction,
        m,
        node_count,
    })
}

// ─── HnswPersistence ─────────────────────────────────────────────────────────

/// Provides serialization and file persistence for HNSW index snapshots.
#[derive(Debug, Default, Clone)]
pub struct HnswPersistence;

impl HnswPersistence {
    /// Create a new persistence instance.
    pub fn new() -> Self {
        Self
    }

    /// Serialize `meta` + `layers` into a byte vector.
    ///
    /// The format is described in the module documentation.
    pub fn serialize(&self, meta: &HnswMeta, layers: &[HnswLayer]) -> Vec<u8> {
        // Serialize meta
        let meta_bytes = serialize_meta(meta);

        // Serialize layers
        let mut layer_bytes: Vec<u8> = Vec::new();
        for layer in layers {
            write_u32_le(&mut layer_bytes, layer.level as u32);
            write_u32_le(&mut layer_bytes, layer.nodes.len() as u32);
            for node in &layer.nodes {
                write_u64_le(&mut layer_bytes, node.id);
                write_u32_le(&mut layer_bytes, node.neighbors.len() as u32);
                for &nb in &node.neighbors {
                    write_u64_le(&mut layer_bytes, nb);
                }
            }
        }

        // Build the payload (meta + layers) for checksumming
        let mut payload: Vec<u8> = Vec::new();
        write_u32_le(&mut payload, meta_bytes.len() as u32);
        payload.extend_from_slice(&meta_bytes);
        payload.extend_from_slice(&layer_bytes);

        let checksum = crc32(&payload);

        // Build the final buffer
        let mut buf: Vec<u8> = Vec::with_capacity(10 + payload.len());
        buf.extend_from_slice(MAGIC);
        write_u16_le(&mut buf, FORMAT_VERSION);
        write_u32_le(&mut buf, checksum);
        buf.extend_from_slice(&payload);

        buf
    }

    /// Deserialize bytes into `(HnswMeta, Vec<HnswLayer>)`.
    pub fn deserialize(&self, bytes: &[u8]) -> Result<(HnswMeta, Vec<HnswLayer>), PersistError> {
        if bytes.len() < 10 {
            return Err(PersistError::Malformed("too short".into()));
        }

        // Magic
        if &bytes[..4] != MAGIC {
            return Err(PersistError::BadMagic);
        }
        let mut pos = 4;

        // Version
        let version = read_u16_le(bytes, &mut pos)?;
        if version != FORMAT_VERSION {
            return Err(PersistError::UnsupportedVersion(version));
        }

        // Checksum
        let stored_crc = read_u32_le(bytes, &mut pos)?;
        let computed_crc = crc32(&bytes[pos..]);
        if stored_crc != computed_crc {
            return Err(PersistError::ChecksumMismatch);
        }

        // Meta
        let meta_len = read_u32_le(bytes, &mut pos)? as usize;
        if pos + meta_len > bytes.len() {
            return Err(PersistError::Malformed("meta_len out of bounds".into()));
        }
        let meta = deserialize_meta(bytes, &mut pos)?;

        // Layers
        let mut layers = Vec::new();
        while pos < bytes.len() {
            let level = read_u32_le(bytes, &mut pos)? as usize;
            let node_count = read_u32_le(bytes, &mut pos)?;
            let mut nodes = Vec::with_capacity(node_count as usize);
            for _ in 0..node_count {
                let id = read_u64_le(bytes, &mut pos)?;
                let neighbor_count = read_u32_le(bytes, &mut pos)?;
                let mut neighbors = Vec::with_capacity(neighbor_count as usize);
                for _ in 0..neighbor_count {
                    neighbors.push(read_u64_le(bytes, &mut pos)?);
                }
                nodes.push(LayerNode { id, neighbors });
            }
            layers.push(HnswLayer { level, nodes });
        }

        Ok((meta, layers))
    }

    /// Write a snapshot to a file atomically (write → temp, rename).
    pub fn save_to_file(
        &self,
        path: &Path,
        meta: &HnswMeta,
        layers: &[HnswLayer],
    ) -> Result<(), PersistError> {
        let bytes = self.serialize(meta, layers);

        // Write to a temporary file alongside the destination
        let mut tmp_path = path.to_path_buf();
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("hnsw");
        tmp_path.set_file_name(format!(".{file_name}.tmp"));

        {
            let mut file = std::fs::File::create(&tmp_path)?;
            file.write_all(&bytes)?;
            file.flush()?;
        }

        // Atomic rename
        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }

    /// Read a snapshot from a file.
    pub fn load_from_file(&self, path: &Path) -> Result<(HnswMeta, Vec<HnswLayer>), PersistError> {
        let bytes = std::fs::read(path)?;
        self.deserialize(&bytes)
    }

    /// Validate the CRC32 checksum of serialized bytes.
    ///
    /// Returns `true` if the checksum field matches the computed checksum
    /// of the payload, `false` otherwise.
    pub fn validate_checksum(&self, bytes: &[u8]) -> bool {
        if bytes.len() < 10 {
            return false;
        }
        if &bytes[..4] != MAGIC {
            return false;
        }
        let stored_crc = u32::from_le_bytes(bytes[6..10].try_into().expect("4 bytes"));
        let computed_crc = crc32(&bytes[10..]);
        stored_crc == computed_crc
    }

    /// Merge a base snapshot with a delta snapshot.
    ///
    /// Merge strategy:
    /// - `meta` is taken from `delta` (delta always has the newer configuration)
    /// - For each layer in delta, its nodes override/supplement the base layer nodes
    ///   (keyed by `node.id`)
    /// - Layers present only in base are preserved; layers only in delta are added
    pub fn merge_snapshots(&self, base: &HnswSnapshot, delta: &HnswSnapshot) -> HnswSnapshot {
        // Index base layers by level
        let mut layer_map: HashMap<usize, HashMap<u64, LayerNode>> = HashMap::new();
        for layer in &base.layers {
            let node_map: HashMap<u64, LayerNode> =
                layer.nodes.iter().map(|n| (n.id, n.clone())).collect();
            layer_map.insert(layer.level, node_map);
        }

        // Apply delta layers
        for layer in &delta.layers {
            let node_map = layer_map.entry(layer.level).or_default();
            for node in &layer.nodes {
                node_map.insert(node.id, node.clone());
            }
        }

        // Reconstruct ordered layers
        let mut levels: Vec<usize> = layer_map.keys().copied().collect();
        levels.sort();

        let merged_layers: Vec<HnswLayer> = levels
            .into_iter()
            .map(|level| {
                let node_map = &layer_map[&level];
                let mut nodes: Vec<LayerNode> = node_map.values().cloned().collect();
                nodes.sort_by_key(|n| n.id);
                HnswLayer { level, nodes }
            })
            .collect();

        HnswSnapshot {
            meta: delta.meta.clone(),
            layers: merged_layers,
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use std::env::temp_dir;
    use std::path::PathBuf;

    fn sample_meta() -> HnswMeta {
        HnswMeta {
            max_layer: 3,
            entry_point: Some(42),
            ef_construction: 200,
            m: 16,
            node_count: 1000,
        }
    }

    fn sample_layer(level: usize, node_ids: &[u64]) -> HnswLayer {
        let nodes: Vec<LayerNode> = node_ids
            .iter()
            .map(|&id| LayerNode {
                id,
                neighbors: node_ids
                    .iter()
                    .copied()
                    .filter(|&n| n != id)
                    .take(4)
                    .collect(),
            })
            .collect();
        HnswLayer { level, nodes }
    }

    fn persist() -> HnswPersistence {
        HnswPersistence::new()
    }

    fn tmp_path(name: &str) -> PathBuf {
        temp_dir().join(format!("hnsw_test_{name}.bin"))
    }

    // ── CRC32 ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_crc32_empty() {
        let c = crc32(&[]);
        // Standard CRC32 of empty input = 0x00000000
        assert_eq!(c, 0x0000_0000);
    }

    #[test]
    fn test_crc32_hello() {
        // CRC32 of "hello" = 0x3610a686
        let c = crc32(b"hello");
        assert_eq!(c, 0x3610_a686);
    }

    #[test]
    fn test_crc32_deterministic() {
        let data = b"deterministic test data";
        assert_eq!(crc32(data), crc32(data));
    }

    #[test]
    fn test_crc32_different_data() {
        assert_ne!(crc32(b"foo"), crc32(b"bar"));
    }

    // ── HnswMeta ─────────────────────────────────────────────────────────────

    #[test]
    fn test_meta_roundtrip() {
        let meta = sample_meta();
        let bytes = serialize_meta(&meta);
        let mut pos = 0;
        let decoded = deserialize_meta(&bytes, &mut pos).expect("ok");
        assert_eq!(decoded, meta);
    }

    #[test]
    fn test_meta_no_entry_point() {
        let meta = HnswMeta {
            max_layer: 0,
            entry_point: None,
            ef_construction: 100,
            m: 8,
            node_count: 0,
        };
        let bytes = serialize_meta(&meta);
        let mut pos = 0;
        let decoded = deserialize_meta(&bytes, &mut pos).expect("ok");
        assert_eq!(decoded.entry_point, None);
    }

    // ── Serialize / Deserialize ───────────────────────────────────────────────

    #[test]
    fn test_serialize_deserialize_empty() {
        let meta = sample_meta();
        let layers: Vec<HnswLayer> = vec![];
        let bytes = persist().serialize(&meta, &layers);
        let (decoded_meta, decoded_layers) = persist().deserialize(&bytes).expect("ok");
        assert_eq!(decoded_meta, meta);
        assert!(decoded_layers.is_empty());
    }

    #[test]
    fn test_serialize_deserialize_single_layer() {
        let meta = sample_meta();
        let layer = sample_layer(0, &[1, 2, 3, 4, 5]);
        let bytes = persist().serialize(&meta, std::slice::from_ref(&layer));
        let (_, layers) = persist().deserialize(&bytes).expect("ok");
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].level, 0);
        assert_eq!(layers[0].nodes.len(), 5);
    }

    #[test]
    fn test_serialize_deserialize_multiple_layers() {
        let meta = sample_meta();
        let l0 = sample_layer(0, &[1, 2, 3]);
        let l1 = sample_layer(1, &[1, 2]);
        let l2 = sample_layer(2, &[1]);
        let bytes = persist().serialize(&meta, &[l0, l1, l2]);
        let (_, layers) = persist().deserialize(&bytes).expect("ok");
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].level, 0);
        assert_eq!(layers[1].level, 1);
        assert_eq!(layers[2].level, 2);
    }

    #[test]
    fn test_serialize_deserialize_neighbors() {
        let meta = sample_meta();
        let layer = HnswLayer {
            level: 0,
            nodes: vec![LayerNode {
                id: 42,
                neighbors: vec![1, 2, 3],
            }],
        };
        let bytes = persist().serialize(&meta, &[layer]);
        let (_, layers) = persist().deserialize(&bytes).expect("ok");
        assert_eq!(layers[0].nodes[0].id, 42);
        assert_eq!(layers[0].nodes[0].neighbors, vec![1, 2, 3]);
    }

    #[test]
    fn test_serialize_magic() {
        let meta = sample_meta();
        let bytes = persist().serialize(&meta, &[]);
        assert_eq!(&bytes[..4], b"HNSW");
    }

    #[test]
    fn test_serialize_version() {
        let meta = sample_meta();
        let bytes = persist().serialize(&meta, &[]);
        let version = u16::from_le_bytes(bytes[4..6].try_into().expect("2 bytes"));
        assert_eq!(version, FORMAT_VERSION);
    }

    #[test]
    fn test_deserialize_bad_magic() {
        let mut bytes = persist().serialize(&sample_meta(), &[]);
        bytes[0] = b'X';
        let result = persist().deserialize(&bytes);
        assert!(matches!(result, Err(PersistError::BadMagic)));
    }

    #[test]
    fn test_deserialize_checksum_mismatch() {
        let mut bytes = persist().serialize(&sample_meta(), &[]);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF; // corrupt last byte
        let result = persist().deserialize(&bytes);
        assert!(matches!(result, Err(PersistError::ChecksumMismatch)));
    }

    #[test]
    fn test_deserialize_too_short() {
        let result = persist().deserialize(&[0u8; 5]);
        assert!(matches!(result, Err(PersistError::Malformed(_))));
    }

    #[test]
    fn test_deserialize_unsupported_version() {
        let mut bytes = persist().serialize(&sample_meta(), &[]);
        // Overwrite version field with 99
        bytes[4] = 99;
        bytes[5] = 0;
        // Recompute checksum
        let checksum = crc32(&bytes[10..]);
        bytes[6..10].copy_from_slice(&checksum.to_le_bytes());
        let result = persist().deserialize(&bytes);
        assert!(matches!(result, Err(PersistError::UnsupportedVersion(99))));
    }

    // ── validate_checksum ─────────────────────────────────────────────────────

    #[test]
    fn test_validate_checksum_valid() {
        let bytes = persist().serialize(&sample_meta(), &[]);
        assert!(persist().validate_checksum(&bytes));
    }

    #[test]
    fn test_validate_checksum_corrupted() {
        let mut bytes = persist().serialize(&sample_meta(), &[]);
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert!(!persist().validate_checksum(&bytes));
    }

    #[test]
    fn test_validate_checksum_too_short() {
        assert!(!persist().validate_checksum(&[0u8; 5]));
    }

    #[test]
    fn test_validate_checksum_bad_magic() {
        let mut bytes = persist().serialize(&sample_meta(), &[]);
        bytes[0] = b'X';
        assert!(!persist().validate_checksum(&bytes));
    }

    // ── File I/O ──────────────────────────────────────────────────────────────

    #[test]
    fn test_save_and_load() {
        let path = tmp_path("save_load");
        let meta = sample_meta();
        let layer = sample_layer(0, &[10, 20, 30]);

        persist()
            .save_to_file(&path, &meta, std::slice::from_ref(&layer))
            .expect("save ok");

        let (loaded_meta, loaded_layers) = persist().load_from_file(&path).expect("load ok");

        assert_eq!(loaded_meta, meta);
        assert_eq!(loaded_layers.len(), 1);
        assert_eq!(loaded_layers[0].nodes.len(), layer.nodes.len());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_atomic_no_tmp_left() -> Result<()> {
        let path = tmp_path("atomic_no_tmp");
        persist()
            .save_to_file(&path, &sample_meta(), &[])
            .expect("save ok");

        // Temporary file should not exist
        let mut tmp = path.clone();
        let name = path
            .file_name()
            .expect("path has a file name")
            .to_str()
            .expect("file name is valid UTF-8");
        tmp.set_file_name(format!(".{name}.tmp"));
        assert!(!tmp.exists(), "temp file should have been cleaned up");

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn test_load_nonexistent_file() {
        let path = tmp_path("nonexistent_xyz_12345");
        let result = persist().load_from_file(&path);
        assert!(matches!(result, Err(PersistError::Io(_))));
    }

    // ── merge_snapshots ────────────────────────────────────────────────────────

    #[test]
    fn test_merge_empty_delta() {
        let base = HnswSnapshot {
            meta: sample_meta(),
            layers: vec![sample_layer(0, &[1, 2, 3])],
        };
        let delta = HnswSnapshot {
            meta: HnswMeta {
                max_layer: 3,
                entry_point: Some(1),
                ef_construction: 200,
                m: 16,
                node_count: 4,
            },
            layers: vec![],
        };
        let merged = persist().merge_snapshots(&base, &delta);
        // Meta from delta
        assert_eq!(merged.meta.node_count, 4);
        // Layer from base preserved
        assert_eq!(merged.layers.len(), 1);
        assert_eq!(merged.layers[0].nodes.len(), 3);
    }

    #[test]
    fn test_merge_delta_adds_nodes() {
        let base = HnswSnapshot {
            meta: sample_meta(),
            layers: vec![sample_layer(0, &[1, 2, 3])],
        };
        let delta = HnswSnapshot {
            meta: HnswMeta {
                node_count: 5,
                ..sample_meta()
            },
            layers: vec![HnswLayer {
                level: 0,
                nodes: vec![
                    LayerNode {
                        id: 4,
                        neighbors: vec![1, 2],
                    },
                    LayerNode {
                        id: 5,
                        neighbors: vec![3],
                    },
                ],
            }],
        };
        let merged = persist().merge_snapshots(&base, &delta);
        let l0 = &merged.layers[0];
        let ids: Vec<u64> = l0.nodes.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&4));
        assert!(ids.contains(&5));
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_merge_delta_overrides_node() {
        let base = HnswSnapshot {
            meta: sample_meta(),
            layers: vec![HnswLayer {
                level: 0,
                nodes: vec![LayerNode {
                    id: 1,
                    neighbors: vec![2, 3],
                }],
            }],
        };
        let delta = HnswSnapshot {
            meta: sample_meta(),
            layers: vec![HnswLayer {
                level: 0,
                nodes: vec![LayerNode {
                    id: 1,
                    neighbors: vec![4, 5, 6],
                }], // new neighbors
            }],
        };
        let merged = persist().merge_snapshots(&base, &delta);
        let node1 = merged.layers[0]
            .nodes
            .iter()
            .find(|n| n.id == 1)
            .expect("node 1");
        assert_eq!(node1.neighbors, vec![4, 5, 6]);
    }

    #[test]
    fn test_merge_adds_new_layer() {
        let base = HnswSnapshot {
            meta: sample_meta(),
            layers: vec![sample_layer(0, &[1, 2])],
        };
        let delta = HnswSnapshot {
            meta: sample_meta(),
            layers: vec![sample_layer(1, &[1])],
        };
        let merged = persist().merge_snapshots(&base, &delta);
        assert_eq!(merged.layers.len(), 2);
        let levels: Vec<usize> = merged.layers.iter().map(|l| l.level).collect();
        assert!(levels.contains(&0));
        assert!(levels.contains(&1));
    }

    #[test]
    fn test_merge_meta_from_delta() {
        let base = HnswSnapshot {
            meta: HnswMeta {
                max_layer: 0,
                entry_point: Some(1),
                ef_construction: 100,
                m: 8,
                node_count: 10,
            },
            layers: vec![],
        };
        let delta = HnswSnapshot {
            meta: HnswMeta {
                max_layer: 2,
                entry_point: Some(99),
                ef_construction: 200,
                m: 16,
                node_count: 500,
            },
            layers: vec![],
        };
        let merged = persist().merge_snapshots(&base, &delta);
        assert_eq!(merged.meta.node_count, 500);
        assert_eq!(merged.meta.entry_point, Some(99));
    }

    // ── Large-scale roundtrip ─────────────────────────────────────────────────

    #[test]
    fn test_large_roundtrip() {
        let meta = HnswMeta {
            max_layer: 4,
            entry_point: Some(0),
            ef_construction: 400,
            m: 32,
            node_count: 10_000,
        };
        let mut layers = Vec::new();
        for level in 0..5 {
            let node_count = 1000 >> level;
            let ids: Vec<u64> = (0..node_count).map(|i| i as u64).collect();
            layers.push(sample_layer(level, &ids));
        }
        let bytes = persist().serialize(&meta, &layers);
        assert!(persist().validate_checksum(&bytes));
        let (decoded_meta, decoded_layers) = persist().deserialize(&bytes).expect("ok");
        assert_eq!(decoded_meta.m, 32);
        assert_eq!(decoded_layers.len(), 5);
    }
}
