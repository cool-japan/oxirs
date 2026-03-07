//! HNSW Index Snapshot - Save and Restore without rebuilding
//!
//! Binary format (big-endian throughout):
//!   [4 bytes]  magic "HNSW"
//!   [4 bytes]  format version (u32)
//!   [8 bytes]  num_nodes (u64)
//!   [8 bytes]  num_layers (u64)
//!   [8 bytes]  dimension (u64)
//!   [8 bytes]  ef_construction (u64)
//!   [8 bytes]  m (u64)
//!   [8 bytes]  m_l0 (u64)
//!   [1 byte]   has_entry_point (u8: 0 or 1)
//!   [8 bytes]  entry_point (u64, only present if has_entry_point == 1)
//!   For each node:
//!     \[8 bytes\]  uri_len (u64)
//!     \[uri_len\]  uri bytes (UTF-8)
//!     [8 bytes]  vector_len (u64) -- number of f32 elements
//!     [4*n bytes] vector data (f32 little-endian)
//!     [8 bytes]  num_connection_layers (u64)
//!     For each layer:
//!       [8 bytes]  num_connections (u64)
//!       For each connection:
//!         [8 bytes]  connected_node_id (u64)

use crate::hnsw::{HnswConfig, HnswIndex, Node};
use crate::Vector;
use crate::VectorError;
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::Path;

/// Magic bytes identifying an HNSW snapshot file
const SNAPSHOT_MAGIC: &[u8; 4] = b"HNSW";

/// Snapshot format version
const SNAPSHOT_VERSION: u32 = 1;

/// Header decoded from a snapshot
#[derive(Debug, Clone)]
pub struct SnapshotHeader {
    /// File magic — always b"HNSW"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Number of nodes stored
    pub num_nodes: usize,
    /// Number of hierarchy layers
    pub num_layers: usize,
    /// Vector dimension
    pub dimension: usize,
    /// ef_construction parameter at snapshot time
    pub ef_construction: usize,
    /// M parameter at snapshot time
    pub m: usize,
    /// M_l0 parameter at snapshot time
    pub m_l0: usize,
    /// Entry point node id (None when index is empty)
    pub entry_point: Option<usize>,
}

/// Snapshot I/O for an [`HnswIndex`].
///
/// All multi-byte integers are stored as little-endian `u64` / `u32`.
/// Floating-point values are stored as little-endian `f32`.
pub struct IndexSnapshot;

impl IndexSnapshot {
    // ──────────────────────────────────────────────────────────────────────────
    // Public API
    // ──────────────────────────────────────────────────────────────────────────

    /// Serialize `index` into `writer`.
    ///
    /// Returns the total number of bytes written.
    pub fn save<W: Write>(index: &HnswIndex, writer: &mut W) -> Result<usize, VectorError> {
        let mut written = 0usize;

        // ── magic ──────────────────────────────────────────────────────────────
        writer
            .write_all(SNAPSHOT_MAGIC)
            .map_err(VectorError::IoError)?;
        written += 4;

        // ── version ────────────────────────────────────────────────────────────
        Self::write_u32(writer, SNAPSHOT_VERSION).map_err(VectorError::IoError)?;
        written += 4;

        let nodes = index.nodes();
        let config = index.config();

        // Derive the maximum layer count from the stored nodes
        let num_layers = nodes.iter().map(|n| n.connections.len()).max().unwrap_or(0);

        let dimension = nodes.first().map(|n| n.vector_data_f32.len()).unwrap_or(0);

        // ── header scalars ─────────────────────────────────────────────────────
        Self::write_u64(writer, nodes.len() as u64).map_err(VectorError::IoError)?;
        written += 8;
        Self::write_u64(writer, num_layers as u64).map_err(VectorError::IoError)?;
        written += 8;
        Self::write_u64(writer, dimension as u64).map_err(VectorError::IoError)?;
        written += 8;
        Self::write_u64(writer, config.ef_construction as u64).map_err(VectorError::IoError)?;
        written += 8;
        Self::write_u64(writer, config.m as u64).map_err(VectorError::IoError)?;
        written += 8;
        Self::write_u64(writer, config.m_l0 as u64).map_err(VectorError::IoError)?;
        written += 8;

        // ── entry point ────────────────────────────────────────────────────────
        match index.entry_point() {
            None => {
                Self::write_u8(writer, 0).map_err(VectorError::IoError)?;
                written += 1;
            }
            Some(ep) => {
                Self::write_u8(writer, 1).map_err(VectorError::IoError)?;
                written += 1;
                Self::write_u64(writer, ep as u64).map_err(VectorError::IoError)?;
                written += 8;
            }
        }

        // ── nodes ──────────────────────────────────────────────────────────────
        for node in nodes {
            written += Self::write_node(writer, node).map_err(VectorError::IoError)?;
        }

        writer.flush().map_err(VectorError::IoError)?;
        Ok(written)
    }

    /// Deserialize an [`HnswIndex`] from `reader`.
    pub fn load<R: Read>(reader: &mut R) -> Result<HnswIndex, VectorError> {
        // ── magic ──────────────────────────────────────────────────────────────
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(VectorError::IoError)?;
        if &magic != SNAPSHOT_MAGIC {
            return Err(VectorError::InvalidData(format!(
                "Invalid snapshot magic: expected {:?}, got {:?}",
                SNAPSHOT_MAGIC, magic
            )));
        }

        // ── version ────────────────────────────────────────────────────────────
        let version = Self::read_u32(reader).map_err(VectorError::IoError)?;
        if version != SNAPSHOT_VERSION {
            return Err(VectorError::InvalidData(format!(
                "Unsupported snapshot version: {}",
                version
            )));
        }

        // ── header scalars ─────────────────────────────────────────────────────
        let num_nodes = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
        let _num_layers = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
        let _dimension = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
        let ef_construction = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
        let m = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
        let m_l0 = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;

        // ── entry point ────────────────────────────────────────────────────────
        let has_entry = Self::read_u8(reader).map_err(VectorError::IoError)?;
        let entry_point = if has_entry == 1 {
            Some(Self::read_u64(reader).map_err(VectorError::IoError)? as usize)
        } else {
            None
        };

        // ── reconstruct config ─────────────────────────────────────────────────
        let config = HnswConfig {
            m,
            m_l0,
            ef_construction,
            ..HnswConfig::default()
        };

        // ── nodes ──────────────────────────────────────────────────────────────
        let mut nodes: Vec<Node> = Vec::with_capacity(num_nodes);
        let mut uri_to_id: std::collections::HashMap<String, usize> =
            std::collections::HashMap::with_capacity(num_nodes);

        for idx in 0..num_nodes {
            let node = Self::read_node(reader)?;
            uri_to_id.insert(node.uri.clone(), idx);
            nodes.push(node);
        }

        // ── assemble index ─────────────────────────────────────────────────────
        let mut index = HnswIndex::new_cpu_only(config);
        // Replace internal state via the provided accessors
        *index.nodes_mut() = nodes;
        *index.uri_to_id_mut() = uri_to_id;
        index.set_entry_point(entry_point);

        Ok(index)
    }

    /// Persist `index` to a file at `path`.
    ///
    /// The file is created (or truncated) atomically via a temporary sibling file.
    pub fn save_to_file(index: &HnswIndex, path: &Path) -> Result<usize, VectorError> {
        // Write to a temporary file first, then rename for atomicity
        let tmp_path = path.with_extension("hnsw.tmp");
        let file = std::fs::File::create(&tmp_path).map_err(VectorError::IoError)?;
        let mut writer = std::io::BufWriter::new(file);

        let written = Self::save(index, &mut writer)?;
        drop(writer);

        std::fs::rename(&tmp_path, path).map_err(VectorError::IoError)?;
        Ok(written)
    }

    /// Load an index from a file at `path`.
    pub fn load_from_file(path: &Path) -> Result<HnswIndex, VectorError> {
        let file = std::fs::File::open(path).map_err(VectorError::IoError)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load(&mut reader)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Private node I/O
    // ──────────────────────────────────────────────────────────────────────────

    fn write_node<W: Write>(writer: &mut W, node: &Node) -> std::io::Result<usize> {
        let mut written = 0usize;

        // uri
        let uri_bytes = node.uri.as_bytes();
        Self::write_u64(writer, uri_bytes.len() as u64)?;
        written += 8;
        writer.write_all(uri_bytes)?;
        written += uri_bytes.len();

        // vector data (f32 array)
        Self::write_u64(writer, node.vector_data_f32.len() as u64)?;
        written += 8;
        for &v in &node.vector_data_f32 {
            Self::write_f32(writer, v)?;
            written += 4;
        }

        // connections per layer
        Self::write_u64(writer, node.connections.len() as u64)?;
        written += 8;
        for layer_connections in &node.connections {
            Self::write_u64(writer, layer_connections.len() as u64)?;
            written += 8;
            for &neighbor_id in layer_connections {
                Self::write_u64(writer, neighbor_id as u64)?;
                written += 8;
            }
        }

        Ok(written)
    }

    fn read_node<R: Read>(reader: &mut R) -> Result<Node, VectorError> {
        // uri
        let uri_len = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
        let mut uri_bytes = vec![0u8; uri_len];
        reader
            .read_exact(&mut uri_bytes)
            .map_err(VectorError::IoError)?;
        let uri = String::from_utf8(uri_bytes)
            .map_err(|e| VectorError::InvalidData(format!("Invalid UTF-8 in URI: {}", e)))?;

        // vector data
        let vec_len = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
        let mut vector_data_f32 = Vec::with_capacity(vec_len);
        for _ in 0..vec_len {
            let v = Self::read_f32(reader).map_err(VectorError::IoError)?;
            vector_data_f32.push(v);
        }

        // Reconstruct Vector from f32 data
        let vector = Vector::new(vector_data_f32.clone());

        // connections
        let num_layers = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
        let mut connections: Vec<HashSet<usize>> = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let num_conn = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
            let mut layer_set = HashSet::with_capacity(num_conn);
            for _ in 0..num_conn {
                let neighbor = Self::read_u64(reader).map_err(VectorError::IoError)? as usize;
                layer_set.insert(neighbor);
            }
            connections.push(layer_set);
        }

        let max_level = num_layers.saturating_sub(1);
        let mut node = Node::new(uri, vector, max_level);
        node.connections = connections;
        node.vector_data_f32 = vector_data_f32;

        Ok(node)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Low-level I/O helpers (no external serialization crates)
    // ──────────────────────────────────────────────────────────────────────────

    fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }

    fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }

    fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn write_u8<W: Write>(w: &mut W, v: u8) -> std::io::Result<()> {
        w.write_all(&[v])
    }

    fn read_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
        let mut buf = [0u8; 1];
        r.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn write_f32<W: Write>(w: &mut W, v: f32) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }

    fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{HnswConfig, HnswIndex};
    use crate::VectorIndex;

    fn make_index_with_vectors(n: usize, dim: usize) -> HnswIndex {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new_cpu_only(config);

        for i in 0..n {
            let data: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 / 1000.0).collect();
            let uri = format!("http://example.org/v{}", i);
            let vec = Vector::new(data);
            index.insert(uri, vec).expect("insert failed");
        }

        index
    }

    #[test]
    fn test_save_and_load_empty_index() {
        let index = HnswIndex::new_cpu_only(HnswConfig::default());
        let mut buf = Vec::new();
        let bytes = IndexSnapshot::save(&index, &mut buf).expect("save failed");
        assert!(bytes > 0);

        let loaded = IndexSnapshot::load(&mut buf.as_slice()).expect("load failed");
        assert_eq!(loaded.len(), 0);
        assert_eq!(loaded.entry_point(), None);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let original = make_index_with_vectors(20, 8);
        assert_eq!(original.len(), 20);

        let mut buf = Vec::new();
        IndexSnapshot::save(&original, &mut buf).expect("save failed");

        let restored = IndexSnapshot::load(&mut buf.as_slice()).expect("load failed");

        // Node count preserved
        assert_eq!(restored.len(), original.len());

        // URI mapping preserved
        for uri in original.uri_to_id().keys() {
            assert!(
                restored.uri_to_id().contains_key(uri),
                "URI {} missing after restore",
                uri
            );
        }

        // Entry point preserved
        assert_eq!(original.entry_point(), restored.entry_point());
    }

    #[test]
    fn test_save_and_load_vectors_preserved() {
        let original = make_index_with_vectors(10, 4);

        let mut buf = Vec::new();
        IndexSnapshot::save(&original, &mut buf).expect("save failed");
        let restored = IndexSnapshot::load(&mut buf.as_slice()).expect("load failed");

        // Check each node's vector data matches
        for (orig_node, rest_node) in original.nodes().iter().zip(restored.nodes().iter()) {
            assert_eq!(orig_node.uri, rest_node.uri);
            assert_eq!(
                orig_node.vector_data_f32.len(),
                rest_node.vector_data_f32.len()
            );
            for (a, b) in orig_node
                .vector_data_f32
                .iter()
                .zip(rest_node.vector_data_f32.iter())
            {
                assert!((a - b).abs() < 1e-6, "Vector data mismatch: {} vs {}", a, b);
            }
        }
    }

    #[test]
    fn test_save_and_load_connections_preserved() {
        let original = make_index_with_vectors(30, 8);

        let mut buf = Vec::new();
        IndexSnapshot::save(&original, &mut buf).expect("save failed");
        let restored = IndexSnapshot::load(&mut buf.as_slice()).expect("load failed");

        // Verify connection structure is preserved for each node
        for (i, (orig, rest)) in original
            .nodes()
            .iter()
            .zip(restored.nodes().iter())
            .enumerate()
        {
            assert_eq!(
                orig.connections.len(),
                rest.connections.len(),
                "Node {} layer count mismatch",
                i
            );
            for (layer, (oc, rc)) in orig
                .connections
                .iter()
                .zip(rest.connections.iter())
                .enumerate()
            {
                assert_eq!(oc, rc, "Node {} layer {} connections mismatch", i, layer);
            }
        }
    }

    #[test]
    fn test_file_save_and_load() {
        let original = make_index_with_vectors(15, 6);

        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_snapshot_test.hnsw");

        let bytes = IndexSnapshot::save_to_file(&original, &path).expect("save_to_file failed");
        assert!(bytes > 0);
        assert!(path.exists());

        let restored = IndexSnapshot::load_from_file(&path).expect("load_from_file failed");
        assert_eq!(restored.len(), original.len());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_corrupt_magic_rejected() {
        let mut buf = vec![0u8; 100];
        buf[0] = b'X'; // corrupt magic
        let result = IndexSnapshot::load(&mut buf.as_slice());
        assert!(result.is_err());
    }

    #[test]
    fn test_config_restored() {
        let config = HnswConfig {
            m: 8,
            m_l0: 16,
            ef_construction: 50,
            ..Default::default()
        };

        let mut index = HnswIndex::new_cpu_only(config);
        let vec_a = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        index
            .insert("http://example.org/a".to_string(), vec_a)
            .expect("insert");

        let mut buf = Vec::new();
        IndexSnapshot::save(&index, &mut buf).expect("save");
        let restored = IndexSnapshot::load(&mut buf.as_slice()).expect("load");

        assert_eq!(restored.config().m, 8);
        assert_eq!(restored.config().m_l0, 16);
        assert_eq!(restored.config().ef_construction, 50);
    }
}
