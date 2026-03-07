//! Compressed Prefix Table for IRI storage in triple indexes.
//!
//! Assigns compact `u32` identifiers to IRI namespace prefixes and decomposes
//! full IRIs into `(PrefixId, local_name)` pairs, enabling significant storage
//! savings in large triple stores.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_tdb::prefix_table::{PrefixTable, CompressedIri};
//!
//! let mut table = PrefixTable::new();
//! let id = table.intern_prefix("http://www.w3.org/1999/02/22-rdf-syntax-ns#");
//! let compressed = table.compress_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
//! assert_eq!(compressed.prefix_id, id);
//! assert_eq!(compressed.local, "type");
//! let full = table.expand_iri(&compressed);
//! assert_eq!(full, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A compact numeric identifier for an IRI prefix string.
pub type PrefixId = u32;

/// A compressed representation of a full IRI.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompressedIri {
    /// The prefix identifier (maps to the IRI namespace string).
    pub prefix_id: PrefixId,
    /// The local name (the part of the IRI after the prefix).
    pub local: String,
}

impl CompressedIri {
    /// Create a new `CompressedIri`.
    pub fn new(prefix_id: PrefixId, local: impl Into<String>) -> Self {
        Self {
            prefix_id,
            local: local.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in prefix table operations.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum PrefixTableError {
    /// A prefix ID was not found in the table.
    #[error("Prefix ID {0} not found in table")]
    PrefixIdNotFound(PrefixId),

    /// The IRI does not match any registered prefix.
    #[error("No registered prefix matches IRI '{0}'")]
    NoPrefixMatch(String),

    /// Prefix string is empty.
    #[error("Prefix string cannot be empty")]
    EmptyPrefix,
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about the prefix table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefixTableStats {
    /// Number of registered prefixes.
    pub prefix_count: usize,
    /// Average length of the registered prefix strings (in bytes).
    pub average_prefix_length: f64,
    /// Estimated compression ratio: (total IRI bytes saved) / (total IRI bytes).
    ///
    /// This is computed from a sample of compressed IRIs if available.
    pub compression_ratio: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PrefixTable
// ─────────────────────────────────────────────────────────────────────────────

/// Bidirectional prefix ID ↔ IRI prefix string mapping optimized for triple
/// store compression.
///
/// Prefix strings are deduplicated and assigned compact `u32` identifiers.
/// The table supports O(1) forward lookup (`id → prefix`) and O(1) reverse
/// lookup (`prefix string → id`).
///
/// For `compress_iri` (IRI → CompressedIri), a longest-prefix match is
/// performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixTable {
    /// Forward map: ID → prefix string.
    id_to_prefix: Vec<String>,
    /// Reverse map: prefix string → ID.
    prefix_to_id: HashMap<String, PrefixId>,
    /// Counter for tracking total compressed bytes (for statistics).
    total_original_bytes: u64,
    /// Total bytes saved by compression.
    total_saved_bytes: u64,
}

impl Default for PrefixTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixTable {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create an empty prefix table.
    pub fn new() -> Self {
        Self {
            id_to_prefix: Vec::new(),
            prefix_to_id: HashMap::new(),
            total_original_bytes: 0,
            total_saved_bytes: 0,
        }
    }

    /// Create a prefix table pre-populated with common well-known RDF prefixes.
    pub fn with_well_known() -> Self {
        let mut table = Self::new();
        let well_known = [
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "http://www.w3.org/2000/01/rdf-schema#",
            "http://www.w3.org/2002/07/owl#",
            "http://www.w3.org/2001/XMLSchema#",
            "http://purl.org/dc/elements/1.1/",
            "http://purl.org/dc/terms/",
            "http://www.w3.org/2004/02/skos/core#",
            "http://xmlns.com/foaf/0.1/",
            "http://schema.org/",
        ];
        for prefix in &well_known {
            table.intern_prefix(prefix);
        }
        table
    }

    // ── Interning ────────────────────────────────────────────────────────────

    /// Intern a prefix string and return its `PrefixId`.
    ///
    /// If the prefix is already registered the existing ID is returned without
    /// allocating a new entry.
    pub fn intern_prefix(&mut self, prefix_str: &str) -> PrefixId {
        if let Some(&id) = self.prefix_to_id.get(prefix_str) {
            return id;
        }
        let id = self.id_to_prefix.len() as PrefixId;
        self.id_to_prefix.push(prefix_str.to_owned());
        self.prefix_to_id.insert(prefix_str.to_owned(), id);
        id
    }

    // ── Lookup ──────────────────────────────────────────────────────────────

    /// Look up the prefix string by `PrefixId`.
    pub fn lookup_prefix(&self, id: PrefixId) -> Option<&str> {
        self.id_to_prefix.get(id as usize).map(String::as_str)
    }

    /// Look up the `PrefixId` for an exact prefix string.
    pub fn lookup_id(&self, prefix_str: &str) -> Option<PrefixId> {
        self.prefix_to_id.get(prefix_str).copied()
    }

    /// Returns `true` if the prefix string is registered.
    pub fn contains(&self, prefix_str: &str) -> bool {
        self.prefix_to_id.contains_key(prefix_str)
    }

    /// Number of registered prefixes.
    pub fn len(&self) -> usize {
        self.id_to_prefix.len()
    }

    /// Returns `true` if no prefixes are registered.
    pub fn is_empty(&self) -> bool {
        self.id_to_prefix.is_empty()
    }

    // ── Compression / decompression ──────────────────────────────────────────

    /// Find the `PrefixId` of the longest registered prefix that is a prefix
    /// of `iri`.
    ///
    /// Returns `None` if no registered prefix matches.
    pub fn find_best_prefix(&self, iri: &str) -> Option<PrefixId> {
        let mut best_id: Option<PrefixId> = None;
        let mut best_len = 0usize;
        for (prefix, &id) in &self.prefix_to_id {
            if iri.starts_with(prefix.as_str()) && prefix.len() > best_len {
                best_len = prefix.len();
                best_id = Some(id);
            }
        }
        best_id
    }

    /// Compress a full IRI into a `(PrefixId, local)` pair.
    ///
    /// Uses longest-prefix match. If no registered prefix matches, a virtual
    /// prefix `""`(empty) is interned and the entire IRI becomes the local
    /// name — this means compression is always lossless.
    pub fn compress_iri(&mut self, iri: &str) -> CompressedIri {
        if let Some(id) = self.find_best_prefix(iri) {
            let prefix_str = &self.id_to_prefix[id as usize];
            let local = iri[prefix_str.len()..].to_owned();
            // Update statistics
            let saved = prefix_str.len().saturating_sub(4); // 4 = sizeof(u32)
            self.total_original_bytes += iri.len() as u64;
            self.total_saved_bytes += saved as u64;
            CompressedIri {
                prefix_id: id,
                local,
            }
        } else {
            // No match: use a synthetic prefix = "" (empty IRI prefix)
            let id = self.intern_prefix("");
            self.total_original_bytes += iri.len() as u64;
            CompressedIri {
                prefix_id: id,
                local: iri.to_owned(),
            }
        }
    }

    /// Try to compress a full IRI, returning an error if no prefix matches.
    pub fn try_compress_iri(&self, iri: &str) -> Result<CompressedIri, PrefixTableError> {
        let id = self
            .find_best_prefix(iri)
            .ok_or_else(|| PrefixTableError::NoPrefixMatch(iri.to_owned()))?;
        let prefix_str = &self.id_to_prefix[id as usize];
        let local = iri[prefix_str.len()..].to_owned();
        Ok(CompressedIri {
            prefix_id: id,
            local,
        })
    }

    /// Reconstruct the full IRI from a `CompressedIri`.
    ///
    /// Returns the concatenation of the prefix string and local name.
    pub fn expand_iri(&self, compressed: &CompressedIri) -> String {
        let prefix = self.lookup_prefix(compressed.prefix_id).unwrap_or("");
        format!("{}{}", prefix, compressed.local)
    }

    /// Try to expand a `CompressedIri`, returning an error if the `PrefixId`
    /// is not registered.
    pub fn try_expand_iri(&self, compressed: &CompressedIri) -> Result<String, PrefixTableError> {
        let prefix = self
            .lookup_prefix(compressed.prefix_id)
            .ok_or(PrefixTableError::PrefixIdNotFound(compressed.prefix_id))?;
        Ok(format!("{}{}", prefix, compressed.local))
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Return statistics about the current prefix table.
    pub fn stats(&self) -> PrefixTableStats {
        let prefix_count = self.id_to_prefix.len();
        let average_prefix_length = if prefix_count == 0 {
            0.0
        } else {
            let total: usize = self.id_to_prefix.iter().map(|s| s.len()).sum();
            total as f64 / prefix_count as f64
        };
        let compression_ratio = if self.total_original_bytes == 0 {
            0.0
        } else {
            self.total_saved_bytes as f64 / self.total_original_bytes as f64
        };
        PrefixTableStats {
            prefix_count,
            average_prefix_length,
            compression_ratio,
        }
    }

    /// Iterate over all `(id, prefix_str)` pairs in ID order.
    pub fn iter(&self) -> impl Iterator<Item = (PrefixId, &str)> {
        self.id_to_prefix
            .iter()
            .enumerate()
            .map(|(i, s)| (i as PrefixId, s.as_str()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    const RDFS_NS: &str = "http://www.w3.org/2000/01/rdf-schema#";
    const OWL_NS: &str = "http://www.w3.org/2002/07/owl#";
    const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let table = PrefixTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_default_is_empty() {
        let table = PrefixTable::default();
        assert!(table.is_empty());
    }

    #[test]
    fn test_well_known_has_nine_prefixes() {
        let table = PrefixTable::with_well_known();
        assert_eq!(table.len(), 9);
    }

    // ── Interning ────────────────────────────────────────────────────────────

    #[test]
    fn test_intern_first_prefix_gets_id_zero() {
        let mut table = PrefixTable::new();
        let id = table.intern_prefix(RDF_NS);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_intern_second_prefix_gets_id_one() {
        let mut table = PrefixTable::new();
        table.intern_prefix(RDF_NS);
        let id = table.intern_prefix(RDFS_NS);
        assert_eq!(id, 1);
    }

    #[test]
    fn test_intern_same_prefix_returns_same_id() {
        let mut table = PrefixTable::new();
        let id1 = table.intern_prefix(RDF_NS);
        let id2 = table.intern_prefix(RDF_NS);
        assert_eq!(id1, id2);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_intern_multiple_distinct_prefixes() {
        let mut table = PrefixTable::new();
        let id0 = table.intern_prefix(RDF_NS);
        let id1 = table.intern_prefix(RDFS_NS);
        let id2 = table.intern_prefix(OWL_NS);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(table.len(), 3);
    }

    // ── Lookup ──────────────────────────────────────────────────────────────

    #[test]
    fn test_lookup_prefix_by_id() {
        let mut table = PrefixTable::new();
        let id = table.intern_prefix(RDF_NS);
        assert_eq!(table.lookup_prefix(id), Some(RDF_NS));
    }

    #[test]
    fn test_lookup_prefix_nonexistent_id() {
        let table = PrefixTable::new();
        assert!(table.lookup_prefix(42).is_none());
    }

    #[test]
    fn test_lookup_id_by_prefix_string() {
        let mut table = PrefixTable::new();
        let id = table.intern_prefix(RDF_NS);
        assert_eq!(table.lookup_id(RDF_NS), Some(id));
    }

    #[test]
    fn test_lookup_id_nonexistent_prefix() {
        let table = PrefixTable::new();
        assert!(table.lookup_id("http://nope.example/").is_none());
    }

    #[test]
    fn test_contains_registered_prefix() {
        let mut table = PrefixTable::new();
        table.intern_prefix(RDF_NS);
        assert!(table.contains(RDF_NS));
        assert!(!table.contains(RDFS_NS));
    }

    // ── Best prefix matching ─────────────────────────────────────────────────

    #[test]
    fn test_find_best_prefix_exact() {
        let mut table = PrefixTable::new();
        let id = table.intern_prefix(RDF_NS);
        assert_eq!(table.find_best_prefix(&format!("{RDF_NS}type")), Some(id));
    }

    #[test]
    fn test_find_best_prefix_no_match() {
        let table = PrefixTable::new();
        assert!(table
            .find_best_prefix("http://totally-unknown.example/Foo")
            .is_none());
    }

    #[test]
    fn test_find_best_prefix_longest_match_wins() {
        let mut table = PrefixTable::new();
        let short_id = table.intern_prefix("http://example.org/");
        let long_id = table.intern_prefix("http://example.org/sub/");
        let best = table
            .find_best_prefix("http://example.org/sub/Thing")
            .unwrap();
        assert_eq!(best, long_id);
        assert_ne!(best, short_id);
    }

    // ── Compress ────────────────────────────────────────────────────────────

    #[test]
    fn test_compress_iri_basic() {
        let mut table = PrefixTable::new();
        let id = table.intern_prefix(RDF_NS);
        let compressed = table.compress_iri(&format!("{RDF_NS}type"));
        assert_eq!(compressed.prefix_id, id);
        assert_eq!(compressed.local, "type");
    }

    #[test]
    fn test_compress_iri_no_match_uses_empty_prefix() {
        let mut table = PrefixTable::new();
        let compressed = table.compress_iri("http://unknown.example/Foo");
        // Should fall back to empty prefix interning
        assert_eq!(compressed.local, "http://unknown.example/Foo");
    }

    #[test]
    fn test_compress_iri_rdfs_label() {
        let mut table = PrefixTable::new();
        let id = table.intern_prefix(RDFS_NS);
        let compressed = table.compress_iri(&format!("{RDFS_NS}label"));
        assert_eq!(compressed.prefix_id, id);
        assert_eq!(compressed.local, "label");
    }

    #[test]
    fn test_try_compress_iri_success() {
        let mut table = PrefixTable::new();
        let id = table.intern_prefix(RDF_NS);
        let compressed = table.try_compress_iri(&format!("{RDF_NS}type")).unwrap();
        assert_eq!(compressed.prefix_id, id);
    }

    #[test]
    fn test_try_compress_iri_no_match_error() {
        let table = PrefixTable::new();
        let err = table
            .try_compress_iri("http://nope.example/Foo")
            .unwrap_err();
        assert!(matches!(err, PrefixTableError::NoPrefixMatch(_)));
    }

    // ── Expand ──────────────────────────────────────────────────────────────

    #[test]
    fn test_expand_iri_roundtrip() {
        let mut table = PrefixTable::new();
        table.intern_prefix(RDF_NS);
        let original = format!("{RDF_NS}type");
        let compressed = table.compress_iri(&original);
        let expanded = table.expand_iri(&compressed);
        assert_eq!(expanded, original);
    }

    #[test]
    fn test_try_expand_iri_invalid_id() {
        let table = PrefixTable::new();
        let compressed = CompressedIri::new(999, "local");
        let err = table.try_expand_iri(&compressed).unwrap_err();
        assert!(matches!(err, PrefixTableError::PrefixIdNotFound(_)));
    }

    #[test]
    fn test_expand_iri_xsd_integer() {
        let mut table = PrefixTable::new();
        table.intern_prefix(XSD_NS);
        let original = format!("{XSD_NS}integer");
        let compressed = table.compress_iri(&original);
        let expanded = table.expand_iri(&compressed);
        assert_eq!(expanded, original);
    }

    #[test]
    fn test_expand_iri_owl_class() {
        let mut table = PrefixTable::new();
        table.intern_prefix(OWL_NS);
        let original = format!("{OWL_NS}Class");
        let compressed = table.compress_iri(&original);
        let expanded = table.expand_iri(&compressed);
        assert_eq!(expanded, original);
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_table() {
        let table = PrefixTable::new();
        let stats = table.stats();
        assert_eq!(stats.prefix_count, 0);
        assert_eq!(stats.average_prefix_length, 0.0);
        assert_eq!(stats.compression_ratio, 0.0);
    }

    #[test]
    fn test_stats_prefix_count() {
        let mut table = PrefixTable::new();
        table.intern_prefix(RDF_NS);
        table.intern_prefix(RDFS_NS);
        let stats = table.stats();
        assert_eq!(stats.prefix_count, 2);
    }

    #[test]
    fn test_stats_average_prefix_length_non_zero() {
        let mut table = PrefixTable::new();
        table.intern_prefix(RDF_NS);
        let stats = table.stats();
        assert!(stats.average_prefix_length > 0.0);
        assert!((stats.average_prefix_length - RDF_NS.len() as f64).abs() < 1e-9);
    }

    #[test]
    fn test_stats_compression_ratio_after_compress() {
        let mut table = PrefixTable::new();
        table.intern_prefix(RDF_NS);
        // Compress a few IRIs to populate statistics
        for local in &["type", "Property", "value", "List"] {
            table.compress_iri(&format!("{RDF_NS}{local}"));
        }
        let stats = table.stats();
        // Ratio should be between 0 and 1 since prefix (43 chars) > sizeof(u32) (4 bytes)
        assert!(stats.compression_ratio >= 0.0);
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let mut table = PrefixTable::new();
        table.intern_prefix(RDF_NS);
        table.intern_prefix(RDFS_NS);
        let json = serde_json::to_string(&table).unwrap();
        let restored: PrefixTable = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.lookup_prefix(0), Some(RDF_NS));
    }

    // ── CompressedIri ─────────────────────────────────────────────────────────

    #[test]
    fn test_compressed_iri_new() {
        let c = CompressedIri::new(5, "type");
        assert_eq!(c.prefix_id, 5);
        assert_eq!(c.local, "type");
    }

    #[test]
    fn test_compressed_iri_equality() {
        let c1 = CompressedIri::new(0, "type");
        let c2 = CompressedIri::new(0, "type");
        assert_eq!(c1, c2);
    }

    // ── Iteration ─────────────────────────────────────────────────────────────

    #[test]
    fn test_iter_prefix_table() {
        let mut table = PrefixTable::new();
        table.intern_prefix(RDF_NS);
        table.intern_prefix(RDFS_NS);
        let entries: Vec<_> = table.iter().collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], (0, RDF_NS));
        assert_eq!(entries[1], (1, RDFS_NS));
    }

    // ── Error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_prefix_id_not_found_display() {
        let err = PrefixTableError::PrefixIdNotFound(42);
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn test_error_no_prefix_match_display() {
        let err = PrefixTableError::NoPrefixMatch("http://x.y/z".to_owned());
        assert!(err.to_string().contains("http://x.y/z"));
    }

    // ── Well-known table ─────────────────────────────────────────────────────

    #[test]
    fn test_well_known_contains_rdf_ns() {
        let table = PrefixTable::with_well_known();
        assert!(table.contains(RDF_NS));
    }

    #[test]
    fn test_well_known_compress_rdf_type() {
        let mut table = PrefixTable::with_well_known();
        let compressed = table.compress_iri(&format!("{RDF_NS}type"));
        assert_eq!(compressed.local, "type");
        let expanded = table.expand_iri(&compressed);
        assert_eq!(expanded, format!("{RDF_NS}type"));
    }
}
