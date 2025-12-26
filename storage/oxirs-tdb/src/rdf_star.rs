//! RDF-star support for quoted triples
//!
//! This module implements RDF-star (RDF 1.2) support, allowing triples to be
//! quoted and used as subjects or objects in other triples. This enables
//! powerful metadata and provenance tracking capabilities.
//!
//! ## RDF-star Overview
//!
//! RDF-star extends RDF by allowing triples to appear as subjects or objects:
//! ```turtle
//! <<:alice :likes :bob>> :certainty 0.9 .
//! <<:alice :likes :bob>> :source :survey2023 .
//! ```
//!
//! ## Encoding Strategy
//!
//! Quoted triples are encoded using a special NodeId marker in the high byte,
//! similar to inline values. The NodeId references an entry in a special
//! quoted triple table that stores the actual (subject, predicate, object) tuple.

use crate::dictionary::NodeId;
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// A quoted triple that can be used as a subject or object in RDF-star
///
/// Represents `<<subject predicate object>>` in Turtle* syntax
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord, Encode, Decode,
)]
pub struct QuotedTriple {
    /// Subject of the quoted triple
    pub subject: NodeId,
    /// Predicate of the quoted triple
    pub predicate: NodeId,
    /// Object of the quoted triple
    pub object: NodeId,
}

impl QuotedTriple {
    /// Create a new quoted triple
    pub fn new(subject: NodeId, predicate: NodeId, object: NodeId) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Check if this quoted triple contains another quoted triple (nested)
    pub fn is_nested(&self, quoted_triple_table: &QuotedTripleTable) -> bool {
        quoted_triple_table.is_quoted_triple(self.subject)
            || quoted_triple_table.is_quoted_triple(self.object)
    }

    /// Get maximum nesting depth
    pub fn max_nesting_depth(&self, quoted_triple_table: &QuotedTripleTable) -> usize {
        let subject_depth = if quoted_triple_table.is_quoted_triple(self.subject) {
            if let Some(qt) = quoted_triple_table.get(self.subject) {
                1 + qt.max_nesting_depth(quoted_triple_table)
            } else {
                0
            }
        } else {
            0
        };

        let object_depth = if quoted_triple_table.is_quoted_triple(self.object) {
            if let Some(qt) = quoted_triple_table.get(self.object) {
                1 + qt.max_nesting_depth(quoted_triple_table)
            } else {
                0
            }
        } else {
            0
        };

        subject_depth.max(object_depth)
    }
}

/// Type marker for quoted triples (stored in high byte of NodeId)
pub const QUOTED_TRIPLE_MARKER: u8 = 0x90;

/// Maximum nesting depth for quoted triples to prevent stack overflow
pub const MAX_NESTING_DEPTH: usize = 100;

/// Quoted triple table for RDF-star support
///
/// Stores quoted triples and assigns them NodeIds for use in other triples.
/// The NodeId uses the QUOTED_TRIPLE_MARKER (0x90) in the high byte to
/// distinguish quoted triples from regular terms.
#[derive(Debug, Clone)]
pub struct QuotedTripleTable {
    /// Mapping from NodeId to QuotedTriple
    triples: std::collections::HashMap<NodeId, QuotedTriple>,
    /// Reverse mapping from QuotedTriple to NodeId for deduplication
    reverse: std::collections::HashMap<QuotedTriple, NodeId>,
    /// Next available ID for quoted triples
    next_id: u64,
}

impl QuotedTripleTable {
    /// Create a new quoted triple table
    pub fn new() -> Self {
        Self {
            triples: std::collections::HashMap::new(),
            reverse: std::collections::HashMap::new(),
            next_id: 0,
        }
    }

    /// Create a NodeId for a quoted triple with the marker
    fn create_node_id(&self, id: u64) -> NodeId {
        let encoded = ((QUOTED_TRIPLE_MARKER as u64) << 56) | (id & 0x00FFFFFFFFFFFFFF);
        NodeId::from(encoded)
    }

    /// Extract the ID from a quoted triple NodeId
    fn extract_id(node_id: NodeId) -> u64 {
        node_id.as_u64() & 0x00FFFFFFFFFFFFFF
    }

    /// Check if a NodeId represents a quoted triple
    pub fn is_quoted_triple(&self, node_id: NodeId) -> bool {
        let type_byte = ((node_id.as_u64() >> 56) & 0xFF) as u8;
        type_byte == QUOTED_TRIPLE_MARKER
    }

    /// Get or create a NodeId for a quoted triple
    ///
    /// Returns an error if nesting depth exceeds MAX_NESTING_DEPTH
    pub fn get_or_create(&mut self, triple: QuotedTriple) -> Result<NodeId, RdfStarError> {
        // Check nesting depth to prevent stack overflow
        let depth = triple.max_nesting_depth(self);
        if depth > MAX_NESTING_DEPTH {
            return Err(RdfStarError::ExcessiveNesting(depth));
        }

        // Check if this quoted triple already exists
        if let Some(&node_id) = self.reverse.get(&triple) {
            return Ok(node_id);
        }

        // Create new NodeId
        let id = self.next_id;
        self.next_id += 1;

        let node_id = self.create_node_id(id);

        // Store bidirectional mapping
        self.triples.insert(node_id, triple);
        self.reverse.insert(triple, node_id);

        Ok(node_id)
    }

    /// Get a quoted triple by NodeId
    pub fn get(&self, node_id: NodeId) -> Option<QuotedTriple> {
        self.triples.get(&node_id).copied()
    }

    /// Get the number of quoted triples in the table
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Clear all quoted triples
    pub fn clear(&mut self) {
        self.triples.clear();
        self.reverse.clear();
        self.next_id = 0;
    }

    /// Get all quoted triples
    pub fn all_triples(&self) -> Vec<(NodeId, QuotedTriple)> {
        self.triples
            .iter()
            .map(|(&id, &triple)| (id, triple))
            .collect()
    }
}

impl Default for QuotedTripleTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors specific to RDF-star operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdfStarError {
    /// Nesting depth exceeds maximum allowed
    ExcessiveNesting(usize),
    /// Invalid quoted triple NodeId
    InvalidQuotedTripleId(NodeId),
    /// Quoted triple not found
    NotFound(NodeId),
}

impl std::fmt::Display for RdfStarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfStarError::ExcessiveNesting(depth) => {
                write!(
                    f,
                    "Excessive nesting depth {} exceeds maximum {}",
                    depth, MAX_NESTING_DEPTH
                )
            }
            RdfStarError::InvalidQuotedTripleId(id) => {
                write!(f, "Invalid quoted triple NodeId: {:?}", id)
            }
            RdfStarError::NotFound(id) => write!(f, "Quoted triple not found: {:?}", id),
        }
    }
}

impl std::error::Error for RdfStarError {}

/// Statistics about RDF-star usage
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct RdfStarStats {
    /// Total number of quoted triples
    pub total_quoted_triples: usize,
    /// Number of nested quoted triples (depth > 0)
    pub nested_count: usize,
    /// Maximum nesting depth observed
    pub max_depth: usize,
    /// Average nesting depth
    pub avg_depth: f64,
}

impl QuotedTripleTable {
    /// Collect statistics about RDF-star usage
    pub fn statistics(&self) -> RdfStarStats {
        let total = self.triples.len();

        if total == 0 {
            return RdfStarStats {
                total_quoted_triples: 0,
                nested_count: 0,
                max_depth: 0,
                avg_depth: 0.0,
            };
        }

        let mut max_depth = 0;
        let mut total_depth = 0;
        let mut nested_count = 0;

        for triple in self.triples.values() {
            let depth = triple.max_nesting_depth(self);
            if depth > 0 {
                nested_count += 1;
            }
            max_depth = max_depth.max(depth);
            total_depth += depth;
        }

        RdfStarStats {
            total_quoted_triples: total,
            nested_count,
            max_depth,
            avg_depth: total_depth as f64 / total as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quoted_triple_creation() {
        let triple = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));

        assert_eq!(triple.subject, NodeId::from(1));
        assert_eq!(triple.predicate, NodeId::from(2));
        assert_eq!(triple.object, NodeId::from(3));
    }

    #[test]
    fn test_quoted_triple_table_basic() {
        let mut table = QuotedTripleTable::new();

        let triple = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));

        let node_id = table.get_or_create(triple).unwrap();

        // Verify it's marked as quoted triple
        assert!(table.is_quoted_triple(node_id));

        // Verify we can retrieve it
        assert_eq!(table.get(node_id), Some(triple));
    }

    #[test]
    fn test_quoted_triple_deduplication() {
        let mut table = QuotedTripleTable::new();

        let triple = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));

        let id1 = table.get_or_create(triple).unwrap();
        let id2 = table.get_or_create(triple).unwrap();

        // Same quoted triple should get same NodeId
        assert_eq!(id1, id2);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_quoted_triple_nested() {
        let mut table = QuotedTripleTable::new();

        // Create first quoted triple: <<s1 p1 o1>>
        let triple1 = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));
        let qt_id1 = table.get_or_create(triple1).unwrap();

        // Create nested quoted triple: <<qt1 p2 o2>>
        let triple2 = QuotedTriple::new(
            qt_id1, // Use first quoted triple as subject
            NodeId::from(4),
            NodeId::from(5),
        );
        let qt_id2 = table.get_or_create(triple2).unwrap();

        assert!(table.is_quoted_triple(qt_id1));
        assert!(table.is_quoted_triple(qt_id2));
        assert!(triple2.is_nested(&table));
        assert_eq!(triple2.max_nesting_depth(&table), 1);
    }

    #[test]
    fn test_quoted_triple_max_nesting_depth() {
        let mut table = QuotedTripleTable::new();

        let mut current = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));
        let mut current_id = table.get_or_create(current).unwrap();

        // Create nested chain of depth 5
        for i in 4..9 {
            current = QuotedTriple::new(current_id, NodeId::from(i), NodeId::from(i + 1));
            current_id = table.get_or_create(current).unwrap();
        }

        let final_triple = table.get(current_id).unwrap();
        assert_eq!(final_triple.max_nesting_depth(&table), 5);
    }

    #[test]
    fn test_excessive_nesting_prevention() {
        let mut table = QuotedTripleTable::new();

        let mut current = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));
        let mut current_id = table.get_or_create(current).unwrap();

        // Try to create chain exceeding MAX_NESTING_DEPTH
        for i in 0..MAX_NESTING_DEPTH + 5 {
            current = QuotedTriple::new(
                current_id,
                NodeId::from(i as u64 + 100),
                NodeId::from(i as u64 + 200),
            );

            match table.get_or_create(current) {
                Ok(id) => current_id = id,
                Err(RdfStarError::ExcessiveNesting(depth)) => {
                    assert!(depth > MAX_NESTING_DEPTH);
                    return; // Test passed
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        panic!("Should have prevented excessive nesting");
    }

    #[test]
    fn test_quoted_triple_table_clear() {
        let mut table = QuotedTripleTable::new();

        for i in 0..10 {
            let triple = QuotedTriple::new(
                NodeId::from(i * 3),
                NodeId::from(i * 3 + 1),
                NodeId::from(i * 3 + 2),
            );
            table.get_or_create(triple).unwrap();
        }

        assert_eq!(table.len(), 10);

        table.clear();

        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_rdf_star_statistics() {
        let mut table = QuotedTripleTable::new();

        // Add simple quoted triples
        for i in 0..5 {
            let triple = QuotedTriple::new(
                NodeId::from(i * 3),
                NodeId::from(i * 3 + 1),
                NodeId::from(i * 3 + 2),
            );
            table.get_or_create(triple).unwrap();
        }

        // Add one nested triple
        let triple1 = QuotedTriple::new(NodeId::from(100), NodeId::from(101), NodeId::from(102));
        let qt_id1 = table.get_or_create(triple1).unwrap();

        let triple2 = QuotedTriple::new(qt_id1, NodeId::from(200), NodeId::from(201));
        table.get_or_create(triple2).unwrap();

        let stats = table.statistics();

        assert_eq!(stats.total_quoted_triples, 7);
        assert_eq!(stats.nested_count, 1);
        assert_eq!(stats.max_depth, 1);
    }

    #[test]
    fn test_quoted_triple_marker() {
        let mut table = QuotedTripleTable::new();

        let triple = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));

        let qt_id = table.get_or_create(triple).unwrap();

        // Check high byte is QUOTED_TRIPLE_MARKER
        let high_byte = ((qt_id.as_u64() >> 56) & 0xFF) as u8;
        assert_eq!(high_byte, QUOTED_TRIPLE_MARKER);

        // Regular NodeIds should not be marked as quoted triples
        assert!(!table.is_quoted_triple(NodeId::from(123)));
    }

    #[test]
    fn test_quoted_triple_serialization() {
        let triple = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));

        let serialized = bincode::encode_to_vec(triple, bincode::config::standard()).unwrap();
        let deserialized: QuotedTriple =
            bincode::decode_from_slice(&serialized, bincode::config::standard())
                .unwrap()
                .0;

        assert_eq!(triple, deserialized);
    }

    #[test]
    fn test_quoted_triple_ordering() {
        let triple1 = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(3));

        let triple2 = QuotedTriple::new(NodeId::from(1), NodeId::from(2), NodeId::from(4));

        assert!(triple1 < triple2);
    }
}
