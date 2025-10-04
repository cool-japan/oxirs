//! NodeId type for dictionary encoding

use serde::{Deserialize, Serialize};
use std::fmt;

/// 8-byte node identifier for RDF terms
///
/// NodeIds are assigned sequentially and used to compress
/// IRIs, literals, and blank nodes to fixed-size integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    /// Create a new NodeId from a raw u64 value
    pub const fn new(id: u64) -> Self {
        NodeId(id)
    }

    /// Get the raw u64 value
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// First valid node ID (0 is reserved as NULL)
    pub const FIRST: NodeId = NodeId(1);

    /// Reserved NULL node ID
    pub const NULL: NodeId = NodeId(0);

    /// Check if this is a null node ID
    pub const fn is_null(&self) -> bool {
        self.0 == 0
    }

    /// Get the next node ID
    pub const fn next(&self) -> NodeId {
        NodeId(self.0 + 1)
    }
}

impl Default for NodeId {
    fn default() -> Self {
        NodeId::NULL
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

impl From<u64> for NodeId {
    fn from(id: u64) -> Self {
        NodeId(id)
    }
}

impl From<NodeId> for u64 {
    fn from(id: NodeId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_creation() {
        let id = NodeId::new(42);
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn test_node_id_null() {
        let null_id = NodeId::NULL;
        assert!(null_id.is_null());
        assert_eq!(null_id.as_u64(), 0);

        let valid_id = NodeId::FIRST;
        assert!(!valid_id.is_null());
        assert_eq!(valid_id.as_u64(), 1);
    }

    #[test]
    fn test_node_id_next() {
        let id = NodeId::new(10);
        let next = id.next();
        assert_eq!(next.as_u64(), 11);
    }

    #[test]
    fn test_node_id_ordering() {
        let id1 = NodeId::new(10);
        let id2 = NodeId::new(20);
        assert!(id1 < id2);
        assert!(id2 > id1);
    }

    #[test]
    fn test_node_id_conversions() {
        let raw: u64 = 42;
        let id: NodeId = raw.into();
        assert_eq!(id.as_u64(), 42);

        let back: u64 = id.into();
        assert_eq!(back, 42);
    }

    #[test]
    fn test_node_id_serialization() {
        let id = NodeId::new(123);
        let serialized = bincode::serialize(&id).unwrap();
        let deserialized: NodeId = bincode::deserialize(&serialized).unwrap();
        assert_eq!(id, deserialized);
    }
}
