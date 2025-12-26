//! RDF Triple representation with encoded NodeIds

use crate::dictionary::NodeId;
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::fmt;

/// RDF Triple with encoded NodeIds
///
/// Represents an RDF statement (subject, predicate, object) where
/// each component is encoded as a NodeId for efficient storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub struct Triple {
    /// Subject node ID
    pub subject: NodeId,
    /// Predicate node ID
    pub predicate: NodeId,
    /// Object node ID
    pub object: NodeId,
}

impl Triple {
    /// Create a new triple
    pub const fn new(subject: NodeId, predicate: NodeId, object: NodeId) -> Self {
        Triple {
            subject,
            predicate,
            object,
        }
    }

    /// Get SPO ordering (Subject, Predicate, Object)
    pub const fn spo(&self) -> (NodeId, NodeId, NodeId) {
        (self.subject, self.predicate, self.object)
    }

    /// Get POS ordering (Predicate, Object, Subject)
    pub const fn pos(&self) -> (NodeId, NodeId, NodeId) {
        (self.predicate, self.object, self.subject)
    }

    /// Get OSP ordering (Object, Subject, Predicate)
    pub const fn osp(&self) -> (NodeId, NodeId, NodeId) {
        (self.object, self.subject, self.predicate)
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.subject, self.predicate, self.object)
    }
}

/// SPO composite key for B+Tree indexing
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Encode, Decode,
)]
pub struct SpoKey(pub NodeId, pub NodeId, pub NodeId);

impl From<Triple> for SpoKey {
    fn from(triple: Triple) -> Self {
        let (s, p, o) = triple.spo();
        SpoKey(s, p, o)
    }
}

/// POS composite key for B+Tree indexing
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Encode, Decode,
)]
pub struct PosKey(pub NodeId, pub NodeId, pub NodeId);

impl From<Triple> for PosKey {
    fn from(triple: Triple) -> Self {
        let (p, o, s) = triple.pos();
        PosKey(p, o, s)
    }
}

/// OSP composite key for B+Tree indexing
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Encode, Decode,
)]
pub struct OspKey(pub NodeId, pub NodeId, pub NodeId);

impl From<Triple> for OspKey {
    fn from(triple: Triple) -> Self {
        let (o, s, p) = triple.osp();
        OspKey(o, s, p)
    }
}

/// Empty value type for index (we only need the key)
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Encode, Decode,
)]
pub struct EmptyValue;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_creation() {
        let s = NodeId::new(1);
        let p = NodeId::new(2);
        let o = NodeId::new(3);

        let triple = Triple::new(s, p, o);

        assert_eq!(triple.subject, s);
        assert_eq!(triple.predicate, p);
        assert_eq!(triple.object, o);
    }

    #[test]
    fn test_triple_orderings() {
        let triple = Triple::new(NodeId::new(10), NodeId::new(20), NodeId::new(30));

        assert_eq!(
            triple.spo(),
            (NodeId::new(10), NodeId::new(20), NodeId::new(30))
        );
        assert_eq!(
            triple.pos(),
            (NodeId::new(20), NodeId::new(30), NodeId::new(10))
        );
        assert_eq!(
            triple.osp(),
            (NodeId::new(30), NodeId::new(10), NodeId::new(20))
        );
    }

    #[test]
    fn test_spo_key() {
        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));

        let key: SpoKey = triple.into();
        assert_eq!(key, SpoKey(NodeId::new(1), NodeId::new(2), NodeId::new(3)));
    }

    #[test]
    fn test_pos_key() {
        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));

        let key: PosKey = triple.into();
        assert_eq!(key, PosKey(NodeId::new(2), NodeId::new(3), NodeId::new(1)));
    }

    #[test]
    fn test_osp_key() {
        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));

        let key: OspKey = triple.into();
        assert_eq!(key, OspKey(NodeId::new(3), NodeId::new(1), NodeId::new(2)));
    }

    #[test]
    fn test_key_ordering() {
        let key1 = SpoKey(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let key2 = SpoKey(NodeId::new(1), NodeId::new(2), NodeId::new(4));
        let key3 = SpoKey(NodeId::new(1), NodeId::new(3), NodeId::new(1));

        assert!(key1 < key2);
        assert!(key2 < key3);
    }

    #[test]
    fn test_triple_serialization() {
        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));

        let serialized = bincode::encode_to_vec(triple, bincode::config::standard()).unwrap();
        let deserialized: Triple =
            bincode::decode_from_slice(&serialized, bincode::config::standard())
                .unwrap()
                .0;

        assert_eq!(triple, deserialized);
    }
}
