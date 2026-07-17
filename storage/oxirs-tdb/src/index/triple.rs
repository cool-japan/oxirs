//! RDF Triple representation with encoded NodeIds

use crate::dictionary::NodeId;
use serde::{Deserialize, Serialize};
use std::fmt;

/// RDF Triple with encoded NodeIds
///
/// Represents an RDF statement (subject, predicate, object) where
/// each component is encoded as a NodeId for efficient storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SpoKey(pub NodeId, pub NodeId, pub NodeId);

impl From<Triple> for SpoKey {
    fn from(triple: Triple) -> Self {
        let (s, p, o) = triple.spo();
        SpoKey(s, p, o)
    }
}

/// POS composite key for B+Tree indexing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PosKey(pub NodeId, pub NodeId, pub NodeId);

impl From<Triple> for PosKey {
    fn from(triple: Triple) -> Self {
        let (p, o, s) = triple.pos();
        PosKey(p, o, s)
    }
}

/// OSP composite key for B+Tree indexing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OspKey(pub NodeId, pub NodeId, pub NodeId);

impl From<Triple> for OspKey {
    fn from(triple: Triple) -> Self {
        let (o, s, p) = triple.osp();
        OspKey(o, s, p)
    }
}

/// Empty value type for index (we only need the key)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EmptyValue;

/// Compute an inclusive-start / exclusive-end key-component range that bounds
/// the longest leading run of *specified* (`Some`) components of an index whose
/// key order is `components` (already reordered into the index's own ordering).
///
/// The returned arrays are the raw component tuples for the start and end keys.
/// Any interior or trailing wildcards (a `None` after the leading run, or bound
/// components that follow a wildcard) are left for the caller to apply as a
/// residual filter. Returns `(None, None)` when the very first component is a
/// wildcard, meaning a full index scan is required.
///
/// [`NodeId::NULL`] (`0`) is used as the minimum sentinel for trailing
/// components; because dictionary ids start at [`NodeId::FIRST`] (`1`), a real
/// key component is never `0`, so `(v0, .., v_{k-1}, 0, ..)` is a correct
/// inclusive lower bound and `(v0, .., v_{k-1}.next(), 0, ..)` a correct
/// exclusive upper bound for the leading prefix `v0..v_{k-1}`.
pub(crate) fn prefix_bounds<const N: usize>(
    components: [Option<NodeId>; N],
) -> (Option<[NodeId; N]>, Option<[NodeId; N]>) {
    let mut prefix = 0usize;
    while prefix < N && components[prefix].is_some() {
        prefix += 1;
    }
    if prefix == 0 {
        return (None, None);
    }

    let mut start = [NodeId::NULL; N];
    for (slot, comp) in start.iter_mut().zip(components.iter()).take(prefix) {
        if let Some(value) = comp {
            *slot = *value;
        }
    }

    let mut end = start;
    end[prefix - 1] = start[prefix - 1].next();
    (Some(start), Some(end))
}

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
    fn test_prefix_bounds_full_prefix() {
        let (start, end) = prefix_bounds([
            Some(NodeId::new(3)),
            Some(NodeId::new(4)),
            Some(NodeId::new(5)),
        ]);
        assert_eq!(
            start,
            Some([NodeId::new(3), NodeId::new(4), NodeId::new(5)])
        );
        // Exclusive upper bound bumps the last bound component.
        assert_eq!(end, Some([NodeId::new(3), NodeId::new(4), NodeId::new(6)]));
    }

    #[test]
    fn test_prefix_bounds_partial_prefix() {
        // Only the leading run (s) is bounded; a wildcard breaks the run so the
        // trailing bound object must be handled by a residual filter.
        let (start, end) = prefix_bounds([Some(NodeId::new(7)), None, Some(NodeId::new(9))]);
        assert_eq!(start, Some([NodeId::new(7), NodeId::NULL, NodeId::NULL]));
        assert_eq!(end, Some([NodeId::new(8), NodeId::NULL, NodeId::NULL]));
    }

    #[test]
    fn test_prefix_bounds_full_scan() {
        let (start, end) = prefix_bounds::<3>([None, None, None]);
        assert!(start.is_none());
        assert!(end.is_none());
    }

    #[test]
    fn test_triple_serialization() {
        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));

        let serialized =
            oxicode::serde::encode_to_vec(&triple, oxicode::config::standard()).unwrap();
        let deserialized: Triple =
            oxicode::serde::decode_from_slice(&serialized, oxicode::config::standard())
                .unwrap()
                .0;

        assert_eq!(triple, deserialized);
    }
}
