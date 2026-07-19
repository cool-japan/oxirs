//! Quad index structures for RDF datasets (named graphs)
//!
//! This module provides quad indexes (GSPO, GPOS, GOSP) for efficient
//! querying of RDF datasets with named graphs, complementing the triple
//! indexes (SPO, POS, OSP).

use crate::btree::iterator::BTreeIterator;
use crate::btree::BTree;
use crate::dictionary::NodeId;
use crate::error::Result;
use crate::index::triple::{prefix_bounds, EmptyValue};
use crate::storage::{BufferPool, PageId};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A quad representing (graph, subject, predicate, object)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Quad {
    /// Graph (named graph identifier)
    pub graph: NodeId,
    /// Subject node
    pub subject: NodeId,
    /// Predicate node
    pub predicate: NodeId,
    /// Object node
    pub object: NodeId,
}

impl Quad {
    /// Create a new quad
    pub fn new(graph: NodeId, subject: NodeId, predicate: NodeId, object: NodeId) -> Self {
        Self {
            graph,
            subject,
            predicate,
            object,
        }
    }

    /// Convert to GSPO key for indexing
    pub fn to_gspo_key(&self) -> GspoKey {
        GspoKey(self.graph, self.subject, self.predicate, self.object)
    }

    /// Convert to GPOS key for indexing
    pub fn to_gpos_key(&self) -> GposKey {
        GposKey(self.graph, self.predicate, self.object, self.subject)
    }

    /// Convert to GOSP key for indexing
    pub fn to_gosp_key(&self) -> GospKey {
        GospKey(self.graph, self.object, self.subject, self.predicate)
    }
}

/// GSPO index key (Graph, Subject, Predicate, Object)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GspoKey(pub NodeId, pub NodeId, pub NodeId, pub NodeId);

/// GPOS index key (Graph, Predicate, Object, Subject)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GposKey(pub NodeId, pub NodeId, pub NodeId, pub NodeId);

/// GOSP index key (Graph, Object, Subject, Predicate)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GospKey(pub NodeId, pub NodeId, pub NodeId, pub NodeId);

/// Quad indexes for named graph support
///
/// Provides three complementary indexes:
/// - GSPO: Efficient for queries with graph and subject specified
/// - GPOS: Efficient for queries with graph and predicate specified
/// - GOSP: Efficient for queries with graph and object specified
pub struct QuadIndexes {
    /// GSPO index (Graph, Subject, Predicate, Object)
    gspo: BTree<GspoKey, EmptyValue>,
    /// GPOS index (Graph, Predicate, Object, Subject)
    gpos: BTree<GposKey, EmptyValue>,
    /// GOSP index (Graph, Object, Subject, Predicate)
    gosp: BTree<GospKey, EmptyValue>,
    /// Buffer pool for storage
    buffer_pool: Arc<BufferPool>,
}

impl QuadIndexes {
    /// Create new quad indexes
    pub fn new(buffer_pool: Arc<BufferPool>) -> Self {
        Self {
            gspo: BTree::new(buffer_pool.clone()),
            gpos: BTree::new(buffer_pool.clone()),
            gosp: BTree::new(buffer_pool.clone()),
            buffer_pool,
        }
    }

    /// Reconstruct the quad indexes from persisted B+Tree root pages (on reopen).
    ///
    /// A `None` root rebuilds an empty tree; a `Some(page)` root rehydrates the
    /// on-disk B+Tree via [`BTree::from_root`]. The roots are recorded in, and
    /// restored from, the store superblock (see
    /// [`Superblock`](crate::storage::superblock::Superblock)).
    pub fn from_roots(
        buffer_pool: Arc<BufferPool>,
        gspo_root: Option<PageId>,
        gpos_root: Option<PageId>,
        gosp_root: Option<PageId>,
    ) -> Self {
        let gspo = match gspo_root {
            Some(page) => BTree::from_root(buffer_pool.clone(), page),
            None => BTree::new(buffer_pool.clone()),
        };
        let gpos = match gpos_root {
            Some(page) => BTree::from_root(buffer_pool.clone(), page),
            None => BTree::new(buffer_pool.clone()),
        };
        let gosp = match gosp_root {
            Some(page) => BTree::from_root(buffer_pool.clone(), page),
            None => BTree::new(buffer_pool.clone()),
        };
        Self {
            gspo,
            gpos,
            gosp,
            buffer_pool,
        }
    }

    /// Current root page of the GSPO index (for superblock persistence).
    pub fn gspo_root(&self) -> Option<PageId> {
        self.gspo.root_page()
    }

    /// Current root page of the GPOS index (for superblock persistence).
    pub fn gpos_root(&self) -> Option<PageId> {
        self.gpos.root_page()
    }

    /// Current root page of the GOSP index (for superblock persistence).
    pub fn gosp_root(&self) -> Option<PageId> {
        self.gosp.root_page()
    }

    /// Insert a quad into all indexes.
    ///
    /// Returns `true` if the quad was newly added and `false` if it was already
    /// present, so the caller can maintain an accurate quad count.
    pub fn insert(&mut self, quad: Quad) -> Result<bool> {
        let is_new = self.gspo.insert(quad.to_gspo_key(), EmptyValue)?.is_none();
        self.gpos.insert(quad.to_gpos_key(), EmptyValue)?;
        self.gosp.insert(quad.to_gosp_key(), EmptyValue)?;
        Ok(is_new)
    }

    /// Bulk-insert quads with per-index sorted, sequential leaf appends (F6).
    ///
    /// Mirrors [`TripleIndexes::insert_sorted`](crate::index::TripleIndexes::insert_sorted):
    /// each index is fed the whole batch pre-sorted in its own key order — GSPO
    /// by `(g, s, p, o)`, GPOS by `(g, p, o, s)`, GOSP by `(g, o, s, p)` — so the
    /// B+Trees see monotonically non-decreasing keys and append to the right-most
    /// leaf. Returns the number of genuinely new quads (batch/tree duplicates
    /// excluded; GSPO carries the authoritative new/duplicate signal).
    pub fn insert_sorted(&mut self, quads: &[Quad]) -> Result<usize> {
        let mut gspo_keys: Vec<GspoKey> = quads.iter().map(|q| q.to_gspo_key()).collect();
        gspo_keys.sort_unstable();
        let mut new_count = 0usize;
        for key in gspo_keys {
            if self.gspo.insert(key, EmptyValue)?.is_none() {
                new_count += 1;
            }
        }

        let mut gpos_keys: Vec<GposKey> = quads.iter().map(|q| q.to_gpos_key()).collect();
        gpos_keys.sort_unstable();
        for key in gpos_keys {
            self.gpos.insert(key, EmptyValue)?;
        }

        let mut gosp_keys: Vec<GospKey> = quads.iter().map(|q| q.to_gosp_key()).collect();
        gosp_keys.sort_unstable();
        for key in gosp_keys {
            self.gosp.insert(key, EmptyValue)?;
        }

        Ok(new_count)
    }

    /// Delete a quad from all indexes
    pub fn delete(&mut self, quad: Quad) -> Result<bool> {
        let gspo_deleted = self.gspo.delete(&quad.to_gspo_key())?.is_some();
        let gpos_deleted = self.gpos.delete(&quad.to_gpos_key())?.is_some();
        let gosp_deleted = self.gosp.delete(&quad.to_gosp_key())?.is_some();

        // All three should be consistent
        Ok(gspo_deleted && gpos_deleted && gosp_deleted)
    }

    /// Check if a quad exists
    pub fn contains(&self, quad: &Quad) -> Result<bool> {
        Ok(self.gspo.search(&quad.to_gspo_key())?.is_some())
    }

    /// Open a lazy, streaming scan over the quads matching a pattern.
    ///
    /// Selects the optimal index based on which components are specified and
    /// bounds the range scan to the pattern's leading prefix:
    /// - graph+subject bound: GSPO, prefix `(g, s, ..)`
    /// - graph+predicate bound (no subject): GPOS, prefix `(g, p, ..)`
    /// - graph+object bound (no subject/predicate): GOSP, prefix `(g, o, ..)`
    /// - only graph bound: GSPO, prefix `(g, ..)`
    /// - graph unbound: a full index scan filtered by the residual predicate
    ///   (cross-graph patterns are inherently less selective without an
    ///   SPO-across-graphs index).
    ///
    /// The returned [`QuadScan`] yields one [`Quad`] at a time without
    /// materializing the whole result set (F5).
    pub fn scan(
        &self,
        graph: Option<NodeId>,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
    ) -> Result<QuadScan> {
        let inner = if subject.is_some() {
            // GSPO key order: (g, s, p, o).
            let (start, end) = prefix_bounds([graph, subject, predicate, object]);
            QuadScanInner::Gspo(self.gspo.range_scan(
                start.map(|k| GspoKey(k[0], k[1], k[2], k[3])),
                end.map(|k| GspoKey(k[0], k[1], k[2], k[3])),
            )?)
        } else if predicate.is_some() {
            // GPOS key order: (g, p, o, s).
            let (start, end) = prefix_bounds([graph, predicate, object, subject]);
            QuadScanInner::Gpos(self.gpos.range_scan(
                start.map(|k| GposKey(k[0], k[1], k[2], k[3])),
                end.map(|k| GposKey(k[0], k[1], k[2], k[3])),
            )?)
        } else if object.is_some() {
            // GOSP key order: (g, o, s, p).
            let (start, end) = prefix_bounds([graph, object, subject, predicate]);
            QuadScanInner::Gosp(self.gosp.range_scan(
                start.map(|k| GospKey(k[0], k[1], k[2], k[3])),
                end.map(|k| GospKey(k[0], k[1], k[2], k[3])),
            )?)
        } else {
            // Only graph may be bound: GSPO prefix on the graph column.
            let (start, end) = prefix_bounds([graph, None, None, None]);
            QuadScanInner::Gspo(self.gspo.range_scan(
                start.map(|k| GspoKey(k[0], k[1], k[2], k[3])),
                end.map(|k| GspoKey(k[0], k[1], k[2], k[3])),
            )?)
        };

        Ok(QuadScan {
            inner,
            graph,
            subject,
            predicate,
            object,
        })
    }

    /// Query quads with pattern matching, materializing the results.
    ///
    /// A thin `Vec`-collecting wrapper around [`QuadIndexes::scan`]; prefer
    /// `scan` for large result sets to avoid buffering everything in memory.
    pub fn query_pattern(
        &self,
        graph: Option<NodeId>,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
    ) -> Result<Vec<Quad>> {
        self.scan(graph, subject, predicate, object)?.collect()
    }

    /// Get GSPO index reference
    pub fn gspo(&self) -> &BTree<GspoKey, EmptyValue> {
        &self.gspo
    }

    /// Get GPOS index reference
    pub fn gpos(&self) -> &BTree<GposKey, EmptyValue> {
        &self.gpos
    }

    /// Get GOSP index reference
    pub fn gosp(&self) -> &BTree<GospKey, EmptyValue> {
        &self.gosp
    }
}

/// The selected quad index iterator backing a [`QuadScan`].
///
/// Each variant reconstructs a [`Quad`] from that index's key ordering.
enum QuadScanInner {
    /// GSPO-ordered key stream: key is `(g, s, p, o)`.
    Gspo(BTreeIterator<GspoKey, EmptyValue>),
    /// GPOS-ordered key stream: key is `(g, p, o, s)`.
    Gpos(BTreeIterator<GposKey, EmptyValue>),
    /// GOSP-ordered key stream: key is `(g, o, s, p)`.
    Gosp(BTreeIterator<GospKey, EmptyValue>),
}

/// A lazy, streaming iterator over quads matching a pattern.
///
/// Created by [`QuadIndexes::scan`]. It streams keys out of the selected B+Tree
/// one leaf page at a time (never materializing the full result set),
/// reconstructs each [`Quad`] from the index ordering, and applies a residual
/// filter for any pattern component not covered by the bounded prefix.
pub struct QuadScan {
    inner: QuadScanInner,
    graph: Option<NodeId>,
    subject: Option<NodeId>,
    predicate: Option<NodeId>,
    object: Option<NodeId>,
}

impl QuadScan {
    /// Residual filter for pattern components not covered by the scan prefix.
    fn matches(&self, quad: &Quad) -> bool {
        self.graph.map_or(true, |g| quad.graph == g)
            && self.subject.map_or(true, |s| quad.subject == s)
            && self.predicate.map_or(true, |p| quad.predicate == p)
            && self.object.map_or(true, |o| quad.object == o)
    }
}

impl Iterator for QuadScan {
    type Item = Result<Quad>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = match &mut self.inner {
                QuadScanInner::Gspo(it) => it
                    .next()
                    .map(|r| r.map(|(k, _)| Quad::new(k.0, k.1, k.2, k.3))),
                // GposKey is (g, p, o, s) -> Quad (g, s, p, o).
                QuadScanInner::Gpos(it) => it
                    .next()
                    .map(|r| r.map(|(k, _)| Quad::new(k.0, k.3, k.1, k.2))),
                // GospKey is (g, o, s, p) -> Quad (g, s, p, o).
                QuadScanInner::Gosp(it) => it
                    .next()
                    .map(|r| r.map(|(k, _)| Quad::new(k.0, k.2, k.3, k.1))),
            };

            match next {
                None => return None,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(quad)) => {
                    if self.matches(&quad) {
                        return Some(Ok(quad));
                    }
                    // Otherwise keep scanning (residual filter rejected it).
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::FileManager;
    use std::env;
    use tempfile::TempDir;

    fn setup_indexes() -> QuadIndexes {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));
        QuadIndexes::new(buffer_pool)
    }

    #[test]
    fn test_quad_creation() {
        let quad = Quad::new(
            NodeId::from(0),
            NodeId::from(1),
            NodeId::from(2),
            NodeId::from(3),
        );
        assert_eq!(quad.graph, NodeId::from(0));
        assert_eq!(quad.subject, NodeId::from(1));
        assert_eq!(quad.predicate, NodeId::from(2));
        assert_eq!(quad.object, NodeId::from(3));
    }

    #[test]
    fn test_quad_indexes_creation() {
        let _indexes = setup_indexes();
    }

    #[test]
    fn test_quad_insert() {
        let mut indexes = setup_indexes();
        let quad = Quad::new(
            NodeId::from(0),
            NodeId::from(1),
            NodeId::from(2),
            NodeId::from(3),
        );
        indexes.insert(quad).unwrap();
        assert!(indexes.contains(&quad).unwrap());
    }

    #[test]
    fn test_quad_delete() {
        let mut indexes = setup_indexes();
        let quad = Quad::new(
            NodeId::from(0),
            NodeId::from(1),
            NodeId::from(2),
            NodeId::from(3),
        );
        indexes.insert(quad).unwrap();
        assert!(indexes.contains(&quad).unwrap());

        let deleted = indexes.delete(quad).unwrap();
        assert!(deleted);
        assert!(!indexes.contains(&quad).unwrap());
    }

    #[test]
    fn test_quad_query_by_graph() {
        let mut indexes = setup_indexes();

        // Insert quads in different graphs
        indexes
            .insert(Quad::new(
                NodeId::from(0),
                NodeId::from(1),
                NodeId::from(2),
                NodeId::from(3),
            ))
            .unwrap();
        indexes
            .insert(Quad::new(
                NodeId::from(0),
                NodeId::from(4),
                NodeId::from(5),
                NodeId::from(6),
            ))
            .unwrap();
        indexes
            .insert(Quad::new(
                NodeId::from(1),
                NodeId::from(7),
                NodeId::from(8),
                NodeId::from(9),
            ))
            .unwrap();

        // Query for graph 0
        let results = indexes
            .query_pattern(Some(NodeId::from(0)), None, None, None)
            .unwrap();
        assert_eq!(results.len(), 2);
        for quad in &results {
            assert_eq!(quad.graph, NodeId::from(0));
        }
    }

    #[test]
    fn test_quad_query_full_pattern() {
        let mut indexes = setup_indexes();

        let quad = Quad::new(
            NodeId::from(0),
            NodeId::from(1),
            NodeId::from(2),
            NodeId::from(3),
        );
        indexes.insert(quad).unwrap();

        let results = indexes
            .query_pattern(
                Some(NodeId::from(0)),
                Some(NodeId::from(1)),
                Some(NodeId::from(2)),
                Some(NodeId::from(3)),
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], quad);
    }

    #[test]
    fn test_quad_query_by_predicate() {
        let mut indexes = setup_indexes();

        // Insert quads with same graph and predicate
        indexes
            .insert(Quad::new(
                NodeId::from(0),
                NodeId::from(1),
                NodeId::from(2),
                NodeId::from(3),
            ))
            .unwrap();
        indexes
            .insert(Quad::new(
                NodeId::from(0),
                NodeId::from(4),
                NodeId::from(2),
                NodeId::from(6),
            ))
            .unwrap();
        indexes
            .insert(Quad::new(
                NodeId::from(0),
                NodeId::from(7),
                NodeId::from(8),
                NodeId::from(9),
            ))
            .unwrap();

        // Query for graph 0, predicate 2
        let results = indexes
            .query_pattern(Some(NodeId::from(0)), None, Some(NodeId::from(2)), None)
            .unwrap();
        assert_eq!(results.len(), 2);
        for quad in &results {
            assert_eq!(quad.graph, NodeId::from(0));
            assert_eq!(quad.predicate, NodeId::from(2));
        }
    }

    #[test]
    fn test_quad_key_conversions() {
        let quad = Quad::new(
            NodeId::from(0),
            NodeId::from(1),
            NodeId::from(2),
            NodeId::from(3),
        );

        let gspo = quad.to_gspo_key();
        assert_eq!(gspo.0, NodeId::from(0));
        assert_eq!(gspo.1, NodeId::from(1));
        assert_eq!(gspo.2, NodeId::from(2));
        assert_eq!(gspo.3, NodeId::from(3));

        let gpos = quad.to_gpos_key();
        assert_eq!(gpos.0, NodeId::from(0));
        assert_eq!(gpos.1, NodeId::from(2));
        assert_eq!(gpos.2, NodeId::from(3));
        assert_eq!(gpos.3, NodeId::from(1));

        let gosp = quad.to_gosp_key();
        assert_eq!(gosp.0, NodeId::from(0));
        assert_eq!(gosp.1, NodeId::from(3));
        assert_eq!(gosp.2, NodeId::from(1));
        assert_eq!(gosp.3, NodeId::from(2));
    }
}
