//! Quad index structures for RDF datasets (named graphs)
//!
//! This module provides quad indexes (GSPO, GPOS, GOSP) for efficient
//! querying of RDF datasets with named graphs, complementing the triple
//! indexes (SPO, POS, OSP).

use crate::btree::BTree;
use crate::dictionary::NodeId;
use crate::error::Result;
use crate::index::triple::EmptyValue;
use crate::storage::BufferPool;
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

    /// Insert a quad into all indexes
    pub fn insert(&mut self, quad: Quad) -> Result<()> {
        self.gspo.insert(quad.to_gspo_key(), EmptyValue)?;
        self.gpos.insert(quad.to_gpos_key(), EmptyValue)?;
        self.gosp.insert(quad.to_gosp_key(), EmptyValue)?;
        Ok(())
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

    /// Query quads with pattern matching
    ///
    /// Uses the most selective index based on which components are specified:
    /// - If graph+subject specified: use GSPO
    /// - If graph+predicate specified: use GPOS
    /// - If graph+object specified: use GOSP
    /// - If only graph specified: use GSPO (ordered by subject)
    pub fn query_pattern(
        &self,
        graph: Option<NodeId>,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
    ) -> Result<Vec<Quad>> {
        // Select optimal index based on pattern
        let quads = if graph.is_some() && subject.is_some() {
            // Use GSPO index
            self.query_gspo(graph, subject, predicate, object)?
        } else if graph.is_some() && predicate.is_some() {
            // Use GPOS index
            self.query_gpos(graph, predicate, object, subject)?
        } else if graph.is_some() && object.is_some() {
            // Use GOSP index
            self.query_gosp(graph, object, subject, predicate)?
        } else if graph.is_some() {
            // Use GSPO index (ordered by subject)
            self.query_gspo(graph, None, None, None)?
        } else {
            // No graph specified - scan all indexes (less efficient)
            self.query_all_graphs(subject, predicate, object)?
        };

        Ok(quads)
    }

    /// Query using GSPO index
    fn query_gspo(
        &self,
        graph: Option<NodeId>,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
    ) -> Result<Vec<Quad>> {
        let iter = self.gspo.range_scan(None, None)?;
        let mut results = Vec::new();

        for item in iter {
            let (key, _value) = item?;
            let quad = Quad::new(key.0, key.1, key.2, key.3);
            if self.matches_pattern(&quad, graph, subject, predicate, object) {
                results.push(quad);
            }
        }

        Ok(results)
    }

    /// Query using GPOS index
    fn query_gpos(
        &self,
        graph: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
        subject: Option<NodeId>,
    ) -> Result<Vec<Quad>> {
        let iter = self.gpos.range_scan(None, None)?;
        let mut results = Vec::new();

        for item in iter {
            let (key, _value) = item?;
            let quad = Quad::new(key.0, key.3, key.1, key.2);
            if self.matches_pattern(&quad, graph, subject, predicate, object) {
                results.push(quad);
            }
        }

        Ok(results)
    }

    /// Query using GOSP index
    fn query_gosp(
        &self,
        graph: Option<NodeId>,
        object: Option<NodeId>,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
    ) -> Result<Vec<Quad>> {
        let iter = self.gosp.range_scan(None, None)?;
        let mut results = Vec::new();

        for item in iter {
            let (key, _value) = item?;
            let quad = Quad::new(key.0, key.2, key.3, key.1);
            if self.matches_pattern(&quad, graph, subject, predicate, object) {
                results.push(quad);
            }
        }

        Ok(results)
    }

    /// Query across all graphs (less efficient)
    fn query_all_graphs(
        &self,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
    ) -> Result<Vec<Quad>> {
        // Use GSPO and scan all entries
        self.query_gspo(None, subject, predicate, object)
    }

    /// Check if a quad matches the given pattern
    fn matches_pattern(
        &self,
        quad: &Quad,
        graph: Option<NodeId>,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
    ) -> bool {
        if let Some(g) = graph {
            if quad.graph != g {
                return false;
            }
        }
        if let Some(s) = subject {
            if quad.subject != s {
                return false;
            }
        }
        if let Some(p) = predicate {
            if quad.predicate != p {
                return false;
            }
        }
        if let Some(o) = object {
            if quad.object != o {
                return false;
            }
        }
        true
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
