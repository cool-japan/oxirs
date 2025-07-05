//! Lock-free graph implementation for high-performance concurrent access
//!
//! This module provides a wait-free reader, lock-free writer graph structure
//! using epoch-based memory reclamation and atomic operations.

use super::epoch::{EpochManager, HazardPointer};
use crate::model::{Object, Predicate, Subject, Triple};
use crate::OxirsError;
use crossbeam_epoch::Owned;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Index type for fast triple lookups
/// A lock-free graph node containing triples
struct GraphNode {
    /// The triples stored in this node
    triples: Arc<DashMap<u64, Triple>>,
    /// Version number for optimistic concurrency control
    version: AtomicU64,
    /// Index for SPO (Subject-Predicate-Object) lookups
    spo_index: Arc<DashMap<Subject, DashMap<Predicate, HashSet<Object>>>>,
    /// Index for POS (Predicate-Object-Subject) lookups
    #[allow(dead_code)]
    pos_index: Arc<DashMap<Predicate, DashMap<Object, HashSet<Subject>>>>,
    /// Index for OSP (Object-Subject-Predicate) lookups
    osp_index: Arc<DashMap<Object, DashMap<Subject, HashSet<Predicate>>>>,
}

impl GraphNode {
    fn new() -> Self {
        Self {
            triples: Arc::new(DashMap::new()),
            version: AtomicU64::new(0),
            spo_index: Arc::new(DashMap::new()),
            pos_index: Arc::new(DashMap::new()),
            osp_index: Arc::new(DashMap::new()),
        }
    }

    fn increment_version(&self) -> u64 {
        self.version.fetch_add(1, Ordering::Release)
    }
}

/// A concurrent, lock-free graph implementation
pub struct ConcurrentGraph {
    /// The current graph state
    graph: Arc<HazardPointer<GraphNode>>,
    /// Epoch manager for memory reclamation
    epoch_manager: Arc<EpochManager>,
    /// Triple counter
    triple_count: Arc<AtomicUsize>,
    /// Operation counter for metrics
    operation_count: Arc<AtomicU64>,
}

impl ConcurrentGraph {
    /// Create a new concurrent graph
    pub fn new() -> Self {
        let graph_node = GraphNode::new();
        Self {
            graph: Arc::new(HazardPointer::new(graph_node)),
            epoch_manager: Arc::new(EpochManager::new()),
            triple_count: Arc::new(AtomicUsize::new(0)),
            operation_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Insert a triple into the graph (lock-free)
    pub fn insert(&self, triple: Triple) -> Result<bool, OxirsError> {
        let guard = self.epoch_manager.pin();
        self.operation_count.fetch_add(1, Ordering::Relaxed);

        // Generate a unique ID for this triple
        let triple_id = self.hash_triple(&triple);

        // Load current graph state
        let current = self.graph.load(&guard);
        let graph_node = unsafe {
            current
                .as_ref()
                .ok_or_else(|| OxirsError::Store("Graph not initialized".to_string()))?
        };

        // Check if triple already exists (wait-free read)
        if graph_node.triples.contains_key(&triple_id) {
            return Ok(false);
        }

        // Insert into main storage
        graph_node.triples.insert(triple_id, triple.clone());

        // Update indices
        self.update_indices_insert(graph_node, &triple);

        // Increment version
        graph_node.increment_version();

        // Update counter
        self.triple_count.fetch_add(1, Ordering::Release);

        Ok(true)
    }

    /// Remove a triple from the graph (lock-free)
    pub fn remove(&self, triple: &Triple) -> Result<bool, OxirsError> {
        let guard = self.epoch_manager.pin();
        self.operation_count.fetch_add(1, Ordering::Relaxed);

        let triple_id = self.hash_triple(triple);

        // Load current graph state
        let current = self.graph.load(&guard);
        let graph_node = unsafe {
            current
                .as_ref()
                .ok_or_else(|| OxirsError::Store("Graph not initialized".to_string()))?
        };

        // Remove from main storage
        if graph_node.triples.remove(&triple_id).is_none() {
            return Ok(false);
        }

        // Update indices
        self.update_indices_remove(graph_node, triple);

        // Increment version
        graph_node.increment_version();

        // Update counter
        self.triple_count.fetch_sub(1, Ordering::Release);

        Ok(true)
    }

    /// Check if a triple exists (wait-free)
    pub fn contains(&self, triple: &Triple) -> bool {
        let guard = self.epoch_manager.pin();
        let triple_id = self.hash_triple(triple);

        if let Some(graph_node) = unsafe { self.graph.load(&guard).as_ref() } {
            graph_node.triples.contains_key(&triple_id)
        } else {
            false
        }
    }

    /// Get the number of triples (wait-free)
    pub fn len(&self) -> usize {
        self.triple_count.load(Ordering::Acquire)
    }

    /// Check if the graph is empty (wait-free)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all triples (wait-free snapshot)
    pub fn iter(&self) -> impl Iterator<Item = Triple> + '_ {
        let guard = self.epoch_manager.pin();
        let snapshot = if let Some(graph_node) = unsafe { self.graph.load(&guard).as_ref() } {
            graph_node
                .triples
                .iter()
                .map(|entry| entry.value().clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        snapshot.into_iter()
    }

    /// Find triples matching a pattern (wait-free)
    pub fn match_pattern(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Vec<Triple> {
        let guard = self.epoch_manager.pin();
        let graph_node = match unsafe { self.graph.load(&guard).as_ref() } {
            Some(node) => node,
            None => return Vec::new(),
        };

        match (subject, predicate, object) {
            // All components specified
            (Some(s), Some(p), Some(o)) => {
                let triple = Triple::new(s.clone(), p.clone(), o.clone());
                if self.contains(&triple) {
                    vec![triple]
                } else {
                    Vec::new()
                }
            }
            // Subject and predicate specified
            (Some(s), Some(p), None) => {
                if let Some(pred_map) = graph_node.spo_index.get(s) {
                    if let Some(obj_set) = pred_map.get(p) {
                        obj_set
                            .iter()
                            .map(|o| Triple::new(s.clone(), p.clone(), o.clone()))
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            }
            // Only subject specified
            (Some(s), None, None) => {
                if let Some(pred_map) = graph_node.spo_index.get(s) {
                    pred_map
                        .iter()
                        .flat_map(|pred_entry| {
                            let p = pred_entry.key().clone();
                            let s = s.clone();
                            pred_entry
                                .value()
                                .iter()
                                .map(move |o| Triple::new(s.clone(), p.clone(), o.clone()))
                                .collect::<Vec<_>>()
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            }
            // Object specified
            (None, None, Some(o)) => {
                if let Some(subj_map) = graph_node.osp_index.get(o) {
                    subj_map
                        .iter()
                        .flat_map(|subj_entry| {
                            let s = subj_entry.key().clone();
                            let o = o.clone();
                            subj_entry
                                .value()
                                .iter()
                                .map(move |p| Triple::new(s.clone(), p.clone(), o.clone()))
                                .collect::<Vec<_>>()
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            }
            // Other patterns - scan all triples
            _ => graph_node
                .triples
                .iter()
                .map(|entry| entry.value().clone())
                .filter(|t| {
                    subject.map_or(true, |s| t.subject() == s)
                        && predicate.map_or(true, |p| t.predicate() == p)
                        && object.map_or(true, |o| t.object() == o)
                })
                .collect(),
        }
    }

    /// Get statistics about the graph
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            triple_count: self.len(),
            operation_count: self.operation_count.load(Ordering::Relaxed),
            current_epoch: self.epoch_manager.current_epoch(),
        }
    }

    /// Force memory reclamation
    pub fn collect(&self) {
        let guard = self.epoch_manager.pin();
        self.epoch_manager.flush(&guard);
        self.epoch_manager.advance();
    }

    // Helper methods

    fn hash_triple(&self, triple: &Triple) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        triple.subject().hash(&mut hasher);
        triple.predicate().hash(&mut hasher);
        triple.object().hash(&mut hasher);
        hasher.finish()
    }

    fn update_indices_insert(&self, graph_node: &GraphNode, triple: &Triple) {
        // Update SPO index
        graph_node
            .spo_index
            .entry(triple.subject().clone())
            .or_default()
            .entry(triple.predicate().clone())
            .or_default()
            .insert(triple.object().clone());

        // Update OSP index
        graph_node
            .osp_index
            .entry(triple.object().clone())
            .or_default()
            .entry(triple.subject().clone())
            .or_default()
            .insert(triple.predicate().clone());
    }

    fn update_indices_remove(&self, graph_node: &GraphNode, triple: &Triple) {
        // Update SPO index
        if let Some(pred_map) = graph_node.spo_index.get_mut(triple.subject()) {
            if let Some(mut obj_set) = pred_map.get_mut(triple.predicate()) {
                obj_set.remove(triple.object());
                if obj_set.is_empty() {
                    drop(obj_set);
                    pred_map.remove(triple.predicate());
                }
            }
            if pred_map.is_empty() {
                drop(pred_map);
                graph_node.spo_index.remove(triple.subject());
            }
        }

        // Update OSP index
        if let Some(subj_map) = graph_node.osp_index.get_mut(triple.object()) {
            if let Some(mut pred_set) = subj_map.get_mut(triple.subject()) {
                pred_set.remove(triple.predicate());
                if pred_set.is_empty() {
                    drop(pred_set);
                    subj_map.remove(triple.subject());
                }
            }
            if subj_map.is_empty() {
                drop(subj_map);
                graph_node.osp_index.remove(triple.object());
            }
        }
    }
}

impl Default for ConcurrentGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the concurrent graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub triple_count: usize,
    pub operation_count: u64,
    pub current_epoch: usize,
}

/// Batch operations for improved performance
impl ConcurrentGraph {
    /// Insert multiple triples in a batch
    pub fn insert_batch(&self, triples: Vec<Triple>) -> Result<usize, OxirsError> {
        let mut inserted = 0;
        for triple in triples {
            if self.insert(triple)? {
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    /// Remove multiple triples in a batch
    pub fn remove_batch(&self, triples: &[Triple]) -> Result<usize, OxirsError> {
        let mut removed = 0;
        for triple in triples {
            if self.remove(triple)? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Clear all triples from the graph
    pub fn clear(&self) -> Result<(), OxirsError> {
        let guard = self.epoch_manager.pin();

        // Create new empty graph node
        let new_node = GraphNode::new();
        self.graph.store(Owned::new(new_node), &guard);

        // Reset counter
        self.triple_count.store(0, Ordering::Release);

        // Force collection of old data
        self.collect();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;

    fn create_test_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(
            Subject::NamedNode(NamedNode::new(s).unwrap()),
            Predicate::NamedNode(NamedNode::new(p).unwrap()),
            Object::NamedNode(NamedNode::new(o).unwrap()),
        )
    }

    #[test]
    fn test_concurrent_insert() {
        let graph = ConcurrentGraph::new();
        let triple = create_test_triple("http://s", "http://p", "http://o");

        assert!(graph.insert(triple.clone()).unwrap());
        assert!(!graph.insert(triple.clone()).unwrap());
        assert_eq!(graph.len(), 1);
        assert!(graph.contains(&triple));
    }

    #[test]
    fn test_concurrent_remove() {
        let graph = ConcurrentGraph::new();
        let triple = create_test_triple("http://s", "http://p", "http://o");

        assert!(graph.insert(triple.clone()).unwrap());
        assert!(graph.remove(&triple).unwrap());
        assert!(!graph.remove(&triple).unwrap());
        assert_eq!(graph.len(), 0);
        assert!(!graph.contains(&triple));
    }

    #[test]
    fn test_pattern_matching() {
        let graph = ConcurrentGraph::new();

        // Insert test data
        let t1 = create_test_triple("http://s1", "http://p1", "http://o1");
        let t2 = create_test_triple("http://s1", "http://p1", "http://o2");
        let t3 = create_test_triple("http://s1", "http://p2", "http://o1");
        let t4 = create_test_triple("http://s2", "http://p1", "http://o1");

        graph.insert(t1.clone()).unwrap();
        graph.insert(t2.clone()).unwrap();
        graph.insert(t3.clone()).unwrap();
        graph.insert(t4.clone()).unwrap();

        // Test subject pattern
        let s1 = Subject::NamedNode(NamedNode::new("http://s1").unwrap());
        let matches = graph.match_pattern(Some(&s1), None, None);
        assert_eq!(matches.len(), 3);

        // Test subject-predicate pattern
        let p1 = Predicate::NamedNode(NamedNode::new("http://p1").unwrap());
        let matches = graph.match_pattern(Some(&s1), Some(&p1), None);
        assert_eq!(matches.len(), 2);

        // Test object pattern
        let o1 = Object::NamedNode(NamedNode::new("http://o1").unwrap());
        let matches = graph.match_pattern(None, None, Some(&o1));
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_concurrent_operations() {
        use std::thread;

        let graph = Arc::new(ConcurrentGraph::new());
        let num_threads = 4;
        let ops_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let graph = graph.clone();
                thread::spawn(move || {
                    for j in 0..ops_per_thread {
                        let triple = create_test_triple(
                            &format!("http://s{}", i),
                            &format!("http://p{}", j),
                            &format!("http://o{}", i * ops_per_thread + j),
                        );
                        graph.insert(triple).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(graph.len(), num_threads * ops_per_thread);
    }

    #[test]
    fn test_batch_operations() {
        let graph = ConcurrentGraph::new();

        let triples: Vec<_> = (0..10)
            .map(|i| create_test_triple(&format!("http://s{}", i), "http://p", "http://o"))
            .collect();

        let inserted = graph.insert_batch(triples.clone()).unwrap();
        assert_eq!(inserted, 10);
        assert_eq!(graph.len(), 10);

        let removed = graph.remove_batch(&triples[0..5]).unwrap();
        assert_eq!(removed, 5);
        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn test_clear() {
        let graph = ConcurrentGraph::new();

        for i in 0..10 {
            let triple = create_test_triple(&format!("http://s{}", i), "http://p", "http://o");
            graph.insert(triple).unwrap();
        }

        assert_eq!(graph.len(), 10);
        graph.clear().unwrap();
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
    }
}
