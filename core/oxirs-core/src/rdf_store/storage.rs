//! RDF storage backends and implementations

use crate::indexing::UltraIndex;
use crate::model::*;
use crate::optimization::RdfArena;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

/// Storage backend for RDF quads
#[derive(Debug)]
pub enum StorageBackend {
    /// Ultra-high performance in-memory storage
    UltraMemory(Arc<UltraIndex>, Arc<RdfArena>),
    /// Legacy in-memory storage using collections
    Memory(Arc<RwLock<MemoryStorage>>),
    /// File-based storage (future: will use disk-backed storage)
    Persistent(Arc<RwLock<MemoryStorage>>, std::path::PathBuf),
}

/// In-memory storage implementation
#[derive(Debug, Clone, Default)]
pub struct MemoryStorage {
    /// All quads in the store
    pub quads: BTreeSet<Quad>,
    /// Index by subject for efficient lookups
    subject_index: BTreeMap<Subject, BTreeSet<Quad>>,
    /// Index by predicate for efficient lookups
    predicate_index: BTreeMap<Predicate, BTreeSet<Quad>>,
    /// Index by object for efficient lookups
    object_index: BTreeMap<Object, BTreeSet<Quad>>,
    /// Index by graph name for efficient lookups
    graph_index: BTreeMap<GraphName, BTreeSet<Quad>>,
    /// Named graphs in the dataset
    pub named_graphs: BTreeSet<NamedNode>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        MemoryStorage {
            quads: BTreeSet::new(),
            subject_index: BTreeMap::new(),
            predicate_index: BTreeMap::new(),
            object_index: BTreeMap::new(),
            graph_index: BTreeMap::new(),
            named_graphs: BTreeSet::new(),
        }
    }

    pub fn insert_quad(&mut self, quad: Quad) -> bool {
        let is_new = self.quads.insert(quad.clone());

        if is_new {
            // Update indexes
            self.subject_index
                .entry(quad.subject().clone())
                .or_default()
                .insert(quad.clone());

            self.predicate_index
                .entry(quad.predicate().clone())
                .or_default()
                .insert(quad.clone());

            self.object_index
                .entry(quad.object().clone())
                .or_default()
                .insert(quad.clone());

            self.graph_index
                .entry(quad.graph_name().clone())
                .or_default()
                .insert(quad.clone());

            // Add to named graphs if not default graph
            if let GraphName::NamedNode(graph_name) = quad.graph_name() {
                self.named_graphs.insert(graph_name.clone());
            }
        }

        is_new
    }

    pub fn remove_quad(&mut self, quad: &Quad) -> bool {
        let was_present = self.quads.remove(quad);

        if was_present {
            // Update indexes
            if let Some(subject_quads) = self.subject_index.get_mut(quad.subject()) {
                subject_quads.remove(quad);
                if subject_quads.is_empty() {
                    self.subject_index.remove(quad.subject());
                }
            }

            if let Some(predicate_quads) = self.predicate_index.get_mut(quad.predicate()) {
                predicate_quads.remove(quad);
                if predicate_quads.is_empty() {
                    self.predicate_index.remove(quad.predicate());
                }
            }

            if let Some(object_quads) = self.object_index.get_mut(quad.object()) {
                object_quads.remove(quad);
                if object_quads.is_empty() {
                    self.object_index.remove(quad.object());
                }
            }

            if let Some(graph_quads) = self.graph_index.get_mut(quad.graph_name()) {
                graph_quads.remove(quad);
                if graph_quads.is_empty() {
                    self.graph_index.remove(quad.graph_name());
                    // Remove from named graphs if it was a named graph
                    if let GraphName::NamedNode(graph_name) = quad.graph_name() {
                        self.named_graphs.remove(graph_name);
                    }
                }
            }
        }

        was_present
    }

    pub fn contains_quad(&self, quad: &Quad) -> bool {
        self.quads.contains(quad)
    }

    #[allow(dead_code)]
    fn iter_quads(&self) -> impl Iterator<Item = &Quad> {
        self.quads.iter()
    }

    pub fn query_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Vec<Quad> {
        // Start with the most selective index
        let mut candidates: Option<BTreeSet<Quad>> = None;

        // Use subject index if provided
        if let Some(s) = subject {
            if let Some(subject_quads) = self.subject_index.get(s) {
                candidates = Some(subject_quads.clone());
            } else {
                return Vec::new(); // No quads with this subject
            }
        }

        // Intersect with predicate index if provided
        if let Some(p) = predicate {
            if let Some(predicate_quads) = self.predicate_index.get(p) {
                if let Some(ref mut cand) = candidates {
                    *cand = cand.intersection(predicate_quads).cloned().collect();
                } else {
                    candidates = Some(predicate_quads.clone());
                }
            } else {
                return Vec::new(); // No quads with this predicate
            }
        }

        // Intersect with object index if provided
        if let Some(o) = object {
            if let Some(object_quads) = self.object_index.get(o) {
                if let Some(ref mut cand) = candidates {
                    *cand = cand.intersection(object_quads).cloned().collect();
                } else {
                    candidates = Some(object_quads.clone());
                }
            } else {
                return Vec::new(); // No quads with this object
            }
        }

        // Intersect with graph index if provided
        if let Some(g) = graph_name {
            if let Some(graph_quads) = self.graph_index.get(g) {
                if let Some(ref mut cand) = candidates {
                    *cand = cand.intersection(graph_quads).cloned().collect();
                } else {
                    candidates = Some(graph_quads.clone());
                }
            } else {
                return Vec::new(); // No quads in this graph
            }
        }

        // If no specific criteria provided, return all quads
        let quads = candidates.unwrap_or_else(|| self.quads.clone());
        quads.into_iter().collect()
    }

    pub fn len(&self) -> usize {
        self.quads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }
}
