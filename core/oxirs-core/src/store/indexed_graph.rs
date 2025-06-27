//! High-performance multi-index graph implementation
//!
//! This module provides an indexed graph structure with SPO, POS, and OSP indexes
//! for efficient triple pattern matching. All terms are interned for memory efficiency.

use crate::model::{Triple, Subject, Predicate, Object};
use crate::store::term_interner::TermInterner;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, RwLock};

/// Interned triple representation using term IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InternedTriple {
    pub subject_id: u32,
    pub predicate_id: u32,
    pub object_id: u32,
}

/// Index type for selecting the best index for a query pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    SPO, // Subject -> Predicate -> Object
    POS, // Predicate -> Object -> Subject
    OSP, // Object -> Subject -> Predicate
}

/// Multi-indexed graph with term interning
#[derive(Debug)]
pub struct IndexedGraph {
    /// Term interner for string deduplication
    interner: Arc<TermInterner>,
    /// SPO index: Subject -> Predicate -> Object
    spo_index: Arc<RwLock<BTreeMap<u32, BTreeMap<u32, BTreeSet<u32>>>>>,
    /// POS index: Predicate -> Object -> Subject
    pos_index: Arc<RwLock<BTreeMap<u32, BTreeMap<u32, BTreeSet<u32>>>>>,
    /// OSP index: Object -> Subject -> Predicate
    osp_index: Arc<RwLock<BTreeMap<u32, BTreeMap<u32, BTreeSet<u32>>>>>,
    /// Total number of triples
    triple_count: Arc<RwLock<usize>>,
    /// Index statistics for query optimization
    stats: Arc<RwLock<IndexStats>>,
}

/// Statistics for index usage and performance
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub spo_lookups: usize,
    pub pos_lookups: usize,
    pub osp_lookups: usize,
    pub total_insertions: usize,
    pub total_deletions: usize,
    pub batch_insertions: usize,
}

impl IndexStats {
    /// Get the most frequently used index
    pub fn most_used_index(&self) -> IndexType {
        if self.spo_lookups >= self.pos_lookups && self.spo_lookups >= self.osp_lookups {
            IndexType::SPO
        } else if self.pos_lookups >= self.osp_lookups {
            IndexType::POS
        } else {
            IndexType::OSP
        }
    }
}

impl IndexedGraph {
    /// Create a new indexed graph
    pub fn new() -> Self {
        IndexedGraph {
            interner: Arc::new(TermInterner::new()),
            spo_index: Arc::new(RwLock::new(BTreeMap::new())),
            pos_index: Arc::new(RwLock::new(BTreeMap::new())),
            osp_index: Arc::new(RwLock::new(BTreeMap::new())),
            triple_count: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(IndexStats::default())),
        }
    }

    /// Create a new indexed graph with a custom interner
    pub fn with_interner(interner: Arc<TermInterner>) -> Self {
        IndexedGraph {
            interner,
            spo_index: Arc::new(RwLock::new(BTreeMap::new())),
            pos_index: Arc::new(RwLock::new(BTreeMap::new())),
            osp_index: Arc::new(RwLock::new(BTreeMap::new())),
            triple_count: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(IndexStats::default())),
        }
    }

    /// Insert a triple into the graph
    pub fn insert(&self, triple: &Triple) -> bool {
        // Intern the terms
        let s_id = self.interner.intern_subject(triple.subject());
        let p_id = self.interner.intern_predicate(triple.predicate());
        let o_id = self.interner.intern_object(triple.object());

        let interned = InternedTriple {
            subject_id: s_id,
            predicate_id: p_id,
            object_id: o_id,
        };

        self.insert_interned(interned)
    }

    /// Insert an already interned triple
    fn insert_interned(&self, triple: InternedTriple) -> bool {
        let mut spo = self.spo_index.write().unwrap();
        let mut pos = self.pos_index.write().unwrap();
        let mut osp = self.osp_index.write().unwrap();

        // Check if triple already exists in SPO index
        if let Some(po_map) = spo.get(&triple.subject_id) {
            if let Some(o_set) = po_map.get(&triple.predicate_id) {
                if o_set.contains(&triple.object_id) {
                    return false; // Triple already exists
                }
            }
        }

        // Insert into SPO index
        spo.entry(triple.subject_id)
            .or_insert_with(BTreeMap::new)
            .entry(triple.predicate_id)
            .or_insert_with(BTreeSet::new)
            .insert(triple.object_id);

        // Insert into POS index
        pos.entry(triple.predicate_id)
            .or_insert_with(BTreeMap::new)
            .entry(triple.object_id)
            .or_insert_with(BTreeSet::new)
            .insert(triple.subject_id);

        // Insert into OSP index
        osp.entry(triple.object_id)
            .or_insert_with(BTreeMap::new)
            .entry(triple.subject_id)
            .or_insert_with(BTreeSet::new)
            .insert(triple.predicate_id);

        // Update counts
        *self.triple_count.write().unwrap() += 1;
        self.stats.write().unwrap().total_insertions += 1;

        true
    }

    /// Batch insert multiple triples for better performance
    pub fn batch_insert(&self, triples: &[Triple]) -> Vec<bool> {
        // Pre-intern all terms to minimize lock contention
        let interned_triples: Vec<InternedTriple> = triples
            .iter()
            .map(|t| InternedTriple {
                subject_id: self.interner.intern_subject(t.subject()),
                predicate_id: self.interner.intern_predicate(t.predicate()),
                object_id: self.interner.intern_object(t.object()),
            })
            .collect();

        // Batch insert with single lock acquisition
        let mut spo = self.spo_index.write().unwrap();
        let mut pos = self.pos_index.write().unwrap();
        let mut osp = self.osp_index.write().unwrap();
        let mut count = self.triple_count.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let mut results = Vec::with_capacity(triples.len());
        let mut inserted_count = 0;

        for triple in interned_triples {
            // Check existence
            let exists = spo
                .get(&triple.subject_id)
                .and_then(|po| po.get(&triple.predicate_id))
                .map_or(false, |o_set| o_set.contains(&triple.object_id));

            if !exists {
                // Insert into all indexes
                spo.entry(triple.subject_id)
                    .or_insert_with(BTreeMap::new)
                    .entry(triple.predicate_id)
                    .or_insert_with(BTreeSet::new)
                    .insert(triple.object_id);

                pos.entry(triple.predicate_id)
                    .or_insert_with(BTreeMap::new)
                    .entry(triple.object_id)
                    .or_insert_with(BTreeSet::new)
                    .insert(triple.subject_id);

                osp.entry(triple.object_id)
                    .or_insert_with(BTreeMap::new)
                    .entry(triple.subject_id)
                    .or_insert_with(BTreeSet::new)
                    .insert(triple.predicate_id);

                inserted_count += 1;
                results.push(true);
            } else {
                results.push(false);
            }
        }

        *count += inserted_count;
        stats.total_insertions += inserted_count;
        stats.batch_insertions += 1;

        results
    }

    /// Remove a triple from the graph
    pub fn remove(&self, triple: &Triple) -> bool {
        let s_id = match self.interner.get_subject_id(triple.subject()) {
            Some(id) => id,
            None => return false, // Subject not in graph
        };
        let p_id = match self.interner.get_predicate_id(triple.predicate()) {
            Some(id) => id,
            None => return false, // Predicate not in graph
        };
        let o_id = match self.interner.get_object_id(triple.object()) {
            Some(id) => id,
            None => return false, // Object not in graph
        };

        self.remove_interned(s_id, p_id, o_id)
    }

    /// Remove an interned triple
    fn remove_interned(&self, s_id: u32, p_id: u32, o_id: u32) -> bool {
        let mut spo = self.spo_index.write().unwrap();
        let mut pos = self.pos_index.write().unwrap();
        let mut osp = self.osp_index.write().unwrap();

        let mut removed = false;

        // Remove from SPO index
        if let Some(po_map) = spo.get_mut(&s_id) {
            if let Some(o_set) = po_map.get_mut(&p_id) {
                removed = o_set.remove(&o_id);
                if o_set.is_empty() {
                    po_map.remove(&p_id);
                    if po_map.is_empty() {
                        spo.remove(&s_id);
                    }
                }
            }
        }

        if removed {
            // Remove from POS index
            if let Some(os_map) = pos.get_mut(&p_id) {
                if let Some(s_set) = os_map.get_mut(&o_id) {
                    s_set.remove(&s_id);
                    if s_set.is_empty() {
                        os_map.remove(&o_id);
                        if os_map.is_empty() {
                            pos.remove(&p_id);
                        }
                    }
                }
            }

            // Remove from OSP index
            if let Some(sp_map) = osp.get_mut(&o_id) {
                if let Some(p_set) = sp_map.get_mut(&s_id) {
                    p_set.remove(&p_id);
                    if p_set.is_empty() {
                        sp_map.remove(&s_id);
                        if sp_map.is_empty() {
                            osp.remove(&o_id);
                        }
                    }
                }
            }

            *self.triple_count.write().unwrap() -= 1;
            self.stats.write().unwrap().total_deletions += 1;
        }

        removed
    }

    /// Query triples matching a pattern using the most efficient index
    pub fn query(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Vec<Triple> {
        // Determine the best index to use
        let index_type = self.select_index(subject.is_some(), predicate.is_some(), object.is_some());
        
        // Update stats
        match index_type {
            IndexType::SPO => self.stats.write().unwrap().spo_lookups += 1,
            IndexType::POS => self.stats.write().unwrap().pos_lookups += 1,
            IndexType::OSP => self.stats.write().unwrap().osp_lookups += 1,
        }

        // Convert terms to IDs if provided
        let s_id = subject.and_then(|s| self.interner.get_subject_id(s));
        let p_id = predicate.and_then(|p| self.interner.get_predicate_id(p));
        let o_id = object.and_then(|o| self.interner.get_object_id(o));

        // If any required term is not found, return empty result
        if (subject.is_some() && s_id.is_none()) ||
           (predicate.is_some() && p_id.is_none()) ||
           (object.is_some() && o_id.is_none()) {
            return Vec::new();
        }

        // Query using the selected index
        let interned_results = match index_type {
            IndexType::SPO => self.query_spo(s_id, p_id, o_id),
            IndexType::POS => self.query_pos(p_id, o_id, s_id),
            IndexType::OSP => self.query_osp(o_id, s_id, p_id),
        };

        // Convert back to triples
        interned_results
            .into_iter()
            .filter_map(|it| self.interned_to_triple(it))
            .collect()
    }

    /// Select the best index based on the query pattern
    fn select_index(&self, has_s: bool, has_p: bool, has_o: bool) -> IndexType {
        match (has_s, has_p, has_o) {
            (true, true, _) => IndexType::SPO,   // S+P given, use SPO
            (true, false, true) => IndexType::OSP, // S+O given, use OSP
            (false, true, true) => IndexType::POS, // P+O given, use POS
            (true, false, false) => IndexType::SPO, // Only S given
            (false, true, false) => IndexType::POS, // Only P given
            (false, false, true) => IndexType::OSP, // Only O given
            _ => IndexType::SPO, // No constraint or all given, default to SPO
        }
    }

    /// Query using SPO index
    fn query_spo(&self, s_id: Option<u32>, p_id: Option<u32>, o_id: Option<u32>) -> Vec<InternedTriple> {
        let spo = self.spo_index.read().unwrap();
        let mut results = Vec::new();

        match (s_id, p_id, o_id) {
            (Some(s), Some(p), Some(o)) => {
                // Exact match
                if let Some(po_map) = spo.get(&s) {
                    if let Some(o_set) = po_map.get(&p) {
                        if o_set.contains(&o) {
                            results.push(InternedTriple {
                                subject_id: s,
                                predicate_id: p,
                                object_id: o,
                            });
                        }
                    }
                }
            }
            (Some(s), Some(p), None) => {
                // S+P given
                if let Some(po_map) = spo.get(&s) {
                    if let Some(o_set) = po_map.get(&p) {
                        for &o in o_set {
                            results.push(InternedTriple {
                                subject_id: s,
                                predicate_id: p,
                                object_id: o,
                            });
                        }
                    }
                }
            }
            (Some(s), None, None) => {
                // Only S given
                if let Some(po_map) = spo.get(&s) {
                    for (&p, o_set) in po_map {
                        for &o in o_set {
                            results.push(InternedTriple {
                                subject_id: s,
                                predicate_id: p,
                                object_id: o,
                            });
                        }
                    }
                }
            }
            (None, None, None) => {
                // All triples
                for (&s, po_map) in spo.iter() {
                    for (&p, o_set) in po_map {
                        for &o in o_set {
                            results.push(InternedTriple {
                                subject_id: s,
                                predicate_id: p,
                                object_id: o,
                            });
                        }
                    }
                }
            }
            _ => {
                // Other patterns better served by different indexes
                // But we can still handle them, just less efficiently
                for (&s, po_map) in spo.iter() {
                    if s_id.map_or(true, |id| id == s) {
                        for (&p, o_set) in po_map {
                            if p_id.map_or(true, |id| id == p) {
                                for &o in o_set {
                                    if o_id.map_or(true, |id| id == o) {
                                        results.push(InternedTriple {
                                            subject_id: s,
                                            predicate_id: p,
                                            object_id: o,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Query using POS index
    fn query_pos(&self, p_id: Option<u32>, o_id: Option<u32>, s_id: Option<u32>) -> Vec<InternedTriple> {
        let pos = self.pos_index.read().unwrap();
        let mut results = Vec::new();

        match (p_id, o_id) {
            (Some(p), Some(o)) => {
                // P+O given
                if let Some(os_map) = pos.get(&p) {
                    if let Some(s_set) = os_map.get(&o) {
                        for &s in s_set {
                            if s_id.map_or(true, |id| id == s) {
                                results.push(InternedTriple {
                                    subject_id: s,
                                    predicate_id: p,
                                    object_id: o,
                                });
                            }
                        }
                    }
                }
            }
            (Some(p), None) => {
                // Only P given
                if let Some(os_map) = pos.get(&p) {
                    for (&o, s_set) in os_map {
                        for &s in s_set {
                            if s_id.map_or(true, |id| id == s) {
                                results.push(InternedTriple {
                                    subject_id: s,
                                    predicate_id: p,
                                    object_id: o,
                                });
                            }
                        }
                    }
                }
            }
            _ => {
                // Fallback to SPO for other patterns
                return self.query_spo(s_id, p_id, o_id);
            }
        }

        results
    }

    /// Query using OSP index
    fn query_osp(&self, o_id: Option<u32>, s_id: Option<u32>, p_id: Option<u32>) -> Vec<InternedTriple> {
        let osp = self.osp_index.read().unwrap();
        let mut results = Vec::new();

        match (o_id, s_id) {
            (Some(o), Some(s)) => {
                // O+S given
                if let Some(sp_map) = osp.get(&o) {
                    if let Some(p_set) = sp_map.get(&s) {
                        for &p in p_set {
                            if p_id.map_or(true, |id| id == p) {
                                results.push(InternedTriple {
                                    subject_id: s,
                                    predicate_id: p,
                                    object_id: o,
                                });
                            }
                        }
                    }
                }
            }
            (Some(o), None) => {
                // Only O given
                if let Some(sp_map) = osp.get(&o) {
                    for (&s, p_set) in sp_map {
                        for &p in p_set {
                            if p_id.map_or(true, |id| id == p) {
                                results.push(InternedTriple {
                                    subject_id: s,
                                    predicate_id: p,
                                    object_id: o,
                                });
                            }
                        }
                    }
                }
            }
            _ => {
                // Fallback to SPO for other patterns
                return self.query_spo(s_id, p_id, o_id);
            }
        }

        results
    }

    /// Convert an interned triple back to a regular triple
    fn interned_to_triple(&self, interned: InternedTriple) -> Option<Triple> {
        let subject = self.interner.get_subject(interned.subject_id)?;
        let predicate = self.interner.get_predicate(interned.predicate_id)?;
        let object = self.interner.get_object(interned.object_id)?;
        Some(Triple::new(subject, predicate, object))
    }

    /// Get the number of triples in the graph
    pub fn len(&self) -> usize {
        *self.triple_count.read().unwrap()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        let spo = self.spo_index.read().unwrap();
        let pos = self.pos_index.read().unwrap();
        let osp = self.osp_index.read().unwrap();

        let spo_entries = count_index_entries(&*spo);
        let pos_entries = count_index_entries(&*pos);
        let osp_entries = count_index_entries(&*osp);

        MemoryUsage {
            term_interner_bytes: self.interner.memory_usage(),
            spo_index_bytes: estimate_index_memory(&*spo),
            pos_index_bytes: estimate_index_memory(&*pos),
            osp_index_bytes: estimate_index_memory(&*osp),
            total_triple_count: self.len(),
            index_entry_count: spo_entries + pos_entries + osp_entries,
        }
    }

    /// Get index statistics
    pub fn index_stats(&self) -> IndexStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear all data from the graph
    pub fn clear(&self) {
        self.spo_index.write().unwrap().clear();
        self.pos_index.write().unwrap().clear();
        self.osp_index.write().unwrap().clear();
        *self.triple_count.write().unwrap() = 0;
        self.interner.clear();
    }

    /// Get a reference to the term interner
    pub fn interner(&self) -> &Arc<TermInterner> {
        &self.interner
    }

    /// Parallel insert of multiple triples using rayon
    #[cfg(feature = "parallel")]
    pub fn par_insert_batch(&self, triples: Vec<Triple>) -> Vec<bool> {
        use rayon::prelude::*;
        
        // First pass: intern all terms in parallel
        let interned_triples: Vec<_> = triples
            .par_iter()
            .map(|triple| InternedTriple {
                subject_id: self.interner.intern_subject(triple.subject()),
                predicate_id: self.interner.intern_predicate(triple.predicate()),
                object_id: self.interner.intern_object(triple.object()),
            })
            .collect();
        
        // Group by subject for better lock granularity
        let mut grouped: HashMap<u32, Vec<InternedTriple>> = HashMap::new();
        for interned in interned_triples {
            grouped.entry(interned.subject_id).or_default().push(interned);
        }
        
        // Process groups in parallel
        let results: Vec<_> = grouped
            .into_par_iter()
            .flat_map(|(_, group)| {
                let mut group_results = Vec::new();
                for interned in group {
                    let inserted = self.insert_interned(interned);
                    group_results.push(inserted);
                }
                group_results
            })
            .collect();
        
        results
    }

    /// Parallel remove of multiple triples
    #[cfg(feature = "parallel")]
    pub fn par_remove_batch(&self, triples: &[Triple]) -> Vec<bool> {
        use rayon::prelude::*;
        
        triples
            .par_iter()
            .map(|triple| self.remove(triple))
            .collect()
    }

    /// Parallel query with multiple patterns
    #[cfg(feature = "parallel")]
    pub fn par_query_batch(
        &self,
        patterns: Vec<(Option<Subject>, Option<Predicate>, Option<Object>)>,
    ) -> Vec<Vec<Triple>> {
        use rayon::prelude::*;
        
        patterns
            .into_par_iter()
            .map(|(s, p, o)| {
                self.query(s.as_ref(), p.as_ref(), o.as_ref())
            })
            .collect()
    }

    /// Apply a transformation to all triples in parallel
    #[cfg(feature = "parallel")]
    pub fn par_transform<F>(&self, transform: F) -> Vec<Triple>
    where
        F: Fn(&Triple) -> Option<Triple> + Send + Sync,
    {
        use rayon::prelude::*;
        
        // Get all triples
        let all_triples = self.query(None, None, None);
        
        // Transform in parallel
        all_triples
            .into_par_iter()
            .filter_map(|triple| transform(&triple))
            .collect()
    }

    /// Parallel map over all triples
    #[cfg(feature = "parallel")]
    pub fn par_map<F, R>(&self, mapper: F) -> Vec<R>
    where
        F: Fn(&Triple) -> R + Send + Sync,
        R: Send,
    {
        use rayon::prelude::*;
        
        let all_triples = self.query(None, None, None);
        all_triples
            .into_par_iter()
            .map(|triple| mapper(&triple))
            .collect()
    }

    /// Parallel filter triples
    #[cfg(feature = "parallel")]
    pub fn par_filter<F>(&self, predicate: F) -> Vec<Triple>
    where
        F: Fn(&Triple) -> bool + Send + Sync,
    {
        use rayon::prelude::*;
        
        let all_triples = self.query(None, None, None);
        all_triples
            .into_par_iter()
            .filter(|triple| predicate(triple))
            .collect()
    }

    /// Parallel fold operation
    #[cfg(feature = "parallel")]
    pub fn par_fold<F, R>(&self, init: R, fold_fn: F) -> R
    where
        F: Fn(R, &Triple) -> R + Send + Sync,
        R: Send + Sync + Clone + 'static,
    {
        use rayon::prelude::*;
        
        let all_triples = self.query(None, None, None);
        all_triples
            .into_par_iter()
            .fold(|| init.clone(), |acc, triple| fold_fn(acc, &triple))
            .reduce(|| init.clone(), |acc1, acc2| {
                // For a proper fold, we need a reduce function that makes sense
                // This is a simplified version - in practice, you'd want a proper combine function
                acc1
            })
    }
}

impl Default for IndexedGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub term_interner_bytes: usize,
    pub spo_index_bytes: usize,
    pub pos_index_bytes: usize,
    pub osp_index_bytes: usize,
    pub total_triple_count: usize,
    pub index_entry_count: usize,
}

impl MemoryUsage {
    /// Get total memory usage in bytes
    pub fn total_bytes(&self) -> usize {
        self.term_interner_bytes + self.spo_index_bytes + self.pos_index_bytes + self.osp_index_bytes
    }

    /// Get average bytes per triple
    pub fn bytes_per_triple(&self) -> f64 {
        if self.total_triple_count == 0 {
            0.0
        } else {
            self.total_bytes() as f64 / self.total_triple_count as f64
        }
    }
}

/// Count entries in an index
fn count_index_entries(index: &BTreeMap<u32, BTreeMap<u32, BTreeSet<u32>>>) -> usize {
    index.values().map(|inner| inner.len()).sum()
}

/// Estimate memory usage of an index
fn estimate_index_memory(index: &BTreeMap<u32, BTreeMap<u32, BTreeSet<u32>>>) -> usize {
    let mut total = 0;
    for (_, inner) in index {
        total += 4; // Key size
        total += 24; // BTreeMap overhead
        for (_, set) in inner {
            total += 4; // Key size
            total += 24; // BTreeSet overhead
            total += set.len() * 4; // Values
        }
    }
    total
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Literal};

    fn create_test_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(
            NamedNode::new(s).unwrap(),
            NamedNode::new(p).unwrap(),
            Literal::new(o),
        )
    }

    #[test]
    fn test_basic_operations() {
        let graph = IndexedGraph::new();
        
        let triple = create_test_triple("http://example.org/s1", "http://example.org/p1", "object1");
        
        // Test insertion
        assert!(graph.insert(&triple));
        assert!(!graph.insert(&triple)); // Duplicate
        assert_eq!(graph.len(), 1);
        
        // Test query
        let results = graph.query(Some(triple.subject()), None, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], triple);
        
        // Test removal
        assert!(graph.remove(&triple));
        assert!(!graph.remove(&triple)); // Already removed
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_batch_insert() {
        let graph = IndexedGraph::new();
        
        let triples = vec![
            create_test_triple("http://example.org/s1", "http://example.org/p1", "o1"),
            create_test_triple("http://example.org/s1", "http://example.org/p2", "o2"),
            create_test_triple("http://example.org/s2", "http://example.org/p1", "o3"),
            create_test_triple("http://example.org/s1", "http://example.org/p1", "o1"), // Duplicate
        ];
        
        let results = graph.batch_insert(&triples);
        assert_eq!(results, vec![true, true, true, false]);
        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_query_patterns() {
        let graph = IndexedGraph::new();
        
        // Insert test data
        let triples = vec![
            create_test_triple("http://example.org/s1", "http://example.org/p1", "o1"),
            create_test_triple("http://example.org/s1", "http://example.org/p2", "o2"),
            create_test_triple("http://example.org/s2", "http://example.org/p1", "o3"),
            create_test_triple("http://example.org/s2", "http://example.org/p2", "o4"),
        ];
        
        for triple in &triples {
            graph.insert(triple);
        }
        
        // Test S pattern
        let s1 = Subject::NamedNode(NamedNode::new("http://example.org/s1").unwrap());
        let results = graph.query(Some(&s1), None, None);
        assert_eq!(results.len(), 2);
        
        // Test P pattern
        let p1 = Predicate::NamedNode(NamedNode::new("http://example.org/p1").unwrap());
        let results = graph.query(None, Some(&p1), None);
        assert_eq!(results.len(), 2);
        
        // Test O pattern
        let o1 = Object::Literal(Literal::new("o1"));
        let results = graph.query(None, None, Some(&o1));
        assert_eq!(results.len(), 1);
        
        // Test SP pattern
        let results = graph.query(Some(&s1), Some(&p1), None);
        assert_eq!(results.len(), 1);
        
        // Test PO pattern
        let results = graph.query(None, Some(&p1), Some(&o1));
        assert_eq!(results.len(), 1);
        
        // Test SO pattern
        let o2 = Object::Literal(Literal::new("o2"));
        let results = graph.query(Some(&s1), None, Some(&o2));
        assert_eq!(results.len(), 1);
        
        // Test SPO pattern (exact match)
        let results = graph.query(Some(&s1), Some(&p1), Some(&o1));
        assert_eq!(results.len(), 1);
        
        // Test all triples
        let results = graph.query(None, None, None);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_index_selection() {
        let graph = IndexedGraph::new();
        
        // SPO patterns
        assert_eq!(graph.select_index(true, true, true), IndexType::SPO);
        assert_eq!(graph.select_index(true, true, false), IndexType::SPO);
        assert_eq!(graph.select_index(true, false, false), IndexType::SPO);
        
        // POS patterns
        assert_eq!(graph.select_index(false, true, true), IndexType::POS);
        assert_eq!(graph.select_index(false, true, false), IndexType::POS);
        
        // OSP patterns
        assert_eq!(graph.select_index(true, false, true), IndexType::OSP);
        assert_eq!(graph.select_index(false, false, true), IndexType::OSP);
    }

    #[test]
    fn test_memory_usage() {
        let graph = IndexedGraph::new();
        
        // Insert some triples
        for i in 0..10 {
            let triple = create_test_triple(
                &format!("http://example.org/s{}", i),
                "http://example.org/p1",
                &format!("object{}", i),
            );
            graph.insert(&triple);
        }
        
        let usage = graph.memory_usage();
        assert_eq!(usage.total_triple_count, 10);
        assert!(usage.total_bytes() > 0);
        assert!(usage.bytes_per_triple() > 0.0);
    }

    #[test]
    fn test_clear() {
        let graph = IndexedGraph::new();
        
        // Insert and then clear
        let triple = create_test_triple("http://example.org/s1", "http://example.org/p1", "o1");
        graph.insert(&triple);
        assert_eq!(graph.len(), 1);
        
        graph.clear();
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
        
        // Should be able to insert again after clear
        assert!(graph.insert(&triple));
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let graph = Arc::new(IndexedGraph::new());
        let mut handles = vec![];

        // Concurrent insertions
        for i in 0..10 {
            let graph_clone = Arc::clone(&graph);
            let handle = thread::spawn(move || {
                let triple = create_test_triple(
                    &format!("http://example.org/s{}", i),
                    "http://example.org/p1",
                    &format!("o{}", i),
                );
                graph_clone.insert(&triple)
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(graph.len(), 10);

        // Concurrent queries
        let mut handles = vec![];
        for _ in 0..10 {
            let graph_clone = Arc::clone(&graph);
            let handle = thread::spawn(move || {
                graph_clone.query(None, None, None).len()
            });
            handles.push(handle);
        }

        for handle in handles {
            let count = handle.join().unwrap();
            assert_eq!(count, 10);
        }
    }
}