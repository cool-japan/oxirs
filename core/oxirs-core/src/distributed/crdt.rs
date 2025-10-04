//! CRDTs for conflict-free replicated RDF
//!
//! This module implements Conflict-free Replicated Data Types (CRDTs) optimized
//! for RDF data, enabling eventual consistency without coordination.

#![allow(dead_code)]

use crate::model::{Triple, TriplePattern};
use crate::OxirsError;
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;

/// CRDT configuration
#[derive(Debug, Clone)]
pub struct CrdtConfig {
    /// Node ID for this replica
    pub node_id: String,
    /// CRDT type to use
    pub crdt_type: CrdtType,
    /// Garbage collection configuration
    pub gc_config: GcConfig,
    /// Delta sync configuration
    pub delta_config: DeltaConfig,
}

/// CRDT types available
#[derive(Debug, Clone)]
pub enum CrdtType {
    /// Grow-only set (2P-Set without removals)
    GSet,
    /// Two-phase set (add and remove)
    TwoPhaseSet,
    /// Add-remove partial order
    AddRemovePartialOrder,
    /// Observed-remove set
    OrSet,
    /// Last-write-wins element set
    LwwSet,
    /// Multi-value register for conflicts
    MvRegister,
    /// RDF-specific CRDT
    RdfCrdt,
}

/// Garbage collection configuration
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Enable automatic GC
    pub auto_gc: bool,
    /// GC interval in seconds
    pub interval_secs: u64,
    /// Maximum tombstone age in seconds
    pub tombstone_ttl_secs: u64,
    /// Batch size for GC
    pub batch_size: usize,
}

impl Default for GcConfig {
    fn default() -> Self {
        GcConfig {
            auto_gc: true,
            interval_secs: 3600,           // 1 hour
            tombstone_ttl_secs: 86400 * 7, // 1 week
            batch_size: 1000,
        }
    }
}

/// Delta sync configuration
#[derive(Debug, Clone)]
pub struct DeltaConfig {
    /// Enable delta synchronization
    pub enabled: bool,
    /// Maximum delta size before full sync
    pub max_delta_size: usize,
    /// Delta buffer size
    pub buffer_size: usize,
    /// Compression for deltas
    pub compression: bool,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        DeltaConfig {
            enabled: true,
            max_delta_size: 10000,
            buffer_size: 100000,
            compression: true,
        }
    }
}

/// Base trait for CRDTs
pub trait Crdt: Send + Sync {
    /// Type of delta for this CRDT
    type Delta: Send + Sync + Clone + Serialize + for<'de> Deserialize<'de>;

    /// Merge with another CRDT state
    fn merge(&mut self, other: &Self);

    /// Get delta since last checkpoint
    fn delta(&self) -> Option<Self::Delta>;

    /// Apply delta
    fn apply_delta(&mut self, delta: Self::Delta);

    /// Reset delta tracking
    fn reset_delta(&mut self);
}

/// Unique ID for CRDT elements
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ElementId {
    /// Timestamp (Lamport clock)
    pub timestamp: u64,
    /// Node ID
    pub node_id: String,
    /// Random component for uniqueness
    pub random: u64,
}

impl ElementId {
    /// Create new element ID
    pub fn new(timestamp: u64, node_id: String) -> Self {
        ElementId {
            timestamp,
            node_id,
            random: {
                let mut rng = Random::default();
                rng.random::<u64>()
            },
        }
    }
}

/// Grow-only set CRDT
#[derive(Debug, Clone)]
pub struct GrowSet<T: Clone + Ord + Send + Sync> {
    /// Elements in the set
    elements: BTreeSet<T>,
    /// Delta tracking
    delta_elements: Option<BTreeSet<T>>,
}

impl<T: Clone + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>> Default for GrowSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>> GrowSet<T> {
    /// Create new grow-only set
    pub fn new() -> Self {
        GrowSet {
            elements: BTreeSet::new(),
            delta_elements: Some(BTreeSet::new()),
        }
    }

    /// Add element
    pub fn add(&mut self, element: T) {
        if self.elements.insert(element.clone()) {
            if let Some(ref mut delta) = self.delta_elements {
                delta.insert(element);
            }
        }
    }

    /// Check if contains element
    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }

    /// Get all elements
    pub fn elements(&self) -> &BTreeSet<T> {
        &self.elements
    }
}

impl<T: Clone + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>> Crdt for GrowSet<T> {
    type Delta = BTreeSet<T>;

    fn merge(&mut self, other: &Self) {
        for element in &other.elements {
            self.add(element.clone());
        }
    }

    fn delta(&self) -> Option<Self::Delta> {
        self.delta_elements.clone()
    }

    fn apply_delta(&mut self, delta: Self::Delta) {
        for element in delta {
            self.elements.insert(element);
        }
    }

    fn reset_delta(&mut self) {
        self.delta_elements = Some(BTreeSet::new());
    }
}

/// Two-phase set CRDT
#[derive(Debug, Clone)]
pub struct TwoPhaseSet<T: Clone + Ord + Send + Sync> {
    /// Added elements
    added: BTreeSet<T>,
    /// Removed elements (tombstones)
    removed: BTreeSet<T>,
    /// Delta tracking
    delta_added: Option<BTreeSet<T>>,
    delta_removed: Option<BTreeSet<T>>,
}

impl<T: Clone + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>> Default
    for TwoPhaseSet<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>> TwoPhaseSet<T> {
    /// Create new two-phase set
    pub fn new() -> Self {
        TwoPhaseSet {
            added: BTreeSet::new(),
            removed: BTreeSet::new(),
            delta_added: Some(BTreeSet::new()),
            delta_removed: Some(BTreeSet::new()),
        }
    }

    /// Add element
    pub fn add(&mut self, element: T) {
        if !self.removed.contains(&element) && self.added.insert(element.clone()) {
            if let Some(ref mut delta) = self.delta_added {
                delta.insert(element);
            }
        }
    }

    /// Remove element
    pub fn remove(&mut self, element: T) {
        if self.added.contains(&element) && self.removed.insert(element.clone()) {
            if let Some(ref mut delta) = self.delta_removed {
                delta.insert(element);
            }
        }
    }

    /// Check if contains element
    pub fn contains(&self, element: &T) -> bool {
        self.added.contains(element) && !self.removed.contains(element)
    }

    /// Get current elements
    pub fn elements(&self) -> BTreeSet<T> {
        self.added.difference(&self.removed).cloned().collect()
    }
}

/// Observed-Remove Set (OR-Set) CRDT
#[derive(Debug, Clone)]
pub struct OrSet<T: Clone + Ord + Send + Sync> {
    /// Elements with their unique tags
    elements: BTreeMap<T, BTreeSet<ElementId>>,
    /// Tombstones for removed elements
    tombstones: BTreeMap<T, BTreeSet<ElementId>>,
    /// Node ID for generating tags
    node_id: String,
    /// Lamport clock
    clock: u64,
    /// Delta tracking
    delta: Option<OrSetDelta<T>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrSetDelta<T: Clone + Ord> {
    /// Added elements with tags
    added: BTreeMap<T, BTreeSet<ElementId>>,
    /// Removed tombstones
    removed: BTreeMap<T, BTreeSet<ElementId>>,
}

impl<T: Clone + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>> OrSet<T> {
    /// Create new OR-Set
    pub fn new(node_id: String) -> Self {
        OrSet {
            elements: BTreeMap::new(),
            tombstones: BTreeMap::new(),
            node_id,
            clock: 0,
            delta: Some(OrSetDelta {
                added: BTreeMap::new(),
                removed: BTreeMap::new(),
            }),
        }
    }

    /// Add element
    pub fn add(&mut self, element: T) {
        self.clock += 1;
        let tag = ElementId::new(self.clock, self.node_id.clone());

        self.elements
            .entry(element.clone())
            .or_default()
            .insert(tag.clone());

        if let Some(ref mut delta) = self.delta {
            delta
                .added
                .entry(element)
                .or_insert_with(BTreeSet::new)
                .insert(tag);
        }
    }

    /// Remove element
    pub fn remove(&mut self, element: &T) {
        if let Some(tags) = self.elements.get(element).cloned() {
            self.tombstones.insert(element.clone(), tags.clone());
            self.elements.remove(element);

            if let Some(ref mut delta) = self.delta {
                delta.removed.insert(element.clone(), tags);
            }
        }
    }

    /// Check if contains element
    pub fn contains(&self, element: &T) -> bool {
        if let Some(tags) = self.elements.get(element) {
            if let Some(tombstone_tags) = self.tombstones.get(element) {
                // Element exists if it has tags not in tombstones
                !tags.is_subset(tombstone_tags)
            } else {
                true
            }
        } else {
            false
        }
    }

    /// Get current elements
    pub fn elements(&self) -> BTreeSet<T> {
        self.elements
            .keys()
            .filter(|e| self.contains(e))
            .cloned()
            .collect()
    }
}

impl<T: Clone + Ord + Send + Sync + Serialize + for<'de> Deserialize<'de>> Crdt for OrSet<T> {
    type Delta = OrSetDelta<T>;

    fn merge(&mut self, other: &Self) {
        // Merge elements
        for (element, tags) in &other.elements {
            self.elements
                .entry(element.clone())
                .or_default()
                .extend(tags.iter().cloned());
        }

        // Merge tombstones
        for (element, tags) in &other.tombstones {
            self.tombstones
                .entry(element.clone())
                .or_default()
                .extend(tags.iter().cloned());
        }

        // Remove elements that are fully tombstoned
        let to_remove: Vec<_> = self
            .elements
            .iter()
            .filter(|(e, tags)| {
                if let Some(tombstone_tags) = self.tombstones.get(e) {
                    tags.is_subset(tombstone_tags)
                } else {
                    false
                }
            })
            .map(|(e, _)| e.clone())
            .collect();

        for element in to_remove {
            self.elements.remove(&element);
        }

        // Update clock
        self.clock = self.clock.max(other.clock);
    }

    fn delta(&self) -> Option<Self::Delta> {
        self.delta.clone()
    }

    fn apply_delta(&mut self, delta: Self::Delta) {
        // Apply added elements
        for (element, tags) in delta.added {
            self.elements.entry(element).or_default().extend(tags);
        }

        // Apply removed tombstones
        for (element, tags) in delta.removed {
            self.tombstones
                .entry(element.clone())
                .or_default()
                .extend(tags);

            // Remove element if fully tombstoned
            if let Some(elem_tags) = self.elements.get(&element) {
                if let Some(tombstone_tags) = self.tombstones.get(&element) {
                    if elem_tags.is_subset(tombstone_tags) {
                        self.elements.remove(&element);
                    }
                }
            }
        }
    }

    fn reset_delta(&mut self) {
        self.delta = Some(OrSetDelta {
            added: BTreeMap::new(),
            removed: BTreeMap::new(),
        });
    }
}

/// RDF-specific CRDT optimized for triple stores
pub struct RdfCrdt {
    /// Configuration
    config: CrdtConfig,
    /// Triple OR-Set for conflict-free triple management
    triples: OrSet<Triple>,
    /// Predicate-indexed sets for efficient queries
    predicate_index: HashMap<String, OrSet<Triple>>,
    /// Subject-indexed sets
    subject_index: HashMap<String, OrSet<Triple>>,
    /// Statistics
    stats: Arc<RwLock<CrdtStats>>,
}

/// CRDT statistics
#[derive(Debug, Default)]
struct CrdtStats {
    /// Total operations
    total_ops: u64,
    /// Add operations
    add_ops: u64,
    /// Remove operations
    remove_ops: u64,
    /// Merge operations
    merge_ops: u64,
    /// Current triple count
    triple_count: usize,
    /// Tombstone count
    #[allow(dead_code)]
    tombstone_count: usize,
}

impl RdfCrdt {
    /// Create new RDF CRDT
    pub async fn new(config: CrdtConfig) -> Result<Self, OxirsError> {
        let node_id = config.node_id.clone();

        Ok(RdfCrdt {
            config,
            triples: OrSet::new(node_id),
            predicate_index: HashMap::new(),
            subject_index: HashMap::new(),
            stats: Arc::new(RwLock::new(CrdtStats::default())),
        })
    }

    /// Add triple
    pub async fn add_triple(&mut self, triple: Triple) -> Result<(), OxirsError> {
        // Add to main set
        self.triples.add(triple.clone());

        // Update predicate index
        let predicate_str = match triple.predicate() {
            crate::model::Predicate::NamedNode(nn) => nn.as_str(),
            crate::model::Predicate::Variable(v) => v.as_str(),
        };
        self.predicate_index
            .entry(predicate_str.to_string())
            .or_insert_with(|| OrSet::new(self.config.node_id.clone()))
            .add(triple.clone());

        // Update subject index
        let subject_str = match triple.subject() {
            crate::model::Subject::NamedNode(nn) => nn.as_str(),
            crate::model::Subject::BlankNode(bn) => bn.as_str(),
            crate::model::Subject::Variable(v) => v.as_str(),
            crate::model::Subject::QuotedTriple(_qt) => "<<quoted-triple>>",
        };
        self.subject_index
            .entry(subject_str.to_string())
            .or_insert_with(|| OrSet::new(self.config.node_id.clone()))
            .add(triple);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_ops += 1;
        stats.add_ops += 1;
        stats.triple_count = self.triples.elements().len();

        Ok(())
    }

    /// Remove triple
    pub async fn remove_triple(&mut self, triple: &Triple) -> Result<(), OxirsError> {
        // Remove from main set
        self.triples.remove(triple);

        // Update predicate index
        let predicate_str = match triple.predicate() {
            crate::model::Predicate::NamedNode(nn) => nn.as_str(),
            crate::model::Predicate::Variable(v) => v.as_str(),
        };
        if let Some(predicate_set) = self.predicate_index.get_mut(predicate_str) {
            predicate_set.remove(triple);
        }

        // Update subject index
        let subject_str = match triple.subject() {
            crate::model::Subject::NamedNode(nn) => nn.as_str(),
            crate::model::Subject::BlankNode(bn) => bn.as_str(),
            crate::model::Subject::Variable(v) => v.as_str(),
            crate::model::Subject::QuotedTriple(_qt) => "<<quoted-triple>>",
        };
        if let Some(subject_set) = self.subject_index.get_mut(subject_str) {
            subject_set.remove(triple);
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_ops += 1;
        stats.remove_ops += 1;
        stats.triple_count = self.triples.elements().len();

        Ok(())
    }

    /// Query triples by pattern
    pub async fn query(&self, pattern: &TriplePattern) -> Result<Vec<Triple>, OxirsError> {
        let results = match (pattern.subject(), pattern.predicate(), pattern.object()) {
            (Some(subject), Some(_predicate), _) => {
                // Use both subject and predicate index
                if let Some(subject_set) = self.subject_index.get(subject.as_str()) {
                    subject_set
                        .elements()
                        .into_iter()
                        .filter(|t| pattern.matches(t))
                        .collect()
                } else {
                    Vec::new()
                }
            }
            (Some(subject), None, _) => {
                // Use subject index
                if let Some(subject_set) = self.subject_index.get(subject.as_str()) {
                    subject_set
                        .elements()
                        .into_iter()
                        .filter(|t| pattern.matches(t))
                        .collect()
                } else {
                    Vec::new()
                }
            }
            (None, Some(predicate), _) => {
                // Use predicate index
                if let Some(predicate_set) = self.predicate_index.get(predicate.as_str()) {
                    predicate_set
                        .elements()
                        .into_iter()
                        .filter(|t| pattern.matches(t))
                        .collect()
                } else {
                    Vec::new()
                }
            }
            _ => {
                // Full scan
                self.triples
                    .elements()
                    .into_iter()
                    .filter(|t| pattern.matches(t))
                    .collect()
            }
        };

        Ok(results)
    }

    /// Merge with another RDF CRDT
    pub async fn merge(&mut self, other: &RdfCrdt) -> Result<(), OxirsError> {
        // Merge main triple set
        self.triples.merge(&other.triples);

        // Merge predicate indexes
        for (predicate, other_set) in &other.predicate_index {
            self.predicate_index
                .entry(predicate.clone())
                .or_insert_with(|| OrSet::new(self.config.node_id.clone()))
                .merge(other_set);
        }

        // Merge subject indexes
        for (subject, other_set) in &other.subject_index {
            self.subject_index
                .entry(subject.clone())
                .or_insert_with(|| OrSet::new(self.config.node_id.clone()))
                .merge(other_set);
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.merge_ops += 1;
        stats.triple_count = self.triples.elements().len();

        Ok(())
    }

    /// Get delta for synchronization
    pub fn get_delta(&self) -> RdfCrdtDelta {
        RdfCrdtDelta {
            triples_delta: self.triples.delta(),
            predicate_deltas: self
                .predicate_index
                .iter()
                .filter_map(|(p, set)| set.delta().map(|d| (p.clone(), d)))
                .collect(),
            subject_deltas: self
                .subject_index
                .iter()
                .filter_map(|(s, set)| set.delta().map(|d| (s.clone(), d)))
                .collect(),
        }
    }

    /// Apply delta from another replica
    pub async fn apply_delta(&mut self, delta: RdfCrdtDelta) -> Result<(), OxirsError> {
        // Apply main triple delta
        if let Some(triples_delta) = delta.triples_delta {
            self.triples.apply_delta(triples_delta);
        }

        // Apply predicate deltas
        for (predicate, pred_delta) in delta.predicate_deltas {
            self.predicate_index
                .entry(predicate)
                .or_insert_with(|| OrSet::new(self.config.node_id.clone()))
                .apply_delta(pred_delta);
        }

        // Apply subject deltas
        for (subject, subj_delta) in delta.subject_deltas {
            self.subject_index
                .entry(subject)
                .or_insert_with(|| OrSet::new(self.config.node_id.clone()))
                .apply_delta(subj_delta);
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.triple_count = self.triples.elements().len();

        Ok(())
    }

    /// Reset delta tracking
    pub fn reset_delta(&mut self) {
        self.triples.reset_delta();
        for set in self.predicate_index.values_mut() {
            set.reset_delta();
        }
        for set in self.subject_index.values_mut() {
            set.reset_delta();
        }
    }

    /// Garbage collect tombstones
    pub async fn garbage_collect(&mut self) -> Result<GcReport, OxirsError> {
        let start_tombstones = self.triples.tombstones.len();

        // Remove old tombstones based on age
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - self.config.gc_config.tombstone_ttl_secs;

        // Filter tombstones by age
        self.triples
            .tombstones
            .retain(|_, tags| tags.iter().any(|tag| tag.timestamp > cutoff));

        // Same for indexes
        for set in self.predicate_index.values_mut() {
            set.tombstones
                .retain(|_, tags| tags.iter().any(|tag| tag.timestamp > cutoff));
        }

        for set in self.subject_index.values_mut() {
            set.tombstones
                .retain(|_, tags| tags.iter().any(|tag| tag.timestamp > cutoff));
        }

        let removed = start_tombstones - self.triples.tombstones.len();

        Ok(GcReport {
            tombstones_removed: removed,
            space_reclaimed: removed * std::mem::size_of::<(Triple, BTreeSet<ElementId>)>(),
        })
    }

    /// Get statistics
    pub async fn stats(&self) -> CrdtStatsReport {
        let stats = self.stats.read().await;
        CrdtStatsReport {
            total_ops: stats.total_ops,
            add_ops: stats.add_ops,
            remove_ops: stats.remove_ops,
            merge_ops: stats.merge_ops,
            triple_count: stats.triple_count,
            tombstone_count: self.triples.tombstones.len(),
        }
    }
}

/// RDF CRDT delta for efficient synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfCrdtDelta {
    /// Main triple set delta
    pub triples_delta: Option<OrSetDelta<Triple>>,
    /// Predicate index deltas
    pub predicate_deltas: HashMap<String, OrSetDelta<Triple>>,
    /// Subject index deltas
    pub subject_deltas: HashMap<String, OrSetDelta<Triple>>,
}

/// Garbage collection report
#[derive(Debug)]
pub struct GcReport {
    pub tombstones_removed: usize,
    pub space_reclaimed: usize,
}

/// CRDT statistics report
#[derive(Debug)]
pub struct CrdtStatsReport {
    pub total_ops: u64,
    pub add_ops: u64,
    pub remove_ops: u64,
    pub merge_ops: u64,
    pub triple_count: usize,
    pub tombstone_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode, Object};

    #[tokio::test]
    async fn test_grow_set() {
        let mut set1 = GrowSet::new();
        let mut set2 = GrowSet::new();

        set1.add(1);
        set1.add(2);
        set2.add(2);
        set2.add(3);

        set1.merge(&set2);

        assert!(set1.contains(&1));
        assert!(set1.contains(&2));
        assert!(set1.contains(&3));
        assert_eq!(set1.elements().len(), 3);
    }

    #[tokio::test]
    async fn test_or_set() {
        let mut set1 = OrSet::new("node1".to_string());
        let mut set2 = OrSet::new("node2".to_string());

        set1.add(1);
        set1.add(2);
        set2.add(2);
        set2.add(3);

        // Remove from set1
        set1.remove(&2);

        // Merge
        set1.merge(&set2);

        assert!(set1.contains(&1));
        assert!(set1.contains(&2)); // Still exists due to set2
        assert!(set1.contains(&3));
    }

    #[tokio::test]
    async fn test_rdf_crdt() {
        let config = CrdtConfig {
            node_id: "node1".to_string(),
            crdt_type: CrdtType::RdfCrdt,
            gc_config: GcConfig::default(),
            delta_config: DeltaConfig::default(),
        };

        let mut crdt = RdfCrdt::new(config).await.unwrap();

        // Add triples
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/s1").unwrap(),
            NamedNode::new("http://example.org/p1").unwrap(),
            Object::Literal(Literal::new("value1")),
        );

        let triple2 = Triple::new(
            NamedNode::new("http://example.org/s1").unwrap(),
            NamedNode::new("http://example.org/p2").unwrap(),
            Object::Literal(Literal::new("value2")),
        );

        crdt.add_triple(triple1.clone()).await.unwrap();
        crdt.add_triple(triple2.clone()).await.unwrap();

        // Query by subject
        let pattern = TriplePattern::new(
            Some(crate::model::SubjectPattern::NamedNode(
                NamedNode::new("http://example.org/s1").unwrap(),
            )),
            None,
            None,
        );

        let results = crdt.query(&pattern).await.unwrap();
        assert_eq!(results.len(), 2);

        // Remove triple
        crdt.remove_triple(&triple1).await.unwrap();

        let results = crdt.query(&pattern).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], triple2);
    }

    #[tokio::test]
    async fn test_rdf_crdt_merge() {
        let config1 = CrdtConfig {
            node_id: "node1".to_string(),
            crdt_type: CrdtType::RdfCrdt,
            gc_config: GcConfig::default(),
            delta_config: DeltaConfig::default(),
        };

        let config2 = CrdtConfig {
            node_id: "node2".to_string(),
            crdt_type: CrdtType::RdfCrdt,
            gc_config: GcConfig::default(),
            delta_config: DeltaConfig::default(),
        };

        let mut crdt1 = RdfCrdt::new(config1).await.unwrap();
        let mut crdt2 = RdfCrdt::new(config2).await.unwrap();

        // Add different triples to each
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/s1").unwrap(),
            NamedNode::new("http://example.org/p1").unwrap(),
            Object::Literal(Literal::new("value1")),
        );

        let triple2 = Triple::new(
            NamedNode::new("http://example.org/s2").unwrap(),
            NamedNode::new("http://example.org/p2").unwrap(),
            Object::Literal(Literal::new("value2")),
        );

        crdt1.add_triple(triple1.clone()).await.unwrap();
        crdt2.add_triple(triple2.clone()).await.unwrap();

        // Merge
        crdt1.merge(&crdt2).await.unwrap();

        // Both triples should be in crdt1
        let pattern = TriplePattern::new(None, None, None);
        let results = crdt1.query(&pattern).await.unwrap();
        assert_eq!(results.len(), 2);
    }
}
