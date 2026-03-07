//! RDFS inference for OxiRS WASM
//!
//! Implements the core RDFS entailment rules:
//! - `rdfs:subClassOf` transitivity
//! - `rdfs:subPropertyOf` transitivity
//! - `rdf:type` propagation through `rdfs:subClassOf`
//! - `rdfs:domain` inference: if `?p rdfs:domain ?C` and `?s ?p ?o`, then `?s rdf:type ?C`
//! - `rdfs:range` inference: if `?p rdfs:range ?C` and `?s ?p ?o`, then `?o rdf:type ?C`

use crate::store::OxiRSStore;
use std::collections::HashSet;

// -----------------------------------------------------------------------
// RDF/RDFS vocabulary constants
// -----------------------------------------------------------------------

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDFS_SUB_CLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_SUB_PROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";

// -----------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------

/// Configuration for the RDFS reasoner
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum depth for transitivity closure computations.
    /// A value of 0 means unlimited depth.
    pub max_depth: usize,
    /// Whether to use forward chaining only (the only strategy currently implemented)
    pub forward_chaining_only: bool,
    /// Whether to materialise ALL inferred triples into the store immediately,
    /// or only return a count of new triples without inserting them.
    pub materialize_all: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_depth: 0,
            forward_chaining_only: true,
            materialize_all: true,
        }
    }
}

impl InferenceConfig {
    /// Create a configuration with unlimited depth and full materialisation
    pub fn full() -> Self {
        Self::default()
    }

    /// Create a configuration limited to `depth` reasoning steps
    pub fn with_max_depth(depth: usize) -> Self {
        Self {
            max_depth: depth,
            ..Self::default()
        }
    }
}

// -----------------------------------------------------------------------
// RdfsReasoner
// -----------------------------------------------------------------------

/// Applies RDFS entailment rules to an [`OxiRSStore`], materialising inferred triples.
pub struct RdfsReasoner {
    config: InferenceConfig,
}

impl RdfsReasoner {
    /// Create a new reasoner with the given configuration
    pub fn new(config: InferenceConfig) -> Self {
        Self { config }
    }

    /// Create a reasoner with default (full) configuration
    pub fn default_config() -> Self {
        Self::new(InferenceConfig::default())
    }

    /// Apply all RDFS rules until a fixed point is reached.
    ///
    /// Returns the total number of new triples inserted.
    pub fn apply(&self, store: &mut OxiRSStore) -> u32 {
        let mut total_new = 0u32;
        let max_iterations = if self.config.max_depth == 0 {
            usize::MAX
        } else {
            self.config.max_depth
        };

        for _ in 0..max_iterations {
            let new = self.apply_one_round(store);
            if new == 0 {
                break;
            }
            total_new += new;
        }

        total_new
    }

    /// Run a single round of RDFS rule applications.
    /// Returns the number of new triples added in this round.
    fn apply_one_round(&self, store: &mut OxiRSStore) -> u32 {
        let mut inferred: Vec<(String, String, String)> = Vec::new();

        // ------------------------------------------------------------------
        // Rule 1: rdfs:subClassOf transitivity
        //   IF (?A rdfs:subClassOf ?B) AND (?B rdfs:subClassOf ?C)
        //   THEN (?A rdfs:subClassOf ?C)
        // ------------------------------------------------------------------
        {
            let sub_class_pairs = collect_pairs(store, RDFS_SUB_CLASS_OF);
            for (a, b) in &sub_class_pairs {
                for (b2, c) in &sub_class_pairs {
                    if b == b2 && a != c {
                        inferred.push((a.clone(), RDFS_SUB_CLASS_OF.to_string(), c.clone()));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Rule 2: rdfs:subPropertyOf transitivity
        //   IF (?p rdfs:subPropertyOf ?q) AND (?q rdfs:subPropertyOf ?r)
        //   THEN (?p rdfs:subPropertyOf ?r)
        // ------------------------------------------------------------------
        {
            let sub_prop_pairs = collect_pairs(store, RDFS_SUB_PROPERTY_OF);
            for (p, q) in &sub_prop_pairs {
                for (q2, r) in &sub_prop_pairs {
                    if q == q2 && p != r {
                        inferred.push((p.clone(), RDFS_SUB_PROPERTY_OF.to_string(), r.clone()));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Rule 3: rdf:type propagation through rdfs:subClassOf
        //   IF (?x rdf:type ?C) AND (?C rdfs:subClassOf ?D)
        //   THEN (?x rdf:type ?D)
        // ------------------------------------------------------------------
        {
            let sub_class_pairs = collect_pairs(store, RDFS_SUB_CLASS_OF);
            let type_pairs = collect_pairs(store, RDF_TYPE);
            for (x, c) in &type_pairs {
                for (c2, d) in &sub_class_pairs {
                    if c == c2 {
                        inferred.push((x.clone(), RDF_TYPE.to_string(), d.clone()));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Rule 4: rdfs:subPropertyOf propagation
        //   IF (?p rdfs:subPropertyOf ?q) AND (?s ?p ?o)
        //   THEN (?s ?q ?o)
        // ------------------------------------------------------------------
        {
            let sub_prop_pairs = collect_pairs(store, RDFS_SUB_PROPERTY_OF);
            // Snapshot current triples to avoid modifying while iterating
            let all: Vec<(String, String, String)> = store
                .all_triples()
                .map(|t| (t.subject.clone(), t.predicate.clone(), t.object.clone()))
                .collect();
            for (p, q) in &sub_prop_pairs {
                for (s, pred, o) in &all {
                    if pred == p {
                        inferred.push((s.clone(), q.clone(), o.clone()));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Rule 5: rdfs:domain inference
        //   IF (?p rdfs:domain ?C) AND (?s ?p ?o)
        //   THEN (?s rdf:type ?C)
        // ------------------------------------------------------------------
        {
            let domain_pairs = collect_pairs(store, RDFS_DOMAIN);
            let all: Vec<(String, String, String)> = store
                .all_triples()
                .map(|t| (t.subject.clone(), t.predicate.clone(), t.object.clone()))
                .collect();
            for (p, c) in &domain_pairs {
                for (s, pred, _o) in &all {
                    if pred == p {
                        inferred.push((s.clone(), RDF_TYPE.to_string(), c.clone()));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Rule 6: rdfs:range inference
        //   IF (?p rdfs:range ?C) AND (?s ?p ?o)
        //   THEN (?o rdf:type ?C)
        // ------------------------------------------------------------------
        {
            let range_pairs = collect_pairs(store, RDFS_RANGE);
            let all: Vec<(String, String, String)> = store
                .all_triples()
                .map(|t| (t.subject.clone(), t.predicate.clone(), t.object.clone()))
                .collect();
            for (p, c) in &range_pairs {
                for (_s, pred, o) in &all {
                    if pred == p {
                        inferred.push((_s.clone(), RDF_TYPE.to_string(), c.clone()));
                        // rdfs:range says the *object* has the type
                        inferred.push((o.clone(), RDF_TYPE.to_string(), c.clone()));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Insert new triples (only novel ones)
        // ------------------------------------------------------------------
        if !self.config.materialize_all {
            // Just count what would be new
            let new_count = inferred
                .iter()
                .filter(|(s, p, o)| !store.contains(s, p, o))
                .count() as u32;
            return new_count;
        }

        let mut inserted = 0u32;
        // Deduplicate within the inferred set before inserting
        let unique: HashSet<_> = inferred.into_iter().collect();
        for (s, p, o) in unique {
            if store.insert(&s, &p, &o) {
                inserted += 1;
            }
        }

        inserted
    }
}

// -----------------------------------------------------------------------
// Top-level convenience function
// -----------------------------------------------------------------------

/// Apply RDFS inference to the store with default configuration.
///
/// Returns the total number of new triples added.
pub fn apply_rdfs_inference(store: &mut OxiRSStore) -> u32 {
    RdfsReasoner::default_config().apply(store)
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Collect all (subject, object) pairs from triples with the given predicate
fn collect_pairs(store: &OxiRSStore, predicate: &str) -> Vec<(String, String)> {
    store
        .all_triples()
        .filter(|t| t.predicate == predicate)
        .map(|t| (t.subject.clone(), t.object.clone()))
        .collect()
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> OxiRSStore {
        OxiRSStore::new()
    }

    // ---- subClassOf transitivity ----

    #[test]
    fn test_sub_class_of_transitivity() {
        let mut store = make_store();
        // Dog rdfs:subClassOf Animal
        // Animal rdfs:subClassOf LivingThing
        store.insert("http://Dog", RDFS_SUB_CLASS_OF, "http://Animal");
        store.insert("http://Animal", RDFS_SUB_CLASS_OF, "http://LivingThing");

        let added = apply_rdfs_inference(&mut store);
        assert!(added > 0);
        // Dog should be inferred as subClassOf LivingThing
        assert!(store.contains("http://Dog", RDFS_SUB_CLASS_OF, "http://LivingThing"));
    }

    #[test]
    fn test_sub_class_of_chain_of_three() {
        let mut store = make_store();
        store.insert("http://Poodle", RDFS_SUB_CLASS_OF, "http://Dog");
        store.insert("http://Dog", RDFS_SUB_CLASS_OF, "http://Animal");
        store.insert("http://Animal", RDFS_SUB_CLASS_OF, "http://LivingThing");

        apply_rdfs_inference(&mut store);
        assert!(store.contains("http://Poodle", RDFS_SUB_CLASS_OF, "http://Animal"));
        assert!(store.contains("http://Poodle", RDFS_SUB_CLASS_OF, "http://LivingThing"));
        assert!(store.contains("http://Dog", RDFS_SUB_CLASS_OF, "http://LivingThing"));
    }

    // ---- subPropertyOf transitivity ----

    #[test]
    fn test_sub_property_of_transitivity() {
        let mut store = make_store();
        store.insert("http://hasMother", RDFS_SUB_PROPERTY_OF, "http://hasParent");
        store.insert(
            "http://hasParent",
            RDFS_SUB_PROPERTY_OF,
            "http://hasAncestor",
        );

        apply_rdfs_inference(&mut store);
        assert!(store.contains(
            "http://hasMother",
            RDFS_SUB_PROPERTY_OF,
            "http://hasAncestor"
        ));
    }

    // ---- rdf:type propagation ----

    #[test]
    fn test_type_propagation_through_sub_class() {
        let mut store = make_store();
        store.insert("http://fido", RDF_TYPE, "http://Dog");
        store.insert("http://Dog", RDFS_SUB_CLASS_OF, "http://Animal");

        apply_rdfs_inference(&mut store);
        assert!(store.contains("http://fido", RDF_TYPE, "http://Animal"));
    }

    #[test]
    fn test_type_propagation_chain() {
        let mut store = make_store();
        store.insert("http://fido", RDF_TYPE, "http://Dog");
        store.insert("http://Dog", RDFS_SUB_CLASS_OF, "http://Animal");
        store.insert("http://Animal", RDFS_SUB_CLASS_OF, "http://LivingThing");

        apply_rdfs_inference(&mut store);
        assert!(store.contains("http://fido", RDF_TYPE, "http://Animal"));
        assert!(store.contains("http://fido", RDF_TYPE, "http://LivingThing"));
    }

    // ---- rdfs:domain inference ----

    #[test]
    fn test_domain_inference() {
        let mut store = make_store();
        // hasFather rdfs:domain http://Person
        store.insert("http://hasFather", RDFS_DOMAIN, "http://Person");
        // :alice http://hasFather :bob
        store.insert("http://alice", "http://hasFather", "http://bob");

        apply_rdfs_inference(&mut store);
        // alice should now be typed as Person
        assert!(store.contains("http://alice", RDF_TYPE, "http://Person"));
    }

    // ---- rdfs:range inference ----

    #[test]
    fn test_range_inference() {
        let mut store = make_store();
        // hasName rdfs:range xsd:string
        store.insert(
            "http://hasName",
            RDFS_RANGE,
            "http://www.w3.org/2001/XMLSchema#string",
        );
        store.insert("http://alice", "http://hasName", "http://NameNode");

        apply_rdfs_inference(&mut store);
        // The object (NameNode) should be typed as xsd:string
        assert!(store.contains(
            "http://NameNode",
            RDF_TYPE,
            "http://www.w3.org/2001/XMLSchema#string"
        ));
    }

    // ---- subPropertyOf propagation ----

    #[test]
    fn test_sub_property_of_propagation() {
        let mut store = make_store();
        // hasMother rdfs:subPropertyOf hasParent
        store.insert("http://hasMother", RDFS_SUB_PROPERTY_OF, "http://hasParent");
        // :alice hasMother :carol
        store.insert("http://alice", "http://hasMother", "http://carol");

        apply_rdfs_inference(&mut store);
        // alice hasParent carol should be inferred
        assert!(store.contains("http://alice", "http://hasParent", "http://carol"));
    }

    // ---- InferenceConfig ----

    #[test]
    fn test_max_depth_config() {
        let mut store = make_store();
        // Three-level chain: only 2 levels should be resolved with max_depth=1
        store.insert("http://C", RDFS_SUB_CLASS_OF, "http://B");
        store.insert("http://B", RDFS_SUB_CLASS_OF, "http://A");

        let config = InferenceConfig::with_max_depth(1);
        let reasoner = RdfsReasoner::new(config);
        reasoner.apply(&mut store);

        // One transitive step should be inferred (C subClassOf A)
        assert!(store.contains("http://C", RDFS_SUB_CLASS_OF, "http://A"));
    }

    #[test]
    fn test_no_self_loops_from_transitivity() {
        let mut store = make_store();
        store.insert("http://Dog", RDFS_SUB_CLASS_OF, "http://Animal");

        apply_rdfs_inference(&mut store);
        // Dog should NOT be inferred as subClassOf itself
        assert!(!store.contains("http://Dog", RDFS_SUB_CLASS_OF, "http://Dog"));
    }

    #[test]
    fn test_idempotent() {
        let mut store = make_store();
        store.insert("http://Dog", RDFS_SUB_CLASS_OF, "http://Animal");
        store.insert("http://fido", RDF_TYPE, "http://Dog");

        let added_first = apply_rdfs_inference(&mut store);
        let added_second = apply_rdfs_inference(&mut store);
        assert!(added_first > 0);
        // Second run should add nothing new
        assert_eq!(added_second, 0);
    }
}
