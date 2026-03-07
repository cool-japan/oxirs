//! Semantic Reasoner for RDFS/OWL Inference
//!
//! This module provides semantic reasoning capabilities for enhanced schema alignment:
//! - RDFS entailment rules (subClass, subProperty, domain, range)
//! - OWL inference (transitivity, symmetry, inverse properties)
//! - Class hierarchy reasoning
//! - Property chain reasoning
//! - Equivalence reasoning (owl:sameAs, owl:equivalentClass, owl:equivalentProperty)
//! - Disjointness checking
//!
//! Used by schema_alignment module for intelligent ontology matching.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Semantic reasoner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasonerConfig {
    /// Enable RDFS inference
    pub enable_rdfs: bool,
    /// Enable OWL inference
    pub enable_owl: bool,
    /// Maximum reasoning depth
    pub max_depth: usize,
    /// Enable transitive closure
    pub enable_transitive_closure: bool,
    /// Enable property chain reasoning
    pub enable_property_chains: bool,
}

impl Default for ReasonerConfig {
    fn default() -> Self {
        Self {
            enable_rdfs: true,
            enable_owl: true,
            max_depth: 10,
            enable_transitive_closure: true,
            enable_property_chains: true,
        }
    }
}

/// Semantic reasoner for ontology inference
pub struct SemanticReasoner {
    config: ReasonerConfig,
    /// Class hierarchy (subClassOf relationships)
    class_hierarchy: HashMap<String, HashSet<String>>,
    /// Property hierarchy (subPropertyOf relationships)
    property_hierarchy: HashMap<String, HashSet<String>>,
    /// Transitive properties
    transitive_properties: HashSet<String>,
    /// Symmetric properties
    symmetric_properties: HashSet<String>,
    /// Inverse properties
    inverse_properties: HashMap<String, String>,
    /// Functional properties
    functional_properties: HashSet<String>,
    /// Inverse functional properties
    #[allow(dead_code)]
    inverse_functional_properties: HashSet<String>,
    /// Equivalent classes
    equivalent_classes: HashMap<String, HashSet<String>>,
    /// Equivalent properties
    equivalent_properties: HashMap<String, HashSet<String>>,
    /// Disjoint classes
    disjoint_classes: Vec<(String, String)>,
    /// Property domains
    property_domains: HashMap<String, String>,
    /// Property ranges
    property_ranges: HashMap<String, String>,
    /// Property chains
    property_chains: Vec<PropertyChain>,
}

impl SemanticReasoner {
    /// Create a new semantic reasoner
    pub fn new(config: ReasonerConfig) -> Self {
        Self {
            config,
            class_hierarchy: HashMap::new(),
            property_hierarchy: HashMap::new(),
            transitive_properties: HashSet::new(),
            symmetric_properties: HashSet::new(),
            inverse_properties: HashMap::new(),
            functional_properties: HashSet::new(),
            inverse_functional_properties: HashSet::new(),
            equivalent_classes: HashMap::new(),
            equivalent_properties: HashMap::new(),
            disjoint_classes: Vec::new(),
            property_domains: HashMap::new(),
            property_ranges: HashMap::new(),
            property_chains: Vec::new(),
        }
    }

    /// Add a subclass relationship
    pub fn add_subclass(&mut self, subclass: &str, superclass: &str) {
        self.class_hierarchy
            .entry(subclass.to_string())
            .or_default()
            .insert(superclass.to_string());
    }

    /// Add a sub-property relationship
    pub fn add_subproperty(&mut self, subprop: &str, superprop: &str) {
        self.property_hierarchy
            .entry(subprop.to_string())
            .or_default()
            .insert(superprop.to_string());
    }

    /// Mark a property as transitive
    pub fn add_transitive_property(&mut self, property: &str) {
        self.transitive_properties.insert(property.to_string());
    }

    /// Mark a property as symmetric
    pub fn add_symmetric_property(&mut self, property: &str) {
        self.symmetric_properties.insert(property.to_string());
    }

    /// Add inverse property relationship
    pub fn add_inverse_properties(&mut self, prop1: &str, prop2: &str) {
        self.inverse_properties
            .insert(prop1.to_string(), prop2.to_string());
        self.inverse_properties
            .insert(prop2.to_string(), prop1.to_string());
    }

    /// Add equivalent classes
    pub fn add_equivalent_classes(&mut self, class1: &str, class2: &str) {
        self.equivalent_classes
            .entry(class1.to_string())
            .or_default()
            .insert(class2.to_string());
        self.equivalent_classes
            .entry(class2.to_string())
            .or_default()
            .insert(class1.to_string());
    }

    /// Add equivalent properties
    pub fn add_equivalent_properties(&mut self, prop1: &str, prop2: &str) {
        self.equivalent_properties
            .entry(prop1.to_string())
            .or_default()
            .insert(prop2.to_string());
        self.equivalent_properties
            .entry(prop2.to_string())
            .or_default()
            .insert(prop1.to_string());
    }

    /// Add disjoint classes
    pub fn add_disjoint_classes(&mut self, class1: &str, class2: &str) {
        self.disjoint_classes
            .push((class1.to_string(), class2.to_string()));
    }

    /// Set property domain
    pub fn set_property_domain(&mut self, property: &str, domain: &str) {
        self.property_domains
            .insert(property.to_string(), domain.to_string());
    }

    /// Set property range
    pub fn set_property_range(&mut self, property: &str, range: &str) {
        self.property_ranges
            .insert(property.to_string(), range.to_string());
    }

    /// Add property chain (e.g., hasParent o hasParent => hasGrandparent)
    pub fn add_property_chain(&mut self, chain: Vec<String>, result_property: String) {
        self.property_chains.push(PropertyChain {
            chain,
            result_property,
        });
    }

    /// Compute transitive closure of class hierarchy
    pub fn compute_class_closure(&mut self) -> Result<()> {
        if !self.config.enable_transitive_closure {
            return Ok(());
        }

        info!("Computing transitive closure of class hierarchy");

        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < self.config.max_depth {
            changed = false;
            iterations += 1;

            let classes: Vec<String> = self.class_hierarchy.keys().cloned().collect();

            for class in &classes {
                let superclasses: Vec<String> = self
                    .class_hierarchy
                    .get(class)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect();

                for superclass in &superclasses {
                    if let Some(super_superclasses) = self.class_hierarchy.get(superclass) {
                        for super_superclass in super_superclasses.clone() {
                            let entry = self.class_hierarchy.entry(class.clone()).or_default();
                            if entry.insert(super_superclass) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        debug!("Transitive closure computed in {} iterations", iterations);
        Ok(())
    }

    /// Get all superclasses of a class (including transitive)
    pub fn get_superclasses(&self, class: &str) -> HashSet<String> {
        self.class_hierarchy.get(class).cloned().unwrap_or_default()
    }

    /// Get all subclasses of a class
    pub fn get_subclasses(&self, class: &str) -> HashSet<String> {
        let mut subclasses = HashSet::new();

        for (subclass, superclasses) in &self.class_hierarchy {
            if superclasses.contains(class) {
                subclasses.insert(subclass.clone());
            }
        }

        subclasses
    }

    /// Check if class1 is a subclass of class2
    pub fn is_subclass_of(&self, class1: &str, class2: &str) -> bool {
        if class1 == class2 {
            return true;
        }

        self.get_superclasses(class1).contains(class2)
    }

    /// Get all super-properties of a property (including transitive)
    pub fn get_superproperties(&self, property: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut to_visit = vec![property.to_string()];
        let mut visited = HashSet::new();

        while let Some(prop) = to_visit.pop() {
            if !visited.insert(prop.clone()) {
                continue;
            }

            if let Some(superprops) = self.property_hierarchy.get(&prop) {
                for superprop in superprops {
                    result.insert(superprop.clone());
                    to_visit.push(superprop.clone());
                }
            }
        }

        result
    }

    /// Check if property1 is a sub-property of property2
    pub fn is_subproperty_of(&self, property1: &str, property2: &str) -> bool {
        if property1 == property2 {
            return true;
        }

        self.get_superproperties(property1).contains(property2)
    }

    /// Infer new triples based on reasoning rules
    pub fn infer_triples(&self, triples: &[Triple]) -> Vec<Triple> {
        let mut inferred = Vec::new();

        // RDFS subClassOf reasoning
        if self.config.enable_rdfs {
            inferred.extend(self.infer_from_subclass(triples));
            inferred.extend(self.infer_from_domain_range(triples));
        }

        // OWL reasoning
        if self.config.enable_owl {
            inferred.extend(self.infer_from_transitive(triples));
            inferred.extend(self.infer_from_symmetric(triples));
            inferred.extend(self.infer_from_inverse(triples));

            if self.config.enable_property_chains {
                inferred.extend(self.infer_from_property_chains(triples));
            }
        }

        inferred
    }

    /// Infer triples from subclass relationships
    fn infer_from_subclass(&self, triples: &[Triple]) -> Vec<Triple> {
        let mut inferred = Vec::new();

        for triple in triples {
            // If (x rdf:type C1) and (C1 rdfs:subClassOf C2), infer (x rdf:type C2)
            if triple.predicate == "rdf:type" {
                for superclass in self.get_superclasses(&triple.object) {
                    inferred.push(Triple {
                        subject: triple.subject.clone(),
                        predicate: "rdf:type".to_string(),
                        object: superclass,
                    });
                }
            }
        }

        inferred
    }

    /// Infer triples from property domain and range
    fn infer_from_domain_range(&self, triples: &[Triple]) -> Vec<Triple> {
        let mut inferred = Vec::new();

        for triple in triples {
            // Domain inference: if (x P y) and (P rdfs:domain C), infer (x rdf:type C)
            if let Some(domain) = self.property_domains.get(&triple.predicate) {
                inferred.push(Triple {
                    subject: triple.subject.clone(),
                    predicate: "rdf:type".to_string(),
                    object: domain.clone(),
                });
            }

            // Range inference: if (x P y) and (P rdfs:range C), infer (y rdf:type C)
            if let Some(range) = self.property_ranges.get(&triple.predicate) {
                inferred.push(Triple {
                    subject: triple.object.clone(),
                    predicate: "rdf:type".to_string(),
                    object: range.clone(),
                });
            }
        }

        inferred
    }

    /// Infer triples from transitive properties
    fn infer_from_transitive(&self, triples: &[Triple]) -> Vec<Triple> {
        let mut inferred = Vec::new();

        // Build property chains for transitive properties
        for prop in &self.transitive_properties {
            let mut chains: HashMap<String, Vec<String>> = HashMap::new();

            // Collect all (x prop y) triples
            for triple in triples {
                if &triple.predicate == prop {
                    chains
                        .entry(triple.subject.clone())
                        .or_default()
                        .push(triple.object.clone());
                }
            }

            // Compute transitive closure
            for start in chains.keys() {
                let mut reachable = HashSet::new();
                let mut to_visit = vec![start.clone()];

                while let Some(current) = to_visit.pop() {
                    if let Some(targets) = chains.get(&current) {
                        for target in targets {
                            if reachable.insert(target.clone()) {
                                to_visit.push(target.clone());

                                // Infer transitive triple
                                inferred.push(Triple {
                                    subject: start.clone(),
                                    predicate: prop.clone(),
                                    object: target.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        inferred
    }

    /// Infer triples from symmetric properties
    fn infer_from_symmetric(&self, triples: &[Triple]) -> Vec<Triple> {
        let mut inferred = Vec::new();

        for triple in triples {
            if self.symmetric_properties.contains(&triple.predicate) {
                // If (x P y) and P is symmetric, infer (y P x)
                inferred.push(Triple {
                    subject: triple.object.clone(),
                    predicate: triple.predicate.clone(),
                    object: triple.subject.clone(),
                });
            }
        }

        inferred
    }

    /// Infer triples from inverse properties
    fn infer_from_inverse(&self, triples: &[Triple]) -> Vec<Triple> {
        let mut inferred = Vec::new();

        for triple in triples {
            if let Some(inverse_prop) = self.inverse_properties.get(&triple.predicate) {
                // If (x P y) and (P owl:inverseOf Q), infer (y Q x)
                inferred.push(Triple {
                    subject: triple.object.clone(),
                    predicate: inverse_prop.clone(),
                    object: triple.subject.clone(),
                });
            }
        }

        inferred
    }

    /// Infer triples from property chains
    fn infer_from_property_chains(&self, triples: &[Triple]) -> Vec<Triple> {
        let mut inferred = Vec::new();

        for chain_rule in &self.property_chains {
            if chain_rule.chain.len() != 2 {
                continue; // Only support 2-property chains for now
            }

            let prop1 = &chain_rule.chain[0];
            let prop2 = &chain_rule.chain[1];

            // Find all (x prop1 y) triples
            for triple1 in triples {
                if &triple1.predicate == prop1 {
                    // Find all (y prop2 z) triples
                    for triple2 in triples {
                        if &triple2.predicate == prop2 && triple2.subject == triple1.object {
                            // Infer (x result_property z)
                            inferred.push(Triple {
                                subject: triple1.subject.clone(),
                                predicate: chain_rule.result_property.clone(),
                                object: triple2.object.clone(),
                            });
                        }
                    }
                }
            }
        }

        inferred
    }

    /// Check for inconsistencies in the ontology
    pub fn check_consistency(&self, triples: &[Triple]) -> Vec<InconsistencyReport> {
        let mut inconsistencies = Vec::new();

        // Check disjoint classes
        for triple in triples {
            if triple.predicate == "rdf:type" {
                for (class1, class2) in &self.disjoint_classes {
                    let types = triples
                        .iter()
                        .filter(|t| t.subject == triple.subject && t.predicate == "rdf:type")
                        .map(|t| &t.object)
                        .collect::<HashSet<_>>();

                    if types.contains(class1) && types.contains(class2) {
                        inconsistencies.push(InconsistencyReport {
                            inconsistency_type: InconsistencyType::DisjointClasses,
                            description: format!(
                                "Instance {} belongs to disjoint classes {} and {}",
                                triple.subject, class1, class2
                            ),
                            affected_entities: vec![
                                triple.subject.clone(),
                                class1.clone(),
                                class2.clone(),
                            ],
                        });
                    }
                }
            }
        }

        // Check functional properties
        for prop in &self.functional_properties {
            let mut value_map: HashMap<String, Vec<String>> = HashMap::new();

            for triple in triples {
                if &triple.predicate == prop {
                    value_map
                        .entry(triple.subject.clone())
                        .or_default()
                        .push(triple.object.clone());
                }
            }

            for (subject, objects) in value_map {
                if objects.len() > 1 {
                    inconsistencies.push(InconsistencyReport {
                        inconsistency_type: InconsistencyType::FunctionalProperty,
                        description: format!(
                            "Functional property {} has multiple values for subject {}",
                            prop, subject
                        ),
                        affected_entities: vec![subject, prop.clone()],
                    });
                }
            }
        }

        inconsistencies
    }

    /// Compute equivalence classes for owl:sameAs
    pub fn compute_equivalence_classes(
        &self,
        same_as_triples: &[(String, String)],
    ) -> Vec<HashSet<String>> {
        let mut union_find = UnionFind::new();

        for (entity1, entity2) in same_as_triples {
            union_find.union(entity1, entity2);
        }

        union_find.get_components()
    }
}

impl Default for SemanticReasoner {
    fn default() -> Self {
        Self::new(ReasonerConfig::default())
    }
}

/// RDF triple representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Property chain rule
#[derive(Debug, Clone)]
struct PropertyChain {
    chain: Vec<String>,
    result_property: String,
}

/// Inconsistency report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InconsistencyReport {
    pub inconsistency_type: InconsistencyType,
    pub description: String,
    pub affected_entities: Vec<String>,
}

/// Type of inconsistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InconsistencyType {
    DisjointClasses,
    FunctionalProperty,
    InverseFunctionalProperty,
    CardinalityViolation,
}

/// Union-Find data structure for equivalence classes
struct UnionFind {
    parent: HashMap<String, String>,
    rank: HashMap<String, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }

    fn find(&mut self, x: &str) -> String {
        if !self.parent.contains_key(x) {
            self.parent.insert(x.to_string(), x.to_string());
            self.rank.insert(x.to_string(), 0);
            return x.to_string();
        }

        let parent = self.parent[x].clone();
        if parent != x {
            let root = self.find(&parent);
            self.parent.insert(x.to_string(), root.clone());
            root
        } else {
            parent
        }
    }

    fn union(&mut self, x: &str, y: &str) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return;
        }

        let rank_x = self.rank[&root_x];
        let rank_y = self.rank[&root_y];

        if rank_x < rank_y {
            self.parent.insert(root_x, root_y);
        } else if rank_x > rank_y {
            self.parent.insert(root_y, root_x);
        } else {
            self.parent.insert(root_y, root_x.clone());
            *self
                .rank
                .get_mut(&root_x)
                .expect("insertion should succeed") += 1;
        }
    }

    fn get_components(&mut self) -> Vec<HashSet<String>> {
        let mut components: HashMap<String, HashSet<String>> = HashMap::new();

        for entity in self.parent.keys().cloned().collect::<Vec<_>>() {
            let root = self.find(&entity);
            components.entry(root).or_default().insert(entity);
        }

        components.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoner_creation() {
        let config = ReasonerConfig::default();
        let reasoner = SemanticReasoner::new(config);

        assert!(reasoner.class_hierarchy.is_empty());
        assert!(reasoner.transitive_properties.is_empty());
    }

    #[test]
    fn test_subclass_reasoning() {
        let mut reasoner = SemanticReasoner::new(ReasonerConfig::default());

        reasoner.add_subclass("Dog", "Animal");
        reasoner.add_subclass("Animal", "LivingThing");
        reasoner
            .compute_class_closure()
            .expect("reasoning should succeed");

        assert!(reasoner.is_subclass_of("Dog", "Animal"));
        assert!(reasoner.is_subclass_of("Dog", "LivingThing"));
        assert!(reasoner.is_subclass_of("Dog", "Dog"));
    }

    #[test]
    fn test_transitive_property_inference() {
        let config = ReasonerConfig::default();
        let mut reasoner = SemanticReasoner::new(config);

        reasoner.add_transitive_property("ancestorOf");

        let triples = vec![
            Triple {
                subject: "Alice".to_string(),
                predicate: "ancestorOf".to_string(),
                object: "Bob".to_string(),
            },
            Triple {
                subject: "Bob".to_string(),
                predicate: "ancestorOf".to_string(),
                object: "Charlie".to_string(),
            },
        ];

        let inferred = reasoner.infer_triples(&triples);

        // Should infer Alice ancestorOf Charlie
        assert!(inferred
            .iter()
            .any(|t| t.subject == "Alice" && t.predicate == "ancestorOf" && t.object == "Charlie"));
    }

    #[test]
    fn test_symmetric_property_inference() {
        let config = ReasonerConfig::default();
        let mut reasoner = SemanticReasoner::new(config);

        reasoner.add_symmetric_property("friendOf");

        let triples = vec![Triple {
            subject: "Alice".to_string(),
            predicate: "friendOf".to_string(),
            object: "Bob".to_string(),
        }];

        let inferred = reasoner.infer_triples(&triples);

        // Should infer Bob friendOf Alice
        assert!(inferred
            .iter()
            .any(|t| t.subject == "Bob" && t.predicate == "friendOf" && t.object == "Alice"));
    }

    #[test]
    fn test_inverse_property_inference() {
        let config = ReasonerConfig::default();
        let mut reasoner = SemanticReasoner::new(config);

        reasoner.add_inverse_properties("hasChild", "hasParent");

        let triples = vec![Triple {
            subject: "Alice".to_string(),
            predicate: "hasChild".to_string(),
            object: "Bob".to_string(),
        }];

        let inferred = reasoner.infer_triples(&triples);

        // Should infer Bob hasParent Alice
        assert!(inferred
            .iter()
            .any(|t| t.subject == "Bob" && t.predicate == "hasParent" && t.object == "Alice"));
    }

    #[test]
    fn test_equivalence_classes() {
        let config = ReasonerConfig::default();
        let reasoner = SemanticReasoner::new(config);

        let same_as = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
            ("D".to_string(), "E".to_string()),
        ];

        let classes = reasoner.compute_equivalence_classes(&same_as);

        assert_eq!(classes.len(), 2);
        assert!(classes
            .iter()
            .any(|c| c.contains("A") && c.contains("B") && c.contains("C")));
        assert!(classes.iter().any(|c| c.contains("D") && c.contains("E")));
    }
}
