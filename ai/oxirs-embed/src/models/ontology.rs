//! Ontology-aware embedding models
//!
//! This module provides embedding models that understand and respect RDF schema
//! relationships such as class hierarchies, property constraints, and semantic
//! relationships defined in OWL and RDFS ontologies.

use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Ontology relationship types that are semantically meaningful
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OntologyRelation {
    /// rdfs:subClassOf - class hierarchy relationships
    SubClassOf,
    /// owl:equivalentClass - equivalent class relationships
    EquivalentClass,
    /// owl:disjointWith - disjoint class relationships
    DisjointWith,
    /// rdfs:domain - property domain constraints
    Domain,
    /// rdfs:range - property range constraints
    Range,
    /// owl:inverseOf - inverse property relationships
    InverseOf,
    /// owl:functionalProperty - functional property constraints
    FunctionalProperty,
    /// owl:symmetricProperty - symmetric property constraints
    SymmetricProperty,
    /// owl:transitiveProperty - transitive property constraints
    TransitiveProperty,
}

/// Property characteristics for enhanced constraint handling
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropertyCharacteristics {
    pub is_functional: bool,
    pub is_inverse_functional: bool,
    pub is_symmetric: bool,
    pub is_asymmetric: bool,
    pub is_transitive: bool,
    pub is_reflexive: bool,
    pub is_irreflexive: bool,
    pub has_inverse: Option<String>,
    pub domain_classes: HashSet<String>,
    pub range_classes: HashSet<String>,
}

impl PropertyCharacteristics {
    /// Check if property has any domain constraints
    pub fn has_domain_constraints(&self) -> bool {
        !self.domain_classes.is_empty()
    }

    /// Check if property has any range constraints
    pub fn has_range_constraints(&self) -> bool {
        !self.range_classes.is_empty()
    }

    /// Check if entity satisfies domain constraints for this property
    pub fn satisfies_domain(&self, entity_type: &str) -> bool {
        if self.domain_classes.is_empty() {
            true // No constraints means always satisfied
        } else {
            self.domain_classes.contains(entity_type)
        }
    }

    /// Check if entity satisfies range constraints for this property
    pub fn satisfies_range(&self, entity_type: &str) -> bool {
        if self.range_classes.is_empty() {
            true // No constraints means always satisfied
        } else {
            self.range_classes.contains(entity_type)
        }
    }
}

impl OntologyRelation {
    /// Convert from RDF predicate IRI to ontology relation type
    pub fn from_iri(iri: &str) -> Option<Self> {
        match iri {
            "http://www.w3.org/2000/01/rdf-schema#subClassOf" => Some(Self::SubClassOf),
            "http://www.w3.org/2002/07/owl#equivalentClass" => Some(Self::EquivalentClass),
            "http://www.w3.org/2002/07/owl#disjointWith" => Some(Self::DisjointWith),
            "http://www.w3.org/2000/01/rdf-schema#domain" => Some(Self::Domain),
            "http://www.w3.org/2000/01/rdf-schema#range" => Some(Self::Range),
            "http://www.w3.org/2002/07/owl#inverseOf" => Some(Self::InverseOf),
            "http://www.w3.org/2002/07/owl#FunctionalProperty" => Some(Self::FunctionalProperty),
            "http://www.w3.org/2002/07/owl#SymmetricProperty" => Some(Self::SymmetricProperty),
            "http://www.w3.org/2002/07/owl#TransitiveProperty" => Some(Self::TransitiveProperty),
            _ => None,
        }
    }
}

/// Configuration for ontology-aware embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyAwareConfig {
    pub base_config: ModelConfig,
    /// Weight for hierarchy constraint enforcement
    pub hierarchy_weight: f32,
    /// Weight for equivalence constraint enforcement
    pub equivalence_weight: f32,
    /// Weight for disjointness constraint enforcement
    pub disjoint_weight: f32,
    /// Weight for property domain/range constraint enforcement
    pub property_constraint_weight: f32,
    /// Weight for multi-modal alignment
    pub cross_modal_weight: f32,
    /// Whether to use transitive closure for hierarchies
    pub use_transitive_closure: bool,
    /// Maximum depth for transitive closure computation
    pub max_transitive_depth: usize,
    /// Whether to normalize embeddings for hierarchy preservation
    pub normalize_for_hierarchy: bool,
    /// Margin for ranking loss in hierarchy constraints
    pub hierarchy_margin: f32,
    /// Enable contrastive learning for cross-modal alignment
    pub enable_contrastive_learning: bool,
    /// Temperature parameter for contrastive learning
    pub contrastive_temperature: f32,
    /// Enable mutual information maximization
    pub enable_mutual_info_max: bool,
    /// Enable property chain inference
    pub enable_property_chains: bool,
    /// Maximum length for property chains
    pub max_property_chain_length: usize,
}

impl Default for OntologyAwareConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            hierarchy_weight: 1.0,
            equivalence_weight: 2.0,
            disjoint_weight: 1.5,
            property_constraint_weight: 1.2,
            cross_modal_weight: 0.8,
            use_transitive_closure: true,
            max_transitive_depth: 10,
            normalize_for_hierarchy: true,
            hierarchy_margin: 1.0,
            enable_contrastive_learning: true,
            contrastive_temperature: 0.1,
            enable_mutual_info_max: false,
            enable_property_chains: true,
            max_property_chain_length: 3,
        }
    }
}

/// Ontology-aware embedding model that respects semantic relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyAwareEmbedding {
    pub config: OntologyAwareConfig,
    /// Unique model identifier
    pub model_id: Uuid,
    /// Entity embeddings
    pub entity_embeddings: HashMap<String, Array1<f32>>,
    /// Relation embeddings
    pub relation_embeddings: HashMap<String, Array1<f32>>,
    /// Entity to index mapping
    pub entity_to_idx: HashMap<String, usize>,
    /// Relation to index mapping
    pub relation_to_idx: HashMap<String, usize>,
    /// Training triples
    pub triples: Vec<Triple>,
    /// Ontology constraints extracted from triples
    pub ontology_constraints: OntologyConstraints,
    /// Training statistics
    pub training_stats: TrainingStats,
    /// Model statistics
    pub model_stats: ModelStats,
    /// Whether the model has been trained
    pub is_trained: bool,
}

/// Ontology constraints extracted from RDF data
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OntologyConstraints {
    /// Class hierarchy: subclass -> set of superclasses
    pub class_hierarchy: HashMap<String, HashSet<String>>,
    /// Equivalent classes: class -> set of equivalent classes
    pub equivalent_classes: HashMap<String, HashSet<String>>,
    /// Disjoint classes: class -> set of disjoint classes
    pub disjoint_classes: HashMap<String, HashSet<String>>,
    /// Property domains: property -> set of domain classes
    pub property_domains: HashMap<String, HashSet<String>>,
    /// Property ranges: property -> set of range classes
    pub property_ranges: HashMap<String, HashSet<String>>,
    /// Inverse properties: property -> inverse property
    pub inverse_properties: HashMap<String, String>,
    /// Functional properties
    pub functional_properties: HashSet<String>,
    /// Symmetric properties
    pub symmetric_properties: HashSet<String>,
    /// Transitive properties
    pub transitive_properties: HashSet<String>,
    /// Transitive closure of class hierarchy
    pub transitive_hierarchy: HashMap<String, HashSet<String>>,
    /// Property chains: property -> list of property sequences that imply it
    pub property_chains: HashMap<String, Vec<Vec<String>>>,
    /// Cross-modal alignments: entity -> set of aligned entities from other modalities
    pub cross_modal_alignments: HashMap<String, HashSet<String>>,
    /// Property characteristics cache for fast lookup
    pub property_characteristics: HashMap<String, PropertyCharacteristics>,
}

impl OntologyConstraints {
    /// Compute transitive closure of class hierarchy
    pub fn compute_transitive_closure(&mut self, max_depth: usize) {
        self.transitive_hierarchy = self.class_hierarchy.clone();

        for _ in 0..max_depth {
            let mut changed = false;
            let current_hierarchy = self.transitive_hierarchy.clone();

            for (subclass, superclasses) in &current_hierarchy {
                let mut new_superclasses = superclasses.clone();

                // Add superclasses of superclasses
                for superclass in superclasses {
                    if let Some(super_superclasses) = current_hierarchy.get(superclass) {
                        for super_superclass in super_superclasses {
                            if !new_superclasses.contains(super_superclass)
                                && super_superclass != subclass
                            {
                                new_superclasses.insert(super_superclass.clone());
                                changed = true;
                            }
                        }
                    }
                }

                self.transitive_hierarchy
                    .insert(subclass.clone(), new_superclasses);
            }

            if !changed {
                break;
            }
        }
    }

    /// Check if one class is a subclass of another (considering transitive closure)
    pub fn is_subclass_of(&self, subclass: &str, superclass: &str) -> bool {
        if let Some(superclasses) = self.transitive_hierarchy.get(subclass) {
            superclasses.contains(superclass)
        } else {
            false
        }
    }

    /// Check if two classes are equivalent
    pub fn are_equivalent(&self, class1: &str, class2: &str) -> bool {
        if let Some(equivalent) = self.equivalent_classes.get(class1) {
            equivalent.contains(class2)
        } else {
            false
        }
    }

    /// Check if two classes are disjoint
    pub fn are_disjoint(&self, class1: &str, class2: &str) -> bool {
        if let Some(disjoint) = self.disjoint_classes.get(class1) {
            disjoint.contains(class2)
        } else {
            false
        }
    }

    /// Add property chain constraint
    pub fn add_property_chain(&mut self, target_property: &str, chain: Vec<String>) {
        self.property_chains
            .entry(target_property.to_string())
            .or_default()
            .push(chain);
    }

    /// Get property chains that imply the given property
    pub fn get_property_chains(&self, property: &str) -> Option<&Vec<Vec<String>>> {
        self.property_chains.get(property)
    }

    /// Add cross-modal alignment between entities
    pub fn add_cross_modal_alignment(&mut self, entity1: &str, entity2: &str) {
        self.cross_modal_alignments
            .entry(entity1.to_string())
            .or_default()
            .insert(entity2.to_string());

        // Symmetric relationship
        self.cross_modal_alignments
            .entry(entity2.to_string())
            .or_default()
            .insert(entity1.to_string());
    }

    /// Get cross-modal alignments for an entity
    pub fn get_cross_modal_alignments(&self, entity: &str) -> Option<&HashSet<String>> {
        self.cross_modal_alignments.get(entity)
    }

    /// Build property characteristics cache
    pub fn build_property_characteristics_cache(&mut self) {
        // Build comprehensive property characteristics
        let all_properties: HashSet<String> = self
            .property_domains
            .keys()
            .chain(self.property_ranges.keys())
            .chain(self.functional_properties.iter())
            .chain(self.symmetric_properties.iter())
            .chain(self.transitive_properties.iter())
            .chain(self.inverse_properties.keys())
            .cloned()
            .collect();

        for property in all_properties {
            let mut characteristics = PropertyCharacteristics {
                is_functional: self.functional_properties.contains(&property),
                is_symmetric: self.symmetric_properties.contains(&property),
                is_transitive: self.transitive_properties.contains(&property),
                has_inverse: self.inverse_properties.get(&property).cloned(),
                ..Default::default()
            };

            if let Some(domains) = self.property_domains.get(&property) {
                characteristics.domain_classes = domains.clone();
            }

            if let Some(ranges) = self.property_ranges.get(&property) {
                characteristics.range_classes = ranges.clone();
            }

            self.property_characteristics
                .insert(property, characteristics);
        }
    }

    /// Validate property usage based on domain/range constraints
    pub fn validate_property_usage(
        &self,
        subject: &str,
        property: &str,
        object: &str,
        entity_types: &HashMap<String, String>,
    ) -> bool {
        if let Some(characteristics) = self.property_characteristics.get(property) {
            // Check domain constraints
            if characteristics.has_domain_constraints() {
                if let Some(subject_type) = entity_types.get(subject) {
                    if !characteristics.satisfies_domain(subject_type) {
                        return false;
                    }
                }
            }

            // Check range constraints
            if characteristics.has_range_constraints() {
                if let Some(object_type) = entity_types.get(object) {
                    if !characteristics.satisfies_range(object_type) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Infer new triples based on property chains
    pub fn infer_from_property_chains(&self, existing_triples: &[Triple]) -> Vec<Triple> {
        let mut inferred_triples = Vec::new();

        for (target_property, chains) in &self.property_chains {
            for chain in chains {
                if chain.len() >= 2 {
                    // Find sequences of triples that match the property chain
                    inferred_triples.extend(self.find_chain_matches(
                        existing_triples,
                        target_property,
                        chain,
                    ));
                }
            }
        }

        inferred_triples
    }

    /// Find triple sequences that match a property chain pattern
    fn find_chain_matches(
        &self,
        triples: &[Triple],
        target_property: &str,
        chain: &[String],
    ) -> Vec<Triple> {
        let mut matches = Vec::new();

        // Build index of triples by predicate for faster lookup
        let mut triples_by_predicate: HashMap<String, Vec<&Triple>> = HashMap::new();
        for triple in triples {
            triples_by_predicate
                .entry(triple.predicate.iri.clone())
                .or_default()
                .push(triple);
        }

        // For simplicity, handle 2-property chains
        if chain.len() == 2 {
            let prop1 = &chain[0];
            let prop2 = &chain[1];

            if let (Some(triples1), Some(triples2)) = (
                triples_by_predicate.get(prop1),
                triples_by_predicate.get(prop2),
            ) {
                for t1 in triples1 {
                    for t2 in triples2 {
                        // Check if t1.object == t2.subject (chain connection)
                        if t1.object.iri == t2.subject.iri {
                            // Create inferred triple: t1.subject target_property t2.object
                            if let (Ok(subject), Ok(predicate), Ok(object)) = (
                                crate::NamedNode::new(&t1.subject.iri),
                                crate::NamedNode::new(target_property),
                                crate::NamedNode::new(&t2.object.iri),
                            ) {
                                matches.push(Triple::new(subject, predicate, object));
                            }
                        }
                    }
                }
            }
        }

        matches
    }
}

impl Default for TrainingStats {
    fn default() -> Self {
        Self {
            epochs_completed: 0,
            final_loss: 0.0,
            training_time_seconds: 0.0,
            convergence_achieved: false,
            loss_history: Vec::new(),
        }
    }
}

impl OntologyAwareEmbedding {
    /// Create new ontology-aware embedding model
    pub fn new(config: OntologyAwareConfig) -> Self {
        let model_id = Uuid::new_v4();
        let now = Utc::now();

        Self {
            model_id,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entity_to_idx: HashMap::new(),
            relation_to_idx: HashMap::new(),
            triples: Vec::new(),
            ontology_constraints: OntologyConstraints::default(),
            training_stats: TrainingStats::default(),
            model_stats: ModelStats {
                num_entities: 0,
                num_relations: 0,
                num_triples: 0,
                dimensions: config.base_config.dimensions,
                is_trained: false,
                model_type: "OntologyAware".to_string(),
                creation_time: now,
                last_training_time: None,
            },
            is_trained: false,
            config,
        }
    }

    /// Create configuration optimized for class hierarchies
    pub fn hierarchy_optimized_config(dimensions: usize) -> OntologyAwareConfig {
        OntologyAwareConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            hierarchy_weight: 2.0,
            equivalence_weight: 1.0,
            disjoint_weight: 1.0,
            property_constraint_weight: 1.0,
            cross_modal_weight: 0.5,
            use_transitive_closure: true,
            max_transitive_depth: 15,
            normalize_for_hierarchy: true,
            hierarchy_margin: 0.5,
            enable_contrastive_learning: false,
            contrastive_temperature: 0.1,
            enable_mutual_info_max: false,
            enable_property_chains: true,
            max_property_chain_length: 2,
        }
    }

    /// Create configuration optimized for property constraints
    pub fn property_optimized_config(dimensions: usize) -> OntologyAwareConfig {
        OntologyAwareConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            hierarchy_weight: 1.0,
            equivalence_weight: 1.5,
            disjoint_weight: 2.0,
            property_constraint_weight: 2.5,
            cross_modal_weight: 1.0,
            use_transitive_closure: true,
            max_transitive_depth: 8,
            normalize_for_hierarchy: false,
            hierarchy_margin: 1.0,
            enable_contrastive_learning: true,
            contrastive_temperature: 0.05,
            enable_mutual_info_max: true,
            enable_property_chains: true,
            max_property_chain_length: 3,
        }
    }

    /// Extract ontology constraints from triples
    fn extract_ontology_constraints(&mut self) {
        for triple in &self.triples {
            if let Some(relation_type) = OntologyRelation::from_iri(&triple.predicate.iri) {
                match relation_type {
                    OntologyRelation::SubClassOf => {
                        self.ontology_constraints
                            .class_hierarchy
                            .entry(triple.subject.iri.clone())
                            .or_default()
                            .insert(triple.object.iri.clone());
                    }
                    OntologyRelation::EquivalentClass => {
                        self.ontology_constraints
                            .equivalent_classes
                            .entry(triple.subject.iri.clone())
                            .or_default()
                            .insert(triple.object.iri.clone());
                        // Symmetric relationship
                        self.ontology_constraints
                            .equivalent_classes
                            .entry(triple.object.iri.clone())
                            .or_default()
                            .insert(triple.subject.iri.clone());
                    }
                    OntologyRelation::DisjointWith => {
                        self.ontology_constraints
                            .disjoint_classes
                            .entry(triple.subject.iri.clone())
                            .or_default()
                            .insert(triple.object.iri.clone());
                        // Symmetric relationship
                        self.ontology_constraints
                            .disjoint_classes
                            .entry(triple.object.iri.clone())
                            .or_default()
                            .insert(triple.subject.iri.clone());
                    }
                    OntologyRelation::Domain => {
                        self.ontology_constraints
                            .property_domains
                            .entry(triple.subject.iri.clone())
                            .or_default()
                            .insert(triple.object.iri.clone());
                    }
                    OntologyRelation::Range => {
                        self.ontology_constraints
                            .property_ranges
                            .entry(triple.subject.iri.clone())
                            .or_default()
                            .insert(triple.object.iri.clone());
                    }
                    OntologyRelation::InverseOf => {
                        self.ontology_constraints
                            .inverse_properties
                            .insert(triple.subject.iri.clone(), triple.object.iri.clone());
                        self.ontology_constraints
                            .inverse_properties
                            .insert(triple.object.iri.clone(), triple.subject.iri.clone());
                    }
                    OntologyRelation::FunctionalProperty => {
                        self.ontology_constraints
                            .functional_properties
                            .insert(triple.subject.iri.clone());
                    }
                    OntologyRelation::SymmetricProperty => {
                        self.ontology_constraints
                            .symmetric_properties
                            .insert(triple.subject.iri.clone());
                    }
                    OntologyRelation::TransitiveProperty => {
                        self.ontology_constraints
                            .transitive_properties
                            .insert(triple.subject.iri.clone());
                    }
                }
            }
        }

        // Compute transitive closure if enabled
        if self.config.use_transitive_closure {
            self.ontology_constraints
                .compute_transitive_closure(self.config.max_transitive_depth);
        }
    }

    /// Compute hierarchy-preserving loss
    fn compute_hierarchy_loss(&self) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (subclass, superclasses) in &self.ontology_constraints.transitive_hierarchy {
            if let Some(sub_emb) = self.entity_embeddings.get(subclass) {
                for superclass in superclasses {
                    if let Some(super_emb) = self.entity_embeddings.get(superclass) {
                        // Subclass embedding should be "closer" to origin than superclass
                        // in the direction of the hierarchy
                        let sub_norm = sub_emb.dot(sub_emb).sqrt();
                        let super_norm = super_emb.dot(super_emb).sqrt();
                        let similarity = sub_emb.dot(super_emb) / (sub_norm * super_norm + 1e-8);

                        // Hierarchy loss: encourage high similarity and proper ordering
                        let hierarchy_score = similarity + (super_norm - sub_norm) * 0.1;
                        let loss = (self.config.hierarchy_margin - hierarchy_score).max(0.0);
                        total_loss += loss;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Compute equivalence loss
    fn compute_equivalence_loss(&self) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (class1, equivalent_classes) in &self.ontology_constraints.equivalent_classes {
            if let Some(emb1) = self.entity_embeddings.get(class1) {
                for class2 in equivalent_classes {
                    if let Some(emb2) = self.entity_embeddings.get(class2) {
                        // Equivalent classes should have very similar embeddings
                        let distance = (emb1 - emb2).mapv(|x| x * x).sum().sqrt();
                        total_loss += distance;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Compute disjointness loss
    fn compute_disjoint_loss(&self) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (class1, disjoint_classes) in &self.ontology_constraints.disjoint_classes {
            if let Some(emb1) = self.entity_embeddings.get(class1) {
                for class2 in disjoint_classes {
                    if let Some(emb2) = self.entity_embeddings.get(class2) {
                        // Disjoint classes should have low similarity
                        let norm1 = emb1.dot(emb1).sqrt();
                        let norm2 = emb2.dot(emb2).sqrt();
                        let similarity = emb1.dot(emb2) / (norm1 * norm2 + 1e-8);
                        let loss = (similarity + self.config.hierarchy_margin).max(0.0);
                        total_loss += loss;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Compute property constraint loss (domain/range constraints)
    fn compute_property_constraint_loss(&self) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        // Check domain constraints
        for (property, domains) in &self.ontology_constraints.property_domains {
            if let Some(relation_emb) = self.relation_embeddings.get(property) {
                for domain_class in domains {
                    if let Some(domain_emb) = self.entity_embeddings.get(domain_class) {
                        // Domain entities should be compatible with property
                        let compatibility = relation_emb.dot(domain_emb);
                        let loss = (1.0 - compatibility).max(0.0); // Encourage positive compatibility
                        total_loss += loss;
                        count += 1;
                    }
                }
            }
        }

        // Check range constraints
        for (property, ranges) in &self.ontology_constraints.property_ranges {
            if let Some(relation_emb) = self.relation_embeddings.get(property) {
                for range_class in ranges {
                    if let Some(range_emb) = self.entity_embeddings.get(range_class) {
                        // Range entities should be compatible with property
                        let compatibility = relation_emb.dot(range_emb);
                        let loss = (1.0 - compatibility).max(0.0);
                        total_loss += loss;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Compute contrastive learning loss for cross-modal alignment
    fn compute_contrastive_loss(&self) -> f32 {
        if !self.config.enable_contrastive_learning {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut count = 0;
        let temperature = self.config.contrastive_temperature;

        for (entity1, aligned_entities) in &self.ontology_constraints.cross_modal_alignments {
            if let Some(emb1) = self.entity_embeddings.get(entity1) {
                for entity2 in aligned_entities {
                    if let Some(emb2) = self.entity_embeddings.get(entity2) {
                        // Positive similarity
                        let pos_sim = emb1.dot(emb2) / temperature;

                        // Negative similarities (sample random entities)
                        let mut neg_sims = Vec::new();
                        for (neg_entity, neg_emb) in self.entity_embeddings.iter().take(10) {
                            if neg_entity != entity1 && neg_entity != entity2 {
                                let neg_sim = emb1.dot(neg_emb) / temperature;
                                neg_sims.push(neg_sim);
                            }
                        }

                        if !neg_sims.is_empty() {
                            // InfoNCE loss
                            let exp_pos = pos_sim.exp();
                            let sum_exp_neg: f32 = neg_sims.iter().copied().map(|x| x.exp()).sum();
                            let loss = -(exp_pos / (exp_pos + sum_exp_neg)).ln();
                            total_loss += loss;
                            count += 1;
                        }
                    }
                }
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Compute mutual information maximization loss
    fn compute_mutual_info_loss(&self) -> f32 {
        if !self.config.enable_mutual_info_max {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        // Simplified mutual information between entity and relation embeddings
        for (entity, entity_emb) in &self.entity_embeddings {
            for relation_emb in self.relation_embeddings.values() {
                // Check if this entity-relation pair appears in training data
                let pair_exists = self
                    .triples
                    .iter()
                    .any(|t| t.subject.iri == *entity || t.object.iri == *entity);

                if pair_exists {
                    // Encourage high mutual information for related pairs
                    let mi = entity_emb.dot(relation_emb);
                    let loss = (1.0 - mi).max(0.0);
                    total_loss += loss;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Compute property chain consistency loss
    fn compute_property_chain_loss(&self) -> f32 {
        if !self.config.enable_property_chains {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        for (target_property, chains) in &self.ontology_constraints.property_chains {
            if let Some(target_emb) = self.relation_embeddings.get(target_property) {
                for chain in chains {
                    if chain.len() == 2 {
                        // For 2-property chains: target ≈ prop1 + prop2
                        if let (Some(prop1_emb), Some(prop2_emb)) = (
                            self.relation_embeddings.get(&chain[0]),
                            self.relation_embeddings.get(&chain[1]),
                        ) {
                            let chain_emb = prop1_emb + prop2_emb;
                            let distance = (target_emb - &chain_emb).mapv(|x| x * x).sum().sqrt();
                            total_loss += distance;
                            count += 1;
                        }
                    }
                }
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }
}

#[async_trait]
impl EmbeddingModel for OntologyAwareEmbedding {
    fn config(&self) -> &ModelConfig {
        &self.config.base_config
    }

    fn model_id(&self) -> &Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "OntologyAware"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        self.triples.push(triple);
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let start_time = std::time::Instant::now();

        // Extract ontology constraints first
        self.extract_ontology_constraints();

        // Build property characteristics cache for enhanced constraint handling
        self.ontology_constraints
            .build_property_characteristics_cache();

        // Build entity and relation vocabularies
        let mut entity_set = HashSet::new();
        let mut relation_set = HashSet::new();

        for triple in &self.triples {
            entity_set.insert(triple.subject.iri.clone());
            entity_set.insert(triple.object.iri.clone());
            relation_set.insert(triple.predicate.iri.clone());
        }

        // Create mappings
        for (idx, entity) in entity_set.iter().enumerate() {
            self.entity_to_idx.insert(entity.clone(), idx);
        }

        for (idx, relation) in relation_set.iter().enumerate() {
            self.relation_to_idx.insert(relation.clone(), idx);
        }

        // Initialize embeddings
        let dimensions = self.config.base_config.dimensions;
        for entity in &entity_set {
            let embedding = Array1::from_vec(
                (0..dimensions)
                    .map(|_| {
                        use scirs2_core::random::{Random, Rng};
                        let mut random = Random::default();
                        (random.random::<f32>() - 0.5) * 0.1
                    })
                    .collect(),
            );
            self.entity_embeddings.insert(entity.clone(), embedding);
        }

        for relation in &relation_set {
            let embedding = Array1::from_vec(
                (0..dimensions)
                    .map(|_| {
                        use scirs2_core::random::{Random, Rng};
                        let mut random = Random::default();
                        (random.random::<f32>() - 0.5) * 0.1
                    })
                    .collect(),
            );
            self.relation_embeddings.insert(relation.clone(), embedding);
        }

        // Training loop with ontology constraints
        let max_epochs = epochs.unwrap_or(self.config.base_config.max_epochs);
        let learning_rate = self.config.base_config.learning_rate as f32;
        let mut loss_history = Vec::new();

        for epoch in 0..max_epochs {
            let mut total_loss = 0.0;

            // Standard TransE-style training
            for triple in &self.triples {
                if let (Some(h), Some(r), Some(t)) = (
                    self.entity_embeddings.get(&triple.subject.iri).cloned(),
                    self.relation_embeddings.get(&triple.predicate.iri).cloned(),
                    self.entity_embeddings.get(&triple.object.iri).cloned(),
                ) {
                    // Compute score: ||h + r - t||
                    let predicted = &h + &r;
                    let error = &t - &predicted;
                    let loss = error.dot(&error).sqrt();
                    total_loss += loss;

                    // Gradient updates
                    let gradient_scale = learning_rate * 0.01;
                    let h_grad = &error * gradient_scale;
                    let r_grad = &error * gradient_scale;
                    let t_grad = &error * (-gradient_scale);

                    // Update embeddings
                    if let Some(h_emb) = self.entity_embeddings.get_mut(&triple.subject.iri) {
                        *h_emb += &h_grad;
                    }
                    if let Some(r_emb) = self.relation_embeddings.get_mut(&triple.predicate.iri) {
                        *r_emb += &r_grad;
                    }
                    if let Some(t_emb) = self.entity_embeddings.get_mut(&triple.object.iri) {
                        *t_emb += &t_grad;
                    }
                }
            }

            // Add ontology constraint losses
            let hierarchy_loss = self.compute_hierarchy_loss();
            let equivalence_loss = self.compute_equivalence_loss();
            let disjoint_loss = self.compute_disjoint_loss();
            let property_loss = self.compute_property_constraint_loss();
            let contrastive_loss = self.compute_contrastive_loss();
            let mutual_info_loss = self.compute_mutual_info_loss();
            let property_chain_loss = self.compute_property_chain_loss();

            total_loss += hierarchy_loss * self.config.hierarchy_weight;
            total_loss += equivalence_loss * self.config.equivalence_weight;
            total_loss += disjoint_loss * self.config.disjoint_weight;
            total_loss += property_loss * self.config.property_constraint_weight;
            total_loss += contrastive_loss * self.config.cross_modal_weight;
            total_loss += mutual_info_loss * self.config.cross_modal_weight * 0.5;
            total_loss += property_chain_loss * self.config.property_constraint_weight * 0.8;

            loss_history.push(total_loss as f64);

            // Normalize embeddings if configured
            if self.config.normalize_for_hierarchy {
                for embedding in self.entity_embeddings.values_mut() {
                    let norm = embedding.dot(embedding).sqrt();
                    if norm > 0.0 {
                        *embedding /= norm;
                    }
                }
            }

            if epoch % 10 == 0 {
                tracing::info!(
                    "Epoch {}: total_loss={:.6}, hierarchy={:.6}, equiv={:.6}, disjoint={:.6}",
                    epoch,
                    total_loss,
                    hierarchy_loss,
                    equivalence_loss,
                    disjoint_loss
                );
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();
        self.is_trained = true;

        // Update model stats
        self.model_stats.num_entities = entity_set.len();
        self.model_stats.num_relations = relation_set.len();
        self.model_stats.num_triples = self.triples.len();
        self.model_stats.is_trained = true;
        self.model_stats.last_training_time = Some(Utc::now());

        // Update training stats
        self.training_stats = TrainingStats {
            epochs_completed: max_epochs,
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().copied().unwrap_or(0.0) < 0.01,
            loss_history,
        };

        Ok(self.training_stats.clone())
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        self.entity_embeddings
            .get(entity)
            .map(|arr| Vector::new(arr.to_vec()))
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        self.relation_embeddings
            .get(relation)
            .map(|arr| Vector::new(arr.to_vec()))
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let h = self
            .entity_embeddings
            .get(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let r = self
            .relation_embeddings
            .get(predicate)
            .ok_or_else(|| anyhow!("Predicate not found: {}", predicate))?;
        let t = self
            .entity_embeddings
            .get(object)
            .ok_or_else(|| anyhow!("Object not found: {}", object))?;

        // TransE scoring: ||h + r - t||
        let predicted = h + r;
        let distance = (&predicted - t).mapv(|x| x * x).sum().sqrt();
        Ok(-(distance as f64)) // Negative distance as higher scores are better
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let h = self
            .entity_embeddings
            .get(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let r = self
            .relation_embeddings
            .get(predicate)
            .ok_or_else(|| anyhow!("Predicate not found: {}", predicate))?;
        let predicted = h + r;

        let mut scores = Vec::new();
        for (entity, embedding) in &self.entity_embeddings {
            let distance = (&predicted - embedding).mapv(|x| x * x).sum().sqrt();
            scores.push((entity.clone(), -(distance as f64)));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        Ok(scores)
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let r = self
            .relation_embeddings
            .get(predicate)
            .ok_or_else(|| anyhow!("Predicate not found: {}", predicate))?;
        let t = self
            .entity_embeddings
            .get(object)
            .ok_or_else(|| anyhow!("Object not found: {}", object))?;
        let target = t - r; // h = t - r

        let mut scores = Vec::new();
        for (entity, embedding) in &self.entity_embeddings {
            let distance = (embedding - &target).mapv(|x| x * x).sum().sqrt();
            scores.push((entity.clone(), -(distance as f64)));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        Ok(scores)
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let h = self
            .entity_embeddings
            .get(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let t = self
            .entity_embeddings
            .get(object)
            .ok_or_else(|| anyhow!("Object not found: {}", object))?;
        let target = t - h; // r = t - h

        let mut scores = Vec::new();
        for (relation, embedding) in &self.relation_embeddings {
            let distance = (embedding - &target).mapv(|x| x * x).sum().sqrt();
            scores.push((relation.clone(), -(distance as f64)));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        Ok(scores)
    }

    fn get_entities(&self) -> Vec<String> {
        self.entity_embeddings.keys().cloned().collect()
    }

    fn get_relations(&self) -> Vec<String> {
        self.relation_embeddings.keys().cloned().collect()
    }

    fn get_stats(&self) -> ModelStats {
        self.model_stats.clone()
    }

    fn save(&self, path: &str) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let loaded: OntologyAwareEmbedding = serde_json::from_str(&content)?;
        *self = loaded;
        Ok(())
    }

    fn clear(&mut self) {
        self.entity_embeddings.clear();
        self.relation_embeddings.clear();
        self.entity_to_idx.clear();
        self.relation_to_idx.clear();
        self.triples.clear();
        self.ontology_constraints = OntologyConstraints::default();
        self.training_stats = TrainingStats::default();
        self.is_trained = false;
        self.model_stats.is_trained = false;
        self.model_stats.num_entities = 0;
        self.model_stats.num_relations = 0;
        self.model_stats.num_triples = 0;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn encode(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(anyhow!(
            "Knowledge graph embedding model does not support text encoding"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;

    #[test]
    fn test_ontology_relation_from_iri() {
        assert_eq!(
            OntologyRelation::from_iri("http://www.w3.org/2000/01/rdf-schema#subClassOf"),
            Some(OntologyRelation::SubClassOf)
        );
        assert_eq!(
            OntologyRelation::from_iri("http://www.w3.org/2002/07/owl#equivalentClass"),
            Some(OntologyRelation::EquivalentClass)
        );
        assert_eq!(
            OntologyRelation::from_iri("http://example.org/custom"),
            None
        );
    }

    #[test]
    fn test_ontology_constraint_extraction() {
        let config = OntologyAwareEmbedding::hierarchy_optimized_config(50);
        let mut model = OntologyAwareEmbedding::new(config);

        // Create test triples with ontology relationships
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/Dog").unwrap(),
                NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf").unwrap(),
                NamedNode::new("http://example.org/Animal").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/Cat").unwrap(),
                NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf").unwrap(),
                NamedNode::new("http://example.org/Animal").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/Canine").unwrap(),
                NamedNode::new("http://www.w3.org/2002/07/owl#equivalentClass").unwrap(),
                NamedNode::new("http://example.org/Dog").unwrap(),
            ),
        ];

        model.triples = triples;
        model.extract_ontology_constraints();

        // Check class hierarchy extraction
        assert!(model
            .ontology_constraints
            .class_hierarchy
            .contains_key("http://example.org/Dog"));
        assert!(model
            .ontology_constraints
            .class_hierarchy
            .contains_key("http://example.org/Cat"));

        // Check equivalent classes
        assert!(model
            .ontology_constraints
            .equivalent_classes
            .contains_key("http://example.org/Canine"));
        assert!(model
            .ontology_constraints
            .equivalent_classes
            .contains_key("http://example.org/Dog"));
    }

    #[test]
    fn test_transitive_closure_computation() {
        let mut constraints = OntologyConstraints::default();

        // A -> B -> C hierarchy
        constraints.class_hierarchy.insert("A".to_string(), {
            let mut set = HashSet::new();
            set.insert("B".to_string());
            set
        });
        constraints.class_hierarchy.insert("B".to_string(), {
            let mut set = HashSet::new();
            set.insert("C".to_string());
            set
        });

        constraints.compute_transitive_closure(5);

        // A should be a subclass of both B and C
        assert!(constraints.is_subclass_of("A", "B"));
        assert!(constraints.is_subclass_of("A", "C"));
        assert!(constraints.is_subclass_of("B", "C"));
        assert!(!constraints.is_subclass_of("C", "A"));
    }

    #[test]
    fn test_ontology_aware_config_factory_methods() {
        let hierarchy_config = OntologyAwareEmbedding::hierarchy_optimized_config(100);
        assert_eq!(hierarchy_config.base_config.dimensions, 100);
        assert_eq!(hierarchy_config.hierarchy_weight, 2.0);
        assert!(hierarchy_config.use_transitive_closure);

        let property_config = OntologyAwareEmbedding::property_optimized_config(100);
        assert_eq!(property_config.disjoint_weight, 2.0);
        assert_eq!(property_config.max_transitive_depth, 8);
    }

    #[tokio::test]
    async fn test_ontology_aware_embedding_training() {
        let config = OntologyAwareEmbedding::hierarchy_optimized_config(32);
        let mut model = OntologyAwareEmbedding::new(config);

        // Add triples using the trait method
        model
            .add_triple(Triple::new(
                NamedNode::new("http://example.org/Dog").unwrap(),
                NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf").unwrap(),
                NamedNode::new("http://example.org/Animal").unwrap(),
            ))
            .unwrap();

        model
            .add_triple(Triple::new(
                NamedNode::new("http://example.org/Fido").unwrap(),
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
                NamedNode::new("http://example.org/Dog").unwrap(),
            ))
            .unwrap();

        let result = model.train(Some(10)).await;
        assert!(result.is_ok());

        // Check that embeddings were created
        assert!(model
            .entity_embeddings
            .contains_key("http://example.org/Dog"));
        assert!(model
            .entity_embeddings
            .contains_key("http://example.org/Animal"));
        assert!(model
            .entity_embeddings
            .contains_key("http://example.org/Fido"));

        // Test embedding retrieval using the trait method
        let dog_embedding = model.get_entity_embedding("http://example.org/Dog");
        assert!(dog_embedding.is_ok());
        assert_eq!(dog_embedding.unwrap().dimensions, 32);

        // Test that model is trained
        assert!(model.is_trained());
    }
}
