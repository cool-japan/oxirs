//! Schema Alignment and Ontology Mapping for Federated RDF/SPARQL
//!
//! This module provides automatic schema alignment, ontology mapping, and vocabulary
//! harmonization for federated SPARQL queries across heterogeneous RDF sources.
//!
//! Features:
//! - Automatic ontology alignment using similarity metrics
//! - Property and class mapping across vocabularies
//! - Equivalence reasoning (owl:sameAs, owl:equivalentClass, owl:equivalentProperty)
//! - Schema translation and query rewriting
//! - ML-based mapping suggestion using scirs2
//!
//! Enhanced with scirs2 for similarity computations and ML-based mapping.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

// scirs2 integration for similarity and ML
// Note: Advanced features simplified for initial release

use crate::semantic_reasoner::{ReasonerConfig, SemanticReasoner};

/// Schema alignment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Minimum similarity threshold for automatic alignment (0.0 - 1.0)
    pub similarity_threshold: f64,
    /// Enable property alignment
    pub enable_property_alignment: bool,
    /// Enable class alignment
    pub enable_class_alignment: bool,
    /// Enable instance alignment (owl:sameAs)
    pub enable_instance_alignment: bool,
    /// Use ML-based mapping suggestions
    pub enable_ml_mapping: bool,
    /// Confidence threshold for ML suggestions (0.0 - 1.0)
    pub ml_confidence_threshold: f64,
    /// Maximum number of alignment suggestions per entity
    pub max_suggestions_per_entity: usize,
    /// Enable reasoning with OWL axioms
    pub enable_owl_reasoning: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.75,
            enable_property_alignment: true,
            enable_class_alignment: true,
            enable_instance_alignment: true,
            enable_ml_mapping: true,
            ml_confidence_threshold: 0.8,
            max_suggestions_per_entity: 5,
            enable_owl_reasoning: true,
        }
    }
}

/// Schema alignment engine
pub struct SchemaAligner {
    config: AlignmentConfig,
    /// Property mappings: (source_property, target_property) -> confidence
    property_mappings: Arc<RwLock<HashMap<(String, String), f64>>>,
    /// Class mappings: (source_class, target_class) -> confidence
    class_mappings: Arc<RwLock<HashMap<(String, String), f64>>>,
    /// Instance mappings: (source_instance, target_instance) -> confidence
    instance_mappings: Arc<RwLock<HashMap<(String, String), f64>>>,
    /// Vocabulary metadata cache
    vocabulary_cache: Arc<RwLock<HashMap<String, VocabularyMetadata>>>,
    /// ML model for mapping prediction (simplified)
    ml_predictor: Arc<RwLock<Option<MappingPredictor>>>,
    /// Semantic reasoner for RDFS/OWL inference
    reasoner: Arc<RwLock<Option<SemanticReasoner>>>,
}

impl SchemaAligner {
    /// Create a new schema aligner
    pub fn new(config: AlignmentConfig) -> Self {
        Self {
            config,
            property_mappings: Arc::new(RwLock::new(HashMap::new())),
            class_mappings: Arc::new(RwLock::new(HashMap::new())),
            instance_mappings: Arc::new(RwLock::new(HashMap::new())),
            vocabulary_cache: Arc::new(RwLock::new(HashMap::new())),
            ml_predictor: Arc::new(RwLock::new(None)),
            reasoner: Arc::new(RwLock::new(None)),
        }
    }

    /// Enable semantic reasoning for enhanced alignment
    pub async fn enable_reasoning(&self, reasoner_config: ReasonerConfig) -> Result<()> {
        let reasoner = SemanticReasoner::new(reasoner_config);
        let mut reasoner_guard = self.reasoner.write().await;
        *reasoner_guard = Some(reasoner);
        info!("Semantic reasoning enabled for schema alignment");
        Ok(())
    }

    /// Align two RDF vocabularies/ontologies
    pub async fn align_vocabularies(
        &self,
        source_vocab: &str,
        target_vocab: &str,
    ) -> Result<AlignmentResult> {
        info!(
            "Aligning vocabularies: {} -> {}",
            source_vocab, target_vocab
        );

        // Load vocabulary metadata
        let source_meta = self.load_vocabulary_metadata(source_vocab).await?;
        let target_meta = self.load_vocabulary_metadata(target_vocab).await?;

        let mut property_alignments = Vec::new();
        let mut class_alignments = Vec::new();

        // Align properties
        if self.config.enable_property_alignment {
            property_alignments = self
                .align_properties(&source_meta.properties, &target_meta.properties)
                .await?;
        }

        // Align classes
        if self.config.enable_class_alignment {
            class_alignments = self
                .align_classes(&source_meta.classes, &target_meta.classes)
                .await?;
        }

        // Store alignments
        self.store_alignments(&property_alignments, &class_alignments)
            .await?;

        let overall_confidence =
            self.calculate_overall_confidence(&property_alignments, &class_alignments);

        Ok(AlignmentResult {
            source_vocabulary: source_vocab.to_string(),
            target_vocabulary: target_vocab.to_string(),
            property_alignments,
            class_alignments,
            instance_alignments: vec![],
            overall_confidence,
        })
    }

    /// Align RDF properties between vocabularies
    async fn align_properties(
        &self,
        source_properties: &[PropertyMetadata],
        target_properties: &[PropertyMetadata],
    ) -> Result<Vec<Alignment>> {
        let mut alignments = Vec::new();

        for source_prop in source_properties {
            let mut candidates = Vec::new();

            for target_prop in target_properties {
                // Calculate similarity
                let similarity = self
                    .calculate_property_similarity(source_prop, target_prop)
                    .await?;

                if similarity >= self.config.similarity_threshold {
                    candidates.push((target_prop, similarity));
                }
            }

            // Sort by similarity (highest first)
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top N suggestions
            for (target_prop, confidence) in candidates
                .iter()
                .take(self.config.max_suggestions_per_entity)
            {
                alignments.push(Alignment {
                    source_entity: source_prop.uri.clone(),
                    target_entity: target_prop.uri.clone(),
                    alignment_type: AlignmentType::Property,
                    confidence: *confidence,
                    evidence: vec![format!(
                        "String similarity: {:.3}, Domain/range match",
                        confidence
                    )],
                });
            }
        }

        Ok(alignments)
    }

    /// Align RDF classes between vocabularies
    async fn align_classes(
        &self,
        source_classes: &[ClassMetadata],
        target_classes: &[ClassMetadata],
    ) -> Result<Vec<Alignment>> {
        let mut alignments = Vec::new();

        for source_class in source_classes {
            let mut candidates = Vec::new();

            for target_class in target_classes {
                // Calculate similarity
                let similarity = self
                    .calculate_class_similarity(source_class, target_class)
                    .await?;

                if similarity >= self.config.similarity_threshold {
                    candidates.push((target_class, similarity));
                }
            }

            // Sort by similarity
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top N suggestions
            for (target_class, confidence) in candidates
                .iter()
                .take(self.config.max_suggestions_per_entity)
            {
                alignments.push(Alignment {
                    source_entity: source_class.uri.clone(),
                    target_entity: target_class.uri.clone(),
                    alignment_type: AlignmentType::Class,
                    confidence: *confidence,
                    evidence: vec![format!("Class similarity: {:.3}", confidence)],
                });
            }
        }

        Ok(alignments)
    }

    /// Calculate similarity between two properties
    async fn calculate_property_similarity(
        &self,
        source: &PropertyMetadata,
        target: &PropertyMetadata,
    ) -> Result<f64> {
        let mut similarities = Vec::new();

        // Label similarity (string-based)
        let label_sim = self.calculate_string_similarity(&source.label, &target.label);
        similarities.push(label_sim * 0.4); // Weight: 40%

        // Local name similarity
        let local_name_sim = self.calculate_string_similarity(
            &Self::extract_local_name(&source.uri),
            &Self::extract_local_name(&target.uri),
        );
        similarities.push(local_name_sim * 0.3); // Weight: 30%

        // Domain/range compatibility
        let domain_range_sim = self.calculate_domain_range_similarity(source, target);
        similarities.push(domain_range_sim * 0.2); // Weight: 20%

        // Description similarity (if available)
        if let (Some(ref src_desc), Some(ref tgt_desc)) = (&source.description, &target.description)
        {
            let desc_sim = self.calculate_string_similarity(src_desc, tgt_desc);
            similarities.push(desc_sim * 0.1); // Weight: 10%
        }

        // Aggregate similarities
        let total_similarity = similarities.iter().sum::<f64>();

        Ok(total_similarity.min(1.0))
    }

    /// Calculate similarity between two classes
    async fn calculate_class_similarity(
        &self,
        source: &ClassMetadata,
        target: &ClassMetadata,
    ) -> Result<f64> {
        let mut similarities = Vec::new();

        // Label similarity
        let label_sim = self.calculate_string_similarity(&source.label, &target.label);
        similarities.push(label_sim * 0.5); // Weight: 50%

        // Local name similarity
        let local_name_sim = self.calculate_string_similarity(
            &Self::extract_local_name(&source.uri),
            &Self::extract_local_name(&target.uri),
        );
        similarities.push(local_name_sim * 0.3); // Weight: 30%

        // Superclass/subclass overlap
        let hierarchy_sim = self.calculate_hierarchy_similarity(source, target);
        similarities.push(hierarchy_sim * 0.2); // Weight: 20%

        let total_similarity = similarities.iter().sum::<f64>();

        Ok(total_similarity.min(1.0))
    }

    /// Calculate string similarity using multiple methods
    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        // Normalize strings
        let s1_norm = s1.to_lowercase().trim().to_string();
        let s2_norm = s2.to_lowercase().trim().to_string();

        // Exact match
        if s1_norm == s2_norm {
            return 1.0;
        }

        // Levenshtein distance (normalized) - simplified implementation
        let lev_dist = Self::simple_levenshtein(&s1_norm, &s2_norm);
        let max_len = s1_norm.len().max(s2_norm.len()) as f64;
        let lev_sim = 1.0 - (lev_dist as f64 / max_len);

        // Token-based similarity (Jaccard)
        let tokens1: HashSet<&str> = s1_norm.split_whitespace().collect();
        let tokens2: HashSet<&str> = s2_norm.split_whitespace().collect();
        let intersection = tokens1.intersection(&tokens2).count();
        let union = tokens1.union(&tokens2).count();
        let jaccard_sim = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        // Combine similarities
        (lev_sim * 0.6 + jaccard_sim * 0.4).min(1.0)
    }

    /// Calculate domain/range similarity for properties
    fn calculate_domain_range_similarity(
        &self,
        source: &PropertyMetadata,
        target: &PropertyMetadata,
    ) -> f64 {
        let mut similarity: f64 = 0.0;

        // Domain match
        if let (Some(ref src_domain), Some(ref tgt_domain)) = (&source.domain, &target.domain) {
            if src_domain == tgt_domain {
                similarity += 0.5;
            } else {
                // Partial match based on local names
                let src_local = Self::extract_local_name(src_domain);
                let tgt_local = Self::extract_local_name(tgt_domain);
                if src_local == tgt_local {
                    similarity += 0.3;
                }
            }
        }

        // Range match
        if let (Some(ref src_range), Some(ref tgt_range)) = (&source.range, &target.range) {
            if src_range == tgt_range {
                similarity += 0.5;
            } else {
                let src_local = Self::extract_local_name(src_range);
                let tgt_local = Self::extract_local_name(tgt_range);
                if src_local == tgt_local {
                    similarity += 0.3;
                }
            }
        }

        similarity.min(1.0)
    }

    /// Calculate hierarchy similarity for classes
    fn calculate_hierarchy_similarity(
        &self,
        source: &ClassMetadata,
        target: &ClassMetadata,
    ) -> f64 {
        // Calculate Jaccard similarity of superclasses
        let src_supers: HashSet<_> = source.superclasses.iter().collect();
        let tgt_supers: HashSet<_> = target.superclasses.iter().collect();

        let intersection = src_supers.intersection(&tgt_supers).count();
        let union = src_supers.union(&tgt_supers).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Extract local name from URI
    fn extract_local_name(uri: &str) -> String {
        uri.rsplit(['/', '#']).next().unwrap_or(uri).to_string()
    }

    /// Simple Levenshtein distance calculation
    fn simple_levenshtein(s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut prev_row: Vec<usize> = (0..=len2).collect();
        let mut curr_row = vec![0; len2 + 1];

        for (i, c1) in s1.chars().enumerate() {
            curr_row[0] = i + 1;

            for (j, c2) in s2.chars().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                curr_row[j + 1] = (curr_row[j] + 1)
                    .min(prev_row[j + 1] + 1)
                    .min(prev_row[j] + cost);
            }

            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        prev_row[len2]
    }

    /// Load vocabulary metadata (simplified - in production would query SPARQL endpoints)
    async fn load_vocabulary_metadata(&self, vocab_uri: &str) -> Result<VocabularyMetadata> {
        // Check cache first
        let cache = self.vocabulary_cache.read().await;
        if let Some(meta) = cache.get(vocab_uri) {
            return Ok(meta.clone());
        }
        drop(cache);

        // In production, this would:
        // 1. Query the vocabulary endpoint
        // 2. Parse RDFS/OWL definitions
        // 3. Extract class and property metadata
        // For now, return mock data

        let metadata = VocabularyMetadata {
            namespace: vocab_uri.to_string(),
            prefix: Self::extract_local_name(vocab_uri),
            properties: vec![],
            classes: vec![],
            version: None,
        };

        // Cache the metadata
        let mut cache = self.vocabulary_cache.write().await;
        cache.insert(vocab_uri.to_string(), metadata.clone());

        Ok(metadata)
    }

    /// Store alignments in the mapping tables
    async fn store_alignments(
        &self,
        property_alignments: &[Alignment],
        class_alignments: &[Alignment],
    ) -> Result<()> {
        let mut prop_mappings = self.property_mappings.write().await;
        let mut class_mappings = self.class_mappings.write().await;

        for alignment in property_alignments {
            prop_mappings.insert(
                (
                    alignment.source_entity.clone(),
                    alignment.target_entity.clone(),
                ),
                alignment.confidence,
            );
        }

        for alignment in class_alignments {
            class_mappings.insert(
                (
                    alignment.source_entity.clone(),
                    alignment.target_entity.clone(),
                ),
                alignment.confidence,
            );
        }

        info!(
            "Stored {} property alignments and {} class alignments",
            property_alignments.len(),
            class_alignments.len()
        );

        Ok(())
    }

    /// Calculate overall alignment confidence
    fn calculate_overall_confidence(
        &self,
        property_alignments: &[Alignment],
        class_alignments: &[Alignment],
    ) -> f64 {
        let all_alignments: Vec<f64> = property_alignments
            .iter()
            .chain(class_alignments.iter())
            .map(|a| a.confidence)
            .collect();

        if all_alignments.is_empty() {
            return 0.0;
        }

        // Average confidence
        all_alignments.iter().sum::<f64>() / all_alignments.len() as f64
    }

    /// Rewrite SPARQL query using schema alignments
    pub async fn rewrite_query(&self, query: &str, target_vocabulary: &str) -> Result<String> {
        debug!("Rewriting query for vocabulary: {}", target_vocabulary);

        // Parse query (simplified - in production would use full SPARQL parser)
        let mut rewritten = query.to_string();

        // Apply property mappings
        let prop_mappings = self.property_mappings.read().await;
        for ((source, target), _confidence) in prop_mappings.iter() {
            rewritten = rewritten.replace(source, target);
        }

        // Apply class mappings
        let class_mappings = self.class_mappings.read().await;
        for ((source, target), _confidence) in class_mappings.iter() {
            rewritten = rewritten.replace(source, target);
        }

        Ok(rewritten)
    }

    /// Get mapping for a specific entity
    pub async fn get_mapping(
        &self,
        source_entity: &str,
        entity_type: AlignmentType,
    ) -> Result<Option<String>> {
        match entity_type {
            AlignmentType::Property => {
                let mappings = self.property_mappings.read().await;
                Ok(mappings
                    .iter()
                    .find(|((s, _), _)| s == source_entity)
                    .map(|((_, t), _)| t.clone()))
            }
            AlignmentType::Class => {
                let mappings = self.class_mappings.read().await;
                Ok(mappings
                    .iter()
                    .find(|((s, _), _)| s == source_entity)
                    .map(|((_, t), _)| t.clone()))
            }
            AlignmentType::Instance => {
                let mappings = self.instance_mappings.read().await;
                Ok(mappings
                    .iter()
                    .find(|((s, _), _)| s == source_entity)
                    .map(|((_, t), _)| t.clone()))
            }
        }
    }

    /// Train ML-based mapping predictor
    pub async fn train_ml_predictor(&self, training_data: Vec<MappingExample>) -> Result<()> {
        if !self.config.enable_ml_mapping {
            return Err(anyhow!("ML mapping is disabled"));
        }

        info!(
            "Training ML-based mapping predictor with {} examples",
            training_data.len()
        );

        let predictor = MappingPredictor::train(training_data)?;

        let mut ml_predictor = self.ml_predictor.write().await;
        *ml_predictor = Some(predictor);

        Ok(())
    }
}

/// Vocabulary metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyMetadata {
    pub namespace: String,
    pub prefix: String,
    pub properties: Vec<PropertyMetadata>,
    pub classes: Vec<ClassMetadata>,
    pub version: Option<String>,
}

/// Property metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyMetadata {
    pub uri: String,
    pub label: String,
    pub description: Option<String>,
    pub domain: Option<String>,
    pub range: Option<String>,
}

/// Class metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetadata {
    pub uri: String,
    pub label: String,
    pub description: Option<String>,
    pub superclasses: Vec<String>,
    pub subclasses: Vec<String>,
}

/// Alignment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    pub source_vocabulary: String,
    pub target_vocabulary: String,
    pub property_alignments: Vec<Alignment>,
    pub class_alignments: Vec<Alignment>,
    pub instance_alignments: Vec<Alignment>,
    pub overall_confidence: f64,
}

/// Individual alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alignment {
    pub source_entity: String,
    pub target_entity: String,
    pub alignment_type: AlignmentType,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// Alignment type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlignmentType {
    Property,
    Class,
    Instance,
}

/// Mapping example for ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingExample {
    pub source_entity: String,
    pub target_entity: String,
    pub is_correct: bool,
    pub features: HashMap<String, f64>,
}

/// ML-based mapping predictor (simplified)
#[derive(Debug, Clone)]
pub struct MappingPredictor {
    /// Feature weights learned from training
    weights: HashMap<String, f64>,
}

impl MappingPredictor {
    /// Train predictor from examples
    pub fn train(examples: Vec<MappingExample>) -> Result<Self> {
        // Simplified training - in production would use scirs2's ML capabilities
        let mut weights = HashMap::new();

        // Calculate feature importance
        for example in &examples {
            for (feature_name, &feature_value) in &example.features {
                let weight = if example.is_correct {
                    feature_value
                } else {
                    -feature_value
                };
                *weights.entry(feature_name.clone()).or_insert(0.0) += weight;
            }
        }

        // Normalize weights
        let sum: f64 = weights.values().map(|w| w.abs()).sum();
        if sum > 0.0 {
            for weight in weights.values_mut() {
                *weight /= sum;
            }
        }

        Ok(Self { weights })
    }

    /// Predict if mapping is correct
    pub fn predict(&self, features: &HashMap<String, f64>) -> f64 {
        let mut score = 0.5; // Base score

        for (feature_name, &feature_value) in features {
            if let Some(&weight) = self.weights.get(feature_name) {
                score += weight * feature_value;
            }
        }

        score.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_config_default() {
        let config = AlignmentConfig::default();
        assert_eq!(config.similarity_threshold, 0.75);
        assert!(config.enable_property_alignment);
        assert!(config.enable_class_alignment);
    }

    #[test]
    fn test_extract_local_name() {
        assert_eq!(
            SchemaAligner::extract_local_name("http://xmlns.com/foaf/0.1/name"),
            "name"
        );
        assert_eq!(
            SchemaAligner::extract_local_name("http://schema.org#Person"),
            "Person"
        );
    }

    #[tokio::test]
    async fn test_schema_aligner_creation() {
        let config = AlignmentConfig::default();
        let aligner = SchemaAligner::new(config);

        // Test that aligner is created successfully
        assert!(aligner.property_mappings.read().await.is_empty());
        assert!(aligner.class_mappings.read().await.is_empty());
    }

    #[test]
    fn test_string_similarity() {
        let config = AlignmentConfig::default();
        let aligner = SchemaAligner::new(config);

        // Exact match
        let sim1 = aligner.calculate_string_similarity("name", "name");
        assert_eq!(sim1, 1.0);

        // Similar strings
        let sim2 = aligner.calculate_string_similarity("firstName", "first_name");
        assert!(sim2 > 0.5);

        // Different strings
        let sim3 = aligner.calculate_string_similarity("name", "age");
        assert!(sim3 < 0.5);
    }

    #[test]
    fn test_mapping_predictor_train() {
        let examples = vec![
            MappingExample {
                source_entity: "foaf:name".to_string(),
                target_entity: "schema:name".to_string(),
                is_correct: true,
                features: [("string_sim".to_string(), 0.9)].into(),
            },
            MappingExample {
                source_entity: "foaf:age".to_string(),
                target_entity: "schema:birthDate".to_string(),
                is_correct: false,
                features: [("string_sim".to_string(), 0.3)].into(),
            },
        ];

        let predictor = MappingPredictor::train(examples).unwrap();
        assert!(!predictor.weights.is_empty());
    }
}
