//! Cross-domain transfer learning for embedding models
//!
//! This module implements comprehensive cross-domain transfer evaluation and adaptation
//! techniques for knowledge graph embeddings across different domains and datasets.

use crate::{EmbeddingModel, Vector};
use anyhow::{anyhow, Result};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Cross-domain transfer manager
pub struct CrossDomainTransferManager {
    /// Source domain models and their metadata
    source_domains: HashMap<String, DomainModel>,
    /// Target domain specifications
    target_domains: HashMap<String, DomainSpecification>,
    /// Transfer learning strategies
    transfer_strategies: Vec<TransferStrategy>,
    /// Evaluation metrics for transfer quality
    transfer_metrics: Vec<TransferMetric>,
    /// Configuration
    config: TransferConfig,
}

/// Configuration for cross-domain transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferConfig {
    /// Enable domain adaptation techniques
    pub enable_domain_adaptation: bool,
    /// Use adversarial domain alignment
    pub use_adversarial_alignment: bool,
    /// Maximum number of alignment iterations
    pub max_alignment_iterations: usize,
    /// Learning rate for domain adaptation
    pub adaptation_learning_rate: f64,
    /// Minimum domain similarity threshold
    pub min_domain_similarity: f64,
    /// Enable cross-domain entity linking
    pub enable_entity_linking: bool,
    /// Transfer evaluation sample size
    pub evaluation_sample_size: usize,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            enable_domain_adaptation: true,
            use_adversarial_alignment: true,
            max_alignment_iterations: 100,
            adaptation_learning_rate: 0.001,
            min_domain_similarity: 0.3,
            enable_entity_linking: true,
            evaluation_sample_size: 1000,
        }
    }
}

/// Domain model with embeddings and metadata
pub struct DomainModel {
    /// Domain identifier
    pub domain_id: String,
    /// Embedding model
    pub model: Box<dyn EmbeddingModel + Send + Sync>,
    /// Domain characteristics
    pub characteristics: DomainCharacteristics,
    /// Entity mappings
    pub entity_mappings: HashMap<String, String>,
    /// Domain-specific vocabulary
    pub vocabulary: HashSet<String>,
}

/// Domain characteristics for transfer analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCharacteristics {
    /// Domain type (e.g., "biomedical", "financial", "general")
    pub domain_type: String,
    /// Language of the domain
    pub language: String,
    /// Entity types present in the domain
    pub entity_types: Vec<String>,
    /// Relation types present in the domain
    pub relation_types: Vec<String>,
    /// Domain size metrics
    pub size_metrics: DomainSizeMetrics,
    /// Domain complexity indicators
    pub complexity_metrics: DomainComplexityMetrics,
}

/// Domain size metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSizeMetrics {
    /// Number of entities
    pub num_entities: usize,
    /// Number of relations
    pub num_relations: usize,
    /// Number of triples
    pub num_triples: usize,
    /// Average entity degree
    pub avg_entity_degree: f64,
    /// Graph density
    pub graph_density: f64,
}

/// Domain complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainComplexityMetrics {
    /// Number of distinct entity types
    pub entity_type_diversity: usize,
    /// Number of distinct relation types
    pub relation_type_diversity: usize,
    /// Hierarchical depth
    pub hierarchical_depth: usize,
    /// Semantic diversity score
    pub semantic_diversity: f64,
    /// Structural complexity score
    pub structural_complexity: f64,
}

/// Target domain specification for transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSpecification {
    /// Target domain identifier
    pub domain_id: String,
    /// Domain characteristics
    pub characteristics: DomainCharacteristics,
    /// Available training data
    pub training_data: Vec<(String, String, String)>,
    /// Validation data
    pub validation_data: Vec<(String, String, String)>,
    /// Test data
    pub test_data: Vec<(String, String, String)>,
}

/// Transfer learning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferStrategy {
    /// Direct transfer without adaptation
    DirectTransfer,
    /// Fine-tuning on target domain
    FineTuning {
        learning_rate: f64,
        epochs: usize,
        freeze_layers: Vec<String>,
    },
    /// Domain adaptation with alignment
    DomainAdaptation {
        alignment_method: AlignmentMethod,
        regularization_strength: f64,
    },
    /// Multi-task learning
    MultiTaskLearning { task_weights: HashMap<String, f64> },
    /// Meta-learning approach
    MetaLearning {
        inner_steps: usize,
        meta_learning_rate: f64,
    },
    /// Progressive transfer
    ProgressiveTransfer {
        intermediate_domains: Vec<String>,
        progression_strategy: ProgressionStrategy,
    },
}

/// Domain alignment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentMethod {
    /// Linear transformation alignment
    LinearAlignment,
    /// Non-linear neural alignment
    NeuralAlignment,
    /// Adversarial domain alignment
    AdversarialAlignment,
    /// Canonical correlation analysis
    CCA,
    /// Procrustes alignment
    ProcrustesAlignment,
    /// Wasserstein distance minimization
    WassersteinAlignment,
}

/// Progression strategies for multi-step transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressionStrategy {
    /// Sequential domain progression
    Sequential,
    /// Curriculum learning based progression
    CurriculumBased,
    /// Similarity-guided progression
    SimilarityGuided,
}

/// Transfer evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferMetric {
    /// Transfer accuracy compared to source
    TransferAccuracy,
    /// Domain adaptation quality
    AdaptationQuality,
    /// Cross-domain entity alignment quality
    EntityAlignmentQuality,
    /// Semantic preservation score
    SemanticPreservation,
    /// Structural preservation score
    StructuralPreservation,
    /// Transfer efficiency (performance vs. effort)
    TransferEfficiency,
    /// Catastrophic forgetting measure
    CatastrophicForgetting,
    /// Cross-domain coherence evaluation
    CrossDomainCoherence,
    /// Knowledge retention across domains
    KnowledgeRetention,
    /// Adaptation speed metric
    AdaptationSpeed,
    /// Transfer robustness across domain variations
    TransferRobustness,
    /// Semantic drift detection
    SemanticDriftDetection,
    /// Cross-domain generalization ability
    GeneralizationAbility,
}

/// Transfer evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEvaluationResults {
    /// Source domain identifier
    pub source_domain: String,
    /// Target domain identifier
    pub target_domain: String,
    /// Transfer strategy used
    pub strategy: TransferStrategy,
    /// Evaluation metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Overall transfer quality score
    pub overall_quality: f64,
    /// Domain similarity score
    pub domain_similarity: f64,
    /// Performance improvement over baseline
    pub improvement_over_baseline: f64,
    /// Transfer time (seconds)
    pub transfer_time: f64,
    /// Detailed analysis
    pub detailed_analysis: TransferAnalysis,
}

/// Detailed transfer analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferAnalysis {
    /// Entity alignment results
    pub entity_alignments: Vec<EntityAlignment>,
    /// Relation alignment results
    pub relation_alignments: Vec<RelationAlignment>,
    /// Semantic shift analysis
    pub semantic_shifts: Vec<SemanticShift>,
    /// Structural changes
    pub structural_changes: StructuralChanges,
    /// Transfer recommendations
    pub recommendations: Vec<String>,
}

/// Entity alignment between domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAlignment {
    /// Source entity
    pub source_entity: String,
    /// Target entity
    pub target_entity: String,
    /// Alignment confidence
    pub confidence: f64,
    /// Similarity score
    pub similarity: f64,
    /// Alignment method used
    pub method: String,
}

/// Relation alignment between domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationAlignment {
    /// Source relation
    pub source_relation: String,
    /// Target relation
    pub target_relation: String,
    /// Alignment confidence
    pub confidence: f64,
    /// Semantic similarity
    pub semantic_similarity: f64,
    /// Structural similarity
    pub structural_similarity: f64,
}

/// Semantic shift analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticShift {
    /// Concept that shifted
    pub concept: String,
    /// Source domain meaning
    pub source_meaning: String,
    /// Target domain meaning
    pub target_meaning: String,
    /// Shift magnitude
    pub shift_magnitude: f64,
    /// Impact on transfer quality
    pub impact: f64,
}

/// Structural changes between domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralChanges {
    /// Degree distribution changes
    pub degree_distribution_shift: f64,
    /// Clustering coefficient changes
    pub clustering_changes: f64,
    /// Path length changes
    pub path_length_changes: f64,
    /// Community structure changes
    pub community_structure_changes: f64,
}

impl CrossDomainTransferManager {
    /// Create a new cross-domain transfer manager
    pub fn new(config: TransferConfig) -> Self {
        Self {
            source_domains: HashMap::new(),
            target_domains: HashMap::new(),
            transfer_strategies: vec![
                TransferStrategy::DirectTransfer,
                TransferStrategy::FineTuning {
                    learning_rate: 0.001,
                    epochs: 50,
                    freeze_layers: vec![],
                },
                TransferStrategy::DomainAdaptation {
                    alignment_method: AlignmentMethod::AdversarialAlignment,
                    regularization_strength: 0.1,
                },
            ],
            transfer_metrics: vec![
                TransferMetric::TransferAccuracy,
                TransferMetric::AdaptationQuality,
                TransferMetric::SemanticPreservation,
                TransferMetric::StructuralPreservation,
            ],
            config,
        }
    }

    /// Register a source domain
    pub fn register_source_domain(
        &mut self,
        domain_id: String,
        model: Box<dyn EmbeddingModel + Send + Sync>,
        characteristics: DomainCharacteristics,
    ) -> Result<()> {
        let domain_model = DomainModel {
            domain_id: domain_id.clone(),
            model,
            characteristics,
            entity_mappings: HashMap::new(),
            vocabulary: HashSet::new(),
        };

        self.source_domains.insert(domain_id, domain_model);
        Ok(())
    }

    /// Register a target domain
    pub fn register_target_domain(&mut self, domain_spec: DomainSpecification) -> Result<()> {
        self.target_domains
            .insert(domain_spec.domain_id.clone(), domain_spec);
        Ok(())
    }

    /// Evaluate cross-domain transfer quality
    pub async fn evaluate_transfer(
        &self,
        source_domain_id: &str,
        target_domain_id: &str,
        strategy: TransferStrategy,
    ) -> Result<TransferEvaluationResults> {
        let source_domain = self
            .source_domains
            .get(source_domain_id)
            .ok_or_else(|| anyhow!("Source domain not found: {}", source_domain_id))?;

        let target_domain = self
            .target_domains
            .get(target_domain_id)
            .ok_or_else(|| anyhow!("Target domain not found: {}", target_domain_id))?;

        let start_time = std::time::Instant::now();

        // Calculate domain similarity
        let domain_similarity = self.calculate_domain_similarity(
            &source_domain.characteristics,
            &target_domain.characteristics,
        )?;

        // Perform entity alignment
        let entity_alignments = self.align_entities(source_domain, target_domain).await?;

        // Perform relation alignment
        let relation_alignments = self.align_relations(source_domain, target_domain).await?;

        // Analyze semantic shifts
        let semantic_shifts = self
            .analyze_semantic_shifts(source_domain, target_domain)
            .await?;

        // Analyze structural changes
        let structural_changes = self.analyze_structural_changes(
            &source_domain.characteristics,
            &target_domain.characteristics,
        )?;

        // Evaluate transfer metrics
        let mut metric_scores = HashMap::new();
        for metric in &self.transfer_metrics {
            let score = self
                .evaluate_transfer_metric(
                    metric,
                    source_domain,
                    target_domain,
                    &entity_alignments,
                    &relation_alignments,
                )
                .await?;
            metric_scores.insert(format!("{metric:?}"), score);
        }

        // Calculate overall quality
        let overall_quality = if metric_scores.is_empty() {
            0.5 // Default quality when no metrics
        } else {
            let avg_quality = metric_scores.values().sum::<f64>() / metric_scores.len() as f64;
            avg_quality.max(0.0) // Ensure non-negative
        };

        // Calculate baseline performance (random transfer)
        let baseline_performance = 0.1; // Simplified baseline
        let improvement_over_baseline = overall_quality - baseline_performance;

        let transfer_time = start_time.elapsed().as_secs_f64();

        // Generate recommendations
        let recommendations = self.generate_transfer_recommendations(
            domain_similarity,
            &entity_alignments,
            &semantic_shifts,
        );

        let detailed_analysis = TransferAnalysis {
            entity_alignments,
            relation_alignments,
            semantic_shifts,
            structural_changes,
            recommendations,
        };

        Ok(TransferEvaluationResults {
            source_domain: source_domain_id.to_string(),
            target_domain: target_domain_id.to_string(),
            strategy,
            metric_scores,
            overall_quality,
            domain_similarity,
            improvement_over_baseline,
            transfer_time,
            detailed_analysis,
        })
    }

    /// Calculate similarity between two domains
    pub fn calculate_domain_similarity(
        &self,
        source: &DomainCharacteristics,
        target: &DomainCharacteristics,
    ) -> Result<f64> {
        let mut similarity_scores = Vec::new();

        // Language similarity
        let language_similarity = if source.language == target.language {
            1.0
        } else {
            0.5 // Could use more sophisticated language similarity
        };
        similarity_scores.push(language_similarity);

        // Entity type overlap
        let source_entity_types: HashSet<_> = source.entity_types.iter().collect();
        let target_entity_types: HashSet<_> = target.entity_types.iter().collect();
        let entity_overlap = source_entity_types
            .intersection(&target_entity_types)
            .count() as f64;
        let entity_similarity =
            entity_overlap / (source_entity_types.len() + target_entity_types.len()) as f64 * 2.0;
        similarity_scores.push(entity_similarity);

        // Relation type overlap
        let source_relation_types: HashSet<_> = source.relation_types.iter().collect();
        let target_relation_types: HashSet<_> = target.relation_types.iter().collect();
        let relation_overlap = source_relation_types
            .intersection(&target_relation_types)
            .count() as f64;
        let relation_similarity = relation_overlap
            / (source_relation_types.len() + target_relation_types.len()) as f64
            * 2.0;
        similarity_scores.push(relation_similarity);

        // Size similarity
        let size_ratio = (target.size_metrics.num_entities as f64
            / source.size_metrics.num_entities as f64)
            .min(source.size_metrics.num_entities as f64 / target.size_metrics.num_entities as f64);
        similarity_scores.push(size_ratio);

        // Complexity similarity
        let complexity_diff = (source.complexity_metrics.semantic_diversity
            - target.complexity_metrics.semantic_diversity)
            .abs();
        let complexity_similarity = (1.0 - complexity_diff).max(0.0);
        similarity_scores.push(complexity_similarity);

        // Overall similarity
        let overall_similarity =
            similarity_scores.iter().sum::<f64>() / similarity_scores.len() as f64;

        Ok(overall_similarity)
    }

    /// Align entities between source and target domains
    async fn align_entities(
        &self,
        source: &DomainModel,
        target: &DomainSpecification,
    ) -> Result<Vec<EntityAlignment>> {
        let mut alignments = Vec::new();

        let source_entities = source.model.get_entities();
        let target_entities = self.extract_entities_from_triples(&target.training_data);

        // Simple name-based alignment (could be enhanced with semantic matching)
        for source_entity in &source_entities {
            for target_entity in &target_entities {
                let similarity = self.calculate_string_similarity(source_entity, target_entity);

                if similarity > 0.7 {
                    // High similarity threshold
                    alignments.push(EntityAlignment {
                        source_entity: source_entity.clone(),
                        target_entity: target_entity.clone(),
                        confidence: similarity,
                        similarity,
                        method: "string_similarity".to_string(),
                    });
                }
            }
        }

        // Semantic alignment using embeddings
        for source_entity in source_entities.iter().take(50) {
            // Limit for efficiency
            if let Ok(source_embedding) = source.model.get_entity_embedding(source_entity) {
                let mut best_match = None;
                let mut best_similarity = 0.0;

                for target_entity in target_entities.iter().take(50) {
                    // Create a simple embedding for target entity (simplified)
                    let target_embedding = self.create_simple_embedding(target_entity);
                    let similarity = self.cosine_similarity(&source_embedding, &target_embedding);

                    if similarity > best_similarity && similarity > 0.5 {
                        best_similarity = similarity;
                        best_match = Some(target_entity.clone());
                    }
                }

                if let Some(target_entity) = best_match {
                    alignments.push(EntityAlignment {
                        source_entity: source_entity.clone(),
                        target_entity,
                        confidence: best_similarity,
                        similarity: best_similarity,
                        method: "semantic_embedding".to_string(),
                    });
                }
            }
        }

        Ok(alignments)
    }

    /// Align relations between source and target domains
    async fn align_relations(
        &self,
        source: &DomainModel,
        target: &DomainSpecification,
    ) -> Result<Vec<RelationAlignment>> {
        let mut alignments = Vec::new();

        let source_relations = source.model.get_relations();
        let target_relations = self.extract_relations_from_triples(&target.training_data);

        for source_relation in &source_relations {
            for target_relation in &target_relations {
                let semantic_similarity =
                    self.calculate_string_similarity(source_relation, target_relation);

                // Structural similarity (simplified - could analyze usage patterns)
                let structural_similarity = 0.5; // Placeholder

                if semantic_similarity > 0.6 {
                    alignments.push(RelationAlignment {
                        source_relation: source_relation.clone(),
                        target_relation: target_relation.clone(),
                        confidence: (semantic_similarity + structural_similarity) / 2.0,
                        semantic_similarity,
                        structural_similarity,
                    });
                }
            }
        }

        Ok(alignments)
    }

    /// Analyze semantic shifts between domains
    async fn analyze_semantic_shifts(
        &self,
        source: &DomainModel,
        target: &DomainSpecification,
    ) -> Result<Vec<SemanticShift>> {
        let mut shifts = Vec::new();

        // Analyze shifts in common entities/concepts
        let source_entities = source.model.get_entities();
        let target_entities = self.extract_entities_from_triples(&target.training_data);

        for source_entity in source_entities.iter().take(20) {
            for target_entity in target_entities.iter().take(20) {
                if self.calculate_string_similarity(source_entity, target_entity) > 0.8 {
                    // Same concept, different domains
                    let shift_magnitude = self.calculate_semantic_shift_magnitude(
                        source_entity,
                        target_entity,
                        source,
                        target,
                    )?;

                    if shift_magnitude > 0.3 {
                        shifts.push(SemanticShift {
                            concept: source_entity.clone(),
                            source_meaning: format!("Source domain context: {source_entity}"),
                            target_meaning: format!("Target domain context: {target_entity}"),
                            shift_magnitude,
                            impact: shift_magnitude * 0.5, // Simplified impact calculation
                        });
                    }
                }
            }
        }

        Ok(shifts)
    }

    /// Analyze structural changes between domains
    fn analyze_structural_changes(
        &self,
        source: &DomainCharacteristics,
        target: &DomainCharacteristics,
    ) -> Result<StructuralChanges> {
        // Calculate relative changes in structural properties
        let degree_distribution_shift =
            (source.size_metrics.avg_entity_degree - target.size_metrics.avg_entity_degree).abs()
                / source.size_metrics.avg_entity_degree;

        let clustering_changes = 0.1; // Placeholder - would calculate actual clustering coefficients
        let path_length_changes = 0.15; // Placeholder
        let community_structure_changes = 0.2; // Placeholder

        Ok(StructuralChanges {
            degree_distribution_shift,
            clustering_changes,
            path_length_changes,
            community_structure_changes,
        })
    }

    /// Evaluate a specific transfer metric
    async fn evaluate_transfer_metric(
        &self,
        metric: &TransferMetric,
        source: &DomainModel,
        target: &DomainSpecification,
        entity_alignments: &[EntityAlignment],
        relation_alignments: &[RelationAlignment],
    ) -> Result<f64> {
        match metric {
            TransferMetric::TransferAccuracy => {
                // Measure how well source model performs on target tasks
                self.calculate_transfer_accuracy(source, target).await
            }
            TransferMetric::AdaptationQuality => {
                // Quality of domain adaptation
                if entity_alignments.is_empty() {
                    Ok(0.5) // Default quality when no alignments
                } else {
                    Ok(entity_alignments.iter().map(|a| a.confidence).sum::<f64>()
                        / entity_alignments.len() as f64)
                }
            }
            TransferMetric::EntityAlignmentQuality => {
                // Quality of entity alignments
                if entity_alignments.is_empty() {
                    Ok(0.5) // Default quality when no alignments
                } else {
                    Ok(entity_alignments
                        .iter()
                        .filter(|a| a.confidence > 0.7)
                        .count() as f64
                        / entity_alignments.len() as f64)
                }
            }
            TransferMetric::SemanticPreservation => {
                // How well semantics are preserved
                self.calculate_semantic_preservation(source, target, entity_alignments)
                    .await
            }
            TransferMetric::StructuralPreservation => {
                // How well structure is preserved
                self.calculate_structural_preservation(source, target, relation_alignments)
                    .await
            }
            TransferMetric::TransferEfficiency => {
                // Performance improvement per unit of effort
                self.calculate_transfer_efficiency(source, target).await
            }
            TransferMetric::CatastrophicForgetting => {
                // How much original performance is lost
                self.calculate_catastrophic_forgetting(source, target).await
            }
            TransferMetric::CrossDomainCoherence => {
                // Evaluate coherence across domain boundaries
                self.calculate_cross_domain_coherence(source, target, entity_alignments)
                    .await
            }
            TransferMetric::KnowledgeRetention => {
                // Measure knowledge retention during transfer
                self.calculate_knowledge_retention(source, target).await
            }
            TransferMetric::AdaptationSpeed => {
                // Speed of adaptation to new domain
                self.calculate_adaptation_speed(source, target).await
            }
            TransferMetric::TransferRobustness => {
                // Robustness across domain variations
                self.calculate_transfer_robustness(source, target).await
            }
            TransferMetric::SemanticDriftDetection => {
                // Detection of semantic drift during transfer
                self.calculate_semantic_drift_detection(source, target)
                    .await
            }
            TransferMetric::GeneralizationAbility => {
                // Ability to generalize across domains
                self.calculate_generalization_ability(source, target).await
            }
        }
    }

    /// Calculate transfer accuracy
    async fn calculate_transfer_accuracy(
        &self,
        source: &DomainModel,
        target: &DomainSpecification,
    ) -> Result<f64> {
        let mut correct_predictions = 0;
        let total_predictions = target
            .test_data
            .len()
            .min(self.config.evaluation_sample_size);

        if total_predictions == 0 {
            return Ok(0.5); // Default accuracy when no data
        }

        for (subject, predicate, object) in target.test_data.iter().take(total_predictions) {
            // Try to score the triple using the source model
            if let Ok(score) = source.model.score_triple(subject, predicate, object) {
                // Simple threshold-based classification
                if score > 0.0 {
                    correct_predictions += 1;
                }
            }
        }

        Ok(correct_predictions as f64 / total_predictions as f64)
    }

    /// Calculate semantic preservation score
    async fn calculate_semantic_preservation(
        &self,
        source: &DomainModel,
        _target: &DomainSpecification,
        entity_alignments: &[EntityAlignment],
    ) -> Result<f64> {
        if entity_alignments.is_empty() {
            return Ok(0.0);
        }

        let mut preservation_scores = Vec::new();

        for alignment in entity_alignments.iter().take(20) {
            // Limit for efficiency
            if let Ok(source_embedding) =
                source.model.get_entity_embedding(&alignment.source_entity)
            {
                // Create target embedding (simplified)
                let target_embedding = self.create_simple_embedding(&alignment.target_entity);

                // Calculate preservation as cosine similarity
                let preservation = self.cosine_similarity(&source_embedding, &target_embedding);
                preservation_scores.push(preservation);
            }
        }

        if preservation_scores.is_empty() {
            Ok(0.0)
        } else {
            Ok(preservation_scores.iter().sum::<f64>() / preservation_scores.len() as f64)
        }
    }

    /// Calculate structural preservation score
    async fn calculate_structural_preservation(
        &self,
        _source: &DomainModel,
        _target: &DomainSpecification,
        relation_alignments: &[RelationAlignment],
    ) -> Result<f64> {
        if relation_alignments.is_empty() {
            return Ok(0.5); // Neutral score
        }

        // Average structural similarity from relation alignments
        let avg_structural_similarity = relation_alignments
            .iter()
            .map(|a| a.structural_similarity)
            .sum::<f64>()
            / relation_alignments.len() as f64;

        Ok(avg_structural_similarity)
    }

    /// Generate transfer recommendations
    fn generate_transfer_recommendations(
        &self,
        domain_similarity: f64,
        entity_alignments: &[EntityAlignment],
        semantic_shifts: &[SemanticShift],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if domain_similarity < 0.3 {
            recommendations.push(
                "Low domain similarity detected. Consider using domain adaptation techniques."
                    .to_string(),
            );
        }

        if entity_alignments.len() < 10 {
            recommendations.push(
                "Few entity alignments found. Consider improving entity linking methods."
                    .to_string(),
            );
        }

        let high_shift_count = semantic_shifts
            .iter()
            .filter(|s| s.shift_magnitude > 0.5)
            .count();
        if high_shift_count > 5 {
            recommendations.push(
                "Significant semantic shifts detected. Consider gradual domain adaptation."
                    .to_string(),
            );
        }

        if domain_similarity > 0.7 {
            recommendations
                .push("High domain similarity. Direct transfer should work well.".to_string());
        }

        recommendations
    }

    /// Helper: Extract entities from triples
    fn extract_entities_from_triples(
        &self,
        triples: &[(String, String, String)],
    ) -> HashSet<String> {
        let mut entities = HashSet::new();
        for (subject, _, object) in triples {
            entities.insert(subject.clone());
            entities.insert(object.clone());
        }
        entities.into_iter().collect::<HashSet<_>>()
    }

    /// Helper: Extract relations from triples
    fn extract_relations_from_triples(
        &self,
        triples: &[(String, String, String)],
    ) -> HashSet<String> {
        triples
            .iter()
            .map(|(_, predicate, _)| predicate.clone())
            .collect()
    }

    /// Helper: Calculate string similarity
    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            return 1.0;
        }

        // Simple Jaccard similarity on character n-grams
        let n = 3;
        let ngrams1: HashSet<String> = s1
            .chars()
            .collect::<Vec<_>>()
            .windows(n)
            .map(|w| w.iter().collect())
            .collect();
        let ngrams2: HashSet<String> = s2
            .chars()
            .collect::<Vec<_>>()
            .windows(n)
            .map(|w| w.iter().collect())
            .collect();

        if ngrams1.is_empty() && ngrams2.is_empty() {
            return 1.0;
        }

        let intersection = ngrams1.intersection(&ngrams2).count();
        let union = ngrams1.union(&ngrams2).count();

        intersection as f64 / union as f64
    }

    /// Helper: Create simple embedding for target entity
    fn create_simple_embedding(&self, entity: &str) -> Vector {
        // Simple character-based embedding (in practice, use pre-trained embeddings)
        let mut embedding = vec![0.0f32; 100]; // Fixed dimension
        for (i, byte) in entity.bytes().enumerate() {
            if i >= embedding.len() {
                break;
            }
            embedding[i] = (byte as f32) / 255.0;
        }
        Vector::new(embedding)
    }

    /// Helper: Calculate cosine similarity
    fn cosine_similarity(&self, v1: &Vector, v2: &Vector) -> f64 {
        let dot_product: f32 = v1
            .values
            .iter()
            .zip(v2.values.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = v1.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = v2.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot_product / (norm_a * norm_b)) as f64
        } else {
            0.0
        }
    }

    /// Helper: Calculate semantic shift magnitude
    fn calculate_semantic_shift_magnitude(
        &self,
        _source_entity: &str,
        _target_entity: &str,
        _source: &DomainModel,
        _target: &DomainSpecification,
    ) -> Result<f64> {
        // Simplified - in practice would analyze contextual differences
        Ok({
            let mut random = Random::default();
            random.random::<f64>() * 0.8
        }) // Random for demonstration
    }

    /// Get all registered source domains
    pub fn get_source_domains(&self) -> Vec<String> {
        self.source_domains.keys().cloned().collect()
    }

    /// Get all registered target domains
    pub fn get_target_domains(&self) -> Vec<String> {
        self.target_domains.keys().cloned().collect()
    }

    /// Get domain characteristics
    pub fn get_domain_characteristics(&self, domain_id: &str) -> Option<&DomainCharacteristics> {
        self.source_domains
            .get(domain_id)
            .map(|d| &d.characteristics)
            .or_else(|| {
                self.target_domains
                    .get(domain_id)
                    .map(|d| &d.characteristics)
            })
    }

    /// Calculate transfer efficiency
    async fn calculate_transfer_efficiency(
        &self,
        source: &DomainModel,
        target: &DomainSpecification,
    ) -> Result<f64> {
        let start_time = std::time::Instant::now();

        // Calculate domain similarity as a proxy for transfer ease
        let domain_similarity =
            self.calculate_domain_similarity(&source.characteristics, &target.characteristics)?;

        // Simulate transfer performance
        let transfer_accuracy = self.calculate_transfer_accuracy(source, target).await?;
        let transfer_time = start_time.elapsed().as_secs_f64();

        // Efficiency = (accuracy * domain_similarity) / normalized_time
        let normalized_time = (transfer_time / 60.0).clamp(0.01, 1.0); // Normalize to minutes
        let efficiency = (transfer_accuracy * domain_similarity) / normalized_time;

        Ok(efficiency.clamp(0.0, 1.0))
    }

    /// Calculate catastrophic forgetting
    async fn calculate_catastrophic_forgetting(
        &self,
        source: &DomainModel,
        target: &DomainSpecification,
    ) -> Result<f64> {
        // Measure performance degradation on source domain tasks
        let source_entities = source.model.get_entities();
        let sample_size = source_entities.len().min(20);

        if sample_size == 0 {
            return Ok(0.0);
        }

        let mut forgetting_scores = Vec::new();

        // Test retention of source domain knowledge
        for entity in source_entities.iter().take(sample_size) {
            if let Ok(_source_embedding) = source.model.get_entity_embedding(entity) {
                // Simulate post-adaptation embedding quality degradation
                let target_entities = self.extract_entities_from_triples(&target.training_data);
                let domain_overlap = target_entities.contains(entity);

                let degradation = if domain_overlap {
                    // Less forgetting for overlapping entities
                    let mut random = Random::default();
                    0.1 + random.random::<f64>() * 0.2
                } else {
                    // More forgetting for non-overlapping entities
                    let mut random = Random::default();
                    0.3 + random.random::<f64>() * 0.4
                };

                forgetting_scores.push(degradation);
            }
        }

        if forgetting_scores.is_empty() {
            Ok(0.1) // Low forgetting by default
        } else {
            let avg_forgetting =
                forgetting_scores.iter().sum::<f64>() / forgetting_scores.len() as f64;
            Ok(avg_forgetting.clamp(0.0, 1.0))
        }
    }

    /// Calculate cross-domain coherence
    async fn calculate_cross_domain_coherence(
        &self,
        source: &DomainModel,
        target: &DomainSpecification,
        entity_alignments: &[EntityAlignment],
    ) -> Result<f64> {
        if entity_alignments.is_empty() {
            return Ok(0.5);
        }

        let mut coherence_scores = Vec::new();

        // Evaluate coherence across aligned entities
        for alignment in entity_alignments.iter().take(15) {
            if alignment.confidence > 0.6 {
                if let Ok(source_embedding) =
                    source.model.get_entity_embedding(&alignment.source_entity)
                {
                    let target_embedding = self.create_simple_embedding(&alignment.target_entity);

                    // Calculate embedding coherence
                    let embedding_coherence =
                        self.cosine_similarity(&source_embedding, &target_embedding);

                    // Calculate neighborhood coherence
                    let source_neighbors =
                        self.get_source_neighbors(&alignment.source_entity, source);
                    let target_neighbors =
                        self.get_target_neighbors(&alignment.target_entity, target);
                    let neighborhood_coherence = self
                        .calculate_neighborhood_similarity(&source_neighbors, &target_neighbors);

                    // Combined coherence score
                    let combined_coherence = (embedding_coherence + neighborhood_coherence) / 2.0;
                    coherence_scores.push(combined_coherence);
                }
            }
        }

        if coherence_scores.is_empty() {
            Ok(0.5)
        } else {
            let avg_coherence =
                coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64;
            Ok(avg_coherence.clamp(0.0, 1.0))
        }
    }

    /// Calculate knowledge retention
    async fn calculate_knowledge_retention(
        &self,
        _source: &DomainModel,
        _target: &DomainSpecification,
    ) -> Result<f64> {
        // Placeholder implementation
        Ok(0.85)
    }

    /// Calculate adaptation speed
    async fn calculate_adaptation_speed(
        &self,
        _source: &DomainModel,
        _target: &DomainSpecification,
    ) -> Result<f64> {
        // Placeholder implementation
        Ok(0.75)
    }

    /// Calculate transfer robustness
    async fn calculate_transfer_robustness(
        &self,
        _source: &DomainModel,
        _target: &DomainSpecification,
    ) -> Result<f64> {
        // Placeholder implementation
        Ok(0.8)
    }

    /// Calculate semantic drift detection
    async fn calculate_semantic_drift_detection(
        &self,
        _source: &DomainModel,
        _target: &DomainSpecification,
    ) -> Result<f64> {
        // Placeholder implementation
        Ok(0.7)
    }

    /// Calculate generalization ability
    async fn calculate_generalization_ability(
        &self,
        _source: &DomainModel,
        _target: &DomainSpecification,
    ) -> Result<f64> {
        // Placeholder implementation
        Ok(0.8)
    }

    /// Helper: Get source domain neighbors
    fn get_source_neighbors(&self, _entity: &str, source: &DomainModel) -> Vec<String> {
        // Get related entities from source domain
        let relations = source.model.get_relations();
        relations.into_iter().take(5).collect()
    }

    /// Helper: Get target domain neighbors
    fn get_target_neighbors(&self, entity: &str, target: &DomainSpecification) -> Vec<String> {
        let mut neighbors = Vec::new();
        for (subject, predicate, object) in &target.training_data {
            if subject == entity {
                neighbors.push(object.clone());
                neighbors.push(predicate.clone());
            } else if object == entity {
                neighbors.push(subject.clone());
                neighbors.push(predicate.clone());
            }
        }
        neighbors.into_iter().take(5).collect()
    }

    /// Helper: Calculate neighborhood similarity
    fn calculate_neighborhood_similarity(
        &self,
        source_neighbors: &[String],
        target_neighbors: &[String],
    ) -> f64 {
        if source_neighbors.is_empty() && target_neighbors.is_empty() {
            return 1.0;
        }

        if source_neighbors.is_empty() || target_neighbors.is_empty() {
            return 0.0;
        }

        let source_set: HashSet<&String> = source_neighbors.iter().collect();
        let target_set: HashSet<&String> = target_neighbors.iter().collect();

        let intersection = source_set.intersection(&target_set).count();
        let union = source_set.union(&target_set).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

/// Cross-domain transfer utilities
pub struct TransferUtils;

impl TransferUtils {
    /// Create domain characteristics from triples
    pub fn analyze_domain_from_triples(
        _domain_id: String,
        triples: &[(String, String, String)],
    ) -> DomainCharacteristics {
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();

        for (subject, predicate, object) in triples {
            entities.insert(subject.clone());
            entities.insert(object.clone());
            relations.insert(predicate.clone());
        }

        let num_entities = entities.len();
        let num_relations = relations.len();
        let num_triples = triples.len();

        // Calculate average entity degree
        let mut entity_degrees = HashMap::new();
        for (subject, _, object) in triples {
            *entity_degrees.entry(subject.clone()).or_insert(0) += 1;
            *entity_degrees.entry(object.clone()).or_insert(0) += 1;
        }
        let avg_entity_degree = if num_entities > 0 {
            entity_degrees.values().sum::<usize>() as f64 / num_entities as f64
        } else {
            0.0
        };

        // Calculate graph density
        let max_possible_edges = num_entities * (num_entities - 1);
        let graph_density = if max_possible_edges > 0 {
            num_triples as f64 / max_possible_edges as f64
        } else {
            0.0
        };

        DomainCharacteristics {
            domain_type: "unknown".to_string(),
            language: "unknown".to_string(),
            entity_types: vec!["Entity".to_string()], // Simplified
            relation_types: relations.into_iter().collect(),
            size_metrics: DomainSizeMetrics {
                num_entities,
                num_relations,
                num_triples,
                avg_entity_degree,
                graph_density,
            },
            complexity_metrics: DomainComplexityMetrics {
                entity_type_diversity: 1, // Simplified
                relation_type_diversity: num_relations,
                hierarchical_depth: 3,                           // Estimated
                semantic_diversity: 0.5,                         // Estimated
                structural_complexity: avg_entity_degree / 10.0, // Simplified metric
            },
        }
    }

    /// Create a simple domain specification for testing
    pub fn create_test_domain_specification(
        domain_id: String,
        training_data: Vec<(String, String, String)>,
    ) -> DomainSpecification {
        // Split data into train/val/test
        let total = training_data.len();
        let train_size = (total as f64 * 0.7) as usize;
        let val_size = (total as f64 * 0.15) as usize;

        let training = training_data[..train_size].to_vec();
        let validation = training_data[train_size..train_size + val_size].to_vec();
        let test = training_data[train_size + val_size..].to_vec();

        let characteristics = Self::analyze_domain_from_triples(domain_id.clone(), &training);

        DomainSpecification {
            domain_id,
            characteristics,
            training_data: training,
            validation_data: validation,
            test_data: test,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::transe::TransE;

    #[test]
    fn test_transfer_config_default() {
        let config = TransferConfig::default();
        assert!(config.enable_domain_adaptation);
        assert!(config.use_adversarial_alignment);
        assert_eq!(config.max_alignment_iterations, 100);
    }

    #[test]
    fn test_domain_characteristics_creation() {
        let triples = vec![
            ("alice".to_string(), "knows".to_string(), "bob".to_string()),
            ("bob".to_string(), "likes".to_string(), "pizza".to_string()),
            (
                "alice".to_string(),
                "likes".to_string(),
                "coffee".to_string(),
            ),
        ];

        let characteristics =
            TransferUtils::analyze_domain_from_triples("test_domain".to_string(), &triples);

        assert_eq!(characteristics.size_metrics.num_triples, 3);
        assert_eq!(characteristics.size_metrics.num_entities, 4); // alice, bob, pizza, coffee
        assert_eq!(characteristics.size_metrics.num_relations, 2); // knows, likes
    }

    #[test]
    fn test_string_similarity() {
        let manager = CrossDomainTransferManager::new(TransferConfig::default());

        let sim1 = manager.calculate_string_similarity("hello", "hello");
        assert_eq!(sim1, 1.0);

        let sim2 = manager.calculate_string_similarity("hello", "world");
        assert!(sim2 < 0.5);

        let sim3 = manager.calculate_string_similarity("testing", "test");
        assert!(sim3 > 0.3);
    }

    #[tokio::test]
    async fn test_transfer_evaluation() {
        let mut manager = CrossDomainTransferManager::new(TransferConfig::default());

        // Create source domain
        let source_model = Box::new(TransE::new(Default::default()));
        let source_characteristics = DomainCharacteristics {
            domain_type: "test".to_string(),
            language: "en".to_string(),
            entity_types: vec!["Person".to_string()],
            relation_types: vec!["knows".to_string()],
            size_metrics: DomainSizeMetrics {
                num_entities: 100,
                num_relations: 10,
                num_triples: 500,
                avg_entity_degree: 5.0,
                graph_density: 0.01,
            },
            complexity_metrics: DomainComplexityMetrics {
                entity_type_diversity: 2,
                relation_type_diversity: 10,
                hierarchical_depth: 3,
                semantic_diversity: 0.6,
                structural_complexity: 0.5,
            },
        };

        manager
            .register_source_domain("source".to_string(), source_model, source_characteristics)
            .unwrap();

        // Create target domain
        let target_spec = TransferUtils::create_test_domain_specification(
            "target".to_string(),
            vec![
                ("alice".to_string(), "knows".to_string(), "bob".to_string()),
                (
                    "bob".to_string(),
                    "knows".to_string(),
                    "charlie".to_string(),
                ),
            ],
        );

        manager.register_target_domain(target_spec).unwrap();

        // Evaluate transfer
        let results = manager
            .evaluate_transfer("source", "target", TransferStrategy::DirectTransfer)
            .await;

        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.source_domain, "source");
        assert_eq!(results.target_domain, "target");
        assert!(results.overall_quality >= 0.0);
        assert!(results.overall_quality <= 1.0);
    }

    #[test]
    fn test_domain_similarity_calculation() {
        let manager = CrossDomainTransferManager::new(TransferConfig::default());

        let source = DomainCharacteristics {
            domain_type: "biomedical".to_string(),
            language: "en".to_string(),
            entity_types: vec!["Gene".to_string(), "Disease".to_string()],
            relation_types: vec!["causes".to_string(), "treats".to_string()],
            size_metrics: DomainSizeMetrics {
                num_entities: 1000,
                num_relations: 50,
                num_triples: 5000,
                avg_entity_degree: 5.0,
                graph_density: 0.005,
            },
            complexity_metrics: DomainComplexityMetrics {
                entity_type_diversity: 2,
                relation_type_diversity: 50,
                hierarchical_depth: 4,
                semantic_diversity: 0.7,
                structural_complexity: 0.6,
            },
        };

        let target = DomainCharacteristics {
            domain_type: "medical".to_string(),
            language: "en".to_string(),
            entity_types: vec!["Gene".to_string(), "Drug".to_string()],
            relation_types: vec!["treats".to_string(), "interacts".to_string()],
            size_metrics: DomainSizeMetrics {
                num_entities: 800,
                num_relations: 40,
                num_triples: 4000,
                avg_entity_degree: 5.0,
                graph_density: 0.006,
            },
            complexity_metrics: DomainComplexityMetrics {
                entity_type_diversity: 2,
                relation_type_diversity: 40,
                hierarchical_depth: 3,
                semantic_diversity: 0.6,
                structural_complexity: 0.5,
            },
        };

        let similarity = manager
            .calculate_domain_similarity(&source, &target)
            .unwrap();
        assert!(similarity > 0.0);
        assert!(similarity <= 1.0);

        // Should have reasonable similarity due to shared entity and relation types
        assert!(similarity > 0.2);
    }
}
