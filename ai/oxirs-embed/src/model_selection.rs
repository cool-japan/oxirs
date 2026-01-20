//! # Model Selection Guidance
//!
//! This module provides intelligent model selection and recommendation capabilities
//! to help users choose the most appropriate embedding model for their specific
//! knowledge graph and use case.
//!
//! ## Features
//!
//! - **Automatic Model Recommendation**: Based on dataset characteristics
//! - **Model Comparison**: Compare multiple models on the same dataset
//! - **Performance Profiling**: Benchmark model performance
//! - **Resource Requirements**: Estimate memory and compute needs
//! - **Use Case Matching**: Recommend models for specific applications
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use oxirs_embed::model_selection::{ModelSelector, DatasetCharacteristics, UseCaseType};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Define dataset characteristics
//! let characteristics = DatasetCharacteristics {
//!     num_entities: 10000,
//!     num_relations: 50,
//!     num_triples: 50000,
//!     avg_degree: 5.0,
//!     is_sparse: false,
//!     has_hierarchies: true,
//!     has_complex_relations: true,
//!     domain: Some("biomedical".to_string()),
//! };
//!
//! // Get model recommendations
//! let selector = ModelSelector::new();
//! let recommendations = selector.recommend_models(&characteristics, UseCaseType::LinkPrediction)?;
//!
//! for rec in recommendations {
//!     println!("Model: {}, Score: {:.2}, Reason: {}",
//!              rec.model_type, rec.suitability_score, rec.reasoning);
//! }
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Dataset characteristics for model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetCharacteristics {
    /// Number of unique entities
    pub num_entities: usize,
    /// Number of unique relations
    pub num_relations: usize,
    /// Total number of triples
    pub num_triples: usize,
    /// Average node degree
    pub avg_degree: f64,
    /// Whether the graph is sparse (avg_degree << num_entities)
    pub is_sparse: bool,
    /// Whether the graph has hierarchical structure
    pub has_hierarchies: bool,
    /// Whether the graph has complex multi-hop relations
    pub has_complex_relations: bool,
    /// Domain of the knowledge graph (e.g., "biomedical", "general", "social")
    pub domain: Option<String>,
}

impl DatasetCharacteristics {
    /// Automatically infer characteristics from basic statistics
    pub fn infer(num_entities: usize, num_relations: usize, num_triples: usize) -> Self {
        let avg_degree = if num_entities > 0 {
            (num_triples as f64 * 2.0) / num_entities as f64
        } else {
            0.0
        };

        let is_sparse = avg_degree < (num_entities as f64).sqrt();

        Self {
            num_entities,
            num_relations,
            num_triples,
            avg_degree,
            is_sparse,
            has_hierarchies: false, // Conservative default
            has_complex_relations: num_relations > 10,
            domain: None,
        }
    }

    /// Calculate graph density
    pub fn density(&self) -> f64 {
        if self.num_entities == 0 {
            return 0.0;
        }
        let max_possible = (self.num_entities * (self.num_entities - 1)) as f64;
        if max_possible == 0.0 {
            return 0.0;
        }
        self.num_triples as f64 / max_possible
    }

    /// Estimate memory requirements in MB
    pub fn estimated_memory_mb(&self, embedding_dim: usize) -> f64 {
        // Rough estimate: entities + relations + overhead
        let entity_mem = (self.num_entities * embedding_dim * 4) as f64 / 1_048_576.0; // 4 bytes per f32
        let relation_mem = (self.num_relations * embedding_dim * 4) as f64 / 1_048_576.0;
        let overhead = 50.0; // MB for other structures

        entity_mem + relation_mem + overhead
    }
}

/// Type of use case for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UseCaseType {
    /// Link prediction (predicting missing triples)
    LinkPrediction,
    /// Entity classification
    EntityClassification,
    /// Relation extraction
    RelationExtraction,
    /// Question answering
    QuestionAnswering,
    /// Knowledge graph completion
    KGCompletion,
    /// Similarity search
    SimilaritySearch,
    /// General purpose embeddings
    GeneralPurpose,
}

/// Available embedding model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    TransE,
    DistMult,
    ComplEx,
    RotatE,
    HolE,
    ConvE,
    TuckER,
    QuatD,
    GNN,
    Transformer,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::TransE => write!(f, "TransE"),
            ModelType::DistMult => write!(f, "DistMult"),
            ModelType::ComplEx => write!(f, "ComplEx"),
            ModelType::RotatE => write!(f, "RotatE"),
            ModelType::HolE => write!(f, "HolE"),
            ModelType::ConvE => write!(f, "ConvE"),
            ModelType::TuckER => write!(f, "TuckER"),
            ModelType::QuatD => write!(f, "QuatD"),
            ModelType::GNN => write!(f, "GNN"),
            ModelType::Transformer => write!(f, "Transformer"),
        }
    }
}

/// Model recommendation with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub model_type: ModelType,
    pub suitability_score: f64,
    pub reasoning: String,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
    pub recommended_dimensions: usize,
    pub estimated_training_time: TrainingTime,
    pub memory_requirement: MemoryRequirement,
}

/// Training time estimate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingTime {
    Fast,     // < 5 minutes
    Medium,   // 5-30 minutes
    Slow,     // 30-60 minutes
    VerySlow, // > 1 hour
}

impl std::fmt::Display for TrainingTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingTime::Fast => write!(f, "Fast (< 5 min)"),
            TrainingTime::Medium => write!(f, "Medium (5-30 min)"),
            TrainingTime::Slow => write!(f, "Slow (30-60 min)"),
            TrainingTime::VerySlow => write!(f, "Very Slow (> 1 hour)"),
        }
    }
}

/// Memory requirement estimate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryRequirement {
    Low,      // < 500 MB
    Medium,   // 500 MB - 2 GB
    High,     // 2 GB - 8 GB
    VeryHigh, // > 8 GB
}

impl std::fmt::Display for MemoryRequirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryRequirement::Low => write!(f, "Low (< 500 MB)"),
            MemoryRequirement::Medium => write!(f, "Medium (500 MB - 2 GB)"),
            MemoryRequirement::High => write!(f, "High (2 GB - 8 GB)"),
            MemoryRequirement::VeryHigh => write!(f, "Very High (> 8 GB)"),
        }
    }
}

/// Model selector for intelligent recommendation
pub struct ModelSelector {
    model_profiles: HashMap<ModelType, ModelProfile>,
}

/// Profile of a model's characteristics
#[derive(Debug, Clone)]
struct ModelProfile {
    model_type: ModelType,
    /// Strengths of this model
    strengths: Vec<String>,
    /// Weaknesses of this model
    weaknesses: Vec<String>,
    /// Best use cases
    best_for: Vec<UseCaseType>,
    /// Complexity score (1-10, higher = more complex)
    complexity: u8,
    /// Speed score (1-10, higher = faster)
    speed: u8,
    /// Accuracy score (1-10, higher = more accurate)
    accuracy: u8,
    /// Works well with sparse graphs
    handles_sparse: bool,
    /// Works well with hierarchies
    handles_hierarchies: bool,
    /// Works well with complex relations
    handles_complex_relations: bool,
}

impl Default for ModelSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelSelector {
    /// Create a new model selector with predefined model profiles
    pub fn new() -> Self {
        let mut model_profiles = HashMap::new();

        // TransE profile
        model_profiles.insert(
            ModelType::TransE,
            ModelProfile {
                model_type: ModelType::TransE,
                strengths: vec![
                    "Simple and efficient".to_string(),
                    "Good for hierarchical relations".to_string(),
                    "Fast training".to_string(),
                ],
                weaknesses: vec![
                    "Cannot model symmetric relations well".to_string(),
                    "Limited expressiveness".to_string(),
                ],
                best_for: vec![UseCaseType::LinkPrediction, UseCaseType::GeneralPurpose],
                complexity: 2,
                speed: 9,
                accuracy: 6,
                handles_sparse: true,
                handles_hierarchies: true,
                handles_complex_relations: false,
            },
        );

        // DistMult profile
        model_profiles.insert(
            ModelType::DistMult,
            ModelProfile {
                model_type: ModelType::DistMult,
                strengths: vec![
                    "Very fast".to_string(),
                    "Good for symmetric relations".to_string(),
                    "Low memory footprint".to_string(),
                ],
                weaknesses: vec![
                    "Cannot model asymmetric relations".to_string(),
                    "Cannot capture composition".to_string(),
                ],
                best_for: vec![
                    UseCaseType::SimilaritySearch,
                    UseCaseType::EntityClassification,
                ],
                complexity: 1,
                speed: 10,
                accuracy: 5,
                handles_sparse: true,
                handles_hierarchies: false,
                handles_complex_relations: false,
            },
        );

        // ComplEx profile
        model_profiles.insert(
            ModelType::ComplEx,
            ModelProfile {
                model_type: ModelType::ComplEx,
                strengths: vec![
                    "Handles symmetric and asymmetric relations".to_string(),
                    "Good theoretical properties".to_string(),
                    "State-of-the-art performance".to_string(),
                ],
                weaknesses: vec![
                    "More complex than TransE".to_string(),
                    "Requires more memory".to_string(),
                ],
                best_for: vec![UseCaseType::LinkPrediction, UseCaseType::KGCompletion],
                complexity: 5,
                speed: 7,
                accuracy: 8,
                handles_sparse: true,
                handles_hierarchies: true,
                handles_complex_relations: true,
            },
        );

        // RotatE profile
        model_profiles.insert(
            ModelType::RotatE,
            ModelProfile {
                model_type: ModelType::RotatE,
                strengths: vec![
                    "Excellent for complex relations".to_string(),
                    "Handles composition patterns".to_string(),
                    "Strong theoretical foundation".to_string(),
                ],
                weaknesses: vec![
                    "Slower than simpler models".to_string(),
                    "Higher memory usage".to_string(),
                ],
                best_for: vec![UseCaseType::LinkPrediction, UseCaseType::RelationExtraction],
                complexity: 6,
                speed: 6,
                accuracy: 9,
                handles_sparse: true,
                handles_hierarchies: true,
                handles_complex_relations: true,
            },
        );

        // HolE profile
        model_profiles.insert(
            ModelType::HolE,
            ModelProfile {
                model_type: ModelType::HolE,
                strengths: vec![
                    "Memory efficient".to_string(),
                    "Good compositional properties".to_string(),
                    "Fast inference".to_string(),
                ],
                weaknesses: vec![
                    "Training can be slower".to_string(),
                    "Less intuitive than TransE".to_string(),
                ],
                best_for: vec![UseCaseType::KGCompletion, UseCaseType::LinkPrediction],
                complexity: 5,
                speed: 7,
                accuracy: 7,
                handles_sparse: true,
                handles_hierarchies: false,
                handles_complex_relations: true,
            },
        );

        // ConvE profile
        model_profiles.insert(
            ModelType::ConvE,
            ModelProfile {
                model_type: ModelType::ConvE,
                strengths: vec![
                    "State-of-the-art accuracy".to_string(),
                    "Captures complex patterns".to_string(),
                    "Scalable to large graphs".to_string(),
                ],
                weaknesses: vec![
                    "Requires more computational resources".to_string(),
                    "More complex to tune".to_string(),
                    "Slower training".to_string(),
                ],
                best_for: vec![UseCaseType::LinkPrediction, UseCaseType::KGCompletion],
                complexity: 8,
                speed: 4,
                accuracy: 9,
                handles_sparse: false,
                handles_hierarchies: true,
                handles_complex_relations: true,
            },
        );

        // GNN profile
        model_profiles.insert(
            ModelType::GNN,
            ModelProfile {
                model_type: ModelType::GNN,
                strengths: vec![
                    "Leverages graph structure".to_string(),
                    "Good for node classification".to_string(),
                    "Captures neighborhood information".to_string(),
                ],
                weaknesses: vec![
                    "Computationally expensive".to_string(),
                    "Not ideal for very large graphs".to_string(),
                ],
                best_for: vec![
                    UseCaseType::EntityClassification,
                    UseCaseType::QuestionAnswering,
                ],
                complexity: 7,
                speed: 5,
                accuracy: 8,
                handles_sparse: false,
                handles_hierarchies: true,
                handles_complex_relations: true,
            },
        );

        // Transformer profile
        model_profiles.insert(
            ModelType::Transformer,
            ModelProfile {
                model_type: ModelType::Transformer,
                strengths: vec![
                    "Excellent for complex patterns".to_string(),
                    "State-of-the-art on many tasks".to_string(),
                    "Flexible architecture".to_string(),
                ],
                weaknesses: vec![
                    "Very computationally expensive".to_string(),
                    "Requires large amounts of data".to_string(),
                    "High memory usage".to_string(),
                ],
                best_for: vec![UseCaseType::QuestionAnswering, UseCaseType::GeneralPurpose],
                complexity: 9,
                speed: 3,
                accuracy: 9,
                handles_sparse: false,
                handles_hierarchies: true,
                handles_complex_relations: true,
            },
        );

        Self { model_profiles }
    }

    /// Recommend models for given dataset and use case
    pub fn recommend_models(
        &self,
        characteristics: &DatasetCharacteristics,
        use_case: UseCaseType,
    ) -> Result<Vec<ModelRecommendation>> {
        info!(
            "Recommending models for dataset with {} entities, {} relations, {} triples",
            characteristics.num_entities,
            characteristics.num_relations,
            characteristics.num_triples
        );

        let mut recommendations = Vec::new();

        for (model_type, profile) in &self.model_profiles {
            let score = self.calculate_suitability_score(profile, characteristics, use_case);

            if score > 0.3 {
                // Only include models with reasonable suitability
                let recommendation = self.create_recommendation(
                    *model_type,
                    profile,
                    characteristics,
                    score,
                    use_case,
                );
                recommendations.push(recommendation);
            }
        }

        // Sort by suitability score (descending)
        recommendations.sort_by(|a, b| {
            b.suitability_score
                .partial_cmp(&a.suitability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!("Generated {} model recommendations", recommendations.len());

        Ok(recommendations)
    }

    /// Calculate suitability score for a model (0.0 to 1.0)
    fn calculate_suitability_score(
        &self,
        profile: &ModelProfile,
        characteristics: &DatasetCharacteristics,
        use_case: UseCaseType,
    ) -> f64 {
        let mut score: f64 = 0.5; // Base score

        // Use case match (strong signal)
        if profile.best_for.contains(&use_case) {
            score += 0.3;
        }

        // Dataset characteristics match
        if characteristics.is_sparse && profile.handles_sparse {
            score += 0.1;
        }

        if characteristics.has_hierarchies && profile.handles_hierarchies {
            score += 0.1;
        }

        if characteristics.has_complex_relations && profile.handles_complex_relations {
            score += 0.1;
        }

        // Penalize complex models for small datasets
        if characteristics.num_triples < 10000 && profile.complexity > 6 {
            score -= 0.2;
        }

        // Penalize slow models for large datasets (unless accuracy is critical)
        if characteristics.num_triples > 100000 && profile.speed < 5 {
            score -= 0.1;
        }

        // Bonus for high accuracy models in link prediction
        if use_case == UseCaseType::LinkPrediction && profile.accuracy >= 8 {
            score += 0.1;
        }

        // Clamp to [0, 1]
        score.clamp(0.0, 1.0)
    }

    /// Create a detailed recommendation
    fn create_recommendation(
        &self,
        model_type: ModelType,
        profile: &ModelProfile,
        characteristics: &DatasetCharacteristics,
        score: f64,
        use_case: UseCaseType,
    ) -> ModelRecommendation {
        let recommended_dimensions = self.recommend_dimensions(characteristics, profile);

        let training_time =
            self.estimate_training_time(characteristics, profile, recommended_dimensions);

        let memory_requirement =
            self.estimate_memory_requirement(characteristics, recommended_dimensions);

        let reasoning = self.generate_reasoning(profile, characteristics, use_case);

        ModelRecommendation {
            model_type,
            suitability_score: score,
            reasoning,
            pros: profile.strengths.clone(),
            cons: profile.weaknesses.clone(),
            recommended_dimensions,
            estimated_training_time: training_time,
            memory_requirement,
        }
    }

    /// Recommend embedding dimensions based on dataset size
    fn recommend_dimensions(
        &self,
        characteristics: &DatasetCharacteristics,
        profile: &ModelProfile,
    ) -> usize {
        let base_dim = if characteristics.num_entities < 1000 {
            32
        } else if characteristics.num_entities < 10000 {
            64
        } else if characteristics.num_entities < 100000 {
            128
        } else {
            256
        };

        // Adjust for model complexity
        if profile.complexity > 7 {
            base_dim / 2 // Complex models can use lower dimensions
        } else {
            base_dim
        }
    }

    /// Estimate training time
    fn estimate_training_time(
        &self,
        characteristics: &DatasetCharacteristics,
        profile: &ModelProfile,
        _dimensions: usize,
    ) -> TrainingTime {
        let data_size_factor = characteristics.num_triples as f64 / 50000.0;
        let speed_factor = profile.speed as f64 / 10.0;

        let estimated_minutes = data_size_factor / speed_factor * 10.0;

        if estimated_minutes < 5.0 {
            TrainingTime::Fast
        } else if estimated_minutes < 30.0 {
            TrainingTime::Medium
        } else if estimated_minutes < 60.0 {
            TrainingTime::Slow
        } else {
            TrainingTime::VerySlow
        }
    }

    /// Estimate memory requirement
    fn estimate_memory_requirement(
        &self,
        characteristics: &DatasetCharacteristics,
        dimensions: usize,
    ) -> MemoryRequirement {
        let memory_mb = characteristics.estimated_memory_mb(dimensions);

        if memory_mb < 500.0 {
            MemoryRequirement::Low
        } else if memory_mb < 2000.0 {
            MemoryRequirement::Medium
        } else if memory_mb < 8000.0 {
            MemoryRequirement::High
        } else {
            MemoryRequirement::VeryHigh
        }
    }

    /// Generate reasoning explanation
    fn generate_reasoning(
        &self,
        profile: &ModelProfile,
        characteristics: &DatasetCharacteristics,
        use_case: UseCaseType,
    ) -> String {
        let mut reasons = Vec::new();

        if profile.best_for.contains(&use_case) {
            reasons.push(format!("Well-suited for {:?}", use_case));
        }

        if characteristics.is_sparse && profile.handles_sparse {
            reasons.push("Handles sparse graphs effectively".to_string());
        }

        if characteristics.has_hierarchies && profile.handles_hierarchies {
            reasons.push("Good for hierarchical structures".to_string());
        }

        if characteristics.has_complex_relations && profile.handles_complex_relations {
            reasons.push("Capable of modeling complex relations".to_string());
        }

        if profile.speed >= 8 {
            reasons.push("Fast training and inference".to_string());
        }

        if profile.accuracy >= 8 {
            reasons.push("High accuracy on benchmarks".to_string());
        }

        if reasons.is_empty() {
            "General-purpose model".to_string()
        } else {
            reasons.join("; ")
        }
    }

    /// Compare multiple models on the same criteria
    pub fn compare_models(
        &self,
        models: &[ModelType],
        characteristics: &DatasetCharacteristics,
    ) -> Result<ModelComparison> {
        if models.is_empty() {
            return Err(anyhow!("No models provided for comparison"));
        }

        let mut comparisons = HashMap::new();

        for model_type in models {
            if let Some(profile) = self.model_profiles.get(model_type) {
                let dimensions = self.recommend_dimensions(characteristics, profile);
                let training_time =
                    self.estimate_training_time(characteristics, profile, dimensions);
                let memory_req = self.estimate_memory_requirement(characteristics, dimensions);

                comparisons.insert(
                    *model_type,
                    ModelComparisonEntry {
                        model_type: *model_type,
                        complexity: profile.complexity,
                        speed: profile.speed,
                        accuracy: profile.accuracy,
                        recommended_dimensions: dimensions,
                        estimated_training_time: training_time,
                        memory_requirement: memory_req,
                    },
                );
            }
        }

        Ok(ModelComparison {
            models: comparisons,
            dataset_size: characteristics.num_triples,
        })
    }
}

/// Comparison result for multiple models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub models: HashMap<ModelType, ModelComparisonEntry>,
    pub dataset_size: usize,
}

/// Entry in model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonEntry {
    pub model_type: ModelType,
    pub complexity: u8,
    pub speed: u8,
    pub accuracy: u8,
    pub recommended_dimensions: usize,
    pub estimated_training_time: TrainingTime,
    pub memory_requirement: MemoryRequirement,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_characteristics_infer() {
        let chars = DatasetCharacteristics::infer(1000, 10, 5000);
        assert_eq!(chars.num_entities, 1000);
        assert_eq!(chars.num_relations, 10);
        assert_eq!(chars.num_triples, 5000);
        assert!(chars.avg_degree > 0.0);
    }

    #[test]
    fn test_dataset_density() {
        let chars = DatasetCharacteristics {
            num_entities: 100,
            num_relations: 5,
            num_triples: 500,
            avg_degree: 5.0,
            is_sparse: false,
            has_hierarchies: false,
            has_complex_relations: false,
            domain: None,
        };

        let density = chars.density();
        assert!(density > 0.0);
        assert!(density < 1.0);
    }

    #[test]
    fn test_model_selector_creation() {
        let selector = ModelSelector::new();
        assert!(!selector.model_profiles.is_empty());
        assert!(selector.model_profiles.contains_key(&ModelType::TransE));
        assert!(selector.model_profiles.contains_key(&ModelType::ComplEx));
    }

    #[test]
    fn test_model_recommendation() -> Result<()> {
        let selector = ModelSelector::new();
        let characteristics = DatasetCharacteristics::infer(10000, 50, 50000);

        let recommendations =
            selector.recommend_models(&characteristics, UseCaseType::LinkPrediction)?;

        assert!(!recommendations.is_empty());

        // Check that recommendations are sorted by score
        for i in 1..recommendations.len() {
            assert!(
                recommendations[i - 1].suitability_score >= recommendations[i].suitability_score
            );
        }

        Ok(())
    }

    #[test]
    fn test_model_comparison() -> Result<()> {
        let selector = ModelSelector::new();
        let characteristics = DatasetCharacteristics::infer(10000, 50, 50000);

        let models = vec![ModelType::TransE, ModelType::ComplEx, ModelType::RotatE];
        let comparison = selector.compare_models(&models, &characteristics)?;

        assert_eq!(comparison.models.len(), 3);
        assert!(comparison.models.contains_key(&ModelType::TransE));
        assert!(comparison.models.contains_key(&ModelType::ComplEx));
        assert!(comparison.models.contains_key(&ModelType::RotatE));

        Ok(())
    }

    #[test]
    fn test_small_dataset_recommendations() -> Result<()> {
        let selector = ModelSelector::new();
        let characteristics = DatasetCharacteristics::infer(100, 5, 500);

        let recommendations =
            selector.recommend_models(&characteristics, UseCaseType::GeneralPurpose)?;

        // For small datasets, simpler models should be preferred
        let top_model = &recommendations[0];
        assert!(top_model.recommended_dimensions <= 64);

        Ok(())
    }

    #[test]
    fn test_large_dataset_recommendations() -> Result<()> {
        let selector = ModelSelector::new();
        let characteristics = DatasetCharacteristics::infer(100000, 100, 500000);

        let recommendations =
            selector.recommend_models(&characteristics, UseCaseType::LinkPrediction)?;

        let top_model = &recommendations[0];
        assert!(top_model.recommended_dimensions >= 64);

        Ok(())
    }

    #[test]
    fn test_memory_estimation() {
        let characteristics = DatasetCharacteristics::infer(10000, 50, 50000);
        let memory_mb = characteristics.estimated_memory_mb(128);

        assert!(memory_mb > 0.0);
        assert!(memory_mb < 10000.0); // Sanity check
    }

    #[test]
    fn test_use_case_specific_recommendations() -> Result<()> {
        let selector = ModelSelector::new();
        let characteristics = DatasetCharacteristics::infer(10000, 50, 50000);

        let link_pred_recs =
            selector.recommend_models(&characteristics, UseCaseType::LinkPrediction)?;

        let similarity_recs =
            selector.recommend_models(&characteristics, UseCaseType::SimilaritySearch)?;

        // Different use cases may prefer different models
        assert!(!link_pred_recs.is_empty());
        assert!(!similarity_recs.is_empty());

        Ok(())
    }
}
