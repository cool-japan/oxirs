//! Configuration types for multi-modal embeddings

use crate::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cross-modal alignment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    pub base_config: ModelConfig,
    /// Text embedding dimension
    pub text_dim: usize,
    /// Knowledge graph embedding dimension
    pub kg_dim: usize,
    /// Unified embedding dimension
    pub unified_dim: usize,
    /// Alignment objective type
    pub alignment_objective: AlignmentObjective,
    /// Contrastive learning parameters
    pub contrastive_config: ContrastiveConfig,
    /// Multi-task learning weights
    pub task_weights: HashMap<String, f32>,
    /// Cross-domain transfer settings
    pub cross_domain_config: CrossDomainConfig,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        let mut task_weights = HashMap::new();
        task_weights.insert("text_kg_alignment".to_string(), 1.0);
        task_weights.insert("entity_description".to_string(), 0.8);
        task_weights.insert("property_text".to_string(), 0.6);
        task_weights.insert("multilingual".to_string(), 0.4);

        Self {
            base_config: ModelConfig::default(),
            text_dim: 768,
            kg_dim: 128,
            unified_dim: 512,
            alignment_objective: AlignmentObjective::ContrastiveLearning,
            contrastive_config: ContrastiveConfig::default(),
            task_weights,
            cross_domain_config: CrossDomainConfig::default(),
        }
    }
}

/// Alignment objective types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentObjective {
    /// Contrastive learning for positive/negative pairs
    ContrastiveLearning,
    /// Mutual information maximization
    MutualInformation,
    /// Adversarial alignment with discriminator
    AdversarialAlignment,
    /// Multi-task learning with shared representations
    MultiTaskLearning,
    /// Self-supervised objectives
    SelfSupervised,
    /// Meta-learning for few-shot adaptation
    MetaLearning,
}

/// Contrastive learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveConfig {
    /// Temperature parameter for contrastive loss
    pub temperature: f32,
    /// Number of negative samples
    pub negative_samples: usize,
    /// Hard negative mining
    pub hard_negative_mining: bool,
    /// Margin for triplet loss
    pub margin: f32,
    /// Use InfoNCE loss
    pub use_info_nce: bool,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            negative_samples: 64,
            hard_negative_mining: true,
            margin: 0.2,
            use_info_nce: true,
        }
    }
}

/// Cross-domain transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainConfig {
    /// Enable domain adaptation
    pub enable_domain_adaptation: bool,
    /// Source domains for transfer learning
    pub source_domains: Vec<String>,
    /// Target domains
    pub target_domains: Vec<String>,
    /// Domain adversarial training
    pub domain_adversarial: bool,
    /// Gradual domain adaptation
    pub gradual_adaptation: bool,
}

impl Default for CrossDomainConfig {
    fn default() -> Self {
        Self {
            enable_domain_adaptation: true,
            source_domains: vec!["general".to_string(), "scientific".to_string()],
            target_domains: vec!["biomedical".to_string(), "legal".to_string()],
            domain_adversarial: false,
            gradual_adaptation: true,
        }
    }
}
