//! Types for Joint Embedding Spaces
//!
//! Structs, enums, space config types, alignment types, and transfer types.

use crate::{cross_modal_embeddings::Modality, Vector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for joint embedding space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointEmbeddingConfig {
    /// Dimension of the joint embedding space
    pub joint_dim: usize,
    /// Temperature parameter for contrastive learning
    pub temperature: f32,
    /// Learning rate for alignment optimization
    pub learning_rate: f32,
    /// Margin for triplet loss
    pub margin: f32,
    /// Enable contrastive learning
    pub contrastive_learning: bool,
    /// Enable triplet loss
    pub triplet_loss: bool,
    /// Enable hard negative mining
    pub hard_negative_mining: bool,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of negative samples per positive
    pub negative_samples: usize,
    /// Enable curriculum learning
    pub curriculum_learning: bool,
    /// Weight decay for regularization
    pub weight_decay: f32,
    /// Gradient clipping threshold
    pub gradient_clip: f32,
    /// Enable domain adaptation
    pub domain_adaptation: bool,
    /// Cross-modal alignment strength
    pub alignment_strength: f32,
    /// Enable self-supervised learning
    pub self_supervised: bool,
}

impl Default for JointEmbeddingConfig {
    fn default() -> Self {
        Self {
            joint_dim: 512,
            temperature: 0.07,
            learning_rate: 1e-4,
            margin: 0.2,
            contrastive_learning: true,
            triplet_loss: false,
            hard_negative_mining: true,
            batch_size: 256,
            negative_samples: 5,
            curriculum_learning: false,
            weight_decay: 1e-4,
            gradient_clip: 1.0,
            domain_adaptation: true,
            alignment_strength: 1.0,
            self_supervised: false,
        }
    }
}

/// Type alias for contrastive pairs result
pub type ContrastivePairs = (
    Vec<(Modality, Vector, Modality, Vector)>,
    Vec<(Modality, Vector, Modality, Vector)>,
);

/// Alignment pair for caching and training
#[derive(Debug, Clone)]
pub struct AlignmentPair {
    pub(crate) modality1: Modality,
    pub(crate) modality2: Modality,
    pub(crate) embedding1: Vector,
    pub(crate) embedding2: Vector,
    pub(crate) similarity: f32,
    pub(crate) confidence: f32,
    pub(crate) timestamp: std::time::SystemTime,
}

/// Training statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct TrainingStatistics {
    pub(crate) total_samples: u64,
    pub(crate) positive_pairs: u64,
    pub(crate) negative_pairs: u64,
    pub(crate) average_loss: f32,
    pub(crate) average_similarity: f32,
    pub(crate) convergence_rate: f32,
    pub(crate) alignment_accuracy: f32,
    pub(crate) cross_modal_retrieval_acc: HashMap<(Modality, Modality), f32>,
    pub(crate) training_epochs: u32,
    pub(crate) last_improvement: u32,
}

/// Activation functions for projectors
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    Swish,
    Mish,
    LeakyReLU(f32),
}

/// Schedule type for temperature decay
#[derive(Debug, Clone, Copy)]
pub enum ScheduleType {
    Linear,
    Exponential,
    Cosine,
    Warmup,
}

/// Temperature scheduler for contrastive learning
#[derive(Debug, Clone)]
pub struct TemperatureScheduler {
    pub(crate) initial_temperature: f32,
    pub(crate) final_temperature: f32,
    pub(crate) decay_steps: usize,
    pub(crate) current_step: usize,
    pub(crate) schedule_type: ScheduleType,
}

/// Domain statistics for domain adaptation
#[derive(Debug, Clone, Default)]
pub struct DomainStatistics {
    pub(crate) mean: Vec<f32>,
    pub(crate) variance: Vec<f32>,
    pub(crate) sample_count: usize,
    pub(crate) feature_statistics: HashMap<String, f32>,
}

/// Domain classifier for adversarial adaptation
#[derive(Debug, Clone)]
pub struct DomainClassifier {
    pub(crate) weights: Vec<Vec<f32>>,
    pub(crate) bias: Vec<f32>,
    pub(crate) accuracy: f32,
}

/// Domain adaptation module for cross-domain alignment
#[derive(Debug, Clone)]
pub struct DomainAdapter {
    pub(crate) source_stats: DomainStatistics,
    pub(crate) target_stats: DomainStatistics,
    pub(crate) adaptation_weights: Vec<f32>,
    pub(crate) domain_classifier: Option<DomainClassifier>,
    pub(crate) adaptation_strength: f32,
}

/// Linear projector for transforming embeddings to joint space
#[derive(Debug, Clone)]
pub struct LinearProjector {
    pub(crate) weights: Vec<Vec<f32>>,
    pub(crate) bias: Vec<f32>,
    pub(crate) input_dim: usize,
    pub(crate) output_dim: usize,
    pub(crate) dropout_rate: f32,
    pub(crate) activation: ActivationFunction,
}

/// Cross-modal attention mechanism for joint spaces
#[derive(Debug, Clone)]
pub struct CrossModalAttention {
    pub(crate) query_projector: LinearProjector,
    pub(crate) key_projector: LinearProjector,
    pub(crate) value_projector: LinearProjector,
    pub(crate) output_projector: LinearProjector,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) dropout_rate: f32,
    pub(crate) scale: f32,
    pub(crate) enable_relative_pos: bool,
}

/// Learning rate schedule for contrastive optimizer
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule {
    Constant,
    StepDecay { step_size: usize, gamma: f32 },
    ExponentialDecay { gamma: f32 },
    CosineAnnealing { min_lr: f32, max_epochs: usize },
}

/// Contrastive optimizer for alignment training
#[derive(Debug, Clone)]
pub struct ContrastiveOptimizer {
    pub(crate) learning_rate: f32,
    pub(crate) momentum: f32,
    pub(crate) weight_decay: f32,
    pub(crate) gradient_history: HashMap<String, Vec<f32>>,
    pub(crate) adaptive_lr: bool,
    pub(crate) lr_schedule: LearningRateSchedule,
}

/// Text augmentation strategies
#[derive(Debug, Clone)]
pub enum TextAugmentation {
    RandomWordDropout(f32),
    Paraphrasing,
    BackTranslation,
    SynonymReplacement(f32),
    ContextualAugmentation,
}

/// Image augmentation strategies
#[derive(Debug, Clone)]
pub enum ImageAugmentation {
    RandomCrop {
        size: (u32, u32),
    },
    RandomFlip {
        horizontal: bool,
        vertical: bool,
    },
    ColorJitter {
        brightness: f32,
        contrast: f32,
        saturation: f32,
    },
    RandomRotation {
        max_angle: f32,
    },
    GaussianBlur {
        sigma: f32,
    },
}

/// Audio augmentation strategies
#[derive(Debug, Clone)]
pub enum AudioAugmentation {
    TimeStretch { factor: f32 },
    PitchShift { semitones: f32 },
    AddNoise { snr_db: f32 },
    FrequencyMasking { max_freq_mask: f32 },
    TimeMasking { max_time_mask: f32 },
}

/// Data augmentation for improved generalization
#[derive(Debug, Clone)]
pub struct DataAugmentation {
    pub(crate) text_augmentations: Vec<TextAugmentation>,
    pub(crate) image_augmentations: Vec<ImageAugmentation>,
    pub(crate) audio_augmentations: Vec<AudioAugmentation>,
    pub(crate) cross_modal_mixup: bool,
    pub(crate) augmentation_probability: f32,
}

/// Difficulty schedule for curriculum learning
#[derive(Debug, Clone)]
pub enum DifficultySchedule {
    Linear { start: f32, end: f32, epochs: usize },
    Exponential { base: f32, scale: f32 },
    Adaptive { improvement_threshold: f32 },
}

/// Pacing function for curriculum learning
#[derive(Debug, Clone)]
pub enum PacingFunction {
    Root,
    Linear,
    Logarithmic,
    Polynomial(f32),
}

/// Curriculum learning for progressive training
#[derive(Debug, Clone)]
pub struct CurriculumLearning {
    pub(crate) enabled: bool,
    pub(crate) current_difficulty: f32,
    pub(crate) difficulty_schedule: DifficultySchedule,
    pub(crate) pacing_function: PacingFunction,
    pub(crate) competence_threshold: f32,
}

/// Joint embedding space handle shared via Arc
pub(crate) type AlignmentCacheRef = Arc<parking_lot::RwLock<HashMap<String, AlignmentPair>>>;
pub(crate) type TrainingStatsRef = Arc<parking_lot::RwLock<TrainingStatistics>>;
