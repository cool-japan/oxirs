//! Module for vision-language-graph integration

use crate::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for vision-language-graph integration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VisionLanguageGraphConfig {
    pub base_config: ModelConfig,
    /// Vision encoder configuration
    pub vision_config: VisionEncoderConfig,
    /// Language encoder configuration  
    pub language_config: LanguageEncoderConfig,
    /// Graph encoder configuration
    pub graph_config: GraphEncoderConfig,
    /// Multi-modal transformer configuration
    pub transformer_config: MultiModalTransformerConfig,
    /// Meta-learning configuration
    pub meta_learning_config: MetaLearningConfig,
    /// Transfer learning configuration
    pub transfer_config: TransferLearningConfig,
    /// Joint training configuration
    pub joint_training_config: JointTrainingConfig,
}

/// Vision encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionEncoderConfig {
    /// Vision model architecture
    pub architecture: VisionArchitecture,
    /// Input image dimensions
    pub image_size: (usize, usize),
    /// Number of channels
    pub channels: usize,
    /// Patch size for vision transformer
    pub patch_size: (usize, usize),
    /// Vision embedding dimension
    pub vision_dim: usize,
    /// CNN backbone configuration
    pub cnn_config: CNNConfig,
    /// Vision transformer configuration
    pub vit_config: ViTConfig,
}

impl Default for VisionEncoderConfig {
    fn default() -> Self {
        Self {
            architecture: VisionArchitecture::VisionTransformer,
            image_size: (224, 224),
            channels: 3,
            patch_size: (16, 16),
            vision_dim: 768,
            cnn_config: CNNConfig::default(),
            vit_config: ViTConfig::default(),
        }
    }
}

/// Vision architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionArchitecture {
    /// Convolutional Neural Networks
    ResNet,
    EfficientNet,
    DenseNet,
    /// Vision Transformers
    VisionTransformer,
    DeiT,
    Swin,
    /// Hybrid architectures
    ConViT,
    CvT,
    /// CLIP-style encoders
    CLIPVision,
}

/// CNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNNConfig {
    /// Number of layers
    pub num_layers: usize,
    /// Filter sizes per layer
    pub filter_sizes: Vec<usize>,
    /// Stride sizes
    pub stride_sizes: Vec<usize>,
    /// Pooling configuration
    pub pooling: PoolingType,
    /// Normalization type
    pub normalization: NormalizationType,
}

impl Default for CNNConfig {
    fn default() -> Self {
        Self {
            num_layers: 4,
            filter_sizes: vec![64, 128, 256, 512],
            stride_sizes: vec![2, 2, 2, 2],
            pooling: PoolingType::AdaptiveAvgPool,
            normalization: NormalizationType::BatchNorm,
        }
    }
}

/// Vision Transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViTConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP hidden dimension
    pub mlp_dim: usize,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Position encoding type
    pub position_encoding: PositionEncodingType,
}

impl Default for ViTConfig {
    fn default() -> Self {
        Self {
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout_rate: 0.1,
            position_encoding: PositionEncodingType::Learnable,
        }
    }
}

/// Pooling types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingType {
    MaxPool,
    AvgPool,
    AdaptiveAvgPool,
    AdaptiveMaxPool,
    GlobalAvgPool,
}

/// Normalization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
}

/// Position encoding types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionEncodingType {
    Learnable,
    Sinusoidal,
    Relative,
    RoPE, // Rotary Position Embedding
}

/// Language encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageEncoderConfig {
    /// Language model architecture
    pub architecture: LanguageArchitecture,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Language embedding dimension
    pub language_dim: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Transformer configuration
    pub transformer_config: LanguageTransformerConfig,
}

impl Default for LanguageEncoderConfig {
    fn default() -> Self {
        Self {
            architecture: LanguageArchitecture::BERT,
            vocab_size: 30522,
            language_dim: 768,
            max_seq_length: 512,
            transformer_config: LanguageTransformerConfig::default(),
        }
    }
}

/// Language architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LanguageArchitecture {
    BERT,
    RoBERTa,
    DeBERTa,
    ELECTRA,
    GPT,
    T5,
    CLIP,
    ALIGN,
}

/// Language transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageTransformerConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub dropout_rate: f32,
    pub activation: ActivationFunction,
}

impl Default for LanguageTransformerConfig {
    fn default() -> Self {
        Self {
            num_layers: 12,
            num_heads: 12,
            hidden_dim: 768,
            intermediate_dim: 3072,
            dropout_rate: 0.1,
            activation: ActivationFunction::GELU,
        }
    }
}

/// Graph encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEncoderConfig {
    /// Graph neural network architecture
    pub architecture: GraphArchitecture,
    /// Node embedding dimension
    pub node_dim: usize,
    /// Edge embedding dimension
    pub edge_dim: usize,
    /// Graph embedding dimension
    pub graph_dim: usize,
    /// Number of GNN layers
    pub num_layers: usize,
    /// Aggregation function
    pub aggregation: AggregationFunction,
    /// Readout function
    pub readout: ReadoutFunction,
}

impl Default for GraphEncoderConfig {
    fn default() -> Self {
        Self {
            architecture: GraphArchitecture::GraphTransformer,
            node_dim: 256,
            edge_dim: 128,
            graph_dim: 768, // Match unified_dim for proper fusion
            num_layers: 6,
            aggregation: AggregationFunction::Attention,
            readout: ReadoutFunction::GlobalAttention,
        }
    }
}

/// Graph neural network architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphArchitecture {
    GCN,
    GraphSAGE,
    GAT,
    GraphTransformer,
    GIN,
    PNA,
    GPS, // General, Powerful, Scalable Graph Transformer
}

/// Aggregation functions for GNNs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Mean,
    Max,
    Sum,
    Attention,
    LSTM,
    GRU,
}

/// Readout functions for graph-level representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadoutFunction {
    GlobalMean,
    GlobalMax,
    GlobalSum,
    GlobalAttention,
    Set2Set,
    DiffPool,
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    GELU,
    Swish,
    Mish,
    ELU,
    LeakyReLU,
    Tanh,
    Sigmoid,
}

/// Multi-modal transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalTransformerConfig {
    /// Unified embedding dimension
    pub unified_dim: usize,
    /// Number of fusion layers
    pub num_fusion_layers: usize,
    /// Cross-attention configuration
    pub cross_attention_config: CrossAttentionConfig,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Positional encoding for modalities
    pub modality_encoding: ModalityEncoding,
}

impl Default for MultiModalTransformerConfig {
    fn default() -> Self {
        Self {
            unified_dim: 768,
            num_fusion_layers: 6,
            cross_attention_config: CrossAttentionConfig::default(),
            fusion_strategy: FusionStrategy::CrossAttention,
            modality_encoding: ModalityEncoding::Learnable,
        }
    }
}

/// Cross-attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Attention head dimension
    pub head_dim: usize,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Use residual connections
    pub use_residual: bool,
    /// Attention mechanism
    pub attention_mechanism: AttentionMechanism,
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 12,
            head_dim: 64,
            dropout_rate: 0.1,
            use_residual: true,
            attention_mechanism: AttentionMechanism::ScaledDotProduct,
        }
    }
}

/// Attention mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionMechanism {
    ScaledDotProduct,
    MultiHead,
    SparseAttention,
    LinearAttention,
    PerformerAttention,
    CoAttn, // Co-Attention
}

/// Fusion strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Early fusion (concatenation)
    EarlyFusion,
    /// Late fusion (separate processing)
    LateFusion,
    /// Cross-attention between modalities
    CrossAttention,
    /// Progressive fusion
    ProgressiveFusion,
    /// Adaptive fusion
    AdaptiveFusion,
    /// Tensor fusion
    TensorFusion,
}

/// Modality encoding types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityEncoding {
    /// No modality encoding
    None,
    /// Learnable modality embeddings
    Learnable,
    /// Fixed modality embeddings
    Fixed,
    /// Position-aware modality encoding
    PositionAware,
}

/// Meta-learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Meta-learning algorithm
    pub algorithm: MetaLearningAlgorithm,
    /// Support set size for few-shot learning
    pub support_set_size: usize,
    /// Query set size
    pub query_set_size: usize,
    /// Number of adaptation steps
    pub adaptation_steps: usize,
    /// Inner learning rate
    pub inner_lr: f32,
    /// Outer learning rate
    pub outer_lr: f32,
    /// Task-specific parameters
    pub task_specific_params: TaskSpecificParams,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            algorithm: MetaLearningAlgorithm::MAML,
            support_set_size: 5,
            query_set_size: 15,
            adaptation_steps: 5,
            inner_lr: 0.01,
            outer_lr: 0.001,
            task_specific_params: TaskSpecificParams::default(),
        }
    }
}

/// Meta-learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// First-Order MAML
    FOMAML,
    /// Reptile
    Reptile,
    /// Prototypical Networks
    ProtoNet,
    /// Relation Networks
    RelationNet,
    /// Memory-Augmented Neural Networks
    MANN,
    /// Meta-Learning with Adaptive Parameters
    AMAML,
}

/// Task-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpecificParams {
    /// Task categories
    pub task_categories: Vec<TaskCategory>,
    /// Domain-specific weights
    pub domain_weights: HashMap<String, f32>,
    /// Task difficulty adjustment
    pub difficulty_adjustment: bool,
}

impl Default for TaskSpecificParams {
    fn default() -> Self {
        let mut domain_weights = HashMap::new();
        domain_weights.insert("vision".to_string(), 1.0);
        domain_weights.insert("language".to_string(), 1.0);
        domain_weights.insert("graph".to_string(), 1.0);

        Self {
            task_categories: vec![
                TaskCategory::ImageCaptioning,
                TaskCategory::VisualQuestionAnswering,
                TaskCategory::GraphGrounding,
            ],
            domain_weights,
            difficulty_adjustment: true,
        }
    }
}

/// Task categories for multi-modal learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskCategory {
    /// Image captioning
    ImageCaptioning,
    /// Visual question answering
    VisualQuestionAnswering,
    /// Image-text retrieval
    ImageTextRetrieval,
    /// Graph-text alignment
    GraphTextAlignment,
    /// Graph grounding in images
    GraphGrounding,
    /// Multi-modal reasoning
    MultiModalReasoning,
    /// Cross-modal generation
    CrossModalGeneration,
}

/// Transfer learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningConfig {
    /// Transfer strategy
    pub strategy: TransferStrategy,
    /// Source domains
    pub source_domains: Vec<String>,
    /// Target domains
    pub target_domains: Vec<String>,
    /// Domain adaptation configuration
    pub domain_adaptation: DomainAdaptationConfig,
    /// Zero-shot configuration
    pub zero_shot_config: ZeroShotConfig,
    /// Few-shot configuration
    pub few_shot_config: FewShotConfig,
}

impl Default for TransferLearningConfig {
    fn default() -> Self {
        Self {
            strategy: TransferStrategy::ProgressiveTransfer,
            source_domains: vec!["general".to_string(), "imagenet".to_string()],
            target_domains: vec!["medical".to_string(), "scientific".to_string()],
            domain_adaptation: DomainAdaptationConfig::default(),
            zero_shot_config: ZeroShotConfig::default(),
            few_shot_config: FewShotConfig::default(),
        }
    }
}

/// Transfer learning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferStrategy {
    /// Fine-tuning all parameters
    FineTuning,
    /// Feature extraction (frozen backbone)
    FeatureExtraction,
    /// Progressive transfer
    ProgressiveTransfer,
    /// Multi-task learning
    MultiTaskLearning,
    /// Domain adaptation
    DomainAdaptation,
    /// Continual learning
    ContinualLearning,
}

/// Domain adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdaptationConfig {
    /// Adaptation method
    pub method: DomainAdaptationMethod,
    /// Adversarial training
    pub adversarial_training: bool,
    /// Gradient reversal layer
    pub gradient_reversal: bool,
    /// Domain classifier weight
    pub domain_classifier_weight: f32,
}

impl Default for DomainAdaptationConfig {
    fn default() -> Self {
        Self {
            method: DomainAdaptationMethod::DANN,
            adversarial_training: true,
            gradient_reversal: true,
            domain_classifier_weight: 0.1,
        }
    }
}

/// Domain adaptation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainAdaptationMethod {
    /// Domain-Adversarial Neural Networks
    DANN,
    /// Maximum Mean Discrepancy
    MMD,
    /// Correlation Alignment
    CORAL,
    /// Wasserstein Distance
    WDGRL,
    /// Conditional Domain Adaptation
    CDAN,
}

/// Zero-shot learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotConfig {
    /// Zero-shot method
    pub method: ZeroShotMethod,
    /// Semantic space dimension
    pub semantic_dim: usize,
    /// Use auxiliary attributes
    pub use_attributes: bool,
    /// Attribute dimension
    pub attribute_dim: usize,
}

impl Default for ZeroShotConfig {
    fn default() -> Self {
        Self {
            method: ZeroShotMethod::CLIP,
            semantic_dim: 512,
            use_attributes: true,
            attribute_dim: 256,
        }
    }
}

/// Zero-shot learning methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZeroShotMethod {
    /// CLIP-style contrastive learning
    CLIP,
    /// ALIGN-style learning
    ALIGN,
    /// Attribute-based learning
    Attribute,
    /// Semantic embedding
    SemanticEmbedding,
    /// Knowledge graph guided
    KnowledgeGuided,
}

/// Few-shot learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotConfig {
    /// Few-shot method
    pub method: FewShotMethod,
    /// Number of shots
    pub num_shots: usize,
    /// Episode configuration
    pub episode_config: EpisodeConfig,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            method: FewShotMethod::ProtoNet,
            num_shots: 5,
            episode_config: EpisodeConfig::default(),
        }
    }
}

/// Few-shot learning methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FewShotMethod {
    /// Prototypical Networks
    ProtoNet,
    /// Matching Networks
    MatchingNet,
    /// Relation Networks
    RelationNet,
    /// Meta-learning approaches
    MetaLearning,
}

/// Episode configuration for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeConfig {
    /// Number of classes per episode
    pub num_classes: usize,
    /// Support samples per class
    pub support_per_class: usize,
    /// Query samples per class
    pub query_per_class: usize,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            num_classes: 5,
            support_per_class: 5,
            query_per_class: 15,
        }
    }
}

/// Joint training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointTrainingConfig {
    /// Training objectives
    pub objectives: Vec<TrainingObjective>,
    /// Objective weights
    pub objective_weights: HashMap<String, f32>,
    /// Curriculum learning
    pub curriculum_learning: bool,
    /// Progressive training
    pub progressive_training: bool,
}

impl Default for JointTrainingConfig {
    fn default() -> Self {
        let mut objective_weights = HashMap::new();
        objective_weights.insert("vision_language_alignment".to_string(), 1.0);
        objective_weights.insert("language_graph_alignment".to_string(), 0.8);
        objective_weights.insert("vision_graph_alignment".to_string(), 0.6);
        objective_weights.insert("tri_modal_alignment".to_string(), 1.2);

        Self {
            objectives: vec![
                TrainingObjective::ContrastiveLearning,
                TrainingObjective::MaskedLanguageModeling,
                TrainingObjective::ImageTextMatching,
                TrainingObjective::GraphAlignment,
            ],
            objective_weights,
            curriculum_learning: true,
            progressive_training: true,
        }
    }
}

/// Training objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingObjective {
    /// Contrastive learning between modalities
    ContrastiveLearning,
    /// Masked language modeling
    MaskedLanguageModeling,
    /// Image-text matching
    ImageTextMatching,
    /// Graph-text alignment
    GraphAlignment,
    /// Visual question answering
    VisualQuestionAnswering,
    /// Image captioning
    ImageCaptioning,
    /// Graph reasoning
    GraphReasoning,
    /// Multi-modal reasoning
    MultiModalReasoning,
}
