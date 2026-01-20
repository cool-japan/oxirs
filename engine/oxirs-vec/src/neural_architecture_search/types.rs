//! Basic types and enums for Neural Architecture Search

use serde::{Deserialize, Serialize};

/// Neural network layer types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    Linear { input_dim: usize, output_dim: usize },
    Conv1D { filters: usize, kernel_size: usize, stride: usize },
    LSTM { hidden_size: usize, num_layers: usize },
    GRU { hidden_size: usize, num_layers: usize },
    Transformer { d_model: usize, num_heads: usize, num_layers: usize },
    Attention { num_heads: usize, hidden_dim: usize },
    Dropout { rate: f64 },
    BatchNorm,
    LayerNorm,
    Residual,
    MoE { num_experts: usize, expert_dim: usize }, // Mixture of Experts
}

/// Activation function types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    Mish,
    Tanh,
    Sigmoid,
    LeakyReLU { negative_slope: f64 },
    ELU { alpha: f64 },
}

/// Normalization types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    GroupNorm { num_groups: usize },
    InstanceNorm,
    RMSNorm,
    None,
}

/// Attention mechanism types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttentionType {
    MultiHead { num_heads: usize },
    SingleHead,
    SelfAttention,
    CrossAttention,
    SparseAttention { sparsity_ratio: f64 },
    LocalAttention { window_size: usize },
}

/// Skip connection patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SkipPattern {
    None,
    Residual,
    DenseNet,
    Highway,
    SENet { reduction: usize },
}

/// Search strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    EvolutionaryAlgorithm,
    ReinforcementLearning,
    BayesianOptimization,
    RandomSearch,
    GridSearch,
    GradientBased,
    Hybrid {
        strategies: Vec<SearchStrategy>,
        switching_criteria: SwitchingCriteria,
    },
}

/// Selection types for evolutionary algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionType {
    Tournament { size: usize },
    Roulette,
    Rank,
    Elitist { elite_ratio: f64 },
}

/// Mutation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationStrategy {
    Uniform,
    Gaussian { sigma: f64 },
    Adaptive,
}

/// Crossover strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossoverStrategy {
    SinglePoint,
    TwoPoint,
    Uniform,
    Arithmetic,
}

/// Controller types for reinforcement learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControllerType {
    LSTM,
    Transformer,
    MLP,
}

/// Reward function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardFunction {
    Accuracy,
    F1Score,
    MultiObjective,
    Custom(String),
}

/// Exploration strategies for RL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    EpsilonGreedy { epsilon: f64 },
    Boltzmann { temperature: f64 },
    UCB { c: f64 },
    Thompson,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    MaxValueEntropy,
}

/// Surrogate model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurrogateModel {
    GaussianProcess,
    RandomForest,
    NeuralNetwork,
    Ensemble,
}

/// Architecture initialization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitializationStrategy {
    Random,
    Pretrained,
    Progressive,
    HeuristicBased,
}

/// Switching criteria for hybrid strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwitchingCriteria {
    Performance { threshold: f64 },
    Iteration { interval: usize },
    Stagnation { patience: usize },
    Resource { limit: f64 },
}

/// Stage transition criteria for progressive search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageTransitionCriteria {
    Performance { threshold: f64 },
    Convergence { tolerance: f64 },
    TimeLimit { minutes: usize },
    Generation { count: usize },
}

/// Evaluation metrics for architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationMetric {
    Accuracy,
    F1Score,
    BLEU,
    ROUGE,
    Perplexity,
    Custom(String),
}

/// Early stopping modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EarlyStoppingMode {
    Patience,
    RelativeImprovement,
    AbsoluteThreshold,
}

/// Data types for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Text,
    Image,
    Audio,
    Tabular,
    Graph,
    TimeSeries,
    Multimodal,
}

/// Task types for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Regression,
    LanguageModeling,
    MachineTranslation,
    QuestionAnswering,
    SentimentAnalysis,
    NamedEntityRecognition,
    ImageClassification,
    ObjectDetection,
    Custom(String),
}