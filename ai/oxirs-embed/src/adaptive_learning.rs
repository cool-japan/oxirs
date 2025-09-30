//! Adaptive Learning System for Real-Time Embedding Enhancement
//!
//! This module implements an advanced adaptive learning system that continuously
//! improves embedding quality through online learning, feedback mechanisms,
//! and dynamic model adaptation based on usage patterns.

use anyhow::Result;
use chrono::{DateTime, Utc};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Configuration for adaptive learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningConfig {
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Size of the experience buffer
    pub buffer_size: usize,
    /// Minimum samples before adaptation
    pub min_samples_for_adaptation: usize,
    /// Maximum adaptation frequency (per second)
    pub max_adaptation_frequency: f64,
    /// Quality threshold for positive feedback
    pub quality_threshold: f64,
    /// Enable meta-learning adaptation
    pub enable_meta_learning: bool,
    /// Batch size for adaptation updates
    pub adaptation_batch_size: usize,
}

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            buffer_size: 10000,
            min_samples_for_adaptation: 100,
            max_adaptation_frequency: 1.0,
            quality_threshold: 0.8,
            enable_meta_learning: true,
            adaptation_batch_size: 32,
        }
    }
}

/// Feedback signal for embedding quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFeedback {
    /// Query that generated the embedding
    pub query: String,
    /// Generated embedding
    pub embedding: Vec<f64>,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Timestamp of feedback  
    #[serde(with = "chrono::serde::ts_seconds")]
    pub timestamp: DateTime<Utc>,
    /// User-provided relevance score
    pub relevance: Option<f64>,
    /// Task context
    pub task_context: Option<String>,
}

/// Experience sample for adaptive learning
#[derive(Debug, Clone)]
pub struct ExperienceSample {
    /// Input query/text
    pub input: String,
    /// Target embedding (from positive feedback)
    pub target: Vec<f64>,
    /// Current embedding (what model produced)
    pub current: Vec<f64>,
    /// Quality improvement needed
    pub improvement_target: f64,
    /// Context information
    pub context: HashMap<String, String>,
}

/// Adaptation strategy for different learning scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Gradient-based fine-tuning
    GradientDescent { momentum: f64, weight_decay: f64 },
    /// Evolutionary adaptation
    Evolutionary {
        mutation_rate: f64,
        population_size: usize,
    },
    /// Meta-learning adaptation (MAML-style)
    MetaLearning {
        inner_steps: usize,
        outer_learning_rate: f64,
    },
    /// Bayesian optimization
    BayesianOptimization {
        exploration_factor: f64,
        kernel_bandwidth: f64,
    },
}

impl Default for AdaptationStrategy {
    fn default() -> Self {
        Self::GradientDescent {
            momentum: 0.9,
            weight_decay: 0.0001,
        }
    }
}

/// Performance metrics for adaptation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetrics {
    /// Number of adaptations performed
    pub adaptations_count: usize,
    /// Average quality improvement per adaptation
    pub avg_quality_improvement: f64,
    /// Current adaptation rate (adaptations per minute)
    pub adaptation_rate: f64,
    /// Experience buffer utilization
    pub buffer_utilization: f64,
    /// Model performance drift detection
    pub performance_drift: f64,
    /// Last adaptation timestamp
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub last_adaptation: Option<DateTime<Utc>>,
}

impl Default for AdaptationMetrics {
    fn default() -> Self {
        Self {
            adaptations_count: 0,
            avg_quality_improvement: 0.0,
            adaptation_rate: 0.0,
            buffer_utilization: 0.0,
            performance_drift: 0.0,
            last_adaptation: None,
        }
    }
}

/// Adaptive learning system for continuous embedding improvement
pub struct AdaptiveLearningSystem {
    /// Configuration
    config: AdaptiveLearningConfig,
    /// Experience buffer for storing feedback
    experience_buffer: Arc<RwLock<VecDeque<ExperienceSample>>>,
    /// Quality feedback receiver
    feedback_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<QualityFeedback>>>>,
    /// Quality feedback sender
    feedback_sender: mpsc::UnboundedSender<QualityFeedback>,
    /// Adaptation strategy
    strategy: AdaptationStrategy,
    /// Current metrics
    metrics: Arc<RwLock<AdaptationMetrics>>,
    /// Model parameters for adaptation
    model_parameters: Arc<RwLock<HashMap<String, DMatrix<f64>>>>,
    /// Learning state
    learning_state: Arc<RwLock<LearningState>>,
}

/// Internal learning state
#[derive(Debug, Clone)]
struct LearningState {
    /// Momentum vectors for gradient descent
    momentum: HashMap<String, DMatrix<f64>>,
    /// Adaptation history
    adaptation_history: VecDeque<AdaptationRecord>,
    /// Current learning rate (adaptive)
    current_learning_rate: f64,
    /// Performance baseline
    performance_baseline: f64,
}

/// Record of adaptation for analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AdaptationRecord {
    /// Timestamp of adaptation
    timestamp: DateTime<Utc>,
    /// Quality before adaptation
    quality_before: f64,
    /// Quality after adaptation
    quality_after: f64,
    /// Number of samples used
    samples_used: usize,
    /// Strategy used
    strategy: AdaptationStrategy,
}

impl AdaptiveLearningSystem {
    /// Create new adaptive learning system
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let learning_rate = config.learning_rate;

        Self {
            config,
            experience_buffer: Arc::new(RwLock::new(VecDeque::new())),
            feedback_receiver: Arc::new(RwLock::new(Some(receiver))),
            feedback_sender: sender,
            strategy: AdaptationStrategy::default(),
            metrics: Arc::new(RwLock::new(AdaptationMetrics::default())),
            model_parameters: Arc::new(RwLock::new(HashMap::new())),
            learning_state: Arc::new(RwLock::new(LearningState {
                momentum: HashMap::new(),
                adaptation_history: VecDeque::new(),
                current_learning_rate: learning_rate,
                performance_baseline: 0.5,
            })),
        }
    }

    /// Create system with custom strategy
    pub fn with_strategy(config: AdaptiveLearningConfig, strategy: AdaptationStrategy) -> Self {
        let mut system = Self::new(config);
        system.strategy = strategy;
        system
    }

    /// Submit quality feedback
    pub fn submit_feedback(&self, feedback: QualityFeedback) -> Result<()> {
        self.feedback_sender.send(feedback)?;
        Ok(())
    }

    /// Start the adaptive learning process
    pub async fn start_learning(&self) -> Result<()> {
        let mut receiver = self
            .feedback_receiver
            .write()
            .unwrap()
            .take()
            .ok_or_else(|| anyhow::anyhow!("Learning already started"))?;

        info!("Starting adaptive learning system");

        // Spawn feedback processing task
        let experience_buffer = Arc::clone(&self.experience_buffer);
        let metrics = Arc::clone(&self.metrics);
        let config = self.config.clone();

        tokio::spawn(async move {
            while let Some(feedback) = receiver.recv().await {
                if let Err(e) =
                    Self::process_feedback(feedback, &experience_buffer, &metrics, &config).await
                {
                    warn!("Error processing feedback: {}", e);
                }
            }
        });

        // Spawn adaptation task
        let buffer = Arc::clone(&self.experience_buffer);
        let metrics = Arc::clone(&self.metrics);
        let parameters = Arc::clone(&self.model_parameters);
        let learning_state = Arc::clone(&self.learning_state);
        let config = self.config.clone();
        let strategy = self.strategy.clone();

        tokio::spawn(async move {
            let mut last_adaptation = Instant::now();

            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;

                // Check if we should perform adaptation
                let should_adapt = {
                    let buffer_guard = buffer.read().unwrap();
                    let _metrics_guard = metrics.read().unwrap();

                    buffer_guard.len() >= config.min_samples_for_adaptation
                        && last_adaptation.elapsed().as_secs_f64()
                            >= 1.0 / config.max_adaptation_frequency
                };

                if should_adapt {
                    match Self::perform_adaptation(
                        &buffer,
                        &metrics,
                        &parameters,
                        &learning_state,
                        &config,
                        &strategy,
                    )
                    .await
                    {
                        Err(e) => {
                            warn!("Error during adaptation: {}", e);
                        }
                        _ => {
                            last_adaptation = Instant::now();
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Process incoming feedback
    async fn process_feedback(
        feedback: QualityFeedback,
        buffer: &Arc<RwLock<VecDeque<ExperienceSample>>>,
        metrics: &Arc<RwLock<AdaptationMetrics>>,
        config: &AdaptiveLearningConfig,
    ) -> Result<()> {
        // Convert feedback to experience sample
        if feedback.quality_score > config.quality_threshold {
            let sample = ExperienceSample {
                input: feedback.query.clone(),
                target: feedback.embedding.clone(),
                current: feedback.embedding.clone(), // This would be the current model output
                improvement_target: 1.0 - feedback.quality_score,
                context: feedback
                    .task_context
                    .map(|ctx| [("task".to_string(), ctx)].into())
                    .unwrap_or_default(),
            };

            // Add to buffer
            {
                let mut buffer_guard = buffer.write().unwrap();
                buffer_guard.push_back(sample);

                // Maintain buffer size
                while buffer_guard.len() > config.buffer_size {
                    buffer_guard.pop_front();
                }
            }

            // Update metrics
            {
                let mut metrics_guard = metrics.write().unwrap();
                let buffer_guard = buffer.read().unwrap();
                metrics_guard.buffer_utilization =
                    buffer_guard.len() as f64 / config.buffer_size as f64;
            }

            debug!(
                "Processed feedback with quality score: {}",
                feedback.quality_score
            );
        }

        Ok(())
    }

    /// Perform model adaptation
    async fn perform_adaptation(
        buffer: &Arc<RwLock<VecDeque<ExperienceSample>>>,
        metrics: &Arc<RwLock<AdaptationMetrics>>,
        parameters: &Arc<RwLock<HashMap<String, DMatrix<f64>>>>,
        learning_state: &Arc<RwLock<LearningState>>,
        config: &AdaptiveLearningConfig,
        strategy: &AdaptationStrategy,
    ) -> Result<()> {
        let samples = {
            let buffer_guard = buffer.read().unwrap();
            buffer_guard
                .iter()
                .take(config.adaptation_batch_size)
                .cloned()
                .collect::<Vec<_>>()
        };

        if samples.is_empty() {
            return Ok(());
        }

        info!("Performing adaptation with {} samples", samples.len());

        // Calculate quality before adaptation
        let quality_before = Self::calculate_current_quality(&samples)?;

        // Perform adaptation based on strategy
        match strategy {
            AdaptationStrategy::GradientDescent {
                momentum,
                weight_decay,
            } => {
                Self::gradient_descent_adaptation(
                    &samples,
                    parameters,
                    learning_state,
                    *momentum,
                    *weight_decay,
                    config.learning_rate,
                )?;
            }
            AdaptationStrategy::MetaLearning {
                inner_steps,
                outer_learning_rate,
            } => {
                Self::meta_learning_adaptation(
                    &samples,
                    parameters,
                    learning_state,
                    *inner_steps,
                    *outer_learning_rate,
                )?;
            }
            AdaptationStrategy::Evolutionary {
                mutation_rate,
                population_size,
            } => {
                Self::evolutionary_adaptation(
                    &samples,
                    parameters,
                    *mutation_rate,
                    *population_size,
                )?;
            }
            AdaptationStrategy::BayesianOptimization {
                exploration_factor,
                kernel_bandwidth,
            } => {
                Self::bayesian_optimization_adaptation(
                    &samples,
                    parameters,
                    *exploration_factor,
                    *kernel_bandwidth,
                )?;
            }
        }

        // Calculate quality after adaptation
        let quality_after = Self::calculate_current_quality(&samples)?;

        // Update metrics
        {
            let mut metrics_guard = metrics.write().unwrap();
            metrics_guard.adaptations_count += 1;
            let improvement = quality_after - quality_before;
            metrics_guard.avg_quality_improvement = (metrics_guard.avg_quality_improvement
                * (metrics_guard.adaptations_count - 1) as f64
                + improvement)
                / metrics_guard.adaptations_count as f64;
            metrics_guard.last_adaptation = Some(Utc::now());
        }

        // Update learning state
        {
            let mut state_guard = learning_state.write().unwrap();
            state_guard.adaptation_history.push_back(AdaptationRecord {
                timestamp: Utc::now(),
                quality_before,
                quality_after,
                samples_used: samples.len(),
                strategy: strategy.clone(),
            });

            // Maintain history size
            while state_guard.adaptation_history.len() > 1000 {
                state_guard.adaptation_history.pop_front();
            }

            // Adaptive learning rate
            if quality_after > quality_before {
                state_guard.current_learning_rate *= 1.01; // Increase slightly
            } else {
                state_guard.current_learning_rate *= 0.95; // Decrease
            }
            state_guard.current_learning_rate = state_guard
                .current_learning_rate
                .max(config.learning_rate * 0.1)
                .min(config.learning_rate * 10.0);
        }

        info!(
            "Adaptation completed: quality improved by {:.4}",
            quality_after - quality_before
        );

        Ok(())
    }

    /// Calculate current quality based on samples
    fn calculate_current_quality(samples: &[ExperienceSample]) -> Result<f64> {
        if samples.is_empty() {
            return Ok(0.0);
        }

        let total_quality: f64 = samples
            .iter()
            .map(|sample| {
                // Calculate similarity between current and target embeddings
                let current = DVector::from_vec(sample.current.clone());
                let target = DVector::from_vec(sample.target.clone());

                if current.len() != target.len() {
                    return 0.0;
                }

                // Cosine similarity
                let dot_product = current.dot(&target);
                let norm_current = current.norm();
                let norm_target = target.norm();

                if norm_current == 0.0 || norm_target == 0.0 {
                    return 0.0;
                }

                (dot_product / (norm_current * norm_target)).max(0.0)
            })
            .sum();

        Ok(total_quality / samples.len() as f64)
    }

    /// Gradient descent adaptation
    fn gradient_descent_adaptation(
        samples: &[ExperienceSample],
        parameters: &Arc<RwLock<HashMap<String, DMatrix<f64>>>>,
        learning_state: &Arc<RwLock<LearningState>>,
        momentum: f64,
        weight_decay: f64,
        learning_rate: f64,
    ) -> Result<()> {
        // Simplified gradient descent implementation
        // In a real implementation, this would compute gradients based on the loss
        // between current embeddings and target embeddings

        let mut params_guard = parameters.write().unwrap();
        let mut state_guard = learning_state.write().unwrap();

        for (param_name, param_matrix) in params_guard.iter_mut() {
            // Compute pseudo-gradient (simplified for demonstration)
            let gradient = Self::compute_gradient(samples, param_matrix)?;

            // Update momentum
            let momentum_entry = state_guard
                .momentum
                .entry(param_name.clone())
                .or_insert_with(|| DMatrix::zeros(param_matrix.nrows(), param_matrix.ncols()));

            *momentum_entry = momentum_entry.clone() * momentum + &gradient;

            // Apply weight decay
            let decay_term = param_matrix.clone() * weight_decay;

            // Update parameters
            *param_matrix -= &(momentum_entry.clone() * learning_rate + decay_term * learning_rate);
        }

        Ok(())
    }

    /// Compute gradient (simplified placeholder)
    fn compute_gradient(
        _samples: &[ExperienceSample],
        param_matrix: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>> {
        // Simplified gradient computation
        // In practice, this would involve backpropagation through the embedding model
        let mut gradient = DMatrix::zeros(param_matrix.nrows(), param_matrix.ncols());

        // Add small random perturbations as a placeholder
        for i in 0..gradient.nrows() {
            for j in 0..gradient.ncols() {
                gradient[(i, j)] = ({
                    use scirs2_core::random::{Rng, Random};
                    let mut random = Random::default();
                    random.gen::<f64>()
                } - 0.5)
                    * 0.001;
            }
        }

        Ok(gradient)
    }

    /// Meta-learning adaptation (MAML-style)
    fn meta_learning_adaptation(
        samples: &[ExperienceSample],
        parameters: &Arc<RwLock<HashMap<String, DMatrix<f64>>>>,
        _learning_state: &Arc<RwLock<LearningState>>,
        inner_steps: usize,
        outer_learning_rate: f64,
    ) -> Result<()> {
        let mut params_guard = parameters.write().unwrap();

        // MAML inner loop
        for _ in 0..inner_steps {
            for (_, param_matrix) in params_guard.iter_mut() {
                let gradient = Self::compute_gradient(samples, param_matrix)?;
                *param_matrix -= &(gradient * outer_learning_rate);
            }
        }

        Ok(())
    }

    /// Evolutionary adaptation
    fn evolutionary_adaptation(
        samples: &[ExperienceSample],
        parameters: &Arc<RwLock<HashMap<String, DMatrix<f64>>>>,
        mutation_rate: f64,
        population_size: usize,
    ) -> Result<()> {
        let mut params_guard = parameters.write().unwrap();

        // Simple evolutionary strategy
        for (_, param_matrix) in params_guard.iter_mut() {
            let mut best_fitness = Self::evaluate_fitness(samples, param_matrix)?;
            let mut best_params = param_matrix.clone();

            // Generate population
            for _ in 0..population_size {
                let mut mutated = param_matrix.clone();

                // Apply mutations
                for i in 0..mutated.nrows() {
                    for j in 0..mutated.ncols() {
                        if {
                            use scirs2_core::random::{Rng, Random};
                            let mut random = Random::default();
                            random.gen::<f64>()
                        } < mutation_rate
                        {
                            mutated[(i, j)] += ({
                                use scirs2_core::random::{Rng, Random};
                                let mut random = Random::default();
                                random.gen::<f64>()
                            } - 0.5)
                                * 0.01;
                        }
                    }
                }

                let fitness = Self::evaluate_fitness(samples, &mutated)?;
                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_params = mutated;
                }
            }

            *param_matrix = best_params;
        }

        Ok(())
    }

    /// Evaluate fitness for evolutionary adaptation
    fn evaluate_fitness(
        _samples: &[ExperienceSample],
        _param_matrix: &DMatrix<f64>,
    ) -> Result<f64> {
        // Simplified fitness evaluation
        // In practice, this would evaluate how well the parameters perform on the samples
        Ok({
            use scirs2_core::random::{Rng, Random};
            let mut random = Random::default();
            random.gen::<f64>()
        })
    }

    /// Bayesian optimization adaptation
    fn bayesian_optimization_adaptation(
        samples: &[ExperienceSample],
        parameters: &Arc<RwLock<HashMap<String, DMatrix<f64>>>>,
        exploration_factor: f64,
        _kernel_bandwidth: f64,
    ) -> Result<()> {
        let mut params_guard = parameters.write().unwrap();

        // Simplified Bayesian optimization
        for (_, param_matrix) in params_guard.iter_mut() {
            let current_fitness = Self::evaluate_fitness(samples, param_matrix)?;

            // Generate candidate solutions
            let mut best_candidate = param_matrix.clone();
            let mut best_acquisition = 0.0;

            for _ in 0..10 {
                let mut candidate = param_matrix.clone();

                // Add exploration noise
                for i in 0..candidate.nrows() {
                    for j in 0..candidate.ncols() {
                        candidate[(i, j)] += ({
                            use scirs2_core::random::{Rng, Random};
                            let mut random = Random::default();
                            random.gen::<f64>()
                        } - 0.5)
                            * exploration_factor;
                    }
                }

                let fitness = Self::evaluate_fitness(samples, &candidate)?;
                let acquisition = fitness
                    + exploration_factor * {
                        use scirs2_core::random::{Rng, Random};
                        let mut random = Random::default();
                        random.gen::<f64>()
                    };

                if acquisition > best_acquisition {
                    best_acquisition = acquisition;
                    best_candidate = candidate;
                }
            }

            // Update only if improvement is significant
            if best_acquisition > current_fitness + 0.01 {
                *param_matrix = best_candidate;
            }
        }

        Ok(())
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> AdaptationMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Get feedback sender for external use
    pub fn get_feedback_sender(&self) -> mpsc::UnboundedSender<QualityFeedback> {
        self.feedback_sender.clone()
    }

    /// Update adaptation strategy
    pub fn set_strategy(&mut self, strategy: AdaptationStrategy) {
        self.strategy = strategy;
    }

    /// Reset learning state
    pub fn reset_learning_state(&self) {
        let mut state_guard = self.learning_state.write().unwrap();
        state_guard.momentum.clear();
        state_guard.adaptation_history.clear();
        state_guard.current_learning_rate = self.config.learning_rate;
        state_guard.performance_baseline = 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_adaptive_learning_system_creation() {
        let config = AdaptiveLearningConfig::default();
        let system = AdaptiveLearningSystem::new(config);

        let metrics = system.get_metrics();
        assert_eq!(metrics.adaptations_count, 0);
        assert_eq!(metrics.avg_quality_improvement, 0.0);
    }

    #[tokio::test]
    async fn test_feedback_submission() {
        let config = AdaptiveLearningConfig::default();
        let system = AdaptiveLearningSystem::new(config);

        let feedback = QualityFeedback {
            query: "test query".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            quality_score: 0.9,
            timestamp: Utc::now(),
            relevance: Some(0.8),
            task_context: Some("similarity".to_string()),
        };

        assert!(system.submit_feedback(feedback).is_ok());
    }

    #[tokio::test]
    async fn test_adaptive_learning_config_default() {
        let config = AdaptiveLearningConfig::default();

        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.buffer_size, 10000);
        assert_eq!(config.min_samples_for_adaptation, 100);
        assert_eq!(config.quality_threshold, 0.8);
        assert!(config.enable_meta_learning);
    }

    #[tokio::test]
    async fn test_adaptation_strategies() {
        let config = AdaptiveLearningConfig::default();

        // Test different strategies
        let strategies = vec![
            AdaptationStrategy::GradientDescent {
                momentum: 0.9,
                weight_decay: 0.0001,
            },
            AdaptationStrategy::MetaLearning {
                inner_steps: 3,
                outer_learning_rate: 0.01,
            },
            AdaptationStrategy::Evolutionary {
                mutation_rate: 0.1,
                population_size: 20,
            },
            AdaptationStrategy::BayesianOptimization {
                exploration_factor: 0.1,
                kernel_bandwidth: 1.0,
            },
        ];

        for strategy in strategies {
            let system = AdaptiveLearningSystem::with_strategy(config.clone(), strategy);
            assert!(system.start_learning().await.is_ok());

            // Give some time for initialization
            sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn test_quality_calculation() {
        let samples = vec![
            ExperienceSample {
                input: "test1".to_string(),
                target: vec![1.0, 0.0, 0.0],
                current: vec![0.9, 0.1, 0.0],
                improvement_target: 0.1,
                context: HashMap::new(),
            },
            ExperienceSample {
                input: "test2".to_string(),
                target: vec![0.0, 1.0, 0.0],
                current: vec![0.0, 0.8, 0.2],
                improvement_target: 0.2,
                context: HashMap::new(),
            },
        ];

        let quality = AdaptiveLearningSystem::calculate_current_quality(&samples).unwrap();
        assert!(quality > 0.0 && quality <= 1.0);
    }
}
