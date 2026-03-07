//! AutoML for Embedding Optimization - Version 1.2 Feature
//!
//! This module implements comprehensive AutoML capabilities for automatically
//! optimizing embedding configurations, model selection, hyperparameters, and
//! vector search performance. It provides intelligent automation for finding
//! the best embedding strategies for specific datasets and use cases.

use crate::{
    advanced_analytics::VectorAnalyticsEngine,
    benchmarking::{BenchmarkConfig, BenchmarkSuite},
    embeddings::EmbeddingStrategy,
    similarity::SimilarityMetric,
    VectorStore,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{info, span, warn, Level};
use uuid::Uuid;

/// AutoML optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLConfig {
    /// Maximum optimization time budget
    pub max_optimization_time: Duration,
    /// Number of trials per configuration
    pub trials_per_config: usize,
    /// Evaluation metrics to optimize
    pub optimization_metrics: Vec<OptimizationMetric>,
    /// Search space for hyperparameters
    pub search_space: SearchSpace,
    /// Cross-validation folds
    pub cross_validation_folds: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Enable parallel optimization
    pub enable_parallel_optimization: bool,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            max_optimization_time: Duration::from_secs(3600), // 1 hour
            trials_per_config: 5,
            optimization_metrics: vec![
                OptimizationMetric::Accuracy,
                OptimizationMetric::Latency,
                OptimizationMetric::MemoryUsage,
            ],
            search_space: SearchSpace::default(),
            cross_validation_folds: 5,
            early_stopping_patience: 10,
            enable_parallel_optimization: true,
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

/// Metrics to optimize during AutoML
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OptimizationMetric {
    /// Search accuracy (recall@k, precision@k)
    Accuracy,
    /// Query latency
    Latency,
    /// Memory usage
    MemoryUsage,
    /// Throughput (queries per second)
    Throughput,
    /// Index build time
    IndexBuildTime,
    /// Storage efficiency
    StorageEfficiency,
    /// Embedding quality
    EmbeddingQuality,
}

/// Hyperparameter search space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Embedding strategies to evaluate
    pub embedding_strategies: Vec<EmbeddingStrategy>,
    /// Vector dimensions to test
    pub vector_dimensions: Vec<usize>,
    /// Similarity metrics to evaluate
    pub similarity_metrics: Vec<SimilarityMetric>,
    /// Index parameters
    pub index_parameters: IndexParameterSpace,
    /// Learning rate ranges for trainable embeddings
    pub learning_rates: Vec<f32>,
    /// Batch sizes for processing
    pub batch_sizes: Vec<usize>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            embedding_strategies: vec![
                EmbeddingStrategy::TfIdf,
                EmbeddingStrategy::SentenceTransformer,
                EmbeddingStrategy::Custom("default".to_string()),
            ],
            vector_dimensions: vec![128, 256, 512, 768, 1024],
            similarity_metrics: vec![
                SimilarityMetric::Cosine,
                SimilarityMetric::Euclidean,
                SimilarityMetric::DotProduct,
            ],
            index_parameters: IndexParameterSpace::default(),
            learning_rates: vec![0.001, 0.01, 0.1],
            batch_sizes: vec![32, 64, 128, 256],
        }
    }
}

/// Index-specific parameter search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParameterSpace {
    /// HNSW specific parameters
    pub hnsw_m: Vec<usize>,
    pub hnsw_ef_construction: Vec<usize>,
    pub hnsw_ef_search: Vec<usize>,
    /// IVF specific parameters
    pub ivf_nlist: Vec<usize>,
    pub ivf_nprobe: Vec<usize>,
    /// PQ specific parameters
    pub pq_m: Vec<usize>,
    pub pq_nbits: Vec<usize>,
}

impl Default for IndexParameterSpace {
    fn default() -> Self {
        Self {
            hnsw_m: vec![16, 32, 64],
            hnsw_ef_construction: vec![100, 200, 400],
            hnsw_ef_search: vec![50, 100, 200],
            ivf_nlist: vec![100, 1000, 4096],
            ivf_nprobe: vec![1, 10, 50],
            pq_m: vec![8, 16, 32],
            pq_nbits: vec![4, 8],
        }
    }
}

/// Resource constraints for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Maximum CPU cores to use
    pub max_cpu_cores: usize,
    /// Maximum GPU memory if available
    pub max_gpu_memory_bytes: Option<usize>,
    /// Maximum disk usage for indices
    pub max_disk_usage_bytes: usize,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            max_cpu_cores: 4,
            max_gpu_memory_bytes: None,
            max_disk_usage_bytes: 50 * 1024 * 1024 * 1024, // 50GB
        }
    }
}

/// AutoML optimization configuration for a specific trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTrial {
    pub trial_id: String,
    pub embedding_strategy: EmbeddingStrategy,
    pub vector_dimension: usize,
    pub similarity_metric: SimilarityMetric,
    pub index_config: IndexConfiguration,
    pub hyperparameters: HashMap<String, f32>,
    pub timestamp: u64,
}

/// Index configuration for optimization trials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfiguration {
    pub index_type: String,
    pub parameters: HashMap<String, f32>,
}

/// Result of an AutoML optimization trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub trial: OptimizationTrial,
    pub metrics: HashMap<OptimizationMetric, f32>,
    pub cross_validation_scores: Vec<f32>,
    pub training_time: Duration,
    pub evaluation_time: Duration,
    pub memory_peak_usage: usize,
    pub error_message: Option<String>,
    pub success: bool,
}

/// AutoML optimization results summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLResults {
    pub best_configuration: OptimizationTrial,
    pub best_metrics: HashMap<OptimizationMetric, f32>,
    pub pareto_frontier: Vec<TrialResult>,
    pub optimization_history: Vec<TrialResult>,
    pub total_optimization_time: Duration,
    pub trials_completed: usize,
    pub improvement_curve: Vec<(usize, f32)>,
}

/// AutoML optimizer for embedding and vector search configurations
pub struct AutoMLOptimizer {
    config: AutoMLConfig,
    trial_history: Arc<RwLock<Vec<TrialResult>>>,
    best_trial: Arc<RwLock<Option<TrialResult>>>,
    optimization_state: Arc<Mutex<OptimizationState>>,
    #[allow(dead_code)]
    analytics_engine: Arc<Mutex<VectorAnalyticsEngine>>,
    #[allow(dead_code)]
    benchmark_engine: Arc<Mutex<BenchmarkSuite>>,
}

/// Internal optimization state
#[derive(Debug)]
struct OptimizationState {
    #[allow(dead_code)]
    current_trial: usize,
    early_stopping_counter: usize,
    best_score: f32,
    #[allow(dead_code)]
    pareto_frontier: Vec<TrialResult>,
    active_trials: HashMap<String, Instant>,
}

impl AutoMLOptimizer {
    /// Create a new AutoML optimizer
    pub fn new(config: AutoMLConfig) -> Result<Self> {
        Ok(Self {
            config,
            trial_history: Arc::new(RwLock::new(Vec::new())),
            best_trial: Arc::new(RwLock::new(None)),
            optimization_state: Arc::new(Mutex::new(OptimizationState {
                current_trial: 0,
                early_stopping_counter: 0,
                best_score: f32::NEG_INFINITY,
                pareto_frontier: Vec::new(),
                active_trials: HashMap::new(),
            })),
            analytics_engine: Arc::new(Mutex::new(VectorAnalyticsEngine::new())),
            benchmark_engine: Arc::new(Mutex::new(BenchmarkSuite::new(BenchmarkConfig::default()))),
        })
    }

    /// Create optimizer with default configuration
    pub fn with_default_config() -> Result<Self> {
        Self::new(AutoMLConfig::default())
    }

    /// Optimize embedding configuration for given dataset
    pub async fn optimize_embeddings(
        &self,
        training_data: &[(String, String)], // (id, content) pairs
        validation_data: &[(String, String)],
        test_queries: &[(String, Vec<String>)], // (query, relevant_docs)
    ) -> Result<AutoMLResults> {
        let span = span!(Level::INFO, "automl_optimization");
        let _enter = span.enter();

        info!(
            "Starting AutoML optimization for {} training samples",
            training_data.len()
        );

        let start_time = Instant::now();
        let optimization_state = self.optimization_state.lock().await;

        // Generate optimization trials
        let trials = self.generate_optimization_trials()?;
        info!("Generated {} optimization trials", trials.len());

        drop(optimization_state);

        let mut results = Vec::new();
        let mut best_score = f32::NEG_INFINITY;

        // Execute optimization trials
        for (i, trial) in trials.iter().enumerate() {
            if start_time.elapsed() > self.config.max_optimization_time {
                warn!("Optimization time budget exceeded, stopping early");
                break;
            }

            info!(
                "Executing trial {}/{}: {}",
                i + 1,
                trials.len(),
                trial.trial_id
            );

            match self
                .execute_trial(trial, training_data, validation_data, test_queries)
                .await
            {
                Ok(trial_result) => {
                    // Check for improvement
                    let primary_score = self.compute_primary_score(&trial_result.metrics);

                    if primary_score > best_score {
                        best_score = primary_score;
                        {
                            let mut best_trial = self
                                .best_trial
                                .write()
                                .expect("best_trial lock should not be poisoned");
                            *best_trial = Some(trial_result.clone());
                        } // Drop the mutex guard before await

                        // Reset early stopping counter
                        let mut state = self.optimization_state.lock().await;
                        state.early_stopping_counter = 0;
                        state.best_score = best_score;
                    } else {
                        let mut state = self.optimization_state.lock().await;
                        state.early_stopping_counter += 1;

                        // Check early stopping
                        if state.early_stopping_counter >= self.config.early_stopping_patience {
                            info!(
                                "Early stopping triggered after {} trials without improvement",
                                self.config.early_stopping_patience
                            );
                            break;
                        }
                    }

                    results.push(trial_result);
                }
                Err(e) => {
                    warn!("Trial {} failed: {}", trial.trial_id, e);
                    results.push(TrialResult {
                        trial: trial.clone(),
                        metrics: HashMap::new(),
                        cross_validation_scores: Vec::new(),
                        training_time: Duration::from_secs(0),
                        evaluation_time: Duration::from_secs(0),
                        memory_peak_usage: 0,
                        error_message: Some(e.to_string()),
                        success: false,
                    });
                }
            }
        }

        // Store results in history
        {
            let mut history = self
                .trial_history
                .write()
                .expect("trial_history lock should not be poisoned");
            history.extend(results.clone());
        }

        // Generate final results
        let best_trial = self
            .best_trial
            .read()
            .expect("best_trial lock should not be poisoned");
        let best_configuration = best_trial
            .as_ref()
            .map(|r| r.trial.clone())
            .unwrap_or_else(|| trials[0].clone());

        let best_metrics = best_trial
            .as_ref()
            .map(|r| r.metrics.clone())
            .unwrap_or_default();

        let pareto_frontier = self.compute_pareto_frontier(&results);
        let improvement_curve = self.compute_improvement_curve(&results);

        Ok(AutoMLResults {
            best_configuration,
            best_metrics,
            pareto_frontier,
            optimization_history: results,
            total_optimization_time: start_time.elapsed(),
            trials_completed: trials.len(),
            improvement_curve,
        })
    }

    /// Generate optimization trials based on search space
    fn generate_optimization_trials(&self) -> Result<Vec<OptimizationTrial>> {
        let mut trials = Vec::new();

        // Grid search over key parameters
        for embedding_strategy in &self.config.search_space.embedding_strategies {
            for &vector_dimension in &self.config.search_space.vector_dimensions {
                for similarity_metric in &self.config.search_space.similarity_metrics {
                    for &learning_rate in &self.config.search_space.learning_rates {
                        for &batch_size in &self.config.search_space.batch_sizes {
                            let trial = OptimizationTrial {
                                trial_id: Uuid::new_v4().to_string(),
                                embedding_strategy: embedding_strategy.clone(),
                                vector_dimension,
                                similarity_metric: *similarity_metric,
                                index_config: self.generate_index_config()?,
                                hyperparameters: {
                                    let mut params = HashMap::new();
                                    params.insert("learning_rate".to_string(), learning_rate);
                                    params.insert("batch_size".to_string(), batch_size as f32);
                                    params
                                },
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                            };
                            trials.push(trial);
                        }
                    }
                }
            }
        }

        // Add random search trials for exploration
        for _ in 0..20 {
            trials.push(self.generate_random_trial()?);
        }

        Ok(trials)
    }

    /// Execute a single optimization trial
    async fn execute_trial(
        &self,
        trial: &OptimizationTrial,
        training_data: &[(String, String)],
        validation_data: &[(String, String)],
        test_queries: &[(String, Vec<String>)],
    ) -> Result<TrialResult> {
        let trial_start = Instant::now();

        // Record trial start
        {
            let mut state = self.optimization_state.lock().await;
            state
                .active_trials
                .insert(trial.trial_id.clone(), trial_start);
        }

        // Create vector store with trial configuration
        let mut vector_store = self.create_vector_store_for_trial(trial)?;

        // Training phase
        let training_start = Instant::now();
        for (id, content) in training_data {
            vector_store
                .index_resource(id.clone(), content)
                .context("Failed to index training data")?;
        }
        let training_time = training_start.elapsed();

        // Cross-validation evaluation
        let cv_scores = self
            .perform_cross_validation(&vector_store, validation_data, test_queries)
            .await?;

        // Final evaluation
        let eval_start = Instant::now();
        let metrics = self
            .evaluate_trial_performance(&vector_store, test_queries, trial)
            .await?;
        let evaluation_time = eval_start.elapsed();

        // Estimate memory usage (simplified)
        let memory_peak_usage = self.estimate_memory_usage(&vector_store, trial)?;

        // Clean up trial state
        {
            let mut state = self.optimization_state.lock().await;
            state.active_trials.remove(&trial.trial_id);
        }

        Ok(TrialResult {
            trial: trial.clone(),
            metrics,
            cross_validation_scores: cv_scores,
            training_time,
            evaluation_time,
            memory_peak_usage,
            error_message: None,
            success: true,
        })
    }

    /// Perform cross-validation for trial evaluation
    async fn perform_cross_validation(
        &self,
        vector_store: &VectorStore,
        validation_data: &[(String, String)],
        test_queries: &[(String, Vec<String>)],
    ) -> Result<Vec<f32>> {
        let fold_size = validation_data.len() / self.config.cross_validation_folds;
        let mut cv_scores = Vec::new();

        for fold in 0..self.config.cross_validation_folds {
            let _start_idx = fold * fold_size;
            let _end_idx = if fold == self.config.cross_validation_folds - 1 {
                validation_data.len()
            } else {
                (fold + 1) * fold_size
            };

            // Use this fold as test set
            let fold_queries: Vec<_> = test_queries
                .iter()
                .filter(|(query_id, _)| {
                    // Simple hash-based assignment
                    let hash = query_id.chars().map(|c| c as u32).sum::<u32>() as usize;
                    let fold_idx = hash % self.config.cross_validation_folds;
                    fold_idx == fold
                })
                .cloned()
                .collect();

            if fold_queries.is_empty() {
                cv_scores.push(0.0);
                continue;
            }

            // Evaluate on this fold
            let mut total_recall = 0.0;
            for (query, relevant_docs) in &fold_queries {
                let search_results = vector_store.similarity_search(query, 10)?;
                let retrieved_docs: Vec<String> =
                    search_results.iter().map(|r| r.0.clone()).collect();

                let recall = self.compute_recall(&retrieved_docs, relevant_docs);
                total_recall += recall;
            }

            let avg_recall = total_recall / fold_queries.len() as f32;
            cv_scores.push(avg_recall);
        }

        Ok(cv_scores)
    }

    /// Evaluate trial performance metrics
    async fn evaluate_trial_performance(
        &self,
        vector_store: &VectorStore,
        test_queries: &[(String, Vec<String>)],
        trial: &OptimizationTrial,
    ) -> Result<HashMap<OptimizationMetric, f32>> {
        let mut metrics = HashMap::new();

        // Accuracy metrics
        let mut total_recall = 0.0;
        let mut total_precision = 0.0;
        let mut total_latency = 0.0;

        for (query, relevant_docs) in test_queries {
            let query_start = Instant::now();
            let search_results = vector_store.similarity_search(query, 10)?;
            let query_latency = query_start.elapsed().as_millis() as f32;

            total_latency += query_latency;

            let retrieved_docs: Vec<String> = search_results.iter().map(|r| r.0.clone()).collect();

            let recall = self.compute_recall(&retrieved_docs, relevant_docs);
            let precision = self.compute_precision(&retrieved_docs, relevant_docs);

            total_recall += recall;
            total_precision += precision;
        }

        let num_queries = test_queries.len() as f32;
        metrics.insert(
            OptimizationMetric::Accuracy,
            (total_recall + total_precision) / (2.0 * num_queries),
        );
        metrics.insert(OptimizationMetric::Latency, total_latency / num_queries);

        // Throughput (queries per second)
        let avg_latency_seconds = (total_latency / num_queries) / 1000.0;
        metrics.insert(OptimizationMetric::Throughput, 1.0 / avg_latency_seconds);

        // Memory and storage efficiency (simplified estimates)
        metrics.insert(
            OptimizationMetric::MemoryUsage,
            (trial.vector_dimension as f32) * 4.0,
        ); // 4 bytes per f32
        metrics.insert(
            OptimizationMetric::StorageEfficiency,
            1.0 / (trial.vector_dimension as f32).log2(),
        );

        // Index build time (estimated from hyperparameters)
        let build_time_estimate = match trial.embedding_strategy {
            EmbeddingStrategy::TfIdf => 100.0,
            EmbeddingStrategy::SentenceTransformer => 1000.0,
            _ => 500.0,
        };
        metrics.insert(OptimizationMetric::IndexBuildTime, build_time_estimate);

        // Embedding quality (simplified metric)
        let embedding_quality = 1.0 - (1.0 / (trial.vector_dimension as f32).sqrt());
        metrics.insert(OptimizationMetric::EmbeddingQuality, embedding_quality);

        Ok(metrics)
    }

    /// Generate a random trial for exploration
    fn generate_random_trial(&self) -> Result<OptimizationTrial> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        Uuid::new_v4().hash(&mut hasher);
        let random_seed = hasher.finish();

        let search_space = &self.config.search_space;

        Ok(OptimizationTrial {
            trial_id: Uuid::new_v4().to_string(),
            embedding_strategy: search_space.embedding_strategies
                [(random_seed % search_space.embedding_strategies.len() as u64) as usize]
                .clone(),
            vector_dimension: search_space.vector_dimensions
                [((random_seed >> 8) % search_space.vector_dimensions.len() as u64) as usize],
            similarity_metric: search_space.similarity_metrics
                [((random_seed >> 16) % search_space.similarity_metrics.len() as u64) as usize],
            index_config: self.generate_index_config()?,
            hyperparameters: {
                let mut params = HashMap::new();
                let lr_idx =
                    ((random_seed >> 24) % search_space.learning_rates.len() as u64) as usize;
                let bs_idx = ((random_seed >> 32) % search_space.batch_sizes.len() as u64) as usize;
                params.insert(
                    "learning_rate".to_string(),
                    search_space.learning_rates[lr_idx],
                );
                params.insert(
                    "batch_size".to_string(),
                    search_space.batch_sizes[bs_idx] as f32,
                );
                params
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    /// Generate index configuration for trial
    fn generate_index_config(&self) -> Result<IndexConfiguration> {
        Ok(IndexConfiguration {
            index_type: "hnsw".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("m".to_string(), 16.0);
                params.insert("ef_construction".to_string(), 200.0);
                params.insert("ef_search".to_string(), 100.0);
                params
            },
        })
    }

    /// Create vector store for trial configuration
    fn create_vector_store_for_trial(&self, trial: &OptimizationTrial) -> Result<VectorStore> {
        VectorStore::with_embedding_strategy(trial.embedding_strategy.clone())
            .context("Failed to create vector store for trial")
    }

    /// Compute primary optimization score
    fn compute_primary_score(&self, metrics: &HashMap<OptimizationMetric, f32>) -> f32 {
        let mut score = 0.0;

        // Weighted combination of metrics
        if let Some(&accuracy) = metrics.get(&OptimizationMetric::Accuracy) {
            score += accuracy * 0.4;
        }
        if let Some(&latency) = metrics.get(&OptimizationMetric::Latency) {
            // Lower latency is better, so invert
            score += (1.0 / (1.0 + latency / 1000.0)) * 0.3;
        }
        if let Some(&throughput) = metrics.get(&OptimizationMetric::Throughput) {
            score += (throughput / 100.0).min(1.0) * 0.3;
        }

        score
    }

    /// Compute Pareto frontier from trial results
    fn compute_pareto_frontier(&self, results: &[TrialResult]) -> Vec<TrialResult> {
        let mut frontier = Vec::new();

        for result in results {
            if !result.success {
                continue;
            }

            let is_dominated = results.iter().any(|other| {
                if !other.success || other.trial.trial_id == result.trial.trial_id {
                    return false;
                }

                // Check if other dominates result
                let mut better_in_all = true;
                let mut better_in_some = false;

                for metric in &self.config.optimization_metrics {
                    let result_val = result.metrics.get(metric).unwrap_or(&0.0);
                    let other_val = other.metrics.get(metric).unwrap_or(&0.0);

                    match metric {
                        OptimizationMetric::Latency | OptimizationMetric::MemoryUsage => {
                            // Lower is better
                            if other_val > result_val {
                                better_in_all = false;
                            } else if other_val < result_val {
                                better_in_some = true;
                            }
                        }
                        _ => {
                            // Higher is better
                            if other_val < result_val {
                                better_in_all = false;
                            } else if other_val > result_val {
                                better_in_some = true;
                            }
                        }
                    }
                }

                better_in_all && better_in_some
            });

            if !is_dominated {
                frontier.push(result.clone());
            }
        }

        frontier
    }

    /// Compute improvement curve over trials
    fn compute_improvement_curve(&self, results: &[TrialResult]) -> Vec<(usize, f32)> {
        let mut curve = Vec::new();
        let mut best_score = f32::NEG_INFINITY;

        for (i, result) in results.iter().enumerate() {
            if result.success {
                let score = self.compute_primary_score(&result.metrics);
                if score > best_score {
                    best_score = score;
                }
            }
            curve.push((i, best_score));
        }

        curve
    }

    /// Estimate memory usage for trial
    fn estimate_memory_usage(
        &self,
        _vector_store: &VectorStore,
        trial: &OptimizationTrial,
    ) -> Result<usize> {
        // Simplified memory estimation
        let base_memory = 100 * 1024 * 1024; // 100MB base
        let vector_memory = trial.vector_dimension * 4; // 4 bytes per f32
        let index_overhead = vector_memory / 2; // 50% overhead for index structures

        Ok(base_memory + vector_memory + index_overhead)
    }

    /// Compute recall@k metric
    fn compute_recall(&self, retrieved: &[String], relevant: &[String]) -> f32 {
        if relevant.is_empty() {
            return 1.0;
        }

        let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();
        let retrieved_relevant = retrieved
            .iter()
            .filter(|doc| relevant_set.contains(doc))
            .count();

        retrieved_relevant as f32 / relevant.len() as f32
    }

    /// Compute precision@k metric
    fn compute_precision(&self, retrieved: &[String], relevant: &[String]) -> f32 {
        if retrieved.is_empty() {
            return 0.0;
        }

        let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();
        let retrieved_relevant = retrieved
            .iter()
            .filter(|doc| relevant_set.contains(doc))
            .count();

        retrieved_relevant as f32 / retrieved.len() as f32
    }

    /// Get optimization statistics
    pub fn get_optimization_statistics(&self) -> AutoMLStatistics {
        let history = self
            .trial_history
            .read()
            .expect("trial_history lock should not be poisoned");
        let best_trial = self
            .best_trial
            .read()
            .expect("best_trial lock should not be poisoned");

        let total_trials = history.len();
        let successful_trials = history.iter().filter(|r| r.success).count();
        let average_trial_time = if !history.is_empty() {
            history
                .iter()
                .map(|r| r.training_time + r.evaluation_time)
                .sum::<Duration>()
                .as_secs_f32()
                / history.len() as f32
        } else {
            0.0
        };

        let best_score = best_trial
            .as_ref()
            .map(|r| self.compute_primary_score(&r.metrics))
            .unwrap_or(0.0);

        AutoMLStatistics {
            total_trials,
            successful_trials,
            best_score,
            average_trial_time,
            optimization_metrics: self.config.optimization_metrics.clone(),
            search_space_size: self.estimate_search_space_size(),
        }
    }

    fn estimate_search_space_size(&self) -> usize {
        let space = &self.config.search_space;
        space.embedding_strategies.len()
            * space.vector_dimensions.len()
            * space.similarity_metrics.len()
            * space.learning_rates.len()
            * space.batch_sizes.len()
    }
}

/// AutoML optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLStatistics {
    pub total_trials: usize,
    pub successful_trials: usize,
    pub best_score: f32,
    pub average_trial_time: f32,
    pub optimization_metrics: Vec<OptimizationMetric>,
    pub search_space_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automl_config_creation() {
        let config = AutoMLConfig::default();
        assert_eq!(config.cross_validation_folds, 5);
        assert_eq!(config.early_stopping_patience, 10);
        assert!(config.enable_parallel_optimization);
    }

    #[test]
    fn test_search_space_default() {
        let search_space = SearchSpace::default();
        assert!(!search_space.embedding_strategies.is_empty());
        assert!(!search_space.vector_dimensions.is_empty());
        assert!(!search_space.similarity_metrics.is_empty());
    }

    #[tokio::test]
    async fn test_automl_optimizer_creation() {
        let optimizer = AutoMLOptimizer::with_default_config();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_trial_generation() {
        let optimizer = AutoMLOptimizer::with_default_config().unwrap();
        let trials = optimizer.generate_optimization_trials().unwrap();
        assert!(!trials.is_empty());

        // Check trial uniqueness
        let mut trial_ids = std::collections::HashSet::new();
        for trial in &trials {
            assert!(trial_ids.insert(trial.trial_id.clone()));
        }
    }

    #[tokio::test]
    async fn test_optimization_with_sample_data() {
        let _optimizer = AutoMLOptimizer::with_default_config().unwrap();

        let training_data = vec![
            (
                "doc1".to_string(),
                "artificial intelligence machine learning".to_string(),
            ),
            (
                "doc2".to_string(),
                "deep learning neural networks".to_string(),
            ),
            (
                "doc3".to_string(),
                "natural language processing".to_string(),
            ),
        ];

        let validation_data = vec![
            (
                "doc4".to_string(),
                "computer vision image recognition".to_string(),
            ),
            (
                "doc5".to_string(),
                "reinforcement learning algorithms".to_string(),
            ),
        ];

        let test_queries = vec![
            (
                "ai query".to_string(),
                vec!["doc1".to_string(), "doc2".to_string()],
            ),
            ("nlp query".to_string(), vec!["doc3".to_string()]),
        ];

        // Use a very short optimization time for testing
        let config = AutoMLConfig {
            max_optimization_time: Duration::from_secs(1),
            trials_per_config: 1,
            ..Default::default()
        };

        let optimizer = AutoMLOptimizer::new(config).unwrap();
        let results = optimizer
            .optimize_embeddings(&training_data, &validation_data, &test_queries)
            .await;

        // Test should complete without errors even if no trials complete
        assert!(results.is_ok());
    }

    #[test]
    fn test_pareto_frontier_computation() {
        let optimizer = AutoMLOptimizer::with_default_config().unwrap();

        let trial1 = OptimizationTrial {
            trial_id: "trial1".to_string(),
            embedding_strategy: EmbeddingStrategy::TfIdf,
            vector_dimension: 128,
            similarity_metric: SimilarityMetric::Cosine,
            index_config: optimizer.generate_index_config().unwrap(),
            hyperparameters: HashMap::new(),
            timestamp: 0,
        };

        let trial2 = OptimizationTrial {
            trial_id: "trial2".to_string(),
            embedding_strategy: EmbeddingStrategy::SentenceTransformer,
            vector_dimension: 256,
            similarity_metric: SimilarityMetric::Euclidean,
            index_config: optimizer.generate_index_config().unwrap(),
            hyperparameters: HashMap::new(),
            timestamp: 0,
        };

        let mut metrics1 = HashMap::new();
        metrics1.insert(OptimizationMetric::Accuracy, 0.8);
        metrics1.insert(OptimizationMetric::Latency, 100.0);

        let mut metrics2 = HashMap::new();
        metrics2.insert(OptimizationMetric::Accuracy, 0.9);
        metrics2.insert(OptimizationMetric::Latency, 200.0);

        let results = vec![
            TrialResult {
                trial: trial1,
                metrics: metrics1,
                cross_validation_scores: vec![0.8],
                training_time: Duration::from_secs(1),
                evaluation_time: Duration::from_secs(1),
                memory_peak_usage: 1000,
                error_message: None,
                success: true,
            },
            TrialResult {
                trial: trial2,
                metrics: metrics2,
                cross_validation_scores: vec![0.9],
                training_time: Duration::from_secs(2),
                evaluation_time: Duration::from_secs(1),
                memory_peak_usage: 2000,
                error_message: None,
                success: true,
            },
        ];

        let frontier = optimizer.compute_pareto_frontier(&results);
        assert_eq!(frontier.len(), 2); // Both trials should be on frontier
    }

    #[test]
    fn test_recall_precision_computation() {
        let optimizer = AutoMLOptimizer::with_default_config().unwrap();

        let retrieved = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];
        let relevant = vec!["doc1".to_string(), "doc3".to_string(), "doc4".to_string()];

        let recall = optimizer.compute_recall(&retrieved, &relevant);
        let precision = optimizer.compute_precision(&retrieved, &relevant);

        assert_eq!(recall, 2.0 / 3.0); // 2 out of 3 relevant docs retrieved
        assert_eq!(precision, 2.0 / 3.0); // 2 out of 3 retrieved docs are relevant
    }

    #[test]
    fn test_optimization_statistics() {
        let optimizer = AutoMLOptimizer::with_default_config().unwrap();
        let stats = optimizer.get_optimization_statistics();

        assert_eq!(stats.total_trials, 0);
        assert_eq!(stats.successful_trials, 0);
        assert!(stats.search_space_size > 0);
    }
}
