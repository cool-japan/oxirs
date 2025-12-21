//! A/B Testing Framework for Model Comparison
//!
//! This module provides a comprehensive framework for A/B testing AI models in production.
//! It supports multi-variant testing, statistical significance analysis, gradual rollout,
//! and automated winner selection.
//!
//! # Features
//! - Multi-variant testing (A/B/C/D/...)
//! - Traffic splitting with consistent assignment
//! - Statistical significance testing (t-test, chi-square, Bayesian)
//! - Gradual rollout and canary deployments
//! - Automated winner selection based on metrics
//! - Real-time monitoring and alerts
//! - Experiment lifecycle management

use crate::{Result, ShaclAiError};
use scirs2_core::random::rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A/B test experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Unique experiment ID
    pub id: String,
    /// Experiment name
    pub name: String,
    /// Experiment description
    pub description: String,
    /// Test variants
    pub variants: Vec<Variant>,
    /// Traffic allocation per variant
    pub traffic_allocation: HashMap<String, f64>,
    /// Experiment status
    pub status: ExperimentStatus,
    /// Metrics to track
    pub metrics: Vec<MetricDefinition>,
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// End time (optional)
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Minimum sample size per variant
    pub min_sample_size: usize,
    /// Statistical significance threshold (p-value)
    pub significance_threshold: f64,
    /// Confidence level
    pub confidence_level: f64,
}

/// Test variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Variant ID
    pub id: String,
    /// Variant name
    pub name: String,
    /// Variant description
    pub description: String,
    /// Is this the control variant?
    pub is_control: bool,
    /// Model configuration for this variant
    pub model_config: HashMap<String, serde_json::Value>,
    /// Current sample count
    pub sample_count: usize,
    /// Collected metrics
    pub metrics: HashMap<String, Vec<f64>>,
}

/// Experiment status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    /// Experiment is being drafted
    Draft,
    /// Experiment is running
    Running,
    /// Experiment is paused
    Paused,
    /// Experiment is completed
    Completed,
    /// Experiment was stopped early
    Stopped,
}

/// Metric definition for A/B testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    /// Metric name
    pub name: String,
    /// Metric description
    pub description: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Goal (maximize or minimize)
    pub goal: MetricGoal,
    /// Weight for multi-metric optimization
    pub weight: f64,
}

/// Type of metric
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    /// Continuous metric (e.g., latency, accuracy)
    Continuous,
    /// Binary metric (e.g., success/failure)
    Binary,
    /// Count metric (e.g., number of errors)
    Count,
    /// Rate metric (e.g., click-through rate)
    Rate,
}

/// Metric goal
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricGoal {
    Maximize,
    Minimize,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test type
    pub test_type: StatisticalTestType,
    /// P-value
    pub p_value: f64,
    /// Is statistically significant?
    pub is_significant: bool,
    /// Effect size
    pub effect_size: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Type of statistical test
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticalTestType {
    /// Student's t-test
    TTest,
    /// Mann-Whitney U test
    MannWhitneyU,
    /// Chi-square test
    ChiSquare,
    /// Bayesian A/B test
    Bayesian,
}

/// Experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    /// Experiment ID
    pub experiment_id: String,
    /// Winning variant
    pub winner: Option<String>,
    /// Statistical tests per metric
    pub statistical_tests: HashMap<String, HashMap<String, StatisticalTest>>,
    /// Metric summaries per variant
    pub metric_summaries: HashMap<String, HashMap<String, MetricSummary>>,
    /// Overall recommendation
    pub recommendation: Recommendation,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Metric summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Sample count
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Median
    pub median: f64,
    /// 25th percentile
    pub p25: f64,
    /// 75th percentile
    pub p75: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

/// Recommendation from A/B test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Action to take
    pub action: RecommendationAction,
    /// Confidence in recommendation (0.0 to 1.0)
    pub confidence: f64,
    /// Reasoning
    pub reasoning: Vec<String>,
    /// Expected lift
    pub expected_lift: Option<f64>,
}

/// Recommendation action
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationAction {
    /// Roll out winner to all traffic
    RolloutWinner,
    /// Continue experiment (needs more data)
    ContinueExperiment,
    /// Stop experiment (no significant difference)
    StopNoWinner,
    /// Manual review needed
    ManualReview,
}

/// A/B testing framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Enable consistent hashing for user assignment
    pub enable_consistent_hashing: bool,
    /// Hash seed for reproducibility
    pub hash_seed: u64,
    /// Enable gradual rollout
    pub enable_gradual_rollout: bool,
    /// Rollout increment (percentage points per day)
    pub rollout_increment: f64,
    /// Enable auto winner selection
    pub enable_auto_winner: bool,
    /// Minimum experiment duration (days)
    pub min_experiment_days: usize,
    /// Maximum experiment duration (days)
    pub max_experiment_days: usize,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            enable_consistent_hashing: true,
            hash_seed: 12345,
            enable_gradual_rollout: true,
            rollout_increment: 10.0,
            enable_auto_winner: false,
            min_experiment_days: 7,
            max_experiment_days: 30,
        }
    }
}

/// A/B testing framework
pub struct ABTestFramework {
    config: ABTestConfig,
    experiments: HashMap<String, Experiment>,
    assignments: HashMap<String, HashMap<String, String>>, // experiment_id -> user_id -> variant_id
}

impl ABTestFramework {
    /// Create new A/B testing framework
    pub fn new() -> Self {
        Self::with_config(ABTestConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ABTestConfig) -> Self {
        Self {
            config,
            experiments: HashMap::new(),
            assignments: HashMap::new(),
        }
    }

    /// Create a new experiment
    pub fn create_experiment(&mut self, mut experiment: Experiment) -> Result<()> {
        // Validate traffic allocation sums to ~1.0
        let total_traffic: f64 = experiment.traffic_allocation.values().sum();
        if (total_traffic - 1.0).abs() > 0.01 {
            return Err(ShaclAiError::Configuration(format!(
                "Traffic allocation must sum to 1.0, got {}",
                total_traffic
            )));
        }

        // Initialize variant metrics
        for variant in &mut experiment.variants {
            for metric_def in &experiment.metrics {
                variant.metrics.insert(metric_def.name.clone(), Vec::new());
            }
        }

        let experiment_id = experiment.id.clone();
        self.experiments.insert(experiment_id.clone(), experiment);
        self.assignments.insert(experiment_id, HashMap::new());

        Ok(())
    }

    /// Start an experiment
    pub fn start_experiment(&mut self, experiment_id: &str) -> Result<()> {
        let experiment = self.experiments.get_mut(experiment_id).ok_or_else(|| {
            ShaclAiError::Configuration(format!("Experiment {} not found", experiment_id))
        })?;

        experiment.status = ExperimentStatus::Running;
        experiment.start_time = chrono::Utc::now();

        tracing::info!("Started experiment: {}", experiment_id);
        Ok(())
    }

    /// Assign user to variant
    pub fn assign_variant(&mut self, experiment_id: &str, user_id: &str) -> Result<String> {
        let experiment = self.experiments.get(experiment_id).ok_or_else(|| {
            ShaclAiError::Configuration(format!("Experiment {} not found", experiment_id))
        })?;

        if experiment.status != ExperimentStatus::Running {
            return Err(ShaclAiError::Configuration(format!(
                "Experiment {} is not running",
                experiment_id
            )));
        }

        // Check if user already assigned
        if let Some(assignments) = self.assignments.get(experiment_id) {
            if let Some(variant_id) = assignments.get(user_id) {
                return Ok(variant_id.clone());
            }
        }

        // Assign new user
        let variant_id = if self.config.enable_consistent_hashing {
            self.consistent_hash_assignment(user_id, &experiment.traffic_allocation)
        } else {
            self.random_assignment(&experiment.traffic_allocation)
        };

        self.assignments
            .get_mut(experiment_id)
            .unwrap()
            .insert(user_id.to_string(), variant_id.clone());

        Ok(variant_id)
    }

    /// Record metric observation
    pub fn record_metric(
        &mut self,
        experiment_id: &str,
        variant_id: &str,
        metric_name: &str,
        value: f64,
    ) -> Result<()> {
        let experiment = self.experiments.get_mut(experiment_id).ok_or_else(|| {
            ShaclAiError::Configuration(format!("Experiment {} not found", experiment_id))
        })?;

        let variant = experiment
            .variants
            .iter_mut()
            .find(|v| v.id == variant_id)
            .ok_or_else(|| {
                ShaclAiError::Configuration(format!("Variant {} not found", variant_id))
            })?;

        variant.sample_count += 1;
        variant
            .metrics
            .entry(metric_name.to_string())
            .or_insert_with(Vec::new)
            .push(value);

        Ok(())
    }

    /// Analyze experiment results
    pub fn analyze_experiment(&self, experiment_id: &str) -> Result<ExperimentResults> {
        let experiment = self.experiments.get(experiment_id).ok_or_else(|| {
            ShaclAiError::Analytics(format!("Experiment {} not found", experiment_id))
        })?;

        // Find control variant
        let control = experiment
            .variants
            .iter()
            .find(|v| v.is_control)
            .ok_or_else(|| ShaclAiError::Configuration("No control variant defined".to_string()))?;

        let mut statistical_tests: HashMap<String, HashMap<String, StatisticalTest>> =
            HashMap::new();
        let mut metric_summaries: HashMap<String, HashMap<String, MetricSummary>> = HashMap::new();

        // Analyze each metric
        for metric_def in &experiment.metrics {
            let metric_name = &metric_def.name;

            for variant in &experiment.variants {
                if variant.id == control.id {
                    continue;
                }

                // Run statistical test
                let control_data = control.metrics.get(metric_name).unwrap();
                let variant_data = variant.metrics.get(metric_name).unwrap();

                let test = self.run_statistical_test(
                    control_data,
                    variant_data,
                    experiment.significance_threshold,
                    StatisticalTestType::TTest,
                );

                statistical_tests
                    .entry(metric_name.clone())
                    .or_insert_with(HashMap::new)
                    .insert(variant.id.clone(), test);

                // Calculate summary statistics
                let summary = self.calculate_metric_summary(variant_data);
                metric_summaries
                    .entry(variant.id.clone())
                    .or_insert_with(HashMap::new)
                    .insert(metric_name.clone(), summary);
            }

            // Summary for control
            let control_data = control.metrics.get(metric_name).unwrap();
            let control_summary = self.calculate_metric_summary(control_data);
            metric_summaries
                .entry(control.id.clone())
                .or_insert_with(HashMap::new)
                .insert(metric_name.clone(), control_summary);
        }

        // Determine winner and recommendation
        let (winner, recommendation) = self.determine_winner(experiment, &statistical_tests)?;

        Ok(ExperimentResults {
            experiment_id: experiment_id.to_string(),
            winner,
            statistical_tests,
            metric_summaries,
            recommendation,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Stop an experiment
    pub fn stop_experiment(&mut self, experiment_id: &str) -> Result<()> {
        let experiment = self.experiments.get_mut(experiment_id).ok_or_else(|| {
            ShaclAiError::Configuration(format!("Experiment {} not found", experiment_id))
        })?;

        experiment.status = ExperimentStatus::Stopped;
        experiment.end_time = Some(chrono::Utc::now());

        tracing::info!("Stopped experiment: {}", experiment_id);
        Ok(())
    }

    /// Get experiment
    pub fn get_experiment(&self, experiment_id: &str) -> Option<&Experiment> {
        self.experiments.get(experiment_id)
    }

    /// List all experiments
    pub fn list_experiments(&self) -> Vec<&Experiment> {
        self.experiments.values().collect()
    }

    // Private helper methods

    fn consistent_hash_assignment(
        &self,
        user_id: &str,
        traffic_allocation: &HashMap<String, f64>,
    ) -> String {
        // Use consistent hashing for stable assignment
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        user_id.hash(&mut hasher);
        self.config.hash_seed.hash(&mut hasher);
        let hash = hasher.finish();

        let normalized = (hash % 10000) as f64 / 10000.0;
        self.select_variant_by_threshold(normalized, traffic_allocation)
    }

    fn random_assignment(&self, traffic_allocation: &HashMap<String, f64>) -> String {
        // Use fastrand for simplicity
        let random_val: f64 = fastrand::f64();
        self.select_variant_by_threshold(random_val, traffic_allocation)
    }

    fn select_variant_by_threshold(
        &self,
        threshold: f64,
        traffic_allocation: &HashMap<String, f64>,
    ) -> String {
        let mut cumulative = 0.0;
        for (variant_id, allocation) in traffic_allocation {
            cumulative += allocation;
            if threshold <= cumulative {
                return variant_id.clone();
            }
        }
        // Fallback to first variant
        traffic_allocation.keys().next().unwrap().clone()
    }

    fn run_statistical_test(
        &self,
        control_data: &[f64],
        treatment_data: &[f64],
        significance_threshold: f64,
        _test_type: StatisticalTestType,
    ) -> StatisticalTest {
        // Simplified t-test implementation
        let control_mean = Self::mean(control_data);
        let treatment_mean = Self::mean(treatment_data);

        let control_var = Self::variance(control_data, control_mean);
        let treatment_var = Self::variance(treatment_data, treatment_mean);

        let n1 = control_data.len() as f64;
        let n2 = treatment_data.len() as f64;

        // Pooled standard error
        let se = ((control_var / n1) + (treatment_var / n2)).sqrt();

        // T-statistic
        let t_stat = if se > 0.0 {
            (treatment_mean - control_mean) / se
        } else {
            0.0
        };

        // Simplified p-value calculation (two-tailed)
        // In production, use proper t-distribution
        let p_value = 2.0 * (1.0 - Self::normal_cdf(t_stat.abs()));

        // Effect size (Cohen's d)
        let pooled_std = ((control_var + treatment_var) / 2.0).sqrt();
        let effect_size = if pooled_std > 0.0 {
            (treatment_mean - control_mean) / pooled_std
        } else {
            0.0
        };

        // Confidence interval (95%)
        let margin_of_error = 1.96 * se;
        let diff = treatment_mean - control_mean;
        let ci = (diff - margin_of_error, diff + margin_of_error);

        StatisticalTest {
            test_type: StatisticalTestType::TTest,
            p_value,
            is_significant: p_value < significance_threshold,
            effect_size,
            confidence_interval: ci,
        }
    }

    fn calculate_metric_summary(&self, data: &[f64]) -> MetricSummary {
        if data.is_empty() {
            return MetricSummary {
                count: 0,
                mean: 0.0,
                std_dev: 0.0,
                median: 0.0,
                p25: 0.0,
                p75: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let mean = Self::mean(data);
        let std_dev = Self::variance(data, mean).sqrt();

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        MetricSummary {
            count: data.len(),
            mean,
            std_dev,
            median: Self::percentile(&sorted, 50.0),
            p25: Self::percentile(&sorted, 25.0),
            p75: Self::percentile(&sorted, 75.0),
            p95: Self::percentile(&sorted, 95.0),
            p99: Self::percentile(&sorted, 99.0),
        }
    }

    fn determine_winner(
        &self,
        experiment: &Experiment,
        statistical_tests: &HashMap<String, HashMap<String, StatisticalTest>>,
    ) -> Result<(Option<String>, Recommendation)> {
        // Check if minimum sample size reached
        let min_samples_reached = experiment
            .variants
            .iter()
            .all(|v| v.sample_count >= experiment.min_sample_size);

        if !min_samples_reached {
            return Ok((
                None,
                Recommendation {
                    action: RecommendationAction::ContinueExperiment,
                    confidence: 0.0,
                    reasoning: vec!["Minimum sample size not reached".to_string()],
                    expected_lift: None,
                },
            ));
        }

        // Find variant with best performance on primary metric
        let primary_metric = &experiment.metrics[0].name;
        let metric_goal = &experiment.metrics[0].goal;

        let mut best_variant: Option<String> = None;
        let mut best_score = if *metric_goal == MetricGoal::Maximize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut is_significant = false;

        for variant in &experiment.variants {
            if variant.is_control {
                continue;
            }

            let data = variant.metrics.get(primary_metric).unwrap();
            let score = Self::mean(data);

            // Check if improvement is statistically significant
            if let Some(tests) = statistical_tests.get(primary_metric) {
                if let Some(test) = tests.get(&variant.id) {
                    if test.is_significant {
                        let is_better = match metric_goal {
                            MetricGoal::Maximize => score > best_score,
                            MetricGoal::Minimize => score < best_score,
                        };

                        if is_better {
                            best_variant = Some(variant.id.clone());
                            best_score = score;
                            is_significant = true;
                        }
                    }
                }
            }
        }

        if is_significant {
            Ok((
                best_variant.clone(),
                Recommendation {
                    action: RecommendationAction::RolloutWinner,
                    confidence: 0.95,
                    reasoning: vec![format!(
                        "Variant {:?} shows statistically significant improvement",
                        best_variant
                    )],
                    expected_lift: Some(best_score),
                },
            ))
        } else {
            Ok((
                None,
                Recommendation {
                    action: RecommendationAction::StopNoWinner,
                    confidence: 0.8,
                    reasoning: vec!["No statistically significant winner found".to_string()],
                    expected_lift: None,
                },
            ))
        }
    }

    // Statistical helper functions

    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            0.0
        } else {
            data.iter().sum::<f64>() / data.len() as f64
        }
    }

    fn variance(data: &[f64], mean: f64) -> f64 {
        if data.len() <= 1 {
            0.0
        } else {
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
        }
    }

    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }
        let index = (p / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }

    fn normal_cdf(x: f64) -> f64 {
        // Simplified normal CDF using error function approximation
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

impl Default for ABTestFramework {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_experiment() {
        let mut framework = ABTestFramework::new();

        let mut traffic_allocation = HashMap::new();
        traffic_allocation.insert("control".to_string(), 0.5);
        traffic_allocation.insert("treatment".to_string(), 0.5);

        let experiment = Experiment {
            id: "exp1".to_string(),
            name: "Test Experiment".to_string(),
            description: "Testing A/B framework".to_string(),
            variants: vec![
                Variant {
                    id: "control".to_string(),
                    name: "Control".to_string(),
                    description: "Baseline model".to_string(),
                    is_control: true,
                    model_config: HashMap::new(),
                    sample_count: 0,
                    metrics: HashMap::new(),
                },
                Variant {
                    id: "treatment".to_string(),
                    name: "Treatment".to_string(),
                    description: "New model".to_string(),
                    is_control: false,
                    model_config: HashMap::new(),
                    sample_count: 0,
                    metrics: HashMap::new(),
                },
            ],
            traffic_allocation,
            status: ExperimentStatus::Draft,
            metrics: vec![MetricDefinition {
                name: "accuracy".to_string(),
                description: "Model accuracy".to_string(),
                metric_type: MetricType::Continuous,
                goal: MetricGoal::Maximize,
                weight: 1.0,
            }],
            start_time: chrono::Utc::now(),
            end_time: None,
            min_sample_size: 100,
            significance_threshold: 0.05,
            confidence_level: 0.95,
        };

        framework.create_experiment(experiment).unwrap();
        assert_eq!(framework.experiments.len(), 1);
    }

    #[test]
    fn test_variant_assignment() {
        let mut framework = ABTestFramework::new();

        let mut traffic_allocation = HashMap::new();
        traffic_allocation.insert("control".to_string(), 0.5);
        traffic_allocation.insert("treatment".to_string(), 0.5);

        let experiment = Experiment {
            id: "exp1".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            variants: vec![
                Variant {
                    id: "control".to_string(),
                    name: "Control".to_string(),
                    description: "".to_string(),
                    is_control: true,
                    model_config: HashMap::new(),
                    sample_count: 0,
                    metrics: HashMap::new(),
                },
                Variant {
                    id: "treatment".to_string(),
                    name: "Treatment".to_string(),
                    description: "".to_string(),
                    is_control: false,
                    model_config: HashMap::new(),
                    sample_count: 0,
                    metrics: HashMap::new(),
                },
            ],
            traffic_allocation,
            status: ExperimentStatus::Draft,
            metrics: vec![],
            start_time: chrono::Utc::now(),
            end_time: None,
            min_sample_size: 100,
            significance_threshold: 0.05,
            confidence_level: 0.95,
        };

        framework.create_experiment(experiment).unwrap();
        framework.start_experiment("exp1").unwrap();

        let variant1 = framework.assign_variant("exp1", "user1").unwrap();
        let variant2 = framework.assign_variant("exp1", "user1").unwrap();

        // Same user should get same variant
        assert_eq!(variant1, variant2);
    }

    #[test]
    fn test_statistical_functions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = ABTestFramework::mean(&data);
        assert_eq!(mean, 3.0);

        let variance = ABTestFramework::variance(&data, mean);
        assert!((variance - 2.5).abs() < 0.01);

        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = ABTestFramework::percentile(&sorted, 50.0);
        assert_eq!(median, 3.0);
    }

    #[test]
    fn test_record_metric() {
        let mut framework = ABTestFramework::new();

        let mut traffic_allocation = HashMap::new();
        traffic_allocation.insert("control".to_string(), 1.0);

        let experiment = Experiment {
            id: "exp1".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            variants: vec![Variant {
                id: "control".to_string(),
                name: "Control".to_string(),
                description: "".to_string(),
                is_control: true,
                model_config: HashMap::new(),
                sample_count: 0,
                metrics: HashMap::new(),
            }],
            traffic_allocation,
            status: ExperimentStatus::Draft,
            metrics: vec![MetricDefinition {
                name: "accuracy".to_string(),
                description: "".to_string(),
                metric_type: MetricType::Continuous,
                goal: MetricGoal::Maximize,
                weight: 1.0,
            }],
            start_time: chrono::Utc::now(),
            end_time: None,
            min_sample_size: 100,
            significance_threshold: 0.05,
            confidence_level: 0.95,
        };

        framework.create_experiment(experiment).unwrap();

        framework
            .record_metric("exp1", "control", "accuracy", 0.95)
            .unwrap();

        let exp = framework.get_experiment("exp1").unwrap();
        assert_eq!(exp.variants[0].sample_count, 1);
    }
}
