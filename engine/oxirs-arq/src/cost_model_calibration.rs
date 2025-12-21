//! Cost Model Calibration for Adaptive Query Optimization
//!
//! This module provides automatic calibration of cost model parameters based on
//! actual query execution times. It enables the optimizer to learn from real
//! workloads and improve cost estimates over time.
//!
//! # Features
//!
//! - **Adaptive Learning**: Learns cost coefficients from actual execution times
//! - **Statistical Analysis**: Uses regression analysis to fit cost models
//! - **Workload Profiling**: Tracks execution patterns for better predictions
//! - **Online Calibration**: Continuous calibration without system restart
//! - **Confidence Tracking**: Monitors calibration confidence levels
//! - **Anomaly Detection**: Detects outliers that might skew calibration
//!
//! # Example
//!
//! ```rust
//! use oxirs_arq::cost_model_calibration::{CostModelCalibrator, CalibrationConfig};
//!
//! let config = CalibrationConfig::default();
//! let mut calibrator = CostModelCalibrator::new(config);
//!
//! // Record execution samples
//! calibrator.record_scan_execution(1000, std::time::Duration::from_millis(50));
//! calibrator.record_join_execution(5000, 3000, std::time::Duration::from_millis(200));
//!
//! // Get calibrated costs
//! let scan_cost = calibrator.estimate_scan_cost(2000);
//! println!("Estimated scan cost for 2000 tuples: {}", scan_cost);
//! ```

use anyhow::Result;
// SciRS2 array types available for advanced calibration
#[allow(unused_imports)]
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Configuration for cost model calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Minimum samples required for calibration
    pub min_samples: usize,
    /// Maximum samples to keep in history
    pub max_samples: usize,
    /// Learning rate for online updates
    pub learning_rate: f64,
    /// Decay factor for older samples (0.0 - 1.0)
    pub decay_factor: f64,
    /// Outlier detection threshold (in standard deviations)
    pub outlier_threshold: f64,
    /// Minimum confidence level for calibration (0.0 - 1.0)
    pub min_confidence: f64,
    /// Enable automatic recalibration
    pub auto_recalibrate: bool,
    /// Recalibration interval (in samples)
    pub recalibration_interval: usize,
    /// Use weighted regression
    pub weighted_regression: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_samples: 10000,
            learning_rate: 0.01,
            decay_factor: 0.99,
            outlier_threshold: 3.0,
            min_confidence: 0.7,
            auto_recalibrate: true,
            recalibration_interval: 500,
            weighted_regression: true,
        }
    }
}

/// Cost model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModelParameters {
    /// Sequential scan cost per tuple
    pub seq_scan_cost: f64,
    /// Index scan cost per tuple
    pub index_scan_cost: f64,
    /// Hash join build cost per tuple
    pub hash_build_cost: f64,
    /// Hash join probe cost per tuple
    pub hash_probe_cost: f64,
    /// Sort cost per tuple (n log n factor)
    pub sort_cost: f64,
    /// Merge join cost per tuple
    pub merge_join_cost: f64,
    /// Nested loop inner cost
    pub nested_loop_cost: f64,
    /// Filter evaluation cost
    pub filter_cost: f64,
    /// Aggregate computation cost
    pub aggregate_cost: f64,
    /// Network round-trip cost (for federated queries)
    pub network_rtt_cost: f64,
    /// Materialization cost per tuple
    pub materialize_cost: f64,
}

impl Default for CostModelParameters {
    fn default() -> Self {
        Self {
            seq_scan_cost: 1.0,
            index_scan_cost: 0.1,
            hash_build_cost: 1.5,
            hash_probe_cost: 0.5,
            sort_cost: 2.0,
            merge_join_cost: 0.8,
            nested_loop_cost: 0.01,
            filter_cost: 0.1,
            aggregate_cost: 0.2,
            network_rtt_cost: 100.0,
            materialize_cost: 0.5,
        }
    }
}

/// Execution sample for calibration
#[derive(Debug, Clone)]
pub struct ExecutionSample {
    /// Operation type
    pub operation: OperationType,
    /// Input cardinality
    pub input_cardinality: u64,
    /// Output cardinality (if applicable)
    pub output_cardinality: Option<u64>,
    /// Secondary input cardinality (for joins)
    pub secondary_cardinality: Option<u64>,
    /// Actual execution time
    pub execution_time: Duration,
    /// Predicted cost (before execution)
    pub predicted_cost: Option<f64>,
    /// Timestamp of execution
    pub timestamp: Instant,
    /// Memory used (bytes)
    pub memory_used: Option<usize>,
    /// Was this query cached?
    pub cache_hit: bool,
}

/// Types of operations for calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    /// Sequential table/graph scan
    SequentialScan,
    /// Index-based lookup
    IndexScan,
    /// Hash join
    HashJoin,
    /// Sort-merge join
    MergeJoin,
    /// Nested loop join
    NestedLoopJoin,
    /// Sort operation
    Sort,
    /// Filter evaluation
    Filter,
    /// Aggregate computation
    Aggregate,
    /// Union operation
    Union,
    /// Network operation (federated)
    Network,
    /// Result materialization
    Materialize,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::SequentialScan => write!(f, "SeqScan"),
            OperationType::IndexScan => write!(f, "IndexScan"),
            OperationType::HashJoin => write!(f, "HashJoin"),
            OperationType::MergeJoin => write!(f, "MergeJoin"),
            OperationType::NestedLoopJoin => write!(f, "NLJoin"),
            OperationType::Sort => write!(f, "Sort"),
            OperationType::Filter => write!(f, "Filter"),
            OperationType::Aggregate => write!(f, "Aggregate"),
            OperationType::Union => write!(f, "Union"),
            OperationType::Network => write!(f, "Network"),
            OperationType::Materialize => write!(f, "Materialize"),
        }
    }
}

/// Calibration statistics for an operation type
#[derive(Debug, Clone, Default)]
pub struct OperationCalibrationStats {
    /// Total samples collected
    pub sample_count: u64,
    /// Current estimated coefficient
    pub coefficient: f64,
    /// Coefficient standard error
    pub standard_error: f64,
    /// R-squared (goodness of fit)
    pub r_squared: f64,
    /// Mean absolute error
    pub mae: f64,
    /// Mean percentage error
    pub mape: f64,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Last calibration timestamp
    pub last_calibration: Option<Instant>,
}

/// Cost model calibrator
pub struct CostModelCalibrator {
    /// Configuration
    config: CalibrationConfig,
    /// Current cost model parameters
    parameters: Arc<RwLock<CostModelParameters>>,
    /// Execution samples by operation type
    samples: Arc<RwLock<HashMap<OperationType, VecDeque<ExecutionSample>>>>,
    /// Calibration statistics by operation type
    stats: Arc<RwLock<HashMap<OperationType, OperationCalibrationStats>>>,
    /// Total calibrations performed
    calibration_count: AtomicU64,
    /// Samples since last calibration
    samples_since_calibration: AtomicU64,
}

impl CostModelCalibrator {
    /// Create new cost model calibrator
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            config,
            parameters: Arc::new(RwLock::new(CostModelParameters::default())),
            samples: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
            calibration_count: AtomicU64::new(0),
            samples_since_calibration: AtomicU64::new(0),
        }
    }

    /// Create with custom initial parameters
    pub fn with_parameters(config: CalibrationConfig, params: CostModelParameters) -> Self {
        Self {
            config,
            parameters: Arc::new(RwLock::new(params)),
            samples: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
            calibration_count: AtomicU64::new(0),
            samples_since_calibration: AtomicU64::new(0),
        }
    }

    /// Record an execution sample
    pub fn record_sample(&self, sample: ExecutionSample) {
        let mut samples = self.samples.write().unwrap();
        let queue = samples.entry(sample.operation).or_default();

        // Add sample
        queue.push_back(sample);

        // Remove old samples if exceeding limit
        while queue.len() > self.config.max_samples {
            queue.pop_front();
        }

        drop(samples);

        // Check if recalibration is needed
        let count = self
            .samples_since_calibration
            .fetch_add(1, Ordering::Relaxed);
        if self.config.auto_recalibrate && count >= self.config.recalibration_interval as u64 {
            let _ = self.recalibrate_all();
        }
    }

    /// Record a sequential scan execution
    pub fn record_scan_execution(&self, cardinality: u64, execution_time: Duration) {
        self.record_sample(ExecutionSample {
            operation: OperationType::SequentialScan,
            input_cardinality: cardinality,
            output_cardinality: Some(cardinality),
            secondary_cardinality: None,
            execution_time,
            predicted_cost: None,
            timestamp: Instant::now(),
            memory_used: None,
            cache_hit: false,
        });
    }

    /// Record an index scan execution
    pub fn record_index_scan_execution(
        &self,
        cardinality: u64,
        result_count: u64,
        execution_time: Duration,
    ) {
        self.record_sample(ExecutionSample {
            operation: OperationType::IndexScan,
            input_cardinality: cardinality,
            output_cardinality: Some(result_count),
            secondary_cardinality: None,
            execution_time,
            predicted_cost: None,
            timestamp: Instant::now(),
            memory_used: None,
            cache_hit: false,
        });
    }

    /// Record a hash join execution
    pub fn record_hash_join_execution(
        &self,
        left_card: u64,
        right_card: u64,
        execution_time: Duration,
    ) {
        self.record_sample(ExecutionSample {
            operation: OperationType::HashJoin,
            input_cardinality: left_card,
            output_cardinality: None,
            secondary_cardinality: Some(right_card),
            execution_time,
            predicted_cost: None,
            timestamp: Instant::now(),
            memory_used: None,
            cache_hit: false,
        });
    }

    /// Record a merge join execution
    pub fn record_merge_join_execution(
        &self,
        left_card: u64,
        right_card: u64,
        execution_time: Duration,
    ) {
        self.record_sample(ExecutionSample {
            operation: OperationType::MergeJoin,
            input_cardinality: left_card,
            output_cardinality: None,
            secondary_cardinality: Some(right_card),
            execution_time,
            predicted_cost: None,
            timestamp: Instant::now(),
            memory_used: None,
            cache_hit: false,
        });
    }

    /// Record a nested loop join execution
    pub fn record_nested_loop_execution(
        &self,
        outer_card: u64,
        inner_card: u64,
        execution_time: Duration,
    ) {
        self.record_sample(ExecutionSample {
            operation: OperationType::NestedLoopJoin,
            input_cardinality: outer_card,
            output_cardinality: None,
            secondary_cardinality: Some(inner_card),
            execution_time,
            predicted_cost: None,
            timestamp: Instant::now(),
            memory_used: None,
            cache_hit: false,
        });
    }

    /// Record a join execution (generic version)
    pub fn record_join_execution(&self, left_card: u64, right_card: u64, execution_time: Duration) {
        // Default to hash join if not specified
        self.record_hash_join_execution(left_card, right_card, execution_time);
    }

    /// Record a sort operation
    pub fn record_sort_execution(&self, cardinality: u64, execution_time: Duration) {
        self.record_sample(ExecutionSample {
            operation: OperationType::Sort,
            input_cardinality: cardinality,
            output_cardinality: Some(cardinality),
            secondary_cardinality: None,
            execution_time,
            predicted_cost: None,
            timestamp: Instant::now(),
            memory_used: None,
            cache_hit: false,
        });
    }

    /// Record a filter operation
    pub fn record_filter_execution(
        &self,
        input_card: u64,
        output_card: u64,
        execution_time: Duration,
    ) {
        self.record_sample(ExecutionSample {
            operation: OperationType::Filter,
            input_cardinality: input_card,
            output_cardinality: Some(output_card),
            secondary_cardinality: None,
            execution_time,
            predicted_cost: None,
            timestamp: Instant::now(),
            memory_used: None,
            cache_hit: false,
        });
    }

    /// Record an aggregate operation
    pub fn record_aggregate_execution(
        &self,
        input_card: u64,
        group_count: u64,
        execution_time: Duration,
    ) {
        self.record_sample(ExecutionSample {
            operation: OperationType::Aggregate,
            input_cardinality: input_card,
            output_cardinality: Some(group_count),
            secondary_cardinality: None,
            execution_time,
            predicted_cost: None,
            timestamp: Instant::now(),
            memory_used: None,
            cache_hit: false,
        });
    }

    /// Recalibrate all operation types
    pub fn recalibrate_all(&self) -> Result<CalibrationReport> {
        let mut report = CalibrationReport {
            timestamp: Instant::now(),
            operations_calibrated: Vec::new(),
            total_samples_used: 0,
            old_parameters: self.parameters.read().unwrap().clone(),
            new_parameters: CostModelParameters::default(),
            warnings: Vec::new(),
        };

        let samples = self.samples.read().unwrap();

        for (&op_type, queue) in samples.iter() {
            if queue.len() >= self.config.min_samples {
                match self.calibrate_operation(op_type, queue) {
                    Ok(stats) => {
                        report.operations_calibrated.push((op_type, stats.clone()));
                        report.total_samples_used += stats.sample_count as usize;

                        // Update stats
                        let mut all_stats = self.stats.write().unwrap();
                        all_stats.insert(op_type, stats);
                    }
                    Err(e) => {
                        report
                            .warnings
                            .push(format!("Failed to calibrate {:?}: {}", op_type, e));
                    }
                }
            } else {
                report.warnings.push(format!(
                    "Insufficient samples for {:?}: {} < {}",
                    op_type,
                    queue.len(),
                    self.config.min_samples
                ));
            }
        }

        // Store new parameters
        report.new_parameters = self.parameters.read().unwrap().clone();

        self.calibration_count.fetch_add(1, Ordering::Relaxed);
        self.samples_since_calibration.store(0, Ordering::Relaxed);

        Ok(report)
    }

    /// Calibrate a specific operation type
    fn calibrate_operation(
        &self,
        op_type: OperationType,
        samples: &VecDeque<ExecutionSample>,
    ) -> Result<OperationCalibrationStats> {
        // Collect data points
        let mut x_values: Vec<f64> = Vec::with_capacity(samples.len());
        let mut y_values: Vec<f64> = Vec::with_capacity(samples.len());
        let mut weights: Vec<f64> = Vec::with_capacity(samples.len());

        let now = Instant::now();

        for sample in samples.iter() {
            // Filter out cache hits (they don't represent true cost)
            if sample.cache_hit {
                continue;
            }

            // Calculate feature value based on operation type
            let x = match op_type {
                OperationType::SequentialScan | OperationType::IndexScan => {
                    sample.input_cardinality as f64
                }
                OperationType::HashJoin => {
                    let left = sample.input_cardinality as f64;
                    let right = sample.secondary_cardinality.unwrap_or(0) as f64;
                    left + right // Simplified: build + probe
                }
                OperationType::MergeJoin => {
                    let left = sample.input_cardinality as f64;
                    let right = sample.secondary_cardinality.unwrap_or(0) as f64;
                    left + right
                }
                OperationType::NestedLoopJoin => {
                    let outer = sample.input_cardinality as f64;
                    let inner = sample.secondary_cardinality.unwrap_or(0) as f64;
                    outer * inner
                }
                OperationType::Sort => {
                    let n = sample.input_cardinality as f64;
                    n * (n.max(1.0)).ln() // n log n
                }
                OperationType::Filter | OperationType::Aggregate | OperationType::Union => {
                    sample.input_cardinality as f64
                }
                OperationType::Network => 1.0, // Per call
                OperationType::Materialize => sample.input_cardinality as f64,
            };

            let y = sample.execution_time.as_secs_f64() * 1_000_000.0; // Convert to microseconds

            // Calculate weight (more recent = higher weight)
            let age = now.duration_since(sample.timestamp).as_secs_f64();
            let weight = self.config.decay_factor.powf(age / 60.0); // Decay per minute

            if x > 0.0 && y > 0.0 && weight > 0.01 {
                x_values.push(x);
                y_values.push(y);
                weights.push(if self.config.weighted_regression {
                    weight
                } else {
                    1.0
                });
            }
        }

        if x_values.len() < self.config.min_samples / 2 {
            anyhow::bail!("Insufficient valid samples after filtering");
        }

        // Perform weighted linear regression
        let (coefficient, _intercept, _r_squared) = if self.config.weighted_regression {
            weighted_linear_regression(&x_values, &y_values, &weights)?
        } else {
            simple_linear_regression(&x_values, &y_values)?
        };

        // Detect and remove outliers
        let (cleaned_coef, cleaned_rsq, _outlier_count) =
            self.remove_outliers_and_refit(&x_values, &y_values, &weights, coefficient)?;

        // Calculate error metrics
        let (mae, mape) = calculate_error_metrics(&x_values, &y_values, cleaned_coef);

        // Calculate confidence based on R², sample count, and error metrics
        let confidence = self.calculate_confidence(cleaned_rsq, x_values.len(), mape);

        // Update cost model parameters if confidence is high enough
        if confidence >= self.config.min_confidence {
            self.update_parameter(op_type, cleaned_coef);
        }

        Ok(OperationCalibrationStats {
            sample_count: x_values.len() as u64,
            coefficient: cleaned_coef,
            standard_error: calculate_standard_error(&x_values, &y_values, cleaned_coef),
            r_squared: cleaned_rsq,
            mae,
            mape,
            confidence,
            last_calibration: Some(Instant::now()),
        })
    }

    /// Remove outliers and refit the model
    fn remove_outliers_and_refit(
        &self,
        x_values: &[f64],
        y_values: &[f64],
        weights: &[f64],
        initial_coef: f64,
    ) -> Result<(f64, f64, usize)> {
        // Calculate residuals
        let residuals: Vec<f64> = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(&x, &y)| y - x * initial_coef)
            .collect();

        // Calculate mean and std of residuals
        let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let std_residual = (residuals
            .iter()
            .map(|r| (r - mean_residual).powi(2))
            .sum::<f64>()
            / residuals.len() as f64)
            .sqrt();

        // Filter outliers
        let mut filtered_x = Vec::new();
        let mut filtered_y = Vec::new();
        let mut filtered_w = Vec::new();
        let mut outlier_count = 0;

        for i in 0..residuals.len() {
            let z_score = (residuals[i] - mean_residual).abs() / std_residual.max(1e-10);
            if z_score <= self.config.outlier_threshold {
                filtered_x.push(x_values[i]);
                filtered_y.push(y_values[i]);
                filtered_w.push(weights[i]);
            } else {
                outlier_count += 1;
            }
        }

        if filtered_x.len() < self.config.min_samples / 4 {
            // Too many outliers removed, use original
            return Ok((initial_coef, 0.5, outlier_count));
        }

        // Refit with cleaned data
        let (new_coef, _, new_rsq) = if self.config.weighted_regression {
            weighted_linear_regression(&filtered_x, &filtered_y, &filtered_w)?
        } else {
            simple_linear_regression(&filtered_x, &filtered_y)?
        };

        Ok((new_coef, new_rsq, outlier_count))
    }

    /// Calculate confidence level for calibration
    fn calculate_confidence(&self, r_squared: f64, sample_count: usize, mape: f64) -> f64 {
        // Factors contributing to confidence:
        // 1. R² (goodness of fit) - 40%
        // 2. Sample count relative to min_samples - 30%
        // 3. MAPE (mean absolute percentage error) - 30%

        let rsq_factor = r_squared * 0.4;

        let sample_factor = {
            let ratio = sample_count as f64 / self.config.min_samples as f64;
            (ratio.min(5.0) / 5.0) * 0.3 // Cap at 5x min_samples
        };

        let mape_factor = {
            let mape_quality = 1.0 - (mape / 100.0).min(1.0); // Lower MAPE = better
            mape_quality * 0.3
        };

        (rsq_factor + sample_factor + mape_factor).clamp(0.0, 1.0)
    }

    /// Update a specific cost model parameter
    fn update_parameter(&self, op_type: OperationType, coefficient: f64) {
        let mut params = self.parameters.write().unwrap();

        // Convert coefficient from microseconds back to relative cost units
        // We use a base of 1 microsecond = 0.001 cost units
        let cost_value = coefficient * 0.001;

        match op_type {
            OperationType::SequentialScan => params.seq_scan_cost = cost_value,
            OperationType::IndexScan => params.index_scan_cost = cost_value,
            OperationType::HashJoin => {
                // Split between build and probe (roughly 75% build, 25% probe)
                params.hash_build_cost = cost_value * 0.75;
                params.hash_probe_cost = cost_value * 0.25;
            }
            OperationType::MergeJoin => params.merge_join_cost = cost_value,
            OperationType::NestedLoopJoin => params.nested_loop_cost = cost_value,
            OperationType::Sort => params.sort_cost = cost_value,
            OperationType::Filter => params.filter_cost = cost_value,
            OperationType::Aggregate => params.aggregate_cost = cost_value,
            OperationType::Network => params.network_rtt_cost = cost_value,
            OperationType::Materialize => params.materialize_cost = cost_value,
            OperationType::Union => {} // No specific parameter for union
        }
    }

    /// Estimate scan cost for given cardinality
    pub fn estimate_scan_cost(&self, cardinality: u64) -> f64 {
        let params = self.parameters.read().unwrap();
        cardinality as f64 * params.seq_scan_cost
    }

    /// Estimate index scan cost
    pub fn estimate_index_scan_cost(&self, index_cardinality: u64, selectivity: f64) -> f64 {
        let params = self.parameters.read().unwrap();
        let result_rows = (index_cardinality as f64 * selectivity).max(1.0);
        result_rows * params.index_scan_cost
    }

    /// Estimate hash join cost
    pub fn estimate_hash_join_cost(&self, build_size: u64, probe_size: u64) -> f64 {
        let params = self.parameters.read().unwrap();
        (build_size as f64 * params.hash_build_cost) + (probe_size as f64 * params.hash_probe_cost)
    }

    /// Estimate merge join cost
    pub fn estimate_merge_join_cost(&self, left_size: u64, right_size: u64) -> f64 {
        let params = self.parameters.read().unwrap();
        (left_size + right_size) as f64 * params.merge_join_cost
    }

    /// Estimate nested loop join cost
    pub fn estimate_nested_loop_cost(&self, outer_size: u64, inner_size: u64) -> f64 {
        let params = self.parameters.read().unwrap();
        (outer_size as f64 * inner_size as f64) * params.nested_loop_cost
    }

    /// Estimate sort cost (n log n)
    pub fn estimate_sort_cost(&self, cardinality: u64) -> f64 {
        let params = self.parameters.read().unwrap();
        let n = cardinality as f64;
        n * (n.max(1.0)).ln() * params.sort_cost
    }

    /// Estimate filter cost
    pub fn estimate_filter_cost(&self, input_cardinality: u64) -> f64 {
        let params = self.parameters.read().unwrap();
        input_cardinality as f64 * params.filter_cost
    }

    /// Estimate aggregate cost
    pub fn estimate_aggregate_cost(&self, input_cardinality: u64) -> f64 {
        let params = self.parameters.read().unwrap();
        input_cardinality as f64 * params.aggregate_cost
    }

    /// Get current cost model parameters
    pub fn parameters(&self) -> CostModelParameters {
        self.parameters.read().unwrap().clone()
    }

    /// Get calibration statistics for an operation
    pub fn get_operation_stats(&self, op_type: OperationType) -> Option<OperationCalibrationStats> {
        self.stats.read().unwrap().get(&op_type).cloned()
    }

    /// Get sample count for an operation
    pub fn sample_count(&self, op_type: OperationType) -> usize {
        self.samples
            .read()
            .unwrap()
            .get(&op_type)
            .map(|q| q.len())
            .unwrap_or(0)
    }

    /// Get total calibration count
    pub fn calibration_count(&self) -> u64 {
        self.calibration_count.load(Ordering::Relaxed)
    }

    /// Get comprehensive statistics
    pub fn statistics(&self) -> CalibratorStatistics {
        let samples = self.samples.read().unwrap();
        let stats = self.stats.read().unwrap();

        let mut operation_stats = HashMap::new();
        let mut total_samples = 0;

        for (&op_type, queue) in samples.iter() {
            let sample_count = queue.len();
            total_samples += sample_count;

            operation_stats.insert(
                op_type,
                OperationSummary {
                    sample_count,
                    calibration_stats: stats.get(&op_type).cloned(),
                },
            );
        }

        CalibratorStatistics {
            total_samples,
            calibration_count: self.calibration_count.load(Ordering::Relaxed),
            samples_since_last_calibration: self.samples_since_calibration.load(Ordering::Relaxed),
            operation_stats,
            current_parameters: self.parameters.read().unwrap().clone(),
        }
    }

    /// Reset calibration data
    pub fn reset(&self) {
        self.samples.write().unwrap().clear();
        self.stats.write().unwrap().clear();
        *self.parameters.write().unwrap() = CostModelParameters::default();
        self.calibration_count.store(0, Ordering::Relaxed);
        self.samples_since_calibration.store(0, Ordering::Relaxed);
    }

    /// Export calibration data for persistence
    pub fn export(&self) -> CalibrationExport {
        CalibrationExport {
            parameters: self.parameters.read().unwrap().clone(),
            stats: self
                .stats
                .read()
                .unwrap()
                .iter()
                .map(|(&k, v)| {
                    (
                        k,
                        OperationCalibrationExport {
                            sample_count: v.sample_count,
                            coefficient: v.coefficient,
                            r_squared: v.r_squared,
                            confidence: v.confidence,
                        },
                    )
                })
                .collect(),
            calibration_count: self.calibration_count.load(Ordering::Relaxed),
        }
    }

    /// Import calibration data
    pub fn import(&self, data: CalibrationExport) {
        *self.parameters.write().unwrap() = data.parameters;

        let mut stats = self.stats.write().unwrap();
        for (op_type, export) in data.stats {
            stats.insert(
                op_type,
                OperationCalibrationStats {
                    sample_count: export.sample_count,
                    coefficient: export.coefficient,
                    standard_error: 0.0,
                    r_squared: export.r_squared,
                    mae: 0.0,
                    mape: 0.0,
                    confidence: export.confidence,
                    last_calibration: None,
                },
            );
        }

        self.calibration_count
            .store(data.calibration_count, Ordering::Relaxed);
    }
}

/// Calibration report
#[derive(Debug)]
pub struct CalibrationReport {
    /// When calibration was performed
    pub timestamp: Instant,
    /// Operations that were calibrated
    pub operations_calibrated: Vec<(OperationType, OperationCalibrationStats)>,
    /// Total samples used
    pub total_samples_used: usize,
    /// Previous parameters
    pub old_parameters: CostModelParameters,
    /// New parameters
    pub new_parameters: CostModelParameters,
    /// Warnings during calibration
    pub warnings: Vec<String>,
}

/// Calibrator statistics
#[derive(Debug, Clone)]
pub struct CalibratorStatistics {
    /// Total samples across all operations
    pub total_samples: usize,
    /// Number of calibrations performed
    pub calibration_count: u64,
    /// Samples since last calibration
    pub samples_since_last_calibration: u64,
    /// Per-operation statistics
    pub operation_stats: HashMap<OperationType, OperationSummary>,
    /// Current cost model parameters
    pub current_parameters: CostModelParameters,
}

/// Summary for a single operation type
#[derive(Debug, Clone)]
pub struct OperationSummary {
    /// Number of samples
    pub sample_count: usize,
    /// Calibration statistics (if available)
    pub calibration_stats: Option<OperationCalibrationStats>,
}

/// Export format for calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationExport {
    /// Cost model parameters
    pub parameters: CostModelParameters,
    /// Operation statistics
    pub stats: HashMap<OperationType, OperationCalibrationExport>,
    /// Calibration count
    pub calibration_count: u64,
}

/// Export format for operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationCalibrationExport {
    /// Sample count
    pub sample_count: u64,
    /// Coefficient
    pub coefficient: f64,
    /// R-squared
    pub r_squared: f64,
    /// Confidence
    pub confidence: f64,
}

/// Simple linear regression (y = ax + b)
fn simple_linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
    let n = x.len() as f64;
    if n < 2.0 {
        anyhow::bail!("Need at least 2 data points");
    }

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();

    let denominator = n * sum_x2 - sum_x * sum_x;
    if denominator.abs() < 1e-10 {
        anyhow::bail!("Singular matrix in regression");
    }

    let a = (n * sum_xy - sum_x * sum_y) / denominator;
    let b = (sum_y - a * sum_x) / n;

    // Calculate R²
    let y_mean = sum_y / n;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi - (a * xi + b)).powi(2))
        .sum();

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Ok((a.max(0.0), b, r_squared.clamp(0.0, 1.0)))
}

/// Weighted linear regression
fn weighted_linear_regression(x: &[f64], y: &[f64], w: &[f64]) -> Result<(f64, f64, f64)> {
    let sum_w: f64 = w.iter().sum();
    if sum_w < 1e-10 {
        anyhow::bail!("Sum of weights is too small");
    }

    let sum_wx: f64 = x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * xi).sum();
    let sum_wy: f64 = y.iter().zip(w.iter()).map(|(&yi, &wi)| wi * yi).sum();
    let sum_wxy: f64 = x
        .iter()
        .zip(y.iter())
        .zip(w.iter())
        .map(|((&xi, &yi), &wi)| wi * xi * yi)
        .sum();
    let sum_wx2: f64 = x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * xi * xi).sum();

    let denominator = sum_w * sum_wx2 - sum_wx * sum_wx;
    if denominator.abs() < 1e-10 {
        anyhow::bail!("Singular matrix in weighted regression");
    }

    let a = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator;
    let b = (sum_wy - a * sum_wx) / sum_w;

    // Calculate weighted R²
    let y_mean = sum_wy / sum_w;
    let ss_tot: f64 = y
        .iter()
        .zip(w.iter())
        .map(|(&yi, &wi)| wi * (yi - y_mean).powi(2))
        .sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .zip(w.iter())
        .map(|((&xi, &yi), &wi)| wi * (yi - (a * xi + b)).powi(2))
        .sum();

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Ok((a.max(0.0), b, r_squared.clamp(0.0, 1.0)))
}

/// Calculate error metrics
fn calculate_error_metrics(x: &[f64], y: &[f64], coefficient: f64) -> (f64, f64) {
    let mut mae_sum = 0.0;
    let mut mape_sum = 0.0;
    let mut valid_count = 0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let predicted = xi * coefficient;
        let error = (yi - predicted).abs();
        mae_sum += error;

        if yi.abs() > 1e-10 {
            mape_sum += error / yi.abs();
            valid_count += 1;
        }
    }

    let n = x.len() as f64;
    let mae = mae_sum / n;
    let mape = if valid_count > 0 {
        (mape_sum / valid_count as f64) * 100.0
    } else {
        100.0
    };

    (mae, mape)
}

/// Calculate standard error of coefficient
fn calculate_standard_error(x: &[f64], y: &[f64], coefficient: f64) -> f64 {
    let n = x.len() as f64;
    if n <= 2.0 {
        return f64::INFINITY;
    }

    // Calculate residual sum of squares
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi - xi * coefficient).powi(2))
        .sum();

    // Calculate sum of squares of x
    let x_mean = x.iter().sum::<f64>() / n;
    let ss_x: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();

    if ss_x < 1e-10 {
        return f64::INFINITY;
    }

    let mse = ss_res / (n - 2.0);
    (mse / ss_x).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrator_creation() {
        let config = CalibrationConfig::default();
        let calibrator = CostModelCalibrator::new(config);

        assert_eq!(calibrator.calibration_count(), 0);
        assert_eq!(calibrator.sample_count(OperationType::SequentialScan), 0);
    }

    #[test]
    fn test_record_samples() {
        let config = CalibrationConfig {
            min_samples: 10,
            ..Default::default()
        };
        let calibrator = CostModelCalibrator::new(config);

        // Record some scan executions
        for i in 0..20 {
            calibrator.record_scan_execution((i + 1) * 100, Duration::from_micros((i + 1) * 10));
        }

        assert_eq!(calibrator.sample_count(OperationType::SequentialScan), 20);
    }

    #[test]
    fn test_simple_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let (a, b, r_squared) = simple_linear_regression(&x, &y).unwrap();

        assert!((a - 2.0).abs() < 0.01, "Expected coefficient ~2, got {}", a);
        assert!((b - 0.0).abs() < 0.01, "Expected intercept ~0, got {}", b);
        assert!(
            (r_squared - 1.0).abs() < 0.01,
            "Expected R² ~1, got {}",
            r_squared
        );
    }

    #[test]
    fn test_weighted_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let (a, _b, r_squared) = weighted_linear_regression(&x, &y, &w).unwrap();

        assert!((a - 2.0).abs() < 0.01);
        assert!((r_squared - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_costs() {
        let config = CalibrationConfig::default();
        let calibrator = CostModelCalibrator::new(config);

        // Default parameters should give reasonable estimates
        let scan_cost = calibrator.estimate_scan_cost(1000);
        assert!(scan_cost > 0.0);

        let join_cost = calibrator.estimate_hash_join_cost(1000, 500);
        assert!(join_cost > 0.0);

        let sort_cost = calibrator.estimate_sort_cost(1000);
        assert!(sort_cost > 0.0);
    }

    #[test]
    fn test_calibration_with_sufficient_samples() {
        let config = CalibrationConfig {
            min_samples: 10,
            auto_recalibrate: false,
            ..Default::default()
        };
        let calibrator = CostModelCalibrator::new(config);

        // Generate linear data: time = 0.1 * cardinality
        for i in 1..=50 {
            let cardinality = i as u64 * 100;
            let time = Duration::from_micros(cardinality / 10);
            calibrator.record_scan_execution(cardinality, time);
        }

        let report = calibrator.recalibrate_all().unwrap();
        assert!(!report.operations_calibrated.is_empty());
    }

    #[test]
    fn test_export_import() {
        let config = CalibrationConfig::default();
        let calibrator = CostModelCalibrator::new(config.clone());

        // Modify parameters
        {
            let mut params = calibrator.parameters.write().unwrap();
            params.seq_scan_cost = 2.5;
        }

        let export = calibrator.export();
        assert_eq!(export.parameters.seq_scan_cost, 2.5);

        // Import to new calibrator
        let calibrator2 = CostModelCalibrator::new(config);
        calibrator2.import(export);

        assert_eq!(calibrator2.parameters().seq_scan_cost, 2.5);
    }

    #[test]
    fn test_statistics() {
        let config = CalibrationConfig {
            min_samples: 5,
            ..Default::default()
        };
        let calibrator = CostModelCalibrator::new(config);

        for i in 0..10 {
            calibrator.record_scan_execution((i + 1) * 100, Duration::from_micros((i + 1) * 10));
        }

        let stats = calibrator.statistics();
        assert_eq!(stats.total_samples, 10);
        assert!(stats
            .operation_stats
            .contains_key(&OperationType::SequentialScan));
    }

    #[test]
    fn test_reset() {
        let config = CalibrationConfig {
            min_samples: 5,
            ..Default::default()
        };
        let calibrator = CostModelCalibrator::new(config);

        for i in 0..10 {
            calibrator.record_scan_execution((i + 1) * 100, Duration::from_micros((i + 1) * 10));
        }

        assert_eq!(calibrator.sample_count(OperationType::SequentialScan), 10);

        calibrator.reset();

        assert_eq!(calibrator.sample_count(OperationType::SequentialScan), 0);
        assert_eq!(calibrator.calibration_count(), 0);
    }

    #[test]
    fn test_multiple_operation_types() {
        let config = CalibrationConfig {
            min_samples: 5,
            auto_recalibrate: false,
            ..Default::default()
        };
        let calibrator = CostModelCalibrator::new(config);

        // Record different operation types
        for i in 1..=10 {
            calibrator.record_scan_execution(i * 100, Duration::from_micros(i * 10));
            calibrator.record_hash_join_execution(i * 50, i * 50, Duration::from_micros(i * 20));
            calibrator.record_sort_execution(i * 100, Duration::from_micros(i * 15));
        }

        let stats = calibrator.statistics();
        assert_eq!(stats.operation_stats.len(), 3);
        assert_eq!(
            stats.operation_stats[&OperationType::SequentialScan].sample_count,
            10
        );
        assert_eq!(
            stats.operation_stats[&OperationType::HashJoin].sample_count,
            10
        );
        assert_eq!(stats.operation_stats[&OperationType::Sort].sample_count, 10);
    }

    #[test]
    fn test_confidence_calculation() {
        let config = CalibrationConfig::default();
        let calibrator = CostModelCalibrator::new(config);

        // High confidence: high R², many samples, low MAPE
        let high_conf = calibrator.calculate_confidence(0.95, 500, 5.0);
        assert!(high_conf > 0.7);

        // Low confidence: low R², few samples, high MAPE
        let low_conf = calibrator.calculate_confidence(0.3, 50, 50.0);
        assert!(low_conf < 0.5);
    }

    #[test]
    fn test_error_metrics() {
        let x = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0]; // y = 0.1x
        let coefficient = 0.1;

        let (mae, mape) = calculate_error_metrics(&x, &y, coefficient);

        assert!(mae < 1.0, "MAE should be very small for perfect fit");
        assert!(mape < 1.0, "MAPE should be very small for perfect fit");
    }

    #[test]
    fn test_standard_error() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let coefficient = 2.0;

        let se = calculate_standard_error(&x, &y, coefficient);

        // For perfect fit, SE should be very small
        assert!(
            se < 0.1 || se == f64::INFINITY,
            "SE for perfect fit should be ~0"
        );
    }

    #[test]
    fn test_with_custom_parameters() {
        let config = CalibrationConfig::default();
        let custom_params = CostModelParameters {
            seq_scan_cost: 0.5,
            index_scan_cost: 0.05,
            ..Default::default()
        };

        let calibrator = CostModelCalibrator::with_parameters(config, custom_params);

        assert_eq!(calibrator.parameters().seq_scan_cost, 0.5);
        assert_eq!(calibrator.parameters().index_scan_cost, 0.05);
    }

    #[test]
    fn test_join_execution_recording() {
        let config = CalibrationConfig::default();
        let calibrator = CostModelCalibrator::new(config);

        // Test generic join recording
        calibrator.record_join_execution(1000, 500, Duration::from_millis(10));

        assert_eq!(calibrator.sample_count(OperationType::HashJoin), 1);
    }
}
