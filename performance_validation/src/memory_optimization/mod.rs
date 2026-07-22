//! Advanced Memory Optimization Engine
//!
//! This module provides sophisticated memory optimization techniques including:
//! - Adaptive memory pooling with pressure-aware sizing
//! - Smart caching strategies with profiler integration
//! - Real-time memory pressure detection and mitigation
//! - Intelligent garbage collection coordination
//! - Memory-aware workload scheduling

use crate::profiling::memory_profiler::{MemoryProfiler, MemoryReport};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use serde::{Deserialize, Serialize};
use parking_lot::RwLock as ParkingRwLock;

pub mod adaptive_pools;
pub mod smart_cache;
pub mod pressure_mitigation;
pub mod gc_coordination;

/// Advanced memory optimization engine that coordinates all memory management strategies
#[derive(Debug)]
pub struct AdvancedMemoryOptimizer {
    /// Adaptive memory pools
    adaptive_pools: Arc<RwLock<adaptive_pools::AdaptivePoolManager>>,

    /// Smart cache manager
    cache_manager: Arc<RwLock<smart_cache::SmartCacheManager>>,

    /// Memory pressure mitigation system
    pressure_mitigator: Arc<Mutex<pressure_mitigation::PressureMitigator>>,

    /// Garbage collection coordinator
    gc_coordinator: Arc<Mutex<gc_coordination::GCCoordinator>>,

    /// Memory profiler integration
    memory_profiler: Arc<RwLock<MemoryProfiler>>,

    /// Configuration
    config: MemoryOptimizerConfig,

    /// Performance statistics
    stats: Arc<RwLock<OptimizationStats>>,

    /// Optimization trigger
    optimization_trigger: Arc<Notify>,

    /// Current optimization state
    optimization_state: Arc<RwLock<OptimizationState>>,

    /// Rolling window of recent real memory-usage samples (MB, oldest
    /// first), used by `calculate_memory_trend` to detect a genuine trend
    /// via linear regression instead of a hardcoded placeholder.
    usage_history: Arc<RwLock<VecDeque<f64>>>,
}

/// Maximum number of recent usage samples retained for trend detection.
const TREND_HISTORY_CAPACITY: usize = 30;

/// Minimum number of samples required before a trend judgement is made.
/// With fewer samples than this there isn't enough real signal, so the
/// trend is honestly reported as `Stable` rather than guessed from noise.
const TREND_MIN_SAMPLES: usize = 4;

/// Relative slope (as a fraction of mean usage, per sample) above which the
/// trend is classified `Increasing`/`Decreasing` rather than `Stable`.
const TREND_SLOPE_THRESHOLD: f64 = 0.02;

/// Configuration for the memory optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizerConfig {
    /// Enable adaptive memory pooling
    pub enable_adaptive_pools: bool,

    /// Enable smart caching
    pub enable_smart_cache: bool,

    /// Enable pressure mitigation
    pub enable_pressure_mitigation: bool,

    /// Enable GC coordination
    pub enable_gc_coordination: bool,

    /// Memory pressure thresholds
    pub pressure_thresholds: PressureThresholds,

    /// Optimization intervals
    pub optimization_intervals: OptimizationIntervals,

    /// Pool configuration
    pub pool_config: adaptive_pools::AdaptivePoolConfig,

    /// Cache configuration
    pub cache_config: smart_cache::SmartCacheConfig,

    /// Maximum memory usage before emergency actions (bytes)
    pub emergency_memory_threshold: usize,

    /// Target memory usage for normal operations (bytes)
    pub target_memory_usage: usize,
}

/// Memory pressure thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureThresholds {
    /// Warning threshold (0.0-1.0)
    pub warning: f64,

    /// Critical threshold (0.0-1.0)
    pub critical: f64,

    /// Emergency threshold (0.0-1.0)
    pub emergency: f64,

    /// Recovery threshold (0.0-1.0)
    pub recovery: f64,
}

/// Optimization intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationIntervals {
    /// Continuous monitoring interval
    pub monitoring: Duration,

    /// Lightweight optimization interval
    pub lightweight: Duration,

    /// Full optimization interval
    pub full: Duration,

    /// Emergency optimization interval
    pub emergency: Duration,
}

/// Current optimization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationState {
    /// Current memory pressure level
    pub pressure_level: MemoryPressureLevel,

    /// Active optimization operations
    pub active_operations: Vec<OptimizationOperation>,

    /// Last optimization timestamp
    pub last_optimization: Option<Instant>,

    /// Memory usage trend
    pub memory_trend: MemoryTrend,

    /// Predicted memory exhaustion time
    pub predicted_exhaustion: Option<Duration>,
}

/// Memory pressure levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Normal,
    Warning,
    Critical,
    Emergency,
}

/// Memory usage trends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTrend {
    Stable,
    Increasing,
    Decreasing,
    Oscillating,
}

/// Optimization operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOperation {
    pub operation_type: OptimizationType,
    pub start_time: Instant,
    pub estimated_duration: Duration,
    pub priority: OptimizationPriority,
}

/// Types of optimization operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    PoolCompaction,
    CacheEviction,
    GarbageCollection,
    MemoryDefragmentation,
    WorkloadRebalancing,
    EmergencyCleanup,
}

/// Optimization priorities
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

/// Comprehensive optimization statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Total optimizations performed
    pub total_optimizations: u64,

    /// Memory saved by optimizations (bytes)
    pub total_memory_saved: u64,

    /// Time spent on optimizations
    pub total_optimization_time: Duration,

    /// Pool optimization statistics
    pub pool_stats: adaptive_pools::PoolOptimizationStats,

    /// Cache optimization statistics
    pub cache_stats: smart_cache::CacheOptimizationStats,

    /// Pressure mitigation statistics
    pub pressure_stats: pressure_mitigation::MitigationStats,

    /// GC coordination statistics
    pub gc_stats: gc_coordination::GCStats,

    /// Average memory pressure
    pub average_pressure: f64,

    /// Peak memory usage
    pub peak_memory_usage: u64,

    /// Memory efficiency score (0.0-1.0)
    pub efficiency_score: f64,
}

impl Default for MemoryOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_pools: true,
            enable_smart_cache: true,
            enable_pressure_mitigation: true,
            enable_gc_coordination: true,
            pressure_thresholds: PressureThresholds {
                warning: 0.7,
                critical: 0.85,
                emergency: 0.95,
                recovery: 0.6,
            },
            optimization_intervals: OptimizationIntervals {
                monitoring: Duration::from_millis(100),
                lightweight: Duration::from_secs(30),
                full: Duration::from_secs(300),
                emergency: Duration::from_millis(500),
            },
            pool_config: adaptive_pools::AdaptivePoolConfig::default(),
            cache_config: smart_cache::SmartCacheConfig::default(),
            emergency_memory_threshold: 8 * 1024 * 1024 * 1024, // 8GB
            target_memory_usage: 4 * 1024 * 1024 * 1024, // 4GB
        }
    }
}

/// Ordinary-least-squares fit of a real usage-history sample sequence (MB,
/// oldest first, evenly spaced at the monitoring interval).
#[derive(Debug, Clone, Copy)]
struct UsageTrendFit {
    /// MB change per sample (the raw OLS slope).
    slope_mb_per_sample: f64,
    /// `slope_mb_per_sample` normalized by mean usage, making the judgement
    /// scale-independent (works whether usage is in the tens of MB or
    /// hundreds of GB).
    relative_slope: f64,
    /// Residual standard deviation around the fitted line, normalized by
    /// mean usage -- large values mean the samples bounce around without a
    /// clean monotonic direction.
    relative_noise: f64,
}

impl AdvancedMemoryOptimizer {
    /// Create a new advanced memory optimizer
    pub fn new(config: MemoryOptimizerConfig) -> Self {
        let memory_profiler = Arc::new(RwLock::new(MemoryProfiler::new()));

        Self {
            adaptive_pools: Arc::new(RwLock::new(
                adaptive_pools::AdaptivePoolManager::new(config.pool_config.clone())
            )),
            cache_manager: Arc::new(RwLock::new(
                smart_cache::SmartCacheManager::new(config.cache_config.clone())
            )),
            pressure_mitigator: Arc::new(Mutex::new(
                pressure_mitigation::PressureMitigator::new(
                    config.pressure_thresholds.clone(),
                    memory_profiler.clone()
                )
            )),
            gc_coordinator: Arc::new(Mutex::new(
                gc_coordination::GCCoordinator::new()
            )),
            memory_profiler,
            config,
            stats: Arc::new(RwLock::new(OptimizationStats::default())),
            optimization_trigger: Arc::new(Notify::new()),
            optimization_state: Arc::new(RwLock::new(OptimizationState {
                pressure_level: MemoryPressureLevel::Normal,
                active_operations: Vec::new(),
                last_optimization: None,
                memory_trend: MemoryTrend::Stable,
                predicted_exhaustion: None,
            })),
            usage_history: Arc::new(RwLock::new(VecDeque::with_capacity(TREND_HISTORY_CAPACITY))),
        }
    }

    /// Start the optimization engine
    pub async fn start(&self) -> Result<(), MemoryOptimizationError> {
        // Start monitoring task
        let monitor_task = self.start_monitoring_task();

        // Start optimization task
        let optimization_task = self.start_optimization_task();

        // Start both tasks concurrently
        tokio::try_join!(monitor_task, optimization_task)?;

        Ok(())
    }

    /// Start memory monitoring task
    async fn start_monitoring_task(&self) -> Result<(), MemoryOptimizationError> {
        let profiler = self.memory_profiler.clone();
        let state = self.optimization_state.clone();
        let config = self.config.clone();
        let trigger = self.optimization_trigger.clone();
        let usage_history = self.usage_history.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.optimization_intervals.monitoring);

            loop {
                interval.tick().await;

                // Get current memory usage
                let memory_report = {
                    let mut profiler_guard = profiler.write().expect("lock should not be poisoned");
                    profiler_guard.generate_report()
                };

                // `average_usage_mb` is the real, already-measured usage
                // figure the profiler reports (MB); convert to bytes for
                // the pressure ratio below.
                let current_usage_mb = memory_report.average_usage_mb;
                let current_usage = (current_usage_mb * 1024.0 * 1024.0) as u64;
                let pressure = current_usage as f64 / config.emergency_memory_threshold as f64;

                // Record this real sample into the rolling trend-detection
                // history (bounded so it never grows unboundedly).
                let history_snapshot: Vec<f64> = {
                    let mut history_guard =
                        usage_history.write().expect("lock should not be poisoned");
                    if history_guard.len() >= TREND_HISTORY_CAPACITY {
                        history_guard.pop_front();
                    }
                    history_guard.push_back(current_usage_mb);
                    history_guard.iter().copied().collect()
                };

                // Update optimization state
                let mut state_guard = state.write().expect("lock should not be poisoned");
                let old_pressure = state_guard.pressure_level.clone();

                state_guard.pressure_level = if pressure >= config.pressure_thresholds.emergency {
                    MemoryPressureLevel::Emergency
                } else if pressure >= config.pressure_thresholds.critical {
                    MemoryPressureLevel::Critical
                } else if pressure >= config.pressure_thresholds.warning {
                    MemoryPressureLevel::Warning
                } else {
                    MemoryPressureLevel::Normal
                };

                // Calculate memory trend from the real, accumulated history
                // of measured usage samples (never a hardcoded placeholder).
                state_guard.memory_trend = Self::calculate_memory_trend(&history_snapshot);

                // Predict memory exhaustion from the same real, fitted
                // growth rate (rather than a hardcoded 1MB/s placeholder).
                state_guard.predicted_exhaustion = Self::predict_memory_exhaustion(
                    current_usage,
                    config.emergency_memory_threshold,
                    &state_guard.memory_trend,
                    &history_snapshot,
                    config.optimization_intervals.monitoring,
                );

                // Trigger optimization if pressure increased
                if state_guard.pressure_level != old_pressure &&
                   matches!(state_guard.pressure_level,
                           MemoryPressureLevel::Warning |
                           MemoryPressureLevel::Critical |
                           MemoryPressureLevel::Emergency) {
                    trigger.notify_one();
                }
            }
        });

        Ok(())
    }

    /// Start optimization task
    async fn start_optimization_task(&self) -> Result<(), MemoryOptimizationError> {
        let state = self.optimization_state.clone();
        let trigger = self.optimization_trigger.clone();
        let config = self.config.clone();
        let adaptive_pools = self.adaptive_pools.clone();
        let cache_manager = self.cache_manager.clone();
        let pressure_mitigator = self.pressure_mitigator.clone();
        let gc_coordinator = self.gc_coordinator.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            loop {
                // Wait for optimization trigger or timeout
                let timeout = tokio::time::sleep(config.optimization_intervals.lightweight);
                tokio::select! {
                    _ = trigger.notified() => {
                        // Immediate optimization needed
                        Self::perform_optimization(
                            &state,
                            &config,
                            &adaptive_pools,
                            &cache_manager,
                            &pressure_mitigator,
                            &gc_coordinator,
                            &stats,
                            true // urgent
                        ).await;
                    }
                    _ = timeout => {
                        // Regular optimization
                        Self::perform_optimization(
                            &state,
                            &config,
                            &adaptive_pools,
                            &cache_manager,
                            &pressure_mitigator,
                            &gc_coordinator,
                            &stats,
                            false // not urgent
                        ).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Perform optimization operations
    async fn perform_optimization(
        state: &Arc<RwLock<OptimizationState>>,
        config: &MemoryOptimizerConfig,
        adaptive_pools: &Arc<RwLock<adaptive_pools::AdaptivePoolManager>>,
        cache_manager: &Arc<RwLock<smart_cache::SmartCacheManager>>,
        pressure_mitigator: &Arc<Mutex<pressure_mitigation::PressureMitigator>>,
        gc_coordinator: &Arc<Mutex<gc_coordination::GCCoordinator>>,
        stats: &Arc<RwLock<OptimizationStats>>,
        urgent: bool,
    ) {
        let start_time = Instant::now();
        let pressure_level = {
            let state_guard = state.read().expect("lock should not be poisoned");
            state_guard.pressure_level.clone()
        };

        // Determine optimization strategy based on pressure level and urgency
        let operations = Self::plan_optimization_operations(&pressure_level, urgent);

        // Execute optimization operations
        for operation in operations {
            match operation.operation_type {
                OptimizationType::PoolCompaction => {
                    if config.enable_adaptive_pools {
                        let mut pools = adaptive_pools.write().expect("lock should not be poisoned");
                        pools.compact_all().await;
                    }
                }
                OptimizationType::CacheEviction => {
                    if config.enable_smart_cache {
                        let mut cache = cache_manager.write().expect("lock should not be poisoned");
                        cache.intelligent_eviction().await;
                    }
                }
                OptimizationType::GarbageCollection => {
                    if config.enable_gc_coordination {
                        let mut gc = gc_coordinator.lock().expect("lock should not be poisoned");
                        gc.coordinate_collection().await;
                    }
                }
                OptimizationType::EmergencyCleanup => {
                    if config.enable_pressure_mitigation {
                        let mut mitigator = pressure_mitigator.lock().expect("lock should not be poisoned");
                        mitigator.emergency_cleanup().await;
                    }
                }
                _ => {
                    // Other optimization types would be implemented here
                }
            }
        }

        // Update statistics
        let optimization_duration = start_time.elapsed();
        let mut stats_guard = stats.write().expect("lock should not be poisoned");
        stats_guard.total_optimizations += 1;
        stats_guard.total_optimization_time += optimization_duration;

        // Update last optimization time
        let mut state_guard = state.write().expect("lock should not be poisoned");
        state_guard.last_optimization = Some(start_time);
    }

    /// Plan optimization operations based on current conditions
    fn plan_optimization_operations(
        pressure_level: &MemoryPressureLevel,
        urgent: bool,
    ) -> Vec<OptimizationOperation> {
        let mut operations = Vec::new();
        let current_time = Instant::now();

        match pressure_level {
            MemoryPressureLevel::Normal => {
                if !urgent {
                    // Light maintenance operations
                    operations.push(OptimizationOperation {
                        operation_type: OptimizationType::PoolCompaction,
                        start_time: current_time,
                        estimated_duration: Duration::from_millis(100),
                        priority: OptimizationPriority::Low,
                    });
                }
            }
            MemoryPressureLevel::Warning => {
                operations.push(OptimizationOperation {
                    operation_type: OptimizationType::CacheEviction,
                    start_time: current_time,
                    estimated_duration: Duration::from_millis(200),
                    priority: OptimizationPriority::Normal,
                });
                operations.push(OptimizationOperation {
                    operation_type: OptimizationType::PoolCompaction,
                    start_time: current_time,
                    estimated_duration: Duration::from_millis(150),
                    priority: OptimizationPriority::Normal,
                });
            }
            MemoryPressureLevel::Critical => {
                operations.push(OptimizationOperation {
                    operation_type: OptimizationType::GarbageCollection,
                    start_time: current_time,
                    estimated_duration: Duration::from_millis(500),
                    priority: OptimizationPriority::High,
                });
                operations.push(OptimizationOperation {
                    operation_type: OptimizationType::CacheEviction,
                    start_time: current_time,
                    estimated_duration: Duration::from_millis(300),
                    priority: OptimizationPriority::High,
                });
            }
            MemoryPressureLevel::Emergency => {
                operations.push(OptimizationOperation {
                    operation_type: OptimizationType::EmergencyCleanup,
                    start_time: current_time,
                    estimated_duration: Duration::from_millis(100),
                    priority: OptimizationPriority::Emergency,
                });
                operations.push(OptimizationOperation {
                    operation_type: OptimizationType::GarbageCollection,
                    start_time: current_time,
                    estimated_duration: Duration::from_millis(1000),
                    priority: OptimizationPriority::Critical,
                });
            }
        }

        // Sort by priority
        operations.sort_by_key(|op| op.priority);
        operations
    }

    /// Analyze a real history of recent memory-usage samples (MB, oldest
    /// first) to determine the current trend via linear-regression slope,
    /// instead of unconditionally returning `Stable`. Returns `Stable` (not
    /// a guess) when there isn't yet enough history for a real judgement.
    fn calculate_memory_trend(usage_history_mb: &[f64]) -> MemoryTrend {
        match Self::fit_trend(usage_history_mb) {
            Some(fit)
                if fit.relative_noise > TREND_SLOPE_THRESHOLD * 4.0
                    && fit.relative_slope.abs() < TREND_SLOPE_THRESHOLD * 2.0 =>
            {
                MemoryTrend::Oscillating
            }
            Some(fit) if fit.relative_slope > TREND_SLOPE_THRESHOLD => MemoryTrend::Increasing,
            Some(fit) if fit.relative_slope < -TREND_SLOPE_THRESHOLD => MemoryTrend::Decreasing,
            _ => MemoryTrend::Stable,
        }
    }

    /// Fit an OLS line through the real usage-history samples. `None` when
    /// there isn't enough history (`< TREND_MIN_SAMPLES`) or the mean usage
    /// is (near) zero to support a meaningful fit.
    fn fit_trend(usage_history_mb: &[f64]) -> Option<UsageTrendFit> {
        if usage_history_mb.len() < TREND_MIN_SAMPLES {
            return None;
        }

        let n = usage_history_mb.len() as f64;
        let mean_x = (n - 1.0) / 2.0;
        let mean_y = usage_history_mb.iter().sum::<f64>() / n;
        if mean_y.abs() < f64::EPSILON {
            return None;
        }

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, &y) in usage_history_mb.iter().enumerate() {
            let dx = i as f64 - mean_x;
            numerator += dx * (y - mean_y);
            denominator += dx * dx;
        }
        if denominator == 0.0 {
            return None;
        }

        let slope_mb_per_sample = numerator / denominator;
        let relative_slope = slope_mb_per_sample / mean_y;

        let residual_variance = usage_history_mb
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let predicted = mean_y + slope_mb_per_sample * (i as f64 - mean_x);
                (y - predicted).powi(2)
            })
            .sum::<f64>()
            / n;
        let relative_noise = residual_variance.sqrt() / mean_y.abs();

        Some(UsageTrendFit {
            slope_mb_per_sample,
            relative_slope,
            relative_noise,
        })
    }

    /// Predict when memory will be exhausted based on the real, fitted
    /// growth rate derived from `usage_history_mb`, rather than a fixed
    /// "1MB per second" placeholder. Returns `None` when the trend isn't
    /// really increasing (nothing to predict) or the fit is unavailable.
    fn predict_memory_exhaustion(
        current_usage: u64,
        max_memory: usize,
        trend: &MemoryTrend,
        usage_history_mb: &[f64],
        sample_interval: Duration,
    ) -> Option<Duration> {
        if !matches!(trend, MemoryTrend::Increasing) {
            return None;
        }

        let fit = Self::fit_trend(usage_history_mb)?;
        if fit.slope_mb_per_sample <= 0.0 || sample_interval.as_secs_f64() <= 0.0 {
            return None;
        }

        let bytes_per_sample = fit.slope_mb_per_sample * 1024.0 * 1024.0;
        let bytes_per_sec = bytes_per_sample / sample_interval.as_secs_f64();
        if bytes_per_sec <= 0.0 {
            return None;
        }

        let remaining = (max_memory as u64).saturating_sub(current_usage);
        Some(Duration::from_secs_f64(remaining as f64 / bytes_per_sec))
    }

    /// Get current optimization statistics
    pub fn get_statistics(&self) -> OptimizationStats {
        self.stats.read().expect("lock should not be poisoned").clone()
    }

    /// Get current optimization state
    pub fn get_state(&self) -> OptimizationState {
        self.optimization_state.read().expect("lock should not be poisoned").clone()
    }

    /// Manually trigger optimization
    pub fn trigger_optimization(&self) {
        self.optimization_trigger.notify_one();
    }

    /// Allocate memory with optimization integration
    pub async fn optimized_allocate(&self, size: usize) -> Result<Vec<u8>, MemoryOptimizationError> {
        // Check current memory pressure
        let pressure_level = {
            let state = self.optimization_state.read().expect("lock should not be poisoned");
            state.pressure_level.clone()
        };

        // If pressure is high, try to free some memory first
        if matches!(pressure_level, MemoryPressureLevel::Critical | MemoryPressureLevel::Emergency) {
            self.trigger_optimization();

            // Wait a bit for optimization to complete
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Try pool allocation first
        if self.config.enable_adaptive_pools {
            let pools = self.adaptive_pools.read().expect("lock should not be poisoned");
            if let Ok(memory) = pools.allocate(size).await {
                return Ok(memory);
            }
        }

        // Fall back to regular allocation
        Ok(vec![0u8; size])
    }
}

/// Memory optimization errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryOptimizationError {
    #[error("Pool allocation failed: {0}")]
    PoolAllocationFailed(String),

    #[error("Cache operation failed: {0}")]
    CacheOperationFailed(String),

    #[error("Memory pressure mitigation failed: {0}")]
    PressureMitigationFailed(String),

    #[error("GC coordination failed: {0}")]
    GCCoordinationFailed(String),

    #[error("Task execution failed: {0}")]
    TaskExecutionFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_optimizer_creation() {
        let config = MemoryOptimizerConfig::default();
        let optimizer = AdvancedMemoryOptimizer::new(config);

        let state = optimizer.get_state();
        assert_eq!(state.pressure_level, MemoryPressureLevel::Normal);
        assert!(state.active_operations.is_empty());
    }

    #[tokio::test]
    async fn test_optimization_planning() {
        let operations = AdvancedMemoryOptimizer::plan_optimization_operations(
            &MemoryPressureLevel::Critical,
            true
        );

        assert!(!operations.is_empty());
        assert!(operations.iter().any(|op| matches!(op.operation_type, OptimizationType::GarbageCollection)));
    }

    /// regression: too little history must not be enough to declare a
    /// trend -- `calculate_memory_trend` must stay honestly `Stable`
    /// rather than guess from noise (was previously *always* Stable
    /// regardless of input, which this also covers, but the real
    /// implementation must specifically choose `Stable` here for the
    /// *right* reason: insufficient samples).
    #[test]
    fn regression_memory_trend_insufficient_history_is_stable() {
        let history = vec![1000.0, 1010.0];
        assert_eq!(
            AdvancedMemoryOptimizer::calculate_memory_trend(&history),
            MemoryTrend::Stable
        );
    }

    /// regression: a clearly, consistently climbing series of real usage
    /// samples must be detected as `Increasing` -- proving the trend
    /// detector actually reads its input instead of being hardcoded.
    #[test]
    fn regression_memory_trend_detects_increasing_usage() {
        let history: Vec<f64> = (0..10).map(|i| 1000.0 + i as f64 * 50.0).collect();
        assert_eq!(
            AdvancedMemoryOptimizer::calculate_memory_trend(&history),
            MemoryTrend::Increasing
        );
    }

    /// regression: a clearly, consistently shrinking series must be
    /// detected as `Decreasing`.
    #[test]
    fn regression_memory_trend_detects_decreasing_usage() {
        let history: Vec<f64> = (0..10).map(|i| 2000.0 - i as f64 * 50.0).collect();
        assert_eq!(
            AdvancedMemoryOptimizer::calculate_memory_trend(&history),
            MemoryTrend::Decreasing
        );
    }

    /// regression: a flat series (no real drift) must stay `Stable`.
    #[test]
    fn regression_memory_trend_flat_usage_is_stable() {
        let history = vec![1000.0; 10];
        assert_eq!(
            AdvancedMemoryOptimizer::calculate_memory_trend(&history),
            MemoryTrend::Stable
        );
    }

    /// regression: predict_memory_exhaustion must now actually fire on a
    /// genuinely `Increasing` trend (previously dead code, since the trend
    /// was hardcoded `Stable` and this arm was therefore unreachable), and
    /// the predicted duration must be derived from the real fitted growth
    /// rate rather than a fixed "1MB/s" placeholder.
    #[test]
    fn regression_predict_memory_exhaustion_fires_on_real_increasing_trend() {
        // Usage climbing by ~50MB per sample, sampled every 1 second.
        let history: Vec<f64> = (0..10).map(|i| 1000.0 + i as f64 * 50.0).collect();
        let trend = AdvancedMemoryOptimizer::calculate_memory_trend(&history);
        assert_eq!(trend, MemoryTrend::Increasing);

        let current_usage_bytes = 1450 * 1024 * 1024; // matches the last sample
        let max_memory_bytes = 2000 * 1024 * 1024; // 550MB of headroom left
        let prediction = AdvancedMemoryOptimizer::predict_memory_exhaustion(
            current_usage_bytes,
            max_memory_bytes,
            &trend,
            &history,
            Duration::from_secs(1),
        );

        let predicted = prediction.expect("an increasing trend must produce a real prediction");
        // ~50MB/s growth against ~550MB headroom -> roughly 11s, not the
        // old placeholder's ~550s (which assumed a fixed 1MB/s).
        assert!(
            predicted.as_secs_f64() < 60.0,
            "expected a prediction in the tens of seconds given the real ~50MB/s growth rate, got {predicted:?}"
        );
    }

    /// regression: a `Stable`/`Decreasing` trend must never produce an
    /// exhaustion prediction (nothing to predict).
    #[test]
    fn regression_predict_memory_exhaustion_none_when_not_increasing() {
        let flat_history = vec![1000.0; 10];
        let trend = AdvancedMemoryOptimizer::calculate_memory_trend(&flat_history);
        assert_eq!(trend, MemoryTrend::Stable);

        let prediction = AdvancedMemoryOptimizer::predict_memory_exhaustion(
            1000 * 1024 * 1024,
            2000 * 1024 * 1024,
            &trend,
            &flat_history,
            Duration::from_secs(1),
        );
        assert_eq!(prediction, None);
    }

    #[test]
    fn test_pressure_thresholds() {
        let thresholds = PressureThresholds {
            warning: 0.7,
            critical: 0.85,
            emergency: 0.95,
            recovery: 0.6,
        };

        assert!(thresholds.warning < thresholds.critical);
        assert!(thresholds.critical < thresholds.emergency);
        assert!(thresholds.recovery < thresholds.warning);
    }
}