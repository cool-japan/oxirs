//! Cross-Module Performance — Profiler
//!
//! Profiling logic: module timing, memory tracking, cross-module call tracing.

use crate::cross_module_performance_types::{
    AllocationEvent, AllocationStrategy, AllocationType, AnomalyAlgorithm, AnomalyEvent,
    AnomalyType, ModuleMetrics, OptimizationRecommendation, OptimizationType, PerformanceBaseline,
    PerformanceImpact, PerformanceModel, PerformanceSnapshot, Priority, ResourceAllocation,
    SeverityLevel, TrainingSample,
};
use anyhow::Result;
use chrono::Utc;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::debug;

// ── ResourceTracker ───────────────────────────────────────────────────────────

/// Resource tracker for monitoring resource usage
#[derive(Debug, Clone)]
pub struct ResourceTracker {
    cpu_history: Arc<RwLock<VecDeque<f64>>>,
    memory_history: Arc<RwLock<VecDeque<u64>>>,
    last_update: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
}

impl ResourceTracker {
    pub fn new() -> Self {
        Self {
            cpu_history: Arc::new(RwLock::new(VecDeque::new())),
            memory_history: Arc::new(RwLock::new(VecDeque::new())),
            last_update: Arc::new(RwLock::new(Utc::now())),
        }
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── PredictionModel ───────────────────────────────────────────────────────────

/// Simple per-module prediction model (separate from the config-level PerformanceModel)
#[derive(Debug, Clone)]
pub struct PredictionModel {
    pub parameters: HashMap<String, f64>,
    pub last_trained: chrono::DateTime<chrono::Utc>,
}

impl PredictionModel {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            last_trained: Utc::now(),
        }
    }
}

impl Default for PredictionModel {
    fn default() -> Self {
        Self::new()
    }
}

// ── ModulePerformanceMonitor ──────────────────────────────────────────────────

/// Module performance monitor
#[derive(Debug, Clone)]
pub struct ModulePerformanceMonitor {
    pub module_name: String,
    pub metrics: Arc<RwLock<ModuleMetrics>>,
    pub resource_tracker: ResourceTracker,
    pub history: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    pub(crate) prediction_model: PredictionModel,
}

impl ModulePerformanceMonitor {
    pub fn new(module_name: String) -> Self {
        Self {
            module_name,
            metrics: Arc::new(RwLock::new(ModuleMetrics {
                cpu_usage: 0.0,
                memory_usage: 0,
                gpu_memory_usage: None,
                network_io_bps: 0,
                disk_io_bps: 0,
                request_rate: 0.0,
                avg_response_time: Duration::from_millis(0),
                error_rate: 0.0,
                cache_hit_rate: 0.0,
                active_connections: 0,
                queue_depth: 0,
            })),
            resource_tracker: ResourceTracker::new(),
            history: Arc::new(RwLock::new(VecDeque::new())),
            prediction_model: PredictionModel::new(),
        }
    }

    pub async fn update_metrics(&self, new_metrics: ModuleMetrics) -> Result<()> {
        {
            let mut metrics = self.metrics.write().expect("lock poisoned");
            *metrics = new_metrics.clone();
        }
        let snapshot = PerformanceSnapshot {
            metrics: new_metrics,
            timestamp: Utc::now(),
        };
        {
            let mut history = self.history.write().expect("lock poisoned");
            history.push_back(snapshot);
            if history.len() > 1000 {
                history.pop_front();
            }
        }
        Ok(())
    }

    pub async fn get_current_metrics(&self) -> Result<ModuleMetrics> {
        let metrics = self.metrics.read().expect("lock poisoned");
        Ok(metrics.clone())
    }
}

// ── ResourceAllocator ─────────────────────────────────────────────────────────

/// Resource allocator for dynamic resource management
#[derive(Debug)]
pub struct ResourceAllocator {
    pub available_cores: AtomicUsize,
    pub available_memory: AtomicU64,
    pub available_gpu_memory: AtomicU64,
    pub allocation_history: Arc<RwLock<VecDeque<AllocationEvent>>>,
    pub current_allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    pub optimization_strategies: Vec<AllocationStrategy>,
}

impl ResourceAllocator {
    pub fn new() -> Self {
        Self {
            available_cores: AtomicUsize::new(8),
            available_memory: AtomicU64::new(16_000_000_000),
            available_gpu_memory: AtomicU64::new(8_000_000_000),
            allocation_history: Arc::new(RwLock::new(VecDeque::new())),
            current_allocations: Arc::new(RwLock::new(HashMap::new())),
            optimization_strategies: Vec::new(),
        }
    }

    pub async fn reallocate_resources(
        &self,
        module_name: &str,
        recommendation: &OptimizationRecommendation,
    ) -> Result<()> {
        debug!("Reallocating resources for module: {}", module_name);
        let current_allocation = self.get_current_allocation(module_name).await?;
        let new_allocation = self
            .calculate_new_allocation(&current_allocation, recommendation)
            .await?;
        self.apply_allocation(module_name, new_allocation).await?;
        Ok(())
    }

    async fn get_current_allocation(&self, module_name: &str) -> Result<ResourceAllocation> {
        let allocations = self.current_allocations.read().expect("lock poisoned");
        if let Some(allocation) = allocations.get(module_name) {
            Ok(allocation.clone())
        } else {
            Ok(ResourceAllocation {
                cpu_cores: 2,
                memory_bytes: 2_000_000_000,
                gpu_memory_bytes: Some(1_000_000_000),
                priority: 50,
                allocated_at: Utc::now(),
                expected_duration: None,
            })
        }
    }

    async fn calculate_new_allocation(
        &self,
        current: &ResourceAllocation,
        recommendation: &OptimizationRecommendation,
    ) -> Result<ResourceAllocation> {
        let mut new_allocation = current.clone();
        match recommendation.optimization_type {
            OptimizationType::ResourceReallocation => match recommendation.priority {
                Priority::Critical => {
                    new_allocation.cpu_cores = (current.cpu_cores * 2).min(8);
                    new_allocation.memory_bytes = (current.memory_bytes * 2).min(8_000_000_000);
                }
                Priority::High => {
                    new_allocation.cpu_cores = (current.cpu_cores + 2).min(6);
                    new_allocation.memory_bytes =
                        (current.memory_bytes + 1_000_000_000).min(6_000_000_000);
                }
                _ => {
                    new_allocation.cpu_cores = (current.cpu_cores + 1).min(4);
                    new_allocation.memory_bytes =
                        (current.memory_bytes + 500_000_000).min(4_000_000_000);
                }
            },
            _ => {
                new_allocation.priority = (current.priority + 10).min(100);
            }
        }
        new_allocation.allocated_at = Utc::now();
        Ok(new_allocation)
    }

    async fn apply_allocation(
        &self,
        module_name: &str,
        allocation: ResourceAllocation,
    ) -> Result<()> {
        {
            let mut allocations = self.current_allocations.write().expect("lock poisoned");
            allocations.insert(module_name.to_string(), allocation.clone());
        }
        let event = AllocationEvent {
            module_name: module_name.to_string(),
            event_type: AllocationType::Rebalance,
            allocation,
            performance_impact: None,
            timestamp: Utc::now(),
        };
        {
            let mut history = self.allocation_history.write().expect("lock poisoned");
            history.push_back(event);
            if history.len() > 1000 {
                history.pop_front();
            }
        }
        Ok(())
    }
}

impl Default for ResourceAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ── LearningEngine ────────────────────────────────────────────────────────────

/// Learning engine for continuous improvement
#[derive(Debug)]
pub struct LearningEngine {
    learning_rate: f64,
    training_samples: Arc<RwLock<VecDeque<TrainingSample>>>,
    update_frequency: Duration,
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
}

impl LearningEngine {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            training_samples: Arc::new(RwLock::new(VecDeque::new())),
            update_frequency: Duration::from_secs(3600),
            baselines: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn update_model(
        &self,
        recommendation: &OptimizationRecommendation,
        actual_impact: &PerformanceImpact,
    ) -> Result<()> {
        let sample = TrainingSample {
            features: vec![
                recommendation.estimated_impact.overall_score,
                recommendation.priority.clone() as u8 as f64,
            ],
            target: actual_impact.overall_score,
            context: HashMap::new(),
            weight: 1.0,
            timestamp: Utc::now(),
        };
        {
            let mut samples = self.training_samples.write().expect("lock poisoned");
            samples.push_back(sample);
            if samples.len() > 10000 {
                samples.pop_front();
            }
        }
        Ok(())
    }
}

impl Default for LearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── AnomalyDetector ───────────────────────────────────────────────────────────

/// Anomaly detector for performance issues
#[derive(Debug)]
pub struct AnomalyDetector {
    algorithms: Vec<AnomalyAlgorithm>,
    thresholds: HashMap<String, f64>,
    anomaly_history: Arc<RwLock<VecDeque<AnomalyEvent>>>,
    false_positive_rate: f64,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            algorithms: vec![
                AnomalyAlgorithm::StatisticalOutlier { z_threshold: 3.0 },
                AnomalyAlgorithm::IsolationForest { contamination: 0.1 },
            ],
            thresholds: HashMap::new(),
            anomaly_history: Arc::new(RwLock::new(VecDeque::new())),
            false_positive_rate: 0.05,
        }
    }

    pub async fn detect(
        &self,
        performance_data: &HashMap<String, ModuleMetrics>,
    ) -> Result<Vec<AnomalyEvent>> {
        let mut anomalies = Vec::new();
        for (module_name, metrics) in performance_data {
            if metrics.cpu_usage > 90.0
                || metrics.error_rate > 5.0
                || metrics.avg_response_time > Duration::from_millis(1000)
            {
                let anomaly = AnomalyEvent {
                    module_name: module_name.clone(),
                    anomaly_type: AnomalyType::PerformanceDegradation,
                    severity: if metrics.cpu_usage > 95.0 || metrics.error_rate > 10.0 {
                        SeverityLevel::Critical
                    } else {
                        SeverityLevel::High
                    },
                    score: calculate_anomaly_score(metrics),
                    affected_metrics: vec![
                        "cpu_usage".to_string(),
                        "error_rate".to_string(),
                        "response_time".to_string(),
                    ],
                    recommended_actions: vec![
                        "Increase resource allocation".to_string(),
                        "Investigate error sources".to_string(),
                        "Optimize critical paths".to_string(),
                    ],
                    detected_at: Utc::now(),
                    resolved_at: None,
                };
                anomalies.push(anomaly);
            }
            if metrics.memory_usage > 12_000_000_000 {
                let anomaly = AnomalyEvent {
                    module_name: module_name.clone(),
                    anomaly_type: AnomalyType::MemoryLeak,
                    severity: SeverityLevel::High,
                    score: (metrics.memory_usage as f64 / 16_000_000_000.0) * 100.0,
                    affected_metrics: vec!["memory_usage".to_string()],
                    recommended_actions: vec![
                        "Investigate memory usage patterns".to_string(),
                        "Enable memory profiling".to_string(),
                        "Implement memory cleanup".to_string(),
                    ],
                    detected_at: Utc::now(),
                    resolved_at: None,
                };
                anomalies.push(anomaly);
            }
        }
        {
            let mut history = self.anomaly_history.write().expect("lock poisoned");
            for anomaly in &anomalies {
                history.push_back(anomaly.clone());
            }
            while history.len() > 1000 {
                history.pop_front();
            }
        }
        Ok(anomalies)
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── PredictivePerformanceEngine ───────────────────────────────────────────────

/// Predictive performance engine
#[derive(Debug)]
pub struct PredictivePerformanceEngine {
    pub models: Arc<RwLock<HashMap<String, PerformanceModel>>>,
    pub prediction_cache:
        Arc<RwLock<HashMap<String, crate::cross_module_performance_types::CachedPrediction>>>,
    pub learning_engine: LearningEngine,
    pub anomaly_detector: AnomalyDetector,
}

impl PredictivePerformanceEngine {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            learning_engine: LearningEngine::new(),
            anomaly_detector: AnomalyDetector::new(),
        }
    }

    pub async fn detect_anomalies(
        &self,
        performance_data: &HashMap<String, ModuleMetrics>,
    ) -> Result<Vec<AnomalyEvent>> {
        self.anomaly_detector.detect(performance_data).await
    }

    pub async fn update_models(
        &self,
        recommendation: &OptimizationRecommendation,
        actual_impact: &PerformanceImpact,
    ) -> Result<()> {
        self.learning_engine
            .update_model(recommendation, actual_impact)
            .await
    }
}

impl Default for PredictivePerformanceEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Calculate anomaly score from module metrics
pub fn calculate_anomaly_score(metrics: &ModuleMetrics) -> f64 {
    let cpu_score = if metrics.cpu_usage > 80.0 {
        metrics.cpu_usage
    } else {
        0.0
    };
    let error_score = metrics.error_rate * 10.0;
    let latency_score = if metrics.avg_response_time > Duration::from_millis(500) {
        metrics.avg_response_time.as_millis() as f64 / 10.0
    } else {
        0.0
    };
    (cpu_score + error_score + latency_score) / 3.0
}

/// Calculate percentage change between two values
pub fn calculate_percentage_change(old_value: f64, new_value: f64) -> f64 {
    if old_value == 0.0 {
        return 0.0;
    }
    ((new_value - old_value) / old_value) * 100.0
}
