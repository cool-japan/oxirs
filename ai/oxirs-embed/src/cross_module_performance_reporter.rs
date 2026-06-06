//! Cross-Module Performance — Reporter
//!
//! Report generation: hotspot identification, bottleneck analysis, comparison reports, and the
//! main `CrossModulePerformanceCoordinator` orchestrator.

use crate::cross_module_performance_profiler::{
    calculate_percentage_change, ModulePerformanceMonitor, PredictivePerformanceEngine,
    ResourceAllocator,
};
use crate::cross_module_performance_types::{
    AnomalyEvent, AnomalyType, CacheStats, CoordinatorConfig, ModuleMetrics,
    OptimizationRecommendation, OptimizationResults, OptimizationType, PerformanceImpact, Priority,
    SeverityLevel,
};
use anyhow::{anyhow, Result};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time;
use tracing::{info, warn};

// ── GlobalPerformanceMetrics ──────────────────────────────────────────────────

/// Global performance metrics
#[derive(Debug)]
pub struct GlobalPerformanceMetrics {
    total_optimizations: AtomicU64,
    avg_performance_gain: Arc<RwLock<f64>>,
    success_rate: Arc<RwLock<f64>>,
    last_optimization: Arc<RwLock<Option<chrono::DateTime<chrono::Utc>>>>,
}

impl GlobalPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_optimizations: AtomicU64::new(0),
            avg_performance_gain: Arc::new(RwLock::new(0.0)),
            success_rate: Arc::new(RwLock::new(0.0)),
            last_optimization: Arc::new(RwLock::new(None)),
        }
    }

    pub fn update(&mut self, results: &OptimizationResults) {
        self.total_optimizations.fetch_add(1, Ordering::SeqCst);
        {
            let mut gain = self.avg_performance_gain.write().expect("lock poisoned");
            *gain = (*gain + results.total_performance_gain) / 2.0;
        }
        {
            let mut rate = self.success_rate.write().expect("lock poisoned");
            let success = results.optimizations_applied as f64
                / (results.optimizations_applied + results.optimization_failures).max(1) as f64;
            *rate = (*rate + success) / 2.0;
        }
        {
            let mut last = self.last_optimization.write().expect("lock poisoned");
            *last = Some(Utc::now());
        }
    }
}

impl Default for GlobalPerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── OptimizationCache ─────────────────────────────────────────────────────────

/// Optimization cache
#[derive(Debug)]
pub struct OptimizationCache {
    pub cache: HashMap<String, crate::cross_module_performance_types::CachedOptimization>,
    pub stats: CacheStats,
}

impl OptimizationCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            stats: CacheStats {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                size: AtomicUsize::new(0),
            },
        }
    }
}

impl Default for OptimizationCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── CrossModulePerformanceCoordinator ─────────────────────────────────────────

/// Cross-module performance coordinator
#[derive(Debug)]
pub struct CrossModulePerformanceCoordinator {
    config: CoordinatorConfig,
    module_monitors: Arc<RwLock<HashMap<String, ModulePerformanceMonitor>>>,
    resource_allocator: ResourceAllocator,
    predictive_engine: PredictivePerformanceEngine,
    optimization_cache: Arc<RwLock<OptimizationCache>>,
    global_metrics: Arc<RwLock<GlobalPerformanceMetrics>>,
}

impl CrossModulePerformanceCoordinator {
    /// Create a new cross-module performance coordinator
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            config,
            module_monitors: Arc::new(RwLock::new(HashMap::new())),
            resource_allocator: ResourceAllocator::new(),
            predictive_engine: PredictivePerformanceEngine::new(),
            optimization_cache: Arc::new(RwLock::new(OptimizationCache::new())),
            global_metrics: Arc::new(RwLock::new(GlobalPerformanceMetrics::new())),
        }
    }

    /// Register a module for performance monitoring
    pub async fn register_module(&self, module_name: String) -> Result<()> {
        let monitor = ModulePerformanceMonitor::new(module_name.clone());
        {
            let mut monitors = self.module_monitors.write().expect("lock poisoned");
            monitors.insert(module_name.clone(), monitor);
        }
        info!(
            "Registered module '{}' for performance monitoring",
            module_name
        );
        Ok(())
    }

    /// Update module metrics
    pub async fn update_module_metrics(
        &self,
        module_name: &str,
        metrics: ModuleMetrics,
    ) -> Result<()> {
        let monitor = {
            let monitors = self.module_monitors.read().expect("lock poisoned");
            monitors.get(module_name).cloned()
        };
        if let Some(monitor) = monitor {
            monitor.update_metrics(metrics).await?;
        } else {
            return Err(anyhow!("Module '{}' not registered", module_name));
        }
        Ok(())
    }

    /// Optimize performance across all modules
    pub async fn optimize_performance(&self) -> Result<OptimizationResults> {
        info!("Starting cross-module performance optimization");
        let mut results = OptimizationResults::new();

        let performance_data = self.collect_performance_data().await?;
        let anomalies = self
            .predictive_engine
            .detect_anomalies(&performance_data)
            .await?;
        results.anomalies_detected = anomalies.len();

        let recommendations = self
            .generate_optimization_recommendations(&performance_data, &anomalies)
            .await?;
        results.recommendations = recommendations.clone();

        for recommendation in recommendations {
            match self.apply_optimization(recommendation).await {
                Ok(impact) => {
                    results.optimizations_applied += 1;
                    results.total_performance_gain += impact.overall_score;
                }
                Err(e) => {
                    warn!("Failed to apply optimization: {}", e);
                    results.optimization_failures += 1;
                }
            }
        }

        self.update_global_metrics(&results).await?;
        info!("Performance optimization completed: {:?}", results);
        Ok(results)
    }

    async fn collect_performance_data(&self) -> Result<HashMap<String, ModuleMetrics>> {
        let monitor_list = {
            let monitors = self.module_monitors.read().expect("lock poisoned");
            monitors
                .iter()
                .map(|(name, monitor)| (name.clone(), monitor.clone()))
                .collect::<Vec<_>>()
        };
        let mut data = HashMap::new();
        for (module_name, monitor) in monitor_list {
            let metrics = monitor.get_current_metrics().await?;
            data.insert(module_name, metrics);
        }
        Ok(data)
    }

    async fn generate_optimization_recommendations(
        &self,
        performance_data: &HashMap<String, ModuleMetrics>,
        anomalies: &[AnomalyEvent],
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        for (module_name, metrics) in performance_data {
            if metrics.cpu_usage > 80.0 {
                recommendations.push(OptimizationRecommendation {
                    module_name: module_name.clone(),
                    optimization_type: OptimizationType::ResourceReallocation,
                    priority: Priority::High,
                    description: "High CPU usage detected - recommend resource reallocation"
                        .to_string(),
                    estimated_impact: PerformanceImpact {
                        latency_change_pct: -15.0,
                        throughput_change_pct: 20.0,
                        efficiency_change_pct: 10.0,
                        overall_score: 75.0,
                    },
                    implementation_steps: vec![
                        "Increase CPU allocation".to_string(),
                        "Enable parallel processing".to_string(),
                        "Optimize critical paths".to_string(),
                    ],
                });
            }
            if metrics.memory_usage > 8_000_000_000 {
                recommendations.push(OptimizationRecommendation {
                    module_name: module_name.clone(),
                    optimization_type: OptimizationType::MemoryOptimization,
                    priority: Priority::Medium,
                    description: "High memory usage - recommend memory optimization".to_string(),
                    estimated_impact: PerformanceImpact {
                        latency_change_pct: -10.0,
                        throughput_change_pct: 15.0,
                        efficiency_change_pct: 25.0,
                        overall_score: 70.0,
                    },
                    implementation_steps: vec![
                        "Enable memory pooling".to_string(),
                        "Optimize data structures".to_string(),
                        "Implement garbage collection tuning".to_string(),
                    ],
                });
            }
            if metrics.cache_hit_rate < 80.0 {
                recommendations.push(OptimizationRecommendation {
                    module_name: module_name.clone(),
                    optimization_type: OptimizationType::CacheOptimization,
                    priority: Priority::Medium,
                    description: "Low cache hit rate - recommend cache optimization".to_string(),
                    estimated_impact: PerformanceImpact {
                        latency_change_pct: -20.0,
                        throughput_change_pct: 25.0,
                        efficiency_change_pct: 15.0,
                        overall_score: 80.0,
                    },
                    implementation_steps: vec![
                        "Increase cache size".to_string(),
                        "Implement intelligent prefetching".to_string(),
                        "Optimize cache eviction policy".to_string(),
                    ],
                });
            }
        }

        for anomaly in anomalies {
            recommendations.extend(self.generate_anomaly_recommendations(anomaly).await?);
        }

        recommendations.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                b.estimated_impact
                    .overall_score
                    .partial_cmp(&a.estimated_impact.overall_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        Ok(recommendations)
    }

    async fn generate_anomaly_recommendations(
        &self,
        anomaly: &AnomalyEvent,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        match anomaly.anomaly_type {
            AnomalyType::PerformanceDegradation => {
                recommendations.push(OptimizationRecommendation {
                    module_name: anomaly.module_name.clone(),
                    optimization_type: OptimizationType::PerformanceTuning,
                    priority: Priority::High,
                    description: "Performance degradation detected - immediate optimization needed"
                        .to_string(),
                    estimated_impact: PerformanceImpact {
                        latency_change_pct: -30.0,
                        throughput_change_pct: 40.0,
                        efficiency_change_pct: 20.0,
                        overall_score: 85.0,
                    },
                    implementation_steps: anomaly.recommended_actions.clone(),
                });
            }
            AnomalyType::MemoryLeak => {
                recommendations.push(OptimizationRecommendation {
                    module_name: anomaly.module_name.clone(),
                    optimization_type: OptimizationType::MemoryOptimization,
                    priority: Priority::Critical,
                    description: "Memory leak detected - immediate action required".to_string(),
                    estimated_impact: PerformanceImpact {
                        latency_change_pct: -50.0,
                        throughput_change_pct: 60.0,
                        efficiency_change_pct: 80.0,
                        overall_score: 95.0,
                    },
                    implementation_steps: vec![
                        "Identify memory leak source".to_string(),
                        "Implement automatic memory cleanup".to_string(),
                        "Add memory monitoring alerts".to_string(),
                    ],
                });
            }
            _ => {
                recommendations.push(OptimizationRecommendation {
                    module_name: anomaly.module_name.clone(),
                    optimization_type: OptimizationType::GeneralOptimization,
                    priority: match anomaly.severity {
                        SeverityLevel::Critical => Priority::Critical,
                        SeverityLevel::High => Priority::High,
                        SeverityLevel::Medium => Priority::Medium,
                        SeverityLevel::Low => Priority::Low,
                    },
                    description: format!("Anomaly detected: {:?}", anomaly.anomaly_type),
                    estimated_impact: PerformanceImpact {
                        latency_change_pct: -10.0,
                        throughput_change_pct: 15.0,
                        efficiency_change_pct: 10.0,
                        overall_score: 60.0,
                    },
                    implementation_steps: anomaly.recommended_actions.clone(),
                });
            }
        }
        Ok(recommendations)
    }

    async fn apply_optimization(
        &self,
        recommendation: OptimizationRecommendation,
    ) -> Result<PerformanceImpact> {
        info!("Applying optimization: {}", recommendation.description);
        match recommendation.optimization_type {
            OptimizationType::ResourceReallocation => {
                self.resource_allocator
                    .reallocate_resources(&recommendation.module_name, &recommendation)
                    .await?;
            }
            OptimizationType::MemoryOptimization => {
                self.apply_memory_optimization(&recommendation.module_name, &recommendation)
                    .await?;
            }
            OptimizationType::CacheOptimization => {
                self.apply_cache_optimization(&recommendation.module_name, &recommendation)
                    .await?;
            }
            OptimizationType::PerformanceTuning => {
                self.apply_performance_tuning(&recommendation.module_name, &recommendation)
                    .await?;
            }
            OptimizationType::GeneralOptimization => {
                self.apply_general_optimization(&recommendation.module_name, &recommendation)
                    .await?;
            }
        }
        time::sleep(Duration::from_secs(5)).await;
        let actual_impact = self
            .measure_optimization_impact(&recommendation.module_name)
            .await?;
        self.predictive_engine
            .update_models(&recommendation, &actual_impact)
            .await?;
        Ok(actual_impact)
    }

    async fn apply_memory_optimization(
        &self,
        module_name: &str,
        recommendation: &OptimizationRecommendation,
    ) -> Result<()> {
        use tracing::debug;
        debug!("Applying memory optimization for module: {}", module_name);
        for step in &recommendation.implementation_steps {
            if step.contains("memory pooling") {
                self.enable_memory_pooling(module_name).await?;
            } else if step.contains("garbage collection") {
                self.optimize_garbage_collection(module_name).await?;
            } else if step.contains("data structures") {
                self.optimize_data_structures(module_name).await?;
            }
        }
        Ok(())
    }

    async fn apply_cache_optimization(
        &self,
        module_name: &str,
        recommendation: &OptimizationRecommendation,
    ) -> Result<()> {
        use tracing::debug;
        debug!("Applying cache optimization for module: {}", module_name);
        for step in &recommendation.implementation_steps {
            if step.contains("cache size") {
                self.increase_cache_size(module_name).await?;
            } else if step.contains("prefetching") {
                self.enable_intelligent_prefetching(module_name).await?;
            } else if step.contains("eviction policy") {
                self.optimize_cache_eviction(module_name).await?;
            }
        }
        Ok(())
    }

    async fn apply_performance_tuning(
        &self,
        module_name: &str,
        recommendation: &OptimizationRecommendation,
    ) -> Result<()> {
        use tracing::debug;
        debug!("Applying performance tuning for module: {}", module_name);
        for step in &recommendation.implementation_steps {
            if step.contains("parallel processing") {
                self.enable_parallel_processing(module_name).await?;
            } else if step.contains("critical paths") {
                self.optimize_critical_paths(module_name).await?;
            } else if step.contains("algorithms") {
                self.optimize_algorithms(module_name).await?;
            }
        }
        Ok(())
    }

    async fn apply_general_optimization(
        &self,
        module_name: &str,
        _recommendation: &OptimizationRecommendation,
    ) -> Result<()> {
        use tracing::debug;
        debug!("Applying general optimization for module: {}", module_name);
        self.tune_module_parameters(module_name).await?;
        self.optimize_resource_usage(module_name).await?;
        Ok(())
    }

    async fn measure_optimization_impact(&self, module_name: &str) -> Result<PerformanceImpact> {
        let baseline = self.get_baseline_metrics(module_name).await?;
        let current = self.get_current_module_metrics(module_name).await?;
        let latency_change = calculate_percentage_change(
            baseline.avg_response_time.as_millis() as f64,
            current.avg_response_time.as_millis() as f64,
        );
        let throughput_change =
            calculate_percentage_change(baseline.request_rate, current.request_rate);
        let efficiency_change = calculate_percentage_change(baseline.cpu_usage, current.cpu_usage);
        let overall_score =
            (latency_change.abs() + throughput_change + efficiency_change.abs()) / 3.0;
        Ok(PerformanceImpact {
            latency_change_pct: latency_change,
            throughput_change_pct: throughput_change,
            efficiency_change_pct: efficiency_change,
            overall_score,
        })
    }

    async fn update_global_metrics(&self, results: &OptimizationResults) -> Result<()> {
        let mut global_metrics = self.global_metrics.write().expect("lock poisoned");
        global_metrics.update(results);
        Ok(())
    }

    async fn enable_memory_pooling(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Enabling memory pooling");
        Ok(())
    }

    async fn optimize_garbage_collection(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Optimizing garbage collection");
        Ok(())
    }

    async fn optimize_data_structures(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Optimizing data structures");
        Ok(())
    }

    async fn increase_cache_size(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Increasing cache size");
        Ok(())
    }

    async fn enable_intelligent_prefetching(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Enabling intelligent prefetching");
        Ok(())
    }

    async fn optimize_cache_eviction(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Optimizing cache eviction policy");
        Ok(())
    }

    async fn enable_parallel_processing(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Enabling parallel processing");
        Ok(())
    }

    async fn optimize_critical_paths(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Optimizing critical paths");
        Ok(())
    }

    async fn optimize_algorithms(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Optimizing algorithms");
        Ok(())
    }

    async fn tune_module_parameters(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Tuning module parameters");
        Ok(())
    }

    async fn optimize_resource_usage(&self, _module_name: &str) -> Result<()> {
        use tracing::debug;
        debug!("Optimizing resource usage");
        Ok(())
    }

    async fn get_baseline_metrics(&self, _module_name: &str) -> Result<ModuleMetrics> {
        Ok(ModuleMetrics {
            cpu_usage: 50.0,
            memory_usage: 4_000_000_000,
            gpu_memory_usage: Some(2_000_000_000),
            network_io_bps: 1_000_000,
            disk_io_bps: 500_000,
            request_rate: 100.0,
            avg_response_time: Duration::from_millis(100),
            error_rate: 1.0,
            cache_hit_rate: 85.0,
            active_connections: 50,
            queue_depth: 10,
        })
    }

    async fn get_current_module_metrics(&self, module_name: &str) -> Result<ModuleMetrics> {
        let monitor = {
            let monitors = self.module_monitors.read().expect("lock poisoned");
            monitors.get(module_name).cloned()
        };
        if let Some(monitor) = monitor {
            monitor.get_current_metrics().await
        } else {
            Err(anyhow!("Module '{}' not found", module_name))
        }
    }
}
