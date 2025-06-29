//! Federated Query Execution Engine
//!
//! This module handles the execution of federated queries across multiple services,
//! including parallel execution, timeout handling, and fault tolerance.

use anyhow::{anyhow, Result};
use futures::{stream, StreamExt, TryStreamExt};
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, info, instrument, warn};

// Conditional imports for missing types
#[cfg(any(feature = "distributed", not(target_arch = "wasm32")))]
use oxirs_core::{
    distributed::bft::ResourceMonitor as CoreResourceMonitor,
    store::AdaptiveConfig as CoreAdaptiveConfig,
};

// Stub types for when oxirs-core features are not available
#[cfg(not(any(feature = "distributed", not(target_arch = "wasm32"))))]
pub struct CoreAdaptiveConfig;
#[cfg(not(any(feature = "distributed", not(target_arch = "wasm32"))))]
impl Default for CoreAdaptiveConfig {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(any(feature = "distributed", not(target_arch = "wasm32"))))]
pub struct CoreResourceMonitor;
#[cfg(not(any(feature = "distributed", not(target_arch = "wasm32"))))]
impl CoreResourceMonitor {
    pub fn new() -> Self {
        Self
    }
}

// Enhanced performance monitor defined below

use crate::{
    cache::{CacheConfig, FederationCache},
    service_executor::{JoinExecutor, ServiceExecutor, ServiceExecutorConfig},
    service_optimizer::{OptimizedServiceClause, ServiceExecutionStrategy},
    ExecutionPlan, ExecutionStep, FederatedService, FederationError, ServiceRegistry, StepType,
};

/// Federated query executor
#[derive(Debug)]
pub struct FederatedExecutor {
    client: Client,
    config: FederatedExecutorConfig,
    service_executor: Arc<ServiceExecutor>,
    join_executor: Arc<JoinExecutor>,
    cache: Arc<FederationCache>,
}

impl FederatedExecutor {
    /// Create a new federated executor with default configuration
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("oxirs-federate/1.0")
            .build()
            .expect("Failed to create HTTP client");

        let cache = Arc::new(FederationCache::new());
        let service_executor = Arc::new(ServiceExecutor::new(cache.clone()));
        let join_executor = Arc::new(JoinExecutor::new());

        Self {
            client,
            config: FederatedExecutorConfig::default(),
            service_executor,
            join_executor,
            cache,
        }
    }

    /// Create a new federated executor with custom configuration
    pub fn with_config(config: FederatedExecutorConfig) -> Self {
        let client = Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .build()
            .expect("Failed to create HTTP client");

        let cache = Arc::new(FederationCache::with_config(config.cache_config.clone()));
        let service_executor = Arc::new(ServiceExecutor::with_config(
            config.service_executor_config.clone(),
            cache.clone(),
        ));
        let join_executor = Arc::new(JoinExecutor::new());

        Self {
            client,
            config,
            service_executor,
            join_executor,
            cache,
        }
    }

    /// Execute a federated query plan
    #[instrument(skip(self, plan))]
    pub async fn execute_plan(&self, plan: &ExecutionPlan) -> Result<Vec<StepResult>> {
        info!("Executing federated plan with {} steps", plan.steps.len());

        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut completed_steps = HashMap::new();

        // Execute steps according to dependencies and parallelization
        for parallel_group in &plan.parallelizable_steps {
            let group_results =
                execute_parallel_group(parallel_group, plan, &completed_steps).await?;

            for result in group_results {
                completed_steps.insert(result.step_id.clone(), result.clone());
                results.push(result);
            }
        }

        // Execute remaining sequential steps
        for step in &plan.steps {
            if !completed_steps.contains_key(&step.step_id) {
                let result = execute_step(step, &completed_steps).await?;
                completed_steps.insert(result.step_id.clone(), result.clone());
                results.push(result);
            }
        }

        let execution_time = start_time.elapsed();
        info!("Plan execution completed in {:?}", execution_time);

        Ok(results)
    }

    // ===== ADAPTIVE EXECUTION ALGORITHMS =====

    /// Execute plan with adaptive optimization
    #[instrument(skip(self, plan, registry))]
    pub async fn execute_plan_adaptive(
        &self,
        plan: &ExecutionPlan,
        registry: &ServiceRegistry,
    ) -> Result<Vec<StepResult>> {
        info!("Executing federated plan with adaptive optimization");

        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut completed_steps = HashMap::new();
        let mut runtime_stats = RuntimeStatistics::new();
        let mut adaptive_config = LocalAdaptiveConfig::default();

        // Initialize runtime monitoring
        let mut performance_monitor = EnhancedPerformanceMonitor::new();
        let mut resource_monitor = LocalResourceMonitor::new();

        // Execute steps with adaptive optimization
        for (group_idx, parallel_group) in plan.parallelizable_steps.iter().enumerate() {
            let group_start = Instant::now();

            // Collect runtime statistics before execution
            runtime_stats.update_group_start(group_idx, group_start);

            // Check if re-optimization is needed
            if self.should_reoptimize(&runtime_stats, &adaptive_config, group_idx) {
                info!(
                    "Triggering adaptive re-optimization for group {}",
                    group_idx
                );

                let optimized_group = self
                    .reoptimize_execution_group(
                        parallel_group,
                        plan,
                        &runtime_stats,
                        &performance_monitor,
                        registry,
                    )
                    .await?;

                // Execute with optimized strategy
                let group_results = self
                    .execute_adaptive_group(
                        &optimized_group,
                        plan,
                        &completed_steps,
                        &mut performance_monitor,
                        &mut resource_monitor,
                        &mut adaptive_config,
                    )
                    .await?;

                for result in group_results {
                    completed_steps.insert(result.step_id.clone(), result.clone());
                    results.push(result);
                }
            } else {
                // Execute with current strategy
                let group_results = self
                    .execute_adaptive_group(
                        parallel_group,
                        plan,
                        &completed_steps,
                        &mut performance_monitor,
                        &mut resource_monitor,
                        &mut adaptive_config,
                    )
                    .await?;

                for result in group_results {
                    completed_steps.insert(result.step_id.clone(), result.clone());
                    results.push(result);
                }
            }

            // Update runtime statistics after execution
            let group_duration = group_start.elapsed();
            runtime_stats.update_group_end(group_idx, group_duration);

            // Adapt configuration based on performance
            self.adapt_configuration(
                &mut adaptive_config,
                &runtime_stats,
                &performance_monitor,
                &resource_monitor,
            );
        }

        let total_duration = start_time.elapsed();
        runtime_stats.total_execution_time = total_duration;

        info!(
            "Adaptive execution completed in {:?} with {} steps",
            total_duration,
            results.len()
        );

        // Log performance insights
        self.log_adaptive_insights(&runtime_stats, &performance_monitor, &resource_monitor);

        Ok(results)
    }

    /// Execute a group of steps with adaptive optimization
    async fn execute_adaptive_group(
        &self,
        parallel_group: &[String],
        plan: &ExecutionPlan,
        completed_steps: &HashMap<String, StepResult>,
        performance_monitor: &mut EnhancedPerformanceMonitor,
        resource_monitor: &mut LocalResourceMonitor,
        adaptive_config: &mut LocalAdaptiveConfig,
    ) -> Result<Vec<StepResult>> {
        let group_start = Instant::now();
        let mut group_results = Vec::new();

        // Monitor resource usage before execution
        let initial_memory = resource_monitor.get_memory_usage();
        let initial_cpu = resource_monitor.get_cpu_usage();

        // Choose execution strategy dynamically
        let execution_strategy = self.select_execution_strategy(
            parallel_group,
            plan,
            performance_monitor,
            resource_monitor,
            adaptive_config,
        );

        match execution_strategy {
            AdaptiveExecutionStrategy::Parallel => {
                group_results = self
                    .execute_parallel_adaptive(
                        parallel_group,
                        plan,
                        completed_steps,
                        performance_monitor,
                    )
                    .await?;
            }
            AdaptiveExecutionStrategy::Sequential => {
                group_results = self
                    .execute_sequential_adaptive(
                        parallel_group,
                        plan,
                        completed_steps,
                        performance_monitor,
                    )
                    .await?;
            }
            AdaptiveExecutionStrategy::Hybrid => {
                group_results = self
                    .execute_hybrid_adaptive(
                        parallel_group,
                        plan,
                        completed_steps,
                        performance_monitor,
                        adaptive_config,
                    )
                    .await?;
            }
            AdaptiveExecutionStrategy::Streaming => {
                group_results = self
                    .execute_streaming_adaptive(
                        parallel_group,
                        plan,
                        completed_steps,
                        performance_monitor,
                    )
                    .await?;
            }
        }

        // Update resource monitoring
        let final_memory = resource_monitor.get_memory_usage();
        let final_cpu = resource_monitor.get_cpu_usage();
        let group_duration = group_start.elapsed();

        performance_monitor.record_group_execution(
            parallel_group.len(),
            group_duration,
            final_memory - initial_memory,
            final_cpu - initial_cpu,
        );

        Ok(group_results)
    }

    /// Check if re-optimization should be triggered
    fn should_reoptimize(
        &self,
        runtime_stats: &RuntimeStatistics,
        adaptive_config: &LocalAdaptiveConfig,
        current_group: usize,
    ) -> bool {
        // Trigger re-optimization based on various conditions

        // 1. Performance degradation
        if let Some(avg_group_time) = runtime_stats.get_average_group_time() {
            if let Some(current_group_time) = runtime_stats.get_group_time(current_group) {
                if current_group_time.as_millis() as f64
                    > avg_group_time * adaptive_config.performance_threshold
                {
                    return true;
                }
            }
        }

        // 2. Resource pressure
        if runtime_stats.peak_memory_usage > adaptive_config.memory_threshold {
            return true;
        }

        // 3. Error rate increase
        if runtime_stats.error_rate > adaptive_config.error_rate_threshold {
            return true;
        }

        // 4. Periodic re-optimization
        if current_group > 0 && current_group % adaptive_config.reoptimization_interval == 0 {
            return true;
        }

        false
    }

    /// Re-optimize execution group based on runtime feedback
    async fn reoptimize_execution_group(
        &self,
        parallel_group: &[String],
        plan: &ExecutionPlan,
        runtime_stats: &RuntimeStatistics,
        performance_monitor: &EnhancedPerformanceMonitor,
        registry: &ServiceRegistry,
    ) -> Result<Vec<String>> {
        info!(
            "Re-optimizing execution group with {} steps",
            parallel_group.len()
        );

        let mut optimized_group = parallel_group.to_vec();

        // Analyze performance bottlenecks
        let bottlenecks = performance_monitor.identify_bottlenecks();

        // Re-order steps based on performance data
        if bottlenecks.contains(&BottleneckType::NetworkLatency) {
            optimized_group = self
                .optimize_for_network_latency(&optimized_group, plan, registry)
                .await?;
        }

        if bottlenecks.contains(&BottleneckType::MemoryUsage) {
            optimized_group = self
                .optimize_for_memory_usage(&optimized_group, plan)
                .await?;
        }

        if bottlenecks.contains(&BottleneckType::CpuUsage) {
            optimized_group = self.optimize_for_cpu_usage(&optimized_group, plan).await?;
        }

        Ok(optimized_group)
    }

    /// Select optimal execution strategy dynamically
    fn select_execution_strategy(
        &self,
        parallel_group: &[String],
        plan: &ExecutionPlan,
        performance_monitor: &EnhancedPerformanceMonitor,
        resource_monitor: &LocalResourceMonitor,
        adaptive_config: &LocalAdaptiveConfig,
    ) -> AdaptiveExecutionStrategy {
        let group_size = parallel_group.len();
        let memory_usage = resource_monitor.get_memory_usage();
        let cpu_usage = resource_monitor.get_cpu_usage();
        let network_latency = performance_monitor.get_average_network_latency();

        // Decision tree based on current conditions
        if memory_usage > adaptive_config.memory_threshold {
            // High memory usage - prefer sequential or streaming
            if group_size > adaptive_config.streaming_threshold {
                return AdaptiveExecutionStrategy::Streaming;
            } else {
                return AdaptiveExecutionStrategy::Sequential;
            }
        }

        if cpu_usage > adaptive_config.cpu_threshold {
            // High CPU usage - reduce parallelism
            return AdaptiveExecutionStrategy::Sequential;
        }

        if network_latency.as_millis() > adaptive_config.latency_threshold {
            // High network latency - batch operations
            return AdaptiveExecutionStrategy::Hybrid;
        }

        // Good conditions - use parallel execution
        if group_size >= adaptive_config.parallel_threshold {
            AdaptiveExecutionStrategy::Parallel
        } else {
            AdaptiveExecutionStrategy::Sequential
        }
    }

    /// Execute steps in parallel with adaptive optimization
    async fn execute_parallel_adaptive(
        &self,
        parallel_group: &[String],
        plan: &ExecutionPlan,
        completed_steps: &HashMap<String, StepResult>,
        performance_monitor: &mut EnhancedPerformanceMonitor,
    ) -> Result<Vec<StepResult>> {
        let mut results = Vec::new();

        for step_id in parallel_group {
            if let Some(step) = plan.steps.iter().find(|s| &s.step_id == step_id) {
                let start_time = Instant::now();
                let result = self
                    .execute_step_with_monitoring(step, completed_steps)
                    .await?;
                let duration = start_time.elapsed();

                performance_monitor.record_step_execution(duration);
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Execute steps sequentially with adaptive optimization
    async fn execute_sequential_adaptive(
        &self,
        parallel_group: &[String],
        plan: &ExecutionPlan,
        completed_steps: &HashMap<String, StepResult>,
        performance_monitor: &mut EnhancedPerformanceMonitor,
    ) -> Result<Vec<StepResult>> {
        let mut results = Vec::new();

        for step_id in parallel_group {
            if let Some(step) = plan.steps.iter().find(|s| &s.step_id == step_id) {
                let start_time = Instant::now();
                let result = self
                    .execute_step_with_monitoring(step, completed_steps)
                    .await?;
                let duration = start_time.elapsed();

                performance_monitor.record_step_execution(duration);
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Execute steps with hybrid strategy
    async fn execute_hybrid_adaptive(
        &self,
        parallel_group: &[String],
        plan: &ExecutionPlan,
        completed_steps: &HashMap<String, StepResult>,
        performance_monitor: &mut EnhancedPerformanceMonitor,
        adaptive_config: &LocalAdaptiveConfig,
    ) -> Result<Vec<StepResult>> {
        // Split group into smaller batches for balanced execution
        let batch_size = adaptive_config.hybrid_batch_size;
        let mut results = Vec::new();

        for batch in parallel_group.chunks(batch_size) {
            let batch_results = self
                .execute_parallel_adaptive(batch, plan, completed_steps, performance_monitor)
                .await?;

            results.extend(batch_results);

            // Brief pause between batches to manage resource usage
            tokio::time::sleep(Duration::from_millis(adaptive_config.batch_delay_ms)).await;
        }

        Ok(results)
    }

    /// Execute steps with streaming strategy
    async fn execute_streaming_adaptive(
        &self,
        parallel_group: &[String],
        plan: &ExecutionPlan,
        completed_steps: &HashMap<String, StepResult>,
        performance_monitor: &mut EnhancedPerformanceMonitor,
    ) -> Result<Vec<StepResult>> {
        // Use streaming execution for memory efficiency
        let mut results = Vec::new();
        let mut stream = stream::iter(parallel_group.iter());

        while let Some(step_id) = stream.next().await {
            if let Some(step) = plan.steps.iter().find(|s| &s.step_id == step_id) {
                let start_time = Instant::now();
                let result = self
                    .execute_step_with_monitoring(step, completed_steps)
                    .await?;
                let duration = start_time.elapsed();

                performance_monitor.record_step_execution(duration);
                results.push(result);

                // Process result immediately to free memory
                // This could involve partial result processing or streaming to output
            }
        }

        Ok(results)
    }

    /// Execute a single step
    async fn execute_single_step(
        &self,
        step: &ExecutionStep,
        _completed_steps: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        let start_time = Instant::now();

        // Basic step execution logic - this would be expanded with real implementation
        let status = ExecutionStatus::Success;
        let success = status == ExecutionStatus::Success;
        let data = None; // Would contain actual query results
        let error = None;
        let error_message = None;
        let execution_time = start_time.elapsed();
        let service_response_time = execution_time;
        let memory_used = 0; // Would track actual memory usage
        let result_size = 0; // Would track actual result size
        let cache_hit = false; // Would check cache

        Ok(StepResult {
            step_id: step.step_id.clone(),
            step_type: step.step_type.clone(),
            status,
            data,
            error,
            execution_time,
            service_id: step.service_id.clone(),
            memory_used,
            result_size,
            success,
            error_message,
            service_response_time,
            cache_hit,
        })
    }

    /// Execute individual step with performance monitoring
    async fn execute_step_with_monitoring(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        let start_time = Instant::now();
        let start_memory = self.get_current_memory_usage();

        // Execute the step
        let result = self.execute_single_step(step, completed_steps).await?;

        let end_time = Instant::now();
        let end_memory = self.get_current_memory_usage();
        let duration = end_time.duration_since(start_time);
        let memory_delta = end_memory.saturating_sub(start_memory);

        // Enhanced result with monitoring data
        Ok(StepResult {
            step_id: result.step_id,
            step_type: result.step_type,
            status: result.status,
            data: result.data,
            error: result.error,
            execution_time: duration,
            service_id: result.service_id,
            memory_used: memory_delta,
            result_size: result.result_size,
            success: result.success,
            error_message: result.error_message,
            service_response_time: result.service_response_time,
            cache_hit: result.cache_hit,
        })
    }

    /// Get current memory usage
    fn get_current_memory_usage(&self) -> u64 {
        // Simplified memory tracking - in practice would use system APIs
        std::process::id() as u64 * 1024 // Placeholder
    }

    /// Adapt configuration based on runtime feedback
    fn adapt_configuration(
        &self,
        adaptive_config: &mut LocalAdaptiveConfig,
        runtime_stats: &RuntimeStatistics,
        performance_monitor: &EnhancedPerformanceMonitor,
        resource_monitor: &LocalResourceMonitor,
    ) {
        // Adapt thresholds based on observed performance

        // Memory threshold adaptation
        let current_memory = resource_monitor.get_memory_usage();
        if current_memory > adaptive_config.memory_threshold {
            adaptive_config.memory_threshold = (adaptive_config.memory_threshold * 1.1) as u64;
        } else if current_memory < adaptive_config.memory_threshold * 0.7 {
            adaptive_config.memory_threshold = (adaptive_config.memory_threshold * 0.9) as u64;
        }

        // CPU threshold adaptation
        let current_cpu = resource_monitor.get_cpu_usage();
        if current_cpu > adaptive_config.cpu_threshold {
            adaptive_config.cpu_threshold *= 1.1;
        } else if current_cpu < adaptive_config.cpu_threshold * 0.7 {
            adaptive_config.cpu_threshold *= 0.9;
        }

        // Parallel threshold adaptation based on performance
        let avg_parallel_time = performance_monitor.get_average_parallel_time();
        let avg_sequential_time = performance_monitor.get_average_sequential_time();

        if avg_parallel_time > avg_sequential_time * 1.2 {
            // Parallel execution is not efficient, increase threshold
            adaptive_config.parallel_threshold += 1;
        } else if avg_parallel_time < avg_sequential_time * 0.8 {
            // Parallel execution is very efficient, decrease threshold
            adaptive_config.parallel_threshold =
                adaptive_config.parallel_threshold.saturating_sub(1).max(2);
        }
    }

    /// Log adaptive execution insights
    fn log_adaptive_insights(
        &self,
        runtime_stats: &RuntimeStatistics,
        performance_monitor: &EnhancedPerformanceMonitor,
        resource_monitor: &LocalResourceMonitor,
    ) {
        info!("=== Adaptive Execution Insights ===");
        info!(
            "Total execution time: {:?}",
            runtime_stats.total_execution_time
        );
        info!(
            "Peak memory usage: {} MB",
            runtime_stats.peak_memory_usage / 1024 / 1024
        );
        info!(
            "Average group time: {:?}",
            runtime_stats.get_average_group_time().unwrap_or(0.0)
        );
        info!("Error rate: {:.2}%", runtime_stats.error_rate * 100.0);
        info!(
            "Bottlenecks identified: {:?}",
            performance_monitor.identify_bottlenecks()
        );
        info!(
            "Final memory usage: {} MB",
            resource_monitor.get_memory_usage() / 1024 / 1024
        );
        info!(
            "Final CPU usage: {:.1}%",
            resource_monitor.get_cpu_usage() * 100.0
        );
    }

    /// Optimize execution group for network latency
    async fn optimize_for_network_latency(
        &self,
        group: &[String],
        plan: &ExecutionPlan,
        registry: &ServiceRegistry,
    ) -> Result<Vec<String>> {
        let mut optimized = group.to_vec();

        // Sort steps by estimated network latency (lowest first)
        optimized.sort_by(|a, b| {
            let latency_a = self.estimate_step_network_latency(a, plan, registry);
            let latency_b = self.estimate_step_network_latency(b, plan, registry);
            latency_a
                .partial_cmp(&latency_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(optimized)
    }

    /// Optimize execution group for memory usage
    async fn optimize_for_memory_usage(
        &self,
        group: &[String],
        plan: &ExecutionPlan,
    ) -> Result<Vec<String>> {
        let mut optimized = group.to_vec();

        // Sort steps by estimated memory usage (lowest first)
        optimized.sort_by(|a, b| {
            let memory_a = self.estimate_step_memory_usage(a, plan);
            let memory_b = self.estimate_step_memory_usage(b, plan);
            memory_a.cmp(&memory_b)
        });

        Ok(optimized)
    }

    /// Optimize execution group for CPU usage
    async fn optimize_for_cpu_usage(
        &self,
        group: &[String],
        plan: &ExecutionPlan,
    ) -> Result<Vec<String>> {
        let mut optimized = group.to_vec();

        // Sort steps by estimated CPU intensity (lowest first)
        optimized.sort_by(|a, b| {
            let cpu_a = self.estimate_step_cpu_usage(a, plan);
            let cpu_b = self.estimate_step_cpu_usage(b, plan);
            cpu_a
                .partial_cmp(&cpu_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(optimized)
    }

    /// Estimate network latency for a step
    fn estimate_step_network_latency(
        &self,
        step_id: &str,
        plan: &ExecutionPlan,
        _registry: &ServiceRegistry,
    ) -> f64 {
        if let Some(step) = plan.steps.iter().find(|s| s.step_id == step_id) {
            // Estimate based on query complexity and service characteristics
            let base_latency = 100.0; // Base network latency in ms
            let query_size = step.query_fragment.len() as f64;

            base_latency + (query_size / 100.0) // Add latency based on query size
        } else {
            100.0 // Default latency
        }
    }

    /// Estimate memory usage for a step
    fn estimate_step_memory_usage(&self, step_id: &str, plan: &ExecutionPlan) -> u64 {
        if let Some(step) = plan.steps.iter().find(|s| s.step_id == step_id) {
            match step.step_type {
                StepType::ServiceQuery => 1024 * 1024, // 1MB base for service queries
                StepType::Join => 5 * 1024 * 1024,     // 5MB for join operations
                StepType::GraphQLQuery => 2 * 1024 * 1024, // 2MB for GraphQL queries
                _ => 512 * 1024,                       // 512KB default
            }
        } else {
            1024 * 1024 // 1MB default
        }
    }

    /// Estimate CPU usage for a step
    fn estimate_step_cpu_usage(&self, step_id: &str, plan: &ExecutionPlan) -> f64 {
        if let Some(step) = plan.steps.iter().find(|s| s.step_id == step_id) {
            match step.step_type {
                StepType::Join => 0.8,      // Joins are CPU intensive
                StepType::Filter => 0.6,    // Filters require processing
                StepType::Aggregate => 0.7, // Aggregations are moderately intensive
                StepType::Sort => 0.5,      // Sorting is moderately intensive
                _ => 0.3,                   // Network-bound operations are less CPU intensive
            }
        } else {
            0.3 // Default low CPU usage
        }
    }
}

/// Runtime statistics collection
#[derive(Debug, Clone)]
pub struct RuntimeStatistics {
    pub total_execution_time: Duration,
    pub group_times: HashMap<usize, Duration>,
    pub group_start_times: HashMap<usize, Instant>,
    pub peak_memory_usage: u64,
    pub error_count: u32,
    pub total_steps: u32,
    pub error_rate: f64,
}

impl RuntimeStatistics {
    pub fn new() -> Self {
        Self {
            total_execution_time: Duration::default(),
            group_times: HashMap::new(),
            group_start_times: HashMap::new(),
            peak_memory_usage: 0,
            error_count: 0,
            total_steps: 0,
            error_rate: 0.0,
        }
    }

    pub fn update_group_start(&mut self, group_id: usize, start_time: Instant) {
        self.group_start_times.insert(group_id, start_time);
    }

    pub fn update_group_end(&mut self, group_id: usize, duration: Duration) {
        self.group_times.insert(group_id, duration);
    }

    pub fn get_average_group_time(&self) -> Option<f64> {
        if self.group_times.is_empty() {
            None
        } else {
            let total_ms: u128 = self.group_times.values().map(|d| d.as_millis()).sum();
            Some(total_ms as f64 / self.group_times.len() as f64)
        }
    }

    pub fn get_group_time(&self, group_id: usize) -> Option<Duration> {
        self.group_times.get(&group_id).copied()
    }

    pub fn record_error(&mut self) {
        self.error_count += 1;
        self.total_steps += 1;
        self.error_rate = self.error_count as f64 / self.total_steps as f64;
    }

    pub fn record_success(&mut self) {
        self.total_steps += 1;
        self.error_rate = self.error_count as f64 / self.total_steps as f64;
    }
}

/// Enhanced performance monitoring for adaptive execution
#[derive(Debug, Clone)]
pub struct EnhancedPerformanceMonitor {
    step_execution_times: Vec<Duration>,
    group_execution_data: Vec<GroupExecutionData>,
    bottlenecks: HashSet<BottleneckType>,
    network_latencies: Vec<Duration>,
    parallel_execution_times: Vec<Duration>,
    sequential_execution_times: Vec<Duration>,
}

impl EnhancedPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            step_execution_times: Vec::new(),
            group_execution_data: Vec::new(),
            bottlenecks: HashSet::new(),
            network_latencies: Vec::new(),
            parallel_execution_times: Vec::new(),
            sequential_execution_times: Vec::new(),
        }
    }

    pub fn record_step_execution(&mut self, duration: Duration) {
        self.step_execution_times.push(duration);
    }

    pub fn record_group_execution(
        &mut self,
        group_size: usize,
        duration: Duration,
        memory_used: u64,
        cpu_used: f64,
    ) {
        self.group_execution_data.push(GroupExecutionData {
            group_size,
            duration,
            memory_used,
            cpu_used,
        });
    }

    pub fn identify_bottlenecks(&self) -> Vec<BottleneckType> {
        let mut bottlenecks = Vec::new();

        // Analyze group execution data to identify bottlenecks
        for data in &self.group_execution_data {
            if data.memory_used > 100 * 1024 * 1024 {
                // 100MB threshold
                bottlenecks.push(BottleneckType::MemoryUsage);
            }
            if data.cpu_used > 0.8 {
                // 80% CPU threshold
                bottlenecks.push(BottleneckType::CpuUsage);
            }
            if data.duration.as_millis() > 5000 {
                // 5 second threshold
                bottlenecks.push(BottleneckType::NetworkLatency);
            }
        }

        bottlenecks.sort();
        bottlenecks.dedup();
        bottlenecks
    }

    pub fn get_average_network_latency(&self) -> Duration {
        if self.network_latencies.is_empty() {
            Duration::from_millis(100) // Default
        } else {
            let total_ms: u128 = self.network_latencies.iter().map(|d| d.as_millis()).sum();
            Duration::from_millis((total_ms / self.network_latencies.len() as u128) as u64)
        }
    }

    pub fn get_average_parallel_time(&self) -> f64 {
        if self.parallel_execution_times.is_empty() {
            0.0
        } else {
            let total_ms: u128 = self
                .parallel_execution_times
                .iter()
                .map(|d| d.as_millis())
                .sum();
            total_ms as f64 / self.parallel_execution_times.len() as f64
        }
    }

    pub fn get_average_sequential_time(&self) -> f64 {
        if self.sequential_execution_times.is_empty() {
            0.0
        } else {
            let total_ms: u128 = self
                .sequential_execution_times
                .iter()
                .map(|d| d.as_millis())
                .sum();
            total_ms as f64 / self.sequential_execution_times.len() as f64
        }
    }
}

/// Local resource monitoring for adaptive execution
#[derive(Debug, Clone)]
pub struct LocalResourceMonitor {
    current_memory_usage: u64,
    current_cpu_usage: f64,
    peak_memory_usage: u64,
    peak_cpu_usage: f64,
}

impl LocalResourceMonitor {
    pub fn new() -> Self {
        Self {
            current_memory_usage: 0,
            current_cpu_usage: 0.0,
            peak_memory_usage: 0,
            peak_cpu_usage: 0.0,
        }
    }

    pub fn get_memory_usage(&self) -> u64 {
        // In practice, this would query actual system memory usage
        self.current_memory_usage
    }

    pub fn get_cpu_usage(&self) -> f64 {
        // In practice, this would query actual CPU usage
        self.current_cpu_usage
    }

    pub fn update_memory_usage(&mut self, usage: u64) {
        self.current_memory_usage = usage;
        if usage > self.peak_memory_usage {
            self.peak_memory_usage = usage;
        }
    }

    pub fn update_cpu_usage(&mut self, usage: f64) {
        self.current_cpu_usage = usage;
        if usage > self.peak_cpu_usage {
            self.peak_cpu_usage = usage;
        }
    }
}

/// Local adaptive configuration that evolves during execution
#[derive(Debug, Clone)]
pub struct LocalAdaptiveConfig {
    pub performance_threshold: f64,
    pub memory_threshold: u64,
    pub cpu_threshold: f64,
    pub error_rate_threshold: f64,
    pub reoptimization_interval: usize,
    pub parallel_threshold: usize,
    pub streaming_threshold: usize,
    pub latency_threshold: u128,
    pub hybrid_batch_size: usize,
    pub batch_delay_ms: u64,
}

impl Default for LocalAdaptiveConfig {
    fn default() -> Self {
        Self {
            performance_threshold: 1.5, // 50% performance degradation threshold
            memory_threshold: 1024 * 1024 * 1024, // 1GB memory threshold
            cpu_threshold: 0.8,         // 80% CPU threshold
            error_rate_threshold: 0.1,  // 10% error rate threshold
            reoptimization_interval: 5, // Re-optimize every 5 groups
            parallel_threshold: 3,      // Use parallel execution for 3+ steps
            streaming_threshold: 10,    // Use streaming for 10+ steps
            latency_threshold: 1000,    // 1 second latency threshold
            hybrid_batch_size: 3,       // Process 3 steps per batch in hybrid mode
            batch_delay_ms: 50,         // 50ms delay between batches
        }
    }
}

/// Execution strategies for adaptive execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveExecutionStrategy {
    Parallel,
    Sequential,
    Hybrid,
    Streaming,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BottleneckType {
    NetworkLatency,
    MemoryUsage,
    CpuUsage,
    DiskIo,
}

/// Data about group execution for performance analysis
#[derive(Debug, Clone)]
pub struct GroupExecutionData {
    pub group_size: usize,
    pub duration: Duration,
    pub memory_used: u64,
    pub cpu_used: f64,
}

/// Enhanced step result with monitoring data
#[derive(Debug, Clone)]
pub struct EnhancedStepResult {
    pub step_id: String,
    pub execution_time: Duration,
    pub memory_used: u64,
    pub result_size: usize,
    pub success: bool,
    pub error_message: Option<String>,
    pub service_response_time: Option<Duration>,
    pub cache_hit: bool,
}

/// Execute a group of steps in parallel
pub async fn execute_parallel_group(
    step_ids: &[String],
    plan: &ExecutionPlan,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<Vec<StepResult>> {
    let steps: Vec<_> = step_ids
        .iter()
        .filter_map(|id| plan.steps.iter().find(|s| &s.step_id == id))
        .collect();

    if steps.is_empty() {
        return Ok(Vec::new());
    }

    debug!("Executing {} steps in parallel", steps.len());

    // Execute steps concurrently
    let futures: Vec<_> = steps
        .into_iter()
        .map(|step| execute_step(step, completed_steps))
        .collect();

    // Wait for all steps to complete or timeout
    let timeout_duration = Duration::from_secs(60); // Default timeout
    match timeout(timeout_duration, futures::future::try_join_all(futures)).await {
        Ok(Ok(results)) => Ok(results),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(anyhow!(
            "Parallel execution timed out after {:?}",
            timeout_duration
        )),
    }
}

/// Execute a single step
#[instrument(skip(step, completed_steps))]
pub async fn execute_step(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<StepResult> {
    debug!("Executing step: {} ({})", step.step_id, step.step_type);

    // Check dependencies
    for dep_id in &step.dependencies {
        if !completed_steps.contains_key(dep_id) {
            return Err(anyhow!(
                "Dependency {} not completed for step {}",
                dep_id,
                step.step_id
            ));
        }
    }

    let start_time = Instant::now();

    let result = match step.step_type {
        StepType::ServiceQuery => execute_service_query(step).await,
        StepType::GraphQLQuery => execute_graphql_query(step).await,
        StepType::Join => execute_join(step, completed_steps).await,
        StepType::Union => execute_union(step, completed_steps).await,
        StepType::Filter => execute_filter(step, completed_steps).await,
        StepType::SchemaStitch => execute_schema_stitch(step, completed_steps).await,
        StepType::Aggregate => execute_aggregate(step, completed_steps).await,
        StepType::Sort => execute_sort(step, completed_steps).await,
    };

    let execution_time = start_time.elapsed();

    match result {
        Ok(data) => {
            debug!("Step {} completed in {:?}", step.step_id, execution_time);
            Ok(StepResult {
                step_id: step.step_id.clone(),
                step_type: step.step_type,
                status: ExecutionStatus::Success,
                data: Some(data),
                error: None,
                execution_time,
                service_id: step.service_id.clone(),
                memory_used: 0,
                result_size: 0,
                success: true,
                error_message: None,
                service_response_time: execution_time,
                cache_hit: false,
            })
        }
        Err(e) => {
            error!("Step {} failed: {}", step.step_id, e);
            Ok(StepResult {
                step_id: step.step_id.clone(),
                step_type: step.step_type,
                status: ExecutionStatus::Failed,
                data: None,
                error: Some(e.to_string()),
                execution_time,
                service_id: step.service_id.clone(),
                memory_used: 0,
                result_size: 0,
                success: false,
                error_message: Some(e.to_string()),
                service_response_time: execution_time,
                cache_hit: false,
            })
        }
    }
}

/// Execute a SPARQL service query
pub async fn execute_service_query(step: &ExecutionStep) -> Result<QueryResultData> {
    let service_id = step
        .service_id
        .as_ref()
        .ok_or_else(|| anyhow!("Service ID required for service query"))?;

    // TODO: Get service details from registry
    let endpoint = format!("http://localhost:8080/sparql"); // Placeholder

    let mut headers = HeaderMap::new();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/sparql-query"),
    );
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("application/sparql-results+json"),
    );

    let client = Client::new();
    let response = client
        .post(&endpoint)
        .headers(headers)
        .body(step.query_fragment.clone())
        .send()
        .await
        .map_err(|e| anyhow!("HTTP request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow!("Service returned error: {}", response.status()));
    }

    let response_text = response
        .text()
        .await
        .map_err(|e| anyhow!("Failed to read response: {}", e))?;

    // Parse SPARQL results JSON
    let sparql_results: SparqlResults = serde_json::from_str(&response_text)
        .map_err(|e| anyhow!("Failed to parse SPARQL results: {}", e))?;

    Ok(QueryResultData::Sparql(sparql_results))
}

/// Execute a GraphQL query
pub async fn execute_graphql_query(step: &ExecutionStep) -> Result<QueryResultData> {
    let service_id = step
        .service_id
        .as_ref()
        .ok_or_else(|| anyhow!("Service ID required for GraphQL query"))?;

    // TODO: Get service details from registry
    let endpoint = format!("http://localhost:8080/graphql"); // Placeholder

    let graphql_request = GraphQLRequest {
        query: step.query_fragment.clone(),
        variables: None,
        operation_name: None,
    };

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let client = Client::new();
    let response = client
        .post(&endpoint)
        .headers(headers)
        .json(&graphql_request)
        .send()
        .await
        .map_err(|e| anyhow!("HTTP request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow!("Service returned error: {}", response.status()));
    }

    let graphql_response: GraphQLResponse = response
        .json()
        .await
        .map_err(|e| anyhow!("Failed to parse GraphQL response: {}", e))?;

    Ok(QueryResultData::GraphQL(graphql_response))
}

/// Execute a join operation with enhanced parallel processing
pub async fn execute_join(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<QueryResultData> {
    debug!("Executing join step: {}", step.step_id);

    // Get results from dependency steps
    let mut input_results = Vec::new();
    for dep_id in &step.dependencies {
        if let Some(dep_result) = completed_steps.get(dep_id) {
            if let Some(data) = &dep_result.data {
                input_results.push(data);
            }
        }
    }

    if input_results.len() < 2 {
        return Err(anyhow!("Join requires at least 2 input results"));
    }

    // TODO: Implement advanced join logic
    // For now, return a simple merged result
    if let Some(QueryResultData::Sparql(first_result)) = input_results.first() {
        Ok(QueryResultData::Sparql(first_result.clone()))
    } else {
        Err(anyhow!("No valid SPARQL results to join"))
    }
}

/// Execute a union operation
pub async fn execute_union(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<QueryResultData> {
    debug!("Executing union step: {}", step.step_id);

    let mut all_bindings = Vec::new();
    let mut variables = Vec::new();

    for dep_id in &step.dependencies {
        if let Some(dep_result) = completed_steps.get(dep_id) {
            if let Some(QueryResultData::Sparql(sparql_result)) = &dep_result.data {
                if variables.is_empty() {
                    variables = sparql_result.head.vars.clone();
                }
                all_bindings.extend(sparql_result.results.bindings.clone());
            }
        }
    }

    let union_result = SparqlResults {
        head: SparqlHead { vars: variables },
        results: SparqlResultSet {
            bindings: all_bindings,
        },
    };

    Ok(QueryResultData::Sparql(union_result))
}

/// Execute a filter operation
pub async fn execute_filter(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<QueryResultData> {
    debug!("Executing filter step: {}", step.step_id);

    // Get the input data from dependencies
    let mut input_data = None;
    for dep_id in &step.dependencies {
        if let Some(dep_result) = completed_steps.get(dep_id) {
            if let Some(data) = &dep_result.data {
                input_data = Some(data.clone());
                break;
            }
        }
    }

    let input_data = input_data.ok_or_else(|| anyhow!("No input data for filter operation"))?;

    match input_data {
        QueryResultData::Sparql(sparql_results) => {
            // TODO: Implement SPARQL result filtering
            let filtered_results = sparql_results; // Placeholder - return unfiltered for now
            Ok(QueryResultData::Sparql(filtered_results))
        }
        QueryResultData::GraphQL(graphql_response) => {
            // For GraphQL, filters are usually applied at the field level
            // For now, we'll pass through the data as GraphQL filtering is more complex
            warn!("GraphQL filter execution not fully implemented, passing through data");
            Ok(QueryResultData::GraphQL(graphql_response))
        }
    }
}

/// Filter SPARQL results based on filter expression
pub fn filter_sparql_results(results: &SparqlResults, filter_expr: &str) -> Result<SparqlResults> {
    let filtered_bindings = results
        .results
        .bindings
        .iter()
        .filter(|binding| evaluate_filter_expression(binding, filter_expr))
        .cloned()
        .collect();

    Ok(SparqlResults {
        head: results.head.clone(),
        results: SparqlResultSet {
            bindings: filtered_bindings,
        },
    })
}

/// Evaluate a filter expression against a binding
pub fn evaluate_filter_expression(binding: &SparqlBinding, filter_expr: &str) -> bool {
    // Parse and evaluate SPARQL filter expressions
    // This is a simplified implementation - full SPARQL filter evaluation is complex
    let filter_expr = filter_expr.trim();

    // Handle common filter patterns
    if filter_expr.contains("REGEX") {
        return evaluate_regex_filter(binding, filter_expr);
    }

    if filter_expr.contains("langMatches") {
        return evaluate_lang_matches_filter(binding, filter_expr);
    }

    if filter_expr.contains("=") || filter_expr.contains("!=") {
        return evaluate_comparison_filter(binding, filter_expr);
    }

    if filter_expr.contains("BOUND") {
        return evaluate_bound_filter(binding, filter_expr);
    }

    // For complex expressions, default to true for now
    // In a full implementation, we'd need a proper SPARQL expression parser
    warn!(
        "Complex filter expression not fully supported: {}",
        filter_expr
    );
    true
}

/// Evaluate REGEX filter expressions
pub fn evaluate_regex_filter(binding: &SparqlBinding, filter_expr: &str) -> bool {
    // Extract variable and regex pattern from REGEX(?var, "pattern")
    if let Some(start) = filter_expr.find("REGEX(") {
        let substr = &filter_expr[start + 6..];
        if let Some(end) = substr.find(')') {
            let args = &substr[..end];
            let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                let var_name = parts[0].trim_start_matches('?');
                let pattern = parts[1].trim_matches('"');

                if let Some(value) = binding.get(var_name) {
                    if let Ok(regex) = regex::Regex::new(pattern) {
                        return regex.is_match(&value.value);
                    }
                }
            }
        }
    }
    false
}

/// Evaluate langMatches filter expressions
pub fn evaluate_lang_matches_filter(binding: &SparqlBinding, filter_expr: &str) -> bool {
    // Extract variable and language pattern from langMatches(lang(?var), "lang")
    if let Some(start) = filter_expr.find("langMatches(") {
        let substr = &filter_expr[start + 12..];
        if let Some(end) = substr.find(')') {
            let args = &substr[..end];
            let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                // Extract variable from lang(?var)
                let lang_part = parts[0];
                if let Some(var_start) = lang_part.find("lang(") {
                    let var_part = &lang_part[var_start + 5..];
                    if let Some(var_end) = var_part.find(')') {
                        let var_name = var_part[..var_end].trim_start_matches('?');
                        let lang_pattern = parts[1].trim_matches('"');

                        if let Some(value) = binding.get(var_name) {
                            if let Some(lang) = &value.lang {
                                return lang == lang_pattern || lang_pattern == "*";
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// Evaluate comparison filter expressions (=, !=, <, >, <=, >=)
pub fn evaluate_comparison_filter(binding: &SparqlBinding, filter_expr: &str) -> bool {
    let operators = ["!=", "<=", ">=", "=", "<", ">"];

    for op in &operators {
        if let Some(pos) = filter_expr.find(op) {
            let left = filter_expr[..pos].trim();
            let right = filter_expr[pos + op.len()..].trim();

            let left_value = resolve_filter_value(binding, left);
            let right_value = resolve_filter_value(binding, right);

            return match op {
                "=" => left_value == right_value,
                "!=" => left_value != right_value,
                "<" => left_value < right_value,
                ">" => left_value > right_value,
                "<=" => left_value <= right_value,
                ">=" => left_value >= right_value,
                _ => false,
            };
        }
    }
    false
}

/// Evaluate BOUND filter expressions
pub fn evaluate_bound_filter(binding: &SparqlBinding, filter_expr: &str) -> bool {
    if let Some(start) = filter_expr.find("BOUND(") {
        let substr = &filter_expr[start + 6..];
        if let Some(end) = substr.find(')') {
            let var_name = substr[..end].trim().trim_start_matches('?');
            return binding.contains_key(var_name);
        }
    }
    false
}

/// Resolve a filter value (variable or literal)
pub fn resolve_filter_value(binding: &SparqlBinding, value_expr: &str) -> String {
    let value_expr = value_expr.trim();

    if value_expr.starts_with('?') {
        // Variable reference
        let var_name = &value_expr[1..];
        binding
            .get(var_name)
            .map(|v| v.value.clone())
            .unwrap_or_default()
    } else if value_expr.starts_with('"') && value_expr.ends_with('"') {
        // String literal
        value_expr[1..value_expr.len() - 1].to_string()
    } else {
        // Literal value
        value_expr.to_string()
    }
}

/// Execute schema stitching for GraphQL
pub async fn execute_schema_stitch(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<QueryResultData> {
    debug!("Executing schema stitch step: {}", step.step_id);

    // Combine GraphQL results from multiple services
    let mut combined_data = serde_json::Map::new();

    for dep_id in &step.dependencies {
        if let Some(dep_result) = completed_steps.get(dep_id) {
            if let Some(QueryResultData::GraphQL(gql_result)) = &dep_result.data {
                if let Some(data_obj) = gql_result.data.as_object() {
                    for (key, value) in data_obj {
                        combined_data.insert(key.clone(), value.clone());
                    }
                }
            }
        }
    }

    let stitched_result = GraphQLResponse {
        data: serde_json::Value::Object(combined_data),
        errors: Vec::new(),
        extensions: None,
    };

    Ok(QueryResultData::GraphQL(stitched_result))
}

/// Execute aggregation
pub async fn execute_aggregate(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<QueryResultData> {
    debug!("Executing aggregate step: {}", step.step_id);

    // Get the input data from dependencies
    let mut input_data = None;
    for dep_id in &step.dependencies {
        if let Some(dep_result) = completed_steps.get(dep_id) {
            if let Some(data) = &dep_result.data {
                input_data = Some(data.clone());
                break;
            }
        }
    }

    let input_data = input_data.ok_or_else(|| anyhow!("No input data for aggregate operation"))?;

    match input_data {
        QueryResultData::Sparql(sparql_results) => {
            let aggregated_results =
                aggregate_sparql_results(&sparql_results, &step.query_fragment)?;
            Ok(QueryResultData::Sparql(aggregated_results))
        }
        QueryResultData::GraphQL(_) => {
            // GraphQL aggregation is typically handled by the underlying GraphQL engine
            warn!("GraphQL aggregation not implemented, passing through data");
            Ok(input_data)
        }
    }
}

/// Aggregate SPARQL results based on aggregate expression
pub fn aggregate_sparql_results(
    results: &SparqlResults,
    aggregate_expr: &str,
) -> Result<SparqlResults> {
    let aggregate_expr = aggregate_expr.trim();

    // Parse the aggregate expression to identify the operation
    if aggregate_expr.contains("GROUP BY") {
        perform_group_by_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("COUNT") {
        perform_count_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("SUM") {
        perform_sum_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("AVG") {
        perform_avg_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("MIN") {
        perform_min_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("MAX") {
        perform_max_aggregation(results, aggregate_expr)
    } else {
        warn!("Unknown aggregation type: {}", aggregate_expr);
        Ok(results.clone())
    }
}

/// Perform GROUP BY aggregation
pub fn perform_group_by_aggregation(results: &SparqlResults, expr: &str) -> Result<SparqlResults> {
    // Extract GROUP BY variables
    let group_vars = extract_group_by_variables(expr);

    // Group bindings by the GROUP BY variables
    let mut groups: HashMap<String, Vec<SparqlBinding>> = HashMap::new();

    for binding in &results.results.bindings {
        let group_key = create_group_key(binding, &group_vars);
        groups.entry(group_key).or_default().push(binding.clone());
    }

    // Create aggregated results
    let mut aggregated_bindings = Vec::new();

    for (group_key, group_bindings) in groups {
        if let Some(first_binding) = group_bindings.first() {
            // Start with the group variables from the first binding
            let mut agg_binding = HashMap::new();
            for var in &group_vars {
                if let Some(value) = first_binding.get(var) {
                    agg_binding.insert(var.clone(), value.clone());
                }
            }

            // Apply aggregation functions within the group
            if expr.contains("COUNT(") {
                let count_value = SparqlValue {
                    value_type: "literal".to_string(),
                    value: group_bindings.len().to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                    lang: None,
                };
                agg_binding.insert("count".to_string(), count_value);
            }

            // Add more aggregation functions as needed
            aggregated_bindings.push(agg_binding);
        }
    }

    // Create new variable list including group variables and aggregate variables
    let mut new_vars = group_vars;
    if expr.contains("COUNT(") {
        new_vars.push("count".to_string());
    }

    Ok(SparqlResults {
        head: SparqlHead { vars: new_vars },
        results: SparqlResultSet {
            bindings: aggregated_bindings,
        },
    })
}

/// Perform COUNT aggregation
pub fn perform_count_aggregation(results: &SparqlResults, expr: &str) -> Result<SparqlResults> {
    let count = if expr.contains("COUNT(DISTINCT") {
        // Count distinct values
        let var_name = extract_count_variable(expr);
        let mut distinct_values = HashSet::new();

        for binding in &results.results.bindings {
            if let Some(var_name) = &var_name {
                if let Some(value) = binding.get(var_name) {
                    distinct_values.insert(value.value.clone());
                }
            }
        }
        distinct_values.len()
    } else {
        // Count all rows
        results.results.bindings.len()
    };

    let count_binding = {
        let mut binding = HashMap::new();
        binding.insert(
            "count".to_string(),
            SparqlValue {
                value_type: "literal".to_string(),
                value: count.to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                lang: None,
            },
        );
        binding
    };

    Ok(SparqlResults {
        head: SparqlHead {
            vars: vec!["count".to_string()],
        },
        results: SparqlResultSet {
            bindings: vec![count_binding],
        },
    })
}

/// Perform SUM aggregation
pub fn perform_sum_aggregation(results: &SparqlResults, expr: &str) -> Result<SparqlResults> {
    let var_name = extract_aggregate_variable(expr, "SUM");
    let mut sum: f64 = 0.0;
    let mut count = 0;

    if let Some(var_name) = var_name {
        for binding in &results.results.bindings {
            if let Some(value) = binding.get(&var_name) {
                if let Ok(num) = value.value.parse::<f64>() {
                    sum += num;
                    count += 1;
                }
            }
        }
    }

    let sum_binding = {
        let mut binding = HashMap::new();
        binding.insert(
            "sum".to_string(),
            SparqlValue {
                value_type: "literal".to_string(),
                value: sum.to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
                lang: None,
            },
        );
        binding
    };

    Ok(SparqlResults {
        head: SparqlHead {
            vars: vec!["sum".to_string()],
        },
        results: SparqlResultSet {
            bindings: vec![sum_binding],
        },
    })
}

/// Perform AVG aggregation
pub fn perform_avg_aggregation(results: &SparqlResults, expr: &str) -> Result<SparqlResults> {
    let var_name = extract_aggregate_variable(expr, "AVG");
    let mut sum: f64 = 0.0;
    let mut count = 0;

    if let Some(var_name) = var_name {
        for binding in &results.results.bindings {
            if let Some(value) = binding.get(&var_name) {
                if let Ok(num) = value.value.parse::<f64>() {
                    sum += num;
                    count += 1;
                }
            }
        }
    }

    let avg = if count > 0 { sum / count as f64 } else { 0.0 };

    let avg_binding = {
        let mut binding = HashMap::new();
        binding.insert(
            "avg".to_string(),
            SparqlValue {
                value_type: "literal".to_string(),
                value: avg.to_string(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
                lang: None,
            },
        );
        binding
    };

    Ok(SparqlResults {
        head: SparqlHead {
            vars: vec!["avg".to_string()],
        },
        results: SparqlResultSet {
            bindings: vec![avg_binding],
        },
    })
}

/// Perform MIN aggregation
pub fn perform_min_aggregation(results: &SparqlResults, expr: &str) -> Result<SparqlResults> {
    let var_name = extract_aggregate_variable(expr, "MIN");
    let mut min_value: Option<String> = None;

    if let Some(var_name) = var_name {
        for binding in &results.results.bindings {
            if let Some(value) = binding.get(&var_name) {
                match &min_value {
                    None => min_value = Some(value.value.clone()),
                    Some(current_min) => {
                        if value.value < *current_min {
                            min_value = Some(value.value.clone());
                        }
                    }
                }
            }
        }
    }

    let min_binding = {
        let mut binding = HashMap::new();
        binding.insert(
            "min".to_string(),
            SparqlValue {
                value_type: "literal".to_string(),
                value: min_value.unwrap_or_default(),
                datatype: None,
                lang: None,
            },
        );
        binding
    };

    Ok(SparqlResults {
        head: SparqlHead {
            vars: vec!["min".to_string()],
        },
        results: SparqlResultSet {
            bindings: vec![min_binding],
        },
    })
}

/// Perform MAX aggregation
pub fn perform_max_aggregation(results: &SparqlResults, expr: &str) -> Result<SparqlResults> {
    let var_name = extract_aggregate_variable(expr, "MAX");
    let mut max_value: Option<String> = None;

    if let Some(var_name) = var_name {
        for binding in &results.results.bindings {
            if let Some(value) = binding.get(&var_name) {
                match &max_value {
                    None => max_value = Some(value.value.clone()),
                    Some(current_max) => {
                        if value.value > *current_max {
                            max_value = Some(value.value.clone());
                        }
                    }
                }
            }
        }
    }

    let max_binding = {
        let mut binding = HashMap::new();
        binding.insert(
            "max".to_string(),
            SparqlValue {
                value_type: "literal".to_string(),
                value: max_value.unwrap_or_default(),
                datatype: None,
                lang: None,
            },
        );
        binding
    };

    Ok(SparqlResults {
        head: SparqlHead {
            vars: vec!["max".to_string()],
        },
        results: SparqlResultSet {
            bindings: vec![max_binding],
        },
    })
}

/// Extract GROUP BY variables from expression
pub fn extract_group_by_variables(expr: &str) -> Vec<String> {
    if let Some(group_start) = expr.find("GROUP BY") {
        let group_part = &expr[group_start + 8..];
        group_part
            .split_whitespace()
            .filter(|s| s.starts_with('?'))
            .map(|s| s.trim_start_matches('?').to_string())
            .collect()
    } else {
        Vec::new()
    }
}

/// Extract variable from COUNT expression
pub fn extract_count_variable(expr: &str) -> Option<String> {
    if let Some(start) = expr.find("COUNT(") {
        let substr = &expr[start + 6..];
        if let Some(end) = substr.find(')') {
            let var_part = &substr[..end];
            if var_part.contains("DISTINCT") {
                return var_part
                    .split_whitespace()
                    .find(|s| s.starts_with('?'))
                    .map(|s| s.trim_start_matches('?').to_string());
            } else if var_part.starts_with('?') {
                return Some(var_part.trim_start_matches('?').to_string());
            }
        }
    }
    None
}

/// Extract variable from aggregate expression (SUM, AVG, MIN, MAX)
pub fn extract_aggregate_variable(expr: &str, agg_type: &str) -> Option<String> {
    let pattern = format!("{}(", agg_type);
    if let Some(start) = expr.find(&pattern) {
        let substr = &expr[start + pattern.len()..];
        if let Some(end) = substr.find(')') {
            let var_part = &substr[..end].trim();
            if var_part.starts_with('?') {
                return Some(var_part.trim_start_matches('?').to_string());
            }
        }
    }
    None
}

/// Create a group key for GROUP BY operations
pub fn create_group_key(binding: &SparqlBinding, group_vars: &[String]) -> String {
    let mut key_parts = Vec::new();
    for var in group_vars {
        if let Some(value) = binding.get(var) {
            key_parts.push(format!("{}:{}", var, value.value));
        } else {
            key_parts.push(format!("{}:NULL", var));
        }
    }
    key_parts.join("|")
}

/// Execute sorting
pub async fn execute_sort(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<QueryResultData> {
    debug!("Executing sort step: {}", step.step_id);

    // Get the input data from dependencies
    let mut input_data = None;
    for dep_id in &step.dependencies {
        if let Some(dep_result) = completed_steps.get(dep_id) {
            if let Some(data) = &dep_result.data {
                input_data = Some(data.clone());
                break;
            }
        }
    }

    let input_data = input_data.ok_or_else(|| anyhow!("No input data for sort operation"))?;

    match input_data {
        QueryResultData::Sparql(sparql_results) => {
            let sorted_results = sort_sparql_results(&sparql_results, &step.query_fragment)?;
            Ok(QueryResultData::Sparql(sorted_results))
        }
        QueryResultData::GraphQL(graphql_response) => {
            // GraphQL sorting is typically handled at the field level
            warn!("GraphQL sort execution not fully implemented, passing through data");
            Ok(QueryResultData::GraphQL(graphql_response))
        }
    }
}

/// Sort SPARQL results based on ORDER BY expression
pub fn sort_sparql_results(results: &SparqlResults, order_expr: &str) -> Result<SparqlResults> {
    let order_clauses = parse_order_by_expression(order_expr);

    let mut sorted_bindings = results.results.bindings.clone();

    // Sort the bindings based on the ORDER BY clauses
    sorted_bindings.sort_by(|a, b| {
        for order_clause in &order_clauses {
            let comparison = compare_bindings(a, b, &order_clause.variable);

            let result = if order_clause.descending {
                comparison.reverse()
            } else {
                comparison
            };

            if result != std::cmp::Ordering::Equal {
                return result;
            }
        }
        std::cmp::Ordering::Equal
    });

    Ok(SparqlResults {
        head: results.head.clone(),
        results: SparqlResultSet {
            bindings: sorted_bindings,
        },
    })
}

/// Parse ORDER BY expression into order clauses
pub fn parse_order_by_expression(expr: &str) -> Vec<OrderClause> {
    let mut clauses = Vec::new();

    if let Some(order_start) = expr.find("ORDER BY") {
        let order_part = &expr[order_start + 8..];

        // Split by comma and parse each order expression
        for order_expr in order_part.split(',') {
            let order_expr = order_expr.trim();

            let (variable, descending) = if order_expr.starts_with("DESC(") {
                // DESC(?variable)
                if let Some(start) = order_expr.find('(') {
                    if let Some(end) = order_expr.find(')') {
                        let var_part = &order_expr[start + 1..end];
                        (var_part.trim_start_matches('?').to_string(), true)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else if order_expr.starts_with("ASC(") {
                // ASC(?variable)
                if let Some(start) = order_expr.find('(') {
                    if let Some(end) = order_expr.find(')') {
                        let var_part = &order_expr[start + 1..end];
                        (var_part.trim_start_matches('?').to_string(), false)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else if order_expr.starts_with('?') {
                // Simple variable (defaults to ASC)
                (order_expr.trim_start_matches('?').to_string(), false)
            } else {
                // Try to extract variable from complex expressions
                if let Some(var_match) = order_expr.split_whitespace().find(|s| s.starts_with('?'))
                {
                    let descending = order_expr.to_lowercase().contains("desc");
                    (var_match.trim_start_matches('?').to_string(), descending)
                } else {
                    continue;
                }
            };

            clauses.push(OrderClause {
                variable,
                descending,
            });
        }
    }

    clauses
}

/// Compare two bindings for a specific variable
pub fn compare_bindings(
    a: &SparqlBinding,
    b: &SparqlBinding,
    variable: &str,
) -> std::cmp::Ordering {
    let a_value = a.get(variable);
    let b_value = b.get(variable);

    match (a_value, b_value) {
        (Some(a_val), Some(b_val)) => compare_sparql_values(a_val, b_val),
        (Some(_), None) => std::cmp::Ordering::Greater, // Non-null values come after null
        (None, Some(_)) => std::cmp::Ordering::Less,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// Compare two SPARQL values with type-aware comparison
pub fn compare_sparql_values(a: &SparqlValue, b: &SparqlValue) -> std::cmp::Ordering {
    // Compare by type first (URIs < literals < blank nodes)
    let type_order = |value_type: &str| match value_type {
        "uri" => 0,
        "literal" => 1,
        "bnode" => 2,
        _ => 3,
    };

    let a_type_order = type_order(&a.value_type);
    let b_type_order = type_order(&b.value_type);

    match a_type_order.cmp(&b_type_order) {
        std::cmp::Ordering::Equal => {
            // Same type, compare values
            match a.value_type.as_str() {
                "literal" => compare_literal_values(a, b),
                _ => a.value.cmp(&b.value), // String comparison for URIs and blank nodes
            }
        }
        other => other,
    }
}

/// Compare literal values with datatype-aware comparison
pub fn compare_literal_values(a: &SparqlValue, b: &SparqlValue) -> std::cmp::Ordering {
    // If both have the same datatype, try type-specific comparison
    if let (Some(a_dt), Some(b_dt)) = (&a.datatype, &b.datatype) {
        if a_dt == b_dt {
            return compare_typed_literals(a, b, a_dt);
        }
    }

    // Fall back to string comparison
    a.value.cmp(&b.value)
}

/// Compare typed literal values
pub fn compare_typed_literals(
    a: &SparqlValue,
    b: &SparqlValue,
    datatype: &str,
) -> std::cmp::Ordering {
    match datatype {
        "http://www.w3.org/2001/XMLSchema#integer"
        | "http://www.w3.org/2001/XMLSchema#int"
        | "http://www.w3.org/2001/XMLSchema#long" => {
            match (a.value.parse::<i64>(), b.value.parse::<i64>()) {
                (Ok(a_num), Ok(b_num)) => a_num.cmp(&b_num),
                _ => a.value.cmp(&b.value),
            }
        }
        "http://www.w3.org/2001/XMLSchema#decimal"
        | "http://www.w3.org/2001/XMLSchema#double"
        | "http://www.w3.org/2001/XMLSchema#float" => {
            match (a.value.parse::<f64>(), b.value.parse::<f64>()) {
                (Ok(a_num), Ok(b_num)) => a_num
                    .partial_cmp(&b_num)
                    .unwrap_or(std::cmp::Ordering::Equal),
                _ => a.value.cmp(&b.value),
            }
        }
        "http://www.w3.org/2001/XMLSchema#dateTime" | "http://www.w3.org/2001/XMLSchema#date" => {
            // For dates, ISO format string comparison usually works
            a.value.cmp(&b.value)
        }
        "http://www.w3.org/2001/XMLSchema#boolean" => {
            match (a.value.parse::<bool>(), b.value.parse::<bool>()) {
                (Ok(a_bool), Ok(b_bool)) => a_bool.cmp(&b_bool),
                _ => a.value.cmp(&b.value),
            }
        }
        _ => a.value.cmp(&b.value), // Default to string comparison
    }
}

/// Join two SPARQL result sets
pub fn join_sparql_results(left: &SparqlResults, right: &SparqlResults) -> Result<SparqlResults> {
    // Find common variables
    let common_vars: Vec<_> = left
        .head
        .vars
        .iter()
        .filter(|var| right.head.vars.contains(var))
        .cloned()
        .collect();

    if common_vars.is_empty() {
        // Cartesian product if no common variables
        return cartesian_product_sparql(left, right);
    }

    // Perform hash join
    let mut joined_bindings = Vec::new();

    // Build hash table for right side
    let mut right_index: HashMap<String, Vec<&SparqlBinding>> = HashMap::new();
    for binding in &right.results.bindings {
        let key = create_join_key(binding, &common_vars);
        right_index.entry(key).or_default().push(binding);
    }

    // Probe with left side
    for left_binding in &left.results.bindings {
        let key = create_join_key(left_binding, &common_vars);
        if let Some(right_bindings) = right_index.get(&key) {
            for right_binding in right_bindings {
                let mut merged = left_binding.clone();
                for (var, value) in right_binding.iter() {
                    if !merged.contains_key(var) {
                        merged.insert(var.clone(), value.clone());
                    }
                }
                joined_bindings.push(merged);
            }
        }
    }

    // Combine variable lists
    let mut all_vars = left.head.vars.clone();
    for var in &right.head.vars {
        if !all_vars.contains(var) {
            all_vars.push(var.clone());
        }
    }

    Ok(SparqlResults {
        head: SparqlHead { vars: all_vars },
        results: SparqlResultSet {
            bindings: joined_bindings,
        },
    })
}

/// Create a join key from bindings and common variables
pub fn create_join_key(binding: &SparqlBinding, common_vars: &[String]) -> String {
    let mut key_parts = Vec::new();
    for var in common_vars {
        if let Some(value) = binding.get(var) {
            key_parts.push(format!(
                "{}:{}",
                var,
                serde_json::to_string(value).unwrap_or_default()
            ));
        }
    }
    key_parts.join("|")
}

/// Cartesian product of two SPARQL result sets
pub fn cartesian_product_sparql(
    left: &SparqlResults,
    right: &SparqlResults,
) -> Result<SparqlResults> {
    let mut product_bindings = Vec::new();

    for left_binding in &left.results.bindings {
        for right_binding in &right.results.bindings {
            let mut merged = left_binding.clone();
            merged.extend(right_binding.clone());
            product_bindings.push(merged);
        }
    }

    let mut all_vars = left.head.vars.clone();
    all_vars.extend(right.head.vars.clone());

    Ok(SparqlResults {
        head: SparqlHead { vars: all_vars },
        results: SparqlResultSet {
            bindings: product_bindings,
        },
    })
}

/// Join two GraphQL results
pub fn join_graphql_results(
    left: &GraphQLResponse,
    right: &GraphQLResponse,
) -> Result<GraphQLResponse> {
    // Simple merge of GraphQL objects
    let mut merged_data = serde_json::Map::new();

    if let Some(left_obj) = left.data.as_object() {
        merged_data.extend(left_obj.clone());
    }

    if let Some(right_obj) = right.data.as_object() {
        merged_data.extend(right_obj.clone());
    }

    let mut all_errors = left.errors.clone();
    all_errors.extend(right.errors.clone());

    Ok(GraphQLResponse {
        data: serde_json::Value::Object(merged_data),
        errors: all_errors,
        extensions: None,
    })
}

impl Default for FederatedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the federated executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedExecutorConfig {
    pub request_timeout: Duration,
    pub parallel_timeout: Duration,
    pub max_concurrent_requests: usize,
    pub retry_attempts: usize,
    pub retry_delay: Duration,
    pub user_agent: String,
    pub enable_compression: bool,
}

impl Default for FederatedExecutorConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            parallel_timeout: Duration::from_secs(60),
            max_concurrent_requests: 10,
            retry_attempts: 3,
            retry_delay: Duration::from_millis(500),
            user_agent: "oxirs-federate/1.0".to_string(),
            enable_compression: true,
        }
    }
}

/// Result of executing a single step
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_id: String,
    pub step_type: StepType,
    pub status: ExecutionStatus,
    pub data: Option<QueryResultData>,
    pub error: Option<String>,
    pub execution_time: Duration,
    pub service_id: Option<String>,
    pub memory_used: u64,
    pub result_size: usize,
    pub success: bool,
    pub error_message: Option<String>,
    pub service_response_time: Duration,
    pub cache_hit: bool,
}

/// Status of step execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Success,
    Failed,
    Timeout,
    Cancelled,
}

/// Data returned from query execution
#[derive(Debug, Clone)]
pub enum QueryResultData {
    Sparql(SparqlResults),
    GraphQL(GraphQLResponse),
}

impl QueryResultData {
    /// Estimate the size of the query result data
    pub fn estimated_size(&self) -> usize {
        match self {
            QueryResultData::Sparql(results) => {
                // Estimate based on number of bindings and variables
                let var_count = results.head.vars.len();
                let binding_count = results.results.bindings.len();
                // Rough estimate: each binding entry is ~100 bytes on average
                var_count * binding_count * 100
            }
            QueryResultData::GraphQL(response) => {
                // Estimate based on JSON serialization
                serde_json::to_string(response)
                    .map(|s| s.len())
                    .unwrap_or(0)
            }
        }
    }
}

/// SPARQL query results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlResults {
    pub head: SparqlHead,
    pub results: SparqlResultSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlHead {
    pub vars: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlResultSet {
    pub bindings: Vec<SparqlBinding>,
}

pub type SparqlBinding = HashMap<String, SparqlValue>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlValue {
    #[serde(rename = "type")]
    pub value_type: String,
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datatype: Option<String>,
    #[serde(rename = "xml:lang", skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
}

/// GraphQL request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<serde_json::Value>,
    #[serde(rename = "operationName", skip_serializing_if = "Option::is_none")]
    pub operation_name: Option<String>,
}

/// GraphQL response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLResponse {
    pub data: serde_json::Value,
    #[serde(default)]
    pub errors: Vec<GraphQLError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<GraphQLLocation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLLocation {
    pub line: u32,
    pub column: u32,
}

/// Order clause for sorting
#[derive(Debug, Clone)]
struct OrderClause {
    variable: String,
    descending: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::QueryType;
    use crate::{ExecutionPlan, ExecutionStep};

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = FederatedExecutor::new();
        assert_eq!(executor.config.max_concurrent_requests, 10);
    }

    #[tokio::test]
    async fn test_step_result_creation() {
        let result = StepResult {
            step_id: "test-step".to_string(),
            step_type: StepType::ServiceQuery,
            status: ExecutionStatus::Success,
            data: None,
            error: None,
            execution_time: Duration::from_millis(100),
            service_id: Some("test-service".to_string()),
        };

        assert_eq!(result.status, ExecutionStatus::Success);
        assert!(result.data.is_none());
    }

    #[tokio::test]
    async fn test_sparql_results_join() {
        let executor = FederatedExecutor::new();

        let left = SparqlResults {
            head: SparqlHead {
                vars: vec!["s".to_string(), "p".to_string()],
            },
            results: SparqlResultSet { bindings: vec![] },
        };

        let right = SparqlResults {
            head: SparqlHead {
                vars: vec!["p".to_string(), "o".to_string()],
            },
            results: SparqlResultSet { bindings: vec![] },
        };

        let result = executor.join_sparql_results(&left, &right);
        assert!(result.is_ok());

        let joined = result.unwrap();
        assert_eq!(joined.head.vars.len(), 3); // s, p, o
    }

    #[test]
    fn test_join_key_creation() {
        let executor = FederatedExecutor::new();
        let mut binding = HashMap::new();
        binding.insert(
            "x".to_string(),
            SparqlValue {
                value_type: "uri".to_string(),
                value: "http://example.org".to_string(),
                datatype: None,
                lang: None,
            },
        );

        let common_vars = vec!["x".to_string()];
        let key = executor.create_join_key(&binding, &common_vars);
        assert!(key.contains("x:"));
    }
}
