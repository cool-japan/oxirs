//! Adaptive execution algorithms for federated queries
//!
//! This module implements adaptive optimization techniques that adjust execution
//! strategies based on runtime performance and resource utilization.

use anyhow::Result;
use futures::{stream, StreamExt};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, instrument};

use crate::{planner::ExecutionPlan, ServiceRegistry};

use super::step_execution::execute_step;
use super::types::*;

impl FederatedExecutor {
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
    pub async fn execute_adaptive_group(
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

        // Monitor resource usage after execution
        let final_memory = resource_monitor.get_memory_usage();
        let final_cpu = resource_monitor.get_cpu_usage();
        let group_duration = group_start.elapsed();

        // Record performance metrics
        performance_monitor.record_memory_usage(final_memory);
        performance_monitor.record_cpu_usage(final_cpu);

        // Update resource monitor
        resource_monitor.update_memory_usage(final_memory);
        resource_monitor.update_cpu_usage(final_cpu);

        // Record execution time based on strategy
        match execution_strategy {
            AdaptiveExecutionStrategy::Parallel => {
                performance_monitor.record_parallel_execution(group_duration);
            }
            AdaptiveExecutionStrategy::Sequential => {
                performance_monitor.record_sequential_execution(group_duration);
            }
            _ => {
                performance_monitor.record_step_execution(group_duration);
            }
        }

        Ok(group_results)
    }

    /// Execute steps in parallel with adaptive monitoring
    async fn execute_parallel_adaptive(
        &self,
        parallel_group: &[String],
        plan: &ExecutionPlan,
        completed_steps: &HashMap<String, StepResult>,
        performance_monitor: &mut EnhancedPerformanceMonitor,
    ) -> Result<Vec<StepResult>> {
        let mut results = Vec::new();

        // Execute all steps concurrently
        let mut futures = Vec::new();
        for step_id in parallel_group {
            if let Some(step) = plan.steps.iter().find(|s| &s.step_id == step_id) {
                futures.push(async {
                    let start_time = Instant::now();
                    let result = execute_step(step, completed_steps).await;
                    let duration = start_time.elapsed();
                    (result, duration)
                });
            }
        }

        // Wait for all futures to complete
        let future_results = futures::future::join_all(futures).await;

        for (result, duration) in future_results {
            match result {
                Ok(step_result) => {
                    performance_monitor.record_step_execution(duration);
                    results.push(step_result);
                }
                Err(e) => {
                    performance_monitor.record_error(e.to_string());
                    return Err(e);
                }
            }
        }

        Ok(results)
    }

    /// Execute steps sequentially with adaptive monitoring
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
                let result = execute_step(step, completed_steps).await?;
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

    /// Select the optimal execution strategy based on current conditions
    fn select_execution_strategy(
        &self,
        parallel_group: &[String],
        _plan: &ExecutionPlan,
        performance_monitor: &EnhancedPerformanceMonitor,
        resource_monitor: &LocalResourceMonitor,
        adaptive_config: &LocalAdaptiveConfig,
    ) -> AdaptiveExecutionStrategy {
        let group_size = parallel_group.len();
        let memory_usage = resource_monitor.get_memory_usage();
        let cpu_usage = resource_monitor.get_cpu_usage();

        // Check memory pressure
        if memory_usage > adaptive_config.memory_threshold {
            if group_size >= adaptive_config.streaming_threshold {
                return AdaptiveExecutionStrategy::Streaming;
            } else {
                return AdaptiveExecutionStrategy::Sequential;
            }
        }

        // Check CPU pressure
        if cpu_usage > adaptive_config.cpu_threshold {
            return AdaptiveExecutionStrategy::Sequential;
        }

        // Check if parallel execution has been effective
        let avg_parallel_time = performance_monitor.get_average_parallel_time();
        let avg_sequential_time = performance_monitor.get_average_sequential_time();

        if avg_parallel_time.as_millis() as f64
            > avg_sequential_time.as_millis() as f64 * adaptive_config.performance_threshold
        {
            // Parallel execution is not efficient
            if group_size >= adaptive_config.hybrid_batch_size {
                return AdaptiveExecutionStrategy::Hybrid;
            } else {
                return AdaptiveExecutionStrategy::Sequential;
            }
        }

        // Default to parallel execution for groups above threshold
        if group_size >= adaptive_config.parallel_threshold {
            AdaptiveExecutionStrategy::Parallel
        } else {
            AdaptiveExecutionStrategy::Sequential
        }
    }

    /// Check if re-optimization should be triggered
    fn should_reoptimize(
        &self,
        runtime_stats: &RuntimeStatistics,
        adaptive_config: &LocalAdaptiveConfig,
        group_idx: usize,
    ) -> bool {
        // Check reoptimization interval
        if group_idx % adaptive_config.reoptimization_interval != 0 {
            return false;
        }

        // Check if we have enough data to make decisions
        if runtime_stats.groups_executed < 2 {
            return false;
        }

        // Check performance degradation
        let success_rate = runtime_stats.get_success_rate();
        if success_rate < (1.0 - adaptive_config.error_rate_threshold) {
            return true;
        }

        // Check if average execution time has increased significantly
        let current_avg = runtime_stats.average_group_time;
        let threshold_duration = Duration::from_millis(adaptive_config.latency_threshold as u64);

        if current_avg > threshold_duration {
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
        _registry: &ServiceRegistry,
    ) -> Result<Vec<String>> {
        // For now, return the original group
        // In a full implementation, this would analyze bottlenecks and reorder steps

        // Identify primary bottleneck
        if let Some(bottleneck) = performance_monitor.get_primary_bottleneck() {
            match bottleneck {
                BottleneckType::NetworkLatency => {
                    // Reorder to minimize network calls
                    return Ok(self.reorder_for_network_optimization(parallel_group, plan));
                }
                BottleneckType::MemoryUsage => {
                    // Prioritize memory-efficient operations
                    return Ok(self.reorder_for_memory_optimization(parallel_group, plan));
                }
                BottleneckType::CpuUsage => {
                    // Balance CPU-intensive operations
                    return Ok(self.reorder_for_cpu_optimization(parallel_group, plan));
                }
                BottleneckType::DiskIo => {
                    // Minimize disk I/O operations
                    return Ok(self.reorder_for_io_optimization(parallel_group, plan));
                }
            }
        }

        Ok(parallel_group.to_vec())
    }

    /// Reorder steps to optimize network usage
    fn reorder_for_network_optimization(
        &self,
        parallel_group: &[String],
        _plan: &ExecutionPlan,
    ) -> Vec<String> {
        // Simple implementation - in practice would analyze network patterns
        parallel_group.to_vec()
    }

    /// Reorder steps to optimize memory usage
    fn reorder_for_memory_optimization(
        &self,
        parallel_group: &[String],
        _plan: &ExecutionPlan,
    ) -> Vec<String> {
        // Simple implementation - in practice would prioritize memory-efficient steps
        parallel_group.to_vec()
    }

    /// Reorder steps to optimize CPU usage
    fn reorder_for_cpu_optimization(
        &self,
        parallel_group: &[String],
        _plan: &ExecutionPlan,
    ) -> Vec<String> {
        // Simple implementation - in practice would balance CPU-intensive operations
        parallel_group.to_vec()
    }

    /// Reorder steps to optimize I/O usage
    fn reorder_for_io_optimization(
        &self,
        parallel_group: &[String],
        _plan: &ExecutionPlan,
    ) -> Vec<String> {
        // Simple implementation - in practice would minimize I/O operations
        parallel_group.to_vec()
    }

    /// Adapt configuration based on runtime feedback
    pub fn adapt_configuration(
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
            adaptive_config.memory_threshold =
                (adaptive_config.memory_threshold as f64 * 1.1) as u64;
        } else if current_memory < (adaptive_config.memory_threshold as f64 * 0.7) as u64 {
            adaptive_config.memory_threshold =
                ((adaptive_config.memory_threshold as f64) * 0.9) as u64;
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

        if avg_parallel_time.as_millis() as f64 > avg_sequential_time.as_millis() as f64 * 1.2 {
            // Parallel execution is not efficient, increase threshold
            adaptive_config.parallel_threshold += 1;
        } else if (avg_parallel_time.as_millis() as f64)
            < avg_sequential_time.as_millis() as f64 * 0.8
        {
            // Parallel execution is very efficient, decrease threshold
            adaptive_config.parallel_threshold =
                adaptive_config.parallel_threshold.saturating_sub(1).max(2);
        }
    }

    /// Log adaptive execution insights
    pub fn log_adaptive_insights(
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
            "Groups executed: {}, Steps executed: {}",
            runtime_stats.groups_executed, runtime_stats.total_steps_executed
        );
        info!(
            "Success rate: {:.2}%",
            runtime_stats.get_success_rate() * 100.0
        );
        info!("Average group time: {:?}", runtime_stats.average_group_time);
        info!(
            "Peak memory usage: {} MB",
            runtime_stats.peak_memory_usage / 1024 / 1024
        );
        info!(
            "Peak CPU usage: {:.2}%",
            runtime_stats.peak_cpu_usage * 100.0
        );

        if let Some(bottleneck) = performance_monitor.get_primary_bottleneck() {
            info!("Primary bottleneck: {:?}", bottleneck);
        }

        info!(
            "Error rate: {:.2}%",
            performance_monitor.get_error_rate() * 100.0
        );
        info!(
            "Average step time: {:?}",
            performance_monitor.get_average_step_time()
        );
        info!("=== End Insights ===");
    }
}
