//! Core federated executor implementation
//!
//! This module contains the main FederatedExecutor implementation with basic
//! execution methods and constructor functions.

use anyhow::Result;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, instrument};

use crate::{
    cache::FederationCache,
    planner::{ExecutionPlan, ExecutionStep},
    service_executor::{JoinExecutor, ServiceExecutor},
};

use super::step_execution::{execute_parallel_group, execute_step};
use super::types::*;

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

    /// Get current process memory usage (resident set size, in bytes).
    ///
    /// Backed by a real [`sysinfo`] sample of this process, not a
    /// process-id-derived placeholder (`std::process::id() * 1024` used to
    /// be returned here, which is nonsense unrelated to actual memory use
    /// and made every `memory_used` delta in
    /// [`Self::execute_step_with_monitoring`] meaningless). Only this one
    /// process's stats are refreshed per call (not a full-system refresh),
    /// keeping the before/after sampling in that method cheap.
    pub fn get_current_memory_usage(&self) -> u64 {
        let pid = sysinfo::Pid::from_u32(std::process::id());
        let mut system = sysinfo::System::new();
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        system.process(pid).map(|p| p.memory()).unwrap_or(0)
    }

    /// Execute a single step.
    ///
    /// Delegates to the real [`execute_step`] implementation (the same one
    /// the non-adaptive [`Self::execute_plan`] path uses) so this is a
    /// genuine network/query call, not a fabricated always-`Success`,
    /// `data: None` result. This method is reachable via
    /// [`Self::execute_step_with_monitoring`] from the adaptive executor's
    /// `Streaming` strategy, so a caller relying on that path previously got
    /// silently fake results.
    pub async fn execute_single_step(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        execute_step(step, completed_steps).await
    }

    /// Execute individual step with performance monitoring
    pub async fn execute_step_with_monitoring(
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
            memory_used: memory_delta as usize,
            result_size: result.result_size,
            success: result.success,
            error_message: result.error_message,
            service_response_time: result.service_response_time,
            cache_hit: result.cache_hit,
        })
    }
}

impl Default for FederatedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::StepType;
    use std::time::Duration as StdDuration;

    /// Regression test for executor/core.rs:110 — `execute_single_step` used
    /// to be a fake step executor: it never inspected `step` at all and
    /// always returned `Ok(StepResult { status: Success, success: true,
    /// data: None, .. })` unconditionally, reachable via the public
    /// adaptive-execution `Streaming` strategy. It must now delegate to the
    /// real `step_execution::execute_step`, which genuinely attempts the
    /// query. `execute_step` reports per-step failures inside `StepResult`
    /// (status/success/error fields) rather than via the outer `Result`, so
    /// the observable proof that this is real execution (not the old stub)
    /// is that an unrunnable step (no `service_id`) comes back marked
    /// failed, with a real error message, instead of a fabricated success.
    #[tokio::test]
    async fn test_execute_single_step_delegates_to_real_execution() {
        let executor = FederatedExecutor::new();

        let step = ExecutionStep {
            step_id: "step-1".to_string(),
            step_type: StepType::ServiceQuery,
            service_id: None, // Deliberately unrunnable.
            service_url: None,
            auth_config: None,
            query_fragment: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            dependencies: Vec::new(),
            estimated_cost: 1.0,
            timeout: StdDuration::from_secs(5),
            retry_config: None,
        };
        let completed_steps = HashMap::new();

        let result = executor
            .execute_single_step(&step, &completed_steps)
            .await
            .expect("execute_step wraps failures in StepResult, not the outer Result");

        assert!(
            !result.success,
            "a ServiceQuery step with no service_id must genuinely fail, proving \
             execute_single_step is no longer a fake always-Success stub"
        );
        assert_eq!(result.status, ExecutionStatus::Failed);
        assert!(
            result.error.is_some(),
            "a real failure must carry a real error message, not a fabricated `None`"
        );
        assert!(
            result.data.is_none(),
            "a failed step must not carry fabricated result data"
        );
    }

    /// Regression test for executor/core.rs:106 — `get_current_memory_usage`
    /// used to return `std::process::id() * 1024`, a number with no
    /// relationship to actual memory use (e.g. PID 5 would "use" 5120
    /// bytes). It must now report this test process's real resident set
    /// size, which is always well above a single-digit-PID's fake reading
    /// for any running Rust test binary.
    #[test]
    fn test_get_current_memory_usage_reports_real_process_memory() {
        let executor = FederatedExecutor::new();
        let memory = executor.get_current_memory_usage();

        assert!(
            memory > 1_000_000,
            "a running test binary's RSS should be well over 1MB; got {memory} bytes, which \
             looks like the old process-id-derived placeholder rather than real memory"
        );
    }
}
