//! Benchmark definitions and management
//!
//! This module contains the benchmark and benchmark suite definitions
//! for organizing and executing performance tests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use super::config::BenchmarkSuiteConfig;
use super::types::{
    BenchmarkResult, BenchmarkStatus, BenchmarkType, MeasurementConfig, SuccessCriteria,
    TargetComponent, WorkloadConfig,
};
use crate::{Result, ShaclAiError};

/// Type alias for benchmark function
pub type BenchmarkFunction = Arc<dyn Fn(&WorkloadConfig) -> Result<BenchmarkResult> + Send + Sync>;

/// Individual benchmark definition
#[derive(Clone)]
pub struct Benchmark {
    /// Unique benchmark identifier
    pub id: Uuid,
    /// Benchmark name
    pub name: String,
    /// Benchmark description
    pub description: String,
    /// Benchmark type
    pub benchmark_type: BenchmarkType,
    /// Target component
    pub target_component: TargetComponent,
    /// Workload configuration
    pub workload_config: WorkloadConfig,
    /// Measurement configuration
    pub measurement_config: MeasurementConfig,
    /// Success criteria
    pub success_criteria: SuccessCriteria,
    /// Benchmark function
    pub benchmark_fn: BenchmarkFunction,
    /// Benchmark metadata
    pub metadata: HashMap<String, String>,
    /// Enable this benchmark
    pub enabled: bool,
    /// Benchmark timeout
    pub timeout: Duration,
    /// Dependencies (other benchmark IDs that must run first)
    pub dependencies: Vec<Uuid>,
}

impl std::fmt::Debug for Benchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Benchmark")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("benchmark_type", &self.benchmark_type)
            .field("target_component", &self.target_component)
            .field("workload_config", &self.workload_config)
            .field("measurement_config", &self.measurement_config)
            .field("success_criteria", &self.success_criteria)
            .field("benchmark_fn", &"<function>")
            .field("metadata", &self.metadata)
            .field("enabled", &self.enabled)
            .field("timeout", &self.timeout)
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

impl Benchmark {
    /// Create a new benchmark
    pub fn new(
        name: String,
        description: String,
        benchmark_type: BenchmarkType,
        target_component: TargetComponent,
        benchmark_fn: BenchmarkFunction,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            description,
            benchmark_type,
            target_component,
            workload_config: WorkloadConfig::default(),
            measurement_config: MeasurementConfig::default(),
            success_criteria: SuccessCriteria::default(),
            benchmark_fn,
            metadata: HashMap::new(),
            enabled: true,
            timeout: Duration::from_secs(300),
            dependencies: Vec::new(),
        }
    }

    /// Execute the benchmark
    pub fn execute(&self) -> Result<BenchmarkResult> {
        (self.benchmark_fn)(&self.workload_config)
    }

    /// Add metadata to the benchmark
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set workload configuration
    pub fn with_workload_config(mut self, config: WorkloadConfig) -> Self {
        self.workload_config = config;
        self
    }

    /// Set measurement configuration
    pub fn with_measurement_config(mut self, config: MeasurementConfig) -> Self {
        self.measurement_config = config;
        self
    }

    /// Set success criteria
    pub fn with_success_criteria(mut self, criteria: SuccessCriteria) -> Self {
        self.success_criteria = criteria;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add dependency
    pub fn with_dependency(mut self, dependency_id: Uuid) -> Self {
        self.dependencies.push(dependency_id);
        self
    }

    /// Disable the benchmark
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Check if the benchmark is ready to run (all dependencies satisfied)
    pub fn is_ready_to_run(&self, completed_benchmarks: &[Uuid]) -> bool {
        self.enabled
            && self
                .dependencies
                .iter()
                .all(|dep_id| completed_benchmarks.contains(dep_id))
    }
}

/// Benchmark suite containing multiple related benchmarks
#[derive(Debug)]
pub struct BenchmarkSuite {
    /// Suite identifier
    pub id: Uuid,
    /// Suite configuration
    pub config: BenchmarkSuiteConfig,
    /// Benchmarks in this suite
    pub benchmarks: Vec<Benchmark>,
    /// Suite metadata
    pub metadata: HashMap<String, String>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkSuiteConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            benchmarks: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a benchmark to the suite
    pub fn add_benchmark(&mut self, benchmark: Benchmark) {
        self.benchmarks.push(benchmark);
    }

    /// Add multiple benchmarks to the suite
    pub fn add_benchmarks(&mut self, benchmarks: Vec<Benchmark>) {
        self.benchmarks.extend(benchmarks);
    }

    /// Get benchmark by ID
    pub fn get_benchmark(&self, benchmark_id: &Uuid) -> Option<&Benchmark> {
        self.benchmarks.iter().find(|b| &b.id == benchmark_id)
    }

    /// Get benchmarks by type
    pub fn get_benchmarks_by_type(&self, benchmark_type: &BenchmarkType) -> Vec<&Benchmark> {
        self.benchmarks
            .iter()
            .filter(|b| &b.benchmark_type == benchmark_type)
            .collect()
    }

    /// Get benchmarks by target component
    pub fn get_benchmarks_by_component(&self, component: &TargetComponent) -> Vec<&Benchmark> {
        self.benchmarks
            .iter()
            .filter(|b| &b.target_component == component)
            .collect()
    }

    /// Get enabled benchmarks
    pub fn get_enabled_benchmarks(&self) -> Vec<&Benchmark> {
        self.benchmarks.iter().filter(|b| b.enabled).collect()
    }

    /// Get benchmarks ready to run (dependencies satisfied)
    pub fn get_ready_benchmarks(&self, completed_benchmarks: &[Uuid]) -> Vec<&Benchmark> {
        self.benchmarks
            .iter()
            .filter(|b| b.is_ready_to_run(completed_benchmarks))
            .collect()
    }

    /// Validate the benchmark suite for consistency
    pub fn validate(&self) -> Result<()> {
        // Check for circular dependencies
        for benchmark in &self.benchmarks {
            if self.has_circular_dependency(
                &benchmark.id,
                &benchmark.dependencies,
                &mut Vec::new(),
            )? {
                return Err(ShaclAiError::Benchmark(format!(
                    "Circular dependency detected for benchmark: {}",
                    benchmark.name
                )));
            }
        }

        // Check that all dependencies exist
        let benchmark_ids: std::collections::HashSet<Uuid> =
            self.benchmarks.iter().map(|b| b.id).collect();

        for benchmark in &self.benchmarks {
            for dep_id in &benchmark.dependencies {
                if !benchmark_ids.contains(dep_id) {
                    return Err(ShaclAiError::Benchmark(format!(
                        "Dependency {} not found for benchmark: {}",
                        dep_id, benchmark.name
                    )));
                }
            }
        }

        Ok(())
    }

    /// Check for circular dependencies recursively
    fn has_circular_dependency(
        &self,
        current_id: &Uuid,
        dependencies: &[Uuid],
        visited: &mut Vec<Uuid>,
    ) -> Result<bool> {
        if visited.contains(current_id) {
            return Ok(true);
        }

        visited.push(*current_id);

        for dep_id in dependencies {
            if let Some(dep_benchmark) = self.get_benchmark(dep_id) {
                if self.has_circular_dependency(dep_id, &dep_benchmark.dependencies, visited)? {
                    return Ok(true);
                }
            }
        }

        visited.pop();
        Ok(false)
    }

    /// Get execution order respecting dependencies
    pub fn get_execution_order(&self) -> Result<Vec<&Benchmark>> {
        let mut order = Vec::new();
        let mut completed = Vec::new();
        let enabled_benchmarks: Vec<&Benchmark> = self.get_enabled_benchmarks();

        while order.len() < enabled_benchmarks.len() {
            let ready_benchmarks: Vec<&Benchmark> = enabled_benchmarks
                .iter()
                .filter(|b| !order.iter().any(|ordered: &&Benchmark| ordered.id == b.id))
                .filter(|b| b.is_ready_to_run(&completed))
                .cloned()
                .collect();

            if ready_benchmarks.is_empty() {
                return Err(ShaclAiError::Benchmark(
                    "Cannot resolve benchmark dependencies - possible circular dependency"
                        .to_string(),
                ));
            }

            for benchmark in ready_benchmarks {
                order.push(benchmark);
                completed.push(benchmark.id);
            }
        }

        Ok(order)
    }

    /// Add metadata to the suite
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get suite statistics
    pub fn get_statistics(&self) -> BenchmarkSuiteStatistics {
        let total_benchmarks = self.benchmarks.len();
        let enabled_benchmarks = self.get_enabled_benchmarks().len();
        let benchmark_types: std::collections::HashSet<BenchmarkType> = self
            .benchmarks
            .iter()
            .map(|b| b.benchmark_type.clone())
            .collect();
        let target_components: std::collections::HashSet<TargetComponent> = self
            .benchmarks
            .iter()
            .map(|b| b.target_component.clone())
            .collect();

        BenchmarkSuiteStatistics {
            total_benchmarks,
            enabled_benchmarks,
            benchmark_types: benchmark_types.len(),
            target_components: target_components.len(),
            has_dependencies: self.benchmarks.iter().any(|b| !b.dependencies.is_empty()),
            estimated_duration: self.estimate_total_duration(),
        }
    }

    /// Estimate total execution duration
    fn estimate_total_duration(&self) -> Duration {
        if self.config.enable_parallel_execution {
            // Estimate based on critical path
            self.estimate_critical_path_duration()
        } else {
            // Sum all benchmark timeouts
            self.benchmarks
                .iter()
                .filter(|b| b.enabled)
                .map(|b| b.timeout)
                .sum()
        }
    }

    /// Estimate critical path duration for parallel execution
    fn estimate_critical_path_duration(&self) -> Duration {
        // Simplified critical path calculation
        // In reality, this would be more sophisticated
        let max_timeout = self
            .benchmarks
            .iter()
            .filter(|b| b.enabled)
            .map(|b| b.timeout)
            .max()
            .unwrap_or(Duration::from_secs(0));

        max_timeout * 2 // Conservative estimate
    }
}

/// Benchmark suite statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteStatistics {
    pub total_benchmarks: usize,
    pub enabled_benchmarks: usize,
    pub benchmark_types: usize,
    pub target_components: usize,
    pub has_dependencies: bool,
    pub estimated_duration: Duration,
}

/// Benchmark builder for fluent API
pub struct BenchmarkBuilder {
    benchmark: Benchmark,
}

impl BenchmarkBuilder {
    /// Start building a new benchmark
    pub fn new(
        name: String,
        benchmark_type: BenchmarkType,
        target_component: TargetComponent,
        benchmark_fn: BenchmarkFunction,
    ) -> Self {
        Self {
            benchmark: Benchmark::new(
                name.clone(),
                format!("{} benchmark", name),
                benchmark_type,
                target_component,
                benchmark_fn,
            ),
        }
    }

    /// Set description
    pub fn description(mut self, description: String) -> Self {
        self.benchmark.description = description;
        self
    }

    /// Set workload configuration
    pub fn workload_config(mut self, config: WorkloadConfig) -> Self {
        self.benchmark.workload_config = config;
        self
    }

    /// Set measurement configuration
    pub fn measurement_config(mut self, config: MeasurementConfig) -> Self {
        self.benchmark.measurement_config = config;
        self
    }

    /// Set success criteria
    pub fn success_criteria(mut self, criteria: SuccessCriteria) -> Self {
        self.benchmark.success_criteria = criteria;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.benchmark.timeout = timeout;
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.benchmark.metadata.insert(key, value);
        self
    }

    /// Add dependency
    pub fn dependency(mut self, dependency_id: Uuid) -> Self {
        self.benchmark.dependencies.push(dependency_id);
        self
    }

    /// Disable the benchmark
    pub fn disabled(mut self) -> Self {
        self.benchmark.enabled = false;
        self
    }

    /// Build the benchmark
    pub fn build(self) -> Benchmark {
        self.benchmark
    }
}
