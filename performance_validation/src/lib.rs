//! Revolutionary Performance Validation Module for OxiRS
//!
//! Comprehensive benchmarking, validation, and performance analysis suite
//! for the OxiRS semantic web platform with SciRS2 integration.

pub mod revolutionary_benchmarking_suite;

/// GPU and network performance profiling primitives.
///
/// Only the standalone, self-contained profiler submodules are wired in
/// here. `cpu_profiler`/`memory_profiler` and the orchestrating `mod.rs` in
/// `src/profiling/` depend on a pre-`sysinfo`-0.30 API surface
/// (`CpuExt`/`SystemExt`, since removed) and reference mismatched field
/// names against each other; that breakage predates this change and is
/// out of scope here. Only `gpu_profiler` and `network_profiler` -- which
/// are self-contained -- are compiled and tested.
pub mod profiling {
    #[path = "gpu_profiler.rs"]
    pub mod gpu_profiler;
    #[path = "network_profiler.rs"]
    pub mod network_profiler;
}

pub use revolutionary_benchmarking_suite::{
    AIReasoningMetrics, BenchmarkResultsAnalyzer, DistributedCoordinationMetrics,
    MemoryEfficiencyMetrics, QuantumOptimizationMetrics, QueryPerformanceMetrics,
    RevolutionaryBenchmarkConfig, RevolutionaryBenchmarkMetrics, RevolutionaryBenchmarkingSuite,
    SystemHealthMetrics,
};

use scirs2_core::error::CoreResult;

/// Initialize performance validation subsystem
pub async fn initialize_performance_validation() -> CoreResult<()> {
    println!("Revolutionary Performance Validation System initialized");
    Ok(())
}

/// Create a default revolutionary benchmarking suite
pub async fn create_default_benchmark_suite() -> CoreResult<RevolutionaryBenchmarkingSuite> {
    let config = RevolutionaryBenchmarkConfig::default();
    RevolutionaryBenchmarkingSuite::new(config).await
}

/// Run quick performance validation check
pub async fn quick_validation_check() -> CoreResult<bool> {
    let mut suite = create_default_benchmark_suite().await?;
    let metrics = suite.run_comprehensive_benchmarks().await?;

    // Simple health check: overall performance should be above 70%
    Ok(metrics.overall_system_health.overall_performance_score > 0.7)
}
