//! Production Tuning Configurations for Query Optimizer
//!
//! This module provides pre-configured optimizer settings optimized for different
//! production workloads and deployment scenarios.

use super::config::OptimizerConfig;
use crate::statistics::EstimationMethod;

/// Production tuning preset for different workload types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadProfile {
    /// High-throughput OLTP-style queries (simple patterns, high QPS)
    HighThroughput,
    /// Complex analytical queries (OLAP-style, low QPS)
    AnalyticalQueries,
    /// Mixed workload (balanced OLTP/OLAP)
    Mixed,
    /// Memory-constrained environments
    LowMemory,
    /// CPU-constrained environments
    LowCpu,
    /// Maximum performance (high resources)
    MaxPerformance,
}

/// Production optimizer configuration with workload-specific tuning
#[derive(Debug, Clone)]
pub struct ProductionOptimizerConfig {
    /// Base optimizer configuration
    pub base_config: OptimizerConfig,
    /// Cardinality estimation method
    pub estimation_method: EstimationMethod,
    /// Maximum plan cache size
    pub max_plan_cache_size: usize,
    /// Enable adaptive learning from execution
    pub adaptive_learning: bool,
    /// ML model training threshold (minimum samples before training)
    pub ml_training_threshold: usize,
    /// Statistics update frequency (queries between updates)
    pub stats_update_frequency: usize,
    /// Enable query result caching
    pub enable_result_cache: bool,
    /// Result cache TTL in seconds
    pub result_cache_ttl_secs: u64,
    /// Workload profile
    pub workload_profile: WorkloadProfile,
}

impl ProductionOptimizerConfig {
    /// Create configuration for high-throughput workloads
    ///
    /// Optimized for:
    /// - Simple queries (2-5 triple patterns)
    /// - High query rate (>1000 QPS)
    /// - Low latency requirements (<10ms p95)
    ///
    /// Characteristics:
    /// - **Adaptive optimization**: Uses fast path for simple queries (≤5 patterns)
    /// - **Minimal overhead**: ~3.0 µs for simple queries (simple heuristics)
    /// - **Cost-based for complex**: Histogram estimation for queries >5 patterns
    /// - **Aggressive caching**: 10,000 plans for repeated queries
    /// - **Early convergence**: Exits when optimal plan found (1-2 passes typical)
    ///
    /// Performance Improvements:
    /// - Simple queries (≤5 patterns): 3.0 µs (adaptive) vs 10.8 µs (cost-based) = **3.6x faster**
    /// - Complex queries (>5 patterns): Full cost-based optimization with histogram estimation
    /// - Avoids "optimization overhead paradox" where optimization time exceeds execution time
    ///
    /// Implementation Note:
    /// The optimizer automatically detects query complexity and selects the optimal
    /// strategy. For simple queries (≤5 patterns), it uses simple heuristics (max 2 passes).
    /// For complex queries (>5 patterns), it enables full cost-based optimization.
    pub fn high_throughput() -> Self {
        Self {
            base_config: OptimizerConfig {
                join_reordering: true,
                filter_pushdown: true,
                projection_pushdown: true,
                constant_folding: true,
                dead_code_elimination: false, // Skip for speed
                cost_based: true,             // Faster due to early convergence
                max_passes: 5,                // Early convergence typically exits at 1-2
            },
            estimation_method: EstimationMethod::Histogram, // Fast and accurate
            max_plan_cache_size: 10000,                     // Large cache for repeated queries
            adaptive_learning: false,                       // Disabled for minimal overhead
            ml_training_threshold: 0,                       // Not used
            stats_update_frequency: 10000,
            enable_result_cache: true,  // Cache results aggressively
            result_cache_ttl_secs: 300, // 5 minutes
            workload_profile: WorkloadProfile::HighThroughput,
        }
    }

    /// Create configuration for analytical queries
    ///
    /// Optimized for:
    /// - Complex queries (10-100 triple patterns)
    /// - Low query rate (<10 QPS)
    /// - Large result sets (>10K rows)
    ///
    /// Characteristics:
    /// - Comprehensive optimization
    /// - ML-based cardinality estimation
    /// - Adaptive learning enabled
    /// - Cost-based join ordering
    pub fn analytical_queries() -> Self {
        Self {
            base_config: OptimizerConfig {
                join_reordering: true,
                filter_pushdown: true,
                projection_pushdown: true,
                constant_folding: true,
                dead_code_elimination: true,
                cost_based: true, // Full cost-based optimization
                max_passes: 20,   // Aggressive optimization
            },
            estimation_method: EstimationMethod::MachineLearning,
            max_plan_cache_size: 1000, // Smaller cache (complex queries less repeated)
            adaptive_learning: true,   // Learn from execution
            ml_training_threshold: 100, // Start training after 100 samples
            stats_update_frequency: 100,
            enable_result_cache: true,
            result_cache_ttl_secs: 3600, // 1 hour (stable data)
            workload_profile: WorkloadProfile::AnalyticalQueries,
        }
    }

    /// Create configuration for mixed workloads
    ///
    /// Balanced configuration for both simple and complex queries
    pub fn mixed() -> Self {
        Self {
            base_config: OptimizerConfig {
                join_reordering: true,
                filter_pushdown: true,
                projection_pushdown: true,
                constant_folding: true,
                dead_code_elimination: true,
                cost_based: true,
                max_passes: 10, // Moderate optimization
            },
            estimation_method: EstimationMethod::Histogram, // Good balance
            max_plan_cache_size: 5000,
            adaptive_learning: true,
            ml_training_threshold: 500,
            stats_update_frequency: 1000,
            enable_result_cache: true,
            result_cache_ttl_secs: 600, // 10 minutes
            workload_profile: WorkloadProfile::Mixed,
        }
    }

    /// Create configuration for low-memory environments
    ///
    /// Optimized for:
    /// - Limited RAM (<2GB available)
    /// - Containerized deployments
    /// - Edge computing
    ///
    /// Characteristics:
    /// - Small caches (100 plans)
    /// - Sketch-based estimation (16KB per predicate)
    /// - Minimal intermediate results
    pub fn low_memory() -> Self {
        Self {
            base_config: OptimizerConfig {
                join_reordering: true,
                filter_pushdown: true,     // Critical for reducing memory
                projection_pushdown: true, // Critical for reducing memory
                constant_folding: true,
                dead_code_elimination: true,
                cost_based: true,
                max_passes: 5,
            },
            estimation_method: EstimationMethod::Sketch, // HyperLogLog: 16KB memory
            max_plan_cache_size: 100,                    // Minimal cache
            adaptive_learning: false,                    // Avoid ML memory overhead
            ml_training_threshold: 0,
            stats_update_frequency: 5000,
            enable_result_cache: false, // Disabled to save memory
            result_cache_ttl_secs: 0,
            workload_profile: WorkloadProfile::LowMemory,
        }
    }

    /// Create configuration for CPU-constrained environments
    ///
    /// Minimizes CPU usage while maintaining reasonable performance
    pub fn low_cpu() -> Self {
        Self {
            base_config: OptimizerConfig {
                join_reordering: true,
                filter_pushdown: true,
                projection_pushdown: true,
                constant_folding: false, // Skip expensive analysis
                dead_code_elimination: false,
                cost_based: false, // Use simple heuristics
                max_passes: 2,     // Minimal passes
            },
            estimation_method: EstimationMethod::Simple,
            max_plan_cache_size: 1000,
            adaptive_learning: false,
            ml_training_threshold: 0,
            stats_update_frequency: 10000,
            enable_result_cache: true, // Cache to avoid re-computation
            result_cache_ttl_secs: 600,
            workload_profile: WorkloadProfile::LowCpu,
        }
    }

    /// Create configuration for maximum performance
    ///
    /// Optimized for:
    /// - Dedicated servers
    /// - High resources (16+ cores, 32GB+ RAM)
    /// - Mission-critical queries
    ///
    /// Characteristics:
    /// - All optimizations enabled
    /// - Large caches
    /// - ML-based estimation with continuous learning
    /// - Aggressive result caching
    pub fn max_performance() -> Self {
        Self {
            base_config: OptimizerConfig {
                join_reordering: true,
                filter_pushdown: true,
                projection_pushdown: true,
                constant_folding: true,
                dead_code_elimination: true,
                cost_based: true,
                max_passes: 30, // Very aggressive optimization
            },
            estimation_method: EstimationMethod::MachineLearning,
            max_plan_cache_size: 50000, // Very large cache
            adaptive_learning: true,
            ml_training_threshold: 50, // Train early and often
            stats_update_frequency: 50,
            enable_result_cache: true,
            result_cache_ttl_secs: 7200, // 2 hours
            workload_profile: WorkloadProfile::MaxPerformance,
        }
    }

    /// Get recommended configuration based on workload profile
    pub fn for_workload(profile: WorkloadProfile) -> Self {
        match profile {
            WorkloadProfile::HighThroughput => Self::high_throughput(),
            WorkloadProfile::AnalyticalQueries => Self::analytical_queries(),
            WorkloadProfile::Mixed => Self::mixed(),
            WorkloadProfile::LowMemory => Self::low_memory(),
            WorkloadProfile::LowCpu => Self::low_cpu(),
            WorkloadProfile::MaxPerformance => Self::max_performance(),
        }
    }

    /// Estimate resource requirements for this configuration
    pub fn estimate_resource_requirements(&self) -> ResourceRequirements {
        let memory_mb = match self.workload_profile {
            WorkloadProfile::LowMemory => 100,          // 100 MB
            WorkloadProfile::LowCpu => 200,             // 200 MB
            WorkloadProfile::HighThroughput => 500,     // 500 MB
            WorkloadProfile::Mixed => 800,              // 800 MB
            WorkloadProfile::AnalyticalQueries => 1500, // 1.5 GB
            WorkloadProfile::MaxPerformance => 4000,    // 4 GB
        };

        let cpu_cores = match self.workload_profile {
            WorkloadProfile::LowCpu => 1,
            WorkloadProfile::LowMemory => 2,
            WorkloadProfile::HighThroughput => 4,
            WorkloadProfile::Mixed => 4,
            WorkloadProfile::AnalyticalQueries => 8,
            WorkloadProfile::MaxPerformance => 16,
        };

        let cache_memory_mb = (self.max_plan_cache_size as f64 * 0.001) as usize; // ~1KB per plan

        ResourceRequirements {
            memory_mb,
            cpu_cores,
            cache_memory_mb,
            max_concurrent_queries: self.estimate_max_concurrent_queries(),
        }
    }

    /// Estimate maximum concurrent queries this config can handle
    fn estimate_max_concurrent_queries(&self) -> usize {
        match self.workload_profile {
            WorkloadProfile::LowMemory => 10,
            WorkloadProfile::LowCpu => 5,
            WorkloadProfile::HighThroughput => 1000,
            WorkloadProfile::Mixed => 100,
            WorkloadProfile::AnalyticalQueries => 20,
            WorkloadProfile::MaxPerformance => 500,
        }
    }

    /// Validate configuration and return warnings
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check for incompatible settings
        if !self.adaptive_learning
            && matches!(self.estimation_method, EstimationMethod::MachineLearning)
        {
            warnings.push(
                "ML estimation enabled but adaptive_learning disabled - model won't improve"
                    .to_string(),
            );
        }

        if self.base_config.max_passes < 2 && self.base_config.cost_based {
            warnings.push(
                "Cost-based optimization needs at least 3 passes for effectiveness".to_string(),
            );
        }

        if self.max_plan_cache_size < 100
            && matches!(self.workload_profile, WorkloadProfile::HighThroughput)
        {
            warnings.push(
                "High-throughput workload needs larger plan cache (recommended: 5000+)".to_string(),
            );
        }

        if !self.enable_result_cache
            && matches!(self.workload_profile, WorkloadProfile::HighThroughput)
        {
            warnings
                .push("High-throughput workload benefits greatly from result caching".to_string());
        }

        warnings
    }
}

impl Default for ProductionOptimizerConfig {
    fn default() -> Self {
        Self::mixed() // Balanced default
    }
}

/// Resource requirements estimation
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Estimated memory usage in MB
    pub memory_mb: usize,
    /// Recommended CPU cores
    pub cpu_cores: usize,
    /// Cache memory in MB
    pub cache_memory_mb: usize,
    /// Maximum concurrent queries
    pub max_concurrent_queries: usize,
}

impl ResourceRequirements {
    /// Format as human-readable string
    pub fn summary(&self) -> String {
        format!(
            "Memory: {}MB, CPU: {} cores, Cache: {}MB, Max Concurrent: {}",
            self.memory_mb, self.cpu_cores, self.cache_memory_mb, self.max_concurrent_queries
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_throughput_config() {
        let config = ProductionOptimizerConfig::high_throughput();

        // Updated configuration based on benchmark findings:
        // Cost-based with early convergence is faster than fixed-pass heuristics
        assert_eq!(config.base_config.max_passes, 5);
        assert!(config.base_config.cost_based); // Enabled for early convergence
        assert!(matches!(
            config.estimation_method,
            EstimationMethod::Histogram
        )); // Fast and accurate
        assert_eq!(config.max_plan_cache_size, 10000);
        assert!(!config.adaptive_learning);

        let warnings = config.validate();
        assert!(warnings.is_empty(), "Config should be valid");
    }

    #[test]
    fn test_analytical_config() {
        let config = ProductionOptimizerConfig::analytical_queries();

        assert_eq!(config.base_config.max_passes, 20);
        assert!(config.base_config.cost_based);
        assert!(config.adaptive_learning);
        assert!(matches!(
            config.estimation_method,
            EstimationMethod::MachineLearning
        ));

        let warnings = config.validate();
        assert!(warnings.is_empty(), "Config should be valid");
    }

    #[test]
    fn test_low_memory_config() {
        let config = ProductionOptimizerConfig::low_memory();

        assert_eq!(config.max_plan_cache_size, 100);
        assert!(!config.enable_result_cache);
        assert!(matches!(config.estimation_method, EstimationMethod::Sketch));

        let resources = config.estimate_resource_requirements();
        assert_eq!(resources.memory_mb, 100);
    }

    #[test]
    fn test_max_performance_config() {
        let config = ProductionOptimizerConfig::max_performance();

        assert_eq!(config.base_config.max_passes, 30);
        assert_eq!(config.max_plan_cache_size, 50000);
        assert!(config.adaptive_learning);

        let resources = config.estimate_resource_requirements();
        assert_eq!(resources.memory_mb, 4000);
        assert_eq!(resources.cpu_cores, 16);
    }

    #[test]
    fn test_config_validation() {
        // Create invalid config
        let mut config = ProductionOptimizerConfig::high_throughput();
        config.estimation_method = EstimationMethod::MachineLearning;
        config.adaptive_learning = false;

        let warnings = config.validate();
        assert!(!warnings.is_empty(), "Should have warnings");
        assert!(warnings[0].contains("adaptive_learning disabled"));
    }

    #[test]
    fn test_workload_profile_selection() {
        let profiles = vec![
            WorkloadProfile::HighThroughput,
            WorkloadProfile::AnalyticalQueries,
            WorkloadProfile::Mixed,
            WorkloadProfile::LowMemory,
            WorkloadProfile::LowCpu,
            WorkloadProfile::MaxPerformance,
        ];

        for profile in profiles {
            let config = ProductionOptimizerConfig::for_workload(profile);
            assert_eq!(config.workload_profile, profile);

            let warnings = config.validate();
            // Each profile should be self-consistent
            assert!(
                warnings.is_empty(),
                "Profile {:?} should be valid, got warnings: {:?}",
                profile,
                warnings
            );
        }
    }

    #[test]
    fn test_resource_requirements() {
        let config = ProductionOptimizerConfig::max_performance();
        let resources = config.estimate_resource_requirements();

        assert!(resources.memory_mb >= 1000);
        assert!(resources.cpu_cores >= 4);
        assert!(resources.max_concurrent_queries > 100);

        let summary = resources.summary();
        assert!(summary.contains("Memory:"));
        assert!(summary.contains("CPU:"));
    }

    #[test]
    fn test_mixed_workload_balance() {
        let config = ProductionOptimizerConfig::mixed();

        // Should balance between throughput and analytical
        assert!(config.base_config.max_passes > 3); // More than high-throughput
        assert!(config.base_config.max_passes < 20); // Less than analytical
        assert!(config.base_config.cost_based); // Enable cost-based
        assert!(matches!(
            config.estimation_method,
            EstimationMethod::Histogram
        ));
    }

    #[test]
    fn test_resource_scaling() {
        // Verify resource requirements scale appropriately
        let low_mem = ProductionOptimizerConfig::low_memory();
        let max_perf = ProductionOptimizerConfig::max_performance();

        let low_mem_req = low_mem.estimate_resource_requirements();
        let max_perf_req = max_perf.estimate_resource_requirements();

        assert!(low_mem_req.memory_mb < max_perf_req.memory_mb);
        assert!(low_mem_req.cpu_cores < max_perf_req.cpu_cores);
        assert!(low_mem_req.max_concurrent_queries < max_perf_req.max_concurrent_queries);
    }
}
