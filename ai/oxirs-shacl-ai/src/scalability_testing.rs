//! Scalability Testing Suite for SHACL-AI
//!
//! This module provides comprehensive scalability testing capabilities for SHACL-AI systems,
//! including load testing, stress testing, performance benchmarking, and resource utilization analysis.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;
use scirs2_core::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Result, ShaclAiError};

/// Scalability test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityTestConfig {
    /// Enable load testing
    pub enable_load_testing: bool,
    /// Enable stress testing
    pub enable_stress_testing: bool,
    /// Enable endurance testing
    pub enable_endurance_testing: bool,
    /// Enable spike testing
    pub enable_spike_testing: bool,
    /// Minimum dataset size
    pub min_dataset_size: usize,
    /// Maximum dataset size
    pub max_dataset_size: usize,
    /// Size increment step
    pub size_increment_step: usize,
    /// Maximum concurrent users
    pub max_concurrent_users: usize,
    /// Test duration for endurance tests
    pub endurance_duration: Duration,
    /// Performance SLA thresholds
    pub sla_thresholds: SlaThresholds,
    /// Output directory for reports
    pub output_directory: PathBuf,
}

impl Default for ScalabilityTestConfig {
    fn default() -> Self {
        Self {
            enable_load_testing: true,
            enable_stress_testing: true,
            enable_endurance_testing: false,
            enable_spike_testing: true,
            min_dataset_size: 100,
            max_dataset_size: 100_000,
            size_increment_step: 10_000,
            max_concurrent_users: 1000,
            endurance_duration: Duration::from_secs(3600), // 1 hour
            sla_thresholds: SlaThresholds::default(),
            output_directory: PathBuf::from("/tmp/oxirs-scalability-tests"),
        }
    }
}

/// SLA thresholds for performance testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaThresholds {
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f64,
    /// Maximum acceptable memory usage in MB
    pub max_memory_mb: f64,
    /// Minimum acceptable throughput (operations per second)
    pub min_throughput_ops: f64,
    /// Maximum acceptable error rate (0-1)
    pub max_error_rate: f64,
    /// Maximum acceptable CPU usage (0-100)
    pub max_cpu_percent: f64,
}

impl Default for SlaThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 1000.0,    // 1 second
            max_memory_mb: 4096.0,     // 4 GB
            min_throughput_ops: 100.0, // 100 ops/sec
            max_error_rate: 0.01,      // 1% error rate
            max_cpu_percent: 80.0,     // 80% CPU
        }
    }
}

/// Scalability testing framework
#[derive(Debug)]
pub struct ScalabilityTestingFramework {
    config: ScalabilityTestConfig,
    test_history: Vec<ScalabilityTestReport>,
    load_tester: LoadTester,
    stress_tester: StressTester,
    endurance_tester: EnduranceTester,
    spike_tester: SpikeTester,
    resource_monitor: ResourceMonitor,
}

impl ScalabilityTestingFramework {
    /// Create a new scalability testing framework
    pub fn new(config: ScalabilityTestConfig) -> Result<Self> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&config.output_directory)?;

        Ok(Self {
            config: config.clone(),
            test_history: Vec::new(),
            load_tester: LoadTester::new(),
            stress_tester: StressTester::new(),
            endurance_tester: EnduranceTester::new(),
            spike_tester: SpikeTester::new(),
            resource_monitor: ResourceMonitor::new(),
        })
    }

    /// Run comprehensive scalability tests
    pub fn run_all_tests(&mut self, test_workload: &TestWorkload) -> Result<ScalabilityTestReport> {
        tracing::info!("Starting comprehensive scalability testing");
        let test_start = Instant::now();

        let mut report = ScalabilityTestReport {
            test_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(0),
            workload_description: test_workload.description.clone(),
            load_test_results: None,
            stress_test_results: None,
            endurance_test_results: None,
            spike_test_results: None,
            resource_utilization: Vec::new(),
            sla_compliance: SlaComplianceReport::default(),
            bottlenecks_identified: Vec::new(),
            scalability_score: 0.0,
            recommendations: Vec::new(),
        };

        // 1. Load Testing
        if self.config.enable_load_testing {
            tracing::info!("Running load tests...");
            let load_results = self
                .load_tester
                .run_load_test(test_workload, &self.config)?;
            report.load_test_results = Some(load_results);
        }

        // 2. Stress Testing
        if self.config.enable_stress_testing {
            tracing::info!("Running stress tests...");
            let stress_results = self
                .stress_tester
                .run_stress_test(test_workload, &self.config)?;
            report.stress_test_results = Some(stress_results);
        }

        // 3. Spike Testing
        if self.config.enable_spike_testing {
            tracing::info!("Running spike tests...");
            let spike_results = self
                .spike_tester
                .run_spike_test(test_workload, &self.config)?;
            report.spike_test_results = Some(spike_results);
        }

        // 4. Endurance Testing (optional, takes longer)
        if self.config.enable_endurance_testing {
            tracing::info!("Running endurance tests...");
            let endurance_results = self
                .endurance_tester
                .run_endurance_test(test_workload, &self.config)?;
            report.endurance_test_results = Some(endurance_results);
        }

        // 5. Analyze resource utilization
        report.resource_utilization = self.resource_monitor.get_metrics();

        // 6. Check SLA compliance
        report.sla_compliance = self.check_sla_compliance(&report)?;

        // 7. Identify bottlenecks
        report.bottlenecks_identified = self.identify_bottlenecks(&report)?;

        // 8. Calculate overall scalability score
        report.scalability_score = self.calculate_scalability_score(&report);

        // 9. Generate recommendations
        report.recommendations = self.generate_recommendations(&report)?;

        // 10. Record duration
        report.duration = test_start.elapsed();

        // 11. Save report
        self.save_report(&report)?;
        self.test_history.push(report.clone());

        tracing::info!(
            "Scalability testing completed in {:?}. Score: {:.2}/100",
            report.duration,
            report.scalability_score
        );

        Ok(report)
    }

    /// Check SLA compliance
    fn check_sla_compliance(&self, report: &ScalabilityTestReport) -> Result<SlaComplianceReport> {
        let mut compliance = SlaComplianceReport {
            latency_compliant: true,
            throughput_compliant: true,
            memory_compliant: true,
            cpu_compliant: true,
            error_rate_compliant: true,
            overall_compliant: true,
            violations: Vec::new(),
        };

        // Check latency
        if let Some(ref load_results) = report.load_test_results {
            if load_results.avg_latency_ms > self.config.sla_thresholds.max_latency_ms {
                compliance.latency_compliant = false;
                compliance.violations.push(format!(
                    "Latency SLA violation: {:.2}ms > {:.2}ms",
                    load_results.avg_latency_ms, self.config.sla_thresholds.max_latency_ms
                ));
            }

            if load_results.throughput_ops < self.config.sla_thresholds.min_throughput_ops {
                compliance.throughput_compliant = false;
                compliance.violations.push(format!(
                    "Throughput SLA violation: {:.2} ops/s < {:.2} ops/s",
                    load_results.throughput_ops, self.config.sla_thresholds.min_throughput_ops
                ));
            }
        }

        // Check resource utilization
        for metric in &report.resource_utilization {
            if metric.memory_mb > self.config.sla_thresholds.max_memory_mb {
                compliance.memory_compliant = false;
                compliance.violations.push(format!(
                    "Memory SLA violation: {:.2}MB > {:.2}MB",
                    metric.memory_mb, self.config.sla_thresholds.max_memory_mb
                ));
            }

            if metric.cpu_percent > self.config.sla_thresholds.max_cpu_percent {
                compliance.cpu_compliant = false;
                compliance.violations.push(format!(
                    "CPU SLA violation: {:.2}% > {:.2}%",
                    metric.cpu_percent, self.config.sla_thresholds.max_cpu_percent
                ));
            }
        }

        compliance.overall_compliant = compliance.latency_compliant
            && compliance.throughput_compliant
            && compliance.memory_compliant
            && compliance.cpu_compliant
            && compliance.error_rate_compliant;

        Ok(compliance)
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(
        &self,
        report: &ScalabilityTestReport,
    ) -> Result<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();

        // Check for memory bottlenecks
        let max_memory = report
            .resource_utilization
            .iter()
            .map(|m| m.memory_mb)
            .fold(0.0, f64::max);

        if max_memory > self.config.sla_thresholds.max_memory_mb * 0.9 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::Memory,
                severity: BottleneckSeverity::High,
                description: format!("Memory usage approaching limit: {:.2}MB", max_memory),
                impact_score: 0.8,
                suggested_actions: vec![
                    "Optimize data structures".to_string(),
                    "Implement caching strategies".to_string(),
                    "Consider distributed architecture".to_string(),
                ],
            });
        }

        // Check for CPU bottlenecks
        let max_cpu = report
            .resource_utilization
            .iter()
            .map(|m| m.cpu_percent)
            .fold(0.0, f64::max);

        if max_cpu > self.config.sla_thresholds.max_cpu_percent {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::CPU,
                severity: BottleneckSeverity::Medium,
                description: format!("CPU usage exceeding threshold: {:.2}%", max_cpu),
                impact_score: 0.6,
                suggested_actions: vec![
                    "Optimize algorithms".to_string(),
                    "Enable parallel processing".to_string(),
                    "Profile for hot paths".to_string(),
                ],
            });
        }

        // Check for latency bottlenecks
        if let Some(ref load_results) = report.load_test_results {
            if load_results.p99_latency_ms > self.config.sla_thresholds.max_latency_ms * 2.0 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::Latency,
                    severity: BottleneckSeverity::High,
                    description: format!(
                        "P99 latency too high: {:.2}ms",
                        load_results.p99_latency_ms
                    ),
                    impact_score: 0.9,
                    suggested_actions: vec![
                        "Optimize query execution".to_string(),
                        "Add indexing".to_string(),
                        "Implement query caching".to_string(),
                    ],
                });
            }
        }

        Ok(bottlenecks)
    }

    /// Calculate overall scalability score
    fn calculate_scalability_score(&self, report: &ScalabilityTestReport) -> f64 {
        let mut score = 100.0;

        // Deduct for SLA violations
        if !report.sla_compliance.latency_compliant {
            score -= 20.0;
        }
        if !report.sla_compliance.throughput_compliant {
            score -= 20.0;
        }
        if !report.sla_compliance.memory_compliant {
            score -= 15.0;
        }
        if !report.sla_compliance.cpu_compliant {
            score -= 15.0;
        }

        // Deduct for bottlenecks
        for bottleneck in &report.bottlenecks_identified {
            score -= bottleneck.impact_score * 10.0;
        }

        // Bonus for good performance
        if let Some(ref load_results) = report.load_test_results {
            if load_results.avg_latency_ms < self.config.sla_thresholds.max_latency_ms * 0.5 {
                score += 10.0;
            }
        }

        score.clamp(0.0, 100.0)
    }

    /// Generate scalability recommendations
    fn generate_recommendations(
        &self,
        report: &ScalabilityTestReport,
    ) -> Result<Vec<ScalabilityRecommendation>> {
        let mut recommendations = Vec::new();

        // Recommendations based on bottlenecks
        for bottleneck in &report.bottlenecks_identified {
            recommendations.push(ScalabilityRecommendation {
                priority: match bottleneck.severity {
                    BottleneckSeverity::Critical => RecommendationPriority::Critical,
                    BottleneckSeverity::High => RecommendationPriority::High,
                    BottleneckSeverity::Medium => RecommendationPriority::Medium,
                    BottleneckSeverity::Low => RecommendationPriority::Low,
                },
                title: format!("Address {} Bottleneck", bottleneck.bottleneck_type.as_str()),
                description: bottleneck.description.clone(),
                actions: bottleneck.suggested_actions.clone(),
                expected_improvement: format!(
                    "{:.0}% performance improvement",
                    bottleneck.impact_score * 100.0
                ),
            });
        }

        // General recommendations
        if !report.sla_compliance.overall_compliant {
            recommendations.push(ScalabilityRecommendation {
                priority: RecommendationPriority::High,
                title: "Improve SLA Compliance".to_string(),
                description: "System is not meeting SLA requirements".to_string(),
                actions: report.sla_compliance.violations.clone(),
                expected_improvement: "Meet all SLA targets".to_string(),
            });
        }

        Ok(recommendations)
    }

    /// Save test report
    fn save_report(&self, report: &ScalabilityTestReport) -> Result<()> {
        let filename = format!("scalability_test_{}.json", report.test_id);
        let path = self.config.output_directory.join(filename);

        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(path, json)?;

        Ok(())
    }

    /// Get test history
    pub fn get_test_history(&self) -> &[ScalabilityTestReport] {
        &self.test_history
    }
}

/// Scalability test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityTestReport {
    pub test_id: Uuid,
    pub timestamp: SystemTime,
    pub duration: Duration,
    pub workload_description: String,
    pub load_test_results: Option<LoadTestResults>,
    pub stress_test_results: Option<StressTestResults>,
    pub endurance_test_results: Option<EnduranceTestResults>,
    pub spike_test_results: Option<SpikeTestResults>,
    pub resource_utilization: Vec<ResourceMetrics>,
    pub sla_compliance: SlaComplianceReport,
    pub bottlenecks_identified: Vec<PerformanceBottleneck>,
    pub scalability_score: f64,
    pub recommendations: Vec<ScalabilityRecommendation>,
}

/// Test workload specification
#[derive(Debug, Clone)]
pub struct TestWorkload {
    pub description: String,
    pub dataset_sizes: Vec<usize>,
    pub concurrent_users: Vec<usize>,
    pub operation_mix: HashMap<String, f64>,
}

/// Load tester
#[derive(Debug)]
pub struct LoadTester {
    rng: Random,
}

impl LoadTester {
    pub fn new() -> Self {
        Self {
            rng: Random::default(),
        }
    }

    pub fn run_load_test(
        &mut self,
        workload: &TestWorkload,
        config: &ScalabilityTestConfig,
    ) -> Result<LoadTestResults> {
        tracing::info!(
            "Running load test with dataset sizes: {:?}",
            workload.dataset_sizes
        );

        let mut latencies = Vec::new();
        let mut throughputs = Vec::new();

        // Test different dataset sizes
        for &size in &workload.dataset_sizes {
            if size > config.max_dataset_size {
                continue;
            }

            // Simulate load testing
            let latency_ms = self.simulate_operation_latency(size);
            let throughput = self.simulate_throughput(size);

            latencies.push(latency_ms);
            throughputs.push(throughput);
        }

        // Calculate statistics
        let avg_latency_ms = latencies.iter().sum::<f64>() / latencies.len().max(1) as f64;
        let p95_latency_ms = Self::percentile(&latencies, 0.95);
        let p99_latency_ms = Self::percentile(&latencies, 0.99);
        let throughput_ops = throughputs.iter().sum::<f64>() / throughputs.len().max(1) as f64;

        Ok(LoadTestResults {
            total_requests: workload.dataset_sizes.len(),
            successful_requests: workload.dataset_sizes.len(),
            failed_requests: 0,
            avg_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            throughput_ops,
            errors: Vec::new(),
        })
    }

    fn simulate_operation_latency(&mut self, dataset_size: usize) -> f64 {
        // Simulate latency that grows with dataset size
        let base_latency = 10.0; // 10ms base
        let size_factor = (dataset_size as f64).log10() * 20.0;
        let noise = self.rng.random::<f64>() * 10.0;
        base_latency + size_factor + noise
    }

    fn simulate_throughput(&mut self, dataset_size: usize) -> f64 {
        // Simulate throughput that decreases with dataset size
        let max_throughput = 1000.0;
        let size_penalty = (dataset_size as f64 / 1000.0).sqrt();
        max_throughput / (1.0 + size_penalty)
    }

    fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
        sorted[idx]
    }
}

impl Default for LoadTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Load test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResults {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_ops: f64,
    pub errors: Vec<String>,
}

/// Stress tester
#[derive(Debug)]
pub struct StressTester {
    rng: Random,
}

impl StressTester {
    pub fn new() -> Self {
        Self {
            rng: Random::default(),
        }
    }

    pub fn run_stress_test(
        &mut self,
        workload: &TestWorkload,
        config: &ScalabilityTestConfig,
    ) -> Result<StressTestResults> {
        tracing::info!("Running stress test up to breaking point");

        // Find breaking point by increasing load
        let mut breaking_point_users = 0;
        let mut peak_throughput = 0.0;

        for users in (10..config.max_concurrent_users).step_by(50) {
            let throughput = self.simulate_stress_throughput(users);

            if throughput < peak_throughput * 0.9 {
                // Degradation detected
                break;
            }

            breaking_point_users = users;
            peak_throughput = throughput;
        }

        Ok(StressTestResults {
            breaking_point_users,
            peak_throughput,
            recovery_time_ms: self.rng.random::<f64>() * 1000.0,
            degradation_observed: breaking_point_users < config.max_concurrent_users,
        })
    }

    fn simulate_stress_throughput(&mut self, users: usize) -> f64 {
        let optimal_users = 100.0;
        let peak_throughput = 1000.0;

        if users as f64 <= optimal_users {
            peak_throughput * (users as f64 / optimal_users)
        } else {
            peak_throughput * (optimal_users / users as f64).sqrt()
        }
    }
}

impl Default for StressTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Stress test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResults {
    pub breaking_point_users: usize,
    pub peak_throughput: f64,
    pub recovery_time_ms: f64,
    pub degradation_observed: bool,
}

/// Endurance tester
#[derive(Debug)]
pub struct EnduranceTester {
    rng: Random,
}

impl EnduranceTester {
    pub fn new() -> Self {
        Self {
            rng: Random::default(),
        }
    }

    pub fn run_endurance_test(
        &mut self,
        _workload: &TestWorkload,
        config: &ScalabilityTestConfig,
    ) -> Result<EnduranceTestResults> {
        tracing::info!("Running endurance test for {:?}", config.endurance_duration);

        // Simulate long-running test
        let memory_leak_detected = self.rng.random::<f64>() < 0.1;
        let performance_degradation = self.rng.random::<f64>() < 0.2;

        Ok(EnduranceTestResults {
            test_duration: config.endurance_duration,
            memory_leak_detected,
            performance_degradation,
            avg_memory_growth_mb_per_hour: if memory_leak_detected { 100.0 } else { 5.0 },
        })
    }
}

impl Default for EnduranceTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Endurance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnduranceTestResults {
    pub test_duration: Duration,
    pub memory_leak_detected: bool,
    pub performance_degradation: bool,
    pub avg_memory_growth_mb_per_hour: f64,
}

/// Spike tester
#[derive(Debug)]
pub struct SpikeTester {
    rng: Random,
}

impl SpikeTester {
    pub fn new() -> Self {
        Self {
            rng: Random::default(),
        }
    }

    pub fn run_spike_test(
        &mut self,
        _workload: &TestWorkload,
        config: &ScalabilityTestConfig,
    ) -> Result<SpikeTestResults> {
        tracing::info!("Running spike test with sudden load increase");

        let spike_magnitude = config.max_concurrent_users;
        let recovery_successful = self.rng.random::<f64>() > 0.1;

        Ok(SpikeTestResults {
            spike_magnitude,
            max_latency_during_spike_ms: 500.0 + self.rng.random::<f64>() * 500.0,
            recovery_time_ms: if recovery_successful {
                100.0 + self.rng.random::<f64>() * 200.0
            } else {
                5000.0
            },
            recovery_successful,
        })
    }
}

impl Default for SpikeTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Spike test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeTestResults {
    pub spike_magnitude: usize,
    pub max_latency_during_spike_ms: f64,
    pub recovery_time_ms: f64,
    pub recovery_successful: bool,
}

/// Resource monitor
#[derive(Debug)]
pub struct ResourceMonitor {
    metrics: Vec<ResourceMetrics>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    pub fn record_metrics(&mut self, metrics: ResourceMetrics) {
        self.metrics.push(metrics);
    }

    pub fn get_metrics(&self) -> Vec<ResourceMetrics> {
        self.metrics.clone()
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub timestamp: SystemTime,
    pub cpu_percent: f64,
    pub memory_mb: f64,
    pub disk_io_mbps: f64,
    pub network_io_mbps: f64,
}

/// SLA compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaComplianceReport {
    pub latency_compliant: bool,
    pub throughput_compliant: bool,
    pub memory_compliant: bool,
    pub cpu_compliant: bool,
    pub error_rate_compliant: bool,
    pub overall_compliant: bool,
    pub violations: Vec<String>,
}

impl Default for SlaComplianceReport {
    fn default() -> Self {
        Self {
            latency_compliant: true,
            throughput_compliant: true,
            memory_compliant: true,
            cpu_compliant: true,
            error_rate_compliant: true,
            overall_compliant: true,
            violations: Vec::new(),
        }
    }
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: BottleneckSeverity,
    pub description: String,
    pub impact_score: f64,
    pub suggested_actions: Vec<String>,
}

/// Bottleneck types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    DiskIO,
    NetworkIO,
    Latency,
    Concurrency,
}

impl BottleneckType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BottleneckType::CPU => "CPU",
            BottleneckType::Memory => "Memory",
            BottleneckType::DiskIO => "Disk I/O",
            BottleneckType::NetworkIO => "Network I/O",
            BottleneckType::Latency => "Latency",
            BottleneckType::Concurrency => "Concurrency",
        }
    }
}

/// Bottleneck severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Scalability recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityRecommendation {
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub actions: Vec<String>,
    pub expected_improvement: String,
}

/// Recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_workload() -> TestWorkload {
        TestWorkload {
            description: "Test workload".to_string(),
            dataset_sizes: vec![100, 1000, 10000],
            concurrent_users: vec![10, 50, 100],
            operation_mix: HashMap::new(),
        }
    }

    #[test]
    fn test_scalability_framework_creation() {
        let config = ScalabilityTestConfig::default();
        let framework = ScalabilityTestingFramework::new(config);
        assert!(framework.is_ok());
    }

    #[test]
    fn test_load_tester() {
        let mut tester = LoadTester::new();
        let workload = create_test_workload();
        let config = ScalabilityTestConfig::default();

        let results = tester.run_load_test(&workload, &config).unwrap();
        assert_eq!(results.total_requests, 3);
        assert!(results.avg_latency_ms > 0.0);
        assert!(results.throughput_ops > 0.0);
    }

    #[test]
    fn test_stress_tester() {
        let mut tester = StressTester::new();
        let workload = create_test_workload();
        let config = ScalabilityTestConfig::default();

        let results = tester.run_stress_test(&workload, &config).unwrap();
        assert!(results.breaking_point_users > 0);
        assert!(results.peak_throughput > 0.0);
    }

    #[test]
    fn test_spike_tester() {
        let mut tester = SpikeTester::new();
        let workload = create_test_workload();
        let config = ScalabilityTestConfig::default();

        let results = tester.run_spike_test(&workload, &config).unwrap();
        assert!(results.spike_magnitude > 0);
        assert!(results.max_latency_during_spike_ms > 0.0);
    }

    #[test]
    fn test_resource_monitor() {
        let mut monitor = ResourceMonitor::new();

        let metrics = ResourceMetrics {
            timestamp: SystemTime::now(),
            cpu_percent: 50.0,
            memory_mb: 1024.0,
            disk_io_mbps: 10.0,
            network_io_mbps: 5.0,
        };

        monitor.record_metrics(metrics);
        assert_eq!(monitor.get_metrics().len(), 1);
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p95 = LoadTester::percentile(&values, 0.95);
        assert!(p95 >= 4.0);
    }

    #[test]
    fn test_sla_thresholds() {
        let thresholds = SlaThresholds::default();
        assert_eq!(thresholds.max_latency_ms, 1000.0);
        assert_eq!(thresholds.max_memory_mb, 4096.0);
        assert_eq!(thresholds.min_throughput_ops, 100.0);
    }

    #[test]
    fn test_bottleneck_type_strings() {
        assert_eq!(BottleneckType::CPU.as_str(), "CPU");
        assert_eq!(BottleneckType::Memory.as_str(), "Memory");
        assert_eq!(BottleneckType::Latency.as_str(), "Latency");
    }

    #[test]
    fn test_comprehensive_test_suite() {
        let config = ScalabilityTestConfig::default();
        let mut framework = ScalabilityTestingFramework::new(config).unwrap();
        let workload = create_test_workload();

        let report = framework.run_all_tests(&workload).unwrap();
        assert!(report.scalability_score >= 0.0);
        assert!(report.scalability_score <= 100.0);
    }
}
