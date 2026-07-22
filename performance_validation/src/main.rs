//! OxiRS Performance Validation Suite
//!
//! Comprehensive performance validation framework for testing real-world performance
//! improvements from GPU acceleration, SIMD optimizations, scirs2 integration,
//! and federated query optimizations.

use anyhow::Result;
use clap::{Arg, Command};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;
use tokio::fs;
use tracing::info;

mod benchmarks;
mod datasets;
mod scenarios;

use benchmarks::{
    ai_ml::{self, AiMlBenchmarkConfig},
    federation::{self, FederationBenchmarkConfig},
    gpu_acceleration::{self, GpuBenchmarkConfig},
    simd_operations::{self, SimdBenchmarkConfig},
};
use datasets::DatasetManager;
use scenarios::{ScenarioResult, ScenarioRunner};

/// Average the real, measured per-suite scores (each 0-100), skipping any
/// suite that did not produce a score (e.g. it failed to run or was
/// disabled) rather than substituting a fixed "typical performance"
/// placeholder value for it.
fn average_measured_scores(scores: &[Option<f64>]) -> f64 {
    let measured: Vec<f64> = scores.iter().filter_map(|s| *s).collect();
    if measured.is_empty() {
        0.0
    } else {
        measured.iter().sum::<f64>() / measured.len() as f64
    }
}

/// Performance validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub output_directory: PathBuf,
    pub run_all_benchmarks: bool,
    pub run_scenarios: bool,
    pub save_detailed_reports: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            output_directory: PathBuf::from("./performance_reports"),
            run_all_benchmarks: true,
            run_scenarios: true,
            save_detailed_reports: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub timestamp: String,
    pub config: ValidationConfig,
    pub system_info: SystemInfo,
    pub benchmark_results: BenchmarkResults,
    pub scenario_results: Vec<ScenarioResult>,
    pub summary: ValidationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub platform: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub gpu_available: bool,
    pub simd_support: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub gpu_acceleration: Option<String>,
    pub simd_operations: Option<String>,
    pub federation: Option<String>,
    pub ai_ml: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub total_scenarios_run: usize,
    pub scenarios_passed: usize,
    pub overall_performance_score: f64,
    pub key_findings: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Performance validation runner
pub struct ValidationRunner {
    config: ValidationConfig,
}

impl ValidationRunner {
    /// Create new validation runner
    pub fn new(config: ValidationConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Run complete performance validation suite
    pub async fn run_validation(&mut self) -> Result<ValidationResults> {
        info!("🚀 Starting OxiRS Performance Validation Suite");

        let start_time = Instant::now();

        // Collect system information
        let system_info = self.collect_system_info().await?;
        info!("System Info: {:?}", system_info);

        // Create output directory
        tokio::fs::create_dir_all(&self.config.output_directory).await?;

        let mut benchmark_results = BenchmarkResults {
            gpu_acceleration: None,
            simd_operations: None,
            federation: None,
            ai_ml: None,
        };

        let mut scenario_results = Vec::new();
        let mut all_findings = Vec::new();
        let mut all_recommendations = Vec::new();

        // Real, measured per-suite scores (0-100) collected as each benchmark
        // suite actually runs. `None` means the suite did not run / failed,
        // and is excluded from the headline average rather than papered over
        // with a fixed "typical performance" constant.
        let mut gpu_score: Option<f64> = None;
        let mut simd_score: Option<f64> = None;
        let mut federation_score: Option<f64> = None;
        let mut ai_ml_score: Option<f64> = None;

        // Run individual benchmarks if enabled
        if self.config.run_all_benchmarks {
            info!("🎮 Running GPU acceleration benchmarks...");
            if let Ok(gpu_suite) = self.run_gpu_benchmarks().await {
                let report_path = format!(
                    "{}/gpu_benchmark_report.md",
                    self.config.output_directory.display()
                );
                gpu_acceleration::report::save_gpu_benchmark_report(&gpu_suite, &report_path)?;
                benchmark_results.gpu_acceleration = Some(format!("Saved to {}", report_path));

                // Real pass-rate score derived from the suite's own measured
                // test outcomes (0-100), not a fixed "typical" constant.
                let total_gpu_tests =
                    gpu_suite.summary.tests_passed + gpu_suite.summary.tests_failed;
                gpu_score = Some(if total_gpu_tests > 0 {
                    (gpu_suite.summary.tests_passed as f64 / total_gpu_tests as f64) * 100.0
                } else {
                    0.0
                });

                if gpu_suite.summary.performance_target_met {
                    all_findings.push("GPU acceleration targets achieved".to_string());
                } else {
                    all_findings.push("GPU acceleration below target performance".to_string());
                    all_recommendations.push("Review GPU acceleration implementation".to_string());
                }
            }

            info!("⚡ Running SIMD optimization benchmarks...");
            if let Ok(simd_suite) = self.run_simd_benchmarks().await {
                let report_path = format!(
                    "{}/simd_benchmark_report.md",
                    self.config.output_directory.display()
                );
                simd_operations::report::save_simd_benchmark_report(&simd_suite, &report_path)?;
                benchmark_results.simd_operations = Some(format!("Saved to {}", report_path));

                // Real pass-rate score: fraction of operations that met the
                // configured SIMD performance threshold.
                simd_score = Some(if simd_suite.summary.total_operations_tested > 0 {
                    (simd_suite.summary.operations_meeting_target as f64
                        / simd_suite.summary.total_operations_tested as f64)
                        * 100.0
                } else {
                    0.0
                });

                all_findings.push(format!(
                    "SIMD average speedup: {:.2}x",
                    simd_suite.summary.average_speedup
                ));
                if simd_suite.summary.average_speedup < 2.0 {
                    all_recommendations
                        .push("Consider expanding SIMD optimization coverage".to_string());
                }
            }

            info!("🔗 Running federation optimization benchmarks...");
            if let Ok(federation_suite) = self.run_federation_benchmarks().await {
                let report_path = format!(
                    "{}/federation_benchmark_report.md",
                    self.config.output_directory.display()
                );
                federation::report::save_federation_benchmark_report(
                    &federation_suite,
                    &report_path,
                )?;
                benchmark_results.federation = Some(format!("Saved to {}", report_path));

                // Real pass-rate score: fraction of federation tests that met
                // their target improvement.
                federation_score = Some(if federation_suite.summary.total_tests > 0 {
                    (federation_suite.summary.tests_meeting_target as f64
                        / federation_suite.summary.total_tests as f64)
                        * 100.0
                } else {
                    0.0
                });

                all_findings.push(format!(
                    "Federation average improvement: {:.1}%",
                    federation_suite.summary.average_improvement_percent
                ));
                if federation_suite.summary.average_improvement_percent < 50.0 {
                    all_recommendations
                        .push("Enhance ML-driven federation optimization".to_string());
                }
            }

            info!("🧠 Running AI/ML optimization benchmarks...");
            if let Ok(ai_ml_suite) = self.run_ai_ml_benchmarks().await {
                let report_path = format!(
                    "{}/ai_ml_benchmark_report.md",
                    self.config.output_directory.display()
                );
                ai_ml::report::save_ai_ml_benchmark_report(&ai_ml_suite, &report_path)?;
                benchmark_results.ai_ml = Some(format!("Saved to {}", report_path));

                // AI/ML suite already reports a real 0-100 performance score.
                ai_ml_score = Some(ai_ml_suite.summary.overall_performance_score);

                all_findings.push(format!(
                    "AI/ML performance score: {:.1}/100",
                    ai_ml_suite.summary.overall_performance_score
                ));
                if ai_ml_suite.summary.overall_performance_score < 80.0 {
                    all_recommendations
                        .push("Optimize AI/ML algorithms for better performance".to_string());
                }
            }
        }

        // Run real-world scenarios if enabled
        if self.config.run_scenarios {
            info!("🌍 Running real-world performance scenarios...");

            // Initialize dataset manager and scenario runner
            let datasets_path = format!("{}/datasets", self.config.output_directory.display());
            let mut dataset_manager = DatasetManager::new(datasets_path);
            dataset_manager.generate_validation_datasets().await?;
            dataset_manager.save_datasets_to_disk().await?;

            let mut scenario_runner = ScenarioRunner::new(dataset_manager);
            scenario_results = scenario_runner.run_all_scenarios().await?;

            // Save scenario report
            let scenario_report_path = format!(
                "{}/scenario_validation_report.md",
                self.config.output_directory.display()
            );
            scenarios::report::save_scenario_report(&scenario_results, &scenario_report_path)?;

            // Extract findings and recommendations from scenarios
            for result in &scenario_results {
                all_findings.extend(result.detailed_findings.clone());
                all_recommendations.extend(result.recommendations.clone());
            }
        }

        let total_scenarios = scenario_results.len();
        let scenarios_passed = scenario_results
            .iter()
            .filter(|r| r.objectives_met.overall_objectives_met)
            .count();

        // Calculate overall performance score from the real, measured suite
        // scores collected above plus the real scenario pass rate. Suites
        // that did not run are excluded from the average rather than
        // substituted with a fixed "typical performance" constant.
        let scenario_score = if total_scenarios > 0 {
            Some((scenarios_passed as f64 / total_scenarios as f64) * 100.0)
        } else {
            None
        };

        let overall_performance_score = average_measured_scores(&[
            gpu_score,
            simd_score,
            federation_score,
            ai_ml_score,
            scenario_score,
        ]);

        let summary = ValidationSummary {
            total_scenarios_run: total_scenarios,
            scenarios_passed,
            overall_performance_score,
            key_findings: all_findings,
            recommendations: all_recommendations,
        };

        let results = ValidationResults {
            timestamp: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            config: self.config.clone(),
            system_info,
            benchmark_results,
            scenario_results,
            summary,
        };

        info!(
            "✅ Performance validation completed in {:.2}s",
            start_time.elapsed().as_secs_f64()
        );
        Ok(results)
    }

    async fn collect_system_info(&self) -> Result<SystemInfo> {
        let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let memory_gb = self.detect_total_memory_gb();
        let gpu_available = self.detect_gpu_availability();
        let simd_support = self.detect_simd_features();

        let cpu_model = self.get_cpu_model();

        Ok(SystemInfo {
            platform,
            cpu_model,
            cpu_cores,
            memory_gb,
            gpu_available,
            simd_support,
        })
    }

    async fn run_gpu_benchmarks(&self) -> Result<gpu_acceleration::GpuBenchmarkSuite> {
        let config = GpuBenchmarkConfig {
            embedding_sizes: vec![10000, 100000],
            batch_sizes: vec![128, 512],
            iterations: 5,
            warmup_iterations: 2,
            dimensions: 512,
            use_mixed_precision: true,
            test_adaptive_switching: true,
            gpu_memory_limit_mb: Some(4096),
        };

        gpu_acceleration::run_gpu_acceleration_benchmark(config).await
    }

    async fn run_simd_benchmarks(&self) -> Result<simd_operations::SimdBenchmarkSuite> {
        let config = SimdBenchmarkConfig {
            vector_sizes: vec![128, 512, 1024, 2048],
            iterations: 20,
            warmup_iterations: 3,
            test_operations: vec![
                simd_operations::SimdOperation::VectorAdd,
                simd_operations::SimdOperation::DotProduct,
                simd_operations::SimdOperation::CosineSimilarity,
                simd_operations::SimdOperation::EuclideanDistance,
            ],
            cross_platform_test: true,
            performance_threshold: 2.0,
        };

        simd_operations::run_simd_benchmark(config).await
    }

    async fn run_federation_benchmarks(&self) -> Result<federation::FederationBenchmarkSuite> {
        let config = FederationBenchmarkConfig {
            endpoint_counts: vec![3, 5],
            query_complexities: vec![
                federation::QueryComplexity::Medium,
                federation::QueryComplexity::Complex,
            ],
            concurrent_queries: vec![5, 10],
            iterations: 10,
            warmup_iterations: 2,
            test_ml_optimization: true,
            test_scirs2_integration: true,
            network_simulation: federation::NetworkSimulation {
                simulate_latency: true,
                latency_range_ms: (50, 200),
                simulate_bandwidth_limits: true,
                packet_loss_percent: 0.5,
                variable_endpoint_performance: true,
            },
            expected_improvement_percent: 50.0,
        };

        federation::run_federation_benchmark(config).await
    }

    async fn run_ai_ml_benchmarks(&self) -> Result<ai_ml::AiMlBenchmarkSuite> {
        let config = AiMlBenchmarkConfig {
            embedding_dimensions: vec![256, 512],
            dataset_sizes: vec![10000, 50000],
            batch_sizes: vec![128, 512],
            neural_architectures: vec![
                ai_ml::NeuralArchitecture::TransE,
                ai_ml::NeuralArchitecture::ComplEx,
            ],
            iterations: 5,
            warmup_iterations: 2,
            test_gpu_acceleration: true,
            test_simd_optimization: true,
            test_scirs2_integration: true,
            performance_thresholds: ai_ml::PerformanceThresholds {
                embedding_generation_speedup: 3.0,
                similarity_computation_speedup: 4.0,
                training_speedup: 2.5,
                memory_efficiency_improvement: 25.0,
                accuracy_tolerance: 0.01,
            },
        };

        ai_ml::run_ai_ml_benchmark(config).await
    }

    /// Query the actual host's total physical memory, in GB, instead of
    /// reporting a fixed value that would be wrong on any machine that
    /// doesn't happen to have exactly 16GB of RAM.
    fn detect_total_memory_gb(&self) -> f64 {
        let mut system = sysinfo::System::new_all();
        system.refresh_memory();
        // total_memory() is reported in bytes; convert to GB (base-1024).
        system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0
    }

    #[allow(unreachable_code)]
    fn detect_gpu_availability(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            if std::process::Command::new("nvidia-smi").output().is_ok() {
                return true;
            }
        }

        #[cfg(target_os = "macos")]
        {
            return true; // Assume Metal support on macOS
        }

        false
    }

    fn detect_simd_features(&self) -> Vec<String> {
        let mut features = Vec::new();

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                features.push("AVX2".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx") {
                features.push("AVX".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse4.2") {
                features.push("SSE4.2".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                features.push("NEON".to_string());
            }
        }

        if features.is_empty() {
            features.push("Scalar".to_string());
        }

        features
    }

    fn get_cpu_model(&self) -> String {
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
            {
                return String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in contents.lines() {
                    if line.starts_with("model name") {
                        if let Some(model) = line.split(':').nth(1) {
                            return model.trim().to_string();
                        }
                    }
                }
            }
        }

        format!("{} CPU", std::env::consts::ARCH)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let matches = Command::new("oxirs-performance-validation")
        .version("1.0.0")
        .about("OxiRS Performance Validation Suite")
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("DIR")
                .help("Output directory for reports")
                .default_value("./performance_reports"),
        )
        .arg(
            Arg::new("benchmarks-only")
                .long("benchmarks-only")
                .help("Run only individual benchmarks (skip scenarios)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("scenarios-only")
                .long("scenarios-only")
                .help("Run only real-world scenarios (skip individual benchmarks)")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Create configuration
    let output_directory = PathBuf::from(
        matches
            .get_one::<String>("output")
            .expect("'output' has a default value"),
    );
    let benchmarks_only = matches.get_flag("benchmarks-only");
    let scenarios_only = matches.get_flag("scenarios-only");

    let config = ValidationConfig {
        output_directory: output_directory.clone(),
        run_all_benchmarks: !scenarios_only,
        run_scenarios: !benchmarks_only,
        save_detailed_reports: true,
    };

    // Create validation runner
    let mut runner = ValidationRunner::new(config)?;

    // Run validation
    let results = runner.run_validation().await?;

    // Save comprehensive report
    let summary_path = format!("{}/validation_summary.json", output_directory.display());
    let results_json = serde_json::to_string_pretty(&results)?;
    fs::write(&summary_path, results_json).await?;

    // Print summary
    println!("\n🎯 OxiRS Performance Validation Summary");
    println!("==========================================");
    println!("📅 Timestamp: {}", results.timestamp);
    println!("🖥️  Platform: {}", results.system_info.platform);
    println!("🔧 CPU: {}", results.system_info.cpu_model);
    println!("⚡ SIMD: {:?}", results.system_info.simd_support);
    println!(
        "🎮 GPU: {}",
        if results.system_info.gpu_available {
            "Available"
        } else {
            "Not Available"
        }
    );
    println!();
    println!("📊 Performance Results:");
    println!("- Scenarios Run: {}", results.summary.total_scenarios_run);
    println!("- Scenarios Passed: {}", results.summary.scenarios_passed);
    println!(
        "- Overall Score: {:.1}/100",
        results.summary.overall_performance_score
    );
    println!();

    if !results.summary.key_findings.is_empty() {
        println!("🔍 Key Findings:");
        for finding in &results.summary.key_findings {
            println!("  • {}", finding);
        }
        println!();
    }

    if !results.summary.recommendations.is_empty() {
        println!("💡 Recommendations:");
        for recommendation in &results.summary.recommendations {
            println!("  • {}", recommendation);
        }
        println!();
    }

    // Print report locations
    println!("📋 Reports Generated:");
    if let Some(gpu_report) = &results.benchmark_results.gpu_acceleration {
        println!("  🎮 GPU: {}", gpu_report);
    }
    if let Some(simd_report) = &results.benchmark_results.simd_operations {
        println!("  ⚡ SIMD: {}", simd_report);
    }
    if let Some(federation_report) = &results.benchmark_results.federation {
        println!("  🔗 Federation: {}", federation_report);
    }
    if let Some(ai_ml_report) = &results.benchmark_results.ai_ml {
        println!("  🧠 AI/ML: {}", ai_ml_report);
    }
    println!("  📄 Summary: {}", summary_path);

    println!("\n✅ Performance validation completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// regression: overall_performance_score must average only the suites
    /// that actually produced a real measurement, never fall back to fixed
    /// "typical performance" constants for suites that didn't run.
    #[test]
    fn regression_average_measured_scores_skips_missing_suites() {
        // Two suites ran (real scores 100.0 and 0.0), two never ran (None).
        let scores = [Some(100.0), None, Some(0.0), None, None];
        let average = average_measured_scores(&scores);
        // Average of only the two real measurements: (100.0 + 0.0) / 2 = 50.0
        // NOT (100.0 + 0.0 + 85.0 + 80.0 + 75.0) / 5 style fabricated padding.
        assert_eq!(average, 50.0);
    }

    #[test]
    fn regression_average_measured_scores_all_present() {
        let scores = [Some(80.0), Some(90.0), Some(70.0), Some(60.0), Some(100.0)];
        let average = average_measured_scores(&scores);
        assert!((average - 80.0).abs() < 1e-9);
    }

    #[test]
    fn regression_average_measured_scores_none_present_is_zero_not_fabricated() {
        let scores: [Option<f64>; 5] = [None, None, None, None, None];
        assert_eq!(average_measured_scores(&scores), 0.0);
    }

    /// regression: total system memory must be queried from the real host
    /// instead of the previous hardcoded `16.0`.
    #[test]
    fn regression_detect_total_memory_gb_uses_real_host_telemetry() {
        let runner = ValidationRunner::new(ValidationConfig::default())
            .expect("runner construction is infallible");
        let reported_gb = runner.detect_total_memory_gb();

        // Independently query the same telemetry source and compare, proving
        // the value is derived from the live host rather than a literal.
        let mut system = sysinfo::System::new_all();
        system.refresh_memory();
        let expected_gb = system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

        assert!(reported_gb > 0.0, "host must report nonzero memory");
        assert!(
            (reported_gb - expected_gb).abs() < 0.01,
            "reported {reported_gb} GB should match live sysinfo query {expected_gb} GB"
        );
    }
}
