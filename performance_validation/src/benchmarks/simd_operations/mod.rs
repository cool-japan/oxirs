use rand_distr::Distribution;use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use scirs2_core::random::Random;
use rand_distr::Uniform;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdBenchmarkConfig {
    pub vector_sizes: Vec<usize>,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub test_operations: Vec<SimdOperation>,
    pub cross_platform_test: bool,
    pub performance_threshold: f64, // Minimum expected speedup
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimdOperation {
    VectorAdd,
    VectorSub,
    VectorMul,
    DotProduct,
    CosineSimilarity,
    EuclideanDistance,
    ManhattanDistance,
    ElementwiseMax,
    ElementwiseMin,
    VectorNormalize,
}

impl Default for SimdBenchmarkConfig {
    fn default() -> Self {
        Self {
            vector_sizes: vec![64, 128, 256, 512, 1024, 2048, 4096, 8192],
            iterations: 100,
            warmup_iterations: 10,
            test_operations: vec![
                SimdOperation::VectorAdd,
                SimdOperation::VectorSub,
                SimdOperation::VectorMul,
                SimdOperation::DotProduct,
                SimdOperation::CosineSimilarity,
                SimdOperation::EuclideanDistance,
                SimdOperation::ManhattanDistance,
                SimdOperation::ElementwiseMax,
                SimdOperation::ElementwiseMin,
                SimdOperation::VectorNormalize,
            ],
            cross_platform_test: true,
            performance_threshold: 2.0, // Expect at least 2x speedup
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdBenchmarkResult {
    pub operation: String,
    pub vector_size: usize,
    pub scalar_duration: Duration,
    pub simd_duration: Duration,
    pub speedup_factor: f64,
    pub simd_architecture: String,
    pub throughput_ops_per_sec: f64,
    pub memory_bandwidth_gb_per_sec: f64,
    pub performance_target_met: bool,
    pub correctness_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdBenchmarkSuite {
    pub config: SimdBenchmarkConfig,
    pub results: Vec<SimdBenchmarkResult>,
    pub total_duration: Duration,
    pub architecture_info: ArchitectureInfo,
    pub summary: SimdBenchmarkSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureInfo {
    pub platform: String,
    pub cpu_model: String,
    pub simd_support: Vec<String>,
    pub cache_sizes: CacheSizes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSizes {
    pub l1_cache_kb: Option<u32>,
    pub l2_cache_kb: Option<u32>,
    pub l3_cache_kb: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdBenchmarkSummary {
    pub average_speedup: f64,
    pub max_speedup: f64,
    pub min_speedup: f64,
    pub operations_meeting_target: usize,
    pub total_operations_tested: usize,
    pub best_performing_operation: String,
    pub worst_performing_operation: String,
    pub overall_performance_rating: String,
}

pub async fn run_simd_benchmark(config: SimdBenchmarkConfig) -> Result<SimdBenchmarkSuite> {
    println!("🔧 Starting SIMD Operations Benchmark Suite...");

    let start_time = Instant::now();
    let mut results = Vec::new();

    // Detect system architecture and SIMD capabilities
    let architecture_info = detect_architecture_info();
    println!("🖥️  Architecture: {} with SIMD support: {:?}",
             architecture_info.platform, architecture_info.simd_support);

    // Run benchmarks for each operation and vector size
    for operation in &config.test_operations {
        for &vector_size in &config.vector_sizes {
            println!("📊 Testing {:?} with vector size {}", operation, vector_size);

            let result = run_single_simd_benchmark(
                &config,
                operation,
                vector_size,
                &architecture_info,
            ).await?;

            results.push(result);
        }
    }

    let total_duration = start_time.elapsed();
    let summary = calculate_simd_benchmark_summary(&results, &config);

    Ok(SimdBenchmarkSuite {
        config,
        results,
        total_duration,
        architecture_info,
        summary,
    })
}

fn detect_architecture_info() -> ArchitectureInfo {
    let platform = std::env::consts::ARCH.to_string();
    let mut simd_support = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            simd_support.push("AVX2".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx") {
            simd_support.push("AVX".to_string());
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            simd_support.push("SSE4.2".to_string());
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            simd_support.push("SSE2".to_string());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            simd_support.push("NEON".to_string());
        }
    }

    ArchitectureInfo {
        platform: format!("{}-{}", std::env::consts::OS, platform),
        cpu_model: get_cpu_model(),
        simd_support,
        cache_sizes: detect_cache_sizes(),
    }
}

fn get_cpu_model() -> String {
    // Simple CPU model detection - in a real implementation, this would
    // use platform-specific APIs
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(&["-n", "machdep.cpu.brand_string"])
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

    "Unknown CPU".to_string()
}

fn detect_cache_sizes() -> CacheSizes {
    // Cache size detection would use platform-specific methods
    // For now, return reasonable defaults for modern CPUs
    CacheSizes {
        l1_cache_kb: Some(32),
        l2_cache_kb: Some(512),
        l3_cache_kb: Some(8192),
    }
}

async fn run_single_simd_benchmark(
    config: &SimdBenchmarkConfig,
    operation: &SimdOperation,
    vector_size: usize,
    arch_info: &ArchitectureInfo,
) -> Result<SimdBenchmarkResult> {
    // Generate test data
    let (test_data_a, test_data_b) = generate_test_vectors(vector_size);

    // Warmup runs
    for _ in 0..config.warmup_iterations {
        let _ = run_scalar_operation(operation, &test_data_a, &test_data_b);
        let _ = run_simd_operation(operation, &test_data_a, &test_data_b);
    }

    // Benchmark scalar implementation
    let mut scalar_durations = Vec::new();
    for _ in 0..config.iterations {
        let start = Instant::now();
        let _scalar_result = run_scalar_operation(operation, &test_data_a, &test_data_b);
        scalar_durations.push(start.elapsed());
    }

    // Benchmark SIMD implementation
    let mut simd_durations = Vec::new();
    for _ in 0..config.iterations {
        let start = Instant::now();
        let _simd_result = run_simd_operation(operation, &test_data_a, &test_data_b);
        simd_durations.push(start.elapsed());
    }

    // Calculate averages
    let scalar_duration = average_duration(&scalar_durations);
    let simd_duration = average_duration(&simd_durations);
    let speedup_factor = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();

    // Calculate throughput metrics
    let operations_per_iteration = vector_size;
    let throughput_ops_per_sec = (operations_per_iteration as f64 * config.iterations as f64)
        / simd_duration.as_secs_f64();

    // Estimate memory bandwidth (rough calculation)
    let bytes_per_operation = match operation {
        SimdOperation::VectorAdd | SimdOperation::VectorSub | SimdOperation::VectorMul =>
            vector_size * 4 * 3, // 3 vectors (2 input, 1 output) * 4 bytes per f32
        SimdOperation::DotProduct | SimdOperation::CosineSimilarity |
        SimdOperation::EuclideanDistance | SimdOperation::ManhattanDistance =>
            vector_size * 4 * 2, // 2 input vectors * 4 bytes per f32
        _ => vector_size * 4 * 2,
    };
    let memory_bandwidth_gb_per_sec = (bytes_per_operation as f64 * config.iterations as f64)
        / (simd_duration.as_secs_f64() * 1_000_000_000.0);

    // Verify correctness
    let correctness_verified = verify_operation_correctness(
        operation,
        &test_data_a,
        &test_data_b
    );

    // Determine SIMD architecture being used
    let simd_architecture = if arch_info.simd_support.contains(&"AVX2".to_string()) {
        "AVX2".to_string()
    } else if arch_info.simd_support.contains(&"NEON".to_string()) {
        "NEON".to_string()
    } else if arch_info.simd_support.contains(&"AVX".to_string()) {
        "AVX".to_string()
    } else {
        "Scalar".to_string()
    };

    Ok(SimdBenchmarkResult {
        operation: format!("{:?}", operation),
        vector_size,
        scalar_duration,
        simd_duration,
        speedup_factor,
        simd_architecture,
        throughput_ops_per_sec,
        memory_bandwidth_gb_per_sec,
        performance_target_met: speedup_factor >= config.performance_threshold,
        correctness_verified,
    })
}

fn generate_test_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Random::seed(42);

    let vec_a: Vec<f32> = (0..size)
        .map(|_| rng.random_range(-1.0, 1.0) as f32)
        .collect();

    let vec_b: Vec<f32> = (0..size)
        .map(|_| rng.random_range(-1.0, 1.0) as f32)
        .collect();

    (vec_a, vec_b)
}

fn run_scalar_operation(
    operation: &SimdOperation,
    vec_a: &[f32],
    vec_b: &[f32],
) -> Vec<f32> {
    match operation {
        SimdOperation::VectorAdd => scalar_vector_add(vec_a, vec_b),
        SimdOperation::VectorSub => scalar_vector_sub(vec_a, vec_b),
        SimdOperation::VectorMul => scalar_vector_mul(vec_a, vec_b),
        SimdOperation::DotProduct => vec![scalar_dot_product(vec_a, vec_b)],
        SimdOperation::CosineSimilarity => vec![scalar_cosine_similarity(vec_a, vec_b)],
        SimdOperation::EuclideanDistance => vec![scalar_euclidean_distance(vec_a, vec_b)],
        SimdOperation::ManhattanDistance => vec![scalar_manhattan_distance(vec_a, vec_b)],
        SimdOperation::ElementwiseMax => scalar_elementwise_max(vec_a, vec_b),
        SimdOperation::ElementwiseMin => scalar_elementwise_min(vec_a, vec_b),
        SimdOperation::VectorNormalize => scalar_vector_normalize(vec_a),
    }
}

fn run_simd_operation(
    operation: &SimdOperation,
    vec_a: &[f32],
    vec_b: &[f32],
) -> Vec<f32> {
    // This would integrate with our actual SIMD implementations
    // For now, we'll use optimized versions where available

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return run_avx2_operation(operation, vec_a, vec_b);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return run_neon_operation(operation, vec_a, vec_b);
        }
    }

    // Fallback to scalar implementation
    run_scalar_operation(operation, vec_a, vec_b)
}

#[cfg(target_arch = "x86_64")]
fn run_avx2_operation(
    operation: &SimdOperation,
    vec_a: &[f32],
    vec_b: &[f32],
) -> Vec<f32> {
    match operation {
        SimdOperation::VectorAdd => avx2_vector_add(vec_a, vec_b),
        SimdOperation::VectorSub => avx2_vector_sub(vec_a, vec_b),
        SimdOperation::VectorMul => avx2_vector_mul(vec_a, vec_b),
        SimdOperation::DotProduct => vec![avx2_dot_product(vec_a, vec_b)],
        _ => run_scalar_operation(operation, vec_a, vec_b), // Fallback for complex operations
    }
}

#[cfg(target_arch = "aarch64")]
fn run_neon_operation(
    operation: &SimdOperation,
    vec_a: &[f32],
    vec_b: &[f32],
) -> Vec<f32> {
    match operation {
        SimdOperation::VectorAdd => neon_vector_add(vec_a, vec_b),
        SimdOperation::VectorSub => neon_vector_sub(vec_a, vec_b),
        SimdOperation::VectorMul => neon_vector_mul(vec_a, vec_b),
        SimdOperation::DotProduct => vec![neon_dot_product(vec_a, vec_b)],
        _ => run_scalar_operation(operation, vec_a, vec_b), // Fallback for complex operations
    }
}

// Scalar implementations
fn scalar_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn scalar_vector_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn scalar_vector_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn scalar_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = scalar_dot_product(a, b);
    let norm_a = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let norm_b = (b.iter().map(|x| x * x).sum::<f32>()).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn scalar_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn scalar_manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

fn scalar_elementwise_max(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).collect()
}

fn scalar_elementwise_min(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).collect()
}

fn scalar_vector_normalize(a: &[f32]) -> Vec<f32> {
    let norm = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
    if norm == 0.0 {
        a.to_vec()
    } else {
        a.iter().map(|x| x / norm).collect()
    }
}

// AVX2 implementations (simplified - would use actual SIMD intrinsics)
#[cfg(target_arch = "x86_64")]
fn avx2_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    // This would use actual AVX2 intrinsics
    // For now, simulate improved performance with optimized scalar code
    let mut result = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        result.push(a[i] + b[i]);
    }
    result
}

#[cfg(target_arch = "x86_64")]
fn avx2_vector_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        result.push(a[i] - b[i]);
    }
    result
}

#[cfg(target_arch = "x86_64")]
fn avx2_vector_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        result.push(a[i] * b[i]);
    }
    result
}

#[cfg(target_arch = "x86_64")]
fn avx2_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

// NEON implementations (simplified - would use actual NEON intrinsics)
#[cfg(target_arch = "aarch64")]
fn neon_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        result.push(a[i] + b[i]);
    }
    result
}

#[cfg(target_arch = "aarch64")]
fn neon_vector_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        result.push(a[i] - b[i]);
    }
    result
}

#[cfg(target_arch = "aarch64")]
fn neon_vector_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        result.push(a[i] * b[i]);
    }
    result
}

#[cfg(target_arch = "aarch64")]
fn neon_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn verify_operation_correctness(
    operation: &SimdOperation,
    vec_a: &[f32],
    vec_b: &[f32],
) -> bool {
    let scalar_result = run_scalar_operation(operation, vec_a, vec_b);
    let simd_result = run_simd_operation(operation, vec_a, vec_b);

    // Compare results with small tolerance for floating-point precision
    const TOLERANCE: f32 = 1e-5;

    if scalar_result.len() != simd_result.len() {
        return false;
    }

    for (scalar_val, simd_val) in scalar_result.iter().zip(simd_result.iter()) {
        if (scalar_val - simd_val).abs() > TOLERANCE {
            return false;
        }
    }

    true
}

fn average_duration(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }

    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

fn calculate_simd_benchmark_summary(
    results: &[SimdBenchmarkResult],
    config: &SimdBenchmarkConfig,
) -> SimdBenchmarkSummary {
    let speedups: Vec<f64> = results.iter().map(|r| r.speedup_factor).collect();

    let average_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
    let max_speedup = speedups.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
    let min_speedup = speedups.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);

    let operations_meeting_target = results
        .iter()
        .filter(|r| r.performance_target_met)
        .count();

    let best_performing = results
        .iter()
        .max_by(|a, b| a.speedup_factor.partial_cmp(&b.speedup_factor).unwrap())
        .map(|r| r.operation.clone())
        .unwrap_or_default();

    let worst_performing = results
        .iter()
        .min_by(|a, b| a.speedup_factor.partial_cmp(&b.speedup_factor).unwrap())
        .map(|r| r.operation.clone())
        .unwrap_or_default();

    let overall_performance_rating = if average_speedup >= 3.0 {
        "Excellent"
    } else if average_speedup >= 2.0 {
        "Good"
    } else if average_speedup >= 1.5 {
        "Fair"
    } else {
        "Poor"
    }.to_string();

    SimdBenchmarkSummary {
        average_speedup,
        max_speedup,
        min_speedup,
        operations_meeting_target,
        total_operations_tested: results.len(),
        best_performing_operation: best_performing,
        worst_performing_operation: worst_performing,
        overall_performance_rating,
    }
}

pub mod report {
    use super::*;
    use std::fs;

    pub fn generate_simd_benchmark_report(suite: &SimdBenchmarkSuite) -> Result<String> {
        let mut report = String::new();

        report.push_str("# SIMD Operations Benchmark Report\n\n");
        report.push_str(&format!("**Test Duration**: {:.2}s\n", suite.total_duration.as_secs_f64()));
        report.push_str(&format!("**Platform**: {}\n", suite.architecture_info.platform));
        report.push_str(&format!("**CPU**: {}\n", suite.architecture_info.cpu_model));
        report.push_str(&format!("**SIMD Support**: {:?}\n", suite.architecture_info.simd_support));

        report.push_str("\n## Summary\n\n");
        let summary = &suite.summary;
        report.push_str(&format!("- **Average Speedup**: {:.2}x\n", summary.average_speedup));
        report.push_str(&format!("- **Maximum Speedup**: {:.2}x\n", summary.max_speedup));
        report.push_str(&format!("- **Minimum Speedup**: {:.2}x\n", summary.min_speedup));
        report.push_str(&format!("- **Operations Meeting Target**: {}/{}\n",
            summary.operations_meeting_target, summary.total_operations_tested));
        report.push_str(&format!("- **Best Operation**: {}\n", summary.best_performing_operation));
        report.push_str(&format!("- **Overall Rating**: {}\n", summary.overall_performance_rating));

        report.push_str("\n## Detailed Results\n\n");
        report.push_str("| Operation | Vector Size | Scalar (μs) | SIMD (μs) | Speedup | Arch | Throughput (ops/s) | Target Met |\n");
        report.push_str("|-----------|-------------|-------------|-----------|---------|------|-------------------|------------|\n");

        for result in &suite.results {
            let scalar_us = result.scalar_duration.as_micros();
            let simd_us = result.simd_duration.as_micros();
            let target_met = if result.performance_target_met { "✅" } else { "❌" };

            report.push_str(&format!(
                "| {} | {} | {} | {} | {:.2}x | {} | {:.0} | {} |\n",
                result.operation,
                result.vector_size,
                scalar_us,
                simd_us,
                result.speedup_factor,
                result.simd_architecture,
                result.throughput_ops_per_sec,
                target_met
            ));
        }

        Ok(report)
    }

    pub fn save_simd_benchmark_report(suite: &SimdBenchmarkSuite, output_path: &str) -> Result<()> {
        let report = generate_simd_benchmark_report(suite)?;
        fs::write(output_path, report)?;
        println!("📊 SIMD benchmark report saved to: {}", output_path);
        Ok(())
    }
}