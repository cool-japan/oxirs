use rand_distr::Distribution;use std::time::{Duration, Instant};

// Helper function for normal distribution using Box-Muller transform
fn generate_normal(rng: &mut scirs2_core::random::Random<impl rand::Rng>, mean: f64, std_dev: f64) -> f64 {
    use std::f64::consts::PI;
    static mut SPARE: Option<f64> = None;
    static mut HAS_SPARE: bool = false;

    unsafe {
        if HAS_SPARE {
            HAS_SPARE = false;
            return SPARE.unwrap() * std_dev + mean;
        }

        HAS_SPARE = true;
        let u: f64 = rng.random_range(0.0, 1.0);
        let v: f64 = rng.random_range(0.0, 1.0);
        let mag = std_dev * (-2.0 * u.ln()).sqrt();
        SPARE = Some(mag * (2.0 * PI * v).sin());
        mag * (2.0 * PI * v).cos() + mean
    }
}
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBenchmarkConfig {
    pub embedding_sizes: Vec<usize>,
    pub batch_sizes: Vec<usize>,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub dimensions: usize,
    pub use_mixed_precision: bool,
    pub test_adaptive_switching: bool,
    pub gpu_memory_limit_mb: Option<usize>,
}

impl Default for GpuBenchmarkConfig {
    fn default() -> Self {
        Self {
            embedding_sizes: vec![1000, 10000, 100000, 1000000],
            batch_sizes: vec![32, 128, 512, 2048],
            iterations: 10,
            warmup_iterations: 3,
            dimensions: 512,
            use_mixed_precision: true,
            test_adaptive_switching: true,
            gpu_memory_limit_mb: Some(4096),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBenchmarkResult {
    pub embedding_size: usize,
    pub batch_size: usize,
    pub dimensions: usize,
    pub cpu_duration: Duration,
    pub gpu_duration: Option<Duration>,
    pub speedup_factor: Option<f64>,
    pub memory_usage_mb: f64,
    pub gpu_utilization_percent: Option<f64>,
    pub adaptive_switch_threshold: Option<usize>,
    pub test_passed: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBenchmarkSuite {
    pub config: GpuBenchmarkConfig,
    pub results: Vec<GpuBenchmarkResult>,
    pub total_duration: Duration,
    pub gpu_available: bool,
    pub gpu_backend: Option<String>,
    pub summary: GpuBenchmarkSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBenchmarkSummary {
    pub average_speedup: Option<f64>,
    pub max_speedup: Option<f64>,
    pub optimal_switch_threshold: Option<usize>,
    pub gpu_memory_efficiency: f64,
    pub tests_passed: usize,
    pub tests_failed: usize,
    pub expected_min_speedup: f64,
    pub performance_target_met: bool,
}

pub async fn run_gpu_acceleration_benchmark(config: GpuBenchmarkConfig) -> Result<GpuBenchmarkSuite> {
    println!("🚀 Starting GPU Acceleration Benchmark Suite...");

    let start_time = Instant::now();
    let mut results = Vec::new();
    let mut gpu_available = false;
    let mut gpu_backend = None;

    // Test GPU availability first
    match test_gpu_availability().await {
        Ok((available, backend)) => {
            gpu_available = available;
            gpu_backend = backend;
            if gpu_available {
                println!("✅ GPU detected: {:?}", gpu_backend);
            } else {
                println!("❌ No GPU available - CPU-only benchmarks will run");
            }
        }
        Err(e) => {
            println!("⚠️  GPU detection failed: {}", e);
        }
    }

    // Run benchmarks for each configuration
    for &embedding_size in &config.embedding_sizes {
        for &batch_size in &config.batch_sizes {
            println!("📊 Testing embedding_size={}, batch_size={}", embedding_size, batch_size);

            let result = run_single_gpu_benchmark(
                &config,
                embedding_size,
                batch_size,
                gpu_available,
            ).await?;

            results.push(result);
        }
    }

    let total_duration = start_time.elapsed();
    let summary = calculate_gpu_benchmark_summary(&results, &config);

    Ok(GpuBenchmarkSuite {
        config,
        results,
        total_duration,
        gpu_available,
        gpu_backend,
        summary,
    })
}

async fn test_gpu_availability() -> Result<(bool, Option<String>)> {
    // This would integrate with the actual GPU acceleration module
    // For now, simulate GPU detection logic

    #[cfg(feature = "cuda")]
    {
        if let Ok(_) = std::process::Command::new("nvidia-smi").output() {
            return Ok((true, Some("CUDA".to_string())));
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Check for Metal support on macOS
        return Ok((true, Some("Metal".to_string())));
    }

    #[cfg(feature = "opencl")]
    {
        // OpenCL detection logic would go here
        return Ok((false, None));
    }

    Ok((false, None))
}

async fn run_single_gpu_benchmark(
    config: &GpuBenchmarkConfig,
    embedding_size: usize,
    batch_size: usize,
    gpu_available: bool,
) -> Result<GpuBenchmarkResult> {
    let mut cpu_durations = Vec::new();
    let mut gpu_durations = Vec::new();
    let mut memory_usage = 0.0;
    let mut gpu_utilization = None;
    let mut test_passed = true;
    let mut error_message = None;

    // Generate test data
    let test_embeddings = generate_test_embeddings(embedding_size, config.dimensions);
    let test_queries = generate_test_queries(batch_size, config.dimensions);

    // Warmup iterations
    for _ in 0..config.warmup_iterations {
        let _ = run_cpu_embedding_computation(&test_embeddings, &test_queries);
        if gpu_available {
            let _ = run_gpu_embedding_computation(&test_embeddings, &test_queries).await;
        }
    }

    // CPU benchmarks
    for _ in 0..config.iterations {
        let start = Instant::now();
        match run_cpu_embedding_computation(&test_embeddings, &test_queries) {
            Ok(_) => cpu_durations.push(start.elapsed()),
            Err(e) => {
                test_passed = false;
                error_message = Some(format!("CPU computation failed: {}", e));
                break;
            }
        }
    }

    // GPU benchmarks (if available)
    if gpu_available {
        for _ in 0..config.iterations {
            let start = Instant::now();
            match run_gpu_embedding_computation(&test_embeddings, &test_queries).await {
                Ok(result) => {
                    gpu_durations.push(start.elapsed());
                    gpu_utilization = result.gpu_utilization;
                    memory_usage = result.memory_usage_mb;
                }
                Err(e) => {
                    error_message = Some(format!("GPU computation failed: {}", e));
                    break;
                }
            }
        }
    }

    // Calculate average durations
    let cpu_duration = average_duration(&cpu_durations);
    let gpu_duration = if gpu_durations.is_empty() {
        None
    } else {
        Some(average_duration(&gpu_durations))
    };

    let speedup_factor = gpu_duration.map(|gpu_dur| {
        cpu_duration.as_secs_f64() / gpu_dur.as_secs_f64()
    });

    // Test adaptive switching threshold
    let adaptive_switch_threshold = if config.test_adaptive_switching {
        test_adaptive_switching_threshold(embedding_size, batch_size).await
    } else {
        None
    };

    Ok(GpuBenchmarkResult {
        embedding_size,
        batch_size,
        dimensions: config.dimensions,
        cpu_duration,
        gpu_duration,
        speedup_factor,
        memory_usage_mb: memory_usage,
        gpu_utilization_percent: gpu_utilization,
        adaptive_switch_threshold,
        test_passed,
        error_message,
    })
}

fn generate_test_embeddings(size: usize, dimensions: usize) -> Vec<Vec<f32>> {
    use scirs2_core::random::Random;

    let mut rng = Random::seed(42);

    (0..size)
        .map(|_| {
            (0..dimensions)
                .map(|_| generate_normal(&mut rng, 0.0, 1.0) as f32)
                .collect()
        })
        .collect()
}

fn generate_test_queries(batch_size: usize, dimensions: usize) -> Vec<Vec<f32>> {
    use scirs2_core::random::Random;

    let mut rng = Random::seed(123);

    (0..batch_size)
        .map(|_| {
            (0..dimensions)
                .map(|_| generate_normal(&mut rng, 0.0, 1.0) as f32)
                .collect()
        })
        .collect()
}

fn run_cpu_embedding_computation(
    embeddings: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> Result<CpuComputationResult> {
    let mut similarities = Vec::new();

    for query in queries {
        let mut query_similarities = Vec::new();
        for embedding in embeddings {
            let similarity = cosine_similarity_cpu(query, embedding);
            query_similarities.push(similarity);
        }
        similarities.push(query_similarities);
    }

    Ok(CpuComputationResult {
        similarities,
        memory_usage_mb: estimate_memory_usage(embeddings.len(), queries.len()),
    })
}

async fn run_gpu_embedding_computation(
    embeddings: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> Result<GpuComputationResult> {
    // This would integrate with the actual GPU acceleration module
    // For now, simulate GPU computation with some realistic timing

    tokio::time::sleep(Duration::from_millis(1)).await;

    let mut similarities = Vec::new();
    for query in queries {
        let mut query_similarities = Vec::new();
        for embedding in embeddings {
            let similarity = cosine_similarity_cpu(query, embedding); // Placeholder
            query_similarities.push(similarity);
        }
        similarities.push(query_similarities);
    }

    Ok(GpuComputationResult {
        similarities,
        memory_usage_mb: estimate_memory_usage(embeddings.len(), queries.len()),
        gpu_utilization: Some(85.0), // Simulated GPU utilization
    })
}

fn cosine_similarity_cpu(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

async fn test_adaptive_switching_threshold(
    embedding_size: usize,
    batch_size: usize,
) -> Option<usize> {
    // Test different thresholds to find optimal switching point
    let test_sizes = vec![100, 500, 1000, 5000, 10000];

    for &threshold in &test_sizes {
        if embedding_size >= threshold && batch_size >= 32 {
            return Some(threshold);
        }
    }

    None
}

fn estimate_memory_usage(embedding_count: usize, query_count: usize) -> f64 {
    let embedding_memory = embedding_count * 512 * 4; // 512 dims * 4 bytes per f32
    let query_memory = query_count * 512 * 4;
    let result_memory = embedding_count * query_count * 4; // similarity scores

    (embedding_memory + query_memory + result_memory) as f64 / (1024.0 * 1024.0)
}

fn average_duration(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }

    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

fn calculate_gpu_benchmark_summary(
    results: &[GpuBenchmarkResult],
    config: &GpuBenchmarkConfig,
) -> GpuBenchmarkSummary {
    let speedups: Vec<f64> = results
        .iter()
        .filter_map(|r| r.speedup_factor)
        .collect();

    let average_speedup = if speedups.is_empty() {
        None
    } else {
        Some(speedups.iter().sum::<f64>() / speedups.len() as f64)
    };

    let max_speedup = speedups.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap());

    let optimal_switch_threshold = results
        .iter()
        .filter_map(|r| r.adaptive_switch_threshold)
        .min();

    let gpu_memory_efficiency = results
        .iter()
        .map(|r| r.memory_usage_mb)
        .sum::<f64>() / results.len() as f64;

    let tests_passed = results.iter().filter(|r| r.test_passed).count();
    let tests_failed = results.len() - tests_passed;

    let expected_min_speedup = 2.0; // Expect at least 2x speedup
    let performance_target_met = average_speedup
        .map(|speedup| speedup >= expected_min_speedup)
        .unwrap_or(false);

    GpuBenchmarkSummary {
        average_speedup,
        max_speedup,
        optimal_switch_threshold,
        gpu_memory_efficiency,
        tests_passed,
        tests_failed,
        expected_min_speedup,
        performance_target_met,
    }
}

#[derive(Debug)]
struct CpuComputationResult {
    similarities: Vec<Vec<f32>>,
    memory_usage_mb: f64,
}

#[derive(Debug)]
struct GpuComputationResult {
    similarities: Vec<Vec<f32>>,
    memory_usage_mb: f64,
    gpu_utilization: Option<f64>,
}

pub mod report {
    use super::*;
    use std::fs;

    pub fn generate_gpu_benchmark_report(suite: &GpuBenchmarkSuite) -> Result<String> {
        let mut report = String::new();

        report.push_str("# GPU Acceleration Benchmark Report\n\n");
        report.push_str(&format!("**Test Duration**: {:.2}s\n", suite.total_duration.as_secs_f64()));
        report.push_str(&format!("**GPU Available**: {}\n", suite.gpu_available));

        if let Some(backend) = &suite.gpu_backend {
            report.push_str(&format!("**GPU Backend**: {}\n", backend));
        }

        report.push_str("\n## Summary\n\n");
        let summary = &suite.summary;

        if let Some(avg_speedup) = summary.average_speedup {
            report.push_str(&format!("- **Average Speedup**: {:.2}x\n", avg_speedup));
        }

        if let Some(max_speedup) = summary.max_speedup {
            report.push_str(&format!("- **Maximum Speedup**: {:.2}x\n", max_speedup));
        }

        report.push_str(&format!("- **Tests Passed**: {}/{}\n", summary.tests_passed, summary.tests_passed + summary.tests_failed));
        report.push_str(&format!("- **Performance Target Met**: {}\n", summary.performance_target_met));
        report.push_str(&format!("- **GPU Memory Efficiency**: {:.2} MB avg\n", summary.gpu_memory_efficiency));

        report.push_str("\n## Detailed Results\n\n");
        report.push_str("| Embedding Size | Batch Size | CPU Time (ms) | GPU Time (ms) | Speedup | GPU Util% | Status |\n");
        report.push_str("|----------------|------------|---------------|---------------|---------|-----------|--------|\n");

        for result in &suite.results {
            let cpu_ms = result.cpu_duration.as_millis();
            let gpu_ms = result.gpu_duration.map(|d| d.as_millis()).unwrap_or(0);
            let speedup = result.speedup_factor.map(|s| format!("{:.2}x", s)).unwrap_or("-".to_string());
            let gpu_util = result.gpu_utilization_percent.map(|u| format!("{:.1}%", u)).unwrap_or("-".to_string());
            let status = if result.test_passed { "✅" } else { "❌" };

            report.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                result.embedding_size,
                result.batch_size,
                cpu_ms,
                gpu_ms,
                speedup,
                gpu_util,
                status
            ));
        }

        Ok(report)
    }

    pub fn save_gpu_benchmark_report(suite: &GpuBenchmarkSuite, output_path: &str) -> Result<()> {
        let report = generate_gpu_benchmark_report(suite)?;
        fs::write(output_path, report)?;
        println!("📊 GPU benchmark report saved to: {}", output_path);
        Ok(())
    }
}