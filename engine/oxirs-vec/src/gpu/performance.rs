//! GPU performance monitoring and statistics

use std::time::{Duration, Instant};

/// GPU performance statistics
#[derive(Debug, Default, Clone)]
pub struct GpuPerformanceStats {
    pub total_operations: u64,
    pub total_compute_time: Duration,
    pub total_memory_transfers: u64,
    pub total_transfer_time: Duration,
    pub peak_memory_usage: usize,
    pub current_memory_usage: usize,
}

impl GpuPerformanceStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a compute operation
    pub fn record_compute_operation(&mut self, duration: Duration) {
        self.total_operations += 1;
        self.total_compute_time += duration;
    }

    /// Record a memory transfer
    pub fn record_memory_transfer(&mut self, duration: Duration) {
        self.total_memory_transfers += 1;
        self.total_transfer_time += duration;
    }

    /// Update memory usage
    pub fn update_memory_usage(&mut self, current_usage: usize) {
        self.current_memory_usage = current_usage;
        if current_usage > self.peak_memory_usage {
            self.peak_memory_usage = current_usage;
        }
    }

    /// Get average compute time per operation
    pub fn average_compute_time(&self) -> Duration {
        if self.total_operations > 0 {
            self.total_compute_time / self.total_operations as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get average transfer time
    pub fn average_transfer_time(&self) -> Duration {
        if self.total_memory_transfers > 0 {
            self.total_transfer_time / self.total_memory_transfers as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get compute throughput (operations per second)
    pub fn compute_throughput(&self) -> f64 {
        if self.total_compute_time.as_secs_f64() > 0.0 {
            self.total_operations as f64 / self.total_compute_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get memory bandwidth (bytes per second)
    pub fn memory_bandwidth(&self, total_bytes_transferred: usize) -> f64 {
        if self.total_transfer_time.as_secs_f64() > 0.0 {
            total_bytes_transferred as f64 / self.total_transfer_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get efficiency ratio (compute time / total time)
    pub fn efficiency_ratio(&self) -> f64 {
        let total_time = self.total_compute_time + self.total_transfer_time;
        if total_time.as_secs_f64() > 0.0 {
            self.total_compute_time.as_secs_f64() / total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get memory utilization ratio
    pub fn memory_utilization(&self, total_memory: usize) -> f64 {
        if total_memory > 0 {
            self.current_memory_usage as f64 / total_memory as f64
        } else {
            0.0
        }
    }
}

/// Performance timer for GPU operations
#[derive(Debug)]
pub struct GpuTimer {
    start: Instant,
    operation_type: String,
}

impl GpuTimer {
    pub fn start(operation_type: &str) -> Self {
        Self {
            start: Instant::now(),
            operation_type: operation_type.to_string(),
        }
    }

    pub fn stop(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn stop_and_record(&self, stats: &mut GpuPerformanceStats) -> Duration {
        let duration = self.stop();
        if self.operation_type.contains("transfer") {
            stats.record_memory_transfer(duration);
        } else {
            stats.record_compute_operation(duration);
        }
        duration
    }
}

/// Benchmarking utilities for GPU operations
pub struct GpuBenchmark;

impl GpuBenchmark {
    /// Benchmark a closure multiple times and return statistics
    pub fn benchmark<F>(name: &str, iterations: usize, mut operation: F) -> BenchmarkResult
    where
        F: FnMut() -> anyhow::Result<()>,
    {
        let mut times = Vec::with_capacity(iterations);
        let mut errors = 0;

        for _ in 0..iterations {
            let start = Instant::now();
            match operation() {
                Ok(_) => times.push(start.elapsed()),
                Err(_) => errors += 1,
            }
        }

        let total_time: Duration = times.iter().sum();
        let avg_time = if !times.is_empty() {
            total_time / times.len() as u32
        } else {
            Duration::ZERO
        };

        let min_time = times.iter().min().copied().unwrap_or(Duration::ZERO);
        let max_time = times.iter().max().copied().unwrap_or(Duration::ZERO);

        // Calculate standard deviation
        let avg_secs = avg_time.as_secs_f64();
        let variance: f64 = times
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - avg_secs;
                diff * diff
            })
            .sum::<f64>() / times.len() as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        BenchmarkResult {
            name: name.to_string(),
            iterations: iterations,
            successful_iterations: times.len(),
            errors,
            total_time,
            average_time: avg_time,
            min_time,
            max_time,
            std_deviation: std_dev,
        }
    }
}

/// Result of a GPU benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub successful_iterations: usize,
    pub errors: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_deviation: Duration,
}

impl BenchmarkResult {
    /// Get throughput (operations per second)
    pub fn throughput(&self) -> f64 {
        if self.total_time.as_secs_f64() > 0.0 {
            self.successful_iterations as f64 / self.total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.iterations > 0 {
            self.successful_iterations as f64 / self.iterations as f64
        } else {
            0.0
        }
    }

    /// Print benchmark results
    pub fn print(&self) {
        println!("Benchmark: {}", self.name);
        println!("  Iterations: {} (success: {}, errors: {})", 
                 self.iterations, self.successful_iterations, self.errors);
        println!("  Total time: {:?}", self.total_time);
        println!("  Average time: {:?}", self.average_time);
        println!("  Min/Max time: {:?} / {:?}", self.min_time, self.max_time);
        println!("  Std deviation: {:?}", self.std_deviation);
        println!("  Throughput: {:.2} ops/sec", self.throughput());
        println!("  Success rate: {:.2}%", self.success_rate() * 100.0);
    }
}
