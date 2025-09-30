//! Memory Performance Profiler
//!
//! Monitors memory usage, allocation patterns, and cache performance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use sysinfo::{System, SystemExt};

/// Memory profiler for monitoring usage and allocation patterns
#[derive(Debug)]
pub struct MemoryProfiler {
    system: System,
    operation_metrics: HashMap<String, MemoryOperationMetrics>,
    global_metrics: MemoryGlobalMetrics,
    start_time: Instant,
    baseline_memory: u64,
}

/// Per-operation memory metrics
#[derive(Debug, Clone)]
pub struct MemoryOperationMetrics {
    pub start_time: Instant,
    pub start_memory_mb: u64,
    pub peak_memory_mb: u64,
    pub total_allocated_mb: u64,
    pub total_deallocated_mb: u64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub page_faults: u64,
}

/// Global memory metrics
#[derive(Debug, Default)]
pub struct MemoryGlobalMetrics {
    pub total_operations: AtomicU64,
    pub peak_usage_mb: AtomicU64,
    pub total_allocated_mb: AtomicU64,
    pub total_deallocated_mb: AtomicU64,
    pub total_cache_hits: AtomicU64,
    pub total_cache_misses: AtomicU64,
    pub memory_leaks_detected: AtomicU64,
}

/// Memory performance report
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryReport {
    pub duration: Duration,
    pub total_available_mb: u64,
    pub peak_usage_mb: u64,
    pub average_usage_mb: f64,
    pub total_operations: u64,
    pub cache_hit_rate: f32,
    pub allocation_efficiency: f32,
    pub fragmentation_score: f32,
    pub memory_leaks: u64,
    pub operation_breakdown: HashMap<String, MemoryOperationSummary>,
    pub memory_recommendations: Vec<String>,
}

/// Summary of memory metrics for an operation
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryOperationSummary {
    pub peak_usage_mb: u64,
    pub total_allocated_mb: u64,
    pub allocation_count: u64,
    pub cache_hit_rate: f32,
    pub memory_efficiency: f32,
    pub allocation_pattern: AllocationPattern,
}

/// Memory allocation pattern classification
#[derive(Debug, Serialize, Deserialize)]
pub enum AllocationPattern {
    Steady,      // Consistent allocations
    Burst,       // Large allocations in short time
    Gradual,     // Slowly increasing allocations
    Fragmented,  // Many small allocations
    Optimal,     // Well-optimized pattern
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_memory();

        let baseline_memory = system.used_memory() / 1024 / 1024; // Convert to MB

        Self {
            system,
            operation_metrics: HashMap::new(),
            global_metrics: MemoryGlobalMetrics::default(),
            start_time: Instant::now(),
            baseline_memory,
        }
    }

    /// Start monitoring an operation
    pub fn start_operation(&mut self, operation_name: &str) {
        self.system.refresh_memory();
        let current_memory = self.system.used_memory() / 1024 / 1024; // Convert to MB

        let metrics = MemoryOperationMetrics {
            start_time: Instant::now(),
            start_memory_mb: current_memory,
            peak_memory_mb: current_memory,
            total_allocated_mb: 0,
            total_deallocated_mb: 0,
            allocation_count: 0,
            deallocation_count: 0,
            cache_hits: 0,
            cache_misses: 0,
            page_faults: 0,
        };

        self.operation_metrics.insert(operation_name.to_string(), metrics);
    }

    /// Finish monitoring an operation
    pub fn finish_operation(&mut self, operation_name: &str) -> MemoryOperationMetrics {
        if let Some(mut metrics) = self.operation_metrics.remove(operation_name) {
            self.system.refresh_memory();
            let current_memory = self.system.used_memory() / 1024 / 1024;

            metrics.peak_memory_mb = metrics.peak_memory_mb.max(current_memory);

            // Update global metrics
            self.global_metrics.total_operations.fetch_add(1, Ordering::Relaxed);
            self.global_metrics.peak_usage_mb.fetch_max(metrics.peak_memory_mb, Ordering::Relaxed);
            self.global_metrics.total_allocated_mb.fetch_add(metrics.total_allocated_mb, Ordering::Relaxed);
            self.global_metrics.total_deallocated_mb.fetch_add(metrics.total_deallocated_mb, Ordering::Relaxed);
            self.global_metrics.total_cache_hits.fetch_add(metrics.cache_hits as u64, Ordering::Relaxed);
            self.global_metrics.total_cache_misses.fetch_add(metrics.cache_misses as u64, Ordering::Relaxed);

            // Check for potential memory leaks
            if metrics.total_allocated_mb > metrics.total_deallocated_mb + 10 { // 10MB threshold
                self.global_metrics.memory_leaks_detected.fetch_add(1, Ordering::Relaxed);
            }

            metrics
        } else {
            // Return default metrics if operation not found
            MemoryOperationMetrics {
                start_time: Instant::now(),
                start_memory_mb: 0,
                peak_memory_mb: 0,
                total_allocated_mb: 0,
                total_deallocated_mb: 0,
                allocation_count: 0,
                deallocation_count: 0,
                cache_hits: 0,
                cache_misses: 0,
                page_faults: 0,
            }
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&mut self, operation_name: &str, size_mb: u64) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.total_allocated_mb += size_mb;
            metrics.allocation_count += 1;

            // Update peak memory if needed
            self.system.refresh_memory();
            let current_memory = self.system.used_memory() / 1024 / 1024;
            metrics.peak_memory_mb = metrics.peak_memory_mb.max(current_memory);
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&mut self, operation_name: &str, size_mb: u64) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.total_deallocated_mb += size_mb;
            metrics.deallocation_count += 1;
        }
    }

    /// Record cache hit
    pub fn record_cache_hit(&mut self, operation_name: &str) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.cache_hits += 1;
        }
    }

    /// Record cache miss
    pub fn record_cache_miss(&mut self, operation_name: &str) {
        if let Some(metrics) = self.operation_metrics.get_mut(operation_name) {
            metrics.cache_misses += 1;
        }
    }

    /// Generate comprehensive memory report
    pub fn generate_report(&mut self) -> MemoryReport {
        self.system.refresh_memory();

        let runtime = self.start_time.elapsed();
        let total_memory = self.system.total_memory() / 1024 / 1024; // Convert to MB
        let current_memory = self.system.used_memory() / 1024 / 1024;
        let peak_usage = self.global_metrics.peak_usage_mb.load(Ordering::Relaxed);
        let total_operations = self.global_metrics.total_operations.load(Ordering::Relaxed);

        // Calculate averages
        let average_usage = if total_operations > 0 {
            (self.baseline_memory + current_memory) as f64 / 2.0
        } else {
            current_memory as f64
        };

        // Calculate cache hit rate
        let total_hits = self.global_metrics.total_cache_hits.load(Ordering::Relaxed);
        let total_misses = self.global_metrics.total_cache_misses.load(Ordering::Relaxed);
        let cache_hit_rate = if total_hits + total_misses > 0 {
            total_hits as f32 / (total_hits + total_misses) as f32
        } else {
            0.0
        };

        // Calculate allocation efficiency
        let total_allocated = self.global_metrics.total_allocated_mb.load(Ordering::Relaxed);
        let total_deallocated = self.global_metrics.total_deallocated_mb.load(Ordering::Relaxed);
        let allocation_efficiency = if total_allocated > 0 {
            total_deallocated as f32 / total_allocated as f32
        } else {
            1.0
        };

        // Calculate fragmentation score (simplified)
        let fragmentation_score = self.calculate_fragmentation_score();

        // Generate operation summaries
        let operation_breakdown: HashMap<String, MemoryOperationSummary> = self.operation_metrics.iter()
            .map(|(name, metrics)| {
                let cache_hit_rate = if metrics.cache_hits + metrics.cache_misses > 0 {
                    metrics.cache_hits as f32 / (metrics.cache_hits + metrics.cache_misses) as f32
                } else {
                    0.0
                };

                let memory_efficiency = if metrics.total_allocated_mb > 0 {
                    metrics.total_deallocated_mb as f32 / metrics.total_allocated_mb as f32
                } else {
                    1.0
                };

                let allocation_pattern = Self::classify_allocation_pattern(metrics);

                let summary = MemoryOperationSummary {
                    peak_usage_mb: metrics.peak_memory_mb,
                    total_allocated_mb: metrics.total_allocated_mb,
                    allocation_count: metrics.allocation_count,
                    cache_hit_rate,
                    memory_efficiency,
                    allocation_pattern,
                };
                (name.clone(), summary)
            })
            .collect();

        let memory_leaks = self.global_metrics.memory_leaks_detected.load(Ordering::Relaxed);
        let recommendations = self.generate_recommendations(cache_hit_rate, allocation_efficiency, memory_leaks);

        MemoryReport {
            duration: runtime,
            total_available_mb: total_memory,
            peak_usage_mb: peak_usage,
            average_usage_mb: average_usage,
            total_operations,
            cache_hit_rate,
            allocation_efficiency,
            fragmentation_score,
            memory_leaks,
            operation_breakdown,
            memory_recommendations: recommendations,
        }
    }

    /// Calculate fragmentation score (simplified approach)
    fn calculate_fragmentation_score(&self) -> f32 {
        // This is a simplified fragmentation calculation
        // In practice, you'd need more sophisticated memory layout analysis

        let total_allocations: u64 = self.operation_metrics.values()
            .map(|m| m.allocation_count)
            .sum();

        let total_allocated: u64 = self.operation_metrics.values()
            .map(|m| m.total_allocated_mb)
            .sum();

        if total_allocations > 0 && total_allocated > 0 {
            let avg_allocation_size = total_allocated as f32 / total_allocations as f32;

            // Lower average allocation size indicates more fragmentation
            if avg_allocation_size < 1.0 {
                0.8 // High fragmentation
            } else if avg_allocation_size < 10.0 {
                0.5 // Medium fragmentation
            } else {
                0.2 // Low fragmentation
            }
        } else {
            0.0
        }
    }

    /// Classify allocation pattern for an operation
    fn classify_allocation_pattern(metrics: &MemoryOperationMetrics) -> AllocationPattern {
        let duration = metrics.start_time.elapsed();
        let allocation_rate = if duration.as_secs() > 0 {
            metrics.allocation_count as f64 / duration.as_secs() as f64
        } else {
            0.0
        };

        let avg_allocation_size = if metrics.allocation_count > 0 {
            metrics.total_allocated_mb as f64 / metrics.allocation_count as f64
        } else {
            0.0
        };

        if allocation_rate > 100.0 {
            AllocationPattern::Burst
        } else if avg_allocation_size < 1.0 && metrics.allocation_count > 1000 {
            AllocationPattern::Fragmented
        } else if allocation_rate > 10.0 && allocation_rate <= 100.0 {
            AllocationPattern::Gradual
        } else if allocation_rate > 0.0 && allocation_rate <= 10.0 {
            AllocationPattern::Steady
        } else {
            AllocationPattern::Optimal
        }
    }

    /// Generate memory optimization recommendations
    fn generate_recommendations(&self, cache_hit_rate: f32, allocation_efficiency: f32, memory_leaks: u64) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Cache recommendations
        if cache_hit_rate < 0.7 {
            recommendations.push("Low cache hit rate detected. Consider optimizing data access patterns or increasing cache size.".to_string());
        }

        // Allocation efficiency recommendations
        if allocation_efficiency < 0.8 {
            recommendations.push("Low allocation efficiency detected. Check for memory leaks or optimize memory management.".to_string());
        }

        // Memory leak recommendations
        if memory_leaks > 0 {
            recommendations.push(format!("Potential memory leaks detected in {} operations. Review memory cleanup code.", memory_leaks));
        }

        // Memory usage recommendations
        self.system.refresh_memory();
        let memory_usage_percent = (self.system.used_memory() as f32 / self.system.total_memory() as f32) * 100.0;

        if memory_usage_percent > 90.0 {
            recommendations.push("High memory usage detected. Consider reducing batch sizes or implementing memory pooling.".to_string());
        } else if memory_usage_percent < 30.0 {
            recommendations.push("Low memory utilization. Consider increasing batch sizes for better performance.".to_string());
        }

        // Fragmentation recommendations
        let fragmentation_score = self.calculate_fragmentation_score();
        if fragmentation_score > 0.6 {
            recommendations.push("High memory fragmentation detected. Consider using memory pools or larger allocation blocks.".to_string());
        }

        recommendations
    }

    /// Reset all profiling data
    pub fn reset(&mut self) {
        self.operation_metrics.clear();
        self.global_metrics = MemoryGlobalMetrics::default();
        self.start_time = Instant::now();

        self.system.refresh_memory();
        self.baseline_memory = self.system.used_memory() / 1024 / 1024;
    }

    /// Get real-time memory statistics
    pub fn get_realtime_stats(&mut self) -> RealtimeMemoryStats {
        self.system.refresh_memory();

        let total_memory = self.system.total_memory() / 1024 / 1024;
        let used_memory = self.system.used_memory() / 1024 / 1024;
        let available_memory = self.system.available_memory() / 1024 / 1024;

        RealtimeMemoryStats {
            total_memory_mb: total_memory,
            used_memory_mb: used_memory,
            available_memory_mb: available_memory,
            usage_percentage: (used_memory as f32 / total_memory as f32) * 100.0,
            active_operations: self.operation_metrics.len(),
            total_operations: self.global_metrics.total_operations.load(Ordering::Relaxed),
        }
    }

    /// Check if system is memory-bound
    pub fn is_memory_bound(&mut self) -> bool {
        self.system.refresh_memory();
        let usage_percent = (self.system.used_memory() as f32 / self.system.total_memory() as f32) * 100.0;
        usage_percent > 90.0
    }

    /// Get memory usage trend
    pub fn get_memory_trend(&self) -> MemoryTrend {
        let current_usage = self.global_metrics.peak_usage_mb.load(Ordering::Relaxed);

        if current_usage > self.baseline_memory + 100 {
            MemoryTrend::Increasing
        } else if current_usage < self.baseline_memory.saturating_sub(100) {
            MemoryTrend::Decreasing
        } else {
            MemoryTrend::Stable
        }
    }
}

/// Real-time memory statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct RealtimeMemoryStats {
    pub total_memory_mb: u64,
    pub used_memory_mb: u64,
    pub available_memory_mb: u64,
    pub usage_percentage: f32,
    pub active_operations: usize,
    pub total_operations: u64,
}

/// Memory usage trend
#[derive(Debug, Serialize, Deserialize)]
pub enum MemoryTrend {
    Increasing,
    Decreasing,
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = MemoryProfiler::new();
        assert!(profiler.baseline_memory > 0);
    }

    #[test]
    fn test_operation_profiling() {
        let mut profiler = MemoryProfiler::new();

        profiler.start_operation("test_op");
        thread::sleep(Duration::from_millis(10));
        let metrics = profiler.finish_operation("test_op");

        assert!(metrics.start_memory_mb > 0);
        assert!(metrics.peak_memory_mb >= metrics.start_memory_mb);
    }

    #[test]
    fn test_allocation_recording() {
        let mut profiler = MemoryProfiler::new();

        profiler.start_operation("alloc_test");
        profiler.record_allocation("alloc_test", 100);
        profiler.record_deallocation("alloc_test", 50);
        let metrics = profiler.finish_operation("alloc_test");

        assert_eq!(metrics.total_allocated_mb, 100);
        assert_eq!(metrics.total_deallocated_mb, 50);
        assert_eq!(metrics.allocation_count, 1);
        assert_eq!(metrics.deallocation_count, 1);
    }

    #[test]
    fn test_cache_recording() {
        let mut profiler = MemoryProfiler::new();

        profiler.start_operation("cache_test");
        profiler.record_cache_hit("cache_test");
        profiler.record_cache_hit("cache_test");
        profiler.record_cache_miss("cache_test");
        let metrics = profiler.finish_operation("cache_test");

        assert_eq!(metrics.cache_hits, 2);
        assert_eq!(metrics.cache_misses, 1);
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = MemoryProfiler::new();

        profiler.start_operation("test_op");
        profiler.record_allocation("test_op", 50);
        profiler.record_cache_hit("test_op");
        thread::sleep(Duration::from_millis(10));
        profiler.finish_operation("test_op");

        let report = profiler.generate_report();
        assert_eq!(report.total_operations, 1);
        assert!(report.duration >= Duration::from_millis(10));
        assert!(report.total_available_mb > 0);
    }

    #[test]
    fn test_allocation_pattern_classification() {
        let metrics = MemoryOperationMetrics {
            start_time: Instant::now(),
            start_memory_mb: 100,
            peak_memory_mb: 150,
            total_allocated_mb: 1000,
            total_deallocated_mb: 950,
            allocation_count: 2000,
            deallocation_count: 1900,
            cache_hits: 100,
            cache_misses: 20,
            page_faults: 5,
        };

        let pattern = MemoryProfiler::classify_allocation_pattern(&metrics);
        assert!(matches!(pattern, AllocationPattern::Fragmented));
    }
}