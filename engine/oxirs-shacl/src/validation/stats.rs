//! Validation statistics and performance tracking

use std::collections::HashMap;
use std::time::Duration;

/// Statistics for validation operations
#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    pub total_validations: usize,
    pub total_node_validations: usize,
    pub total_constraint_evaluations: usize,
    pub total_validation_time: Duration,
    pub last_validation_time: Option<Duration>,
    pub avg_validation_time: Duration,
    pub constraint_evaluation_times: HashMap<String, Duration>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl ValidationStats {
    /// Record a constraint evaluation with its duration
    pub fn record_constraint_evaluation(&mut self, constraint_type: String, duration: Duration) {
        self.total_constraint_evaluations += 1;
        *self
            .constraint_evaluation_times
            .entry(constraint_type)
            .or_insert(Duration::ZERO) += duration;

        if self.total_validations > 0 {
            self.avg_validation_time = self.total_validation_time / self.total_validations as u32;
        }
    }

    /// Get average constraint evaluation time for a specific constraint type
    pub fn get_avg_constraint_time(&self, constraint_type: &str) -> Option<Duration> {
        self.constraint_evaluation_times
            .get(constraint_type)
            .map(|total| *total / self.total_constraint_evaluations as u32)
    }

    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_accesses as f64
        }
    }

    /// Record a cache hit
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record a cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.total_validations = 0;
        self.total_node_validations = 0;
        self.total_constraint_evaluations = 0;
        self.total_validation_time = Duration::ZERO;
        self.last_validation_time = None;
        self.avg_validation_time = Duration::ZERO;
        self.constraint_evaluation_times.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

/// Performance statistics for qualified value shape constraint validation
#[derive(Debug, Clone, Default)]
pub struct QualifiedValidationStats {
    /// Total number of value validations performed
    total_validations: usize,

    /// Total time spent on validations
    total_validation_time: Duration,

    /// Number of conforming validations
    conforming_validations: usize,

    /// Number of non-conforming validations
    non_conforming_validations: usize,

    /// Individual validation times for performance analysis
    validation_times: Vec<Duration>,
}

impl QualifiedValidationStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a validation result and its timing
    pub fn record_validation(&mut self, validation_time: Duration, conforms: bool) {
        self.total_validations += 1;
        self.total_validation_time += validation_time;
        self.validation_times.push(validation_time);

        if conforms {
            self.conforming_validations += 1;
        } else {
            self.non_conforming_validations += 1;
        }
    }

    /// Get average validation time in milliseconds
    pub fn average_validation_time_ms(&self) -> f64 {
        if self.total_validations == 0 {
            0.0
        } else {
            self.total_validation_time.as_secs_f64() * 1000.0 / self.total_validations as f64
        }
    }

    /// Get conformance rate (0.0 to 1.0)
    pub fn conformance_rate(&self) -> f64 {
        if self.total_validations == 0 {
            0.0
        } else {
            self.conforming_validations as f64 / self.total_validations as f64
        }
    }

    /// Get total validation count
    pub fn total_validations(&self) -> usize {
        self.total_validations
    }

    /// Get total validation time
    pub fn total_validation_time(&self) -> Duration {
        self.total_validation_time
    }

    /// Get median validation time (requires sorting, so can be expensive)
    pub fn median_validation_time_ms(&self) -> f64 {
        if self.validation_times.is_empty() {
            return 0.0;
        }

        let mut times = self.validation_times.clone();
        times.sort();

        let len = times.len();
        if len % 2 == 0 {
            let mid1 = times[len / 2 - 1].as_secs_f64() * 1000.0;
            let mid2 = times[len / 2].as_secs_f64() * 1000.0;
            (mid1 + mid2) / 2.0
        } else {
            times[len / 2].as_secs_f64() * 1000.0
        }
    }

    /// Get percentile validation time (p should be between 0.0 and 1.0)
    pub fn percentile_validation_time_ms(&self, p: f64) -> f64 {
        if self.validation_times.is_empty() || p < 0.0 || p > 1.0 {
            return 0.0;
        }

        let mut times = self.validation_times.clone();
        times.sort();

        let index = ((times.len() - 1) as f64 * p).round() as usize;
        times[index].as_secs_f64() * 1000.0
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.total_validations = 0;
        self.total_validation_time = Duration::ZERO;
        self.conforming_validations = 0;
        self.non_conforming_validations = 0;
        self.validation_times.clear();
    }

    /// Get performance summary as a formatted string
    pub fn summary(&self) -> String {
        format!(
            "QualifiedValidationStats {{ total: {}, conformance_rate: {:.2}%, avg_time: {:.2}ms, median_time: {:.2}ms }}",
            self.total_validations,
            self.conformance_rate() * 100.0,
            self.average_validation_time_ms(),
            self.median_validation_time_ms()
        )
    }
}