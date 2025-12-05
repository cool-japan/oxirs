//! Performance profiling utilities for identifying bottlenecks
//!
//! This module provides lightweight profiling helpers for measuring
//! and analyzing spatial operation performance in production and development.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A lightweight performance profiler for spatial operations
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::performance::profiling::Profiler;
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::geometric_properties::area;
///
/// let mut profiler = Profiler::new();
///
/// // Profile geometry parsing
/// profiler.start("parse");
/// let geom = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
/// profiler.stop("parse");
///
/// // Profile area calculation
/// profiler.start("area");
/// let _area = area(&geom).unwrap();
/// profiler.stop("area");
///
/// // Print report
/// profiler.print_report();
/// ```
#[derive(Debug, Clone)]
pub struct Profiler {
    timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    active: Arc<Mutex<HashMap<String, Instant>>>,
}

impl Profiler {
    /// Create a new profiler instance
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(HashMap::new())),
            active: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start profiling a named section
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::profiling::Profiler;
    ///
    /// let mut profiler = Profiler::new();
    /// profiler.start("my_operation");
    /// // ... do work ...
    /// profiler.stop("my_operation");
    /// ```
    pub fn start(&mut self, name: &str) {
        let mut active = self.active.lock().unwrap();
        active.insert(name.to_string(), Instant::now());
    }

    /// Stop profiling a named section and record the duration
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::profiling::Profiler;
    ///
    /// let mut profiler = Profiler::new();
    /// profiler.start("parsing");
    /// // ... parse data ...
    /// profiler.stop("parsing");
    /// ```
    pub fn stop(&mut self, name: &str) {
        let mut active = self.active.lock().unwrap();
        if let Some(start_time) = active.remove(name) {
            let duration = start_time.elapsed();
            let mut timings = self.timings.lock().unwrap();
            timings.entry(name.to_string()).or_default().push(duration);
        }
    }

    /// Record a single timing measurement
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::profiling::Profiler;
    /// use std::time::Duration;
    ///
    /// let mut profiler = Profiler::new();
    /// profiler.record("operation", Duration::from_millis(10));
    /// ```
    pub fn record(&mut self, name: &str, duration: Duration) {
        let mut timings = self.timings.lock().unwrap();
        timings.entry(name.to_string()).or_default().push(duration);
    }

    /// Get statistics for a named section
    ///
    /// Returns (count, total, average, min, max) or None if no data
    pub fn get_stats(&self, name: &str) -> Option<TimingStats> {
        let timings = self.timings.lock().unwrap();
        if let Some(durations) = timings.get(name) {
            if durations.is_empty() {
                return None;
            }

            let total: Duration = durations.iter().sum();
            let count = durations.len();
            let avg = total / count as u32;
            let min = *durations.iter().min().unwrap();
            let max = *durations.iter().max().unwrap();

            Some(TimingStats {
                count,
                total,
                average: avg,
                min,
                max,
            })
        } else {
            None
        }
    }

    /// Get all timing data
    pub fn get_all_stats(&self) -> HashMap<String, TimingStats> {
        let timings = self.timings.lock().unwrap();
        let mut all_stats = HashMap::new();

        for (name, durations) in timings.iter() {
            if durations.is_empty() {
                continue;
            }

            let total: Duration = durations.iter().sum();
            let count = durations.len();
            let avg = total / count as u32;
            let min = *durations.iter().min().unwrap();
            let max = *durations.iter().max().unwrap();

            all_stats.insert(
                name.clone(),
                TimingStats {
                    count,
                    total,
                    average: avg,
                    min,
                    max,
                },
            );
        }

        all_stats
    }

    /// Print a formatted performance report
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::profiling::Profiler;
    ///
    /// let mut profiler = Profiler::new();
    /// // ... do profiling ...
    /// profiler.print_report();
    /// ```
    pub fn print_report(&self) {
        let all_stats = self.get_all_stats();

        if all_stats.is_empty() {
            println!("No profiling data collected");
            return;
        }

        println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
        println!("║                        Performance Report                            ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝");
        println!(
            "{:<25} {:>8} {:>12} {:>12} {:>12} {:>12}",
            "Operation", "Count", "Total", "Average", "Min", "Max"
        );
        println!("{}", "─".repeat(85));

        let mut entries: Vec<_> = all_stats.iter().collect();
        entries.sort_by(|a, b| b.1.total.cmp(&a.1.total)); // Sort by total time

        for (name, stats) in entries {
            println!(
                "{:<25} {:>8} {:>12} {:>12} {:>12} {:>12}",
                truncate(name, 25),
                stats.count,
                format_duration(stats.total),
                format_duration(stats.average),
                format_duration(stats.min),
                format_duration(stats.max),
            );
        }

        println!("{}\n", "─".repeat(85));
    }

    /// Clear all collected timing data
    pub fn clear(&mut self) {
        let mut timings = self.timings.lock().unwrap();
        timings.clear();
        let mut active = self.active.lock().unwrap();
        active.clear();
    }

    /// Export timing data to JSON format for external analysis
    ///
    /// Requires serde_json dependency (always available)
    pub fn export_json(&self) -> serde_json::Value {
        use serde_json::json;

        let all_stats = self.get_all_stats();
        let mut data = Vec::new();

        for (name, stats) in all_stats {
            data.push(json!({
                "name": name,
                "count": stats.count,
                "total_ms": stats.total.as_secs_f64() * 1000.0,
                "average_ms": stats.average.as_secs_f64() * 1000.0,
                "min_ms": stats.min.as_secs_f64() * 1000.0,
                "max_ms": stats.max.as_secs_f64() * 1000.0,
            }));
        }

        json!({ "timings": data })
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a profiled operation
#[derive(Debug, Clone, Copy)]
pub struct TimingStats {
    /// Number of times this operation was executed
    pub count: usize,
    /// Total time spent in this operation
    pub total: Duration,
    /// Average time per execution
    pub average: Duration,
    /// Minimum execution time
    pub min: Duration,
    /// Maximum execution time
    pub max: Duration,
}

/// RAII-style profiling scope that automatically stops profiling when dropped
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::performance::profiling::{Profiler, ProfileScope};
///
/// let mut profiler = Profiler::new();
///
/// {
///     let _scope = ProfileScope::new(&mut profiler, "expensive_operation");
///     // ... do expensive work ...
/// } // Automatically stops profiling when scope ends
///
/// profiler.print_report();
/// ```
pub struct ProfileScope<'a> {
    profiler: &'a mut Profiler,
    name: String,
}

impl<'a> ProfileScope<'a> {
    /// Create a new profiling scope
    pub fn new(profiler: &'a mut Profiler, name: &str) -> Self {
        profiler.start(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ProfileScope<'a> {
    fn drop(&mut self) {
        self.profiler.stop(&self.name);
    }
}

/// Macro for convenient profiling scope creation
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::profile_scope;
/// use oxirs_geosparql::performance::profiling::Profiler;
///
/// let mut profiler = Profiler::new();
///
/// profile_scope!(profiler, "my_operation", {
///     // ... code to profile ...
/// });
/// ```
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr, $body:block) => {{
        $profiler.start($name);
        let result = $body;
        $profiler.stop($name);
        result
    }};
}

// Helper functions

fn format_duration(duration: Duration) -> String {
    let micros = duration.as_micros();
    if micros < 1_000 {
        format!("{}µs", micros)
    } else if micros < 1_000_000 {
        format!("{:.2}ms", duration.as_secs_f64() * 1000.0)
    } else {
        format!("{:.2}s", duration.as_secs_f64())
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_basic() {
        let mut profiler = Profiler::new();

        profiler.start("test_operation");
        thread::sleep(Duration::from_millis(10));
        profiler.stop("test_operation");

        let stats = profiler.get_stats("test_operation").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.total >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_multiple_runs() {
        let mut profiler = Profiler::new();

        for _ in 0..5 {
            profiler.start("loop_operation");
            thread::sleep(Duration::from_millis(1));
            profiler.stop("loop_operation");
        }

        let stats = profiler.get_stats("loop_operation").unwrap();
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_profile_scope() {
        let mut profiler = Profiler::new();

        {
            let _scope = ProfileScope::new(&mut profiler, "scope_test");
            thread::sleep(Duration::from_millis(5));
        }

        let stats = profiler.get_stats("scope_test").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.total >= Duration::from_millis(5));
    }

    #[test]
    fn test_profiler_clear() {
        let mut profiler = Profiler::new();

        profiler.start("test");
        profiler.stop("test");
        assert!(profiler.get_stats("test").is_some());

        profiler.clear();
        assert!(profiler.get_stats("test").is_none());
    }

    #[test]
    fn test_profiler_record() {
        let mut profiler = Profiler::new();
        profiler.record("manual", Duration::from_millis(100));

        let stats = profiler.get_stats("manual").unwrap();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total, Duration::from_millis(100));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_micros(500)), "500µs");
        assert_eq!(format_duration(Duration::from_millis(5)), "5.00ms");
        assert_eq!(format_duration(Duration::from_secs(2)), "2.00s");
    }
}
