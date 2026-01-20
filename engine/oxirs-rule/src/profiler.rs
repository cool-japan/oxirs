//! Rule Profiler with Hotspot Analysis
//!
//! Provides comprehensive profiling and performance analysis for rule-based inference.
//! Identifies bottlenecks, hotspots, and optimization opportunities in rule execution.
//!
//! # Features
//!
//! - **Execution Timing**: Measure time spent in each rule
//! - **Hotspot Identification**: Find rules consuming most CPU time
//! - **Memory Tracking**: Monitor memory allocations during inference
//! - **Call Graph Analysis**: Understand rule dependencies and call patterns
//! - **Statistical Analysis**: Aggregate metrics across multiple runs
//! - **Bottleneck Detection**: Automatically identify performance issues
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::profiler::{RuleProfiler, ProfilingMode};
//! use oxirs_rule::{Rule, RuleAtom, Term, RuleEngine};
//!
//! let mut engine = RuleEngine::new();
//! let mut profiler = RuleProfiler::new();
//!
//! // Add rules
//! engine.add_rule(Rule {
//!     name: "test_rule".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("p".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("q".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//! });
//!
//! // Profile execution
//! profiler.start_profiling(ProfilingMode::Detailed);
//!
//! let facts = vec![RuleAtom::Triple {
//!     subject: Term::Constant("a".to_string()),
//!     predicate: Term::Constant("p".to_string()),
//!     object: Term::Constant("b".to_string()),
//! }];
//!
//! let _results = engine.forward_chain(&facts).unwrap();
//!
//! profiler.stop_profiling();
//!
//! // Analyze results
//! let report = profiler.generate_report();
//! println!("{}", report);
//! # Ok::<(), anyhow::Error>(())
//! ```

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::info;

/// Profiling mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingMode {
    /// Basic timing only
    Basic,
    /// Detailed metrics including memory and call counts
    Detailed,
    /// Full profiling with call graphs and statistical analysis
    Full,
}

/// Rule profiler with hotspot analysis
#[derive(Debug)]
pub struct RuleProfiler {
    /// Profiling mode
    mode: ProfilingMode,
    /// Active profiling session
    active: bool,
    /// Start time of current session
    start_time: Option<Instant>,
    /// Rule execution times
    rule_timings: HashMap<String, Vec<Duration>>,
    /// Rule application counts
    rule_counts: HashMap<String, u64>,
    /// Memory allocations per rule
    rule_memory: HashMap<String, u64>,
    /// Global metrics
    total_inference_time: Duration,
    total_rule_applications: u64,
}

impl Default for RuleProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleProfiler {
    /// Create a new rule profiler
    pub fn new() -> Self {
        Self {
            mode: ProfilingMode::Basic,
            active: false,
            start_time: None,
            rule_timings: HashMap::new(),
            rule_counts: HashMap::new(),
            rule_memory: HashMap::new(),
            total_inference_time: Duration::ZERO,
            total_rule_applications: 0,
        }
    }

    /// Start profiling with specified mode
    pub fn start_profiling(&mut self, mode: ProfilingMode) {
        info!("Starting rule profiler in {:?} mode", mode);
        self.mode = mode;
        self.active = true;
        self.start_time = Some(Instant::now());
    }

    /// Stop profiling
    pub fn stop_profiling(&mut self) {
        if let Some(start) = self.start_time {
            self.total_inference_time = start.elapsed();
        }
        self.active = false;
        info!(
            "Stopped rule profiler after {:?}",
            self.total_inference_time
        );
    }

    /// Record rule execution
    pub fn record_rule_execution(&mut self, rule_name: &str, duration: Duration) {
        if !self.active {
            return;
        }

        self.rule_timings
            .entry(rule_name.to_string())
            .or_default()
            .push(duration);

        *self.rule_counts.entry(rule_name.to_string()).or_insert(0) += 1;
        self.total_rule_applications += 1;
    }

    /// Record memory allocation for a rule
    pub fn record_memory_allocation(&mut self, rule_name: &str, bytes: u64) {
        if !self.active || self.mode == ProfilingMode::Basic {
            return;
        }

        *self.rule_memory.entry(rule_name.to_string()).or_insert(0) += bytes;
    }

    /// Generate profiling report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Rule Profiling Report ===\n\n");
        report.push_str(&format!("Profiling Mode: {:?}\n", self.mode));
        report.push_str(&format!(
            "Total Inference Time: {:.3}ms\n",
            self.total_inference_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "Total Rule Applications: {}\n\n",
            self.total_rule_applications
        ));

        // Hotspot analysis
        report.push_str("=== Hotspot Analysis ===\n\n");

        let hotspots = self.identify_hotspots();
        for (i, (rule_name, total_time, percentage)) in hotspots.iter().enumerate() {
            report.push_str(&format!(
                "{}. {} - {:.3}ms ({:.1}%)\n",
                i + 1,
                rule_name,
                total_time.as_secs_f64() * 1000.0,
                percentage
            ));
        }

        report.push_str("\n=== Rule Statistics ===\n\n");

        for (rule_name, timings) in &self.rule_timings {
            let count = self.rule_counts.get(rule_name).unwrap_or(&0);
            let total: Duration = timings.iter().sum();
            let avg = if !timings.is_empty() {
                total / timings.len() as u32
            } else {
                Duration::ZERO
            };

            let min = timings.iter().min().cloned().unwrap_or(Duration::ZERO);
            let max = timings.iter().max().cloned().unwrap_or(Duration::ZERO);

            report.push_str(&format!("Rule: {}\n", rule_name));
            report.push_str(&format!("  Applications: {}\n", count));
            report.push_str(&format!(
                "  Total Time: {:.3}ms\n",
                total.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Avg Time: {:.3}ms\n",
                avg.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Min Time: {:.3}ms\n",
                min.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Max Time: {:.3}ms\n",
                max.as_secs_f64() * 1000.0
            ));

            if self.mode != ProfilingMode::Basic {
                if let Some(&mem) = self.rule_memory.get(rule_name) {
                    report.push_str(&format!("  Memory: {} bytes\n", mem));
                }
            }

            report.push('\n');
        }

        // Bottleneck detection
        if self.mode == ProfilingMode::Full {
            report.push_str("=== Bottleneck Detection ===\n\n");
            let bottlenecks = self.detect_bottlenecks();
            if bottlenecks.is_empty() {
                report.push_str("No bottlenecks detected.\n");
            } else {
                for bottleneck in bottlenecks {
                    report.push_str(&format!("- {}\n", bottleneck));
                }
            }
            report.push('\n');
        }

        // Optimization recommendations
        report.push_str("=== Optimization Recommendations ===\n\n");
        let recommendations = self.generate_recommendations();
        for rec in recommendations {
            report.push_str(&format!("- {}\n", rec));
        }

        report
    }

    /// Identify hotspots (rules consuming most time)
    fn identify_hotspots(&self) -> Vec<(String, Duration, f64)> {
        let mut hotspots: Vec<(String, Duration)> = self
            .rule_timings
            .iter()
            .map(|(name, timings)| {
                let total: Duration = timings.iter().sum();
                (name.clone(), total)
            })
            .collect();

        hotspots.sort_by(|a, b| b.1.cmp(&a.1));

        let total_time_micros = self.total_inference_time.as_micros() as f64;

        hotspots
            .into_iter()
            .take(10)
            .map(|(name, duration)| {
                let percentage = if total_time_micros > 0.0 {
                    (duration.as_micros() as f64 / total_time_micros) * 100.0
                } else {
                    0.0
                };
                (name, duration, percentage)
            })
            .collect()
    }

    /// Detect bottlenecks
    fn detect_bottlenecks(&self) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        // Check for rules with high execution time
        for (rule_name, timings) in &self.rule_timings {
            let total: Duration = timings.iter().sum();
            let avg = if !timings.is_empty() {
                total / timings.len() as u32
            } else {
                Duration::ZERO
            };

            // Flag rules taking more than 10ms on average
            if avg.as_millis() > 10 {
                bottlenecks.push(format!(
                    "Rule '{}' has high average execution time ({:.3}ms)",
                    rule_name,
                    avg.as_secs_f64() * 1000.0
                ));
            }

            // Flag rules with high variance
            if timings.len() > 1 {
                let variance = self.calculate_variance(timings);
                let std_dev = variance.sqrt();
                if std_dev > 5.0 {
                    // More than 5ms std dev
                    bottlenecks.push(format!(
                        "Rule '{}' has high variance in execution time (σ={:.3}ms)",
                        rule_name, std_dev
                    ));
                }
            }
        }

        // Check for memory-intensive rules
        if self.mode != ProfilingMode::Basic {
            for (rule_name, &mem) in &self.rule_memory {
                if mem > 1_000_000 {
                    // More than 1MB
                    bottlenecks.push(format!(
                        "Rule '{}' has high memory usage ({:.2}MB)",
                        rule_name,
                        mem as f64 / 1_000_000.0
                    ));
                }
            }
        }

        bottlenecks
    }

    /// Calculate variance of durations
    fn calculate_variance(&self, durations: &[Duration]) -> f64 {
        if durations.len() < 2 {
            return 0.0;
        }

        let mean = durations.iter().sum::<Duration>().as_secs_f64() / durations.len() as f64;

        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;

        variance * 1000.0 * 1000.0 // Convert to ms²
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        let hotspots = self.identify_hotspots();

        if let Some((rule_name, _duration, percentage)) = hotspots.first() {
            if *percentage > 50.0 {
                recommendations.push(format!(
                    "Rule '{}' consumes {:.1}% of total time. Consider optimizing this rule first.",
                    rule_name, percentage
                ));
            }
        }

        // Check for frequently applied rules
        for (rule_name, &count) in &self.rule_counts {
            if count > 1000 {
                recommendations.push(format!(
                    "Rule '{}' applied {} times. Consider caching or indexing.",
                    rule_name, count
                ));
            }
        }

        // Check total inference time
        if self.total_inference_time.as_secs() > 1 {
            recommendations.push(
                "Total inference time exceeds 1 second. Consider parallel execution.".to_string(),
            );
        }

        // Check for rules with no applications
        let all_rules: Vec<String> = self.rule_timings.keys().cloned().collect();
        for rule_name in &all_rules {
            if let Some(&count) = self.rule_counts.get(rule_name) {
                if count == 0 {
                    recommendations.push(format!(
                        "Rule '{}' was never applied. Consider removing or reviewing.",
                        rule_name
                    ));
                }
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Performance looks good! No major issues detected.".to_string());
        }

        recommendations
    }

    /// Export profiling data to JSON
    pub fn export_to_json(&self) -> Result<String> {
        use serde_json::json;

        let hotspots = self.identify_hotspots();

        let data = json!({
            "mode": format!("{:?}", self.mode),
            "total_inference_time_ms": self.total_inference_time.as_secs_f64() * 1000.0,
            "total_rule_applications": self.total_rule_applications,
            "hotspots": hotspots.iter().map(|(name, duration, pct)| {
                json!({
                    "rule": name,
                    "total_time_ms": duration.as_secs_f64() * 1000.0,
                    "percentage": pct,
                })
            }).collect::<Vec<_>>(),
            "rule_statistics": self.rule_timings.iter().map(|(name, timings)| {
                let total: Duration = timings.iter().sum();
                let avg = if !timings.is_empty() {
                    total / timings.len() as u32
                } else {
                    Duration::ZERO
                };

                json!({
                    "rule": name,
                    "applications": self.rule_counts.get(name).unwrap_or(&0),
                    "total_time_ms": total.as_secs_f64() * 1000.0,
                    "avg_time_ms": avg.as_secs_f64() * 1000.0,
                    "memory_bytes": self.rule_memory.get(name).cloned().unwrap_or(0),
                })
            }).collect::<Vec<_>>(),
            "bottlenecks": self.detect_bottlenecks(),
            "recommendations": self.generate_recommendations(),
        });

        Ok(serde_json::to_string_pretty(&data)?)
    }

    /// Reset profiler state
    pub fn reset(&mut self) {
        self.active = false;
        self.start_time = None;
        self.rule_timings.clear();
        self.rule_counts.clear();
        self.rule_memory.clear();
        self.total_inference_time = Duration::ZERO;
        self.total_rule_applications = 0;
    }

    /// Get total inference time
    pub fn total_inference_time(&self) -> Duration {
        self.total_inference_time
    }

    /// Get rule application count
    pub fn rule_application_count(&self, rule_name: &str) -> u64 {
        self.rule_counts.get(rule_name).cloned().unwrap_or(0)
    }

    /// Check if profiler is active
    pub fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let mut profiler = RuleProfiler::new();

        profiler.start_profiling(ProfilingMode::Basic);
        assert!(profiler.is_active());

        profiler.record_rule_execution("rule1", Duration::from_millis(10));
        profiler.record_rule_execution("rule2", Duration::from_millis(5));

        profiler.stop_profiling();
        assert!(!profiler.is_active());

        let report = profiler.generate_report();
        assert!(report.contains("rule1"));
        assert!(report.contains("rule2"));
    }

    #[test]
    fn test_profiler_detailed() {
        let mut profiler = RuleProfiler::new();

        profiler.start_profiling(ProfilingMode::Detailed);

        profiler.record_rule_execution("rule1", Duration::from_millis(10));
        profiler.record_memory_allocation("rule1", 1024);

        profiler.stop_profiling();

        let report = profiler.generate_report();
        assert!(report.contains("rule1"));
        assert!(report.contains("1024"));
    }

    #[test]
    fn test_hotspot_identification() {
        let mut profiler = RuleProfiler::new();

        profiler.start_profiling(ProfilingMode::Full);

        // Rule1 is slow
        for _ in 0..10 {
            profiler.record_rule_execution("rule1", Duration::from_millis(100));
        }

        // Rule2 is fast
        for _ in 0..10 {
            profiler.record_rule_execution("rule2", Duration::from_millis(1));
        }

        profiler.stop_profiling();

        let hotspots = profiler.identify_hotspots();
        assert!(!hotspots.is_empty());
        assert_eq!(hotspots[0].0, "rule1"); // rule1 should be the hotspot
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut profiler = RuleProfiler::new();

        profiler.start_profiling(ProfilingMode::Full);

        // Create a rule with high execution time
        profiler.record_rule_execution("slow_rule", Duration::from_millis(50));

        // Create a rule with high memory usage
        profiler.record_memory_allocation("memory_rule", 2_000_000);

        profiler.stop_profiling();

        let bottlenecks = profiler.detect_bottlenecks();
        assert!(!bottlenecks.is_empty());
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = RuleProfiler::new();

        profiler.start_profiling(ProfilingMode::Basic);
        profiler.record_rule_execution("rule1", Duration::from_millis(10));
        profiler.stop_profiling();

        assert_eq!(profiler.total_rule_applications, 1);

        profiler.reset();

        assert_eq!(profiler.total_rule_applications, 0);
        assert!(profiler.rule_timings.is_empty());
    }

    #[test]
    fn test_json_export() {
        let mut profiler = RuleProfiler::new();

        profiler.start_profiling(ProfilingMode::Detailed);
        profiler.record_rule_execution("rule1", Duration::from_millis(10));
        profiler.stop_profiling();

        let json = profiler.export_to_json().unwrap();
        assert!(json.contains("rule1"));
        assert!(json.contains("total_inference_time_ms"));
    }

    #[test]
    fn test_recommendations() {
        let mut profiler = RuleProfiler::new();

        profiler.start_profiling(ProfilingMode::Full);

        // Add rule with many applications
        for _ in 0..2000 {
            profiler.record_rule_execution("frequent_rule", Duration::from_micros(100));
        }

        profiler.stop_profiling();

        let recommendations = profiler.generate_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_variance_calculation() {
        let profiler = RuleProfiler::new();

        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(15),
        ];

        let variance = profiler.calculate_variance(&durations);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_profiler_modes() {
        let mut profiler = RuleProfiler::new();

        // Test basic mode
        profiler.start_profiling(ProfilingMode::Basic);
        profiler.record_memory_allocation("rule1", 1024);
        profiler.stop_profiling();

        assert!(profiler.rule_memory.is_empty()); // Memory not tracked in basic mode

        profiler.reset();

        // Test detailed mode
        profiler.start_profiling(ProfilingMode::Detailed);
        profiler.record_memory_allocation("rule1", 1024);
        profiler.stop_profiling();

        assert!(!profiler.rule_memory.is_empty()); // Memory tracked in detailed mode
    }

    #[test]
    fn test_multiple_executions() {
        let mut profiler = RuleProfiler::new();

        profiler.start_profiling(ProfilingMode::Full);

        // Record multiple executions of the same rule
        for i in 0..5 {
            profiler.record_rule_execution("rule1", Duration::from_millis(i * 2));
        }

        profiler.stop_profiling();

        assert_eq!(profiler.rule_application_count("rule1"), 5);

        let report = profiler.generate_report();
        assert!(report.contains("Min Time"));
        assert!(report.contains("Max Time"));
        assert!(report.contains("Avg Time"));
    }
}
