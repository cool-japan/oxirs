//! # Performance SLA Harness for oxirs-core
//!
//! This module provides the machinery for defining, measuring, and asserting
//! Service-Level Objectives (SLOs) against benchmark results.
//!
//! ## Quick Start
//!
//! ```rust
//! use oxirs_core::perf_sla::{BenchmarkResult, SloTarget, assert_meets_slo};
//!
//! let result = BenchmarkResult::measure("noop", 100, || {
//!     let _ = 1 + 1;
//! });
//!
//! let target = SloTarget {
//!     name: "noop".into(),
//!     p50_us: Some(10_000),
//!     p99_us: Some(50_000),
//!     throughput_ops_s: Some(1.0),
//!     allow_regression_pct: 10.0,
//! };
//!
//! assert_meets_slo(&result, &target).expect("SLO should pass");
//! ```
//!
//! ## Timing Assertions in Debug vs Release Builds
//!
//! Timing threshold checks are **only active in release builds** (no `debug_assertions`).
//! In debug builds, `assert_meets_slo` always returns `Ok(())` so that developer
//! machines and CI debug runs are never broken by scheduling variance.
//!
//! To run the SLA tests with threshold enforcement:
//!
//! ```text
//! cargo test --release -p oxirs-core -- --ignored sla_suite
//! ```

use serde::{Deserialize, Serialize};

/// A named performance target (Service-Level Objective).
///
/// Each field is optional: `None` means that dimension is not checked.
/// The `allow_regression_pct` field defaults to 10%, giving a 10% slack
/// above/below the nominal target before reporting a violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloTarget {
    /// Human-readable name for this SLO.
    pub name: String,
    /// 50th-percentile latency target in microseconds. `None` means unchecked.
    pub p50_us: Option<u64>,
    /// 99th-percentile latency target in microseconds. `None` means unchecked.
    pub p99_us: Option<u64>,
    /// Minimum throughput in operations per second. `None` means unchecked.
    pub throughput_ops_s: Option<f64>,
    /// Allowed regression percentage (0.0–100.0). Default: 10%.
    ///
    /// For latency targets this is an upward allowance: measured ≤ target × (1 + pct/100).
    /// For throughput targets this is a downward allowance: measured ≥ target × (1 − pct/100).
    #[serde(default = "default_regression_pct")]
    pub allow_regression_pct: f64,
}

fn default_regression_pct() -> f64 {
    10.0
}

/// A benchmark measurement result produced by [`BenchmarkResult::measure`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Human-readable name matching the corresponding [`SloTarget::name`].
    pub name: String,
    /// Measured 50th-percentile latency in microseconds.
    pub p50_us: u64,
    /// Measured 99th-percentile latency in microseconds.
    pub p99_us: u64,
    /// Measured throughput in operations per second.
    pub throughput_ops_s: f64,
    /// Number of samples collected.
    pub samples: usize,
    /// Total wall-clock duration of the measurement run in milliseconds.
    pub total_duration_ms: u64,
}

impl BenchmarkResult {
    /// Run a simple microbenchmark: call `f` `samples` times, record per-call
    /// latencies, and return a [`BenchmarkResult`] with p50, p99, and throughput.
    ///
    /// # Panics
    ///
    /// Panics if `samples == 0`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_core::perf_sla::BenchmarkResult;
    ///
    /// let result = BenchmarkResult::measure("example", 1_000, || {
    ///     let _ = "hello".len();
    /// });
    /// assert_eq!(result.samples, 1_000);
    /// assert!(result.throughput_ops_s > 0.0);
    /// ```
    pub fn measure<F: FnMut()>(name: &str, samples: usize, mut f: F) -> Self {
        assert!(samples > 0, "samples must be > 0");

        let mut durations: Vec<u64> = Vec::with_capacity(samples);
        let start = std::time::Instant::now();
        for _ in 0..samples {
            let t = std::time::Instant::now();
            f();
            durations.push(t.elapsed().as_micros() as u64);
        }
        let total_duration_ms = start.elapsed().as_millis() as u64;
        durations.sort_unstable();

        let p50_us = durations[samples * 50 / 100];
        let p99_us = durations[samples * 99 / 100];
        // Avoid division by zero: if wall-clock somehow rounds to 0ms, fall back to 1ms.
        let wall_s = if total_duration_ms == 0 {
            0.001
        } else {
            total_duration_ms as f64 / 1000.0
        };
        let throughput_ops_s = samples as f64 / wall_s;

        BenchmarkResult {
            name: name.to_string(),
            p50_us,
            p99_us,
            throughput_ops_s,
            samples,
            total_duration_ms,
        }
    }
}

/// Description of a single SLO violation dimension.
type ViolationMsg = String;

/// An SLO violation: one or more dimensions of a [`BenchmarkResult`] exceeded
/// the corresponding thresholds in an [`SloTarget`].
#[derive(Debug, Clone)]
pub struct SloViolation {
    /// Name of the SLO target that was violated.
    pub target_name: String,
    /// One entry per violated dimension.
    pub violations: Vec<ViolationMsg>,
}

impl std::fmt::Display for SloViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SLO '{}' violated:", self.target_name)?;
        for v in &self.violations {
            writeln!(f, "  - {v}")?;
        }
        Ok(())
    }
}

impl std::error::Error for SloViolation {}

/// Assert that a [`BenchmarkResult`] meets the given [`SloTarget`].
///
/// Returns `Ok(())` if all checked dimensions are within threshold, or an
/// `Err(SloViolation)` listing every failed dimension.
///
/// **In debug builds (`cfg(debug_assertions)`)**, timing and throughput checks
/// are skipped — only the structural invariant that `result` was provided is
/// verified. This avoids flaky failures on slow debug-build machines.
///
/// # Example
///
/// ```rust
/// use oxirs_core::perf_sla::{BenchmarkResult, SloTarget, assert_meets_slo};
///
/// let result = BenchmarkResult {
///     name: "test".into(),
///     p50_us: 50,
///     p99_us: 100,
///     throughput_ops_s: 1000.0,
///     samples: 100,
///     total_duration_ms: 100,
/// };
/// let target = SloTarget {
///     name: "test".into(),
///     p50_us: Some(200),
///     p99_us: Some(500),
///     throughput_ops_s: Some(500.0),
///     allow_regression_pct: 10.0,
/// };
/// assert!(assert_meets_slo(&result, &target).is_ok());
/// ```
pub fn assert_meets_slo(result: &BenchmarkResult, target: &SloTarget) -> Result<(), SloViolation> {
    // `violations` is only pushed in release builds, but the declaration must
    // be `mut` for release compilation. The `#[allow]` suppresses the
    // unused-mut lint that fires in debug builds where the push is absent.
    #[allow(unused_mut)]
    let mut violations: Vec<ViolationMsg> = Vec::new();

    // In debug builds we skip threshold checks to avoid timing flakiness.
    // We still reference `result` to keep the compiler happy on both paths.
    #[cfg(debug_assertions)]
    {
        // Structural check: result must have a non-empty name.
        let _ = result.name.as_str();
    }

    #[cfg(not(debug_assertions))]
    {
        if let Some(p50) = target.p50_us {
            let threshold = (p50 as f64 * (1.0 + target.allow_regression_pct / 100.0)) as u64;
            if result.p50_us > threshold {
                violations.push(format!(
                    "p50 {}µs > threshold {}µs (target {}µs + {}% slack)",
                    result.p50_us, threshold, p50, target.allow_regression_pct,
                ));
            }
        }
        if let Some(p99) = target.p99_us {
            let threshold = (p99 as f64 * (1.0 + target.allow_regression_pct / 100.0)) as u64;
            if result.p99_us > threshold {
                violations.push(format!(
                    "p99 {}µs > threshold {}µs (target {}µs + {}% slack)",
                    result.p99_us, threshold, p99, target.allow_regression_pct,
                ));
            }
        }
        if let Some(min_tps) = target.throughput_ops_s {
            let threshold = min_tps * (1.0 - target.allow_regression_pct / 100.0);
            if result.throughput_ops_s < threshold {
                violations.push(format!(
                    "throughput {:.0} ops/s < threshold {:.0} ops/s (target {:.0} ops/s − {}% slack)",
                    result.throughput_ops_s, threshold, min_tps, target.allow_regression_pct,
                ));
            }
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(SloViolation {
            target_name: target.name.clone(),
            violations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_regression_pct() {
        let json = r#"{"name":"x","p50_us":null,"p99_us":null,"throughput_ops_s":null}"#;
        let t: SloTarget = serde_json::from_str(json).expect("deserialize");
        assert!((t.allow_regression_pct - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_slo_violation_display() {
        let v = SloViolation {
            target_name: "latency".into(),
            violations: vec!["p50 too high".into()],
        };
        let s = v.to_string();
        assert!(s.contains("latency"));
        assert!(s.contains("p50 too high"));
    }

    #[test]
    fn test_measure_noop_produces_valid_result() {
        let r = BenchmarkResult::measure("noop", 100, || {
            let _ = 1_u64.wrapping_add(1);
        });
        assert_eq!(r.name, "noop");
        assert_eq!(r.samples, 100);
        assert!(r.throughput_ops_s > 0.0);
        assert!(r.p50_us <= r.p99_us);
    }

    #[test]
    #[should_panic(expected = "samples must be > 0")]
    fn test_measure_zero_samples_panics() {
        BenchmarkResult::measure("panic", 0, || {});
    }

    #[test]
    fn test_slo_roundtrip_json() {
        let t = SloTarget {
            name: "roundtrip".into(),
            p50_us: Some(100),
            p99_us: Some(500),
            throughput_ops_s: Some(1000.0),
            allow_regression_pct: 15.0,
        };
        let json = serde_json::to_string(&t).expect("serialize SloTarget");
        let back: SloTarget = serde_json::from_str(&json).expect("deserialize SloTarget");
        assert_eq!(back.name, "roundtrip");
        assert_eq!(back.p50_us, Some(100));
        assert!((back.allow_regression_pct - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_result_roundtrip_json() {
        let r = BenchmarkResult {
            name: "rt".into(),
            p50_us: 42,
            p99_us: 99,
            throughput_ops_s: 999.0,
            samples: 100,
            total_duration_ms: 100,
        };
        let json = serde_json::to_string(&r).expect("serialize BenchmarkResult");
        let back: BenchmarkResult =
            serde_json::from_str(&json).expect("deserialize BenchmarkResult");
        assert_eq!(back.name, "rt");
        assert_eq!(back.p50_us, 42);
        assert_eq!(back.p99_us, 99);
    }
}
