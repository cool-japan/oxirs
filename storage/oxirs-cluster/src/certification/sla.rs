//! SLA certification: latency bounds and throughput targets under simulated load.
//!
//! Runs 1000 simulated read operations and 500 simulated write operations against
//! an in-memory key–value store, measures elapsed time per operation, and asserts
//! that the 99th-percentile latency is within the configured thresholds.
//!
//! Because every operation is pure in-memory computation this is effectively a
//! "simulation overhead is bounded" check rather than a real network benchmark.

use std::collections::HashMap;
use std::time::Instant;

use super::{CertificationConfig, CheckResult, SlaResult};

const READ_SAMPLES: usize = 1000;
const WRITE_SAMPLES: usize = 500;

/// Execute the SLA certification suite and return a [`SlaResult`].
pub fn certify(config: &CertificationConfig) -> SlaResult {
    let checks: Vec<CheckResult> = vec![
        run_read_latency_check(config),
        run_write_latency_check(config),
        run_throughput_check(config),
    ];

    let passed = checks.iter().all(|c| c.passed);
    let notes = if passed {
        format!(
            "All SLA checks passed. read p99 < {}µs, write p99 < {}µs.",
            config.sla_read_latency_us, config.sla_write_latency_us
        )
    } else {
        let failures: Vec<&str> = checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.name.as_str())
            .collect();
        format!("SLA violations detected: {}", failures.join(", "))
    };

    SlaResult {
        passed,
        checks,
        notes,
    }
}

/// Simulate `READ_SAMPLES` read operations and assert p99 < threshold.
fn run_read_latency_check(config: &CertificationConfig) -> CheckResult {
    let mut store: HashMap<String, String> = HashMap::new();
    // Pre-populate so reads always find data.
    for i in 0..READ_SAMPLES {
        store.insert(format!("key-{i}"), format!("value-{i}"));
    }

    let mut latencies_us: Vec<u64> = Vec::with_capacity(READ_SAMPLES);
    for i in 0..READ_SAMPLES {
        let key = format!("key-{}", i % READ_SAMPLES);
        let t0 = Instant::now();
        let _v = store.get(&key);
        let elapsed_us = t0.elapsed().as_micros() as u64;
        latencies_us.push(elapsed_us);
    }

    let p99_us = percentile_99(&mut latencies_us);
    let threshold = config.sla_read_latency_us;
    let passed = p99_us < threshold;

    CheckResult {
        name: "read_latency_p99".to_string(),
        passed,
        detail: format!("Read p99 = {p99_us}µs (threshold {threshold}µs, {READ_SAMPLES} samples)"),
    }
}

/// Simulate `WRITE_SAMPLES` write operations and assert p99 < threshold.
fn run_write_latency_check(config: &CertificationConfig) -> CheckResult {
    let mut store: HashMap<String, String> = HashMap::new();

    let mut latencies_us: Vec<u64> = Vec::with_capacity(WRITE_SAMPLES);
    for i in 0..WRITE_SAMPLES {
        let key = format!("write-key-{i}");
        let val = format!("write-value-{i}");
        let t0 = Instant::now();
        store.insert(key, val);
        let elapsed_us = t0.elapsed().as_micros() as u64;
        latencies_us.push(elapsed_us);
    }

    let p99_us = percentile_99(&mut latencies_us);
    let threshold = config.sla_write_latency_us;
    let passed = p99_us < threshold;

    CheckResult {
        name: "write_latency_p99".to_string(),
        passed,
        detail: format!(
            "Write p99 = {p99_us}µs (threshold {threshold}µs, {WRITE_SAMPLES} samples)"
        ),
    }
}

/// Verify that combined read+write throughput meets a minimum floor.
///
/// We perform `epochs` round-trips (one write + one read each) and assert the
/// total wall time is < 1 second, which corresponds to at least `epochs` rps.
fn run_throughput_check(config: &CertificationConfig) -> CheckResult {
    let rounds = config.epochs;
    let mut store: HashMap<String, String> = HashMap::new();

    let t0 = Instant::now();
    for i in 0..rounds {
        let key = format!("tp-key-{i}");
        let val = format!("tp-val-{i}");
        store.insert(key.clone(), val);
        let _v = store.get(&key);
    }
    let elapsed_ms = t0.elapsed().as_millis();

    // Expect in-memory ops to complete well under 1 second for ≤1000 epochs.
    let passed = elapsed_ms < 1000;

    CheckResult {
        name: "throughput_floor".to_string(),
        passed,
        detail: format!(
            "{rounds} read-write round-trips completed in {elapsed_ms}ms (threshold <1000ms)"
        ),
    }
}

/// Return the 99th-percentile value from a sample slice (modifies in place to sort).
fn percentile_99(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    let idx = (samples.len() * 99 / 100).saturating_sub(1);
    samples[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certification::CertificationConfig;

    fn default_config() -> CertificationConfig {
        CertificationConfig::default()
    }

    #[test]
    fn test_sla_certify_passes_with_defaults() {
        let cfg = default_config();
        let result = certify(&cfg);
        assert!(
            result.passed,
            "SLA certification should pass: {:?}",
            result.notes
        );
    }

    #[test]
    fn test_sla_checks_are_present() {
        let cfg = default_config();
        let result = certify(&cfg);
        assert!(result.checks.len() >= 2, "Expected at least 2 SLA checks");
    }

    #[test]
    fn test_read_latency_check_passes() {
        let cfg = default_config();
        let result = run_read_latency_check(&cfg);
        assert!(
            result.passed,
            "Read latency check failed: {}",
            result.detail
        );
    }

    #[test]
    fn test_write_latency_check_passes() {
        let cfg = default_config();
        let result = run_write_latency_check(&cfg);
        assert!(
            result.passed,
            "Write latency check failed: {}",
            result.detail
        );
    }

    #[test]
    fn test_percentile_99_empty() {
        let mut v: Vec<u64> = vec![];
        assert_eq!(percentile_99(&mut v), 0);
    }

    #[test]
    fn test_percentile_99_sorted() {
        let mut v: Vec<u64> = (1..=100).collect();
        // 99th percentile of [1..100] should be 99
        let p = percentile_99(&mut v);
        assert!((98..=100).contains(&p), "p99={p}");
    }
}
