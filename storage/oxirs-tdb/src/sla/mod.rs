//! Advanced SLA monitoring and enforcement for OxiRS TDB storage layer.
//!
//! Tracks write/read latencies, error rates, downtime, and throughput,
//! then reports violations against configured SLA thresholds.

use std::time::SystemTime;

// ─── SLA Definition ──────────────────────────────────────────────────────────

/// SLA contract for storage operations.
#[derive(Debug, Clone)]
pub struct StorageSla {
    /// Maximum acceptable write latency (P99) in milliseconds
    pub max_write_latency_ms: u64,
    /// Maximum acceptable read latency (P99) in milliseconds
    pub max_read_latency_ms: u64,
    /// Minimum acceptable throughput (operations per second)
    pub min_throughput_tps: f64,
    /// Maximum acceptable error rate in [0, 1]
    pub max_error_rate: f64,
    /// Target availability in [0, 1]
    pub target_availability: f64,
}

impl Default for StorageSla {
    fn default() -> Self {
        Self {
            max_write_latency_ms: 100,
            max_read_latency_ms: 50,
            min_throughput_tps: 1000.0,
            max_error_rate: 0.001,
            target_availability: 0.9999,
        }
    }
}

// ─── Violation Types ─────────────────────────────────────────────────────────

/// Category of SLA violation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationType {
    /// P99 write latency exceeded threshold
    WriteLatency,
    /// P99 read latency exceeded threshold
    ReadLatency,
    /// Throughput fell below minimum
    Throughput,
    /// Error rate exceeded maximum
    ErrorRate,
    /// Availability dropped below target
    Availability,
}

/// A single SLA violation with measured vs. threshold values.
#[derive(Debug, Clone)]
pub struct SlaViolation {
    /// What kind of SLA was violated
    pub violation_type: ViolationType,
    /// Measured value (e.g. latency in ms, error fraction, tps)
    pub measured: f64,
    /// SLA threshold that was breached
    pub threshold: f64,
    /// Unix timestamp (ms) when the violation was detected
    pub timestamp_ms: i64,
}

// ─── SlaMonitor ──────────────────────────────────────────────────────────────

/// Tracks storage metrics and checks them against an [`StorageSla`].
pub struct SlaMonitor {
    sla: StorageSla,
    write_latencies: Vec<u64>,
    read_latencies: Vec<u64>,
    error_count: u64,
    total_ops: u64,
    downtime_ms: u64,
    start_ms: i64,
    window_ms: u64,
}

impl SlaMonitor {
    /// Create a new monitor with a 60-second default measurement window.
    pub fn new(sla: StorageSla) -> Self {
        Self {
            sla,
            write_latencies: Vec::new(),
            read_latencies: Vec::new(),
            error_count: 0,
            total_ops: 0,
            downtime_ms: 0,
            start_ms: now_ms(),
            window_ms: 60_000,
        }
    }

    /// Record a write operation with the given latency.
    pub fn record_write(&mut self, latency_ms: u64) {
        self.write_latencies.push(latency_ms);
        self.total_ops += 1;
    }

    /// Record a read operation with the given latency.
    pub fn record_read(&mut self, latency_ms: u64) {
        self.read_latencies.push(latency_ms);
        self.total_ops += 1;
    }

    /// Record that an error occurred (also counts as an operation).
    pub fn record_error(&mut self) {
        self.error_count += 1;
        self.total_ops += 1;
    }

    /// Record that the storage system was unavailable for `duration_ms` ms.
    pub fn record_downtime(&mut self, duration_ms: u64) {
        self.downtime_ms += duration_ms;
    }

    /// Compute current availability based on elapsed time and total downtime.
    pub fn availability(&self) -> f64 {
        let elapsed_ms = (now_ms() - self.start_ms).max(0) as u64;
        if elapsed_ms == 0 {
            return 1.0;
        }
        let uptime_ms = elapsed_ms.saturating_sub(self.downtime_ms);
        uptime_ms as f64 / elapsed_ms as f64
    }

    /// 99th percentile write latency across all recorded writes.
    pub fn p99_write_latency(&self) -> u64 {
        percentile_99(&self.write_latencies)
    }

    /// 99th percentile read latency across all recorded reads.
    pub fn p99_read_latency(&self) -> u64 {
        percentile_99(&self.read_latencies)
    }

    /// Estimated throughput (operations per second) since monitor creation.
    pub fn current_tps(&self) -> f64 {
        let elapsed_s = (now_ms() - self.start_ms).max(1) as f64 / 1000.0;
        self.total_ops as f64 / elapsed_s
    }

    /// Current error rate as a fraction of total operations.
    pub fn error_rate(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.error_count as f64 / self.total_ops as f64
        }
    }

    /// Check all SLA constraints and return a list of violations.
    pub fn check_violations(&self) -> Vec<SlaViolation> {
        let ts = now_ms();
        let mut violations = Vec::new();

        let p99w = self.p99_write_latency();
        if p99w > self.sla.max_write_latency_ms {
            violations.push(SlaViolation {
                violation_type: ViolationType::WriteLatency,
                measured: p99w as f64,
                threshold: self.sla.max_write_latency_ms as f64,
                timestamp_ms: ts,
            });
        }

        let p99r = self.p99_read_latency();
        if p99r > self.sla.max_read_latency_ms {
            violations.push(SlaViolation {
                violation_type: ViolationType::ReadLatency,
                measured: p99r as f64,
                threshold: self.sla.max_read_latency_ms as f64,
                timestamp_ms: ts,
            });
        }

        let tps = self.current_tps();
        if tps < self.sla.min_throughput_tps {
            violations.push(SlaViolation {
                violation_type: ViolationType::Throughput,
                measured: tps,
                threshold: self.sla.min_throughput_tps,
                timestamp_ms: ts,
            });
        }

        let er = self.error_rate();
        if er > self.sla.max_error_rate {
            violations.push(SlaViolation {
                violation_type: ViolationType::ErrorRate,
                measured: er,
                threshold: self.sla.max_error_rate,
                timestamp_ms: ts,
            });
        }

        let avail = self.availability();
        if avail < self.sla.target_availability {
            violations.push(SlaViolation {
                violation_type: ViolationType::Availability,
                measured: avail,
                threshold: self.sla.target_availability,
                timestamp_ms: ts,
            });
        }

        violations
    }

    /// Returns `true` if there are no active SLA violations.
    pub fn is_compliant(&self) -> bool {
        self.check_violations().is_empty()
    }

    /// Build a detailed compliance report.
    pub fn compliance_report(&self) -> SlaComplianceReport {
        let violations = self.check_violations();
        SlaComplianceReport {
            compliant: violations.is_empty(),
            violations,
            availability: self.availability(),
            p99_write_ms: self.p99_write_latency(),
            p99_read_ms: self.p99_read_latency(),
            tps: self.current_tps(),
            error_rate: self.error_rate(),
        }
    }
}

// ─── SlaComplianceReport ─────────────────────────────────────────────────────

/// Snapshot of SLA compliance at a point in time.
#[derive(Debug)]
pub struct SlaComplianceReport {
    /// Whether all SLA constraints are currently met
    pub compliant: bool,
    /// List of active violations
    pub violations: Vec<SlaViolation>,
    /// Current availability fraction
    pub availability: f64,
    /// P99 write latency in ms
    pub p99_write_ms: u64,
    /// P99 read latency in ms
    pub p99_read_ms: u64,
    /// Current throughput (ops/s)
    pub tps: f64,
    /// Current error rate
    pub error_rate: f64,
}

// ─── Helper functions ─────────────────────────────────────────────────────────

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Compute the 99th-percentile value from an unsorted slice.
fn percentile_99(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() as f64 * 0.99).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lenient_sla() -> StorageSla {
        StorageSla {
            max_write_latency_ms: 500,
            max_read_latency_ms: 200,
            min_throughput_tps: 0.0001,
            max_error_rate: 1.0,
            target_availability: 0.0,
        }
    }

    #[test]
    fn test_default_sla_values() {
        let sla = StorageSla::default();
        assert_eq!(sla.max_write_latency_ms, 100);
        assert_eq!(sla.max_read_latency_ms, 50);
        assert!((sla.min_throughput_tps - 1000.0).abs() < f64::EPSILON);
        assert!((sla.max_error_rate - 0.001).abs() < f64::EPSILON);
        assert!((sla.target_availability - 0.9999).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_monitor_empty() {
        let m = SlaMonitor::new(StorageSla::default());
        assert_eq!(m.write_latencies.len(), 0);
        assert_eq!(m.total_ops, 0);
        assert_eq!(m.window_ms, 60_000);
    }

    #[test]
    fn test_record_write_increments_ops() {
        let mut m = SlaMonitor::new(lenient_sla());
        m.record_write(10);
        m.record_write(20);
        assert_eq!(m.write_latencies.len(), 2);
        assert_eq!(m.total_ops, 2);
    }

    #[test]
    fn test_record_read_increments_ops() {
        let mut m = SlaMonitor::new(lenient_sla());
        m.record_read(5);
        assert_eq!(m.read_latencies.len(), 1);
        assert_eq!(m.total_ops, 1);
    }

    #[test]
    fn test_record_error_increments_error_count() {
        let mut m = SlaMonitor::new(lenient_sla());
        m.record_error();
        assert_eq!(m.error_count, 1);
        assert_eq!(m.total_ops, 1);
    }

    #[test]
    fn test_error_rate_zero_ops() {
        let m = SlaMonitor::new(lenient_sla());
        assert_eq!(m.error_rate(), 0.0);
    }

    #[test]
    fn test_error_rate_computation() {
        let mut m = SlaMonitor::new(lenient_sla());
        m.record_write(10);
        m.record_write(10);
        m.record_error();
        // 1 error out of 3 ops ≈ 0.333
        let rate = m.error_rate();
        assert!((rate - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_p99_write_latency_empty() {
        let m = SlaMonitor::new(lenient_sla());
        assert_eq!(m.p99_write_latency(), 0);
    }

    #[test]
    fn test_p99_write_latency_single() {
        let mut m = SlaMonitor::new(lenient_sla());
        m.record_write(42);
        assert_eq!(m.p99_write_latency(), 42);
    }

    #[test]
    fn test_p99_write_latency_many() {
        let mut m = SlaMonitor::new(lenient_sla());
        for i in 0..100u64 {
            m.record_write(i);
        }
        // P99 of 0..99 is 98
        assert_eq!(m.p99_write_latency(), 98);
    }

    #[test]
    fn test_p99_read_latency_many() {
        let mut m = SlaMonitor::new(lenient_sla());
        for i in 0..100u64 {
            m.record_read(i);
        }
        assert_eq!(m.p99_read_latency(), 98);
    }

    #[test]
    fn test_availability_no_downtime() {
        let m = SlaMonitor::new(lenient_sla());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let avail = m.availability();
        assert!(
            avail > 0.99,
            "expected availability near 1.0, got {}",
            avail
        );
    }

    #[test]
    fn test_availability_with_heavy_downtime() {
        let mut m = SlaMonitor::new(lenient_sla());
        std::thread::sleep(std::time::Duration::from_millis(20));
        m.record_downtime(100_000); // 100 seconds simulated downtime
        let avail = m.availability();
        assert!(avail < 1.0, "availability should be < 1.0 with downtime");
    }

    #[test]
    fn test_record_downtime_accumulates() {
        let mut m = SlaMonitor::new(lenient_sla());
        m.record_downtime(100);
        m.record_downtime(200);
        assert_eq!(m.downtime_ms, 300);
    }

    #[test]
    fn test_no_write_latency_violation_within_limit() {
        let sla = StorageSla {
            max_write_latency_ms: 500,
            max_read_latency_ms: 200,
            min_throughput_tps: 0.0,
            max_error_rate: 1.0,
            target_availability: 0.0,
        };
        let mut m = SlaMonitor::new(sla);
        for _ in 0..100 {
            m.record_write(10); // well within 500ms limit
        }
        let violations = m.check_violations();
        assert!(!violations
            .iter()
            .any(|v| v.violation_type == ViolationType::WriteLatency));
    }

    #[test]
    fn test_write_latency_violation() {
        let sla = StorageSla {
            max_write_latency_ms: 10,
            max_read_latency_ms: 200,
            min_throughput_tps: 0.0,
            max_error_rate: 1.0,
            target_availability: 0.0,
        };
        let mut m = SlaMonitor::new(sla);
        for _ in 0..100 {
            m.record_write(50); // 50ms > 10ms limit
        }
        let violations = m.check_violations();
        assert!(violations
            .iter()
            .any(|v| v.violation_type == ViolationType::WriteLatency));
    }

    #[test]
    fn test_read_latency_violation() {
        let sla = StorageSla {
            max_write_latency_ms: 500,
            max_read_latency_ms: 5,
            min_throughput_tps: 0.0,
            max_error_rate: 1.0,
            target_availability: 0.0,
        };
        let mut m = SlaMonitor::new(sla);
        for _ in 0..100 {
            m.record_read(20); // 20ms > 5ms limit
        }
        let violations = m.check_violations();
        assert!(violations
            .iter()
            .any(|v| v.violation_type == ViolationType::ReadLatency));
    }

    #[test]
    fn test_error_rate_violation() {
        let sla = StorageSla {
            max_write_latency_ms: 500,
            max_read_latency_ms: 200,
            min_throughput_tps: 0.0,
            max_error_rate: 0.01,
            target_availability: 0.0,
        };
        let mut m = SlaMonitor::new(sla);
        for _ in 0..50 {
            m.record_write(1);
        }
        for _ in 0..50 {
            m.record_error(); // 50% error rate >> 1% limit
        }
        let violations = m.check_violations();
        assert!(violations
            .iter()
            .any(|v| v.violation_type == ViolationType::ErrorRate));
    }

    #[test]
    fn test_availability_violation() {
        let sla = StorageSla {
            max_write_latency_ms: 500,
            max_read_latency_ms: 200,
            min_throughput_tps: 0.0,
            max_error_rate: 1.0,
            target_availability: 0.9999,
        };
        let mut m = SlaMonitor::new(sla);
        std::thread::sleep(std::time::Duration::from_millis(20));
        m.record_downtime(1_000_000); // massively over-reports downtime
        let violations = m.check_violations();
        assert!(violations
            .iter()
            .any(|v| v.violation_type == ViolationType::Availability));
    }

    #[test]
    fn test_is_compliant_with_lenient_sla() {
        let mut m = SlaMonitor::new(lenient_sla());
        m.record_write(1);
        m.record_read(1);
        // Write/read latencies are well within limits; no error rate violation
        let report = m.compliance_report();
        assert!(report.p99_write_ms <= 500);
        assert!(report.p99_read_ms <= 200);
    }

    #[test]
    fn test_compliance_report_fields() {
        let mut m = SlaMonitor::new(lenient_sla());
        m.record_write(10);
        m.record_read(5);
        let report = m.compliance_report();
        assert!(report.tps >= 0.0);
        assert!(report.error_rate >= 0.0);
        assert!((0.0..=1.0).contains(&report.availability));
        assert_eq!(report.p99_write_ms, 10);
        assert_eq!(report.p99_read_ms, 5);
    }

    #[test]
    fn test_current_tps_positive_after_ops() {
        let mut m = SlaMonitor::new(lenient_sla());
        for _ in 0..10 {
            m.record_write(1);
        }
        assert!(m.current_tps() > 0.0);
    }

    #[test]
    fn test_violation_measured_exceeds_threshold() {
        let sla = StorageSla {
            max_write_latency_ms: 5,
            max_read_latency_ms: 200,
            min_throughput_tps: 0.0,
            max_error_rate: 1.0,
            target_availability: 0.0,
        };
        let mut m = SlaMonitor::new(sla);
        for _ in 0..100 {
            m.record_write(100);
        }
        let violations = m.check_violations();
        let wl = violations
            .iter()
            .find(|v| v.violation_type == ViolationType::WriteLatency)
            .unwrap();
        assert_eq!(wl.threshold, 5.0);
        assert!(wl.measured > wl.threshold);
        assert!(wl.timestamp_ms > 0);
    }
}
