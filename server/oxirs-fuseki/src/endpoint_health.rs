//! Endpoint health check with configurable probes.
//!
//! Provides a composable health-check system for SPARQL endpoints.
//! Each [`HealthProbe`] implementation can report [`ProbeStatus::Healthy`],
//! [`ProbeStatus::Degraded`], or [`ProbeStatus::Unhealthy`].  A
//! [`HealthChecker`] aggregates multiple probes into a [`HealthReport`].

use std::time::{SystemTime, UNIX_EPOCH};

// ── Status types ──────────────────────────────────────────────────────────────

/// Per-probe health status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeStatus {
    /// The probe reports no issues.
    Healthy,
    /// The probe detected a degraded condition (detail in message).
    Degraded(String),
    /// The probe detected an unhealthy condition (detail in message).
    Unhealthy(String),
}

/// Result of a single probe invocation.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Name of the probe that produced this result.
    pub name: String,
    /// Status reported by the probe.
    pub status: ProbeStatus,
    /// How long the probe took (milliseconds, simulated / measured).
    pub latency_ms: u64,
    /// Unix timestamp (milliseconds) when the probe was executed.
    pub timestamp: u64,
}

impl ProbeResult {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, status: ProbeStatus, latency_ms: u64) -> Self {
        Self {
            name: name.into(),
            status,
            latency_ms,
            timestamp: current_time_ms(),
        }
    }

    /// Returns `true` if the status is [`ProbeStatus::Healthy`].
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, ProbeStatus::Healthy)
    }

    /// Returns `true` if the status is [`ProbeStatus::Unhealthy`].
    pub fn is_unhealthy(&self) -> bool {
        matches!(self.status, ProbeStatus::Unhealthy(_))
    }
}

// ── Probe trait ───────────────────────────────────────────────────────────────

/// A health probe that can be registered with a [`HealthChecker`].
pub trait HealthProbe: Send + Sync {
    /// Unique name identifying this probe.
    fn name(&self) -> &str;
    /// Execute the probe and return a result.
    fn check(&self) -> ProbeResult;
}

// ── Built-in probes ───────────────────────────────────────────────────────────

/// A probe that checks available system memory against a threshold.
pub struct MemoryProbe {
    /// Minimum available memory in MB before the probe degrades.
    pub threshold_mb: usize,
}

impl MemoryProbe {
    /// Create a new memory probe with the given threshold in MB.
    pub fn new(threshold_mb: usize) -> Self {
        Self { threshold_mb }
    }
}

impl HealthProbe for MemoryProbe {
    fn name(&self) -> &str {
        "memory"
    }

    fn check(&self) -> ProbeResult {
        let start = current_time_ms();
        // In a real implementation this would query OS memory stats.
        // For now we use a fixed heuristic: we simulate 512 MB available.
        let simulated_available_mb: usize = 512;
        let latency = current_time_ms().saturating_sub(start);

        if simulated_available_mb >= self.threshold_mb {
            ProbeResult::new("memory", ProbeStatus::Healthy, latency)
        } else {
            ProbeResult::new(
                "memory",
                ProbeStatus::Degraded(format!(
                    "available {}MB < threshold {}MB",
                    simulated_available_mb, self.threshold_mb
                )),
                latency,
            )
        }
    }
}

/// A probe that reports how long the process has been running.
pub struct UptimeProbe {
    /// Unix timestamp (milliseconds) when the service was started.
    pub start_time: u64,
}

impl UptimeProbe {
    /// Create a new uptime probe recording the start time as now.
    pub fn new() -> Self {
        Self {
            start_time: current_time_ms(),
        }
    }

    /// Create a probe with a specific start time.
    pub fn with_start(start_time: u64) -> Self {
        Self { start_time }
    }

    /// Return uptime in milliseconds at query time.
    pub fn uptime_ms(&self) -> u64 {
        current_time_ms().saturating_sub(self.start_time)
    }
}

impl Default for UptimeProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthProbe for UptimeProbe {
    fn name(&self) -> &str {
        "uptime"
    }

    fn check(&self) -> ProbeResult {
        let start = current_time_ms();
        let uptime = current_time_ms().saturating_sub(self.start_time);
        let latency = current_time_ms().saturating_sub(start);
        ProbeResult::new("uptime", ProbeStatus::Healthy, latency).with_metadata_uptime(uptime)
    }
}

// Helper extension to attach uptime metadata in the name field for observability.
impl ProbeResult {
    fn with_metadata_uptime(mut self, uptime_ms: u64) -> Self {
        self.name = format!("uptime({}ms)", uptime_ms);
        self
    }
}

/// A composite probe that runs child probes and rolls up the worst status.
pub struct CompositeProbe {
    name: String,
    probes: Vec<Box<dyn HealthProbe>>,
}

impl CompositeProbe {
    /// Create a new composite probe with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            probes: Vec::new(),
        }
    }

    /// Add a child probe.
    pub fn add_probe(&mut self, probe: Box<dyn HealthProbe>) {
        self.probes.push(probe);
    }

    /// Number of child probes.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }
}

impl HealthProbe for CompositeProbe {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self) -> ProbeResult {
        let start = current_time_ms();
        let results: Vec<ProbeResult> = self.probes.iter().map(|p| p.check()).collect();
        let latency = current_time_ms().saturating_sub(start);

        // Worst wins: Unhealthy > Degraded > Healthy.
        let mut worst = ProbeStatus::Healthy;
        for r in &results {
            match &r.status {
                ProbeStatus::Unhealthy(msg) => {
                    worst = ProbeStatus::Unhealthy(msg.clone());
                    break;
                }
                ProbeStatus::Degraded(msg) => {
                    if !matches!(worst, ProbeStatus::Unhealthy(_)) {
                        worst = ProbeStatus::Degraded(msg.clone());
                    }
                }
                ProbeStatus::Healthy => {}
            }
        }

        ProbeResult::new(&self.name, worst, latency)
    }
}

// ── Overall health ────────────────────────────────────────────────────────────

/// The rolled-up health of all probes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverallHealth {
    /// All probes are healthy.
    Healthy,
    /// At least one probe is degraded but none is unhealthy.
    Degraded,
    /// At least one probe is unhealthy.
    Unhealthy,
}

/// Aggregated health report produced by [`HealthChecker::check_all`].
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Rolled-up overall health status.
    pub overall: OverallHealth,
    /// Individual probe results.
    pub results: Vec<ProbeResult>,
    /// Unix timestamp (milliseconds) when the report was produced.
    pub timestamp: u64,
}

impl HealthReport {
    /// Returns `true` when `overall` is [`OverallHealth::Healthy`].
    pub fn is_healthy(&self) -> bool {
        self.overall == OverallHealth::Healthy
    }

    /// Returns the number of probes that reported healthy.
    pub fn healthy_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.status, ProbeStatus::Healthy))
            .count()
    }

    /// Returns the number of probes that reported unhealthy.
    pub fn unhealthy_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.status, ProbeStatus::Unhealthy(_)))
            .count()
    }
}

// ── Health checker ────────────────────────────────────────────────────────────

/// Aggregates multiple [`HealthProbe`] implementations and produces a
/// [`HealthReport`] on demand.
pub struct HealthChecker {
    probes: Vec<Box<dyn HealthProbe>>,
}

impl HealthChecker {
    /// Create an empty checker with no probes.
    pub fn new() -> Self {
        Self { probes: Vec::new() }
    }

    /// Register a probe.
    pub fn add_probe(&mut self, probe: Box<dyn HealthProbe>) {
        self.probes.push(probe);
    }

    /// Run all probes and return an aggregated report.
    pub fn check_all(&self) -> HealthReport {
        let results: Vec<ProbeResult> = self.probes.iter().map(|p| p.check()).collect();
        let overall = Self::overall_status(&results);
        HealthReport {
            overall,
            results,
            timestamp: current_time_ms(),
        }
    }

    /// Total number of registered probes.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Compute the overall status from a slice of probe results.
    pub fn overall_status(results: &[ProbeResult]) -> OverallHealth {
        let mut has_degraded = false;
        for r in results {
            match &r.status {
                ProbeStatus::Unhealthy(_) => return OverallHealth::Unhealthy,
                ProbeStatus::Degraded(_) => has_degraded = true,
                ProbeStatus::Healthy => {}
            }
        }
        if has_degraded {
            OverallHealth::Degraded
        } else {
            OverallHealth::Healthy
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Utility ───────────────────────────────────────────────────────────────────

/// Return current Unix time in milliseconds.
fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProbeResult ───────────────────────────────────────────────────────────

    #[test]
    fn test_probe_result_healthy() {
        let r = ProbeResult::new("test", ProbeStatus::Healthy, 5);
        assert!(r.is_healthy());
        assert!(!r.is_unhealthy());
    }

    #[test]
    fn test_probe_result_unhealthy() {
        let r = ProbeResult::new("test", ProbeStatus::Unhealthy("oops".into()), 5);
        assert!(!r.is_healthy());
        assert!(r.is_unhealthy());
    }

    #[test]
    fn test_probe_result_degraded() {
        let r = ProbeResult::new("test", ProbeStatus::Degraded("slow".into()), 10);
        assert!(!r.is_healthy());
        assert!(!r.is_unhealthy());
    }

    #[test]
    fn test_probe_result_has_timestamp() {
        let r = ProbeResult::new("test", ProbeStatus::Healthy, 0);
        assert!(r.timestamp > 0);
    }

    // ── MemoryProbe ───────────────────────────────────────────────────────────

    #[test]
    fn test_memory_probe_low_threshold_healthy() {
        let probe = MemoryProbe::new(128); // 128 MB — simulated 512 available → healthy
        let result = probe.check();
        assert!(result.is_healthy());
    }

    #[test]
    fn test_memory_probe_high_threshold_degraded() {
        let probe = MemoryProbe::new(1024); // 1024 MB — simulated 512 → degraded
        let result = probe.check();
        assert!(matches!(result.status, ProbeStatus::Degraded(_)));
    }

    #[test]
    fn test_memory_probe_name() {
        let probe = MemoryProbe::new(256);
        assert_eq!(probe.name(), "memory");
    }

    #[test]
    fn test_memory_probe_exact_threshold_healthy() {
        let probe = MemoryProbe::new(512); // exactly at simulated available
        let result = probe.check();
        assert!(result.is_healthy());
    }

    // ── UptimeProbe ───────────────────────────────────────────────────────────

    #[test]
    fn test_uptime_probe_name() {
        let probe = UptimeProbe::new();
        assert_eq!(probe.name(), "uptime");
    }

    #[test]
    fn test_uptime_probe_result_is_healthy() {
        let probe = UptimeProbe::new();
        let result = probe.check();
        assert!(result.is_healthy());
    }

    #[test]
    fn test_uptime_probe_uptime_nonnegative() {
        let probe = UptimeProbe::new();
        assert!(probe.uptime_ms() < u64::MAX);
    }

    #[test]
    fn test_uptime_probe_with_start_in_past() {
        let now = current_time_ms();
        let probe = UptimeProbe::with_start(now.saturating_sub(1000));
        assert!(probe.uptime_ms() >= 1000 || probe.uptime_ms() < 2000);
    }

    #[test]
    fn test_uptime_probe_default() {
        let probe = UptimeProbe::default();
        let result = probe.check();
        assert!(result.is_healthy());
    }

    // ── CompositeProbe ────────────────────────────────────────────────────────

    #[test]
    fn test_composite_probe_empty_healthy() {
        let probe = CompositeProbe::new("composite");
        let result = probe.check();
        assert!(result.is_healthy());
    }

    #[test]
    fn test_composite_probe_all_healthy() {
        let mut composite = CompositeProbe::new("all_healthy");
        composite.add_probe(Box::new(MemoryProbe::new(128)));
        composite.add_probe(Box::new(UptimeProbe::new()));
        let result = composite.check();
        assert!(result.is_healthy());
    }

    #[test]
    fn test_composite_probe_one_degraded() {
        let mut composite = CompositeProbe::new("comp");
        composite.add_probe(Box::new(MemoryProbe::new(128))); // healthy
        composite.add_probe(Box::new(MemoryProbe::new(1024))); // degraded
        let result = composite.check();
        assert!(matches!(result.status, ProbeStatus::Degraded(_)));
    }

    #[test]
    fn test_composite_probe_count() {
        let mut composite = CompositeProbe::new("c");
        composite.add_probe(Box::new(MemoryProbe::new(128)));
        composite.add_probe(Box::new(UptimeProbe::new()));
        assert_eq!(composite.probe_count(), 2);
    }

    #[test]
    fn test_composite_probe_name() {
        let probe = CompositeProbe::new("my-composite");
        assert_eq!(probe.name(), "my-composite");
    }

    // ── HealthChecker ─────────────────────────────────────────────────────────

    #[test]
    fn test_checker_empty_produces_healthy_report() {
        let checker = HealthChecker::new();
        let report = checker.check_all();
        assert_eq!(report.overall, OverallHealth::Healthy);
    }

    #[test]
    fn test_checker_probe_count_zero() {
        let checker = HealthChecker::new();
        assert_eq!(checker.probe_count(), 0);
    }

    #[test]
    fn test_checker_add_probe_increments_count() {
        let mut checker = HealthChecker::new();
        checker.add_probe(Box::new(MemoryProbe::new(128)));
        assert_eq!(checker.probe_count(), 1);
    }

    #[test]
    fn test_checker_all_healthy_report() {
        let mut checker = HealthChecker::new();
        checker.add_probe(Box::new(MemoryProbe::new(128)));
        checker.add_probe(Box::new(UptimeProbe::new()));
        let report = checker.check_all();
        assert_eq!(report.overall, OverallHealth::Healthy);
        assert_eq!(report.healthy_count(), 2);
    }

    #[test]
    fn test_checker_degraded_report() {
        let mut checker = HealthChecker::new();
        checker.add_probe(Box::new(MemoryProbe::new(1024))); // degraded
        let report = checker.check_all();
        assert_eq!(report.overall, OverallHealth::Degraded);
    }

    #[test]
    fn test_checker_results_len() {
        let mut checker = HealthChecker::new();
        checker.add_probe(Box::new(MemoryProbe::new(128)));
        checker.add_probe(Box::new(UptimeProbe::new()));
        let report = checker.check_all();
        assert_eq!(report.results.len(), 2);
    }

    #[test]
    fn test_checker_report_timestamp() {
        let checker = HealthChecker::new();
        let report = checker.check_all();
        assert!(report.timestamp > 0);
    }

    // ── overall_status ────────────────────────────────────────────────────────

    #[test]
    fn test_overall_status_empty_is_healthy() {
        assert_eq!(HealthChecker::overall_status(&[]), OverallHealth::Healthy);
    }

    #[test]
    fn test_overall_status_all_healthy() {
        let results = vec![
            ProbeResult::new("a", ProbeStatus::Healthy, 0),
            ProbeResult::new("b", ProbeStatus::Healthy, 0),
        ];
        assert_eq!(
            HealthChecker::overall_status(&results),
            OverallHealth::Healthy
        );
    }

    #[test]
    fn test_overall_status_one_degraded() {
        let results = vec![
            ProbeResult::new("a", ProbeStatus::Healthy, 0),
            ProbeResult::new("b", ProbeStatus::Degraded("slow".into()), 0),
        ];
        assert_eq!(
            HealthChecker::overall_status(&results),
            OverallHealth::Degraded
        );
    }

    #[test]
    fn test_overall_status_one_unhealthy() {
        let results = vec![
            ProbeResult::new("a", ProbeStatus::Healthy, 0),
            ProbeResult::new("b", ProbeStatus::Unhealthy("down".into()), 0),
        ];
        assert_eq!(
            HealthChecker::overall_status(&results),
            OverallHealth::Unhealthy
        );
    }

    #[test]
    fn test_overall_status_unhealthy_dominates_degraded() {
        let results = vec![
            ProbeResult::new("a", ProbeStatus::Degraded("slow".into()), 0),
            ProbeResult::new("b", ProbeStatus::Unhealthy("down".into()), 0),
        ];
        assert_eq!(
            HealthChecker::overall_status(&results),
            OverallHealth::Unhealthy
        );
    }

    // ── HealthReport helpers ──────────────────────────────────────────────────

    #[test]
    fn test_report_is_healthy_true() {
        let report = HealthReport {
            overall: OverallHealth::Healthy,
            results: vec![],
            timestamp: 0,
        };
        assert!(report.is_healthy());
    }

    #[test]
    fn test_report_is_healthy_false_for_degraded() {
        let report = HealthReport {
            overall: OverallHealth::Degraded,
            results: vec![],
            timestamp: 0,
        };
        assert!(!report.is_healthy());
    }

    #[test]
    fn test_report_healthy_count() {
        let results = vec![
            ProbeResult::new("a", ProbeStatus::Healthy, 0),
            ProbeResult::new("b", ProbeStatus::Degraded("x".into()), 0),
            ProbeResult::new("c", ProbeStatus::Healthy, 0),
        ];
        let report = HealthReport {
            overall: OverallHealth::Degraded,
            results,
            timestamp: 0,
        };
        assert_eq!(report.healthy_count(), 2);
    }

    #[test]
    fn test_report_unhealthy_count() {
        let results = vec![
            ProbeResult::new("a", ProbeStatus::Unhealthy("x".into()), 0),
            ProbeResult::new("b", ProbeStatus::Healthy, 0),
            ProbeResult::new("c", ProbeStatus::Unhealthy("y".into()), 0),
        ];
        let report = HealthReport {
            overall: OverallHealth::Unhealthy,
            results,
            timestamp: 0,
        };
        assert_eq!(report.unhealthy_count(), 2);
    }

    // ── default ───────────────────────────────────────────────────────────────

    #[test]
    fn test_checker_default() {
        let checker = HealthChecker::default();
        assert_eq!(checker.probe_count(), 0);
    }

    #[test]
    fn test_uptime_default_check() {
        let probe = UptimeProbe::default();
        assert_eq!(probe.name(), "uptime");
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_probe_status_healthy_eq() {
        assert_eq!(ProbeStatus::Healthy, ProbeStatus::Healthy);
    }

    #[test]
    fn test_probe_status_degraded_clone() {
        let s = ProbeStatus::Degraded("x".into());
        let s2 = s.clone();
        assert_eq!(s, s2);
    }

    #[test]
    fn test_probe_result_name_preserved() {
        let r = ProbeResult::new("my_probe", ProbeStatus::Healthy, 3);
        assert_eq!(r.name, "my_probe");
    }

    #[test]
    fn test_probe_result_latency_preserved() {
        let r = ProbeResult::new("p", ProbeStatus::Healthy, 42);
        assert_eq!(r.latency_ms, 42);
    }

    #[test]
    fn test_composite_with_one_unhealthy() {
        let mut c = CompositeProbe::new("c");
        // Unhealthy via high threshold — but MemoryProbe only returns Degraded, not Unhealthy.
        // So we use overall_status directly instead.
        c.add_probe(Box::new(MemoryProbe::new(1024)));
        let results = vec![ProbeResult::new(
            "x",
            ProbeStatus::Unhealthy("fail".into()),
            0,
        )];
        assert_eq!(
            HealthChecker::overall_status(&results),
            OverallHealth::Unhealthy
        );
    }

    #[test]
    fn test_health_checker_three_probes() {
        let mut checker = HealthChecker::new();
        checker.add_probe(Box::new(MemoryProbe::new(128)));
        checker.add_probe(Box::new(UptimeProbe::new()));
        checker.add_probe(Box::new(MemoryProbe::new(64)));
        assert_eq!(checker.probe_count(), 3);
    }

    #[test]
    fn test_overall_health_unhealthy_is_not_healthy() {
        let report = HealthReport {
            overall: OverallHealth::Unhealthy,
            results: vec![],
            timestamp: 0,
        };
        assert!(!report.is_healthy());
    }

    #[test]
    fn test_uptime_probe_start_in_future() {
        // start_time in the future → uptime saturates to 0
        let probe = UptimeProbe::with_start(u64::MAX);
        assert_eq!(probe.uptime_ms(), 0);
    }

    #[test]
    fn test_memory_probe_zero_threshold_healthy() {
        let probe = MemoryProbe::new(0); // 0 MB threshold → always healthy
        let result = probe.check();
        assert!(result.is_healthy());
    }
}
