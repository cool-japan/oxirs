//! Prometheus text exposition format metrics for Modbus operations
//!
//! All metric types use atomic integers internally so they are safe to
//! share across threads. Counters are monotonically increasing; gauges
//! may go up or down.
//!
//! The [`PrometheusExporter`] produces a text body compatible with the
//! Prometheus HTTP scraping endpoint (`text/plain; version=0.0.4`).

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

/// Monotonically increasing counter.
///
/// Suitable for total request counts, error counts, etc.
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Create a new counter initialised to zero.
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Increment by 1.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment by `n`.
    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// Integer gauge (can increase or decrease, including negative values).
///
/// Suitable for connection count, queue depth, etc.
#[derive(Debug, Default)]
pub struct Gauge {
    value: AtomicI64,
}

impl Gauge {
    /// Create a new gauge initialised to zero.
    pub fn new() -> Self {
        Self {
            value: AtomicI64::new(0),
        }
    }

    /// Set the gauge to an absolute value.
    pub fn set(&self, v: i64) {
        self.value.store(v, Ordering::Relaxed);
    }

    /// Increment by 1.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement by 1.
    pub fn dec(&self) {
        self.value.fetch_sub(1, Ordering::Relaxed);
    }

    /// Current value.
    pub fn get(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// Simple histogram that tracks a sum and count of observed values,
/// enabling the computation of arithmetic mean without storing all samples.
///
/// For a true histogram with configurable buckets, use the
/// `metrics` / `prometheus` crate – this implementation is intentionally
/// minimal so it has no external dependencies.
#[derive(Debug, Default)]
pub struct SummaryGauge {
    /// Sum of all observed values (stored as f64 bits in an AtomicU64)
    sum_bits: AtomicU64,
    /// Number of observations
    count: AtomicU64,
    /// Maximum observed value (updated with compare-exchange)
    max_bits: AtomicU64,
    /// Minimum observed value (updated with compare-exchange)
    min_bits: AtomicU64,
}

impl SummaryGauge {
    /// Create a new summary gauge.
    pub fn new() -> Self {
        Self {
            sum_bits: AtomicU64::new(0),
            count: AtomicU64::new(0),
            max_bits: AtomicU64::new(f64::NEG_INFINITY.to_bits()),
            min_bits: AtomicU64::new(f64::INFINITY.to_bits()),
        }
    }

    /// Record one observation.
    pub fn observe(&self, value: f64) {
        // Add to sum (approximate; not perfectly atomic across sum+count)
        let cur_sum = f64::from_bits(self.sum_bits.load(Ordering::Relaxed));
        self.sum_bits
            .store((cur_sum + value).to_bits(), Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        // Update max
        loop {
            let cur_max = f64::from_bits(self.max_bits.load(Ordering::Relaxed));
            if value <= cur_max {
                break;
            }
            if self
                .max_bits
                .compare_exchange(
                    cur_max.to_bits(),
                    value.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }

        // Update min
        loop {
            let cur_min = f64::from_bits(self.min_bits.load(Ordering::Relaxed));
            if value >= cur_min {
                break;
            }
            if self
                .min_bits
                .compare_exchange(
                    cur_min.to_bits(),
                    value.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    /// Arithmetic mean of all observed values (0.0 when no observations).
    pub fn mean(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let sum = f64::from_bits(self.sum_bits.load(Ordering::Relaxed));
        sum / count as f64
    }

    /// Total sum of all observations.
    pub fn sum(&self) -> f64 {
        f64::from_bits(self.sum_bits.load(Ordering::Relaxed))
    }

    /// Number of observations recorded.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Maximum observed value (`f64::NEG_INFINITY` when no observations).
    pub fn max(&self) -> f64 {
        f64::from_bits(self.max_bits.load(Ordering::Relaxed))
    }

    /// Minimum observed value (`f64::INFINITY` when no observations).
    pub fn min(&self) -> f64 {
        f64::from_bits(self.min_bits.load(Ordering::Relaxed))
    }
}

/// Snapshot of all metrics at a point in time.
///
/// All values are plain integers/floats (no atomics), suitable for
/// serialization and display.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    // ── Counters ──────────────────────────────────────────────────────────
    /// Total number of read operations attempted
    pub reads_total: u64,
    /// Total number of successful read operations
    pub reads_success: u64,
    /// Total number of failed read operations
    pub reads_failed: u64,
    /// Total number of write operations attempted
    pub writes_total: u64,
    /// Total number of successful write operations
    pub writes_success: u64,
    /// Total number of failed write operations
    pub writes_failed: u64,
    /// Total Modbus exception responses received
    pub exceptions_total: u64,
    /// Total CRC errors (RTU only)
    pub crc_errors_total: u64,
    /// Total timeout errors
    pub timeouts_total: u64,
    /// Total RDF triples generated
    pub triples_generated_total: u64,
    /// Total reconnection attempts
    pub reconnects_total: u64,

    // ── Gauges ────────────────────────────────────────────────────────────
    /// Currently active TCP connections
    pub active_connections: i64,
    /// Number of devices currently being polled
    pub polling_devices: i64,

    // ── Latency summaries ─────────────────────────────────────────────────
    /// Mean read latency in milliseconds
    pub read_latency_mean_ms: f64,
    /// Maximum read latency in milliseconds
    pub read_latency_max_ms: f64,
    /// Minimum read latency in milliseconds (0.0 when no reads)
    pub read_latency_min_ms: f64,
    /// Total read latency observations
    pub read_latency_count: u64,
}

/// Central metrics registry for a Modbus subsystem.
///
/// Create one instance per application (or per Modbus gateway) and share
/// it via [`Arc`].
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use oxirs_modbus::metrics::{ModbusMetrics, PrometheusExporter};
///
/// let metrics = Arc::new(ModbusMetrics::new());
///
/// // Record a successful read
/// metrics.reads_total.inc();
/// metrics.reads_success.inc();
/// metrics.read_latency_ms.observe(3.2);
///
/// // Expose as Prometheus text
/// let exporter = PrometheusExporter::new(Arc::clone(&metrics));
/// let body = exporter.render();
/// assert!(body.contains("modbus_reads_total"));
/// ```
pub struct ModbusMetrics {
    // Counters
    pub reads_total: Counter,
    pub reads_success: Counter,
    pub reads_failed: Counter,
    pub writes_total: Counter,
    pub writes_success: Counter,
    pub writes_failed: Counter,
    pub exceptions_total: Counter,
    pub crc_errors_total: Counter,
    pub timeouts_total: Counter,
    pub triples_generated_total: Counter,
    pub reconnects_total: Counter,

    // Gauges
    pub active_connections: Gauge,
    pub polling_devices: Gauge,

    // Latency
    pub read_latency_ms: SummaryGauge,
}

impl ModbusMetrics {
    /// Create a new metrics registry with all counters/gauges initialised to zero.
    pub fn new() -> Self {
        Self {
            reads_total: Counter::new(),
            reads_success: Counter::new(),
            reads_failed: Counter::new(),
            writes_total: Counter::new(),
            writes_success: Counter::new(),
            writes_failed: Counter::new(),
            exceptions_total: Counter::new(),
            crc_errors_total: Counter::new(),
            timeouts_total: Counter::new(),
            triples_generated_total: Counter::new(),
            reconnects_total: Counter::new(),
            active_connections: Gauge::new(),
            polling_devices: Gauge::new(),
            read_latency_ms: SummaryGauge::new(),
        }
    }

    /// Record a completed read operation.
    ///
    /// * `success` - `true` for successful reads, `false` for failures.
    /// * `latency_ms` - Round-trip time in milliseconds.
    pub fn record_read(&self, success: bool, latency_ms: f64) {
        self.reads_total.inc();
        if success {
            self.reads_success.inc();
        } else {
            self.reads_failed.inc();
        }
        self.read_latency_ms.observe(latency_ms);
    }

    /// Record a completed write operation.
    ///
    /// * `success` - `true` for successful writes, `false` for failures.
    pub fn record_write(&self, success: bool) {
        self.writes_total.inc();
        if success {
            self.writes_success.inc();
        } else {
            self.writes_failed.inc();
        }
    }

    /// Record a Modbus exception response.
    pub fn record_exception(&self) {
        self.exceptions_total.inc();
        self.reads_failed.inc();
    }

    /// Record a CRC error (RTU transport).
    pub fn record_crc_error(&self) {
        self.crc_errors_total.inc();
        self.reads_failed.inc();
    }

    /// Record a timeout.
    pub fn record_timeout(&self) {
        self.timeouts_total.inc();
    }

    /// Record successful generation of RDF triples.
    pub fn record_triples(&self, count: u64) {
        self.triples_generated_total.inc_by(count);
    }

    /// Record a reconnection attempt.
    pub fn record_reconnect(&self) {
        self.reconnects_total.inc();
    }

    /// Take a snapshot of all metrics (non-atomic; values may be
    /// slightly inconsistent under concurrent writes).
    pub fn snapshot(&self) -> MetricsSnapshot {
        let min_latency = self.read_latency_ms.min();
        let min_latency_display = if min_latency.is_infinite() {
            0.0
        } else {
            min_latency
        };

        MetricsSnapshot {
            reads_total: self.reads_total.get(),
            reads_success: self.reads_success.get(),
            reads_failed: self.reads_failed.get(),
            writes_total: self.writes_total.get(),
            writes_success: self.writes_success.get(),
            writes_failed: self.writes_failed.get(),
            exceptions_total: self.exceptions_total.get(),
            crc_errors_total: self.crc_errors_total.get(),
            timeouts_total: self.timeouts_total.get(),
            triples_generated_total: self.triples_generated_total.get(),
            reconnects_total: self.reconnects_total.get(),
            active_connections: self.active_connections.get(),
            polling_devices: self.polling_devices.get(),
            read_latency_mean_ms: self.read_latency_ms.mean(),
            read_latency_max_ms: {
                let v = self.read_latency_ms.max();
                if v.is_infinite() {
                    0.0
                } else {
                    v
                }
            },
            read_latency_min_ms: min_latency_display,
            read_latency_count: self.read_latency_ms.count(),
        }
    }
}

impl Default for ModbusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Renders [`ModbusMetrics`] in the Prometheus text exposition format
/// (version 0.0.4).
///
/// Mount the output at `/metrics` to make it scrapeable by a Prometheus server.
pub struct PrometheusExporter {
    metrics: Arc<ModbusMetrics>,
    /// Optional static label applied to all metrics (e.g. `instance="gw1"`)
    instance_label: Option<String>,
}

impl PrometheusExporter {
    /// Create an exporter without an instance label.
    pub fn new(metrics: Arc<ModbusMetrics>) -> Self {
        Self {
            metrics,
            instance_label: None,
        }
    }

    /// Set a static instance label attached to every metric.
    pub fn with_instance(mut self, instance: impl Into<String>) -> Self {
        self.instance_label = Some(instance.into());
        self
    }

    /// Render all metrics in the Prometheus text exposition format.
    ///
    /// # Returns
    ///
    /// A `String` with `Content-Type: text/plain; version=0.0.4` semantics.
    pub fn render(&self) -> String {
        let snap = self.metrics.snapshot();
        let lbl = self
            .instance_label
            .as_ref()
            .map(|l| format!("{{instance=\"{}\"}}", l))
            .unwrap_or_default();

        let mut out = String::with_capacity(2048);

        // Helper macro to keep the format calls uniform
        macro_rules! counter {
            ($name:expr, $help:expr, $val:expr) => {
                out.push_str(&format!("# HELP {} {}\n", $name, $help));
                out.push_str(&format!("# TYPE {} counter\n", $name));
                out.push_str(&format!("{}{} {}\n", $name, lbl, $val));
            };
        }
        macro_rules! gauge {
            ($name:expr, $help:expr, $val:expr) => {
                out.push_str(&format!("# HELP {} {}\n", $name, $help));
                out.push_str(&format!("# TYPE {} gauge\n", $name));
                out.push_str(&format!("{}{} {}\n", $name, lbl, $val));
            };
        }

        counter!(
            "modbus_reads_total",
            "Total number of Modbus read operations attempted",
            snap.reads_total
        );
        counter!(
            "modbus_reads_success_total",
            "Total number of successful Modbus read operations",
            snap.reads_success
        );
        counter!(
            "modbus_reads_failed_total",
            "Total number of failed Modbus read operations",
            snap.reads_failed
        );
        counter!(
            "modbus_writes_total",
            "Total number of Modbus write operations attempted",
            snap.writes_total
        );
        counter!(
            "modbus_writes_success_total",
            "Total number of successful Modbus write operations",
            snap.writes_success
        );
        counter!(
            "modbus_writes_failed_total",
            "Total number of failed Modbus write operations",
            snap.writes_failed
        );
        counter!(
            "modbus_exceptions_total",
            "Total number of Modbus exception responses received",
            snap.exceptions_total
        );
        counter!(
            "modbus_crc_errors_total",
            "Total number of CRC checksum errors (RTU transport)",
            snap.crc_errors_total
        );
        counter!(
            "modbus_timeouts_total",
            "Total number of request timeouts",
            snap.timeouts_total
        );
        counter!(
            "modbus_triples_generated_total",
            "Total number of RDF triples generated from register readings",
            snap.triples_generated_total
        );
        counter!(
            "modbus_reconnects_total",
            "Total number of reconnection attempts",
            snap.reconnects_total
        );

        gauge!(
            "modbus_active_connections",
            "Number of currently active TCP connections",
            snap.active_connections
        );
        gauge!(
            "modbus_polling_devices",
            "Number of devices currently being polled",
            snap.polling_devices
        );

        // Latency summary metrics
        out.push_str("# HELP modbus_read_latency_milliseconds Modbus read round-trip latency in milliseconds\n");
        out.push_str("# TYPE modbus_read_latency_milliseconds summary\n");
        out.push_str(&format!(
            "modbus_read_latency_milliseconds_sum{} {}\n",
            lbl,
            self.metrics.read_latency_ms.sum()
        ));
        out.push_str(&format!(
            "modbus_read_latency_milliseconds_count{} {}\n",
            lbl, snap.read_latency_count
        ));
        out.push_str(&format!(
            "modbus_read_latency_milliseconds_mean{} {:.3}\n",
            lbl, snap.read_latency_mean_ms
        ));
        out.push_str(&format!(
            "modbus_read_latency_milliseconds_max{} {:.3}\n",
            lbl, snap.read_latency_max_ms
        ));
        out.push_str(&format!(
            "modbus_read_latency_milliseconds_min{} {:.3}\n",
            lbl, snap.read_latency_min_ms
        ));

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_increments() {
        let c = Counter::new();
        assert_eq!(c.get(), 0);
        c.inc();
        c.inc();
        assert_eq!(c.get(), 2);
        c.inc_by(10);
        assert_eq!(c.get(), 12);
    }

    #[test]
    fn test_gauge_operations() {
        let g = Gauge::new();
        g.set(100);
        assert_eq!(g.get(), 100);
        g.inc();
        assert_eq!(g.get(), 101);
        g.dec();
        g.dec();
        assert_eq!(g.get(), 99);
        g.set(-5);
        assert_eq!(g.get(), -5);
    }

    #[test]
    fn test_summary_gauge_mean() {
        let sg = SummaryGauge::new();
        sg.observe(10.0);
        sg.observe(20.0);
        sg.observe(30.0);

        assert_eq!(sg.count(), 3);
        let mean = sg.mean();
        assert!(
            (mean - 20.0).abs() < 1e-9,
            "mean should be 20.0, got {}",
            mean
        );
    }

    #[test]
    fn test_summary_gauge_min_max() {
        let sg = SummaryGauge::new();
        sg.observe(5.0);
        sg.observe(100.0);
        sg.observe(1.0);

        assert!((sg.min() - 1.0).abs() < 1e-9);
        assert!((sg.max() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_summary_gauge_empty() {
        let sg = SummaryGauge::new();
        assert_eq!(sg.count(), 0);
        assert_eq!(sg.mean(), 0.0);
        assert_eq!(sg.sum(), 0.0);
    }

    #[test]
    fn test_metrics_record_read_success() {
        let m = ModbusMetrics::new();
        m.record_read(true, 4.5);
        m.record_read(true, 3.0);
        m.record_read(false, 100.0);

        let snap = m.snapshot();
        assert_eq!(snap.reads_total, 3);
        assert_eq!(snap.reads_success, 2);
        assert_eq!(snap.reads_failed, 1);
        assert!(snap.read_latency_mean_ms > 0.0);
    }

    #[test]
    fn test_metrics_record_write() {
        let m = ModbusMetrics::new();
        m.record_write(true);
        m.record_write(false);

        let snap = m.snapshot();
        assert_eq!(snap.writes_total, 2);
        assert_eq!(snap.writes_success, 1);
        assert_eq!(snap.writes_failed, 1);
    }

    #[test]
    fn test_metrics_record_exception() {
        let m = ModbusMetrics::new();
        m.record_exception();
        m.record_exception();

        let snap = m.snapshot();
        assert_eq!(snap.exceptions_total, 2);
        assert_eq!(snap.reads_failed, 2); // exceptions also increment failed reads
    }

    #[test]
    fn test_metrics_record_triples() {
        let m = ModbusMetrics::new();
        m.record_triples(100);
        m.record_triples(50);

        assert_eq!(m.snapshot().triples_generated_total, 150);
    }

    #[test]
    fn test_prometheus_exporter_render() {
        let metrics = Arc::new(ModbusMetrics::new());
        metrics.record_read(true, 5.0);
        metrics.record_write(true);
        metrics.active_connections.set(3);

        let exporter = PrometheusExporter::new(Arc::clone(&metrics));
        let output = exporter.render();

        assert!(output.contains("modbus_reads_total"), "missing reads_total");
        assert!(
            output.contains("modbus_writes_total"),
            "missing writes_total"
        );
        assert!(
            output.contains("modbus_active_connections"),
            "missing active_connections"
        );
        assert!(
            output.contains("modbus_read_latency_milliseconds"),
            "missing latency"
        );
        // Values should be present
        assert!(
            output.contains("modbus_reads_total 1"),
            "wrong reads_total value"
        );
        assert!(
            output.contains("modbus_active_connections 3"),
            "wrong active_connections value"
        );
    }

    #[test]
    fn test_prometheus_exporter_with_instance_label() {
        let metrics = Arc::new(ModbusMetrics::new());
        metrics.reads_total.inc();

        let output = PrometheusExporter::new(Arc::clone(&metrics))
            .with_instance("gw_01")
            .render();

        assert!(
            output.contains("instance=\"gw_01\""),
            "missing instance label"
        );
    }

    #[test]
    fn test_metrics_snapshot_consistency() {
        let m = ModbusMetrics::new();
        for _ in 0..5 {
            m.record_read(true, 2.5);
        }
        for _ in 0..2 {
            m.record_read(false, 50.0);
        }

        let snap = m.snapshot();
        assert_eq!(snap.reads_total, snap.reads_success + snap.reads_failed);
    }

    #[test]
    fn test_reconnect_tracking() {
        let m = ModbusMetrics::new();
        m.record_reconnect();
        m.record_reconnect();
        m.record_reconnect();

        assert_eq!(m.snapshot().reconnects_total, 3);
    }

    #[test]
    fn test_latency_extremes() {
        let m = ModbusMetrics::new();
        m.record_read(true, 1.0);
        m.record_read(true, 999.0);

        let snap = m.snapshot();
        assert!(snap.read_latency_min_ms < 2.0, "min should be ~1.0");
        assert!(snap.read_latency_max_ms > 998.0, "max should be ~999.0");
    }
}
