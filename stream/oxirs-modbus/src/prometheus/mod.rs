//! Extended Prometheus / OpenMetrics export for Modbus register values.
//!
//! Complements the existing `metrics::PrometheusExporter` (which tracks
//! internal operational counters) with the ability to expose *device-level*
//! register readings as labelled gauge metrics in the Prometheus text format
//! (OpenMetrics compatible).
//!
//! # Components
//!
//! | Type | Role |
//! |------|------|
//! | `ModbusPrometheusExporter` | Metric registry + text renderer |
//! | `ModbusMetricsCollector`  | Maps register mappings → metric samples |
//! | `MetricSample`            | Snapshot of a single named measurement |
//!
//! # Example
//!
//! ```
//! use oxirs_modbus::prometheus::{ModbusPrometheusExporter, MetricDefinition};
//!
//! let mut exp = ModbusPrometheusExporter::new();
//! exp.register_metric(MetricDefinition {
//!     name: "modbus_temperature_celsius".to_string(),
//!     help: "Temperature reading from holding register".to_string(),
//!     label_names: vec!["unit_id".to_string(), "address".to_string()],
//! });
//! exp.update_gauge("modbus_temperature_celsius", 23.5, &[("unit_id", "1"), ("address", "0")]);
//! let text = exp.render_metrics();
//! assert!(text.contains("modbus_temperature_celsius"));
//! ```

use std::collections::HashMap;
use std::fmt::Write;

use crate::samm::RegisterMapping;

// ── MetricDefinition ──────────────────────────────────────────────────────

/// Static definition of a Prometheus gauge metric.
#[derive(Debug, Clone)]
pub struct MetricDefinition {
    /// Prometheus metric name (must match `[a-zA-Z_:][a-zA-Z0-9_:]*`).
    pub name: String,
    /// Human-readable description shown in `# HELP` lines.
    pub help: String,
    /// Ordered list of label names for this metric.
    pub label_names: Vec<String>,
}

// ── GaugeSeries ───────────────────────────────────────────────────────────

/// A single labelled sample stored inside the exporter.
#[derive(Debug, Clone)]
struct GaugeSample {
    /// Label key=value pairs in the *same order* as `MetricDefinition::label_names`.
    labels: Vec<(String, String)>,
    /// Current gauge value.
    value: f64,
}

// ── ModbusPrometheusExporter ──────────────────────────────────────────────

/// Stores Prometheus gauge metrics and renders them in the standard text
/// exposition format (compatible with OpenMetrics).
///
/// Thread-safety: this struct is **not** internally synchronised.  Wrap in an
/// `Arc<Mutex<…>>` when sharing across threads.
#[derive(Debug, Default)]
pub struct ModbusPrometheusExporter {
    /// Metric definitions keyed by metric name.
    definitions: HashMap<String, MetricDefinition>,
    /// Current gauge samples, keyed by metric name.
    samples: HashMap<String, Vec<GaugeSample>>,
}

impl ModbusPrometheusExporter {
    /// Create an empty exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new gauge metric.
    ///
    /// If a metric with the same name already exists, it is replaced and all
    /// existing samples for it are cleared.
    pub fn register_metric(&mut self, def: MetricDefinition) {
        self.samples.remove(&def.name);
        self.definitions.insert(def.name.clone(), def);
    }

    /// Update (or insert) a gauge value for the given metric and label set.
    ///
    /// `labels` must be `(label_name, label_value)` pairs.  The label **names**
    /// are used to match existing samples; unknown label names are accepted
    /// (the exporter does not enforce the declared `label_names`).
    ///
    /// Returns `false` when no metric with `metric_name` is registered.
    pub fn update_gauge(&mut self, metric_name: &str, value: f64, labels: &[(&str, &str)]) -> bool {
        if !self.definitions.contains_key(metric_name) {
            return false;
        }
        let owned: Vec<(String, String)> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let series = self.samples.entry(metric_name.to_string()).or_default();
        // Update existing sample if label set matches, otherwise append.
        let existing = series.iter_mut().find(|s| labels_match(&s.labels, &owned));
        if let Some(sample) = existing {
            sample.value = value;
        } else {
            series.push(GaugeSample {
                labels: owned,
                value,
            });
        }
        true
    }

    /// Render all registered metrics in Prometheus text exposition format.
    ///
    /// Metrics with no samples still appear with their `# HELP` / `# TYPE` headers.
    pub fn render_metrics(&self) -> String {
        let mut out = String::new();
        // Sort metric names for deterministic output
        let mut names: Vec<&String> = self.definitions.keys().collect();
        names.sort();

        for name in names {
            let def = &self.definitions[name];

            // HELP and TYPE headers
            let help_escaped = def.help.replace('\\', "\\\\").replace('\n', "\\n");
            writeln!(out, "# HELP {name} {help_escaped}").ok();
            writeln!(out, "# TYPE {name} gauge").ok();

            if let Some(series) = self.samples.get(name) {
                for sample in series {
                    let label_str = format_labels(&sample.labels);
                    if sample.value.is_nan() || sample.value.is_infinite() {
                        // Prometheus requires finite float values for gauges
                        writeln!(out, "{name}{label_str} NaN").ok();
                    } else {
                        writeln!(out, "{name}{label_str} {}", sample.value).ok();
                    }
                }
            }
        }
        out
    }

    /// Return the current value of a gauge, if set.
    ///
    /// When multiple samples exist (different label sets) the *first* one is
    /// returned.  Use [`samples_for`](Self::samples_for) to inspect all.
    pub fn get_gauge(&self, metric_name: &str) -> Option<f64> {
        self.samples
            .get(metric_name)
            .and_then(|v| v.first())
            .map(|s| s.value)
    }

    /// Return all samples for a metric (useful for testing).
    pub fn samples_for(&self, metric_name: &str) -> Vec<(Vec<(String, String)>, f64)> {
        self.samples
            .get(metric_name)
            .map(|v| v.iter().map(|s| (s.labels.clone(), s.value)).collect())
            .unwrap_or_default()
    }

    /// Number of registered metric definitions.
    pub fn metric_count(&self) -> usize {
        self.definitions.len()
    }
}

// ── helpers ────────────────────────────────────────────────────────────────

/// Compare two label sets for equality (order-independent).
fn labels_match(a: &[(String, String)], b: &[(String, String)]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .all(|(k, v)| b.iter().any(|(bk, bv)| bk == k && bv == v))
}

/// Format a label set as `{k1="v1",k2="v2"}` or empty string.
fn format_labels(labels: &[(String, String)]) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let inner: Vec<String> = labels
        .iter()
        .map(|(k, v)| {
            let escaped = v
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            format!("{k}=\"{escaped}\"")
        })
        .collect();
    format!("{{{}}}", inner.join(","))
}

// ── MetricSample ──────────────────────────────────────────────────────────

/// A single timestamped measurement collected from a Modbus device.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricSample {
    /// Prometheus metric name (derived from the register mapping name).
    pub name: String,
    /// Decoded and scaled register value.
    pub value: f64,
    /// Label key/value pairs for Prometheus.
    pub labels: Vec<(String, String)>,
    /// Unix timestamp in milliseconds when the sample was collected.
    pub timestamp_ms: i64,
}

// ── ModbusMetricsCollector ────────────────────────────────────────────────

/// Converts Modbus register arrays into `MetricSample` vectors, ready for
/// ingestion into a `ModbusPrometheusExporter`.
#[derive(Debug, Clone)]
pub struct ModbusMetricsCollector {
    /// Base name prefix for all generated metric names (e.g. `"modbus"`).
    pub metric_prefix: String,
}

impl Default for ModbusMetricsCollector {
    fn default() -> Self {
        Self {
            metric_prefix: "modbus".to_string(),
        }
    }
}

impl ModbusMetricsCollector {
    /// Create a collector with the given prefix.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            metric_prefix: prefix.into(),
        }
    }

    /// Collect metric samples from holding registers.
    ///
    /// - `unit_id` — Modbus unit identifier (used as a label value).
    /// - `values` — raw register values starting at `base_address`.
    /// - `mappings` — semantic register mappings.
    /// - `base_address` — the first register address in `values` (often 0).
    /// - `timestamp_ms` — collection timestamp in milliseconds since Unix epoch.
    pub fn collect_holding_registers(
        &self,
        unit_id: u8,
        values: &[u16],
        mappings: &[RegisterMapping],
        base_address: u16,
        timestamp_ms: i64,
    ) -> Vec<MetricSample> {
        self.collect_internal(
            unit_id,
            values,
            mappings,
            base_address,
            timestamp_ms,
            "holding",
        )
    }

    /// Collect metric samples from input registers.
    pub fn collect_input_registers(
        &self,
        unit_id: u8,
        values: &[u16],
        mappings: &[RegisterMapping],
        base_address: u16,
        timestamp_ms: i64,
    ) -> Vec<MetricSample> {
        self.collect_internal(
            unit_id,
            values,
            mappings,
            base_address,
            timestamp_ms,
            "input",
        )
    }

    fn collect_internal(
        &self,
        unit_id: u8,
        values: &[u16],
        mappings: &[RegisterMapping],
        base_address: u16,
        timestamp_ms: i64,
        register_type: &str,
    ) -> Vec<MetricSample> {
        let mut samples = Vec::with_capacity(mappings.len());
        for m in mappings {
            if let Some(value) = m.decode_value(values, base_address) {
                let metric_name = format!(
                    "{}_{}_{}",
                    self.metric_prefix,
                    register_type,
                    sanitize_label(&m.name)
                );
                let labels = vec![
                    ("unit_id".to_string(), unit_id.to_string()),
                    ("address".to_string(), m.address.to_string()),
                    ("unit".to_string(), m.unit.clone()),
                ];
                samples.push(MetricSample {
                    name: metric_name,
                    value,
                    labels,
                    timestamp_ms,
                });
            }
        }
        samples
    }

    /// Feed samples into an exporter, auto-registering metrics if not present.
    pub fn feed_into(
        &self,
        exporter: &mut ModbusPrometheusExporter,
        samples: &[MetricSample],
        help_prefix: &str,
    ) {
        for sample in samples {
            // Auto-register if unknown
            if !exporter.definitions.contains_key(&sample.name) {
                let label_names: Vec<String> =
                    sample.labels.iter().map(|(k, _)| k.clone()).collect();
                exporter.register_metric(MetricDefinition {
                    name: sample.name.clone(),
                    help: format!("{help_prefix} {}", sample.name),
                    label_names,
                });
            }
            let label_refs: Vec<(&str, &str)> = sample
                .labels
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            exporter.update_gauge(&sample.name, sample.value, &label_refs);
        }
    }
}

/// Sanitise a register name into a valid Prometheus metric name component.
fn sanitize_label(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ── tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::samm::{RegisterDataType, RegisterMapping};

    fn make_mappings() -> Vec<RegisterMapping> {
        vec![
            RegisterMapping {
                address: 0,
                name: "temperature".to_string(),
                data_type: RegisterDataType::UInt16,
                scale: 0.1,
                unit: "CEL".to_string(),
            },
            RegisterMapping {
                address: 1,
                name: "humidity".to_string(),
                data_type: RegisterDataType::UInt16,
                scale: 0.1,
                unit: "PCT".to_string(),
            },
        ]
    }

    // ── MetricDefinition / registration ──────────────────────────────────

    #[test]
    fn test_register_metric_adds_definition() {
        let mut exp = ModbusPrometheusExporter::new();
        exp.register_metric(MetricDefinition {
            name: "test_gauge".to_string(),
            help: "A test gauge".to_string(),
            label_names: vec!["host".to_string()],
        });
        assert_eq!(exp.metric_count(), 1);
    }

    #[test]
    fn test_register_metric_replaces_existing() {
        let mut exp = ModbusPrometheusExporter::new();
        for _ in 0..3 {
            exp.register_metric(MetricDefinition {
                name: "same_name".to_string(),
                help: "help".to_string(),
                label_names: vec![],
            });
        }
        assert_eq!(exp.metric_count(), 1);
    }

    // ── update_gauge / get_gauge ──────────────────────────────────────────

    #[test]
    fn test_update_gauge_basic() {
        let mut exp = ModbusPrometheusExporter::new();
        exp.register_metric(MetricDefinition {
            name: "temp".to_string(),
            help: "temperature".to_string(),
            label_names: vec!["unit_id".to_string()],
        });
        let ok = exp.update_gauge("temp", 23.5, &[("unit_id", "1")]);
        assert!(ok);
        assert!((exp.get_gauge("temp").expect("should succeed") - 23.5).abs() < 1e-9);
    }

    #[test]
    fn test_update_gauge_unknown_metric_returns_false() {
        let mut exp = ModbusPrometheusExporter::new();
        assert!(!exp.update_gauge("nonexistent", 1.0, &[]));
    }

    #[test]
    fn test_update_gauge_updates_in_place() {
        let mut exp = ModbusPrometheusExporter::new();
        exp.register_metric(MetricDefinition {
            name: "g".to_string(),
            help: "g".to_string(),
            label_names: vec!["l".to_string()],
        });
        exp.update_gauge("g", 1.0, &[("l", "a")]);
        exp.update_gauge("g", 2.0, &[("l", "a")]);
        // Only one sample (same labels)
        assert_eq!(exp.samples_for("g").len(), 1);
        assert!((exp.get_gauge("g").expect("should succeed") - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_update_gauge_different_labels_appends() {
        let mut exp = ModbusPrometheusExporter::new();
        exp.register_metric(MetricDefinition {
            name: "g".to_string(),
            help: "g".to_string(),
            label_names: vec!["l".to_string()],
        });
        exp.update_gauge("g", 1.0, &[("l", "a")]);
        exp.update_gauge("g", 2.0, &[("l", "b")]);
        assert_eq!(exp.samples_for("g").len(), 2);
    }

    // ── render_metrics ────────────────────────────────────────────────────

    #[test]
    fn test_render_metrics_help_type_headers() {
        let mut exp = ModbusPrometheusExporter::new();
        exp.register_metric(MetricDefinition {
            name: "modbus_freq".to_string(),
            help: "Frequency in Hz".to_string(),
            label_names: vec![],
        });
        let text = exp.render_metrics();
        assert!(
            text.contains("# HELP modbus_freq Frequency in Hz"),
            "got: {text}"
        );
        assert!(text.contains("# TYPE modbus_freq gauge"), "got: {text}");
    }

    #[test]
    fn test_render_metrics_with_labels() {
        let mut exp = ModbusPrometheusExporter::new();
        exp.register_metric(MetricDefinition {
            name: "modbus_temp".to_string(),
            help: "Temp".to_string(),
            label_names: vec!["unit_id".to_string()],
        });
        exp.update_gauge("modbus_temp", 37.0, &[("unit_id", "5")]);
        let text = exp.render_metrics();
        assert!(text.contains("unit_id=\"5\""), "got: {text}");
        assert!(text.contains("37"), "got: {text}");
    }

    #[test]
    fn test_render_metrics_sorted_names() {
        let mut exp = ModbusPrometheusExporter::new();
        for name in &["z_metric", "a_metric", "m_metric"] {
            exp.register_metric(MetricDefinition {
                name: name.to_string(),
                help: "h".to_string(),
                label_names: vec![],
            });
        }
        let text = exp.render_metrics();
        let a_pos = text.find("a_metric").expect("should succeed");
        let m_pos = text.find("m_metric").expect("should succeed");
        let z_pos = text.find("z_metric").expect("should succeed");
        assert!(a_pos < m_pos, "a should precede m");
        assert!(m_pos < z_pos, "m should precede z");
    }

    #[test]
    fn test_render_no_samples_still_emits_header() {
        let mut exp = ModbusPrometheusExporter::new();
        exp.register_metric(MetricDefinition {
            name: "empty_gauge".to_string(),
            help: "no data".to_string(),
            label_names: vec![],
        });
        let text = exp.render_metrics();
        assert!(text.contains("# HELP empty_gauge"));
        assert!(
            !text.contains("empty_gauge{"),
            "no sample lines should appear"
        );
    }

    // ── ModbusMetricsCollector ────────────────────────────────────────────

    #[test]
    fn test_collect_holding_registers_basic() {
        let collector = ModbusMetricsCollector::new("modbus");
        let mappings = make_mappings();
        let values = vec![230u16, 650]; // 23.0, 65.0
        let samples =
            collector.collect_holding_registers(1, &values, &mappings, 0, 1_700_000_000_000);
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn test_collect_holding_registers_values() {
        let collector = ModbusMetricsCollector::new("modbus");
        let mappings = vec![RegisterMapping {
            address: 0,
            name: "voltage".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 0.1,
            unit: "V".to_string(),
        }];
        let samples = collector.collect_holding_registers(1, &[2300], &mappings, 0, 0);
        assert_eq!(samples.len(), 1);
        assert!(
            (samples[0].value - 230.0).abs() < 1e-9,
            "got {}",
            samples[0].value
        );
    }

    #[test]
    fn test_collect_holding_registers_labels() {
        let collector = ModbusMetricsCollector::new("mb");
        let mappings = vec![RegisterMapping {
            address: 5,
            name: "freq".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 0.01,
            unit: "HZ".to_string(),
        }];
        let values = vec![0u16; 6];
        let mut vals = values;
        vals[5] = 5000;
        let samples = collector.collect_holding_registers(3, &vals, &mappings, 0, 42);
        assert_eq!(samples.len(), 1);
        let s = &samples[0];
        assert!(s.labels.iter().any(|(k, v)| k == "unit_id" && v == "3"));
        assert!(s.labels.iter().any(|(k, v)| k == "address" && v == "5"));
        assert!(s.labels.iter().any(|(k, v)| k == "unit" && v == "HZ"));
    }

    #[test]
    fn test_collect_skips_out_of_range() {
        let collector = ModbusMetricsCollector::default();
        let mappings = vec![RegisterMapping {
            address: 99,
            name: "phantom".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 1.0,
            unit: "".to_string(),
        }];
        let samples = collector.collect_holding_registers(1, &[0u16; 5], &mappings, 0, 0);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_collect_input_registers() {
        let collector = ModbusMetricsCollector::new("modbus");
        let mappings = vec![RegisterMapping {
            address: 0,
            name: "current".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 0.001,
            unit: "A".to_string(),
        }];
        let samples = collector.collect_input_registers(2, &[1500], &mappings, 0, 0);
        assert_eq!(samples.len(), 1);
        let name = &samples[0].name;
        assert!(name.contains("input"), "got {name}");
    }

    #[test]
    fn test_feed_into_auto_registers() {
        let collector = ModbusMetricsCollector::new("m");
        let mappings = make_mappings();
        let values = vec![200u16, 700];
        let samples = collector.collect_holding_registers(1, &values, &mappings, 0, 0);

        let mut exp = ModbusPrometheusExporter::new();
        collector.feed_into(&mut exp, &samples, "Modbus register");
        assert_eq!(exp.metric_count(), 2);
    }

    #[test]
    fn test_feed_into_renders_values() {
        let collector = ModbusMetricsCollector::new("m");
        let mappings = vec![RegisterMapping {
            address: 0,
            name: "speed".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 1.0,
            unit: "RPM".to_string(),
        }];
        let samples = collector.collect_holding_registers(1, &[1450], &mappings, 0, 0);
        let mut exp = ModbusPrometheusExporter::new();
        collector.feed_into(&mut exp, &samples, "Help");
        let text = exp.render_metrics();
        assert!(text.contains("1450"), "got: {text}");
    }

    #[test]
    fn test_sanitize_name_with_spaces() {
        // Names with spaces/special chars should be sanitised
        let collector = ModbusMetricsCollector::new("mb");
        let mappings = vec![RegisterMapping {
            address: 0,
            name: "pump flow rate".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 1.0,
            unit: "LPM".to_string(),
        }];
        let samples = collector.collect_holding_registers(1, &[500], &mappings, 0, 0);
        assert!(
            !samples[0].name.contains(' '),
            "spaces in metric name: {}",
            samples[0].name
        );
    }

    #[test]
    fn test_metric_sample_timestamp() {
        let collector = ModbusMetricsCollector::new("t");
        let mappings = vec![RegisterMapping {
            address: 0,
            name: "x".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 1.0,
            unit: "".to_string(),
        }];
        let ts = 1_700_000_000_123_i64;
        let samples = collector.collect_holding_registers(1, &[7], &mappings, 0, ts);
        assert_eq!(samples[0].timestamp_ms, ts);
    }
}
