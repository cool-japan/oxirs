//! # Modbus Data Logger
//!
//! Provides configurable data logging for Modbus register values with ring buffer
//! storage, CSV/JSON export, and threshold-based alerting.
//!
//! ## Features
//!
//! - **Configurable polling intervals**: Per-register polling schedule
//! - **Ring buffer storage**: Bounded in-memory storage with automatic eviction
//! - **CSV/JSON export**: Serialize logged data to CSV or JSON format
//! - **Threshold alerting**: Configurable high/low thresholds with alert generation
//! - **Statistics**: Min, max, mean, standard deviation over logged values
//! - **Tag-based filtering**: Filter log entries by tags

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::Duration;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Modbus data logger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoggerConfig {
    /// Maximum number of log entries to keep in the ring buffer (default: 10_000).
    pub max_entries: usize,
    /// Default polling interval (default: 1s).
    pub default_poll_interval: Duration,
    /// Whether to enable threshold alerting (default: true).
    pub enable_alerting: bool,
    /// Maximum number of alerts to keep (default: 1_000).
    pub max_alerts: usize,
}

impl Default for DataLoggerConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            default_poll_interval: Duration::from_secs(1),
            enable_alerting: true,
            max_alerts: 1_000,
        }
    }
}

/// Configuration for a single register channel to log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Unique channel name.
    pub name: String,
    /// Modbus unit (slave) ID.
    pub unit_id: u8,
    /// Register address.
    pub address: u16,
    /// Optional description.
    pub description: Option<String>,
    /// Engineering unit (e.g., "degC", "bar").
    pub unit: Option<String>,
    /// Polling interval override for this channel.
    pub poll_interval: Option<Duration>,
    /// Optional tags for filtering.
    pub tags: Vec<String>,
    /// Optional high threshold for alerting.
    pub high_threshold: Option<f64>,
    /// Optional low threshold for alerting.
    pub low_threshold: Option<f64>,
    /// Scale factor to apply to raw register value.
    pub scale: f64,
    /// Offset to apply after scaling.
    pub offset: f64,
}

impl ChannelConfig {
    /// Create a new channel config with default scale/offset.
    pub fn new(name: impl Into<String>, unit_id: u8, address: u16) -> Self {
        Self {
            name: name.into(),
            unit_id,
            address,
            description: None,
            unit: None,
            poll_interval: None,
            tags: Vec::new(),
            high_threshold: None,
            low_threshold: None,
            scale: 1.0,
            offset: 0.0,
        }
    }

    /// Set thresholds.
    pub fn with_thresholds(mut self, low: Option<f64>, high: Option<f64>) -> Self {
        self.low_threshold = low;
        self.high_threshold = high;
        self
    }

    /// Set unit.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Set tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Apply scaling to a raw value.
    pub fn apply_scaling(&self, raw: u16) -> f64 {
        (raw as f64) * self.scale + self.offset
    }
}

// ─────────────────────────────────────────────
// Log entry
// ─────────────────────────────────────────────

/// A single logged data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Channel name.
    pub channel: String,
    /// Raw register value.
    pub raw_value: u16,
    /// Scaled/engineered value.
    pub scaled_value: f64,
    /// Timestamp of the reading.
    pub timestamp: DateTime<Utc>,
    /// Quality indicator (0.0 = bad, 1.0 = good).
    pub quality: f64,
    /// Tags inherited from channel config.
    pub tags: Vec<String>,
}

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum AlertSeverity {
    /// Value exceeded high threshold.
    High,
    /// Value dropped below low threshold.
    Low,
    /// Value returned to normal range.
    Cleared,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertSeverity::High => write!(f, "HIGH"),
            AlertSeverity::Low => write!(f, "LOW"),
            AlertSeverity::Cleared => write!(f, "CLEARED"),
        }
    }
}

/// A threshold alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Channel that triggered the alert.
    pub channel: String,
    /// Severity.
    pub severity: AlertSeverity,
    /// The value that triggered the alert.
    pub value: f64,
    /// The threshold that was exceeded.
    pub threshold: f64,
    /// When the alert was generated.
    pub timestamp: DateTime<Utc>,
    /// Alert message.
    pub message: String,
}

/// Statistics for a channel's logged data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStats {
    /// Channel name.
    pub channel: String,
    /// Number of data points.
    pub count: usize,
    /// Minimum scaled value.
    pub min: f64,
    /// Maximum scaled value.
    pub max: f64,
    /// Mean scaled value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Number of alerts generated.
    pub alert_count: usize,
}

/// Export format for logged data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Comma-separated values.
    Csv,
    /// JSON array.
    Json,
}

// ─────────────────────────────────────────────
// ModbusDataLogger
// ─────────────────────────────────────────────

/// The main Modbus data logger.
pub struct ModbusDataLogger {
    config: DataLoggerConfig,
    channels: HashMap<String, ChannelConfig>,
    entries: VecDeque<LogEntry>,
    alerts: VecDeque<Alert>,
    /// Per-channel alert state: whether currently in alert
    alert_active: HashMap<String, AlertSeverity>,
}

impl ModbusDataLogger {
    /// Create a new data logger with default configuration.
    pub fn new() -> Self {
        Self::with_config(DataLoggerConfig::default())
    }

    /// Create a new data logger with the given configuration.
    pub fn with_config(config: DataLoggerConfig) -> Self {
        Self {
            config,
            channels: HashMap::new(),
            entries: VecDeque::new(),
            alerts: VecDeque::new(),
            alert_active: HashMap::new(),
        }
    }

    /// Register a channel for logging.
    pub fn add_channel(&mut self, channel: ChannelConfig) {
        self.channels.insert(channel.name.clone(), channel);
    }

    /// Remove a channel.
    pub fn remove_channel(&mut self, name: &str) -> bool {
        self.channels.remove(name).is_some()
    }

    /// Get registered channel count.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get the number of log entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of alerts.
    pub fn alert_count(&self) -> usize {
        self.alerts.len()
    }

    /// Log a raw register reading for a channel.
    pub fn log_reading(&mut self, channel_name: &str, raw_value: u16) -> Option<Alert> {
        self.log_reading_with_quality(channel_name, raw_value, 1.0)
    }

    /// Log a raw register reading with a quality indicator.
    pub fn log_reading_with_quality(
        &mut self,
        channel_name: &str,
        raw_value: u16,
        quality: f64,
    ) -> Option<Alert> {
        let channel = self.channels.get(channel_name)?;
        let scaled_value = channel.apply_scaling(raw_value);
        let tags = channel.tags.clone();
        let high_threshold = channel.high_threshold;
        let low_threshold = channel.low_threshold;
        let channel_name_owned = channel_name.to_string();

        let entry = LogEntry {
            channel: channel_name_owned.clone(),
            raw_value,
            scaled_value,
            timestamp: Utc::now(),
            quality,
            tags,
        };

        // Add to ring buffer
        if self.entries.len() >= self.config.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);

        // Check thresholds
        if !self.config.enable_alerting {
            return None;
        }

        self.check_thresholds(
            &channel_name_owned,
            scaled_value,
            high_threshold,
            low_threshold,
        )
    }

    fn check_thresholds(
        &mut self,
        channel_name: &str,
        value: f64,
        high: Option<f64>,
        low: Option<f64>,
    ) -> Option<Alert> {
        let currently_active = self.alert_active.get(channel_name).copied();

        if let Some(high_val) = high {
            if value > high_val {
                if currently_active != Some(AlertSeverity::High) {
                    self.alert_active
                        .insert(channel_name.to_string(), AlertSeverity::High);
                    let alert = Alert {
                        channel: channel_name.to_string(),
                        severity: AlertSeverity::High,
                        value,
                        threshold: high_val,
                        timestamp: Utc::now(),
                        message: format!(
                            "Channel '{}' value {:.2} exceeds high threshold {:.2}",
                            channel_name, value, high_val
                        ),
                    };
                    self.push_alert(alert.clone());
                    return Some(alert);
                }
                return None;
            }
        }

        if let Some(low_val) = low {
            if value < low_val {
                if currently_active != Some(AlertSeverity::Low) {
                    self.alert_active
                        .insert(channel_name.to_string(), AlertSeverity::Low);
                    let alert = Alert {
                        channel: channel_name.to_string(),
                        severity: AlertSeverity::Low,
                        value,
                        threshold: low_val,
                        timestamp: Utc::now(),
                        message: format!(
                            "Channel '{}' value {:.2} below low threshold {:.2}",
                            channel_name, value, low_val
                        ),
                    };
                    self.push_alert(alert.clone());
                    return Some(alert);
                }
                return None;
            }
        }

        // Value in range: clear any active alert
        if currently_active.is_some() && currently_active != Some(AlertSeverity::Cleared) {
            self.alert_active
                .insert(channel_name.to_string(), AlertSeverity::Cleared);
            let threshold_val = high.or(low).unwrap_or(0.0);
            let alert = Alert {
                channel: channel_name.to_string(),
                severity: AlertSeverity::Cleared,
                value,
                threshold: threshold_val,
                timestamp: Utc::now(),
                message: format!(
                    "Channel '{}' value {:.2} returned to normal range",
                    channel_name, value
                ),
            };
            self.push_alert(alert.clone());
            return Some(alert);
        }

        None
    }

    fn push_alert(&mut self, alert: Alert) {
        if self.alerts.len() >= self.config.max_alerts {
            self.alerts.pop_front();
        }
        self.alerts.push_back(alert);
    }

    /// Get entries for a specific channel.
    pub fn entries_for_channel(&self, channel_name: &str) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.channel == channel_name)
            .collect()
    }

    /// Get entries matching a tag.
    pub fn entries_with_tag(&self, tag: &str) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Get entries within a time range.
    pub fn entries_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Compute statistics for a channel.
    pub fn channel_stats(&self, channel_name: &str) -> Option<ChannelStats> {
        let values: Vec<f64> = self
            .entries
            .iter()
            .filter(|e| e.channel == channel_name)
            .map(|e| e.scaled_value)
            .collect();

        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let alert_count = self
            .alerts
            .iter()
            .filter(|a| a.channel == channel_name)
            .count();

        Some(ChannelStats {
            channel: channel_name.to_string(),
            count,
            min,
            max,
            mean,
            std_dev,
            alert_count,
        })
    }

    /// Get all alerts for a channel.
    pub fn alerts_for_channel(&self, channel_name: &str) -> Vec<&Alert> {
        self.alerts
            .iter()
            .filter(|a| a.channel == channel_name)
            .collect()
    }

    /// Export logged data to CSV format.
    pub fn export_csv(&self) -> String {
        let mut csv = String::from("timestamp,channel,raw_value,scaled_value,quality,tags\n");
        for entry in &self.entries {
            let tags_str = entry.tags.join(";");
            csv.push_str(&format!(
                "{},{},{},{:.4},{:.2},{}\n",
                entry.timestamp.to_rfc3339(),
                entry.channel,
                entry.raw_value,
                entry.scaled_value,
                entry.quality,
                tags_str,
            ));
        }
        csv
    }

    /// Export logged data for a specific channel to CSV.
    pub fn export_channel_csv(&self, channel_name: &str) -> String {
        let mut csv = String::from("timestamp,raw_value,scaled_value,quality\n");
        for entry in self.entries.iter().filter(|e| e.channel == channel_name) {
            csv.push_str(&format!(
                "{},{},{:.4},{:.2}\n",
                entry.timestamp.to_rfc3339(),
                entry.raw_value,
                entry.scaled_value,
                entry.quality,
            ));
        }
        csv
    }

    /// Export logged data to JSON format.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let entries: Vec<&LogEntry> = self.entries.iter().collect();
        serde_json::to_string_pretty(&entries)
    }

    /// Export alerts to JSON.
    pub fn export_alerts_json(&self) -> Result<String, serde_json::Error> {
        let alerts: Vec<&Alert> = self.alerts.iter().collect();
        serde_json::to_string_pretty(&alerts)
    }

    /// Clear all log entries.
    pub fn clear_entries(&mut self) {
        self.entries.clear();
    }

    /// Clear all alerts.
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
        self.alert_active.clear();
    }

    /// Get the latest entry for a channel.
    pub fn latest_entry(&self, channel_name: &str) -> Option<&LogEntry> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.channel == channel_name)
    }
}

impl Default for ModbusDataLogger {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logger() -> ModbusDataLogger {
        let mut logger = ModbusDataLogger::new();
        logger.add_channel(ChannelConfig::new("temperature", 1, 40001).with_unit("degC"));
        logger.add_channel(
            ChannelConfig::new("pressure", 1, 40002)
                .with_unit("bar")
                .with_thresholds(Some(0.5), Some(10.0)),
        );
        logger
    }

    // ═══ Configuration tests ═════════════════════════════

    #[test]
    fn test_default_config() {
        let config = DataLoggerConfig::default();
        assert_eq!(config.max_entries, 10_000);
        assert!(config.enable_alerting);
    }

    #[test]
    fn test_channel_config_scaling() {
        let ch = ChannelConfig {
            scale: 0.1,
            offset: -10.0,
            ..ChannelConfig::new("test", 1, 40001)
        };
        // 100 * 0.1 + (-10.0) = 0.0
        assert!((ch.apply_scaling(100) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_channel_config_default_scaling() {
        let ch = ChannelConfig::new("test", 1, 40001);
        // scale=1, offset=0: raw=42 => scaled=42
        assert!((ch.apply_scaling(42) - 42.0).abs() < 1e-10);
    }

    // ═══ Channel management tests ════════════════════════

    #[test]
    fn test_add_channel() {
        let mut logger = ModbusDataLogger::new();
        logger.add_channel(ChannelConfig::new("ch1", 1, 40001));
        assert_eq!(logger.channel_count(), 1);
    }

    #[test]
    fn test_remove_channel() {
        let mut logger = make_logger();
        assert!(logger.remove_channel("temperature"));
        assert_eq!(logger.channel_count(), 1);
        assert!(!logger.remove_channel("nonexistent"));
    }

    // ═══ Logging tests ═══════════════════════════════════

    #[test]
    fn test_log_reading() {
        let mut logger = make_logger();
        let alert = logger.log_reading("temperature", 25);
        assert!(alert.is_none()); // no thresholds set
        assert_eq!(logger.entry_count(), 1);
    }

    #[test]
    fn test_log_reading_nonexistent_channel() {
        let mut logger = make_logger();
        let alert = logger.log_reading("nonexistent", 100);
        assert!(alert.is_none());
        assert_eq!(logger.entry_count(), 0);
    }

    #[test]
    fn test_log_reading_with_quality() {
        let mut logger = make_logger();
        logger.log_reading_with_quality("temperature", 25, 0.5);
        let entry = logger.latest_entry("temperature");
        assert!(entry.is_some());
        assert!((entry.expect("entry exists").quality - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let config = DataLoggerConfig {
            max_entries: 5,
            ..Default::default()
        };
        let mut logger = ModbusDataLogger::with_config(config);
        logger.add_channel(ChannelConfig::new("ch1", 1, 40001));
        for i in 0..10 {
            logger.log_reading("ch1", i);
        }
        assert_eq!(logger.entry_count(), 5);
        // Oldest should be evicted
        let entries = logger.entries_for_channel("ch1");
        assert_eq!(entries[0].raw_value, 5);
    }

    // ═══ Threshold alert tests ═══════════════════════════

    #[test]
    fn test_high_threshold_alert() {
        let mut logger = make_logger();
        let alert = logger.log_reading("pressure", 15); // 15.0 > 10.0
        assert!(alert.is_some());
        let alert = alert.expect("alert should exist");
        assert_eq!(alert.severity, AlertSeverity::High);
        assert!((alert.value - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_low_threshold_alert() {
        let mut logger = ModbusDataLogger::new();
        logger.add_channel(
            ChannelConfig::new("temp", 1, 40001).with_thresholds(Some(10.0), Some(50.0)),
        );
        let alert = logger.log_reading("temp", 5); // 5.0 < 10.0
        assert!(alert.is_some());
        let alert = alert.expect("alert should exist");
        assert_eq!(alert.severity, AlertSeverity::Low);
    }

    #[test]
    fn test_no_duplicate_alert() {
        let mut logger = make_logger();
        let alert1 = logger.log_reading("pressure", 15);
        assert!(alert1.is_some());
        let alert2 = logger.log_reading("pressure", 16);
        assert!(alert2.is_none()); // already in alert state
    }

    #[test]
    fn test_alert_cleared() {
        let mut logger = make_logger();
        logger.log_reading("pressure", 15); // high alert
        let alert = logger.log_reading("pressure", 5); // back to normal
        assert!(alert.is_some());
        assert_eq!(
            alert.expect("alert should exist").severity,
            AlertSeverity::Cleared
        );
    }

    #[test]
    fn test_alerting_disabled() {
        let config = DataLoggerConfig {
            enable_alerting: false,
            ..Default::default()
        };
        let mut logger = ModbusDataLogger::with_config(config);
        logger
            .add_channel(ChannelConfig::new("p", 1, 40001).with_thresholds(Some(0.5), Some(10.0)));
        let alert = logger.log_reading("p", 100);
        assert!(alert.is_none());
    }

    // ═══ Query/filter tests ══════════════════════════════

    #[test]
    fn test_entries_for_channel() {
        let mut logger = make_logger();
        logger.log_reading("temperature", 25);
        logger.log_reading("pressure", 5);
        logger.log_reading("temperature", 26);
        let temp_entries = logger.entries_for_channel("temperature");
        assert_eq!(temp_entries.len(), 2);
    }

    #[test]
    fn test_entries_with_tag() {
        let mut logger = ModbusDataLogger::new();
        logger.add_channel(
            ChannelConfig::new("ch1", 1, 40001)
                .with_tags(vec!["zone-a".to_string(), "critical".to_string()]),
        );
        logger
            .add_channel(ChannelConfig::new("ch2", 1, 40002).with_tags(vec!["zone-b".to_string()]));
        logger.log_reading("ch1", 10);
        logger.log_reading("ch2", 20);
        let zone_a = logger.entries_with_tag("zone-a");
        assert_eq!(zone_a.len(), 1);
        assert_eq!(zone_a[0].channel, "ch1");
    }

    #[test]
    fn test_latest_entry() {
        let mut logger = make_logger();
        logger.log_reading("temperature", 25);
        logger.log_reading("temperature", 30);
        let latest = logger.latest_entry("temperature");
        assert!(latest.is_some());
        assert_eq!(latest.expect("entry exists").raw_value, 30);
    }

    #[test]
    fn test_latest_entry_nonexistent() {
        let logger = make_logger();
        assert!(logger.latest_entry("nonexistent").is_none());
    }

    // ═══ Statistics tests ════════════════════════════════

    #[test]
    fn test_channel_stats() {
        let mut logger = make_logger();
        for v in [10, 20, 30, 40, 50] {
            logger.log_reading("temperature", v);
        }
        let stats = logger.channel_stats("temperature");
        assert!(stats.is_some());
        let stats = stats.expect("stats should exist");
        assert_eq!(stats.count, 5);
        assert!((stats.min - 10.0).abs() < 1e-10);
        assert!((stats.max - 50.0).abs() < 1e-10);
        assert!((stats.mean - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_channel_stats_std_dev() {
        let mut logger = make_logger();
        // All same values => std_dev = 0
        for _ in 0..5 {
            logger.log_reading("temperature", 100);
        }
        let stats = logger.channel_stats("temperature").expect("stats exist");
        assert!(stats.std_dev.abs() < 1e-10);
    }

    #[test]
    fn test_channel_stats_empty() {
        let logger = make_logger();
        assert!(logger.channel_stats("temperature").is_none());
    }

    // ═══ Export tests ════════════════════════════════════

    #[test]
    fn test_export_csv() {
        let mut logger = make_logger();
        logger.log_reading("temperature", 25);
        let csv = logger.export_csv();
        assert!(csv.starts_with("timestamp,channel,raw_value,scaled_value,quality,tags\n"));
        assert!(csv.contains("temperature"));
        assert!(csv.contains("25"));
    }

    #[test]
    fn test_export_channel_csv() {
        let mut logger = make_logger();
        logger.log_reading("temperature", 25);
        logger.log_reading("pressure", 5);
        let csv = logger.export_channel_csv("temperature");
        assert!(csv.starts_with("timestamp,raw_value,scaled_value,quality\n"));
        assert!(!csv.contains("pressure"));
    }

    #[test]
    fn test_export_json() {
        let mut logger = make_logger();
        logger.log_reading("temperature", 25);
        let json = logger.export_json();
        assert!(json.is_ok());
        let json = json.expect("JSON export should succeed");
        assert!(json.contains("temperature"));
    }

    #[test]
    fn test_export_alerts_json() {
        let mut logger = make_logger();
        logger.log_reading("pressure", 15); // trigger alert
        let json = logger.export_alerts_json();
        assert!(json.is_ok());
        let json = json.expect("JSON export should succeed");
        assert!(json.contains("HIGH"));
    }

    #[test]
    fn test_export_csv_empty() {
        let logger = make_logger();
        let csv = logger.export_csv();
        assert_eq!(
            csv,
            "timestamp,channel,raw_value,scaled_value,quality,tags\n"
        );
    }

    // ═══ Clear tests ═════════════════════════════════════

    #[test]
    fn test_clear_entries() {
        let mut logger = make_logger();
        logger.log_reading("temperature", 25);
        logger.clear_entries();
        assert_eq!(logger.entry_count(), 0);
    }

    #[test]
    fn test_clear_alerts() {
        let mut logger = make_logger();
        logger.log_reading("pressure", 15);
        logger.clear_alerts();
        assert_eq!(logger.alert_count(), 0);
    }

    // ═══ Alert severity display test ═════════════════════

    #[test]
    fn test_alert_severity_display() {
        assert_eq!(format!("{}", AlertSeverity::High), "HIGH");
        assert_eq!(format!("{}", AlertSeverity::Low), "LOW");
        assert_eq!(format!("{}", AlertSeverity::Cleared), "CLEARED");
    }

    // ═══ Default impl test ═══════════════════════════════

    #[test]
    fn test_default_impl() {
        let logger = ModbusDataLogger::default();
        assert_eq!(logger.channel_count(), 0);
        assert_eq!(logger.entry_count(), 0);
    }

    // ═══ Alerts for channel test ═════════════════════════

    #[test]
    fn test_alerts_for_channel() {
        let mut logger = make_logger();
        logger.log_reading("pressure", 15);
        logger.log_reading("pressure", 5); // cleared
        let alerts = logger.alerts_for_channel("pressure");
        assert_eq!(alerts.len(), 2);
    }

    // ═══ Alert ring buffer bounded test ══════════════════

    #[test]
    fn test_alert_ring_buffer_bounded() {
        let config = DataLoggerConfig {
            max_alerts: 3,
            ..Default::default()
        };
        let mut logger = ModbusDataLogger::with_config(config);
        logger
            .add_channel(ChannelConfig::new("p", 1, 40001).with_thresholds(Some(0.5), Some(10.0)));
        // Generate alerts by toggling high/normal
        for i in 0..10 {
            if i % 2 == 0 {
                logger.log_reading("p", 20); // high
            } else {
                logger.log_reading("p", 5); // cleared
            }
        }
        assert!(logger.alert_count() <= 3);
    }
}
