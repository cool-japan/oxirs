//! Modbus event logging.
//!
//! Provides a ring-buffer-backed, filterable event log for Modbus operations.
//! Events cover register changes, connection state transitions, and protocol
//! errors. The log supports:
//!
//! - Ring buffer storage with configurable capacity (oldest entries evicted).
//! - Filtering by event type, address range, and time range.
//! - CSV and JSON export.
//! - Per-type event statistics and error rate calculation.
//! - Callback-based subscriptions for event types.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ─────────────────────────────────────────────────────────────────────────────
// Event types
// ─────────────────────────────────────────────────────────────────────────────

/// Categories of Modbus events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// A register value changed.
    RegisterChange,
    /// A Modbus device connected.
    Connected,
    /// A Modbus device disconnected.
    Disconnected,
    /// A connection timed out.
    Timeout,
    /// A Modbus exception code was received.
    Error,
}

impl EventKind {
    /// Short label used in CSV/JSON output and statistics.
    pub fn label(&self) -> &'static str {
        match self {
            EventKind::RegisterChange => "RegisterChange",
            EventKind::Connected => "Connected",
            EventKind::Disconnected => "Disconnected",
            EventKind::Timeout => "Timeout",
            EventKind::Error => "Error",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Event payload
// ─────────────────────────────────────────────────────────────────────────────

/// Payload for a register-change event.
#[derive(Debug, Clone)]
pub struct RegisterChangePayload {
    /// Modbus register address.
    pub address: u16,
    /// Previous register value.
    pub old_value: u16,
    /// New register value.
    pub new_value: u16,
}

/// Payload for an error event.
#[derive(Debug, Clone)]
pub struct ErrorPayload {
    /// Modbus exception code (1–11).
    pub exception_code: u8,
    /// Human-readable context about the failing request.
    pub request_context: String,
}

/// Event-specific data.
#[derive(Debug, Clone)]
pub enum EventPayload {
    RegisterChange(RegisterChangePayload),
    Connection {
        /// Device unit identifier.
        unit_id: u8,
        /// Optional endpoint address or description.
        endpoint: String,
    },
    Error(ErrorPayload),
    /// Generic note for Timeout events (no structured data).
    Timeout {
        unit_id: u8,
        endpoint: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Event
// ─────────────────────────────────────────────────────────────────────────────

/// A single Modbus event record.
#[derive(Debug, Clone)]
pub struct ModbusEvent {
    /// Monotonically increasing event ID.
    pub id: u64,
    /// Event category.
    pub kind: EventKind,
    /// Unix-epoch timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Event-specific payload.
    pub payload: EventPayload,
}

impl ModbusEvent {
    /// Return the register address if this is a `RegisterChange` event.
    pub fn register_address(&self) -> Option<u16> {
        if let EventPayload::RegisterChange(ref p) = self.payload {
            Some(p.address)
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter
// ─────────────────────────────────────────────────────────────────────────────

/// Criteria for selecting events from the log.
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// Only include events of these kinds (empty = all kinds).
    pub kinds: Vec<EventKind>,
    /// Only include `RegisterChange` events in this address range (inclusive).
    pub address_range: Option<(u16, u16)>,
    /// Only include events with `timestamp_ms >= start_ms`.
    pub start_ms: Option<u64>,
    /// Only include events with `timestamp_ms <= end_ms`.
    pub end_ms: Option<u64>,
}

impl EventFilter {
    /// Return `true` if `event` matches all criteria in this filter.
    pub fn matches(&self, event: &ModbusEvent) -> bool {
        // Kind filter.
        if !self.kinds.is_empty() && !self.kinds.contains(&event.kind) {
            return false;
        }

        // Address range filter.
        if let Some((lo, hi)) = self.address_range {
            if let Some(addr) = event.register_address() {
                if addr < lo || addr > hi {
                    return false;
                }
            } else if event.kind == EventKind::RegisterChange {
                return false;
            }
        }

        // Time range filter.
        if let Some(start) = self.start_ms {
            if event.timestamp_ms < start {
                return false;
            }
        }
        if let Some(end) = self.end_ms {
            if event.timestamp_ms > end {
                return false;
            }
        }

        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-type event counts and derived error rate.
#[derive(Debug, Clone, Default)]
pub struct EventStats {
    /// Count of events per `EventKind::label()`.
    pub counts: HashMap<String, usize>,
    /// Total events seen (including evicted).
    pub total_seen: u64,
}

impl EventStats {
    /// Error rate = Error events / total events (0.0 if no events).
    pub fn error_rate(&self) -> f64 {
        if self.total_seen == 0 {
            return 0.0;
        }
        let errors = self.counts.get("Error").copied().unwrap_or(0);
        errors as f64 / self.total_seen as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Subscription
// ─────────────────────────────────────────────────────────────────────────────

/// A registered event subscription.
pub struct EventSubscription {
    /// The kinds of events to receive (empty = all).
    pub kinds: Vec<EventKind>,
    /// Callback invoked for every matching event.
    pub callback: Box<dyn Fn(&ModbusEvent) + Send + Sync>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Event log
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`EventLog`].
#[derive(Debug, Clone)]
pub struct EventLogConfig {
    /// Maximum number of events to retain in the ring buffer.
    pub capacity: usize,
}

impl Default for EventLogConfig {
    fn default() -> Self {
        Self { capacity: 10_000 }
    }
}

/// Ring-buffer backed Modbus event log.
pub struct EventLog {
    config: EventLogConfig,
    /// Circular buffer of events (front = oldest, back = newest).
    buffer: std::collections::VecDeque<ModbusEvent>,
    next_id: u64,
    stats: EventStats,
    subscriptions: Vec<EventSubscription>,
}

impl EventLog {
    /// Create a new event log with the given configuration.
    pub fn new(config: EventLogConfig) -> Self {
        Self {
            config,
            buffer: std::collections::VecDeque::new(),
            next_id: 1,
            stats: EventStats::default(),
            subscriptions: Vec::new(),
        }
    }

    // ── writing ───────────────────────────────────────────────────────────────

    /// Record a register-change event.
    pub fn log_register_change(
        &mut self,
        address: u16,
        old_value: u16,
        new_value: u16,
        timestamp_ms: u64,
    ) {
        let event = self.make_event(
            EventKind::RegisterChange,
            timestamp_ms,
            EventPayload::RegisterChange(RegisterChangePayload {
                address,
                old_value,
                new_value,
            }),
        );
        self.push(event);
    }

    /// Record a connection event.
    pub fn log_connected(&mut self, unit_id: u8, endpoint: impl Into<String>, timestamp_ms: u64) {
        let event = self.make_event(
            EventKind::Connected,
            timestamp_ms,
            EventPayload::Connection {
                unit_id,
                endpoint: endpoint.into(),
            },
        );
        self.push(event);
    }

    /// Record a disconnection event.
    pub fn log_disconnected(
        &mut self,
        unit_id: u8,
        endpoint: impl Into<String>,
        timestamp_ms: u64,
    ) {
        let event = self.make_event(
            EventKind::Disconnected,
            timestamp_ms,
            EventPayload::Connection {
                unit_id,
                endpoint: endpoint.into(),
            },
        );
        self.push(event);
    }

    /// Record a timeout event.
    pub fn log_timeout(&mut self, unit_id: u8, endpoint: impl Into<String>, timestamp_ms: u64) {
        let event = self.make_event(
            EventKind::Timeout,
            timestamp_ms,
            EventPayload::Timeout {
                unit_id,
                endpoint: endpoint.into(),
            },
        );
        self.push(event);
    }

    /// Record an error event.
    pub fn log_error(
        &mut self,
        exception_code: u8,
        request_context: impl Into<String>,
        timestamp_ms: u64,
    ) {
        let event = self.make_event(
            EventKind::Error,
            timestamp_ms,
            EventPayload::Error(ErrorPayload {
                exception_code,
                request_context: request_context.into(),
            }),
        );
        self.push(event);
    }

    // ── reading ───────────────────────────────────────────────────────────────

    /// Return all events currently in the buffer that match `filter`.
    pub fn query(&self, filter: &EventFilter) -> Vec<&ModbusEvent> {
        self.buffer.iter().filter(|e| filter.matches(e)).collect()
    }

    /// Return all events in the buffer (unfiltered).
    pub fn all_events(&self) -> Vec<&ModbusEvent> {
        self.buffer.iter().collect()
    }

    /// Number of events currently in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    // ── statistics ────────────────────────────────────────────────────────────

    /// Return a snapshot of the event statistics.
    pub fn stats(&self) -> &EventStats {
        &self.stats
    }

    // ── subscriptions ─────────────────────────────────────────────────────────

    /// Register a callback for specific event kinds (empty kinds = all events).
    pub fn subscribe<F>(&mut self, kinds: Vec<EventKind>, callback: F)
    where
        F: Fn(&ModbusEvent) + Send + Sync + 'static,
    {
        self.subscriptions.push(EventSubscription {
            kinds,
            callback: Box::new(callback),
        });
    }

    // ── export ────────────────────────────────────────────────────────────────

    /// Export matching events as CSV.
    ///
    /// Columns: `id,kind,timestamp_ms,detail`
    pub fn export_csv(&self, filter: &EventFilter) -> String {
        let mut out = String::from("id,kind,timestamp_ms,detail\n");
        for event in self.query(filter) {
            let detail = self.event_detail(event);
            let _ = writeln!(
                out,
                "{},{},{},{}",
                event.id,
                event.kind.label(),
                event.timestamp_ms,
                detail
            );
        }
        out
    }

    /// Export matching events as a JSON array string.
    pub fn export_json(&self, filter: &EventFilter) -> String {
        let events: Vec<String> = self
            .query(filter)
            .iter()
            .map(|e| {
                let detail = self.event_detail(e);
                format!(
                    r#"{{"id":{},"kind":"{}","timestamp_ms":{},"detail":"{}"}}"#,
                    e.id,
                    e.kind.label(),
                    e.timestamp_ms,
                    detail.replace('"', "\\\"")
                )
            })
            .collect();
        format!("[{}]", events.join(","))
    }

    // ── internals ─────────────────────────────────────────────────────────────

    fn make_event(
        &mut self,
        kind: EventKind,
        timestamp_ms: u64,
        payload: EventPayload,
    ) -> ModbusEvent {
        let id = self.next_id;
        self.next_id += 1;
        ModbusEvent {
            id,
            kind,
            timestamp_ms,
            payload,
        }
    }

    fn push(&mut self, event: ModbusEvent) {
        // Update statistics.
        let label = event.kind.label().to_string();
        *self.stats.counts.entry(label).or_insert(0) += 1;
        self.stats.total_seen += 1;

        // Notify subscribers.
        for sub in &self.subscriptions {
            if sub.kinds.is_empty() || sub.kinds.contains(&event.kind) {
                (sub.callback)(&event);
            }
        }

        // Ring buffer: evict oldest when at capacity.
        if self.buffer.len() >= self.config.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(event);
    }

    fn event_detail(&self, event: &ModbusEvent) -> String {
        match &event.payload {
            EventPayload::RegisterChange(p) => {
                format!("addr={} old={} new={}", p.address, p.old_value, p.new_value)
            }
            EventPayload::Connection { unit_id, endpoint } => {
                format!("unit={} endpoint={}", unit_id, endpoint)
            }
            EventPayload::Timeout { unit_id, endpoint } => {
                format!("unit={} endpoint={}", unit_id, endpoint)
            }
            EventPayload::Error(p) => {
                format!("code={} ctx={}", p.exception_code, p.request_context)
            }
        }
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new(EventLogConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn default_log() -> EventLog {
        EventLog::new(EventLogConfig { capacity: 100 })
    }

    // ── basic logging ─────────────────────────────────────────────────────────

    #[test]
    fn test_log_register_change_stores_event() {
        let mut log = default_log();
        log.log_register_change(100, 0, 42, 1000);
        assert_eq!(log.len(), 1);
        let ev = &log.all_events()[0];
        assert_eq!(ev.kind, EventKind::RegisterChange);
        assert_eq!(ev.register_address(), Some(100));
    }

    #[test]
    fn test_log_connected_stores_event() {
        let mut log = default_log();
        log.log_connected(1, "192.168.1.10:502", 500);
        assert_eq!(log.len(), 1);
        assert_eq!(log.all_events()[0].kind, EventKind::Connected);
    }

    #[test]
    fn test_log_disconnected_stores_event() {
        let mut log = default_log();
        log.log_disconnected(2, "192.168.1.11:502", 600);
        assert_eq!(log.all_events()[0].kind, EventKind::Disconnected);
    }

    #[test]
    fn test_log_timeout_stores_event() {
        let mut log = default_log();
        log.log_timeout(3, "192.168.1.12:502", 700);
        assert_eq!(log.all_events()[0].kind, EventKind::Timeout);
    }

    #[test]
    fn test_log_error_stores_event() {
        let mut log = default_log();
        log.log_error(2, "FC03 addr=40001", 800);
        assert_eq!(log.all_events()[0].kind, EventKind::Error);
    }

    // ── ring buffer ───────────────────────────────────────────────────────────

    #[test]
    fn test_ring_buffer_evicts_oldest() {
        let mut log = EventLog::new(EventLogConfig { capacity: 3 });
        log.log_register_change(1, 0, 1, 100);
        log.log_register_change(2, 0, 2, 200);
        log.log_register_change(3, 0, 3, 300);
        log.log_register_change(4, 0, 4, 400); // evicts id=1
        assert_eq!(log.len(), 3);
        // The oldest remaining should have address=2.
        let first = log.all_events()[0];
        assert_eq!(first.register_address(), Some(2));
    }

    #[test]
    fn test_ring_buffer_capacity_respected() {
        let cap = 5usize;
        let mut log = EventLog::new(EventLogConfig { capacity: cap });
        for i in 0..10u16 {
            log.log_register_change(i, 0, 1, i as u64 * 100);
        }
        assert_eq!(log.len(), cap);
    }

    // ── filtering ─────────────────────────────────────────────────────────────

    #[test]
    fn test_filter_by_kind_register_change() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_connected(1, "ep", 200);
        let filter = EventFilter {
            kinds: vec![EventKind::RegisterChange],
            ..Default::default()
        };
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind, EventKind::RegisterChange);
    }

    #[test]
    fn test_filter_by_address_range() {
        let mut log = default_log();
        log.log_register_change(10, 0, 1, 100);
        log.log_register_change(50, 0, 2, 200);
        log.log_register_change(100, 0, 3, 300);
        let filter = EventFilter {
            address_range: Some((10, 50)),
            ..Default::default()
        };
        let results = log.query(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_by_time_range() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_register_change(2, 0, 2, 500);
        log.log_register_change(3, 0, 3, 1000);
        let filter = EventFilter {
            start_ms: Some(400),
            end_ms: Some(600),
            ..Default::default()
        };
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp_ms, 500);
    }

    #[test]
    fn test_filter_no_criteria_returns_all() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_connected(1, "ep", 200);
        let filter = EventFilter::default();
        assert_eq!(log.query(&filter).len(), 2);
    }

    #[test]
    fn test_filter_time_start_only() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_register_change(2, 0, 2, 900);
        let filter = EventFilter {
            start_ms: Some(500),
            ..Default::default()
        };
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp_ms, 900);
    }

    #[test]
    fn test_filter_time_end_only() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_register_change(2, 0, 2, 900);
        let filter = EventFilter {
            end_ms: Some(500),
            ..Default::default()
        };
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp_ms, 100);
    }

    // ── statistics ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_count_per_kind() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_register_change(2, 0, 2, 200);
        log.log_error(1, "ctx", 300);
        let stats = log.stats();
        assert_eq!(stats.counts.get("RegisterChange").copied().unwrap_or(0), 2);
        assert_eq!(stats.counts.get("Error").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_stats_total_seen_includes_evicted() {
        let mut log = EventLog::new(EventLogConfig { capacity: 2 });
        log.log_register_change(1, 0, 1, 100);
        log.log_register_change(2, 0, 2, 200);
        log.log_register_change(3, 0, 3, 300); // evicts first
        assert_eq!(log.stats().total_seen, 3);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_error_rate_zero_when_no_events() {
        let log = default_log();
        assert!((log.stats().error_rate() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_error_rate_calculation() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_register_change(2, 0, 2, 200);
        log.log_error(1, "x", 300);
        log.log_error(2, "y", 400);
        // 2 errors / 4 total = 0.5
        let rate = log.stats().error_rate();
        assert!((rate - 0.5).abs() < 1e-9);
    }

    // ── subscriptions ─────────────────────────────────────────────────────────

    #[test]
    fn test_subscription_receives_matching_events() {
        let mut log = default_log();
        let received = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&received);
        log.subscribe(vec![EventKind::Error], move |event| {
            r.lock().expect("lock").push(event.id);
        });
        log.log_register_change(1, 0, 1, 100);
        log.log_error(3, "ctx", 200);
        let ids = received.lock().expect("lock").clone();
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_subscription_all_kinds_receives_all() {
        let mut log = default_log();
        let count = Arc::new(Mutex::new(0usize));
        let c = Arc::clone(&count);
        log.subscribe(vec![], move |_event| {
            *c.lock().expect("lock") += 1;
        });
        log.log_register_change(1, 0, 1, 100);
        log.log_connected(1, "ep", 200);
        log.log_error(1, "ctx", 300);
        assert_eq!(*count.lock().expect("lock"), 3);
    }

    // ── export ────────────────────────────────────────────────────────────────

    #[test]
    fn test_export_csv_header_present() {
        let log = default_log();
        let csv = log.export_csv(&EventFilter::default());
        assert!(csv.starts_with("id,kind,timestamp_ms,detail"));
    }

    #[test]
    fn test_export_csv_includes_events() {
        let mut log = default_log();
        log.log_register_change(100, 10, 20, 1000);
        let csv = log.export_csv(&EventFilter::default());
        assert!(csv.contains("RegisterChange"));
        assert!(csv.contains("addr=100"));
    }

    #[test]
    fn test_export_json_is_valid_array() {
        let mut log = default_log();
        log.log_connected(1, "ep", 100);
        let json = log.export_json(&EventFilter::default());
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("Connected"));
    }

    #[test]
    fn test_export_json_empty_is_empty_array() {
        let log = default_log();
        let json = log.export_json(&EventFilter::default());
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_export_csv_filtered() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_error(1, "ctx", 200);
        let filter = EventFilter {
            kinds: vec![EventKind::Error],
            ..Default::default()
        };
        let csv = log.export_csv(&filter);
        assert!(csv.contains("Error"));
        assert!(!csv.contains("RegisterChange"));
    }

    // ── IDs ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_event_ids_are_monotonically_increasing() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_register_change(2, 0, 2, 200);
        let events = log.all_events();
        assert_eq!(events[0].id, 1);
        assert_eq!(events[1].id, 2);
    }

    // ── miscellaneous ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_empty_on_new_log() {
        let log = default_log();
        assert!(log.is_empty());
    }

    #[test]
    fn test_register_change_payload_fields() {
        let mut log = default_log();
        log.log_register_change(42, 100, 200, 999);
        let ev = &log.all_events()[0];
        if let EventPayload::RegisterChange(ref p) = ev.payload {
            assert_eq!(p.address, 42);
            assert_eq!(p.old_value, 100);
            assert_eq!(p.new_value, 200);
        } else {
            panic!("wrong payload");
        }
    }

    #[test]
    fn test_error_payload_fields() {
        let mut log = default_log();
        log.log_error(5, "read coils FC01", 1000);
        let ev = &log.all_events()[0];
        if let EventPayload::Error(ref p) = ev.payload {
            assert_eq!(p.exception_code, 5);
            assert_eq!(p.request_context, "read coils FC01");
        } else {
            panic!("wrong payload");
        }
    }

    #[test]
    fn test_default_event_log() {
        let log = EventLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_kind_labels() {
        assert_eq!(EventKind::RegisterChange.label(), "RegisterChange");
        assert_eq!(EventKind::Connected.label(), "Connected");
        assert_eq!(EventKind::Disconnected.label(), "Disconnected");
        assert_eq!(EventKind::Timeout.label(), "Timeout");
        assert_eq!(EventKind::Error.label(), "Error");
    }

    #[test]
    fn test_register_address_none_for_non_register_events() {
        let mut log = default_log();
        log.log_connected(1, "ep", 100);
        let ev = &log.all_events()[0];
        assert_eq!(ev.register_address(), None);
    }

    #[test]
    fn test_multiple_subscriptions_all_fire() {
        let mut log = default_log();
        let c1 = Arc::new(Mutex::new(0usize));
        let c2 = Arc::clone(&c1);
        let c3 = Arc::new(Mutex::new(0usize));
        let c4 = Arc::clone(&c3);
        log.subscribe(vec![], move |_| {
            *c2.lock().expect("lock") += 1;
        });
        log.subscribe(vec![EventKind::Error], move |_| {
            *c4.lock().expect("lock") += 1;
        });
        log.log_error(1, "ctx", 100);
        assert_eq!(*c1.lock().expect("lock"), 1); // all-kinds subscription
        assert_eq!(*c3.lock().expect("lock"), 1); // error-only subscription
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_error_rate_all_errors() {
        let mut log = default_log();
        log.log_error(1, "a", 100);
        log.log_error(2, "b", 200);
        let rate = log.stats().error_rate();
        assert!((rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_error_rate_no_errors() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_connected(1, "ep", 200);
        let rate = log.stats().error_rate();
        assert!((rate - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_filter_by_multiple_kinds() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_connected(1, "ep", 200);
        log.log_error(1, "ctx", 300);
        let filter = EventFilter {
            kinds: vec![EventKind::Connected, EventKind::Error],
            ..Default::default()
        };
        let results = log.query(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_json_export_multiple_events() {
        let mut log = default_log();
        log.log_register_change(10, 0, 5, 100);
        log.log_error(3, "test ctx", 200);
        let json = log.export_json(&EventFilter::default());
        // Should contain two JSON objects.
        assert!(json.contains("RegisterChange"));
        assert!(json.contains("Error"));
    }

    #[test]
    fn test_csv_export_multiple_events() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_timeout(2, "192.168.1.1:502", 200);
        let csv = log.export_csv(&EventFilter::default());
        assert!(csv.contains("Timeout"));
        assert!(csv.contains("RegisterChange"));
    }

    #[test]
    fn test_subscription_not_fired_for_different_kind() {
        let mut log = default_log();
        let fired = Arc::new(Mutex::new(false));
        let f = Arc::clone(&fired);
        log.subscribe(vec![EventKind::Error], move |_| {
            *f.lock().expect("lock") = true;
        });
        // Trigger a non-Error event.
        log.log_register_change(1, 0, 1, 100);
        assert!(!*fired.lock().expect("lock"));
    }

    #[test]
    fn test_log_events_have_unique_ids() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_error(1, "ctx", 200);
        log.log_connected(1, "ep", 300);
        let ids: Vec<u64> = log.all_events().iter().map(|e| e.id).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().cloned().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_query_empty_log_returns_empty() {
        let log = default_log();
        let results = log.query(&EventFilter::default());
        assert!(results.is_empty());
    }

    #[test]
    fn test_filter_by_address_range_excludes_out_of_range() {
        let mut log = default_log();
        log.log_register_change(200, 0, 1, 100); // out of range
        let filter = EventFilter {
            address_range: Some((0, 100)),
            ..Default::default()
        };
        let results = log.query(&filter);
        assert!(results.is_empty());
    }

    #[test]
    fn test_timeout_payload_fields() {
        let mut log = default_log();
        log.log_timeout(5, "10.0.0.1:502", 999);
        let ev = &log.all_events()[0];
        if let EventPayload::Timeout { unit_id, endpoint } = &ev.payload {
            assert_eq!(*unit_id, 5);
            assert_eq!(endpoint, "10.0.0.1:502");
        } else {
            panic!("wrong payload variant");
        }
    }

    #[test]
    fn test_connection_payload_fields() {
        let mut log = default_log();
        log.log_connected(7, "device-A", 100);
        let ev = &log.all_events()[0];
        if let EventPayload::Connection { unit_id, endpoint } = &ev.payload {
            assert_eq!(*unit_id, 7);
            assert_eq!(endpoint, "device-A");
        } else {
            panic!("wrong payload variant");
        }
    }

    #[test]
    fn test_stats_counts_all_kinds() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_connected(1, "ep", 200);
        log.log_disconnected(1, "ep", 300);
        log.log_timeout(1, "ep", 400);
        log.log_error(1, "ctx", 500);
        let stats = log.stats();
        for kind in [
            "RegisterChange",
            "Connected",
            "Disconnected",
            "Timeout",
            "Error",
        ] {
            assert_eq!(
                stats.counts.get(kind).copied().unwrap_or(0),
                1,
                "missing kind {kind}"
            );
        }
    }

    #[test]
    fn test_event_log_len_after_multiple_kinds() {
        let mut log = default_log();
        log.log_register_change(1, 0, 1, 100);
        log.log_connected(1, "ep", 200);
        log.log_error(1, "ctx", 300);
        assert_eq!(log.len(), 3);
    }
}
