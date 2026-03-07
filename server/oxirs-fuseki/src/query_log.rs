//! Structured query audit log for SPARQL endpoints.
//!
//! Provides a ring-buffer-backed audit log that records SPARQL query activity,
//! timings, and errors for monitoring and debugging purposes.

use std::collections::HashMap;
use std::fmt;

/// The type of SPARQL operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryType {
    /// SPARQL SELECT query
    Select,
    /// SPARQL CONSTRUCT query
    Construct,
    /// SPARQL ASK query
    Ask,
    /// SPARQL DESCRIBE query
    Describe,
    /// SPARQL UPDATE operation
    Update,
    /// Unknown or unrecognised operation
    Unknown,
}

impl QueryType {
    /// Detect the query type from the first significant keyword in `query`.
    pub fn from_sparql(query: &str) -> Self {
        let upper = query.trim_start().to_ascii_uppercase();
        if upper.starts_with("SELECT") {
            QueryType::Select
        } else if upper.starts_with("CONSTRUCT") {
            QueryType::Construct
        } else if upper.starts_with("ASK") {
            QueryType::Ask
        } else if upper.starts_with("DESCRIBE") {
            QueryType::Describe
        } else if upper.starts_with("INSERT")
            || upper.starts_with("DELETE")
            || upper.starts_with("LOAD")
            || upper.starts_with("CLEAR")
            || upper.starts_with("DROP")
            || upper.starts_with("ADD")
            || upper.starts_with("MOVE")
            || upper.starts_with("COPY")
            || upper.starts_with("CREATE")
        {
            QueryType::Update
        } else {
            QueryType::Unknown
        }
    }

    /// Return a display name for this query type.
    pub fn as_str(&self) -> &'static str {
        match self {
            QueryType::Select => "SELECT",
            QueryType::Construct => "CONSTRUCT",
            QueryType::Ask => "ASK",
            QueryType::Describe => "DESCRIBE",
            QueryType::Update => "UPDATE",
            QueryType::Unknown => "UNKNOWN",
        }
    }
}

impl fmt::Display for QueryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single recorded query event.
#[derive(Debug, Clone)]
pub struct QueryLogEntry {
    /// Unique monotonic identifier for this entry.
    pub id: u64,
    /// Monotonic timestamp counter (not wall-clock time).
    pub timestamp_ms: u64,
    /// Detected type of SPARQL operation.
    pub query_type: QueryType,
    /// Name of the dataset this query was executed against.
    pub dataset: String,
    /// The raw query text.
    pub query_text: String,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Number of results returned, if applicable.
    pub result_count: Option<usize>,
    /// Error message, if the query failed.
    pub error: Option<String>,
    /// IP address of the requesting client, if available.
    pub client_ip: Option<String>,
}

/// Error type for `QueryLog` operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LogError {
    /// No entry was found with the given id.
    NotFound(u64),
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogError::NotFound(id) => write!(f, "Query log entry not found: id={id}"),
        }
    }
}

impl std::error::Error for LogError {}

/// Aggregated statistics over all recorded queries.
#[derive(Debug, Clone)]
pub struct LogStats {
    /// Total number of queries recorded (including errors).
    pub total_queries: usize,
    /// Average duration across all entries, or 0.0 if no entries.
    pub avg_duration_ms: f64,
    /// Number of entries that contain an error.
    pub error_count: usize,
    /// Per-type counts, keyed by the type's display name.
    pub by_type: HashMap<String, usize>,
}

/// A ring-buffer query audit log.
///
/// Holds at most `capacity` entries; when full, the oldest entry is removed
/// before a new one is inserted.
pub struct QueryLog {
    entries: Vec<QueryLogEntry>,
    capacity: usize,
    next_id: u64,
    /// Monotonic timestamp counter incremented with every logged event.
    clock: u64,
}

impl QueryLog {
    /// Create a new log with the given ring-buffer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity.min(4096)),
            capacity: capacity.max(1),
            next_id: 1,
            clock: 0,
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn push_entry(&mut self, entry: QueryLogEntry) -> u64 {
        let id = entry.id;
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(entry);
        id
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Record a successful (or attempted) query and return its assigned id.
    pub fn log_query(
        &mut self,
        dataset: &str,
        query_text: &str,
        duration_ms: u64,
        result_count: Option<usize>,
        client_ip: Option<String>,
    ) -> u64 {
        let id = self.next_id();
        let ts = self.tick();
        let entry = QueryLogEntry {
            id,
            timestamp_ms: ts,
            query_type: QueryType::from_sparql(query_text),
            dataset: dataset.to_string(),
            query_text: query_text.to_string(),
            duration_ms,
            result_count,
            error: None,
            client_ip,
        };
        self.push_entry(entry)
    }

    /// Record a failed query and return its assigned id.
    pub fn log_error(
        &mut self,
        dataset: &str,
        query_text: &str,
        duration_ms: u64,
        error_msg: &str,
        client_ip: Option<String>,
    ) -> u64 {
        let id = self.next_id();
        let ts = self.tick();
        let entry = QueryLogEntry {
            id,
            timestamp_ms: ts,
            query_type: QueryType::from_sparql(query_text),
            dataset: dataset.to_string(),
            query_text: query_text.to_string(),
            duration_ms,
            result_count: None,
            error: Some(error_msg.to_string()),
            client_ip,
        };
        self.push_entry(entry)
    }

    /// Retrieve an entry by id.
    pub fn get(&self, id: u64) -> Option<&QueryLogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Return the last `n` entries in reverse chronological order (newest first).
    pub fn recent(&self, n: usize) -> Vec<&QueryLogEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Return all entries of a specific query type.
    pub fn by_type(&self, qt: QueryType) -> Vec<&QueryLogEntry> {
        self.entries.iter().filter(|e| e.query_type == qt).collect()
    }

    /// Compute aggregate statistics over all current entries.
    pub fn stats(&self) -> LogStats {
        let total_queries = self.entries.len();
        let error_count = self.entries.iter().filter(|e| e.error.is_some()).count();

        let avg_duration_ms = if total_queries == 0 {
            0.0
        } else {
            self.entries
                .iter()
                .map(|e| e.duration_ms as f64)
                .sum::<f64>()
                / total_queries as f64
        };

        let mut by_type: HashMap<String, usize> = HashMap::new();
        for entry in &self.entries {
            *by_type.entry(entry.query_type.to_string()).or_insert(0) += 1;
        }

        LogStats {
            total_queries,
            avg_duration_ms,
            error_count,
            by_type,
        }
    }

    /// Remove all entries from the log.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Serialise the current log to a simple JSON array string.
    pub fn export_json(&self) -> String {
        let mut buf = String::from("[");
        for (i, e) in self.entries.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            buf.push('{');
            buf.push_str(&format!("\"id\":{},", e.id));
            buf.push_str(&format!("\"timestamp_ms\":{},", e.timestamp_ms));
            buf.push_str(&format!("\"query_type\":\"{}\",", e.query_type));
            buf.push_str(&format!("\"dataset\":\"{}\",", json_escape(&e.dataset)));
            buf.push_str(&format!(
                "\"query_text\":\"{}\",",
                json_escape(&e.query_text)
            ));
            buf.push_str(&format!("\"duration_ms\":{},", e.duration_ms));
            match e.result_count {
                Some(rc) => buf.push_str(&format!("\"result_count\":{},", rc)),
                None => buf.push_str("\"result_count\":null,"),
            }
            match &e.error {
                Some(err) => buf.push_str(&format!("\"error\":\"{}\",", json_escape(err))),
                None => buf.push_str("\"error\":null,"),
            }
            match &e.client_ip {
                Some(ip) => buf.push_str(&format!("\"client_ip\":\"{}\"", json_escape(ip))),
                None => buf.push_str("\"client_ip\":null"),
            }
            buf.push('}');
        }
        buf.push(']');
        buf
    }

    /// Return the total number of entries currently in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Escape a string for embedding in JSON.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── QueryType::from_sparql ─────────────────────────────────────────────────

    #[test]
    fn test_detect_select() {
        assert_eq!(
            QueryType::from_sparql("SELECT ?s WHERE { ?s ?p ?o }"),
            QueryType::Select
        );
    }

    #[test]
    fn test_detect_select_lowercase() {
        assert_eq!(
            QueryType::from_sparql("select ?x where {}"),
            QueryType::Select
        );
    }

    #[test]
    fn test_detect_construct() {
        assert_eq!(
            QueryType::from_sparql("CONSTRUCT { ?s ?p ?o } WHERE {}"),
            QueryType::Construct
        );
    }

    #[test]
    fn test_detect_ask() {
        assert_eq!(QueryType::from_sparql("ASK { ?s ?p ?o }"), QueryType::Ask);
    }

    #[test]
    fn test_detect_describe() {
        assert_eq!(
            QueryType::from_sparql("DESCRIBE <http://example.org/resource>"),
            QueryType::Describe
        );
    }

    #[test]
    fn test_detect_insert() {
        assert_eq!(
            QueryType::from_sparql("INSERT DATA { <s> <p> <o> }"),
            QueryType::Update
        );
    }

    #[test]
    fn test_detect_delete() {
        assert_eq!(
            QueryType::from_sparql("DELETE DATA { <s> <p> <o> }"),
            QueryType::Update
        );
    }

    #[test]
    fn test_detect_unknown() {
        assert_eq!(QueryType::from_sparql("FOOBAR"), QueryType::Unknown);
    }

    #[test]
    fn test_detect_with_leading_whitespace() {
        assert_eq!(
            QueryType::from_sparql("  \n  SELECT ?x WHERE {}"),
            QueryType::Select
        );
    }

    // ── QueryType display ─────────────────────────────────────────────────────

    #[test]
    fn test_query_type_display() {
        assert_eq!(QueryType::Select.to_string(), "SELECT");
        assert_eq!(QueryType::Construct.to_string(), "CONSTRUCT");
        assert_eq!(QueryType::Ask.to_string(), "ASK");
        assert_eq!(QueryType::Describe.to_string(), "DESCRIBE");
        assert_eq!(QueryType::Update.to_string(), "UPDATE");
        assert_eq!(QueryType::Unknown.to_string(), "UNKNOWN");
    }

    // ── QueryLog::new ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_log_is_empty() {
        let log = QueryLog::new(100);
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    // ── log_query / get ───────────────────────────────────────────────────────

    #[test]
    fn test_log_query_returns_id() {
        let mut log = QueryLog::new(100);
        let id = log.log_query("ds1", "SELECT ?s WHERE {}", 10, Some(5), None);
        assert_eq!(id, 1);
    }

    #[test]
    fn test_get_existing_entry() {
        let mut log = QueryLog::new(100);
        let id = log.log_query(
            "ds1",
            "SELECT ?s WHERE {}",
            10,
            Some(3),
            Some("127.0.0.1".to_string()),
        );
        let entry = log.get(id).expect("entry should exist");
        assert_eq!(entry.id, id);
        assert_eq!(entry.query_type, QueryType::Select);
        assert_eq!(entry.dataset, "ds1");
        assert_eq!(entry.duration_ms, 10);
        assert_eq!(entry.result_count, Some(3));
        assert_eq!(entry.client_ip, Some("127.0.0.1".to_string()));
        assert!(entry.error.is_none());
    }

    #[test]
    fn test_get_nonexistent_entry() {
        let log = QueryLog::new(100);
        assert!(log.get(999).is_none());
    }

    // ── log_error ─────────────────────────────────────────────────────────────

    #[test]
    fn test_log_error_entry() {
        let mut log = QueryLog::new(100);
        let id = log.log_error("ds1", "SELECT ?s WHERE {}", 5, "Syntax error", None);
        let entry = log.get(id).expect("entry should exist");
        assert_eq!(entry.error, Some("Syntax error".to_string()));
        assert!(entry.result_count.is_none());
    }

    // ── ring buffer behaviour ─────────────────────────────────────────────────

    #[test]
    fn test_ring_buffer_evicts_oldest() {
        let mut log = QueryLog::new(3);
        let id1 = log.log_query("ds", "SELECT 1", 1, None, None);
        let id2 = log.log_query("ds", "SELECT 2", 1, None, None);
        let id3 = log.log_query("ds", "SELECT 3", 1, None, None);
        // All three fit
        assert!(log.get(id1).is_some());
        assert!(log.get(id2).is_some());
        assert!(log.get(id3).is_some());
        // Adding a fourth evicts the first
        let id4 = log.log_query("ds", "SELECT 4", 1, None, None);
        assert!(log.get(id1).is_none()); // evicted
        assert!(log.get(id2).is_some());
        assert!(log.get(id3).is_some());
        assert!(log.get(id4).is_some());
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_ring_buffer_capacity_one() {
        let mut log = QueryLog::new(1);
        let id1 = log.log_query("ds", "SELECT 1", 1, None, None);
        assert!(log.get(id1).is_some());
        let id2 = log.log_query("ds", "SELECT 2", 1, None, None);
        assert!(log.get(id1).is_none());
        assert!(log.get(id2).is_some());
    }

    // ── recent ────────────────────────────────────────────────────────────────

    #[test]
    fn test_recent_order() {
        let mut log = QueryLog::new(100);
        log.log_query("ds", "SELECT 1", 1, None, None);
        log.log_query("ds", "SELECT 2", 2, None, None);
        log.log_query("ds", "SELECT 3", 3, None, None);
        let r = log.recent(2);
        assert_eq!(r.len(), 2);
        // Newest first
        assert_eq!(r[0].duration_ms, 3);
        assert_eq!(r[1].duration_ms, 2);
    }

    #[test]
    fn test_recent_more_than_available() {
        let mut log = QueryLog::new(100);
        log.log_query("ds", "SELECT 1", 1, None, None);
        let r = log.recent(10);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_recent_empty_log() {
        let log = QueryLog::new(100);
        assert!(log.recent(5).is_empty());
    }

    // ── by_type ───────────────────────────────────────────────────────────────

    #[test]
    fn test_by_type_select() {
        let mut log = QueryLog::new(100);
        log.log_query("ds", "SELECT ?s WHERE {}", 1, Some(0), None);
        log.log_query("ds", "ASK { ?s ?p ?o }", 1, None, None);
        log.log_query("ds", "SELECT ?p WHERE {}", 1, Some(2), None);
        let selects = log.by_type(QueryType::Select);
        assert_eq!(selects.len(), 2);
    }

    #[test]
    fn test_by_type_empty() {
        let mut log = QueryLog::new(100);
        log.log_query("ds", "SELECT ?s WHERE {}", 1, None, None);
        assert!(log.by_type(QueryType::Construct).is_empty());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let log = QueryLog::new(100);
        let s = log.stats();
        assert_eq!(s.total_queries, 0);
        assert_eq!(s.avg_duration_ms, 0.0);
        assert_eq!(s.error_count, 0);
        assert!(s.by_type.is_empty());
    }

    #[test]
    fn test_stats_totals() {
        let mut log = QueryLog::new(100);
        log.log_query("ds", "SELECT ?s WHERE {}", 10, Some(3), None);
        log.log_query("ds", "SELECT ?s WHERE {}", 20, Some(1), None);
        log.log_error("ds", "ASK {}", 5, "err", None);
        let s = log.stats();
        assert_eq!(s.total_queries, 3);
        assert!((s.avg_duration_ms - (35.0 / 3.0)).abs() < 0.001);
        assert_eq!(s.error_count, 1);
        assert_eq!(s.by_type["SELECT"], 2);
        assert_eq!(s.by_type["ASK"], 1);
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear() {
        let mut log = QueryLog::new(100);
        log.log_query("ds", "SELECT 1", 1, None, None);
        log.log_query("ds", "SELECT 2", 1, None, None);
        log.clear();
        assert!(log.is_empty());
    }

    // ── export_json ───────────────────────────────────────────────────────────

    #[test]
    fn test_export_json_empty() {
        let log = QueryLog::new(100);
        assert_eq!(log.export_json(), "[]");
    }

    #[test]
    fn test_export_json_contains_fields() {
        let mut log = QueryLog::new(100);
        log.log_query(
            "myds",
            "SELECT ?s WHERE { ?s ?p ?o }",
            42,
            Some(7),
            Some("10.0.0.1".to_string()),
        );
        let json = log.export_json();
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"query_type\":\"SELECT\""));
        assert!(json.contains("\"dataset\":\"myds\""));
        assert!(json.contains("\"duration_ms\":42"));
        assert!(json.contains("\"result_count\":7"));
        assert!(json.contains("\"client_ip\":\"10.0.0.1\""));
        assert!(json.contains("\"error\":null"));
    }

    #[test]
    fn test_export_json_error_entry() {
        let mut log = QueryLog::new(100);
        log.log_error("ds", "SELECT ?s", 1, "some error", None);
        let json = log.export_json();
        assert!(json.contains("\"error\":\"some error\""));
        assert!(json.contains("\"result_count\":null"));
    }

    #[test]
    fn test_export_json_multiple_entries() {
        let mut log = QueryLog::new(100);
        log.log_query("ds", "SELECT 1", 1, None, None);
        log.log_query("ds", "SELECT 2", 2, None, None);
        let json = log.export_json();
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        // Should have a comma between entries
        assert!(json.contains("},{"));
    }

    // ── monotonic IDs ─────────────────────────────────────────────────────────

    #[test]
    fn test_ids_are_monotonic() {
        let mut log = QueryLog::new(100);
        let id1 = log.log_query("ds", "SELECT 1", 1, None, None);
        let id2 = log.log_query("ds", "SELECT 2", 1, None, None);
        let id3 = log.log_error("ds", "SELECT 3", 1, "err", None);
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    // ── LogError display ──────────────────────────────────────────────────────

    #[test]
    fn test_log_error_display() {
        let e = LogError::NotFound(42);
        assert!(e.to_string().contains("42"));
    }
}
