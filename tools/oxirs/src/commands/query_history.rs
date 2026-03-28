//! Query History Store with Analytics and CSV Export
//!
//! Provides a persistent, analytics-ready store for executed SPARQL queries.
//! Complements the existing `history` module by exposing a richer API:
//! regex search, frequency analysis, per-query average duration tracking,
//! and CSV export.
//!
//! ## Features
//!
//! - `record()` — persist a query execution with duration and result count
//! - `search()` — regex-based full-text search over stored queries
//! - `most_frequent()` — top-N queries by execution count
//! - `slowest_queries()` — top-N queries by average duration
//! - `export_csv()` — write all entries to a CSV file
//! - Test-friendly: uses `std::env::temp_dir()` in tests for isolation

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// A single recorded query execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HistoryEntry {
    /// Auto-incrementing unique identifier
    pub id: u64,
    /// RFC3339 timestamp of execution
    pub timestamp: DateTime<Utc>,
    /// SPARQL query text
    pub query: String,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Number of results returned
    pub result_count: usize,
}

/// Persistent store for SPARQL query execution history
///
/// Serialises to JSON on disk; safe to load from a partially-existing file.
pub struct QueryHistoryStore {
    /// All recorded entries, ordered by insertion
    entries: Vec<HistoryEntry>,
    /// Next available ID
    next_id: u64,
    /// Path to the backing JSON file (None = in-memory only)
    backing_file: Option<std::path::PathBuf>,
}

impl QueryHistoryStore {
    /// Create an in-memory store with no persistence
    pub fn in_memory() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            backing_file: None,
        }
    }

    /// Create (or load) a persisted store at the given path
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut store = Self {
            entries: Vec::new(),
            next_id: 1,
            backing_file: Some(path.clone()),
        };
        if path.exists() {
            store.load()?;
        }
        Ok(store)
    }

    /// Load entries from the backing file
    fn load(&mut self) -> Result<()> {
        let Some(ref path) = self.backing_file else {
            return Ok(());
        };
        if !path.exists() {
            return Ok(());
        }
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read history store: {}", path.display()))?;
        if content.trim().is_empty() {
            return Ok(());
        }
        self.entries = serde_json::from_str(&content)
            .with_context(|| "Failed to deserialise history entries")?;
        self.next_id = self.entries.iter().map(|e| e.id).max().unwrap_or(0) + 1;
        Ok(())
    }

    /// Persist entries to the backing file
    fn save(&self) -> Result<()> {
        let Some(ref path) = self.backing_file else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(&self.entries)
            .with_context(|| "Failed to serialise history entries")?;
        fs::write(path, json)
            .with_context(|| format!("Failed to write history store: {}", path.display()))?;
        Ok(())
    }

    /// Record a query execution and return the created entry
    ///
    /// Persists to disk automatically if a backing file is configured.
    pub fn record(
        &mut self,
        query: &str,
        duration_ms: u64,
        result_count: usize,
    ) -> Result<HistoryEntry> {
        let entry = HistoryEntry {
            id: self.next_id,
            timestamp: Utc::now(),
            query: query.to_string(),
            duration_ms,
            result_count,
        };
        self.next_id += 1;
        self.entries.push(entry.clone());
        self.save()?;
        Ok(entry)
    }

    /// Search entries whose query text matches the given regex pattern
    ///
    /// Returns matching entries in insertion order.
    pub fn search(&self, pattern: &str) -> Result<Vec<&HistoryEntry>> {
        let re =
            Regex::new(pattern).with_context(|| format!("Invalid regex pattern: '{}'", pattern))?;
        Ok(self
            .entries
            .iter()
            .filter(|e| re.is_match(&e.query))
            .collect())
    }

    /// Return the top-N most frequently executed queries
    ///
    /// Frequency is measured by exact query text equality.
    /// Returns `(query_text, count)` pairs sorted descending by count.
    pub fn most_frequent(&self, top_n: usize) -> Vec<(&str, usize)> {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for entry in &self.entries {
            *freq.entry(entry.query.as_str()).or_insert(0) += 1;
        }
        let mut sorted: Vec<(&str, usize)> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        sorted.into_iter().take(top_n).collect()
    }

    /// Return the top-N slowest queries by their average execution duration
    ///
    /// Averages are computed over all recorded executions of the same query text.
    /// Returns `(entry, avg_duration_ms)` pairs sorted descending by average duration.
    ///
    /// The returned `entry` is the *last* recorded execution of that query.
    pub fn slowest_queries(&self, top_n: usize) -> Vec<(&HistoryEntry, u64)> {
        // Accumulate total duration and count per query text
        let mut acc: HashMap<&str, (u64, usize)> = HashMap::new();
        for entry in &self.entries {
            let e = acc.entry(entry.query.as_str()).or_insert((0, 0));
            e.0 += entry.duration_ms;
            e.1 += 1;
        }

        // Find the representative entry (last occurrence) for each query
        let mut last_entry: HashMap<&str, &HistoryEntry> = HashMap::new();
        for entry in &self.entries {
            last_entry.insert(entry.query.as_str(), entry);
        }

        let mut result: Vec<(&HistoryEntry, u64)> = acc
            .iter()
            .filter_map(|(query, (total, count))| {
                let avg = total / (*count as u64).max(1);
                last_entry.get(query).map(|e| (*e, avg))
            })
            .collect();

        result.sort_by_key(|item| std::cmp::Reverse(item.1));
        result.into_iter().take(top_n).collect()
    }

    /// Export all entries to a CSV file at the given path
    pub fn export_csv(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        let file = fs::File::create(path)
            .with_context(|| format!("Failed to create CSV file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        // Header
        writeln!(writer, "id,timestamp,duration_ms,result_count,query")
            .with_context(|| "Failed to write CSV header")?;
        for entry in &self.entries {
            let ts = entry.timestamp.to_rfc3339();
            let escaped_query = csv_escape_field(&entry.query);
            writeln!(
                writer,
                "{},{},{},{},{}",
                entry.id, ts, entry.duration_ms, entry.result_count, escaped_query
            )
            .with_context(|| format!("Failed to write CSV row for entry {}", entry.id))?;
        }
        writer
            .flush()
            .with_context(|| "Failed to flush CSV writer")?;
        Ok(())
    }

    /// Return all entries in insertion order
    pub fn all_entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Return the total number of recorded entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no entries have been recorded
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries (does not affect the backing file until next save)
    pub fn clear(&mut self) -> Result<()> {
        self.entries.clear();
        self.next_id = 1;
        self.save()
    }

    /// Get a single entry by ID
    pub fn get(&self, id: u64) -> Option<&HistoryEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

/// Escape a field value for CSV output (RFC 4180)
fn csv_escape_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Create a temporary file path in the OS temp dir
    fn tmp_path(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!("oxirs_qh_test_{name}_{}.json", std::process::id()))
    }

    fn tmp_csv_path(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!("oxirs_qh_test_{name}_{}.csv", std::process::id()))
    }

    // --- in-memory store basics ---

    #[test]
    fn test_in_memory_store_empty_on_creation() {
        let store = QueryHistoryStore::in_memory();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_record_returns_entry_with_correct_fields() {
        let mut store = QueryHistoryStore::in_memory();
        let q = "SELECT ?s WHERE { ?s rdf:type <http://a> . }";
        let entry = store.record(q, 42, 10).unwrap();
        assert_eq!(entry.query, q);
        assert_eq!(entry.duration_ms, 42);
        assert_eq!(entry.result_count, 10);
        assert_eq!(entry.id, 1);
    }

    #[test]
    fn test_record_increments_id() {
        let mut store = QueryHistoryStore::in_memory();
        let e1 = store
            .record("SELECT ?s WHERE { ?s ?p ?o . }", 10, 5)
            .unwrap();
        let e2 = store
            .record("SELECT ?x WHERE { ?x rdf:type <http://b> . }", 20, 3)
            .unwrap();
        assert_eq!(e1.id, 1);
        assert_eq!(e2.id, 2);
    }

    #[test]
    fn test_len_increases_with_records() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 1, 0)
            .unwrap();
        assert_eq!(store.len(), 1);
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://b> . }", 2, 0)
            .unwrap();
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_all_entries_returns_in_insertion_order() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?a WHERE { ?a rdf:type <http://A> . }", 1, 0)
            .unwrap();
        store
            .record("SELECT ?b WHERE { ?b rdf:type <http://B> . }", 2, 0)
            .unwrap();
        let entries = store.all_entries();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].query.contains('?'));
        assert!(entries[1].query.contains('?'));
    }

    #[test]
    fn test_get_by_id() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 1, 0)
            .unwrap();
        let entry = store.get(1);
        assert!(entry.is_some());
        assert_eq!(store.get(999), None);
    }

    #[test]
    fn test_clear_resets_store() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 5, 0)
            .unwrap();
        store.clear().unwrap();
        assert!(store.is_empty());
    }

    // --- regex search ---

    #[test]
    fn test_search_finds_matching_queries() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://Person> . }", 10, 0)
            .unwrap();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://Product> . }", 10, 0)
            .unwrap();
        let results = store.search("Person").unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].query.contains("Person"));
    }

    #[test]
    fn test_search_returns_empty_on_no_match() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 10, 0)
            .unwrap();
        let results = store.search("NoMatch").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_regex_pattern() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record(
                "SELECT ?s WHERE { ?s <http://schema.org/name> ?name . }",
                10,
                0,
            )
            .unwrap();
        store
            .record("SELECT ?x WHERE { ?x rdf:type <http://b> . }", 10, 0)
            .unwrap();
        let results = store.search(r"schema\.org").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_invalid_regex_returns_error() {
        let store = QueryHistoryStore::in_memory();
        let result = store.search("[invalid(regex");
        assert!(result.is_err());
    }

    #[test]
    fn test_search_multiple_matches() {
        let mut store = QueryHistoryStore::in_memory();
        for i in 0..5 {
            store
                .record(
                    &format!("SELECT ?s WHERE {{ ?s rdf:type <http://example.org/T{i}> . }}"),
                    10,
                    0,
                )
                .unwrap();
        }
        let results = store.search("example.org").unwrap();
        assert_eq!(results.len(), 5);
    }

    // --- most_frequent ---

    #[test]
    fn test_most_frequent_returns_top_n() {
        let mut store = QueryHistoryStore::in_memory();
        let q1 = "SELECT ?s WHERE { ?s rdf:type <http://a> . }";
        let q2 = "SELECT ?s WHERE { ?s rdf:type <http://b> . }";
        let q3 = "SELECT ?s WHERE { ?s rdf:type <http://c> . }";
        for _ in 0..3 {
            store.record(q1, 10, 0).unwrap();
        }
        for _ in 0..5 {
            store.record(q2, 10, 0).unwrap();
        }
        for _ in 0..1 {
            store.record(q3, 10, 0).unwrap();
        }
        let top2 = store.most_frequent(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, q2);
        assert_eq!(top2[0].1, 5);
        assert_eq!(top2[1].0, q1);
        assert_eq!(top2[1].1, 3);
    }

    #[test]
    fn test_most_frequent_empty_store() {
        let store = QueryHistoryStore::in_memory();
        let result = store.most_frequent(5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_most_frequent_n_larger_than_unique() {
        let mut store = QueryHistoryStore::in_memory();
        let q = "SELECT ?s WHERE { ?s rdf:type <http://a> . }";
        store.record(q, 10, 0).unwrap();
        let result = store.most_frequent(10);
        assert_eq!(result.len(), 1);
    }

    // --- slowest_queries ---

    #[test]
    fn test_slowest_queries_returns_top_n_by_avg_duration() {
        let mut store = QueryHistoryStore::in_memory();
        let fast = "SELECT ?s WHERE { ?s rdf:type <http://fast> . }";
        let slow = "SELECT ?s WHERE { ?s rdf:type <http://slow> . }";
        store.record(fast, 10, 0).unwrap();
        store.record(fast, 20, 0).unwrap(); // avg = 15
        store.record(slow, 500, 0).unwrap();
        store.record(slow, 600, 0).unwrap(); // avg = 550

        let top1 = store.slowest_queries(1);
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].1, 550); // avg of slow
    }

    #[test]
    fn test_slowest_queries_empty_store() {
        let store = QueryHistoryStore::in_memory();
        let result = store.slowest_queries(5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_slowest_queries_single_execution() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 300, 0)
            .unwrap();
        let result = store.slowest_queries(1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 300);
    }

    // --- export_csv ---

    #[test]
    fn test_export_csv_creates_file() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 10, 5)
            .unwrap();
        let path = tmp_csv_path("export_creates");
        store.export_csv(&path).unwrap();
        assert!(path.exists());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_export_csv_contains_header() {
        let store = QueryHistoryStore::in_memory();
        let path = tmp_csv_path("export_header");
        store.export_csv(&path).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("id,timestamp,duration_ms,result_count,query"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_export_csv_contains_entry_data() {
        let mut store = QueryHistoryStore::in_memory();
        store
            .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 42, 7)
            .unwrap();
        let path = tmp_csv_path("export_data");
        store.export_csv(&path).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("42"));
        assert!(content.contains("7"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_export_csv_escapes_commas_in_query() {
        let mut store = QueryHistoryStore::in_memory();
        // Query text with a comma (e.g. in a VALUES clause)
        store
            .record(
                "SELECT ?s WHERE { VALUES ?s { <http://a>, <http://b> } }",
                10,
                0,
            )
            .unwrap();
        let path = tmp_csv_path("export_escape");
        store.export_csv(&path).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        // The query field should be quoted
        assert!(content.contains('"'));
        let _ = fs::remove_file(&path);
    }

    // --- persistence ---

    #[test]
    fn test_open_creates_store_and_persists() {
        let path = tmp_path("persist");
        {
            let mut store = QueryHistoryStore::open(&path).unwrap();
            store
                .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 10, 0)
                .unwrap();
        }
        // Re-open and check
        let store2 = QueryHistoryStore::open(&path).unwrap();
        assert_eq!(store2.len(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_open_loads_existing_data() {
        let path = tmp_path("load_existing");
        {
            let mut store = QueryHistoryStore::open(&path).unwrap();
            store
                .record("SELECT ?s WHERE { ?s rdf:type <http://a> . }", 1, 0)
                .unwrap();
            store
                .record("SELECT ?x WHERE { ?x rdf:type <http://b> . }", 2, 0)
                .unwrap();
        }
        let store = QueryHistoryStore::open(&path).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.next_id, 3);
        let _ = fs::remove_file(&path);
    }

    // --- csv_escape_field ---

    #[test]
    fn test_csv_escape_plain_value() {
        assert_eq!(csv_escape_field("hello"), "hello");
    }

    #[test]
    fn test_csv_escape_value_with_comma() {
        let escaped = csv_escape_field("a,b");
        assert!(escaped.starts_with('"'));
        assert!(escaped.ends_with('"'));
    }

    #[test]
    fn test_csv_escape_value_with_quote() {
        let escaped = csv_escape_field("say \"hi\"");
        assert!(escaped.contains("\"\""));
    }
}
