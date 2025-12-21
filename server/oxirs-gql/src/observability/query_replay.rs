//! Query Replay for Debugging
//!
//! Provides query replay capabilities for debugging and testing,
//! allowing recorded queries to be replayed with the same context.
//!
//! # Features
//!
//! - Query recording with full context
//! - Replay with timing and result comparison
//! - Conditional replay (filters)
//! - Replay speed control
//! - Diff generation between original and replay
//! - Batch replay support

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Recorded query with full execution context
#[derive(Debug, Clone)]
pub struct RecordedQuery {
    pub id: String,
    pub timestamp: SystemTime,
    pub query_text: String,
    pub operation_name: Option<String>,
    pub variables: HashMap<String, String>,
    pub request_headers: HashMap<String, String>,
    pub user_id: Option<String>,
    pub result: Option<String>,
    pub duration_ms: f64,
    pub error: Option<String>,
    pub tags: HashMap<String, String>,
}

/// Replay result with comparison
#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub query_id: String,
    pub original_result: Option<String>,
    pub replay_result: Option<String>,
    pub original_duration_ms: f64,
    pub replay_duration_ms: f64,
    pub results_match: bool,
    pub error: Option<String>,
    pub diff: Option<String>,
}

/// Replay configuration
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    pub replay_speed: f64, // 1.0 = normal, 2.0 = 2x speed
    pub stop_on_error: bool,
    pub compare_results: bool,
    pub skip_errors: bool,
    pub timeout_ms: u64,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            replay_speed: 1.0,
            stop_on_error: false,
            compare_results: true,
            skip_errors: false,
            timeout_ms: 30000,
        }
    }
}

/// Query filter for selective replay
#[derive(Debug, Clone)]
pub struct QueryFilter {
    pub operation_names: Option<Vec<String>>,
    pub user_ids: Option<Vec<String>>,
    pub time_range: Option<(SystemTime, SystemTime)>,
    pub min_duration_ms: Option<f64>,
    pub max_duration_ms: Option<f64>,
    pub has_error: Option<bool>,
    pub tags: HashMap<String, String>,
}

impl QueryFilter {
    pub fn matches(&self, query: &RecordedQuery) -> bool {
        // Check operation names
        if let Some(ref names) = self.operation_names {
            if let Some(ref op_name) = query.operation_name {
                if !names.contains(op_name) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check user IDs
        if let Some(ref user_ids) = self.user_ids {
            if let Some(ref user_id) = query.user_id {
                if !user_ids.contains(user_id) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check time range
        if let Some((start, end)) = self.time_range {
            if query.timestamp < start || query.timestamp > end {
                return false;
            }
        }

        // Check duration range
        if let Some(min) = self.min_duration_ms {
            if query.duration_ms < min {
                return false;
            }
        }

        if let Some(max) = self.max_duration_ms {
            if query.duration_ms > max {
                return false;
            }
        }

        // Check error status
        if let Some(has_error) = self.has_error {
            if has_error != query.error.is_some() {
                return false;
            }
        }

        // Check tags
        for (key, value) in &self.tags {
            if query.tags.get(key) != Some(value) {
                return false;
            }
        }

        true
    }
}

/// Query replay manager
pub struct QueryReplay {
    recordings: Vec<RecordedQuery>,
    max_recordings: usize,
}

impl QueryReplay {
    /// Create a new query replay manager
    pub fn new(max_recordings: usize) -> Self {
        Self {
            recordings: Vec::new(),
            max_recordings,
        }
    }

    /// Record a query execution
    pub fn record(
        &mut self,
        query_text: String,
        operation_name: Option<String>,
        variables: HashMap<String, String>,
        result: Option<String>,
        duration_ms: f64,
        error: Option<String>,
    ) -> String {
        // Generate unique ID
        let id = format!("query-{}", self.recordings.len());

        let recording = RecordedQuery {
            id: id.clone(),
            timestamp: SystemTime::now(),
            query_text,
            operation_name,
            variables,
            request_headers: HashMap::new(),
            user_id: None,
            result,
            duration_ms,
            error,
            tags: HashMap::new(),
        };

        self.recordings.push(recording);

        // Trim if at limit
        if self.recordings.len() > self.max_recordings {
            self.recordings
                .drain(0..self.recordings.len() - self.max_recordings);
        }

        id
    }

    /// Record with full context
    pub fn record_full(&mut self, recording: RecordedQuery) {
        self.recordings.push(recording);

        if self.recordings.len() > self.max_recordings {
            self.recordings
                .drain(0..self.recordings.len() - self.max_recordings);
        }
    }

    /// Get a recording by ID
    pub fn get_recording(&self, id: &str) -> Option<&RecordedQuery> {
        self.recordings.iter().find(|r| r.id == id)
    }

    /// Get all recordings
    pub fn get_all_recordings(&self) -> &[RecordedQuery] {
        &self.recordings
    }

    /// Get filtered recordings
    pub fn get_filtered_recordings(&self, filter: &QueryFilter) -> Vec<&RecordedQuery> {
        self.recordings
            .iter()
            .filter(|r| filter.matches(r))
            .collect()
    }

    /// Replay a single query
    pub fn replay_query<F>(
        &self,
        id: &str,
        config: &ReplayConfig,
        executor: &mut F,
    ) -> Option<ReplayResult>
    where
        F: FnMut(&str, &HashMap<String, String>) -> (Option<String>, Duration, Option<String>),
    {
        let recording = self.get_recording(id)?;

        // Execute the query
        let (result, duration, error) = executor(&recording.query_text, &recording.variables);

        let replay_duration_ms = duration.as_secs_f64() * 1000.0;

        // Compare results if enabled
        let results_match = if config.compare_results {
            match (&recording.result, &result) {
                (Some(orig), Some(replay)) => orig == replay,
                (None, None) => true,
                _ => false,
            }
        } else {
            true
        };

        // Generate diff if results don't match
        let diff = if !results_match && config.compare_results {
            Some(Self::generate_diff(
                recording.result.as_deref(),
                result.as_deref(),
            ))
        } else {
            None
        };

        Some(ReplayResult {
            query_id: id.to_string(),
            original_result: recording.result.clone(),
            replay_result: result,
            original_duration_ms: recording.duration_ms,
            replay_duration_ms,
            results_match,
            error,
            diff,
        })
    }

    /// Replay multiple queries
    pub fn replay_batch<F>(
        &self,
        filter: &QueryFilter,
        config: &ReplayConfig,
        mut executor: F,
    ) -> Vec<ReplayResult>
    where
        F: FnMut(&str, &HashMap<String, String>) -> (Option<String>, Duration, Option<String>),
    {
        let queries = self.get_filtered_recordings(filter);
        let mut results = Vec::new();

        for recording in queries {
            if let Some(result) = self.replay_query(&recording.id, config, &mut executor) {
                // Check stop on error
                if config.stop_on_error && result.error.is_some() {
                    results.push(result);
                    break;
                }

                results.push(result);
            }
        }

        results
    }

    /// Clear all recordings
    pub fn clear(&mut self) {
        self.recordings.clear();
    }

    /// Get recording count
    pub fn count(&self) -> usize {
        self.recordings.len()
    }

    /// Export recordings as JSON
    pub fn export_json(&self) -> String {
        let mut json = String::from("[");

        for (i, recording) in self.recordings.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":\"{}\",\"query\":\"{}\",\"duration_ms\":{}}}",
                recording.id,
                recording.query_text.replace('"', "\\\""),
                recording.duration_ms
            ));
        }

        json.push(']');
        json
    }

    /// Generate diff between two results
    fn generate_diff(original: Option<&str>, replay: Option<&str>) -> String {
        match (original, replay) {
            (Some(orig), Some(replay)) => {
                if orig == replay {
                    "No differences".to_string()
                } else {
                    format!("Original:\n{}\n\nReplay:\n{}", orig, replay)
                }
            }
            (Some(orig), None) => format!("Original: {}\nReplay: None", orig),
            (None, Some(replay)) => format!("Original: None\nReplay: {}", replay),
            (None, None) => "Both None".to_string(),
        }
    }

    /// Generate replay summary
    pub fn generate_summary(&self, results: &[ReplayResult]) -> ReplaySummary {
        let mut matching = 0;
        let mut mismatching = 0;
        let mut errors = 0;
        let mut total_original_duration = 0.0;
        let mut total_replay_duration = 0.0;

        for result in results {
            if result.error.is_some() {
                errors += 1;
            } else if result.results_match {
                matching += 1;
            } else {
                mismatching += 1;
            }

            total_original_duration += result.original_duration_ms;
            total_replay_duration += result.replay_duration_ms;
        }

        ReplaySummary {
            total_queries: results.len(),
            matching_results: matching,
            mismatching_results: mismatching,
            errors,
            avg_original_duration_ms: if !results.is_empty() {
                total_original_duration / results.len() as f64
            } else {
                0.0
            },
            avg_replay_duration_ms: if !results.is_empty() {
                total_replay_duration / results.len() as f64
            } else {
                0.0
            },
        }
    }
}

/// Replay summary statistics
#[derive(Debug, Clone)]
pub struct ReplaySummary {
    pub total_queries: usize,
    pub matching_results: usize,
    pub mismatching_results: usize,
    pub errors: usize,
    pub avg_original_duration_ms: f64,
    pub avg_replay_duration_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_query() {
        let mut replay = QueryReplay::new(100);

        let id = replay.record(
            "{ user { id } }".to_string(),
            Some("GetUser".to_string()),
            HashMap::new(),
            Some("result".to_string()),
            10.5,
            None,
        );

        assert_eq!(replay.count(), 1);
        assert!(replay.get_recording(&id).is_some());
    }

    #[test]
    fn test_max_recordings_limit() {
        let mut replay = QueryReplay::new(2);

        replay.record("query1".to_string(), None, HashMap::new(), None, 1.0, None);
        replay.record("query2".to_string(), None, HashMap::new(), None, 2.0, None);
        replay.record("query3".to_string(), None, HashMap::new(), None, 3.0, None);

        assert_eq!(replay.count(), 2);
    }

    #[test]
    fn test_get_recording() {
        let mut replay = QueryReplay::new(100);

        let id = replay.record(
            "query".to_string(),
            None,
            HashMap::new(),
            Some("result".to_string()),
            5.0,
            None,
        );

        let recording = replay.get_recording(&id).unwrap();
        assert_eq!(recording.query_text, "query");
        assert_eq!(recording.duration_ms, 5.0);
    }

    #[test]
    fn test_filter_by_operation_name() {
        let mut replay = QueryReplay::new(100);

        replay.record(
            "query1".to_string(),
            Some("GetUser".to_string()),
            HashMap::new(),
            None,
            1.0,
            None,
        );
        replay.record(
            "query2".to_string(),
            Some("GetPost".to_string()),
            HashMap::new(),
            None,
            2.0,
            None,
        );

        let filter = QueryFilter {
            operation_names: Some(vec!["GetUser".to_string()]),
            user_ids: None,
            time_range: None,
            min_duration_ms: None,
            max_duration_ms: None,
            has_error: None,
            tags: HashMap::new(),
        };

        let filtered = replay.get_filtered_recordings(&filter);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].operation_name, Some("GetUser".to_string()));
    }

    #[test]
    fn test_filter_by_duration() {
        let mut replay = QueryReplay::new(100);

        replay.record("query1".to_string(), None, HashMap::new(), None, 5.0, None);
        replay.record("query2".to_string(), None, HashMap::new(), None, 15.0, None);
        replay.record("query3".to_string(), None, HashMap::new(), None, 25.0, None);

        let filter = QueryFilter {
            operation_names: None,
            user_ids: None,
            time_range: None,
            min_duration_ms: Some(10.0),
            max_duration_ms: Some(20.0),
            has_error: None,
            tags: HashMap::new(),
        };

        let filtered = replay.get_filtered_recordings(&filter);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].duration_ms, 15.0);
    }

    #[test]
    fn test_filter_by_error_status() {
        let mut replay = QueryReplay::new(100);

        replay.record("query1".to_string(), None, HashMap::new(), None, 1.0, None);
        replay.record(
            "query2".to_string(),
            None,
            HashMap::new(),
            None,
            2.0,
            Some("Error".to_string()),
        );

        let filter = QueryFilter {
            operation_names: None,
            user_ids: None,
            time_range: None,
            min_duration_ms: None,
            max_duration_ms: None,
            has_error: Some(true),
            tags: HashMap::new(),
        };

        let filtered = replay.get_filtered_recordings(&filter);
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].error.is_some());
    }

    #[test]
    fn test_replay_query() {
        let mut replay = QueryReplay::new(100);

        let id = replay.record(
            "query".to_string(),
            None,
            HashMap::new(),
            Some("original".to_string()),
            10.0,
            None,
        );

        let mut executor = |_query: &str, _vars: &HashMap<String, String>| {
            (
                Some("original".to_string()),
                Duration::from_millis(12),
                None,
            )
        };

        let result = replay
            .replay_query(&id, &ReplayConfig::default(), &mut executor)
            .unwrap();

        assert!(result.results_match);
        assert_eq!(result.original_duration_ms, 10.0);
    }

    #[test]
    fn test_replay_query_mismatch() {
        let mut replay = QueryReplay::new(100);

        let id = replay.record(
            "query".to_string(),
            None,
            HashMap::new(),
            Some("original".to_string()),
            10.0,
            None,
        );

        let mut executor = |_query: &str, _vars: &HashMap<String, String>| {
            (
                Some("different".to_string()),
                Duration::from_millis(12),
                None,
            )
        };

        let result = replay
            .replay_query(&id, &ReplayConfig::default(), &mut executor)
            .unwrap();

        assert!(!result.results_match);
        assert!(result.diff.is_some());
    }

    #[test]
    fn test_replay_batch() {
        let mut replay = QueryReplay::new(100);

        replay.record("query1".to_string(), None, HashMap::new(), None, 1.0, None);
        replay.record("query2".to_string(), None, HashMap::new(), None, 2.0, None);

        let executor =
            |_query: &str, _vars: &HashMap<String, String>| (None, Duration::from_millis(1), None);

        let filter = QueryFilter {
            operation_names: None,
            user_ids: None,
            time_range: None,
            min_duration_ms: None,
            max_duration_ms: None,
            has_error: None,
            tags: HashMap::new(),
        };

        let results = replay.replay_batch(&filter, &ReplayConfig::default(), executor);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_replay_stop_on_error() {
        let mut replay = QueryReplay::new(100);

        replay.record("query1".to_string(), None, HashMap::new(), None, 1.0, None);
        replay.record("query2".to_string(), None, HashMap::new(), None, 2.0, None);
        replay.record("query3".to_string(), None, HashMap::new(), None, 3.0, None);

        let mut call_count = 0;
        let executor = move |_query: &str, _vars: &HashMap<String, String>| {
            call_count += 1;
            if call_count == 2 {
                (None, Duration::from_millis(1), Some("Error".to_string()))
            } else {
                (None, Duration::from_millis(1), None)
            }
        };

        let config = ReplayConfig {
            stop_on_error: true,
            ..Default::default()
        };

        let filter = QueryFilter {
            operation_names: None,
            user_ids: None,
            time_range: None,
            min_duration_ms: None,
            max_duration_ms: None,
            has_error: None,
            tags: HashMap::new(),
        };

        let results = replay.replay_batch(&filter, &config, executor);
        assert_eq!(results.len(), 2); // Stopped after error
    }

    #[test]
    fn test_export_json() {
        let mut replay = QueryReplay::new(100);

        replay.record("query1".to_string(), None, HashMap::new(), None, 1.0, None);
        replay.record("query2".to_string(), None, HashMap::new(), None, 2.0, None);

        let json = replay.export_json();
        assert!(json.contains("query1"));
        assert!(json.contains("query2"));
    }

    #[test]
    fn test_generate_summary() {
        let replay = QueryReplay::new(100);

        let results = vec![
            ReplayResult {
                query_id: "1".to_string(),
                original_result: Some("a".to_string()),
                replay_result: Some("a".to_string()),
                original_duration_ms: 10.0,
                replay_duration_ms: 12.0,
                results_match: true,
                error: None,
                diff: None,
            },
            ReplayResult {
                query_id: "2".to_string(),
                original_result: Some("a".to_string()),
                replay_result: Some("b".to_string()),
                original_duration_ms: 20.0,
                replay_duration_ms: 18.0,
                results_match: false,
                error: None,
                diff: Some("diff".to_string()),
            },
        ];

        let summary = replay.generate_summary(&results);
        assert_eq!(summary.total_queries, 2);
        assert_eq!(summary.matching_results, 1);
        assert_eq!(summary.mismatching_results, 1);
        assert_eq!(summary.avg_original_duration_ms, 15.0);
    }

    #[test]
    fn test_clear() {
        let mut replay = QueryReplay::new(100);

        replay.record("query".to_string(), None, HashMap::new(), None, 1.0, None);
        assert_eq!(replay.count(), 1);

        replay.clear();
        assert_eq!(replay.count(), 0);
    }

    #[test]
    fn test_generate_diff() {
        let diff1 = QueryReplay::generate_diff(Some("a"), Some("b"));
        assert!(diff1.contains("Original"));
        assert!(diff1.contains("Replay"));

        let diff2 = QueryReplay::generate_diff(Some("a"), Some("a"));
        assert!(diff2.contains("No differences"));

        let diff3 = QueryReplay::generate_diff(None, None);
        assert!(diff3.contains("Both None"));
    }

    #[test]
    fn test_filter_by_tags() {
        let mut replay = QueryReplay::new(100);

        let mut tags1 = HashMap::new();
        tags1.insert("environment".to_string(), "production".to_string());

        let recording1 = RecordedQuery {
            id: "query-0".to_string(),
            timestamp: SystemTime::now(),
            query_text: "query1".to_string(),
            operation_name: None,
            variables: HashMap::new(),
            request_headers: HashMap::new(),
            user_id: None,
            result: None,
            duration_ms: 1.0,
            error: None,
            tags: tags1,
        };

        replay.record_full(recording1);

        let mut filter_tags = HashMap::new();
        filter_tags.insert("environment".to_string(), "production".to_string());

        let filter = QueryFilter {
            operation_names: None,
            user_ids: None,
            time_range: None,
            min_duration_ms: None,
            max_duration_ms: None,
            has_error: None,
            tags: filter_tags,
        };

        let filtered = replay.get_filtered_recordings(&filter);
        assert_eq!(filtered.len(), 1);
    }
}
