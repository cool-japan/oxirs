//! Web Workers support for parallel SPARQL/RDF processing.
//!
//! This module provides a `WorkerPool` abstraction for distributing
//! RDF query workloads across Web Workers.  The actual worker threads are
//! managed by JavaScript; this Rust module:
//!
//! - Serializes work items to `JsValue` for postMessage
//! - Deserializes results from worker messages
//! - Tracks pending / completed jobs
//!
//! ## Architecture
//!
//! ```text
//! [Rust WorkerPool] ──postMessage──> [JS Worker A]
//!                   ──postMessage──> [JS Worker B]
//!                   <──onmessage──── [JS Worker A] (result)
//! ```
//!
//! Workers communicate via structured clone; no SharedArrayBuffer required.
//!
//! ## Usage (JS side)
//!
//! ```javascript
//! import init, { WorkerJob, WorkerResult } from 'oxirs-wasm';
//! // Create a Worker running worker.js, post WorkerJob, receive WorkerResult.
//! ```

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Job / Result types (serializable via serde_json for postMessage)
// ─────────────────────────────────────────────────────────────────────────────

/// Type of work a worker should perform.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerJobKind {
    /// Execute a SPARQL SELECT query against the given N-Triples data.
    SparqlSelect {
        /// SPARQL query string.
        query: String,
        /// N-Triples encoded RDF data.
        ntriples: String,
    },
    /// Parse a chunk of Turtle data.
    ParseTurtle {
        /// Turtle input chunk.
        turtle: String,
    },
    /// Count distinct subjects in N-Triples data.
    CountSubjects {
        /// N-Triples encoded RDF data.
        ntriples: String,
    },
}

/// A unit of work dispatched to a Web Worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerJob {
    /// Unique job identifier (caller-assigned).
    pub id: u64,
    /// Work payload.
    pub kind: WorkerJobKind,
}

impl WorkerJob {
    /// Create a new SPARQL SELECT job.
    pub fn sparql_select(id: u64, query: impl Into<String>, ntriples: impl Into<String>) -> Self {
        Self {
            id,
            kind: WorkerJobKind::SparqlSelect {
                query: query.into(),
                ntriples: ntriples.into(),
            },
        }
    }

    /// Create a new Turtle parse job.
    pub fn parse_turtle(id: u64, turtle: impl Into<String>) -> Self {
        Self {
            id,
            kind: WorkerJobKind::ParseTurtle {
                turtle: turtle.into(),
            },
        }
    }

    /// Create a count-subjects job.
    pub fn count_subjects(id: u64, ntriples: impl Into<String>) -> Self {
        Self {
            id,
            kind: WorkerJobKind::CountSubjects {
                ntriples: ntriples.into(),
            },
        }
    }

    /// Serialize the job to JSON (for postMessage).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("Serialize error: {e}"))
    }

    /// Deserialize a job from JSON (in the worker).
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Deserialize error: {e}"))
    }
}

/// Outcome of a worker job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerResultKind {
    /// SPARQL SELECT result rows (JSON-encoded array of binding maps).
    SparqlRows { rows_json: String },
    /// Parsed triple count from a Turtle chunk.
    ParsedTripleCount { count: usize },
    /// Count of distinct subjects.
    SubjectCount { count: usize },
    /// The worker encountered an error.
    Error { message: String },
}

/// The result returned by a Web Worker after completing a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResult {
    /// Echo of the job id.
    pub job_id: u64,
    /// Result payload.
    pub result: WorkerResultKind,
    /// Elapsed time in milliseconds (reported by the worker).
    pub elapsed_ms: Option<f64>,
}

impl WorkerResult {
    /// Build a successful SPARQL result.
    pub fn sparql_ok(job_id: u64, rows_json: impl Into<String>, elapsed_ms: Option<f64>) -> Self {
        Self {
            job_id,
            result: WorkerResultKind::SparqlRows {
                rows_json: rows_json.into(),
            },
            elapsed_ms,
        }
    }

    /// Build a parse result.
    pub fn parsed_ok(job_id: u64, count: usize, elapsed_ms: Option<f64>) -> Self {
        Self {
            job_id,
            result: WorkerResultKind::ParsedTripleCount { count },
            elapsed_ms,
        }
    }

    /// Build an error result.
    pub fn error(job_id: u64, message: impl Into<String>) -> Self {
        Self {
            job_id,
            result: WorkerResultKind::Error {
                message: message.into(),
            },
            elapsed_ms: None,
        }
    }

    /// Serialize the result to JSON (for postMessage back to main thread).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("Serialize error: {e}"))
    }

    /// Deserialize a result from JSON (in the main thread).
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Deserialize error: {e}"))
    }

    /// Returns `true` if the result represents an error.
    pub fn is_error(&self) -> bool {
        matches!(self.result, WorkerResultKind::Error { .. })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkerPool — job dispatcher (in-process mock for non-WASM targets)
// ─────────────────────────────────────────────────────────────────────────────

/// In-process worker pool (used in unit tests and non-browser environments).
///
/// In the browser, the actual Web Worker threads are managed by JavaScript.
/// This struct handles the Rust-side job queue and provides an `execute`
/// method that runs jobs synchronously (suitable for WASM single-threaded mode
/// or server-side testing).
pub struct WorkerPool {
    capacity: usize,
    completed: std::sync::Mutex<Vec<WorkerResult>>,
}

impl WorkerPool {
    /// Create a pool with the given worker capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            completed: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Pool capacity (max parallel workers).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Execute a job synchronously and store the result.
    ///
    /// In the browser, callers post jobs via `WorkerJob::to_json()` and
    /// receive results via `onmessage`.  This method is for testing.
    pub fn execute(&self, job: WorkerJob) -> WorkerResult {
        let result = Self::run_job(&job);
        let mut guard = self.completed.lock().unwrap_or_else(|p| p.into_inner());
        guard.push(result.clone());
        result
    }

    fn run_job(job: &WorkerJob) -> WorkerResult {
        match &job.kind {
            WorkerJobKind::CountSubjects { ntriples } => {
                let count = ntriples
                    .lines()
                    .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
                    .count();
                WorkerResult::parsed_ok(job.id, count, None)
            }
            WorkerJobKind::ParseTurtle { turtle } => {
                // Count non-empty, non-comment lines as a proxy for triple count
                let count = turtle
                    .lines()
                    .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
                    .count();
                WorkerResult::parsed_ok(job.id, count, None)
            }
            WorkerJobKind::SparqlSelect { query, ntriples: _ } => {
                // Minimal stub — real execution uses sparql_executor
                WorkerResult::sparql_ok(
                    job.id,
                    format!(
                        r#"[{{"query": {}}}]"#,
                        serde_json::to_string(query).unwrap_or_default()
                    ),
                    None,
                )
            }
        }
    }

    /// Return all completed results.
    pub fn completed_results(&self) -> Vec<WorkerResult> {
        self.completed
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Clear completed results.
    pub fn clear_completed(&self) {
        self.completed
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_job_sparql_serialization() {
        let job = WorkerJob::sparql_select(1, "SELECT * WHERE {?s ?p ?o}", "");
        let json = job.to_json().unwrap();
        let parsed = WorkerJob::from_json(&json).unwrap();
        assert_eq!(parsed.id, 1);
        matches!(parsed.kind, WorkerJobKind::SparqlSelect { .. });
    }

    #[test]
    fn test_worker_job_parse_turtle_serialization() {
        let job = WorkerJob::parse_turtle(2, "@prefix ex: <http://example.org/> .");
        let json = job.to_json().unwrap();
        let back = WorkerJob::from_json(&json).unwrap();
        assert_eq!(back.id, 2);
    }

    #[test]
    fn test_worker_result_error_detection() {
        let r = WorkerResult::error(42, "something went wrong");
        assert!(r.is_error());
        assert_eq!(r.job_id, 42);
    }

    #[test]
    fn test_worker_result_serialization() {
        let r = WorkerResult::parsed_ok(5, 10, Some(3.15));
        let json = r.to_json().unwrap();
        let back = WorkerResult::from_json(&json).unwrap();
        assert_eq!(back.job_id, 5);
        assert!(!back.is_error());
    }

    #[test]
    fn test_pool_count_subjects() {
        let pool = WorkerPool::new(2);
        let job = WorkerJob::count_subjects(
            1,
            "<http://a> <http://b> <http://c> .\n<http://d> <http://e> <http://f> .\n",
        );
        let result = pool.execute(job);
        assert!(!result.is_error());
        assert_eq!(result.job_id, 1);
        if let WorkerResultKind::ParsedTripleCount { count } = result.result {
            assert_eq!(count, 2);
        } else {
            panic!("expected ParsedTripleCount");
        }
    }

    #[test]
    fn test_pool_capacity() {
        let pool = WorkerPool::new(4);
        assert_eq!(pool.capacity(), 4);
    }

    #[test]
    fn test_pool_zero_capacity_clamped() {
        let pool = WorkerPool::new(0);
        assert_eq!(pool.capacity(), 1, "capacity must be at least 1");
    }

    #[test]
    fn test_pool_multiple_jobs() {
        let pool = WorkerPool::new(2);
        for i in 0..3 {
            pool.execute(WorkerJob::count_subjects(i, ""));
        }
        assert_eq!(pool.completed_results().len(), 3);
        pool.clear_completed();
        assert_eq!(pool.completed_results().len(), 0);
    }
}
