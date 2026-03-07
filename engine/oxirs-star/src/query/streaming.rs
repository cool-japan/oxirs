//! Streaming SPARQL-star query results.
//!
//! This module provides a lazy, iterator-based interface for consuming large
//! RDF-star query results without materialising the entire result set in memory.
//! It also offers a streaming join that probes a cached "right" side against a
//! lazy "left" stream.
//!
//! # Design goals
//!
//! - **Lazy evaluation**: rows are produced on demand.
//! - **Backpressure**: callers control consumption rate via `take_n`.
//! - **Composable**: `StreamingJoin` wraps two result sources and lazily
//!   produces joined rows.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_star::query::streaming::{StreamingStarResult, StreamingJoin};
//! use oxirs_star::query::parallel::StarBinding;
//! use std::collections::HashMap;
//!
//! // Wrap a vector as a streaming result.
//! let rows = vec![HashMap::new()];
//! let mut stream = StreamingStarResult::from_vec(rows);
//!
//! while stream.has_next() {
//!     let batch = stream.take_n(64);
//!     for row in batch {
//!         let _binding = row.unwrap();
//!     }
//! }
//! println!("consumed {} rows", stream.rows_consumed());
//! ```

use crate::StarResult;

use super::parallel::StarBinding;

// ---- StreamingStarResult ---------------------------------------------------

/// A lazy iterator over SPARQL-star binding rows.
///
/// Rows are produced by the inner iterator on demand.  Exhaustion is tracked
/// so `has_next` is O(1) after the stream is drained.
pub struct StreamingStarResult {
    inner: Box<dyn Iterator<Item = StarResult<StarBinding>> + Send>,
    /// How many rows have been consumed from this stream.
    row_count: usize,
    /// Set to `true` when the inner iterator returns `None`.
    is_exhausted: bool,
    /// Peeked item buffered by `has_next`.
    peeked: Option<StarResult<StarBinding>>,
}

impl StreamingStarResult {
    /// Wrap an arbitrary iterator as a `StreamingStarResult`.
    pub fn new(iter: impl Iterator<Item = StarResult<StarBinding>> + Send + 'static) -> Self {
        Self {
            inner: Box::new(iter),
            row_count: 0,
            is_exhausted: false,
            peeked: None,
        }
    }

    /// Create a `StreamingStarResult` backed by a pre-materialised `Vec`.
    pub fn from_vec(rows: Vec<StarBinding>) -> Self {
        Self::new(rows.into_iter().map(Ok))
    }

    /// Create an empty stream.
    pub fn empty() -> Self {
        Self::new(std::iter::empty())
    }

    /// Consume up to `n` rows from the stream and return them.
    ///
    /// Returns fewer than `n` items when the stream is exhausted.
    pub fn take_n(&mut self, n: usize) -> Vec<StarResult<StarBinding>> {
        let mut batch = Vec::with_capacity(n);
        for _ in 0..n {
            match self.next() {
                Some(item) => batch.push(item),
                None => break,
            }
        }
        batch
    }

    /// Return `true` if there is at least one more row available.
    ///
    /// This may advance the inner iterator by one step and buffer the result.
    pub fn has_next(&mut self) -> bool {
        if self.is_exhausted {
            return false;
        }
        if self.peeked.is_some() {
            return true;
        }
        // Advance the inner iterator to check.
        match self.inner.next() {
            Some(item) => {
                self.peeked = Some(item);
                true
            }
            None => {
                self.is_exhausted = true;
                false
            }
        }
    }

    /// Total number of rows that have been yielded by this stream so far.
    pub fn rows_consumed(&self) -> usize {
        self.row_count
    }

    /// Collect all remaining rows into a `Vec`, exhausting the stream.
    ///
    /// For very large result sets prefer iterating via `take_n` or the
    /// `Iterator` implementation.
    pub fn collect_all(mut self) -> StarResult<Vec<StarBinding>> {
        let mut rows = Vec::new();
        for item in &mut self {
            rows.push(item?);
        }
        Ok(rows)
    }
}

impl Iterator for StreamingStarResult {
    type Item = StarResult<StarBinding>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_exhausted {
            return None;
        }
        // Drain the peeked item first.
        let item = if let Some(peeked) = self.peeked.take() {
            Some(peeked)
        } else {
            self.inner.next()
        };

        match item {
            Some(row) => {
                self.row_count += 1;
                Some(row)
            }
            None => {
                self.is_exhausted = true;
                None
            }
        }
    }
}

// ---- StreamingJoin ---------------------------------------------------------

/// Streaming hash-probe join: the right side is fully cached (assumed to be
/// the smaller side) while the left side is consumed lazily one row at a time.
///
/// For each left row the join probes the right cache and emits all compatible
/// merged rows.  The join variables must be specified explicitly.
pub struct StreamingJoin {
    left: StreamingStarResult,
    right_cache: Vec<StarBinding>,
    join_vars: Vec<String>,
    /// The current left row being probed against the right cache.
    left_current: Option<StarBinding>,
    /// Index into `right_cache` for the current probe.
    right_pos: usize,
    /// Whether the join is fully exhausted.
    exhausted: bool,
}

impl StreamingJoin {
    /// Construct a `StreamingJoin`.
    ///
    /// - `left` – lazy stream of binding rows (larger / outer side).
    /// - `right` – pre-materialised binding rows (smaller / inner side).
    /// - `join_vars` – variable names on which the join condition is equality.
    pub fn new(left: StreamingStarResult, right: Vec<StarBinding>, join_vars: Vec<String>) -> Self {
        Self {
            left,
            right_cache: right,
            join_vars,
            left_current: None,
            right_pos: 0,
            exhausted: false,
        }
    }

    /// Attempt to advance the left stream to the next row.
    ///
    /// Returns `true` if a new left row is available, `false` if the left
    /// stream is exhausted.
    fn advance_left(&mut self) -> Option<StarResult<StarBinding>> {
        match self.left.next() {
            Some(Ok(row)) => {
                self.left_current = Some(row);
                self.right_pos = 0;
                None
            }
            Some(Err(e)) => Some(Err(e)),
            None => {
                self.exhausted = true;
                None
            }
        }
    }

    /// Check whether `left_row` and `right_row` are compatible on all join
    /// variables.
    fn compatible(left_row: &StarBinding, right_row: &StarBinding, join_vars: &[String]) -> bool {
        join_vars.iter().all(|v| {
            match (left_row.get(v.as_str()), right_row.get(v.as_str())) {
                (Some(lv), Some(rv)) => lv == rv,
                _ => true, // Variable absent in one side is compatible.
            }
        })
    }

    /// Merge two compatible binding rows.  Left bindings take precedence for
    /// variables present in both.
    fn merge(left_row: &StarBinding, right_row: &StarBinding) -> StarBinding {
        let mut merged = left_row.clone();
        for (k, v) in right_row {
            merged.entry(k.clone()).or_insert_with(|| v.clone());
        }
        merged
    }
}

impl Iterator for StreamingJoin {
    type Item = StarResult<StarBinding>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.exhausted {
                return None;
            }

            // Ensure we have a current left row.
            if self.left_current.is_none() {
                if let Some(err) = self.advance_left() {
                    return Some(err);
                }
                if self.exhausted {
                    return None;
                }
            }

            let left_row = match self.left_current.as_ref() {
                Some(r) => r,
                None => {
                    self.exhausted = true;
                    return None;
                }
            };

            // Scan the right cache from `right_pos`.
            while self.right_pos < self.right_cache.len() {
                let right_row = &self.right_cache[self.right_pos];
                self.right_pos += 1;
                if Self::compatible(left_row, right_row, &self.join_vars) {
                    return Some(Ok(Self::merge(left_row, right_row)));
                }
            }

            // Exhausted right cache for this left row – move to next left row.
            self.left_current = None;
        }
    }
}

// ---- StreamingFilter -------------------------------------------------------

/// A streaming filter that applies a predicate to binding rows.
pub struct StreamingFilter<F>
where
    F: Fn(&StarBinding) -> bool + Send,
{
    inner: StreamingStarResult,
    predicate: F,
    rows_passed: usize,
    rows_skipped: usize,
}

impl<F> StreamingFilter<F>
where
    F: Fn(&StarBinding) -> bool + Send,
{
    /// Wrap `stream` with a filter predicate.
    pub fn new(stream: StreamingStarResult, predicate: F) -> Self {
        Self {
            inner: stream,
            predicate,
            rows_passed: 0,
            rows_skipped: 0,
        }
    }

    /// Number of rows that passed the filter.
    pub fn rows_passed(&self) -> usize {
        self.rows_passed
    }

    /// Number of rows that were rejected by the filter.
    pub fn rows_skipped(&self) -> usize {
        self.rows_skipped
    }
}

impl<F> Iterator for StreamingFilter<F>
where
    F: Fn(&StarBinding) -> bool + Send,
{
    type Item = StarResult<StarBinding>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next()? {
                Ok(row) => {
                    if (self.predicate)(&row) {
                        self.rows_passed += 1;
                        return Some(Ok(row));
                    }
                    self.rows_skipped += 1;
                    // Continue scanning.
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

// ---- StreamingProject ------------------------------------------------------

/// A streaming projection that retains only the specified variables from each
/// binding row, dropping all others.
pub struct StreamingProject {
    inner: StreamingStarResult,
    variables: Vec<String>,
}

impl StreamingProject {
    /// Wrap `stream`, projecting only `variables`.
    pub fn new(stream: StreamingStarResult, variables: Vec<String>) -> Self {
        Self {
            inner: stream,
            variables,
        }
    }
}

impl Iterator for StreamingProject {
    type Item = StarResult<StarBinding>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next()? {
            Ok(row) => {
                let projected: StarBinding = self
                    .variables
                    .iter()
                    .filter_map(|v| row.get(v.as_str()).map(|t| (v.clone(), t.clone())))
                    .collect();
                Some(Ok(projected))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StarTerm;

    fn make_binding(pairs: &[(&str, &str)]) -> StarBinding {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), StarTerm::iri(v).expect("valid IRI")))
            .collect()
    }

    #[test]
    fn test_empty_stream() {
        let mut stream = StreamingStarResult::empty();
        assert!(!stream.has_next());
        assert_eq!(stream.rows_consumed(), 0);
        let batch = stream.take_n(10);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_from_vec() {
        let rows: Vec<StarBinding> = (0..5_usize)
            .map(|i| {
                let mut b = StarBinding::new();
                b.insert(
                    "x".to_owned(),
                    StarTerm::iri(&format!("http://ex.org/{i}")).expect("valid IRI"),
                );
                b
            })
            .collect();
        let mut stream = StreamingStarResult::from_vec(rows);
        assert!(stream.has_next());
        let batch = stream.take_n(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(stream.rows_consumed(), 3);

        let rest = stream.take_n(100);
        assert_eq!(rest.len(), 2);
        assert_eq!(stream.rows_consumed(), 5);
        assert!(!stream.has_next());
    }

    #[test]
    fn test_collect_all() {
        let rows: Vec<StarBinding> = (0..10_usize).map(|_| StarBinding::new()).collect();
        let stream = StreamingStarResult::from_vec(rows);
        let collected = stream.collect_all().expect("no error");
        assert_eq!(collected.len(), 10);
    }

    #[test]
    fn test_streaming_join_basic() {
        let left_rows = vec![
            make_binding(&[("x", "http://ex.org/alice"), ("y", "http://ex.org/bob")]),
            make_binding(&[("x", "http://ex.org/charlie"), ("y", "http://ex.org/dave")]),
        ];
        let right_rows = vec![
            make_binding(&[("y", "http://ex.org/bob"), ("z", "http://ex.org/lit1")]),
            make_binding(&[("y", "http://ex.org/eve"), ("z", "http://ex.org/lit2")]),
        ];

        let left_stream = StreamingStarResult::from_vec(left_rows);
        let join = StreamingJoin::new(left_stream, right_rows, vec!["y".to_owned()]);

        // Only alice→bob should join with bob→lit1.
        let mut count = 0;
        for item in join {
            let row = item.expect("no error");
            assert_eq!(
                row.get("x").expect("x"),
                &StarTerm::iri("http://ex.org/alice").expect("valid IRI")
            );
            assert_eq!(
                row.get("z").expect("z"),
                &StarTerm::iri("http://ex.org/lit1").expect("valid IRI")
            );
            count += 1;
        }
        assert_eq!(count, 1);
    }

    #[test]
    fn test_streaming_join_empty_right() {
        let left_rows = vec![make_binding(&[("x", "http://ex.org/a")])];
        let left_stream = StreamingStarResult::from_vec(left_rows);
        let mut join = StreamingJoin::new(left_stream, vec![], vec!["x".to_owned()]);
        assert!(join.next().is_none());
    }

    #[test]
    fn test_streaming_filter() {
        let rows: Vec<StarBinding> = (0..10_usize)
            .map(|i| {
                let mut b = StarBinding::new();
                b.insert(
                    "i".to_owned(),
                    StarTerm::iri(&format!("http://ex.org/{i}")).expect("valid IRI"),
                );
                b
            })
            .collect();
        let stream = StreamingStarResult::from_vec(rows);
        // Keep only bindings where "i" is bound (all of them).
        let mut filter = StreamingFilter::new(stream, |row| row.contains_key("i"));
        let all: Vec<_> = (&mut filter).collect();
        assert_eq!(all.len(), 10);
        assert_eq!(filter.rows_passed(), 10);
        assert_eq!(filter.rows_skipped(), 0);
    }

    #[test]
    fn test_streaming_filter_with_exclusions() {
        let target = StarTerm::iri("http://ex.org/keep").expect("valid IRI");
        let keep_binding = {
            let mut b = StarBinding::new();
            b.insert("x".to_owned(), target.clone());
            b
        };
        let skip_binding = {
            let mut b = StarBinding::new();
            b.insert(
                "x".to_owned(),
                StarTerm::iri("http://ex.org/skip").expect("valid IRI"),
            );
            b
        };
        let rows = vec![keep_binding, skip_binding];
        let stream = StreamingStarResult::from_vec(rows);
        let mut filter = StreamingFilter::new(stream, move |row| {
            row.get("x").map(|t| t == &target).unwrap_or(false)
        });
        let results: Vec<_> = (&mut filter).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(filter.rows_passed(), 1);
        assert_eq!(filter.rows_skipped(), 1);
    }

    #[test]
    fn test_streaming_project() {
        let mut b = StarBinding::new();
        b.insert(
            "x".to_owned(),
            StarTerm::iri("http://ex.org/x").expect("valid IRI"),
        );
        b.insert(
            "y".to_owned(),
            StarTerm::iri("http://ex.org/y").expect("valid IRI"),
        );
        let rows = vec![b];
        let stream = StreamingStarResult::from_vec(rows);
        let mut proj = StreamingProject::new(stream, vec!["x".to_owned()]);
        let row = proj.next().expect("item").expect("no error");
        assert!(row.contains_key("x"), "x should be in projected row");
        assert!(!row.contains_key("y"), "y should be projected away");
    }
}
