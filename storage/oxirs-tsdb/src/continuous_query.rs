//! Continuous aggregate queries for time-series data.
//!
//! Defines `ContinuousQuery` materialized views that automatically refresh
//! at configurable intervals, supporting incremental computation over new data.
//! The `ContinuousQueryEngine` manages registration, execution, caching, and
//! status monitoring of all active continuous queries.

use std::collections::HashMap;

// ── Aggregation function ────────────────────────────────────────────────────

/// Supported aggregation functions for continuous queries.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateFunction {
    /// Arithmetic mean.
    Avg,
    /// Summation.
    Sum,
    /// Count of data points.
    Count,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Approximate percentile (0.0 ..= 1.0).
    Percentile(f64),
}

impl AggregateFunction {
    /// Apply the aggregation function to a slice of values.
    ///
    /// Returns `None` if the slice is empty (except for `Count`, which returns 0).
    pub fn apply(&self, values: &[f64]) -> Option<f64> {
        match self {
            AggregateFunction::Count => Some(values.len() as f64),
            _ if values.is_empty() => None,
            AggregateFunction::Avg => {
                let sum: f64 = values.iter().sum();
                Some(sum / values.len() as f64)
            }
            AggregateFunction::Sum => Some(values.iter().sum()),
            AggregateFunction::Min => values.iter().cloned().reduce(f64::min),
            AggregateFunction::Max => values.iter().cloned().reduce(f64::max),
            AggregateFunction::Percentile(p) => percentile_sorted(values, *p),
        }
    }
}

/// Compute the `p`-th percentile of `values` using linear interpolation.
///
/// Expects `p` in [0.0, 1.0].  The values are sorted internally.
fn percentile_sorted(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = p.clamp(0.0, 1.0);
    let idx = p * (sorted.len() as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return Some(sorted[lo]);
    }
    let frac = idx - lo as f64;
    Some(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
}

// ── Time bucket ─────────────────────────────────────────────────────────────

/// Duration of a time bucket for aggregation alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeBucket {
    /// Bucket in seconds.
    Seconds(u64),
    /// Bucket in minutes.
    Minutes(u64),
    /// Bucket in hours.
    Hours(u64),
}

impl TimeBucket {
    /// Return the bucket size in milliseconds.
    pub fn as_millis(&self) -> u64 {
        match self {
            TimeBucket::Seconds(s) => s * 1_000,
            TimeBucket::Minutes(m) => m * 60_000,
            TimeBucket::Hours(h) => h * 3_600_000,
        }
    }

    /// Align a timestamp (ms) to the start of its bucket.
    pub fn align(&self, ts: u64) -> u64 {
        let bucket_ms = self.as_millis();
        if bucket_ms == 0 {
            return ts;
        }
        (ts / bucket_ms) * bucket_ms
    }
}

// ── Refresh policy ──────────────────────────────────────────────────────────

/// How and when a continuous query is refreshed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefreshPolicy {
    /// Refresh at a fixed interval (milliseconds).
    Interval(u64),
    /// Refresh on every new data ingestion.
    OnInsert,
    /// Manual refresh only.
    Manual,
}

// ── Query status ────────────────────────────────────────────────────────────

/// Runtime status of a continuous query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStatus {
    /// The query is registered but has never been executed.
    Idle,
    /// The query is actively being computed.
    Running,
    /// The query has computed results available.
    Ready,
    /// The query was disabled by the user.
    Disabled,
    /// The query encountered an error on last refresh.
    Error(String),
}

// ── ContinuousQuery definition ──────────────────────────────────────────────

/// Definition of a continuous aggregate query.
#[derive(Debug, Clone)]
pub struct ContinuousQuery {
    /// Unique query identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// The series (metric) this query aggregates.
    pub series_id: String,
    /// Aggregation function to apply.
    pub function: AggregateFunction,
    /// Time bucket for alignment.
    pub bucket: TimeBucket,
    /// How the query is refreshed.
    pub refresh_policy: RefreshPolicy,
    /// Whether to compute incrementally (only new data since last refresh).
    pub incremental: bool,
}

impl ContinuousQuery {
    /// Create a new continuous query.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        series_id: impl Into<String>,
        function: AggregateFunction,
        bucket: TimeBucket,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            series_id: series_id.into(),
            function,
            bucket,
            refresh_policy: RefreshPolicy::Interval(60_000),
            incremental: true,
        }
    }

    /// Set the refresh policy.
    pub fn with_refresh(mut self, policy: RefreshPolicy) -> Self {
        self.refresh_policy = policy;
        self
    }

    /// Set incremental mode.
    pub fn with_incremental(mut self, incremental: bool) -> Self {
        self.incremental = incremental;
        self
    }
}

// ── Aggregate result ────────────────────────────────────────────────────────

/// A single bucket result from a continuous query.
#[derive(Debug, Clone)]
pub struct BucketResult {
    /// Aligned start timestamp (ms) of this bucket.
    pub bucket_start: u64,
    /// Aggregated value.
    pub value: f64,
    /// Number of data points that contributed.
    pub count: usize,
}

/// Cached results for a continuous query.
#[derive(Debug, Clone)]
pub struct QueryResultCache {
    /// Query id these results belong to.
    pub query_id: String,
    /// Ordered list of bucket results.
    pub buckets: Vec<BucketResult>,
    /// Timestamp (ms) when these results were last computed.
    pub computed_at: u64,
    /// The highest data timestamp that was included in this computation.
    pub watermark: u64,
}

// ── Query state (internal) ──────────────────────────────────────────────────

/// Internal state per registered query.
struct QueryState {
    query: ContinuousQuery,
    status: QueryStatus,
    cache: Option<QueryResultCache>,
    last_refresh: u64,
    refresh_count: u64,
    error_count: u64,
}

// ── ContinuousQueryEngine ───────────────────────────────────────────────────

/// Manages continuous queries: registration, execution, caching, and monitoring.
pub struct ContinuousQueryEngine {
    queries: HashMap<String, QueryState>,
    /// In-memory time-series data store: series_id -> Vec<(timestamp_ms, value)>.
    data: HashMap<String, Vec<(u64, f64)>>,
}

impl Default for ContinuousQueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousQueryEngine {
    /// Create a new engine with no queries or data.
    pub fn new() -> Self {
        Self {
            queries: HashMap::new(),
            data: HashMap::new(),
        }
    }

    // ── data ingestion ──────────────────────────────────────────────────────

    /// Insert a data point into the time-series store.
    pub fn insert_data(&mut self, series_id: &str, timestamp_ms: u64, value: f64) {
        self.data
            .entry(series_id.to_string())
            .or_default()
            .push((timestamp_ms, value));
    }

    /// Return the number of data points stored for a series.
    pub fn data_count(&self, series_id: &str) -> usize {
        self.data.get(series_id).map_or(0, |v| v.len())
    }

    // ── query registration ──────────────────────────────────────────────────

    /// Register a continuous query.  Returns `false` if a query with the same
    /// id already exists.
    pub fn register(&mut self, query: ContinuousQuery) -> bool {
        if self.queries.contains_key(&query.id) {
            return false;
        }
        let id = query.id.clone();
        self.queries.insert(
            id,
            QueryState {
                query,
                status: QueryStatus::Idle,
                cache: None,
                last_refresh: 0,
                refresh_count: 0,
                error_count: 0,
            },
        );
        true
    }

    /// Unregister a continuous query by id.  Returns `true` if found.
    pub fn unregister(&mut self, query_id: &str) -> bool {
        self.queries.remove(query_id).is_some()
    }

    /// Return the number of registered queries.
    pub fn query_count(&self) -> usize {
        self.queries.len()
    }

    /// Return the query definition for a given id, if registered.
    pub fn get_query(&self, query_id: &str) -> Option<&ContinuousQuery> {
        self.queries.get(query_id).map(|s| &s.query)
    }

    /// Return the current status of a query.
    pub fn query_status(&self, query_id: &str) -> Option<&QueryStatus> {
        self.queries.get(query_id).map(|s| &s.status)
    }

    /// Disable a query (prevents further refreshes).
    pub fn disable(&mut self, query_id: &str) -> bool {
        if let Some(state) = self.queries.get_mut(query_id) {
            state.status = QueryStatus::Disabled;
            true
        } else {
            false
        }
    }

    /// Re-enable a disabled query.
    pub fn enable(&mut self, query_id: &str) -> bool {
        if let Some(state) = self.queries.get_mut(query_id) {
            if state.status == QueryStatus::Disabled {
                state.status = QueryStatus::Idle;
                return true;
            }
        }
        false
    }

    // ── refresh / execution ─────────────────────────────────────────────────

    /// Refresh a specific query, recomputing its results.
    ///
    /// If the query is incremental, only data after the current watermark is
    /// processed and merged with the existing cache.  Returns the number of
    /// buckets produced.
    pub fn refresh(&mut self, query_id: &str, now: u64) -> Option<usize> {
        let state = self.queries.get_mut(query_id)?;
        if state.status == QueryStatus::Disabled {
            return None;
        }
        state.status = QueryStatus::Running;

        let series_data = self.data.get(&state.query.series_id);
        let data_slice = series_data.map_or(&[] as &[(u64, f64)], |v| v.as_slice());

        let watermark = state
            .cache
            .as_ref()
            .filter(|_| state.query.incremental)
            .map_or(0, |c| c.watermark);

        // Filter to only new data (> watermark) if incremental.
        let new_data: Vec<(u64, f64)> = data_slice
            .iter()
            .filter(|(ts, _)| *ts > watermark)
            .cloned()
            .collect();

        // Group by time bucket.
        let bucket_ms = state.query.bucket.as_millis();
        let mut bucket_map: HashMap<u64, Vec<f64>> = HashMap::new();
        for (ts, val) in &new_data {
            let aligned = ts.checked_div(bucket_ms).unwrap_or(0) * bucket_ms;
            bucket_map.entry(aligned).or_default().push(*val);
        }

        // Compute aggregates.
        let mut new_buckets: Vec<BucketResult> = bucket_map
            .into_iter()
            .filter_map(|(bucket_start, values)| {
                state
                    .query
                    .function
                    .apply(&values)
                    .map(|value| BucketResult {
                        bucket_start,
                        value,
                        count: values.len(),
                    })
            })
            .collect();
        new_buckets.sort_by_key(|b| b.bucket_start);

        // Merge with existing cache if incremental.
        let final_buckets = if state.query.incremental {
            if let Some(ref existing) = state.cache {
                let mut merged: HashMap<u64, BucketResult> = HashMap::new();
                for b in &existing.buckets {
                    merged.insert(b.bucket_start, b.clone());
                }
                for b in new_buckets {
                    // For incremental merge, update existing buckets.
                    let entry = merged.entry(b.bucket_start).or_insert(BucketResult {
                        bucket_start: b.bucket_start,
                        value: 0.0,
                        count: 0,
                    });
                    // Simplified merge: recalculate using combined counts.
                    let total_count = entry.count + b.count;
                    if total_count > 0 {
                        match state.query.function {
                            AggregateFunction::Sum | AggregateFunction::Count => {
                                entry.value += b.value;
                            }
                            AggregateFunction::Min => {
                                if b.value < entry.value || entry.count == 0 {
                                    entry.value = b.value;
                                }
                            }
                            AggregateFunction::Max => {
                                if b.value > entry.value || entry.count == 0 {
                                    entry.value = b.value;
                                }
                            }
                            AggregateFunction::Avg => {
                                // Weighted average
                                let old_sum = entry.value * entry.count as f64;
                                let new_sum = b.value * b.count as f64;
                                entry.value = (old_sum + new_sum) / total_count as f64;
                            }
                            AggregateFunction::Percentile(_) => {
                                // Percentile cannot be merged incrementally; take new.
                                entry.value = b.value;
                            }
                        }
                    }
                    entry.count = total_count;
                }
                let mut result: Vec<BucketResult> = merged.into_values().collect();
                result.sort_by_key(|b| b.bucket_start);
                result
            } else {
                new_buckets
            }
        } else {
            new_buckets
        };

        let bucket_count = final_buckets.len();
        let new_watermark = data_slice
            .iter()
            .map(|(ts, _)| *ts)
            .max()
            .unwrap_or(watermark);

        state.cache = Some(QueryResultCache {
            query_id: query_id.to_string(),
            buckets: final_buckets,
            computed_at: now,
            watermark: new_watermark,
        });
        state.status = QueryStatus::Ready;
        state.last_refresh = now;
        state.refresh_count += 1;

        Some(bucket_count)
    }

    /// Refresh all queries whose interval has elapsed.
    pub fn refresh_due(&mut self, now: u64) -> Vec<String> {
        let due_ids: Vec<String> = self
            .queries
            .iter()
            .filter(|(_, state)| {
                if state.status == QueryStatus::Disabled {
                    return false;
                }
                match state.query.refresh_policy {
                    RefreshPolicy::Interval(interval_ms) => {
                        now.saturating_sub(state.last_refresh) >= interval_ms
                    }
                    _ => false,
                }
            })
            .map(|(id, _)| id.clone())
            .collect();

        let mut refreshed = Vec::new();
        for id in due_ids {
            if self.refresh(&id, now).is_some() {
                refreshed.push(id);
            }
        }
        refreshed
    }

    /// Invalidate the cache for a specific query.
    pub fn invalidate(&mut self, query_id: &str) -> bool {
        if let Some(state) = self.queries.get_mut(query_id) {
            state.cache = None;
            if state.status == QueryStatus::Ready {
                state.status = QueryStatus::Idle;
            }
            true
        } else {
            false
        }
    }

    // ── result access ───────────────────────────────────────────────────────

    /// Return the cached results for a query, if available.
    pub fn cached_results(&self, query_id: &str) -> Option<&QueryResultCache> {
        self.queries.get(query_id).and_then(|s| s.cache.as_ref())
    }

    /// Return the watermark for a query.
    pub fn watermark(&self, query_id: &str) -> Option<u64> {
        self.queries
            .get(query_id)
            .and_then(|s| s.cache.as_ref())
            .map(|c| c.watermark)
    }

    // ── monitoring ──────────────────────────────────────────────────────────

    /// Return the refresh count for a query.
    pub fn refresh_count(&self, query_id: &str) -> Option<u64> {
        self.queries.get(query_id).map(|s| s.refresh_count)
    }

    /// Return the last refresh timestamp for a query.
    pub fn last_refresh(&self, query_id: &str) -> Option<u64> {
        self.queries.get(query_id).map(|s| s.last_refresh)
    }

    /// Return the error count for a query.
    pub fn error_count(&self, query_id: &str) -> Option<u64> {
        self.queries.get(query_id).map(|s| s.error_count)
    }

    /// Record an error for a query.
    pub fn record_error(&mut self, query_id: &str, message: impl Into<String>) -> bool {
        if let Some(state) = self.queries.get_mut(query_id) {
            state.error_count += 1;
            state.status = QueryStatus::Error(message.into());
            true
        } else {
            false
        }
    }

    /// List all registered query ids.
    pub fn query_ids(&self) -> Vec<String> {
        self.queries.keys().cloned().collect()
    }

    /// Return a summary of all query statuses.
    pub fn status_summary(&self) -> HashMap<String, QueryStatus> {
        self.queries
            .iter()
            .map(|(id, s)| (id.clone(), s.status.clone()))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_query(id: &str, series: &str) -> ContinuousQuery {
        ContinuousQuery::new(
            id,
            id,
            series,
            AggregateFunction::Avg,
            TimeBucket::Seconds(60),
        )
    }

    fn make_engine_with_data() -> ContinuousQueryEngine {
        let mut engine = ContinuousQueryEngine::new();
        // Insert 10 data points for "temp" every 10 seconds starting at 0.
        for i in 0..10 {
            engine.insert_data("temp", i * 10_000, 20.0 + i as f64);
        }
        engine
    }

    // ── AggregateFunction::apply ────────────────────────────────────────────

    #[test]
    fn test_avg_function() {
        let f = AggregateFunction::Avg;
        let result = f.apply(&[10.0, 20.0, 30.0]);
        assert!((result.expect("has value") - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sum_function() {
        let f = AggregateFunction::Sum;
        let result = f.apply(&[1.0, 2.0, 3.0]);
        assert!((result.expect("has value") - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_count_function() {
        let f = AggregateFunction::Count;
        let result = f.apply(&[1.0, 2.0, 3.0]);
        assert!((result.expect("has value") - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_count_empty() {
        let f = AggregateFunction::Count;
        let result = f.apply(&[]);
        assert!((result.expect("count is 0") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_min_function() {
        let f = AggregateFunction::Min;
        let result = f.apply(&[30.0, 10.0, 20.0]);
        assert!((result.expect("has value") - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_function() {
        let f = AggregateFunction::Max;
        let result = f.apply(&[10.0, 30.0, 20.0]);
        assert!((result.expect("has value") - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_median() {
        let f = AggregateFunction::Percentile(0.5);
        let result = f.apply(&[10.0, 20.0, 30.0]);
        assert!((result.expect("has value") - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_p0() {
        let f = AggregateFunction::Percentile(0.0);
        let result = f.apply(&[10.0, 20.0, 30.0]);
        assert!((result.expect("has value") - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_p100() {
        let f = AggregateFunction::Percentile(1.0);
        let result = f.apply(&[10.0, 20.0, 30.0]);
        assert!((result.expect("has value") - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_empty_returns_none() {
        let f = AggregateFunction::Avg;
        assert!(f.apply(&[]).is_none());
    }

    #[test]
    fn test_min_empty_returns_none() {
        let f = AggregateFunction::Min;
        assert!(f.apply(&[]).is_none());
    }

    // ── TimeBucket ──────────────────────────────────────────────────────────

    #[test]
    fn test_time_bucket_seconds_millis() {
        let b = TimeBucket::Seconds(30);
        assert_eq!(b.as_millis(), 30_000);
    }

    #[test]
    fn test_time_bucket_minutes_millis() {
        let b = TimeBucket::Minutes(5);
        assert_eq!(b.as_millis(), 300_000);
    }

    #[test]
    fn test_time_bucket_hours_millis() {
        let b = TimeBucket::Hours(1);
        assert_eq!(b.as_millis(), 3_600_000);
    }

    #[test]
    fn test_time_bucket_align() {
        let b = TimeBucket::Seconds(60);
        assert_eq!(b.align(75_000), 60_000);
    }

    #[test]
    fn test_time_bucket_align_exact() {
        let b = TimeBucket::Seconds(60);
        assert_eq!(b.align(60_000), 60_000);
    }

    // ── ContinuousQuery construction ────────────────────────────────────────

    #[test]
    fn test_query_new() {
        let q = make_query("q1", "temp");
        assert_eq!(q.id, "q1");
        assert_eq!(q.series_id, "temp");
        assert!(q.incremental);
    }

    #[test]
    fn test_query_with_refresh() {
        let q = make_query("q1", "temp").with_refresh(RefreshPolicy::Manual);
        assert_eq!(q.refresh_policy, RefreshPolicy::Manual);
    }

    #[test]
    fn test_query_with_incremental_false() {
        let q = make_query("q1", "temp").with_incremental(false);
        assert!(!q.incremental);
    }

    // ── Engine basics ───────────────────────────────────────────────────────

    #[test]
    fn test_engine_new_empty() {
        let engine = ContinuousQueryEngine::new();
        assert_eq!(engine.query_count(), 0);
    }

    #[test]
    fn test_engine_default() {
        let engine = ContinuousQueryEngine::default();
        assert_eq!(engine.query_count(), 0);
    }

    #[test]
    fn test_register_query() {
        let mut engine = ContinuousQueryEngine::new();
        assert!(engine.register(make_query("q1", "temp")));
        assert_eq!(engine.query_count(), 1);
    }

    #[test]
    fn test_register_duplicate_returns_false() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "temp"));
        assert!(!engine.register(make_query("q1", "temp")));
    }

    #[test]
    fn test_unregister_query() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "temp"));
        assert!(engine.unregister("q1"));
        assert_eq!(engine.query_count(), 0);
    }

    #[test]
    fn test_unregister_unknown_returns_false() {
        let mut engine = ContinuousQueryEngine::new();
        assert!(!engine.unregister("ghost"));
    }

    #[test]
    fn test_get_query() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "temp"));
        let q = engine.get_query("q1");
        assert!(q.is_some());
        assert_eq!(q.expect("query exists").series_id, "temp");
    }

    #[test]
    fn test_get_query_unknown_returns_none() {
        let engine = ContinuousQueryEngine::new();
        assert!(engine.get_query("ghost").is_none());
    }

    // ── data ingestion ──────────────────────────────────────────────────────

    #[test]
    fn test_insert_data() {
        let mut engine = ContinuousQueryEngine::new();
        engine.insert_data("temp", 1000, 22.5);
        assert_eq!(engine.data_count("temp"), 1);
    }

    #[test]
    fn test_data_count_empty() {
        let engine = ContinuousQueryEngine::new();
        assert_eq!(engine.data_count("missing"), 0);
    }

    // ── refresh / execution ─────────────────────────────────────────────────

    #[test]
    fn test_refresh_produces_buckets() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp"));
        let count = engine.refresh("q1", 100_000);
        assert!(count.is_some());
        assert!(count.expect("bucket count") > 0);
    }

    #[test]
    fn test_refresh_sets_status_ready() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp"));
        engine.refresh("q1", 100_000);
        assert_eq!(engine.query_status("q1"), Some(&QueryStatus::Ready));
    }

    #[test]
    fn test_refresh_unknown_returns_none() {
        let mut engine = ContinuousQueryEngine::new();
        assert!(engine.refresh("ghost", 100).is_none());
    }

    #[test]
    fn test_refresh_disabled_returns_none() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp"));
        engine.disable("q1");
        assert!(engine.refresh("q1", 100_000).is_none());
    }

    #[test]
    fn test_refresh_increments_count() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp"));
        engine.refresh("q1", 100_000);
        engine.refresh("q1", 200_000);
        assert_eq!(engine.refresh_count("q1"), Some(2));
    }

    #[test]
    fn test_incremental_refresh_adds_new_data() {
        let mut engine = ContinuousQueryEngine::new();
        engine.insert_data("s", 1000, 10.0);
        engine.register(
            ContinuousQuery::new(
                "q1",
                "q1",
                "s",
                AggregateFunction::Sum,
                TimeBucket::Seconds(60),
            )
            .with_incremental(true),
        );
        engine.refresh("q1", 2000);
        let first_result = engine.cached_results("q1").expect("has cache").buckets[0].value;

        // Add more data.
        engine.insert_data("s", 2000, 5.0);
        engine.refresh("q1", 3000);
        let second_result = engine.cached_results("q1").expect("has cache").buckets[0].value;
        assert!(second_result > first_result);
    }

    // ── refresh_due ─────────────────────────────────────────────────────────

    #[test]
    fn test_refresh_due_triggers_interval_queries() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp").with_refresh(RefreshPolicy::Interval(60_000)));
        let refreshed = engine.refresh_due(70_000);
        assert!(refreshed.contains(&"q1".to_string()));
    }

    #[test]
    fn test_refresh_due_skips_not_yet_due() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp").with_refresh(RefreshPolicy::Interval(60_000)));
        engine.refresh("q1", 10_000);
        let refreshed = engine.refresh_due(20_000); // only 10s elapsed, need 60s
        assert!(!refreshed.contains(&"q1".to_string()));
    }

    #[test]
    fn test_refresh_due_skips_manual_queries() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp").with_refresh(RefreshPolicy::Manual));
        let refreshed = engine.refresh_due(1_000_000);
        assert!(!refreshed.contains(&"q1".to_string()));
    }

    // ── cache invalidation ──────────────────────────────────────────────────

    #[test]
    fn test_invalidate_clears_cache() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp"));
        engine.refresh("q1", 100_000);
        assert!(engine.cached_results("q1").is_some());
        engine.invalidate("q1");
        assert!(engine.cached_results("q1").is_none());
    }

    #[test]
    fn test_invalidate_resets_status_to_idle() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp"));
        engine.refresh("q1", 100_000);
        engine.invalidate("q1");
        assert_eq!(engine.query_status("q1"), Some(&QueryStatus::Idle));
    }

    #[test]
    fn test_invalidate_unknown_returns_false() {
        let mut engine = ContinuousQueryEngine::new();
        assert!(!engine.invalidate("ghost"));
    }

    // ── disable / enable ────────────────────────────────────────────────────

    #[test]
    fn test_disable_query() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "temp"));
        assert!(engine.disable("q1"));
        assert_eq!(engine.query_status("q1"), Some(&QueryStatus::Disabled));
    }

    #[test]
    fn test_enable_query() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "temp"));
        engine.disable("q1");
        assert!(engine.enable("q1"));
        assert_eq!(engine.query_status("q1"), Some(&QueryStatus::Idle));
    }

    #[test]
    fn test_enable_non_disabled_returns_false() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "temp"));
        assert!(!engine.enable("q1"));
    }

    #[test]
    fn test_disable_unknown_returns_false() {
        let mut engine = ContinuousQueryEngine::new();
        assert!(!engine.disable("ghost"));
    }

    // ── error recording ─────────────────────────────────────────────────────

    #[test]
    fn test_record_error() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "temp"));
        assert!(engine.record_error("q1", "timeout"));
        assert_eq!(engine.error_count("q1"), Some(1));
    }

    #[test]
    fn test_record_error_sets_status() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "temp"));
        engine.record_error("q1", "failed");
        assert_eq!(
            engine.query_status("q1"),
            Some(&QueryStatus::Error("failed".to_string()))
        );
    }

    #[test]
    fn test_record_error_unknown_returns_false() {
        let mut engine = ContinuousQueryEngine::new();
        assert!(!engine.record_error("ghost", "oops"));
    }

    // ── monitoring ──────────────────────────────────────────────────────────

    #[test]
    fn test_last_refresh() {
        let mut engine = make_engine_with_data();
        engine.register(make_query("q1", "temp"));
        engine.refresh("q1", 42_000);
        assert_eq!(engine.last_refresh("q1"), Some(42_000));
    }

    #[test]
    fn test_watermark_advances() {
        let mut engine = ContinuousQueryEngine::new();
        engine.insert_data("s", 1000, 10.0);
        engine.insert_data("s", 2000, 20.0);
        engine.register(make_query("q1", "s"));
        engine.refresh("q1", 5000);
        assert_eq!(engine.watermark("q1"), Some(2000));
    }

    #[test]
    fn test_query_ids() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "a"));
        engine.register(make_query("q2", "b"));
        let ids = engine.query_ids();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_status_summary() {
        let mut engine = ContinuousQueryEngine::new();
        engine.register(make_query("q1", "a"));
        engine.register(make_query("q2", "b"));
        engine.disable("q2");
        let summary = engine.status_summary();
        assert_eq!(summary.get("q1"), Some(&QueryStatus::Idle));
        assert_eq!(summary.get("q2"), Some(&QueryStatus::Disabled));
    }

    // ── aggregate functions with single value ───────────────────────────────

    #[test]
    fn test_sum_single_value() {
        let f = AggregateFunction::Sum;
        assert!((f.apply(&[42.0]).expect("val") - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_single_value() {
        let f = AggregateFunction::Avg;
        assert!((f.apply(&[42.0]).expect("val") - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_single_value() {
        let f = AggregateFunction::Max;
        assert!((f.apply(&[42.0]).expect("val") - 42.0).abs() < f64::EPSILON);
    }

    // ── percentile edge cases ───────────────────────────────────────────────

    #[test]
    fn test_percentile_empty_returns_none() {
        let f = AggregateFunction::Percentile(0.5);
        assert!(f.apply(&[]).is_none());
    }

    #[test]
    fn test_percentile_single_value() {
        let f = AggregateFunction::Percentile(0.5);
        assert!((f.apply(&[7.0]).expect("val") - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_interpolation() {
        let f = AggregateFunction::Percentile(0.25);
        // sorted: [10, 20, 30, 40], idx = 0.25*3=0.75 => 10*0.25 + 20*0.75 = 17.5
        let result = f.apply(&[10.0, 20.0, 30.0, 40.0]);
        assert!((result.expect("val") - 17.5).abs() < 0.01);
    }
}
