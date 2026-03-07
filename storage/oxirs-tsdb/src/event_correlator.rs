//! Time-series event correlation engine (v1.1.0 round 14).
//!
//! Evaluates user-defined `CorrelationRule` definitions against a sliding
//! in-memory window of ingested `Event` values.  When the absolute value of
//! the computed Pearson correlation coefficient between two time-series
//! exceeds a configurable threshold, the result is marked as "triggered".
//!
//! # Design
//!
//! - All Pearson correlation is computed **inline** without any external
//!   numeric crate dependency.
//! - Events are stored in a single `Vec` sorted by insertion order; the
//!   `purge_old_events` method removes
//!   events older than a given timestamp.

use std::collections::HashMap;

/// A single time-series observation.
#[derive(Debug, Clone)]
pub struct Event {
    /// Unix-epoch-style timestamp in milliseconds.
    pub timestamp: u64,
    /// Logical time-series name (e.g. `"temperature"`, `"pressure"`).
    pub series: String,
    /// Measured value.
    pub value: f64,
    /// Arbitrary key-value labels.
    pub tags: HashMap<String, String>,
}

impl Event {
    /// Create a new event without any tags.
    pub fn new(timestamp: u64, series: impl Into<String>, value: f64) -> Self {
        Self {
            timestamp,
            series: series.into(),
            value,
            tags: HashMap::new(),
        }
    }

    /// Create a new event with tags.
    pub fn with_tags(
        timestamp: u64,
        series: impl Into<String>,
        value: f64,
        tags: HashMap<String, String>,
    ) -> Self {
        Self {
            timestamp,
            series: series.into(),
            value,
            tags,
        }
    }
}

/// Definition of a pairwise correlation rule.
///
/// The rule is evaluated by collecting all events for `series_a` and
/// `series_b` whose timestamp falls within `[latest_event - window_ms,
/// latest_event]`.  If at least two matched pairs exist the Pearson
/// correlation coefficient is computed and compared against `min_correlation`.
#[derive(Debug, Clone)]
pub struct CorrelationRule {
    /// Human-readable rule name.
    pub name: String,
    /// First time-series in the correlation pair.
    pub series_a: String,
    /// Second time-series in the correlation pair.
    pub series_b: String,
    /// Look-back window in milliseconds.
    pub window_ms: u64,
    /// Minimum absolute value of the correlation coefficient needed to trigger.
    pub min_correlation: f64,
}

impl CorrelationRule {
    /// Construct a new correlation rule.
    pub fn new(
        name: impl Into<String>,
        series_a: impl Into<String>,
        series_b: impl Into<String>,
        window_ms: u64,
        min_correlation: f64,
    ) -> Self {
        Self {
            name: name.into(),
            series_a: series_a.into(),
            series_b: series_b.into(),
            window_ms,
            min_correlation,
        }
    }
}

/// The outcome of evaluating a single [`CorrelationRule`].
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationResult {
    /// Name of the rule that was evaluated.
    pub rule_name: String,
    /// Computed Pearson correlation coefficient, or `0.0` if there were
    /// insufficient data points.
    pub coefficient: f64,
    /// Number of (a, b) sample pairs that contributed to the calculation.
    pub sample_count: usize,
    /// `true` when `|coefficient| >= rule.min_correlation`.
    pub triggered: bool,
}

/// Time-series event correlation engine.
///
/// Ingests events from multiple series and evaluates registered
/// [`CorrelationRule`] definitions on demand.
#[derive(Debug, Default)]
pub struct EventCorrelator {
    events: Vec<Event>,
    rules: Vec<CorrelationRule>,
}

impl EventCorrelator {
    /// Create a new, empty correlator.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            rules: Vec::new(),
        }
    }

    /// Register a correlation rule.
    pub fn add_rule(&mut self, rule: CorrelationRule) {
        self.rules.push(rule);
    }

    /// Ingest a single event.
    pub fn ingest(&mut self, event: Event) {
        self.events.push(event);
    }

    /// Evaluate all registered rules against the current event buffer.
    ///
    /// For each rule the method:
    /// 1. Determines the most recent event timestamp across both series.
    /// 2. Filters events whose timestamp falls within `[max_ts - window_ms,
    ///    max_ts]`.
    /// 3. Pairs up values by timestamp; uses exact timestamp equality.
    /// 4. Computes the Pearson correlation coefficient over all matched pairs.
    /// 5. Sets `triggered = true` when `|r| >= rule.min_correlation`.
    pub fn evaluate_rules(&self) -> Vec<CorrelationResult> {
        self.rules
            .iter()
            .map(|rule| self.evaluate_rule(rule))
            .collect()
    }

    /// Remove all events with `timestamp < before_timestamp`.
    ///
    /// Returns the number of events removed.
    pub fn purge_old_events(&mut self, before_timestamp: u64) -> usize {
        let before = self.events.len();
        self.events.retain(|e| e.timestamp >= before_timestamp);
        before - self.events.len()
    }

    /// Total number of events currently held in the buffer.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn evaluate_rule(&self, rule: &CorrelationRule) -> CorrelationResult {
        // Collect values for each series separately
        let values_a: Vec<(u64, f64)> = self
            .events
            .iter()
            .filter(|e| e.series == rule.series_a)
            .map(|e| (e.timestamp, e.value))
            .collect();

        let values_b: Vec<(u64, f64)> = self
            .events
            .iter()
            .filter(|e| e.series == rule.series_b)
            .map(|e| (e.timestamp, e.value))
            .collect();

        if values_a.is_empty() || values_b.is_empty() {
            return CorrelationResult {
                rule_name: rule.name.clone(),
                coefficient: 0.0,
                sample_count: 0,
                triggered: false,
            };
        }

        // Determine the time window
        let max_ts_a = values_a.iter().map(|(t, _)| *t).max().unwrap_or(0);
        let max_ts_b = values_b.iter().map(|(t, _)| *t).max().unwrap_or(0);
        let max_ts = max_ts_a.max(max_ts_b);
        let min_ts = max_ts.saturating_sub(rule.window_ms);

        // Filter to the window
        let window_a: HashMap<u64, f64> = values_a
            .into_iter()
            .filter(|(t, _)| *t >= min_ts && *t <= max_ts)
            .collect();

        let window_b: HashMap<u64, f64> = values_b
            .into_iter()
            .filter(|(t, _)| *t >= min_ts && *t <= max_ts)
            .collect();

        // Find common timestamps
        let mut pairs: Vec<(f64, f64)> = window_a
            .iter()
            .filter_map(|(ts, va)| window_b.get(ts).map(|vb| (*va, *vb)))
            .collect();

        // Sort for determinism (not strictly needed for Pearson but good practice)
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n = pairs.len();
        if n < 2 {
            return CorrelationResult {
                rule_name: rule.name.clone(),
                coefficient: 0.0,
                sample_count: n,
                triggered: false,
            };
        }

        let coefficient = pearson_correlation(&pairs);
        let triggered = coefficient.abs() >= rule.min_correlation;

        CorrelationResult {
            rule_name: rule.name.clone(),
            coefficient,
            sample_count: n,
            triggered,
        }
    }
}

// ---------------------------------------------------------------------------
// Pearson correlation (inline, no external crate)
// ---------------------------------------------------------------------------

/// Compute the Pearson correlation coefficient for a slice of (x, y) pairs.
///
/// Returns `0.0` when the standard deviation of either variable is zero
/// (constant series), or when fewer than two pairs are provided.
fn pearson_correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;

    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < f64::EPSILON {
        return 0.0;
    }

    // Clamp to [-1, 1] to handle floating-point rounding near the extremes
    (cov / denom).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(name: &str, sa: &str, sb: &str, window_ms: u64, min_corr: f64) -> CorrelationRule {
        CorrelationRule::new(name, sa, sb, window_ms, min_corr)
    }

    fn ev(ts: u64, series: &str, value: f64) -> Event {
        Event::new(ts, series, value)
    }

    // -- Pearson correlation -----------------------------------------------

    #[test]
    fn test_pearson_perfect_positive() {
        let pairs: Vec<(f64, f64)> = (1..=5).map(|i| (i as f64, i as f64)).collect();
        let r = pearson_correlation(&pairs);
        assert!((r - 1.0).abs() < 1e-9, "Expected ~1.0, got {r}");
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let pairs: Vec<(f64, f64)> = (1..=5).map(|i| (i as f64, -(i as f64))).collect();
        let r = pearson_correlation(&pairs);
        assert!((r + 1.0).abs() < 1e-9, "Expected ~-1.0, got {r}");
    }

    #[test]
    fn test_pearson_no_correlation() {
        // Two constant series → r = 0
        let pairs = vec![(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)];
        let r = pearson_correlation(&pairs);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_pearson_empty() {
        let r = pearson_correlation(&[]);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_pearson_single_pair() {
        let r = pearson_correlation(&[(1.0, 2.0)]);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_pearson_clamped_to_range() {
        let pairs: Vec<(f64, f64)> = (1..=10).map(|i| (i as f64, i as f64 * 2.0)).collect();
        let r = pearson_correlation(&pairs);
        assert!((-1.0..=1.0).contains(&r));
    }

    // -- EventCorrelator basics -------------------------------------------

    #[test]
    fn test_new_is_empty() {
        let c = EventCorrelator::new();
        assert_eq!(c.event_count(), 0);
    }

    #[test]
    fn test_ingest_increments_count() {
        let mut c = EventCorrelator::new();
        c.ingest(ev(1, "a", 1.0));
        c.ingest(ev(2, "a", 2.0));
        assert_eq!(c.event_count(), 2);
    }

    #[test]
    fn test_evaluate_rules_empty_no_rules() {
        let c = EventCorrelator::new();
        let results = c.evaluate_rules();
        assert!(results.is_empty());
    }

    #[test]
    fn test_evaluate_rule_no_events_returns_zero() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("r1", "a", "b", 10_000, 0.8));
        let results = c.evaluate_rules();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].coefficient, 0.0);
        assert!(!results[0].triggered);
    }

    // -- Positive correlation trigger ------------------------------------

    #[test]
    fn test_positive_correlation_triggers() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("temp-pressure", "temp", "pressure", 100_000, 0.8));

        for i in 1u64..=5 {
            c.ingest(ev(i * 1000, "temp", i as f64));
            c.ingest(ev(i * 1000, "pressure", i as f64 * 2.0));
        }

        let results = c.evaluate_rules();
        assert_eq!(results[0].rule_name, "temp-pressure");
        assert!(
            results[0].triggered,
            "Should trigger on positive correlation"
        );
        assert!((results[0].coefficient - 1.0).abs() < 1e-9);
    }

    // -- Negative correlation trigger ------------------------------------

    #[test]
    fn test_negative_correlation_triggers() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("inv", "x", "y", 100_000, 0.8));

        for i in 1u64..=5 {
            c.ingest(ev(i * 1000, "x", i as f64));
            c.ingest(ev(i * 1000, "y", -(i as f64)));
        }

        let results = c.evaluate_rules();
        assert!(
            results[0].triggered,
            "Should trigger on negative correlation"
        );
        assert!((results[0].coefficient + 1.0).abs() < 1e-9);
    }

    // -- Threshold not met -----------------------------------------------

    #[test]
    fn test_low_correlation_does_not_trigger() {
        let mut c = EventCorrelator::new();
        // High threshold — our data has moderate correlation only
        c.add_rule(rule("r", "a", "b", 100_000, 0.99));

        // Introduce some noise so correlation < 0.99
        c.ingest(ev(1000, "a", 1.0));
        c.ingest(ev(1000, "b", 3.0));
        c.ingest(ev(2000, "a", 2.0));
        c.ingest(ev(2000, "b", 1.0));
        c.ingest(ev(3000, "a", 3.0));
        c.ingest(ev(3000, "b", 2.0));

        let results = c.evaluate_rules();
        assert!(!results[0].triggered);
    }

    // -- Window filtering ------------------------------------------------

    #[test]
    fn test_window_excludes_old_events() {
        let mut c = EventCorrelator::new();
        // Window of 1000 ms
        c.add_rule(rule("r", "a", "b", 1000, 0.5));

        // Old events — outside window
        c.ingest(ev(0, "a", 100.0));
        c.ingest(ev(0, "b", 100.0));
        c.ingest(ev(1, "a", 100.0));
        c.ingest(ev(1, "b", 100.0));

        // Recent events (timestamp 5000 ms, window = [4000, 5000])
        c.ingest(ev(5000, "a", 1.0));
        c.ingest(ev(5000, "b", 2.0));
        c.ingest(ev(4500, "a", 2.0));
        c.ingest(ev(4500, "b", 4.0));
        c.ingest(ev(4000, "a", 3.0));
        c.ingest(ev(4000, "b", 6.0));

        let results = c.evaluate_rules();
        // Should only see 3 pairs from the window; old 100.0 values excluded
        assert!(results[0].sample_count <= 3);
    }

    #[test]
    fn test_window_too_narrow_yields_no_samples() {
        let mut c = EventCorrelator::new();
        // 1 ms window — practically empty
        c.add_rule(rule("r", "a", "b", 1, 0.5));

        c.ingest(ev(1000, "a", 1.0));
        c.ingest(ev(2000, "b", 1.0));

        let results = c.evaluate_rules();
        // No common timestamp within 1 ms window
        assert_eq!(results[0].sample_count, 0);
    }

    // -- purge_old_events ------------------------------------------------

    #[test]
    fn test_purge_removes_old_events() {
        let mut c = EventCorrelator::new();
        c.ingest(ev(100, "a", 1.0));
        c.ingest(ev(200, "a", 2.0));
        c.ingest(ev(300, "a", 3.0));
        let removed = c.purge_old_events(200);
        assert_eq!(removed, 1);
        assert_eq!(c.event_count(), 2);
    }

    #[test]
    fn test_purge_all() {
        let mut c = EventCorrelator::new();
        c.ingest(ev(100, "a", 1.0));
        c.ingest(ev(200, "a", 2.0));
        let removed = c.purge_old_events(u64::MAX);
        assert_eq!(removed, 2);
        assert_eq!(c.event_count(), 0);
    }

    #[test]
    fn test_purge_none() {
        let mut c = EventCorrelator::new();
        c.ingest(ev(100, "a", 1.0));
        let removed = c.purge_old_events(0);
        assert_eq!(removed, 0);
        assert_eq!(c.event_count(), 1);
    }

    #[test]
    fn test_purge_boundary_inclusive() {
        let mut c = EventCorrelator::new();
        c.ingest(ev(100, "a", 1.0));
        c.ingest(ev(200, "a", 2.0));
        // Timestamp 100 must survive (>= 100)
        let removed = c.purge_old_events(100);
        assert_eq!(removed, 0);
        assert_eq!(c.event_count(), 2);
    }

    // -- Multiple rules --------------------------------------------------

    #[test]
    fn test_multiple_rules_evaluated_independently() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("r1", "x", "y", 100_000, 0.5));
        c.add_rule(rule("r2", "x", "z", 100_000, 0.5));

        for i in 1u64..=4 {
            let ts = i * 1000;
            c.ingest(ev(ts, "x", i as f64));
            c.ingest(ev(ts, "y", i as f64)); // positive corr with x
            c.ingest(ev(ts, "z", -(i as f64))); // negative corr with x
        }

        let results = c.evaluate_rules();
        assert_eq!(results.len(), 2);
        let r1 = results.iter().find(|r| r.rule_name == "r1").expect("r1");
        let r2 = results.iter().find(|r| r.rule_name == "r2").expect("r2");
        assert!(r1.triggered);
        assert!(r2.triggered);
    }

    // -- Event with tags -------------------------------------------------

    #[test]
    fn test_event_with_tags_ingest() {
        let mut c = EventCorrelator::new();
        let mut tags = HashMap::new();
        tags.insert("region".into(), "eu".into());
        let e = Event::with_tags(1000, "temp", 22.5, tags);
        c.ingest(e);
        assert_eq!(c.event_count(), 1);
    }

    // -- Struct / default -----------------------------------------------

    #[test]
    fn test_event_correlator_default() {
        let _c: EventCorrelator = EventCorrelator::default();
    }

    #[test]
    fn test_correlation_result_fields() {
        let result = CorrelationResult {
            rule_name: "test".into(),
            coefficient: 0.95,
            sample_count: 10,
            triggered: true,
        };
        assert!(result.triggered);
        assert_eq!(result.sample_count, 10);
        assert!((result.coefficient - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_sample_count_matches_pairs() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("r", "a", "b", 100_000, 0.0));
        for i in 1u64..=7 {
            c.ingest(ev(i * 1000, "a", i as f64));
            c.ingest(ev(i * 1000, "b", i as f64));
        }
        let results = c.evaluate_rules();
        assert_eq!(results[0].sample_count, 7);
    }

    #[test]
    fn test_evaluate_single_event_per_series() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("r", "a", "b", 100_000, 0.5));
        c.ingest(ev(1000, "a", 5.0));
        c.ingest(ev(1000, "b", 5.0));
        let results = c.evaluate_rules();
        // Only 1 pair — not enough for Pearson
        assert_eq!(results[0].sample_count, 1);
        assert!(!results[0].triggered);
    }

    #[test]
    fn test_rule_different_timestamps_no_pairs() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("r", "a", "b", 100_000, 0.5));
        c.ingest(ev(1000, "a", 1.0));
        c.ingest(ev(2000, "b", 1.0));
        // No matching timestamps
        let results = c.evaluate_rules();
        assert_eq!(results[0].sample_count, 0);
    }

    #[test]
    fn test_zero_min_correlation_always_triggers_with_pairs() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("r", "a", "b", 100_000, 0.0));
        for i in 1u64..=3 {
            c.ingest(ev(i * 1000, "a", i as f64));
            c.ingest(ev(i * 1000, "b", 42.0));
        }
        let results = c.evaluate_rules();
        // r = 0 for constant "b", but |0| >= 0.0 is true
        assert!(results[0].triggered);
    }

    #[test]
    fn test_correlation_coefficient_in_range() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("r", "x", "y", 100_000, 0.0));
        for i in 1u64..=10 {
            c.ingest(ev(i * 100, "x", i as f64));
            c.ingest(ev(i * 100, "y", (i as f64).sqrt()));
        }
        let results = c.evaluate_rules();
        assert!(results[0].coefficient >= -1.0 && results[0].coefficient <= 1.0);
    }

    #[test]
    fn test_purge_mixed_series() {
        let mut c = EventCorrelator::new();
        c.ingest(ev(50, "a", 1.0));
        c.ingest(ev(50, "b", 2.0));
        c.ingest(ev(150, "a", 3.0));
        c.ingest(ev(150, "b", 4.0));
        let removed = c.purge_old_events(100);
        assert_eq!(removed, 2);
        assert_eq!(c.event_count(), 2);
    }

    #[test]
    fn test_add_multiple_rules_count() {
        let mut c = EventCorrelator::new();
        c.add_rule(rule("r1", "a", "b", 1000, 0.5));
        c.add_rule(rule("r2", "a", "c", 1000, 0.5));
        c.add_rule(rule("r3", "b", "c", 1000, 0.5));
        let results = c.evaluate_rules();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_event_new_no_tags() {
        let e = Event::new(999, "series", 2.71);
        assert_eq!(e.timestamp, 999);
        assert_eq!(e.series, "series");
        assert!((e.value - 2.71).abs() < 1e-9);
        assert!(e.tags.is_empty());
    }

    #[test]
    fn test_correlation_rule_new() {
        let r = CorrelationRule::new("name", "sa", "sb", 5000, 0.7);
        assert_eq!(r.name, "name");
        assert_eq!(r.series_a, "sa");
        assert_eq!(r.series_b, "sb");
        assert_eq!(r.window_ms, 5000);
        assert!((r.min_correlation - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_purge_returns_zero_when_nothing_removed() {
        let mut c = EventCorrelator::new();
        c.ingest(ev(1000, "x", 1.0));
        c.ingest(ev(2000, "x", 2.0));
        let removed = c.purge_old_events(500);
        assert_eq!(removed, 0);
        assert_eq!(c.event_count(), 2);
    }
}
