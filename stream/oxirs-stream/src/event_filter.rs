//! Stream event filtering with composable predicates.
//!
//! Provides a `StreamEvent` type, a `FilterPredicate` enum with logical combinators
//! (And, Or, Not), and an `EventFilter` utility for applying predicates to event slices.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// StreamEvent
// ────────────────────────────────────────────────────────────────────────────

/// A stream event with a key, value, topic, timestamp, and optional headers.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamEvent {
    /// The event key (partition routing hint or entity identifier).
    pub key: String,
    /// The event payload value.
    pub value: String,
    /// The topic (channel) this event was published on.
    pub topic: String,
    /// Timestamp in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Arbitrary key-value metadata headers.
    pub headers: HashMap<String, String>,
}

impl StreamEvent {
    /// Create a new event with no headers.
    pub fn new(key: &str, value: &str, topic: &str, timestamp_ms: u64) -> Self {
        Self {
            key: key.to_string(),
            value: value.to_string(),
            topic: topic.to_string(),
            timestamp_ms,
            headers: HashMap::new(),
        }
    }

    /// Builder-style method to add a header.
    pub fn with_header(
        mut self,
        header_key: impl Into<String>,
        header_val: impl Into<String>,
    ) -> Self {
        self.headers.insert(header_key.into(), header_val.into());
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FilterPredicate
// ────────────────────────────────────────────────────────────────────────────

/// A composable filter predicate for `StreamEvent` matching.
#[derive(Debug, Clone)]
pub enum FilterPredicate {
    /// Key must equal this exact string.
    KeyEquals(String),
    /// Value must contain this substring.
    ValueContains(String),
    /// Topic must match exactly.
    TopicIs(String),
    /// Timestamp must be greater than or equal to this value (inclusive lower bound).
    TimestampAfter(u64),
    /// Timestamp must be less than or equal to this value (inclusive upper bound).
    TimestampBefore(u64),
    /// A header with the given key must have the given value.
    HeaderMatches(String, String),
    /// Logical AND: both sub-predicates must match.
    And(Box<FilterPredicate>, Box<FilterPredicate>),
    /// Logical OR: at least one sub-predicate must match.
    Or(Box<FilterPredicate>, Box<FilterPredicate>),
    /// Logical NOT: the sub-predicate must not match.
    Not(Box<FilterPredicate>),
}

impl FilterPredicate {
    /// Evaluate this predicate against a single `StreamEvent`.
    pub fn matches(&self, event: &StreamEvent) -> bool {
        match self {
            FilterPredicate::KeyEquals(k) => &event.key == k,
            FilterPredicate::ValueContains(sub) => event.value.contains(sub.as_str()),
            FilterPredicate::TopicIs(t) => &event.topic == t,
            FilterPredicate::TimestampAfter(ts) => event.timestamp_ms >= *ts,
            FilterPredicate::TimestampBefore(ts) => event.timestamp_ms <= *ts,
            FilterPredicate::HeaderMatches(hk, hv) => event
                .headers
                .get(hk.as_str())
                .map(|v| v == hv)
                .unwrap_or(false),
            FilterPredicate::And(left, right) => left.matches(event) && right.matches(event),
            FilterPredicate::Or(left, right) => left.matches(event) || right.matches(event),
            FilterPredicate::Not(inner) => !inner.matches(event),
        }
    }

    /// Combine this predicate with another using logical AND.
    pub fn and(self, other: FilterPredicate) -> FilterPredicate {
        FilterPredicate::And(Box::new(self), Box::new(other))
    }

    /// Combine this predicate with another using logical OR.
    pub fn or(self, other: FilterPredicate) -> FilterPredicate {
        FilterPredicate::Or(Box::new(self), Box::new(other))
    }
}

impl std::ops::Not for FilterPredicate {
    type Output = FilterPredicate;

    /// Negate this predicate.
    fn not(self) -> FilterPredicate {
        FilterPredicate::Not(Box::new(self))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// EventFilter
// ────────────────────────────────────────────────────────────────────────────

/// Applies `FilterPredicate`s to slices of `StreamEvent`s.
///
/// This is a stateless utility struct — all methods are pure functions.
#[derive(Debug, Clone, Default)]
pub struct EventFilter;

impl EventFilter {
    /// Create a new `EventFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Return references to all events in `events` that satisfy `predicate`.
    pub fn filter<'a>(
        &self,
        events: &'a [StreamEvent],
        predicate: &FilterPredicate,
    ) -> Vec<&'a StreamEvent> {
        events.iter().filter(|e| predicate.matches(e)).collect()
    }

    /// Return the count of events that satisfy `predicate`.
    pub fn count(&self, events: &[StreamEvent], predicate: &FilterPredicate) -> usize {
        events.iter().filter(|e| predicate.matches(e)).count()
    }

    /// Return `true` if at least one event satisfies `predicate`.
    pub fn any(&self, events: &[StreamEvent], predicate: &FilterPredicate) -> bool {
        events.iter().any(|e| predicate.matches(e))
    }

    /// Return `true` if every event satisfies `predicate`.
    ///
    /// Returns `true` for an empty slice (vacuous truth).
    pub fn all(&self, events: &[StreamEvent], predicate: &FilterPredicate) -> bool {
        events.iter().all(|e| predicate.matches(e))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Not;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_event(key: &str, value: &str, topic: &str, ts: u64) -> StreamEvent {
        StreamEvent::new(key, value, topic, ts)
    }

    fn make_event_with_header(
        key: &str,
        value: &str,
        topic: &str,
        ts: u64,
        hk: &str,
        hv: &str,
    ) -> StreamEvent {
        StreamEvent::new(key, value, topic, ts).with_header(hk, hv)
    }

    fn sample_events() -> Vec<StreamEvent> {
        vec![
            make_event("k1", "hello world", "topic-a", 1000),
            make_event("k2", "foo bar", "topic-b", 2000),
            make_event("k1", "hello again", "topic-a", 3000),
            make_event("k3", "testing", "topic-c", 4000),
            make_event_with_header("k4", "header test", "topic-d", 5000, "x-type", "rdf"),
        ]
    }

    // ── StreamEvent construction ──────────────────────────────────────────

    #[test]
    fn test_stream_event_new_fields() {
        let e = StreamEvent::new("mykey", "myvalue", "mytopic", 9999);
        assert_eq!(e.key, "mykey");
        assert_eq!(e.value, "myvalue");
        assert_eq!(e.topic, "mytopic");
        assert_eq!(e.timestamp_ms, 9999);
        assert!(e.headers.is_empty());
    }

    #[test]
    fn test_stream_event_with_header() {
        let e = StreamEvent::new("k", "v", "t", 1).with_header("content-type", "json");
        assert_eq!(
            e.headers.get("content-type").map(|s| s.as_str()),
            Some("json")
        );
    }

    #[test]
    fn test_stream_event_multiple_headers() {
        let e = StreamEvent::new("k", "v", "t", 1)
            .with_header("a", "1")
            .with_header("b", "2");
        assert_eq!(e.headers.len(), 2);
    }

    #[test]
    fn test_stream_event_equality() {
        let e1 = make_event("k", "v", "t", 10);
        let e2 = make_event("k", "v", "t", 10);
        assert_eq!(e1, e2);
    }

    // ── KeyEquals predicate ───────────────────────────────────────────────

    #[test]
    fn test_key_equals_match() {
        let e = make_event("my-key", "val", "topic", 0);
        assert!(FilterPredicate::KeyEquals("my-key".to_string()).matches(&e));
    }

    #[test]
    fn test_key_equals_no_match() {
        let e = make_event("other-key", "val", "topic", 0);
        assert!(!FilterPredicate::KeyEquals("my-key".to_string()).matches(&e));
    }

    #[test]
    fn test_key_equals_case_sensitive() {
        let e = make_event("Key", "val", "topic", 0);
        assert!(!FilterPredicate::KeyEquals("key".to_string()).matches(&e));
    }

    // ── ValueContains predicate ───────────────────────────────────────────

    #[test]
    fn test_value_contains_match() {
        let e = make_event("k", "hello world", "t", 0);
        assert!(FilterPredicate::ValueContains("hello".to_string()).matches(&e));
    }

    #[test]
    fn test_value_contains_no_match() {
        let e = make_event("k", "hello world", "t", 0);
        assert!(!FilterPredicate::ValueContains("xyz".to_string()).matches(&e));
    }

    #[test]
    fn test_value_contains_exact_match() {
        let e = make_event("k", "exact", "t", 0);
        assert!(FilterPredicate::ValueContains("exact".to_string()).matches(&e));
    }

    #[test]
    fn test_value_contains_empty_substring() {
        // Empty substring always matches
        let e = make_event("k", "anything", "t", 0);
        assert!(FilterPredicate::ValueContains(String::new()).matches(&e));
    }

    // ── TopicIs predicate ─────────────────────────────────────────────────

    #[test]
    fn test_topic_is_match() {
        let e = make_event("k", "v", "events.rdf", 0);
        assert!(FilterPredicate::TopicIs("events.rdf".to_string()).matches(&e));
    }

    #[test]
    fn test_topic_is_no_match() {
        let e = make_event("k", "v", "events.rdf", 0);
        assert!(!FilterPredicate::TopicIs("other".to_string()).matches(&e));
    }

    // ── TimestampAfter predicate ──────────────────────────────────────────

    #[test]
    fn test_timestamp_after_match() {
        let e = make_event("k", "v", "t", 5000);
        assert!(FilterPredicate::TimestampAfter(4999).matches(&e));
    }

    #[test]
    fn test_timestamp_after_equal_is_match() {
        let e = make_event("k", "v", "t", 5000);
        assert!(FilterPredicate::TimestampAfter(5000).matches(&e));
    }

    #[test]
    fn test_timestamp_after_no_match() {
        let e = make_event("k", "v", "t", 3000);
        assert!(!FilterPredicate::TimestampAfter(5000).matches(&e));
    }

    // ── TimestampBefore predicate ─────────────────────────────────────────

    #[test]
    fn test_timestamp_before_match() {
        let e = make_event("k", "v", "t", 1000);
        assert!(FilterPredicate::TimestampBefore(2000).matches(&e));
    }

    #[test]
    fn test_timestamp_before_equal_is_match() {
        let e = make_event("k", "v", "t", 2000);
        assert!(FilterPredicate::TimestampBefore(2000).matches(&e));
    }

    #[test]
    fn test_timestamp_before_no_match() {
        let e = make_event("k", "v", "t", 9000);
        assert!(!FilterPredicate::TimestampBefore(2000).matches(&e));
    }

    // ── HeaderMatches predicate ───────────────────────────────────────────

    #[test]
    fn test_header_matches_match() {
        let e = make_event_with_header("k", "v", "t", 0, "x-source", "sensor-1");
        assert!(
            FilterPredicate::HeaderMatches("x-source".to_string(), "sensor-1".to_string())
                .matches(&e)
        );
    }

    #[test]
    fn test_header_matches_wrong_value() {
        let e = make_event_with_header("k", "v", "t", 0, "x-source", "sensor-1");
        assert!(
            !FilterPredicate::HeaderMatches("x-source".to_string(), "sensor-2".to_string())
                .matches(&e)
        );
    }

    #[test]
    fn test_header_matches_missing_key() {
        let e = make_event("k", "v", "t", 0);
        assert!(
            !FilterPredicate::HeaderMatches("x-source".to_string(), "anything".to_string())
                .matches(&e)
        );
    }

    // ── And combinator ────────────────────────────────────────────────────

    #[test]
    fn test_and_both_true() {
        let e = make_event("k1", "hello world", "topic-a", 1500);
        let pred = FilterPredicate::KeyEquals("k1".to_string())
            .and(FilterPredicate::ValueContains("hello".to_string()));
        assert!(pred.matches(&e));
    }

    #[test]
    fn test_and_first_false() {
        let e = make_event("k2", "hello world", "topic-a", 1500);
        let pred = FilterPredicate::KeyEquals("k1".to_string())
            .and(FilterPredicate::ValueContains("hello".to_string()));
        assert!(!pred.matches(&e));
    }

    #[test]
    fn test_and_second_false() {
        let e = make_event("k1", "goodbye world", "topic-a", 1500);
        let pred = FilterPredicate::KeyEquals("k1".to_string())
            .and(FilterPredicate::ValueContains("hello".to_string()));
        assert!(!pred.matches(&e));
    }

    #[test]
    fn test_and_both_false() {
        let e = make_event("k2", "goodbye world", "topic-a", 1500);
        let pred = FilterPredicate::KeyEquals("k1".to_string())
            .and(FilterPredicate::ValueContains("hello".to_string()));
        assert!(!pred.matches(&e));
    }

    // ── Or combinator ─────────────────────────────────────────────────────

    #[test]
    fn test_or_first_true() {
        let e = make_event("k1", "no match", "topic-a", 0);
        let pred = FilterPredicate::KeyEquals("k1".to_string())
            .or(FilterPredicate::ValueContains("hello".to_string()));
        assert!(pred.matches(&e));
    }

    #[test]
    fn test_or_second_true() {
        let e = make_event("k2", "hello there", "topic-a", 0);
        let pred = FilterPredicate::KeyEquals("k1".to_string())
            .or(FilterPredicate::ValueContains("hello".to_string()));
        assert!(pred.matches(&e));
    }

    #[test]
    fn test_or_both_false() {
        let e = make_event("k2", "nothing", "topic-a", 0);
        let pred = FilterPredicate::KeyEquals("k1".to_string())
            .or(FilterPredicate::ValueContains("hello".to_string()));
        assert!(!pred.matches(&e));
    }

    // ── Not combinator ────────────────────────────────────────────────────

    #[test]
    fn test_not_negates_match() {
        let e = make_event("k1", "v", "t", 0);
        let pred = FilterPredicate::KeyEquals("k1".to_string()).not();
        assert!(!pred.matches(&e));
    }

    #[test]
    fn test_not_negates_non_match() {
        let e = make_event("k2", "v", "t", 0);
        let pred = FilterPredicate::KeyEquals("k1".to_string()).not();
        assert!(pred.matches(&e));
    }

    // ── Nested combinators ────────────────────────────────────────────────

    #[test]
    fn test_nested_and_or() {
        // (key == "k1" OR key == "k2") AND timestamp >= 2000
        let e1 = make_event("k1", "v", "t", 2500);
        let e2 = make_event("k2", "v", "t", 1500);
        let e3 = make_event("k3", "v", "t", 3000);

        let pred = FilterPredicate::KeyEquals("k1".to_string())
            .or(FilterPredicate::KeyEquals("k2".to_string()))
            .and(FilterPredicate::TimestampAfter(2000));

        assert!(pred.matches(&e1)); // k1 AND ts >= 2000
        assert!(!pred.matches(&e2)); // k2 but ts < 2000
        assert!(!pred.matches(&e3)); // ts >= 2000 but key is k3
    }

    #[test]
    fn test_nested_not_and() {
        // NOT (topic == "topic-a") AND timestamp <= 3000
        let e_pass = make_event("k", "v", "topic-b", 2000);
        let e_fail_topic = make_event("k", "v", "topic-a", 2000);
        let e_fail_ts = make_event("k", "v", "topic-b", 5000);

        let pred = FilterPredicate::TopicIs("topic-a".to_string())
            .not()
            .and(FilterPredicate::TimestampBefore(3000));

        assert!(pred.matches(&e_pass));
        assert!(!pred.matches(&e_fail_topic));
        assert!(!pred.matches(&e_fail_ts));
    }

    #[test]
    fn test_triple_nested_predicate() {
        // key == "k1" AND (value contains "hello" OR header x-type == "rdf")
        let e1 = make_event("k1", "hello there", "t", 0);
        let e2 = make_event_with_header("k1", "data", "t", 0, "x-type", "rdf");
        let e3 = make_event("k1", "data", "t", 0); // no hello, no header
        let e4 = make_event("k2", "hello there", "t", 0); // wrong key

        let pred = FilterPredicate::KeyEquals("k1".to_string()).and(
            FilterPredicate::ValueContains("hello".to_string()).or(FilterPredicate::HeaderMatches(
                "x-type".to_string(),
                "rdf".to_string(),
            )),
        );

        assert!(pred.matches(&e1));
        assert!(pred.matches(&e2));
        assert!(!pred.matches(&e3));
        assert!(!pred.matches(&e4));
    }

    // ── EventFilter::filter ───────────────────────────────────────────────

    #[test]
    fn test_filter_returns_matching_subset() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::TopicIs("topic-a".to_string());
        let result = ef.filter(&events, &pred);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|e| e.topic == "topic-a"));
    }

    #[test]
    fn test_filter_empty_input() {
        let ef = EventFilter::new();
        let pred = FilterPredicate::KeyEquals("anything".to_string());
        let result = ef.filter(&[], &pred);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_no_matches() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::KeyEquals("nonexistent".to_string());
        let result = ef.filter(&events, &pred);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_all_match() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::TimestampAfter(0);
        let result = ef.filter(&events, &pred);
        assert_eq!(result.len(), events.len());
    }

    #[test]
    fn test_filter_returns_references_to_original() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::KeyEquals("k1".to_string());
        let result = ef.filter(&events, &pred);
        assert_eq!(result.len(), 2);
        // Verify the references point into the original slice
        assert_eq!(result[0].key, "k1");
        assert_eq!(result[1].key, "k1");
    }

    // ── EventFilter::count ────────────────────────────────────────────────

    #[test]
    fn test_count_matching_events() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::KeyEquals("k1".to_string());
        assert_eq!(ef.count(&events, &pred), 2);
    }

    #[test]
    fn test_count_empty_input() {
        let ef = EventFilter::new();
        let pred = FilterPredicate::KeyEquals("k".to_string());
        assert_eq!(ef.count(&[], &pred), 0);
    }

    #[test]
    fn test_count_zero_matches() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::KeyEquals("ghost".to_string());
        assert_eq!(ef.count(&events, &pred), 0);
    }

    #[test]
    fn test_count_all_match() {
        let events = vec![
            make_event("k", "v", "t", 100),
            make_event("k", "v", "t", 200),
            make_event("k", "v", "t", 300),
        ];
        let ef = EventFilter::new();
        let pred = FilterPredicate::TopicIs("t".to_string());
        assert_eq!(ef.count(&events, &pred), 3);
    }

    // ── EventFilter::any ──────────────────────────────────────────────────

    #[test]
    fn test_any_true_when_one_matches() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::TopicIs("topic-c".to_string());
        assert!(ef.any(&events, &pred));
    }

    #[test]
    fn test_any_false_when_none_match() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::TopicIs("topic-z".to_string());
        assert!(!ef.any(&events, &pred));
    }

    #[test]
    fn test_any_empty_returns_false() {
        let ef = EventFilter::new();
        let pred = FilterPredicate::KeyEquals("k".to_string());
        assert!(!ef.any(&[], &pred));
    }

    // ── EventFilter::all ──────────────────────────────────────────────────

    #[test]
    fn test_all_true_when_all_match() {
        let events = vec![
            make_event("k", "hello", "t", 100),
            make_event("k", "hello there", "t", 200),
        ];
        let ef = EventFilter::new();
        let pred = FilterPredicate::ValueContains("hello".to_string());
        assert!(ef.all(&events, &pred));
    }

    #[test]
    fn test_all_false_when_one_fails() {
        let events = vec![
            make_event("k", "hello", "t", 100),
            make_event("k", "goodbye", "t", 200),
        ];
        let ef = EventFilter::new();
        let pred = FilterPredicate::ValueContains("hello".to_string());
        assert!(!ef.all(&events, &pred));
    }

    #[test]
    fn test_all_empty_returns_true() {
        let ef = EventFilter::new();
        let pred = FilterPredicate::KeyEquals("impossible".to_string());
        assert!(ef.all(&[], &pred));
    }

    // ── Timestamp window (After + Before combined) ────────────────────────

    #[test]
    fn test_timestamp_window_filter() {
        let events = sample_events(); // ts: 1000, 2000, 3000, 4000, 5000
        let ef = EventFilter::new();
        let pred =
            FilterPredicate::TimestampAfter(2000).and(FilterPredicate::TimestampBefore(4000));
        let result = ef.filter(&events, &pred);
        assert_eq!(result.len(), 3); // ts 2000, 3000, 4000
        for e in &result {
            assert!(e.timestamp_ms >= 2000 && e.timestamp_ms <= 4000);
        }
    }

    // ── Default impl ──────────────────────────────────────────────────────

    #[test]
    fn test_event_filter_default() {
        let ef = EventFilter;
        let events = vec![make_event("k", "v", "t", 0)];
        let pred = FilterPredicate::KeyEquals("k".to_string());
        assert_eq!(ef.count(&events, &pred), 1);
    }

    // ── Header with multiple entries ──────────────────────────────────────

    #[test]
    fn test_header_matches_with_multiple_headers() {
        let e = make_event_with_header("k", "v", "t", 0, "h1", "v1")
            .with_header("h2", "v2")
            .with_header("h3", "v3");
        assert!(FilterPredicate::HeaderMatches("h2".to_string(), "v2".to_string()).matches(&e));
        assert!(!FilterPredicate::HeaderMatches("h2".to_string(), "wrong".to_string()).matches(&e));
    }

    // ── TopicIs + KeyEquals combined ─────────────────────────────────────

    #[test]
    fn test_topic_and_key_combined() {
        let events = sample_events();
        let ef = EventFilter::new();
        let pred = FilterPredicate::TopicIs("topic-a".to_string())
            .and(FilterPredicate::KeyEquals("k1".to_string()));
        let result = ef.filter(&events, &pred);
        assert_eq!(result.len(), 2);
    }
}
