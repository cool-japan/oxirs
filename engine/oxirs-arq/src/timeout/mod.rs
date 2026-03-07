//! SPARQL query timeout enforcement
//!
//! Enforces time limits on query execution to prevent runaway queries.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

/// Timeout error types
#[derive(Debug, thiserror::Error)]
pub enum TimeoutError {
    #[error("Query exceeded timeout of {duration_ms}ms")]
    QueryTimeout { duration_ms: u64 },
    #[error("Query was cancelled")]
    QueryCancelled,
}

/// A handle representing an active query timeout.
///
/// Created via [`QueryTimeout::new`]. Call [`QueryTimeout::check`] periodically
/// inside inner query-execution loops to enforce the deadline.
pub struct QueryTimeout {
    pub deadline: Instant,
    pub cancelled: Arc<AtomicBool>,
    duration: Duration,
}

impl QueryTimeout {
    /// Create a new timeout that expires after `duration`.
    pub fn new(duration: Duration) -> Self {
        Self {
            deadline: Instant::now() + duration,
            cancelled: Arc::new(AtomicBool::new(false)),
            duration,
        }
    }

    /// Returns `true` if the timeout deadline has passed.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.deadline
    }

    /// Returns the remaining time until the deadline, or `None` if already expired.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline.checked_duration_since(Instant::now())
    }

    /// Cancel the query externally. Future calls to [`QueryTimeout::check`] will return
    /// [`TimeoutError::QueryCancelled`].
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check whether the timeout or cancellation has been triggered.
    ///
    /// Call this in inner execution loops. Returns `Ok(())` if execution
    /// should continue, or the appropriate `Err` variant otherwise.
    pub fn check(&self) -> Result<(), TimeoutError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(TimeoutError::QueryCancelled);
        }
        if self.is_expired() {
            return Err(TimeoutError::QueryTimeout {
                duration_ms: self.duration.as_millis() as u64,
            });
        }
        Ok(())
    }

    /// Obtain a cloneable cancellation handle that shares the same flag.
    pub fn cancellation_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }
}

/// Configuration for timeout behaviour across a SPARQL endpoint.
pub struct TimeoutConfig {
    /// Default timeout applied when the client does not specify one.
    /// `None` means no timeout by default.
    pub default_timeout: Option<Duration>,
    /// Hard upper bound that even admin requests cannot exceed
    /// unless `admin_bypass` is set.
    pub max_timeout: Duration,
    /// When `true`, requests that carry admin credentials skip timeout enforcement.
    pub admin_bypass: bool,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Some(Duration::from_secs(30)),
            max_timeout: Duration::from_secs(300),
            admin_bypass: true,
        }
    }
}

impl TimeoutConfig {
    /// Resolve the effective timeout for a request.
    ///
    /// * `requested` - the timeout the client asked for (may come from a URL
    ///   parameter or request header).
    /// * `is_admin` - whether the requesting user has admin privileges.
    ///
    /// Returns `None` when no timeout should be applied.
    pub fn resolve(&self, requested: Option<Duration>, is_admin: bool) -> Option<Duration> {
        if is_admin && self.admin_bypass {
            return None;
        }
        let base = requested.or(self.default_timeout)?;
        Some(base.min(self.max_timeout))
    }
}

/// Parse a timeout value supplied as a URL query-parameter or HTTP header value.
///
/// Accepted formats:
/// * `"30s"` - 30 seconds
/// * `"30000ms"` or `"30000"` (raw integer) - milliseconds
/// * `"30m"` - 30 minutes
/// * `"PT30S"` (ISO 8601 duration subset) - 30 seconds
pub fn parse_timeout_param(value: &str) -> Result<Duration, String> {
    let trimmed = value.trim();

    if trimmed.is_empty() {
        return Err("Timeout value is empty".to_string());
    }

    // ISO 8601 subset: PTnS / PTnM
    if trimmed.to_ascii_uppercase().starts_with("PT") {
        let upper = trimmed.to_ascii_uppercase();
        let inner = &upper[2..]; // strip "PT"
        if let Some(s_idx) = inner.find('S') {
            let num_str = &inner[..s_idx];
            if let Ok(secs) = num_str.parse::<f64>() {
                if secs < 0.0 {
                    return Err("Timeout must not be negative".to_string());
                }
                return Ok(Duration::from_millis((secs * 1000.0) as u64));
            }
        }
        return Err(format!("Cannot parse ISO 8601 duration: {trimmed}"));
    }

    // Suffix-based parsing
    if let Some(num_str) = trimmed.strip_suffix("ms") {
        let ms: u64 = num_str
            .trim()
            .parse()
            .map_err(|_| format!("Invalid millisecond value: {num_str}"))?;
        return Ok(Duration::from_millis(ms));
    }
    if let Some(num_str) = trimmed.strip_suffix('s') {
        let secs: f64 = num_str
            .trim()
            .parse()
            .map_err(|_| format!("Invalid second value: {num_str}"))?;
        if secs < 0.0 {
            return Err("Timeout must not be negative".to_string());
        }
        return Ok(Duration::from_millis((secs * 1000.0) as u64));
    }
    if let Some(num_str) = trimmed.strip_suffix('m') {
        let mins: f64 = num_str
            .trim()
            .parse()
            .map_err(|_| format!("Invalid minute value: {num_str}"))?;
        if mins < 0.0 {
            return Err("Timeout must not be negative".to_string());
        }
        return Ok(Duration::from_millis((mins * 60.0 * 1000.0) as u64));
    }

    // Plain integer -> milliseconds
    if let Ok(ms) = trimmed.parse::<u64>() {
        return Ok(Duration::from_millis(ms));
    }

    Err(format!("Unrecognised timeout format: {trimmed}"))
}

/// An iterator adaptor that enforces a [`QueryTimeout`] on every `N`-th item.
///
/// Each call to `next` returns `Some(Ok(item))` while the timeout has not
/// fired, `Some(Err(TimeoutError))` on the first expiry/cancellation, and
/// `None` once the inner iterator is exhausted.
pub struct TimeoutIterator<I: Iterator> {
    inner: I,
    timeout: Arc<QueryTimeout>,
    /// How many items to yield before rechecking the timeout.
    check_interval: usize,
    count: usize,
    /// Set to true after a timeout/cancellation error has been emitted.
    /// After this, the iterator returns `None` to terminate the iteration.
    terminated: bool,
}

impl<I: Iterator> TimeoutIterator<I> {
    /// Wrap `iter` with a timeout check every `check_interval` items.
    pub fn new(iter: I, timeout: Arc<QueryTimeout>, check_interval: usize) -> Self {
        Self {
            inner: iter,
            timeout,
            check_interval: check_interval.max(1),
            count: 0,
            terminated: false,
        }
    }
}

impl<I: Iterator> Iterator for TimeoutIterator<I> {
    type Item = Result<I::Item, TimeoutError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.terminated {
            return None;
        }
        self.count += 1;
        if self.count % self.check_interval == 0 {
            if let Err(e) = self.timeout.check() {
                self.terminated = true;
                return Some(Err(e));
            }
        }

        self.inner.next().map(Ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::Duration;

    // QueryTimeout basic behaviour

    #[test]
    fn test_not_expired_immediately() {
        let t = QueryTimeout::new(Duration::from_secs(60));
        assert!(!t.is_expired(), "Should not be expired immediately");
    }

    #[test]
    fn test_expires_after_deadline() {
        let t = QueryTimeout::new(Duration::from_millis(10));
        thread::sleep(Duration::from_millis(50));
        assert!(t.is_expired(), "Should be expired after deadline");
    }

    #[test]
    fn test_remaining_positive_before_expiry() {
        let t = QueryTimeout::new(Duration::from_secs(60));
        assert!(
            t.remaining().is_some(),
            "Remaining should be Some before expiry"
        );
    }

    #[test]
    fn test_remaining_none_after_expiry() {
        let t = QueryTimeout::new(Duration::from_millis(1));
        thread::sleep(Duration::from_millis(30));
        assert!(
            t.remaining().is_none(),
            "Remaining should be None after expiry"
        );
    }

    #[test]
    fn test_remaining_decreases_over_time() {
        let t = QueryTimeout::new(Duration::from_secs(10));
        let first = t.remaining().expect("first remaining");
        thread::sleep(Duration::from_millis(50));
        let second = t.remaining().expect("second remaining");
        assert!(second < first, "Remaining should decrease");
    }

    #[test]
    fn test_check_ok_before_expiry() {
        let t = QueryTimeout::new(Duration::from_secs(60));
        assert!(t.check().is_ok(), "check() should be Ok before expiry");
    }

    #[test]
    fn test_check_timeout_after_expiry() {
        let t = QueryTimeout::new(Duration::from_millis(10));
        thread::sleep(Duration::from_millis(50));
        match t.check() {
            Err(TimeoutError::QueryTimeout { .. }) => {}
            other => panic!("Expected QueryTimeout, got {other:?}"),
        }
    }

    // Cancellation

    #[test]
    fn test_cancel_not_cancelled_initially() {
        let t = QueryTimeout::new(Duration::from_secs(60));
        assert!(!t.cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn test_cancel_sets_flag() {
        let t = QueryTimeout::new(Duration::from_secs(60));
        t.cancel();
        assert!(t.cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn test_check_returns_cancelled_after_cancel() {
        let t = QueryTimeout::new(Duration::from_secs(60));
        t.cancel();
        match t.check() {
            Err(TimeoutError::QueryCancelled) => {}
            other => panic!("Expected QueryCancelled, got {other:?}"),
        }
    }

    #[test]
    fn test_cancel_takes_priority_over_expiry() {
        let t = QueryTimeout::new(Duration::from_millis(1));
        t.cancel();
        thread::sleep(Duration::from_millis(20));
        // Both cancelled and expired; cancellation flag checked first
        match t.check() {
            Err(TimeoutError::QueryCancelled) => {}
            other => panic!("Expected QueryCancelled, got {other:?}"),
        }
    }

    #[test]
    fn test_cancellation_handle_shares_flag() {
        let t = QueryTimeout::new(Duration::from_secs(60));
        let handle = t.cancellation_handle();
        handle.store(true, Ordering::Release);
        assert!(t.cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn test_cancellation_handle_from_another_thread() {
        let t = Arc::new(QueryTimeout::new(Duration::from_secs(60)));
        let handle = t.cancellation_handle();
        let t2 = Arc::clone(&t);

        let join = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            handle.store(true, Ordering::Release);
        });

        join.join().expect("thread panicked");
        match t2.check() {
            Err(TimeoutError::QueryCancelled) => {}
            other => panic!("Expected QueryCancelled, got {other:?}"),
        }
    }

    // TimeoutConfig::resolve

    #[test]
    fn test_config_default_values() {
        let cfg = TimeoutConfig::default();
        assert_eq!(cfg.default_timeout, Some(Duration::from_secs(30)));
        assert_eq!(cfg.max_timeout, Duration::from_secs(300));
        assert!(cfg.admin_bypass);
    }

    #[test]
    fn test_config_admin_bypass_returns_none() {
        let cfg = TimeoutConfig::default();
        let result = cfg.resolve(Some(Duration::from_secs(10)), true);
        assert!(result.is_none(), "Admin should bypass timeout");
    }

    #[test]
    fn test_config_no_admin_bypass_applies_timeout() {
        let cfg = TimeoutConfig {
            admin_bypass: false,
            ..Default::default()
        };
        let result = cfg.resolve(Some(Duration::from_secs(10)), true);
        assert_eq!(result, Some(Duration::from_secs(10)));
    }

    #[test]
    fn test_config_clamps_to_max() {
        let cfg = TimeoutConfig::default();
        let result = cfg.resolve(Some(Duration::from_secs(9999)), false);
        assert_eq!(result, Some(Duration::from_secs(300)));
    }

    #[test]
    fn test_config_uses_default_when_no_request() {
        let cfg = TimeoutConfig::default();
        let result = cfg.resolve(None, false);
        assert_eq!(result, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_config_none_default_no_request_returns_none() {
        let cfg = TimeoutConfig {
            default_timeout: None,
            ..Default::default()
        };
        let result = cfg.resolve(None, false);
        assert!(result.is_none());
    }

    #[test]
    fn test_config_within_max_passes_through() {
        let cfg = TimeoutConfig::default();
        let result = cfg.resolve(Some(Duration::from_secs(60)), false);
        assert_eq!(result, Some(Duration::from_secs(60)));
    }

    // parse_timeout_param

    #[test]
    fn test_parse_seconds_suffix() {
        let d = parse_timeout_param("30s").expect("parse 30s");
        assert_eq!(d, Duration::from_secs(30));
    }

    #[test]
    fn test_parse_milliseconds_suffix() {
        let d = parse_timeout_param("30000ms").expect("parse 30000ms");
        assert_eq!(d, Duration::from_millis(30_000));
    }

    #[test]
    fn test_parse_plain_integer_milliseconds() {
        let d = parse_timeout_param("30000").expect("parse 30000");
        assert_eq!(d, Duration::from_millis(30_000));
    }

    #[test]
    fn test_parse_minutes_suffix() {
        let d = parse_timeout_param("1m").expect("parse 1m");
        assert_eq!(d, Duration::from_secs(60));
    }

    #[test]
    fn test_parse_fractional_seconds() {
        let d = parse_timeout_param("1.5s").expect("parse 1.5s");
        assert_eq!(d, Duration::from_millis(1500));
    }

    #[test]
    fn test_parse_iso8601_pt30s() {
        let d = parse_timeout_param("PT30S").expect("parse PT30S");
        assert_eq!(d, Duration::from_secs(30));
    }

    #[test]
    fn test_parse_iso8601_lowercase() {
        let d = parse_timeout_param("pt30s").expect("parse pt30s");
        assert_eq!(d, Duration::from_secs(30));
    }

    #[test]
    fn test_parse_zero_milliseconds() {
        let d = parse_timeout_param("0ms").expect("parse 0ms");
        assert_eq!(d, Duration::from_millis(0));
    }

    #[test]
    fn test_parse_zero_seconds() {
        let d = parse_timeout_param("0s").expect("parse 0s");
        assert_eq!(d, Duration::ZERO);
    }

    #[test]
    fn test_parse_empty_string_errors() {
        assert!(parse_timeout_param("").is_err());
    }

    #[test]
    fn test_parse_invalid_string_errors() {
        assert!(parse_timeout_param("abc").is_err());
    }

    #[test]
    fn test_parse_with_leading_whitespace() {
        let d = parse_timeout_param("  30s").expect("parse '  30s'");
        assert_eq!(d, Duration::from_secs(30));
    }

    #[test]
    fn test_parse_large_ms_value() {
        let d = parse_timeout_param("86400000ms").expect("parse 86400000ms");
        assert_eq!(d, Duration::from_secs(86400));
    }

    // TimeoutIterator

    #[test]
    fn test_iterator_yields_all_items_before_timeout() {
        let t = Arc::new(QueryTimeout::new(Duration::from_secs(60)));
        let iter = TimeoutIterator::new(0..5, Arc::clone(&t), 100);
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn test_iterator_stops_on_expiry() {
        let t = Arc::new(QueryTimeout::new(Duration::from_millis(1)));
        thread::sleep(Duration::from_millis(30));
        let iter = TimeoutIterator::new(0..1000, Arc::clone(&t), 1);
        // First item (count==1, 1%1==0) triggers the check
        let first = iter.into_iter().next();
        match first {
            Some(Err(TimeoutError::QueryTimeout { .. })) => {}
            other => panic!("Expected QueryTimeout, got {other:?}"),
        }
    }

    #[test]
    fn test_iterator_stops_on_cancellation() {
        let t = Arc::new(QueryTimeout::new(Duration::from_secs(60)));
        t.cancel();
        let iter = TimeoutIterator::new(0..1000, Arc::clone(&t), 1);
        let first = iter.into_iter().next();
        match first {
            Some(Err(TimeoutError::QueryCancelled)) => {}
            other => panic!("Expected QueryCancelled, got {other:?}"),
        }
    }

    #[test]
    fn test_iterator_checks_at_interval() {
        // check_interval=3: check on item 3, 6, 9, ...
        // Timeout fires before iteration (already expired).
        let t = Arc::new(QueryTimeout::new(Duration::from_millis(1)));
        thread::sleep(Duration::from_millis(30));
        let iter = TimeoutIterator::new(0..100, Arc::clone(&t), 3);
        let results: Vec<_> = iter.collect();
        // Items 1 & 2 pass (no check at those counts), item 3 triggers timeout
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        match &results[2] {
            Err(TimeoutError::QueryTimeout { .. }) => {}
            other => panic!("Expected QueryTimeout at index 2, got {other:?}"),
        }
    }

    #[test]
    fn test_iterator_empty_inner() {
        let t = Arc::new(QueryTimeout::new(Duration::from_secs(60)));
        let iter = TimeoutIterator::new(std::iter::empty::<i32>(), t, 1);
        let results: Vec<_> = iter.collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_iterator_with_string_items() {
        let t = Arc::new(QueryTimeout::new(Duration::from_secs(60)));
        let data = vec!["alpha", "beta", "gamma"];
        let iter = TimeoutIterator::new(data.into_iter(), t, 10);
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 3);
        assert_eq!(*results[0].as_ref().unwrap(), "alpha");
    }

    // Error display

    #[test]
    fn test_timeout_error_display_timeout() {
        let e = TimeoutError::QueryTimeout { duration_ms: 5000 };
        let msg = format!("{e}");
        assert!(msg.contains("5000"));
    }

    #[test]
    fn test_timeout_error_display_cancelled() {
        let e = TimeoutError::QueryCancelled;
        let msg = format!("{e}");
        assert!(msg.to_lowercase().contains("cancel"));
    }

    // Combined scenario

    #[test]
    fn test_full_scenario_config_parse_enforce() {
        let cfg = TimeoutConfig::default();
        // A non-admin request asking for 45s -> allowed (less than 300s max)
        let effective = cfg.resolve(Some(Duration::from_secs(45)), false);
        assert_eq!(effective, Some(Duration::from_secs(45)));

        // Create the timeout and verify it has not yet fired
        let t = QueryTimeout::new(effective.unwrap());
        assert!(t.check().is_ok());
        assert!(t.remaining().is_some());
    }

    #[test]
    fn test_parse_then_new_timeout() {
        let d = parse_timeout_param("500ms").expect("parse");
        let t = QueryTimeout::new(d);
        assert!(t.check().is_ok());
        thread::sleep(Duration::from_millis(600));
        assert!(t.is_expired());
    }

    #[test]
    fn test_config_non_admin_uses_default_when_none_requested() {
        let cfg = TimeoutConfig {
            default_timeout: Some(Duration::from_secs(15)),
            max_timeout: Duration::from_secs(60),
            admin_bypass: true,
        };
        let result = cfg.resolve(None, false);
        assert_eq!(result, Some(Duration::from_secs(15)));
    }

    #[test]
    fn test_parse_2m() {
        let d = parse_timeout_param("2m").expect("parse 2m");
        assert_eq!(d, Duration::from_secs(120));
    }

    #[test]
    fn test_parse_pt10s_float() {
        let d = parse_timeout_param("PT10S").expect("parse PT10S");
        assert_eq!(d, Duration::from_secs(10));
    }

    #[test]
    fn test_iterator_interval_one_checks_every_item() {
        let t = Arc::new(QueryTimeout::new(Duration::from_secs(60)));
        let iter = TimeoutIterator::new(0..10, Arc::clone(&t), 1);
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 10);
        for r in &results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn test_timeout_duration_stored_correctly() {
        let d = Duration::from_millis(250);
        let t = QueryTimeout::new(d);
        // duration field is private, but we can verify behaviour through check()
        thread::sleep(Duration::from_millis(300));
        match t.check() {
            Err(TimeoutError::QueryTimeout { duration_ms }) => {
                assert_eq!(duration_ms, 250);
            }
            other => panic!("Unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_multiple_cancel_calls_idempotent() {
        let t = QueryTimeout::new(Duration::from_secs(60));
        t.cancel();
        t.cancel();
        t.cancel();
        assert!(t.cancelled.load(Ordering::Acquire));
        assert!(matches!(t.check(), Err(TimeoutError::QueryCancelled)));
    }

    #[test]
    fn test_config_max_timeout_boundary() {
        let cfg = TimeoutConfig {
            default_timeout: Some(Duration::from_secs(10)),
            max_timeout: Duration::from_secs(100),
            admin_bypass: false,
        };
        // Exactly at max
        assert_eq!(
            cfg.resolve(Some(Duration::from_secs(100)), false),
            Some(Duration::from_secs(100))
        );
        // One ms over max
        assert_eq!(
            cfg.resolve(Some(Duration::from_millis(100_001)), false),
            Some(Duration::from_secs(100))
        );
    }
}
