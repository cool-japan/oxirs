//! Per-stream SLA admission controller.
//!
//! Wraps [`oxirs_core::sla::AdmissionController`] (a tenant-keyed token bucket)
//! with a stream-keyed view that also enforces stream-specific budgets:
//!
//! * `max_events_per_sec` — sustained rate cap.
//! * `max_lag` — maximum permitted ingestion lag (current-now − event-ts).
//! * `jitter_budget` — maximum allowed wall-clock jitter between successive
//!   admissions (used as a smoothness/jitter contract).
//!
//! Tokens are budgeted from a `SlaClass` (Bronze .. Platinum) — the per-stream
//! `StreamSlaConfig` overrides the rate cap derived from that class.
//!
//! Returns
//!
//! * `Ok(StreamAdmissionDecision::Admit { tokens_left, lag_ms })` on success.
//! * `Err(StreamError::SlaExceeded { … })` for over-rate, over-lag, or
//!   over-jitter rejections.
//! * `Err(StreamError::SlaExceeded { … })` for unregistered streams (the
//!   error variant is reused with `reason = "stream not registered"`).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use oxirs_core::sla::{AdmissionController, AdmissionError, SlaClass};

use crate::error::StreamError;

// ─── StreamSlaConfig ─────────────────────────────────────────────────────────

/// SLA contract attached to a single stream.
///
/// All fields are *limits*; admission rejects when an event would push the
/// stream past any of them.
#[derive(Debug, Clone)]
pub struct StreamSlaConfig {
    /// SLA class used to derive the underlying token bucket
    /// (refill rate, capacity, dispatch priority).
    pub class: SlaClass,
    /// Maximum sustained rate in events per second.  Must be `> 0`.
    /// Defaults to the `token_refill_rate` of `class`.
    pub max_events_per_sec: f64,
    /// Maximum permitted wall-clock lag between event timestamp and now.
    /// `None` means lag is unbounded.
    pub max_lag: Option<Duration>,
    /// Maximum allowed inter-arrival jitter (in milliseconds) between two
    /// successive admitted events for the stream.  `None` disables the check.
    pub jitter_budget_ms: Option<u64>,
    /// Token cost charged per admitted event (default: 1.0).
    pub token_cost: f64,
}

impl StreamSlaConfig {
    /// Build a default configuration for the given SLA class.
    pub fn for_class(class: SlaClass) -> Self {
        let thresholds = class.thresholds();
        Self {
            class,
            max_events_per_sec: thresholds.token_refill_rate,
            max_lag: Some(Duration::from_millis(thresholds.max_latency_p99_ms)),
            jitter_budget_ms: None,
            token_cost: 1.0,
        }
    }

    /// Override the rate cap for this stream.  Must be `> 0`.
    pub fn with_rate(mut self, max_events_per_sec: f64) -> Self {
        self.max_events_per_sec = max_events_per_sec;
        self
    }

    /// Override the maximum lag.
    pub fn with_max_lag(mut self, max_lag: Duration) -> Self {
        self.max_lag = Some(max_lag);
        self
    }

    /// Override the jitter budget in milliseconds.
    pub fn with_jitter_budget(mut self, jitter_budget_ms: u64) -> Self {
        self.jitter_budget_ms = Some(jitter_budget_ms);
        self
    }

    /// Override the per-event token cost.
    pub fn with_token_cost(mut self, cost: f64) -> Self {
        self.token_cost = cost;
        self
    }
}

// ─── Decision and stats types ────────────────────────────────────────────────

/// The outcome of [`StreamAdmissionController::try_admit`].
#[derive(Debug, Clone, PartialEq)]
pub enum StreamAdmissionDecision {
    /// The event was admitted.
    ///
    /// * `tokens_left` — remaining token budget after deduction.
    /// * `lag_ms` — observed wall-clock lag for this event.
    Admit { tokens_left: f64, lag_ms: i64 },
}

/// Cumulative per-stream admission statistics.
#[derive(Debug, Clone, Default)]
pub struct StreamAdmissionStats {
    pub admitted: u64,
    pub rejected_rate: u64,
    pub rejected_lag: u64,
    pub rejected_jitter: u64,
    pub last_admit_at_nanos: Option<u128>,
}

// ─── Internal per-stream state ───────────────────────────────────────────────

struct StreamState {
    config: StreamSlaConfig,
    last_admit: Option<Instant>,
    stats: StreamAdmissionStats,
}

impl StreamState {
    fn new(config: StreamSlaConfig) -> Self {
        Self {
            config,
            last_admit: None,
            stats: StreamAdmissionStats::default(),
        }
    }
}

// ─── StreamAdmissionController ──────────────────────────────────────────────

/// Stream-keyed admission controller backed by per-stream token buckets.
///
/// Internally each registered stream is also registered as a *tenant* with the
/// shared `oxirs_core::sla::AdmissionController` so the token-bucket
/// implementation is reused verbatim.
pub struct StreamAdmissionController {
    /// Underlying tenant-keyed token bucket from `oxirs_core::sla`.
    inner: AdmissionController,
    /// Per-stream metadata layered on top of the inner controller.
    streams: Mutex<HashMap<String, StreamState>>,
}

impl Default for StreamAdmissionController {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAdmissionController {
    /// Create an empty controller.
    pub fn new() -> Self {
        Self {
            inner: AdmissionController::new(),
            streams: Mutex::new(HashMap::new()),
        }
    }

    /// Register a stream.
    ///
    /// Re-registering an existing stream resets its bucket and stats.
    pub fn register_stream(&self, stream_id: &str, config: StreamSlaConfig) {
        // Register with the shared core controller (tenant_id == stream_id).
        self.inner.register_tenant(stream_id, config.class);
        let mut streams = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        streams.insert(stream_id.to_string(), StreamState::new(config));
    }

    /// Deregister a stream.
    pub fn deregister_stream(&self, stream_id: &str) -> bool {
        let mut streams = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed_meta = streams.remove(stream_id).is_some();
        let removed_inner = self.inner.deregister_tenant(stream_id);
        removed_meta || removed_inner
    }

    /// Try to admit a single event for `stream_id`.
    ///
    /// Arguments
    ///
    /// * `stream_id` — registered stream identifier.
    /// * `event_ts` — the event's source timestamp.
    /// * `now` — the current wall-clock instant (typically `Instant::now()`).
    ///
    /// Returns
    ///
    /// * `Ok(StreamAdmissionDecision::Admit { … })` on success.
    /// * `Err(StreamError::SlaExceeded { … })` if any of rate / lag /
    ///   jitter / registration checks fails.
    pub fn try_admit(
        &self,
        stream_id: &str,
        event_ts: Instant,
        now: Instant,
    ) -> Result<StreamAdmissionDecision, StreamError> {
        let mut streams = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let state = streams
            .get_mut(stream_id)
            .ok_or_else(|| StreamError::SlaExceeded {
                stream_id: stream_id.to_string(),
                reason: "stream not registered".to_string(),
            })?;

        // ── Lag check ──────────────────────────────────────────────────────
        let lag = now.saturating_duration_since(event_ts);
        let lag_ms_signed = if now >= event_ts {
            lag.as_millis() as i64
        } else {
            -(event_ts.saturating_duration_since(now).as_millis() as i64)
        };
        if let Some(max_lag) = state.config.max_lag {
            if lag > max_lag {
                state.stats.rejected_lag += 1;
                return Err(StreamError::SlaExceeded {
                    stream_id: stream_id.to_string(),
                    reason: format!(
                        "lag {}ms exceeds max_lag {}ms",
                        lag.as_millis(),
                        max_lag.as_millis()
                    ),
                });
            }
        }

        // ── Jitter check ───────────────────────────────────────────────────
        if let Some(jitter_ms) = state.config.jitter_budget_ms {
            if let Some(prev) = state.last_admit {
                let inter = now.saturating_duration_since(prev).as_millis() as u64;
                if inter > jitter_ms {
                    state.stats.rejected_jitter += 1;
                    return Err(StreamError::SlaExceeded {
                        stream_id: stream_id.to_string(),
                        reason: format!(
                            "inter-arrival {inter}ms exceeds jitter_budget {jitter_ms}ms"
                        ),
                    });
                }
            }
        }

        // ── Rate check (token bucket) ─────────────────────────────────────
        let cost = state.config.token_cost;
        match self.inner.try_admit_with_cost(stream_id, cost) {
            Ok(()) => {
                state.stats.admitted += 1;
                state.last_admit = Some(now);
                state.stats.last_admit_at_nanos =
                    Some(now.saturating_duration_since(*BASELINE_INSTANT).as_nanos());
                let tokens_left = self.inner.available_tokens(stream_id).unwrap_or(0.0);
                Ok(StreamAdmissionDecision::Admit {
                    tokens_left,
                    lag_ms: lag_ms_signed,
                })
            }
            Err(AdmissionError::RateLimitExceeded { .. }) => {
                state.stats.rejected_rate += 1;
                Err(StreamError::SlaExceeded {
                    stream_id: stream_id.to_string(),
                    reason: format!(
                        "rate limit exceeded (max_events_per_sec={})",
                        state.config.max_events_per_sec
                    ),
                })
            }
            Err(AdmissionError::TenantNotRegistered { .. }) => Err(StreamError::SlaExceeded {
                stream_id: stream_id.to_string(),
                reason: "stream not registered with inner controller".to_string(),
            }),
        }
    }

    /// Snapshot per-stream stats.
    pub fn stats(&self, stream_id: &str) -> Option<StreamAdmissionStats> {
        let streams = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        streams.get(stream_id).map(|s| s.stats.clone())
    }

    /// Total number of registered streams.
    pub fn stream_count(&self) -> usize {
        self.streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Lookup the SLA class registered for a stream.
    pub fn sla_class(&self, stream_id: &str) -> Option<SlaClass> {
        self.inner.sla_class(stream_id)
    }
}

// ─── Baseline instant ────────────────────────────────────────────────────────
//
// `Instant` has no public epoch; we capture process start as a baseline so we
// can produce a monotonic nanosecond stamp for `last_admit_at_nanos` without
// dragging in a chrono / SystemTime dependency for the hot path.

use once_cell::sync::Lazy;

static BASELINE_INSTANT: Lazy<Instant> = Lazy::new(Instant::now);

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    fn now() -> Instant {
        Instant::now()
    }

    #[test]
    fn test_register_and_admit_default_class() {
        let ctrl = StreamAdmissionController::new();
        ctrl.register_stream("s", StreamSlaConfig::for_class(SlaClass::Platinum));
        let n = now();
        let dec = ctrl.try_admit("s", n, n).expect("Platinum first admit");
        match dec {
            StreamAdmissionDecision::Admit { tokens_left, .. } => {
                assert!(tokens_left > 0.0);
            }
        }
        assert_eq!(ctrl.stats("s").map(|s| s.admitted), Some(1));
    }

    #[test]
    fn test_unregistered_stream_returns_sla_exceeded() {
        let ctrl = StreamAdmissionController::new();
        let n = now();
        let err = ctrl.try_admit("ghost", n, n).expect_err("ghost");
        assert!(matches!(err, StreamError::SlaExceeded { .. }));
    }

    #[test]
    fn test_lag_check_rejects_event_with_excessive_lag() {
        let ctrl = StreamAdmissionController::new();
        ctrl.register_stream(
            "lagging",
            StreamSlaConfig::for_class(SlaClass::Gold).with_max_lag(Duration::from_millis(50)),
        );
        let event_ts = now();
        // Sleep beyond max_lag to simulate a stale event.
        sleep(Duration::from_millis(80));
        let now_ts = now();
        let err = ctrl
            .try_admit("lagging", event_ts, now_ts)
            .expect_err("over-lag should fail");
        match err {
            StreamError::SlaExceeded { stream_id, reason } => {
                assert_eq!(stream_id, "lagging");
                assert!(reason.contains("max_lag"));
            }
            other => panic!("expected SlaExceeded, got {other:?}"),
        }
        assert_eq!(ctrl.stats("lagging").map(|s| s.rejected_lag), Some(1));
    }

    #[test]
    fn test_rate_check_rejects_when_bucket_drained() {
        let ctrl = StreamAdmissionController::new();
        // Bronze: capacity = 5, refill = 1/s → burst 5 then reject.
        ctrl.register_stream("bronze", StreamSlaConfig::for_class(SlaClass::Bronze));

        let mut admitted = 0usize;
        let mut rejected = 0usize;
        for _ in 0..20 {
            let n = now();
            match ctrl.try_admit("bronze", n, n) {
                Ok(_) => admitted += 1,
                Err(StreamError::SlaExceeded { .. }) => rejected += 1,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(admitted >= 1);
        assert!(rejected >= 1);
        let stats = ctrl.stats("bronze").expect("registered");
        assert_eq!(stats.admitted as usize, admitted);
        assert_eq!(stats.rejected_rate as usize, rejected);
    }

    #[test]
    fn test_jitter_budget_rejects_when_inter_arrival_exceeds_budget() {
        let ctrl = StreamAdmissionController::new();
        ctrl.register_stream(
            "jit",
            StreamSlaConfig::for_class(SlaClass::Platinum).with_jitter_budget(20),
        );
        let n0 = now();
        ctrl.try_admit("jit", n0, n0).expect("first admit");
        sleep(Duration::from_millis(40));
        let n1 = now();
        let err = ctrl.try_admit("jit", n1, n1).expect_err("over-jitter");
        match err {
            StreamError::SlaExceeded { reason, .. } => {
                assert!(reason.contains("jitter_budget"));
            }
            other => panic!("expected SlaExceeded, got {other:?}"),
        }
    }

    #[test]
    fn test_deregister_stream_removes_state() {
        let ctrl = StreamAdmissionController::new();
        ctrl.register_stream("x", StreamSlaConfig::for_class(SlaClass::Silver));
        assert_eq!(ctrl.stream_count(), 1);
        assert!(ctrl.deregister_stream("x"));
        assert_eq!(ctrl.stream_count(), 0);
        let n = now();
        let err = ctrl.try_admit("x", n, n).expect_err("removed");
        assert!(matches!(err, StreamError::SlaExceeded { .. }));
    }

    #[test]
    fn test_sla_class_introspection() {
        let ctrl = StreamAdmissionController::new();
        ctrl.register_stream("g", StreamSlaConfig::for_class(SlaClass::Gold));
        assert_eq!(ctrl.sla_class("g"), Some(SlaClass::Gold));
        assert_eq!(ctrl.sla_class("absent"), None);
    }

    #[test]
    fn test_stream_sla_config_builder() {
        let cfg = StreamSlaConfig::for_class(SlaClass::Silver)
            .with_rate(123.4)
            .with_max_lag(Duration::from_millis(800))
            .with_jitter_budget(15)
            .with_token_cost(2.5);
        assert!((cfg.max_events_per_sec - 123.4).abs() < 1e-9);
        assert_eq!(cfg.max_lag, Some(Duration::from_millis(800)));
        assert_eq!(cfg.jitter_budget_ms, Some(15));
        assert!((cfg.token_cost - 2.5).abs() < 1e-9);
    }
}
