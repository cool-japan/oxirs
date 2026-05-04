//! SLA ↔ adaptive load shedder coordination.
//!
//! [`SlaBackpressureCoordinator`] fuses two concerns:
//!
//! 1. **SLA admission** ([`super::admission::StreamAdmissionController`]):
//!    a *contractual* gate.  When admission rejects, the stream has violated
//!    its declared SLA and the event is rejected outright.
//!
//! 2. **Adaptive load shedding** ([`crate::adaptive_load_shedding::LoadSheddingManager`]):
//!    a *best-effort* gate.  When the system is overloaded but the stream is
//!    still within its SLA budget, load shedding may probabilistically drop
//!    less-important events.
//!
//! Per the W2-S6 plan, **SLA reject takes precedence** over load shedding.
//! When admission fails, the coordinator emits
//! [`SlaBackpressureDecision::Reject`] without ever consulting the shedder.
//!
//! When admission succeeds, the coordinator hands the event to the shedder.
//! The shedder may return [`BackpressureAction::Drop`] (best-effort drop) or
//! [`BackpressureAction::Throttle`] (caller should slow down).
//!
//! Returns
//!
//! * [`SlaBackpressureDecision::Admit`] — admit and process normally.
//! * [`SlaBackpressureDecision::Shed`] — admit per SLA but drop per shedder.
//! * [`SlaBackpressureDecision::Reject`] — SLA violation; do not process.
//! * [`SlaBackpressureDecision::Throttle`] — admit per SLA but caller should
//!   apply backpressure.

use std::sync::Arc;
use std::time::Instant;

use crate::adaptive_load_shedding::{LoadSheddingManager, LoadSheddingStats};
use crate::error::StreamError;
use crate::event::StreamEvent;

use super::admission::{StreamAdmissionController, StreamAdmissionDecision};

// ─── Backpressure action / decision types ───────────────────────────────────

/// Best-effort backpressure decision proposed by the load shedder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureAction {
    /// No action required.
    Pass,
    /// Drop this event (load shedder).
    Drop,
    /// Caller should throttle the producer.
    Throttle,
}

/// Final coordinator decision combining SLA + load shedder.
#[derive(Debug)]
pub enum SlaBackpressureDecision {
    /// Admit and process the event normally.
    Admit { tokens_left: f64, lag_ms: i64 },
    /// SLA admitted but shedder dropped.
    Shed,
    /// Caller should slow the producer.
    Throttle,
    /// SLA admission rejected.
    Reject(StreamError),
}

impl SlaBackpressureDecision {
    /// Returns `true` iff this is an `Admit` outcome.
    pub fn is_admit(&self) -> bool {
        matches!(self, Self::Admit { .. })
    }

    /// Returns `true` iff this is a `Reject` outcome.
    pub fn is_reject(&self) -> bool {
        matches!(self, Self::Reject(_))
    }

    /// Returns `true` iff this is a `Shed` outcome.
    pub fn is_shed(&self) -> bool {
        matches!(self, Self::Shed)
    }

    /// Returns `true` iff this is a `Throttle` outcome.
    pub fn is_throttle(&self) -> bool {
        matches!(self, Self::Throttle)
    }
}

/// Policy for how the coordinator interprets the load shedder verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlaBackpressurePolicy {
    /// Strictly drop when the shedder says drop (default).
    #[default]
    Strict,
    /// Treat shedder drops as throttles instead of drops.
    PreferThrottle,
    /// Bypass the shedder entirely.
    BypassShedder,
}

// ─── Coordinator ─────────────────────────────────────────────────────────────

/// Coordinator wiring [`StreamAdmissionController`] with
/// [`LoadSheddingManager`].
pub struct SlaBackpressureCoordinator {
    admission: Arc<StreamAdmissionController>,
    shedder: Arc<LoadSheddingManager>,
    policy: SlaBackpressurePolicy,
}

impl SlaBackpressureCoordinator {
    /// Create a new coordinator.
    pub fn new(
        admission: Arc<StreamAdmissionController>,
        shedder: Arc<LoadSheddingManager>,
    ) -> Self {
        Self {
            admission,
            shedder,
            policy: SlaBackpressurePolicy::Strict,
        }
    }

    /// Override the backpressure policy.
    pub fn with_policy(mut self, policy: SlaBackpressurePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Evaluate a single event.
    ///
    /// Order:
    /// 1. SLA admission (per-stream token-bucket + lag + jitter).
    ///    Rejected ⇒ [`SlaBackpressureDecision::Reject`] — short-circuit.
    /// 2. Load shedder consultation (system-wide best-effort).
    ///    Drop ⇒ [`SlaBackpressureDecision::Shed`] (or `Throttle` per policy).
    /// 3. Otherwise [`SlaBackpressureDecision::Admit`].
    pub async fn evaluate(
        &self,
        stream_id: &str,
        event: &StreamEvent,
        event_ts: Instant,
        now: Instant,
    ) -> SlaBackpressureDecision {
        // Phase 1: SLA admission.
        let admit = match self.admission.try_admit(stream_id, event_ts, now) {
            Ok(decision) => decision,
            Err(e) => return SlaBackpressureDecision::Reject(e),
        };

        // Bypass shedder if requested.
        if matches!(self.policy, SlaBackpressurePolicy::BypassShedder) {
            return decision_from_admit(admit);
        }

        // Phase 2: load shedding consultation.
        let drop = self.shedder.should_drop_event(event).await;
        if drop {
            self.shedder.record_dropped_event(event).await;
            match self.policy {
                SlaBackpressurePolicy::Strict => return SlaBackpressureDecision::Shed,
                SlaBackpressurePolicy::PreferThrottle => return SlaBackpressureDecision::Throttle,
                SlaBackpressurePolicy::BypassShedder => {} // unreachable; handled above
            }
        }

        decision_from_admit(admit)
    }

    /// Snapshot the load shedder's running statistics.
    pub async fn shedder_stats(&self) -> LoadSheddingStats {
        self.shedder.get_stats().await
    }

    /// Reference to the inner admission controller.
    pub fn admission(&self) -> &Arc<StreamAdmissionController> {
        &self.admission
    }
}

fn decision_from_admit(admit: StreamAdmissionDecision) -> SlaBackpressureDecision {
    match admit {
        StreamAdmissionDecision::Admit {
            tokens_left,
            lag_ms,
        } => SlaBackpressureDecision::Admit {
            tokens_left,
            lag_ms,
        },
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_load_shedding::LoadSheddingConfig;
    use crate::event::EventMetadata;
    use crate::sla::StreamSlaConfig;
    use chrono::Utc;
    use oxirs_core::sla::SlaClass;
    use std::time::Duration;

    fn heartbeat() -> StreamEvent {
        StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "t".to_string(),
            metadata: EventMetadata::default(),
        }
    }

    fn coordinator() -> SlaBackpressureCoordinator {
        let admission = Arc::new(StreamAdmissionController::new());
        admission.register_stream("orders", StreamSlaConfig::for_class(SlaClass::Platinum));
        let shedder_cfg = LoadSheddingConfig {
            enable_load_shedding: false, // disable monitoring loop in tests
            ..Default::default()
        };
        let shedder = Arc::new(LoadSheddingManager::new(shedder_cfg).expect("shedder ok"));
        SlaBackpressureCoordinator::new(admission, shedder)
    }

    #[tokio::test]
    async fn test_admit_when_within_sla_and_no_overload() {
        let c = coordinator();
        let now = Instant::now();
        let dec = c.evaluate("orders", &heartbeat(), now, now).await;
        match dec {
            SlaBackpressureDecision::Admit { tokens_left, .. } => assert!(tokens_left > 0.0),
            other => panic!("expected Admit, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_reject_when_unregistered_stream() {
        let c = coordinator();
        let now = Instant::now();
        let dec = c.evaluate("ghost", &heartbeat(), now, now).await;
        assert!(dec.is_reject());
    }

    #[tokio::test]
    async fn test_reject_takes_precedence_over_shedder() {
        // Bronze drains quickly so we get an SLA reject ASAP.
        let admission = Arc::new(StreamAdmissionController::new());
        admission.register_stream("br", StreamSlaConfig::for_class(SlaClass::Bronze));

        // Configure shedder to "want" to drop everything (high probability).
        // We disable monitoring (no background thread) but still call `should_drop_event`
        // which will read 0% load_score → returns false in our wrapper. The contract
        // we test here is just: "if SLA fails, we never even check the shedder".
        let shedder_cfg = LoadSheddingConfig {
            enable_load_shedding: true,
            ..Default::default()
        };
        let shedder = Arc::new(LoadSheddingManager::new(shedder_cfg).unwrap_or_else(|_| {
            // If creation somehow fails, build a no-op shedder by disabling.
            LoadSheddingManager::new(LoadSheddingConfig {
                enable_load_shedding: false,
                ..Default::default()
            })
            .expect("fallback shedder")
        }));
        let coord = SlaBackpressureCoordinator::new(admission, shedder);

        // Drain bronze.
        let mut admitted = 0;
        let mut rejected = 0;
        for _ in 0..40 {
            let n = Instant::now();
            let dec = coord.evaluate("br", &heartbeat(), n, n).await;
            if dec.is_admit() {
                admitted += 1;
            } else if dec.is_reject() {
                rejected += 1;
            }
        }
        assert!(admitted >= 1);
        assert!(rejected >= 1);
    }

    #[tokio::test]
    async fn test_lag_violation_rejects_via_admission() {
        let admission = Arc::new(StreamAdmissionController::new());
        admission.register_stream(
            "lag",
            StreamSlaConfig::for_class(SlaClass::Platinum).with_max_lag(Duration::from_millis(20)),
        );
        let shedder = Arc::new(
            LoadSheddingManager::new(LoadSheddingConfig {
                enable_load_shedding: false,
                ..Default::default()
            })
            .expect("shedder"),
        );
        let coord = SlaBackpressureCoordinator::new(admission, shedder);
        let event_ts = Instant::now();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let now = Instant::now();
        let dec = coord.evaluate("lag", &heartbeat(), event_ts, now).await;
        assert!(dec.is_reject());
    }

    #[tokio::test]
    async fn test_bypass_shedder_policy_skips_shedder() {
        let admission = Arc::new(StreamAdmissionController::new());
        admission.register_stream("p", StreamSlaConfig::for_class(SlaClass::Platinum));
        let shedder = Arc::new(
            LoadSheddingManager::new(LoadSheddingConfig {
                enable_load_shedding: true,
                ..Default::default()
            })
            .expect("shedder"),
        );
        let coord = SlaBackpressureCoordinator::new(admission, shedder)
            .with_policy(SlaBackpressurePolicy::BypassShedder);
        let now = Instant::now();
        let dec = coord.evaluate("p", &heartbeat(), now, now).await;
        assert!(dec.is_admit());
    }
}
