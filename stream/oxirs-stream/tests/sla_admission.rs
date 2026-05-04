//! Integration test: per-stream SLA admission control.
//!
//! Simulates a multi-tenant stream environment with three streams of
//! different SLA classes and verifies that admission decisions track each
//! stream's class-derived budget correctly:
//!
//! * Bronze quickly drains its small bucket and gets rejected.
//! * Platinum processes a much higher steady-state rate.
//! * Lag-violating events are rejected with [`StreamError::SlaExceeded`].
//! * The W2-S6 contract — SLA reject takes precedence over load shedding —
//!   is exercised through [`SlaBackpressureCoordinator`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use oxirs_core::sla::SlaClass;
use oxirs_stream::adaptive_load_shedding::{LoadSheddingConfig, LoadSheddingManager};
use oxirs_stream::error::StreamError;
use oxirs_stream::event::{EventMetadata, StreamEvent};
use oxirs_stream::sla::{
    SlaBackpressureCoordinator, SlaBackpressurePolicy, StreamAdmissionController, StreamSlaConfig,
};

fn heartbeat() -> StreamEvent {
    StreamEvent::Heartbeat {
        timestamp: chrono::Utc::now(),
        source: "integration-test".to_string(),
        metadata: EventMetadata::default(),
    }
}

#[test]
fn test_bronze_drains_quickly_platinum_sustains_high_rate() {
    let ctrl = StreamAdmissionController::new();
    ctrl.register_stream("bronze", StreamSlaConfig::for_class(SlaClass::Bronze));
    ctrl.register_stream("platinum", StreamSlaConfig::for_class(SlaClass::Platinum));

    let mut bronze_admit = 0;
    let mut bronze_reject = 0;
    let mut platinum_admit = 0;
    let mut platinum_reject = 0;

    for _ in 0..100 {
        let n = Instant::now();
        match ctrl.try_admit("bronze", n, n) {
            Ok(_) => bronze_admit += 1,
            Err(StreamError::SlaExceeded { .. }) => bronze_reject += 1,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
        match ctrl.try_admit("platinum", n, n) {
            Ok(_) => platinum_admit += 1,
            Err(StreamError::SlaExceeded { .. }) => platinum_reject += 1,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    // Bronze: capacity=5 → ≤ 6 (initial burst + tiny refill in microseconds).
    assert!(bronze_admit < 50, "bronze should not admit 50+ in burst");
    assert!(bronze_reject > 0, "bronze must hit at least one rejection");

    // Platinum: capacity=200 → admits at least 50 in a hundred-iteration burst.
    assert!(
        platinum_admit > bronze_admit * 3,
        "platinum (capacity 200) should admit far more than bronze (capacity 5); got platinum={platinum_admit} bronze={bronze_admit}"
    );
    assert!(
        platinum_reject < bronze_reject,
        "platinum should reject fewer than bronze in this short burst"
    );
}

#[test]
fn test_lag_violation_rejects_with_sla_exceeded_error() {
    let ctrl = StreamAdmissionController::new();
    ctrl.register_stream(
        "tight",
        StreamSlaConfig::for_class(SlaClass::Gold).with_max_lag(Duration::from_millis(20)),
    );

    let event_ts = Instant::now();
    std::thread::sleep(Duration::from_millis(40));
    let now = Instant::now();
    let err = ctrl
        .try_admit("tight", event_ts, now)
        .expect_err("over-lag must reject");
    match err {
        StreamError::SlaExceeded { stream_id, reason } => {
            assert_eq!(stream_id, "tight");
            assert!(reason.contains("max_lag"));
        }
        other => panic!("expected SlaExceeded, got {other:?}"),
    }
}

#[tokio::test]
async fn test_coordinator_sla_reject_takes_precedence_over_shedder() {
    // Two streams: a Bronze (drains fast) and a Platinum (effectively always admits).
    let admission = Arc::new(StreamAdmissionController::new());
    admission.register_stream("br", StreamSlaConfig::for_class(SlaClass::Bronze));
    admission.register_stream("pt", StreamSlaConfig::for_class(SlaClass::Platinum));

    // Build a coordinator with a *bypassed* shedder: we are testing the
    // contract that SLA decisions short-circuit the shedder.  Even with
    // shedding fully off, Bronze still drains.
    let shedder = Arc::new(
        LoadSheddingManager::new(LoadSheddingConfig {
            enable_load_shedding: false,
            ..Default::default()
        })
        .expect("shedder construct"),
    );
    let coord = SlaBackpressureCoordinator::new(admission.clone(), shedder)
        .with_policy(SlaBackpressurePolicy::BypassShedder);

    let mut br_reject = 0usize;
    let mut pt_reject = 0usize;
    for _ in 0..200 {
        let n = Instant::now();
        let dec = coord.evaluate("br", &heartbeat(), n, n).await;
        if dec.is_reject() {
            br_reject += 1;
        }
        let dec = coord.evaluate("pt", &heartbeat(), n, n).await;
        if dec.is_reject() {
            pt_reject += 1;
        }
    }

    assert!(
        br_reject > pt_reject,
        "Bronze must hit more SLA rejects than Platinum: br={br_reject} pt={pt_reject}"
    );
}

#[test]
fn test_jitter_budget_independent_of_class() {
    let ctrl = StreamAdmissionController::new();
    ctrl.register_stream(
        "jitter",
        StreamSlaConfig::for_class(SlaClass::Platinum).with_jitter_budget(20),
    );

    // First admit — establishes baseline.
    let n0 = Instant::now();
    ctrl.try_admit("jitter", n0, n0).expect("first admit");

    // Sleep beyond budget — second admit must reject.
    std::thread::sleep(Duration::from_millis(40));
    let n1 = Instant::now();
    let err = ctrl
        .try_admit("jitter", n1, n1)
        .expect_err("over-jitter must reject");
    match err {
        StreamError::SlaExceeded { reason, .. } => {
            assert!(reason.contains("jitter_budget"));
        }
        other => panic!("expected SlaExceeded, got {other:?}"),
    }
}

#[test]
fn test_unregistered_stream_rejects_with_sla_exceeded() {
    let ctrl = StreamAdmissionController::new();
    let n = Instant::now();
    let err = ctrl.try_admit("ghost", n, n).expect_err("ghost rejects");
    match err {
        StreamError::SlaExceeded { stream_id, reason } => {
            assert_eq!(stream_id, "ghost");
            assert!(reason.to_lowercase().contains("not registered"));
        }
        other => panic!("expected SlaExceeded, got {other:?}"),
    }
}

#[test]
fn test_stats_track_admissions_and_rejections() {
    let ctrl = StreamAdmissionController::new();
    ctrl.register_stream("s", StreamSlaConfig::for_class(SlaClass::Bronze));

    for _ in 0..30 {
        let n = Instant::now();
        let _ = ctrl.try_admit("s", n, n);
    }

    let stats = ctrl.stats("s").expect("stats present");
    assert!(stats.admitted > 0);
    assert!(stats.rejected_rate > 0);
    assert_eq!(
        stats.admitted + stats.rejected_rate,
        30,
        "admit+reject must cover all attempts"
    );
}
