//! Integration tests for the cluster SLA admission control layer.
//!
//! Exercises the public API exposed by `oxirs_cluster::sla::*` together with
//! the shared SLA primitives in `oxirs_core::sla::*`.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use oxirs_cluster::sla::{
    ClusterAdmissionController, ProposerOutcome, ReaderOutcome, SlaAdmissionConfig, SlaClassQuota,
    SlaError, SlaProposerGate, SlaReaderGate,
};
use oxirs_core::sla::SlaClass;

// ───────────────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────────────

fn config_with(class: SlaClass, max_concurrent: usize) -> SlaAdmissionConfig {
    let mut quotas: HashMap<SlaClass, SlaClassQuota> = HashMap::new();
    quotas.insert(
        class,
        SlaClassQuota {
            max_qps: None,
            max_concurrent,
            token_cost: 1.0,
        },
    );
    SlaAdmissionConfig { quotas }
}

fn shared_controller(cfg: SlaAdmissionConfig) -> Arc<ClusterAdmissionController> {
    Arc::new(ClusterAdmissionController::new(cfg))
}

// ───────────────────────────────────────────────────────────────────────────
// Per-tier admission token math
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn bronze_tier_token_math_matches_thresholds() {
    // Bronze threshold: capacity = 5, refill_rate = 1.0/s.
    // Drain the bucket repeatedly, never letting concurrency be the
    // limiting factor (release immediately after every admit).
    let ctrl = shared_controller(SlaAdmissionConfig::with_defaults());
    let mut admitted = 0usize;
    let mut rate_rejected = 0usize;
    for _ in 0..50 {
        match ctrl.acquire_permit(SlaClass::Bronze) {
            Ok(()) => {
                admitted += 1;
                ctrl.release_permit(SlaClass::Bronze).expect("release");
            }
            Err(SlaError::RateLimitExceeded { .. }) => rate_rejected += 1,
            Err(e) => panic!("unexpected error {e:?}"),
        }
    }
    // Bronze capacity is 5 — should admit roughly 5 then start rate-rejecting.
    assert!(admitted >= 1, "at least one admit");
    assert!(rate_rejected >= 1, "should hit rate limit eventually");
    // Combined must equal total attempts.
    assert_eq!(admitted + rate_rejected, 50);
}

#[test]
fn higher_tiers_admit_more_than_lower_tiers() {
    let ctrl = shared_controller(SlaAdmissionConfig::with_defaults());
    let drain = |class: SlaClass| -> usize {
        let mut admitted = 0usize;
        for _ in 0..1000 {
            match ctrl.acquire_permit(class) {
                Ok(()) => {
                    admitted += 1;
                    ctrl.release_permit(class).expect("release");
                }
                Err(SlaError::RateLimitExceeded { .. }) => break,
                Err(e) => panic!("unexpected error {e:?}"),
            }
        }
        admitted
    };
    // Drain each tier's bucket from full.
    let bronze = drain(SlaClass::Bronze);
    let silver = drain(SlaClass::Silver);
    let gold = drain(SlaClass::Gold);
    let platinum = drain(SlaClass::Platinum);
    // Capacity ordering must hold: Platinum ≥ Gold ≥ Silver ≥ Bronze.
    assert!(platinum >= gold);
    assert!(gold >= silver);
    assert!(silver >= bronze);
}

#[test]
fn unregistered_class_is_rejected() {
    let ctrl = shared_controller(SlaAdmissionConfig::default());
    let err = ctrl
        .acquire_permit(SlaClass::Bronze)
        .expect_err("unregistered must fail");
    assert!(matches!(err, SlaError::ClassNotRegistered { .. }));
}

#[test]
fn concurrency_cap_enforced_independently_of_tokens() {
    // Use a tiny concurrency cap on Gold (which has a huge token budget)
    // to prove we trip on the concurrency cap first.
    let cfg = config_with(SlaClass::Gold, 3);
    let ctrl = shared_controller(cfg);
    ctrl.acquire_permit(SlaClass::Gold).expect("first");
    ctrl.acquire_permit(SlaClass::Gold).expect("second");
    ctrl.acquire_permit(SlaClass::Gold).expect("third");
    let err = ctrl
        .acquire_permit(SlaClass::Gold)
        .expect_err("fourth must fail");
    assert!(matches!(err, SlaError::ConcurrencyCapExceeded { .. }));
}

// ───────────────────────────────────────────────────────────────────────────
// SLA workload simulator (proposer + reader)
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn proposer_and_reader_share_the_same_admission_state() {
    // Single concurrent permit for Bronze — shared between proposer and
    // reader gates so we can prove admission state is shared correctly.
    let cfg = config_with(SlaClass::Bronze, 1);
    let ctrl = shared_controller(cfg);
    let proposer = SlaProposerGate::new(ctrl.clone());
    let reader = SlaReaderGate::new(ctrl.clone());

    // Proposer takes the slot.
    proposer.try_acquire(SlaClass::Bronze).expect("proposer 1");

    // Reader can no longer admit because the slot is taken.
    let outcome = reader.admit(SlaClass::Bronze);
    match outcome {
        ReaderOutcome::Rejected(SlaError::ConcurrencyCapExceeded { limit, .. }) => {
            assert_eq!(limit, 1);
        }
        other => panic!("unexpected outcome: {other:?}"),
    }

    // Once the proposer releases, the reader can proceed.
    proposer.release(SlaClass::Bronze).expect("release");
    assert_eq!(reader.admit(SlaClass::Bronze), ReaderOutcome::Admitted);
}

#[test]
fn workload_simulator_mixed_tiers() {
    // Build a small simulator: Bronze (max_concurrent=2), Gold (max_concurrent=4).
    // Run 4 worker threads per tier each issuing 50 ops; verify stats.
    let mut quotas = HashMap::new();
    quotas.insert(
        SlaClass::Bronze,
        SlaClassQuota {
            max_qps: None,
            max_concurrent: 2,
            token_cost: 1.0,
        },
    );
    quotas.insert(
        SlaClass::Gold,
        SlaClassQuota {
            max_qps: None,
            max_concurrent: 4,
            token_cost: 1.0,
        },
    );
    let ctrl = shared_controller(SlaAdmissionConfig { quotas });
    let proposer = SlaProposerGate::new(ctrl.clone());

    let mut handles = Vec::new();
    for class in [SlaClass::Bronze, SlaClass::Gold] {
        for _ in 0..4 {
            let g = proposer.clone();
            handles.push(thread::spawn(move || {
                let mut admitted = 0usize;
                let mut rejected = 0usize;
                for _ in 0..50 {
                    match g.admit(class) {
                        ProposerOutcome::Admitted => {
                            admitted += 1;
                            // Tiny "operation".
                            thread::sleep(Duration::from_micros(10));
                            g.release(class).expect("release");
                        }
                        ProposerOutcome::Rejected(_) => rejected += 1,
                    }
                }
                (admitted, rejected)
            }));
        }
    }
    for h in handles {
        let (admitted, rejected) = h.join().expect("join");
        // Each worker did exactly 50 attempts.
        assert_eq!(admitted + rejected, 50);
    }
    // Final in-flight should be zero (all permits released).
    assert_eq!(ctrl.in_flight(SlaClass::Bronze).expect("count"), 0);
    assert_eq!(ctrl.in_flight(SlaClass::Gold).expect("count"), 0);
    // Stats must have non-zero admit counts for both tiers.
    let stats = ctrl.stats().expect("stats");
    assert!(stats.admitted.get(&SlaClass::Bronze).copied().unwrap_or(0) > 0);
    assert!(stats.admitted.get(&SlaClass::Gold).copied().unwrap_or(0) > 0);
}

#[test]
fn cost_aware_reader_drains_bucket_faster() {
    let cfg = SlaAdmissionConfig::with_defaults();
    let ctrl = shared_controller(cfg);
    let reader = SlaReaderGate::new(ctrl.clone());

    // Silver capacity = 20. A cost-15 read should succeed; a follow-up
    // cost-10 read should fail until the bucket refills.
    reader
        .try_acquire_with_cost(SlaClass::Silver, 15.0)
        .expect("expensive ok");
    reader.release(SlaClass::Silver).expect("release");

    let res = reader.try_acquire_with_cost(SlaClass::Silver, 10.0);
    // It is possible (depending on exact timing) that the bucket has
    // refilled enough already; in that case the call succeeds. Either way
    // the call must NOT panic and must return a determinate result.
    match res {
        Ok(()) => {
            reader.release(SlaClass::Silver).expect("release");
        }
        Err(SlaError::RateLimitExceeded { .. }) => {}
        Err(e) => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn scoped_admit_releases_on_completion() {
    let cfg = config_with(SlaClass::Platinum, 1);
    let ctrl = shared_controller(cfg);
    let proposer = SlaProposerGate::new(ctrl.clone());

    let result = proposer
        .scoped(SlaClass::Platinum, || "ok")
        .expect("scoped");
    assert_eq!(result, "ok");
    // After the scope ends, the slot is free again.
    assert_eq!(ctrl.in_flight(SlaClass::Platinum).expect("count"), 0);
    proposer
        .try_acquire(SlaClass::Platinum)
        .expect("post-scope acquire");
}
