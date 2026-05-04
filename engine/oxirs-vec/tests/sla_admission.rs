//! Integration tests for SLA-based resource allocation.

use oxirs_vec::multi_tenancy::{
    admission_controller::{AdmissionController, AdmissionError},
    priority_queue::SlaQueryDispatcher,
    sla::SlaClass,
    types::MultiTenancyError,
};

// ─────────────────────────────────────────────────────────────────────────────
// SlaClass / SlaThresholds tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sla_class_thresholds_ordered() {
    let bronze = SlaClass::Bronze.thresholds();
    let silver = SlaClass::Silver.thresholds();
    let gold = SlaClass::Gold.thresholds();
    let platinum = SlaClass::Platinum.thresholds();

    // Concurrency limit grows with tier
    assert!(bronze.max_concurrent_queries < silver.max_concurrent_queries);
    assert!(silver.max_concurrent_queries < gold.max_concurrent_queries);
    assert!(gold.max_concurrent_queries < platinum.max_concurrent_queries);

    // Token refill rate grows with tier
    assert!(bronze.token_refill_rate < silver.token_refill_rate);
    assert!(silver.token_refill_rate < gold.token_refill_rate);
    assert!(gold.token_refill_rate < platinum.token_refill_rate);

    // Latency budget tightens with tier
    assert!(bronze.max_latency_p99_ms > silver.max_latency_p99_ms);
    assert!(silver.max_latency_p99_ms > gold.max_latency_p99_ms);
    assert!(gold.max_latency_p99_ms > platinum.max_latency_p99_ms);

    // Bandwidth ceiling grows with tier
    assert!(bronze.bandwidth_mb_per_sec < silver.bandwidth_mb_per_sec);
    assert!(silver.bandwidth_mb_per_sec < gold.bandwidth_mb_per_sec);
    assert!(gold.bandwidth_mb_per_sec < platinum.bandwidth_mb_per_sec);
}

#[test]
fn sla_dispatch_priority_order() {
    assert!(SlaClass::Bronze.dispatch_priority() < SlaClass::Silver.dispatch_priority());
    assert!(SlaClass::Silver.dispatch_priority() < SlaClass::Gold.dispatch_priority());
    assert!(SlaClass::Gold.dispatch_priority() < SlaClass::Platinum.dispatch_priority());
}

#[test]
fn sla_class_ord_total_order() {
    let mut classes = vec![
        SlaClass::Platinum,
        SlaClass::Bronze,
        SlaClass::Gold,
        SlaClass::Silver,
    ];
    classes.sort();
    assert_eq!(
        classes,
        vec![
            SlaClass::Bronze,
            SlaClass::Silver,
            SlaClass::Gold,
            SlaClass::Platinum,
        ]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// AdmissionController tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn admission_controller_allows_within_limit() {
    let ctrl = AdmissionController::new();
    ctrl.register_tenant("t1", SlaClass::Platinum);
    // Platinum capacity=200 — many requests should succeed
    for _ in 0..10 {
        assert!(ctrl.try_admit("t1").is_ok(), "should admit within capacity");
    }
}

#[test]
fn admission_controller_rejects_unknown_tenant() {
    let ctrl = AdmissionController::new();
    let err = ctrl.try_admit("unknown").unwrap_err();
    assert!(
        matches!(err, AdmissionError::TenantNotRegistered { .. }),
        "wrong error variant: {:?}",
        err
    );
}

#[test]
fn admission_controller_exhausts_bronze_bucket() {
    let ctrl = AdmissionController::new();
    ctrl.register_tenant("bronze", SlaClass::Bronze);
    // Bronze capacity=5 → 6th+ attempt should fail
    let mut admitted = 0usize;
    let mut rejected = 0usize;
    for _ in 0..10 {
        if ctrl.try_admit("bronze").is_ok() {
            admitted += 1;
        } else {
            rejected += 1;
        }
    }
    assert!(admitted > 0, "should admit at least the initial burst");
    assert!(rejected > 0, "should reject once tokens are depleted");
}

#[test]
fn admission_controller_multiple_tenants_independent() {
    let ctrl = AdmissionController::new();
    ctrl.register_tenant("a", SlaClass::Bronze);
    ctrl.register_tenant("b", SlaClass::Platinum);

    // Exhaust Bronze
    for _ in 0..20 {
        let _ = ctrl.try_admit("a");
    }

    // Platinum should still admit
    assert!(
        ctrl.try_admit("b").is_ok(),
        "Platinum tenant should not be affected by Bronze tenant exhaustion"
    );
}

#[test]
fn admission_error_converts_to_multi_tenancy_error() {
    let rate_err = AdmissionError::RateLimitExceeded {
        tenant_id: "t".into(),
    };
    let mt: MultiTenancyError = rate_err.into();
    assert!(matches!(mt, MultiTenancyError::RateLimitExceeded { .. }));

    let reg_err = AdmissionError::TenantNotRegistered {
        tenant_id: "t".into(),
    };
    let mt2: MultiTenancyError = reg_err.into();
    assert!(matches!(mt2, MultiTenancyError::TenantNotFound { .. }));
}

// ─────────────────────────────────────────────────────────────────────────────
// SlaQueryDispatcher tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn priority_queue_platinum_dequeued_first() {
    let mut dispatcher: SlaQueryDispatcher<&str> = SlaQueryDispatcher::new();
    dispatcher.enqueue("t_bronze".into(), SlaClass::Bronze, "b");
    dispatcher.enqueue("t_gold".into(), SlaClass::Gold, "g");
    dispatcher.enqueue("t_platinum".into(), SlaClass::Platinum, "p");
    dispatcher.enqueue("t_silver".into(), SlaClass::Silver, "s");

    let first = dispatcher.dequeue().expect("non-empty");
    assert_eq!(first.payload, "p", "Platinum should be dequeued first");
    let second = dispatcher.dequeue().expect("non-empty");
    assert_eq!(second.payload, "g", "Gold should be second");
    let third = dispatcher.dequeue().expect("non-empty");
    assert_eq!(third.payload, "s", "Silver should be third");
    let fourth = dispatcher.dequeue().expect("non-empty");
    assert_eq!(fourth.payload, "b", "Bronze should be last");
}

#[test]
fn priority_queue_empty_returns_none() {
    let mut d: SlaQueryDispatcher<i32> = SlaQueryDispatcher::new();
    assert!(d.dequeue().is_none());
}

/// Simulate 10 tenants × 4 SLA classes — verify Platinum items always come
/// before Bronze items in dispatch order.
#[test]
fn sla_workload_platinum_before_bronze() {
    let mut dispatcher: SlaQueryDispatcher<usize> = SlaQueryDispatcher::new();

    for i in 0..10 {
        let class = match i % 4 {
            0 => SlaClass::Bronze,
            1 => SlaClass::Silver,
            2 => SlaClass::Gold,
            _ => SlaClass::Platinum,
        };
        dispatcher.enqueue(format!("tenant_{}", i), class, i);
    }

    // Drain all and collect in dequeue order
    let drained = dispatcher.drain_ordered();
    let priorities: Vec<u8> = drained.iter().map(|q| q.priority).collect();

    // Must be non-increasing (max-heap)
    for window in priorities.windows(2) {
        assert!(
            window[0] >= window[1],
            "priorities must be non-increasing: {:?}",
            priorities
        );
    }

    // At least one Platinum (priority=4) must precede any Bronze (priority=1)
    let first_platinum = drained.iter().position(|q| q.priority == 4);
    let first_bronze = drained.iter().position(|q| q.priority == 1);
    if let (Some(p), Some(b)) = (first_platinum, first_bronze) {
        assert!(
            p < b,
            "first Platinum ({}) must precede first Bronze ({})",
            p,
            b
        );
    }
}
