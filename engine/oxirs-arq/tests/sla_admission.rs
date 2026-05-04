//! Integration tests for the W2-S4 SLA admission control wiring.
//!
//! These tests cover:
//!
//! 1. Multi-tenant simulation with mixed SLA classes — Platinum traffic
//!    completes despite Bronze contention.
//! 2. Strict admission converts a depleted bucket into [`ArqSlaError::SlaExceeded`].
//! 3. Soft admission lets canary tenants bypass admission limits.
//! 4. Priority dispatcher returns higher-tier queries before lower tiers
//!    even when enqueue order is reversed.

use std::time::Duration;

use oxirs_arq::sla_integration::{ArqSlaError, ArqSlaGate};
use oxirs_arq::tenant_config::{TenantConfig, TenantConfigRegistry};
use oxirs_core::sla::SlaClass;

#[test]
fn smoke_multi_tenant_simulator_ten_tenants_four_classes() {
    // Ten tenants split across the four SLA classes.
    let registry = TenantConfigRegistry::new();
    let tenants = vec![
        ("free-1", SlaClass::Bronze),
        ("free-2", SlaClass::Bronze),
        ("std-1", SlaClass::Silver),
        ("std-2", SlaClass::Silver),
        ("std-3", SlaClass::Silver),
        ("pro-1", SlaClass::Gold),
        ("pro-2", SlaClass::Gold),
        ("pro-3", SlaClass::Gold),
        ("vip-1", SlaClass::Platinum),
        ("vip-2", SlaClass::Platinum),
    ];
    for (id, class) in &tenants {
        registry.register(TenantConfig::new(*id, *class));
    }

    let gate = ArqSlaGate::new(registry);

    // Hammer admit() once per tenant, then count survivors.
    let mut admitted_per_tenant = std::collections::HashMap::new();
    for (id, _) in &tenants {
        let mut count = 0usize;
        // Try up to 50 admits; capacity differs by tier so this oversamples
        // intentionally.
        for _ in 0..50 {
            if gate.admit(id, ()).is_ok() {
                count += 1;
            } else {
                break;
            }
        }
        admitted_per_tenant.insert(*id, count);
    }

    // Platinum tenants should get more admits than Bronze.
    let platinum: usize = ["vip-1", "vip-2"]
        .iter()
        .map(|t| admitted_per_tenant[t])
        .sum();
    let bronze: usize = ["free-1", "free-2"]
        .iter()
        .map(|t| admitted_per_tenant[t])
        .sum();
    assert!(
        platinum > bronze,
        "Platinum tenants ({platinum}) must out-admit Bronze ({bronze})"
    );
}

#[test]
fn strict_admission_rejects_after_bucket_depletion() {
    let registry = TenantConfigRegistry::new();
    registry.register(TenantConfig::new("starved", SlaClass::Bronze));

    let gate = ArqSlaGate::new(registry);

    // Drain the bucket
    let mut last_err = None;
    for _ in 0..20 {
        if let Err(err) = gate.admit("starved", ()) {
            last_err = Some(err);
            break;
        }
    }

    let err = last_err.expect("Bronze tenant must eventually be rejected");
    match err {
        ArqSlaError::SlaExceeded {
            tenant_id, class, ..
        } => {
            assert_eq!(tenant_id, "starved");
            assert_eq!(class.0, SlaClass::Bronze);
        }
        other => panic!("expected SlaExceeded, got {other:?}"),
    }
}

#[test]
fn soft_admission_allows_canary_tenants_through() {
    let registry = TenantConfigRegistry::new();
    registry.register(TenantConfig::new("canary", SlaClass::Bronze).with_strict_admission(false));
    let gate = ArqSlaGate::new(registry);

    // 100 admit calls in a row should all succeed in soft mode.
    for _ in 0..100 {
        gate.admit("canary", ()).expect("soft admission must pass");
    }
}

#[test]
fn dispatcher_orders_by_class_regardless_of_enqueue_order() {
    let registry = TenantConfigRegistry::new();
    registry.register(TenantConfig::new("vip", SlaClass::Platinum));
    registry.register(TenantConfig::new("std", SlaClass::Silver));
    registry.register(TenantConfig::new("free", SlaClass::Bronze));
    let gate = ArqSlaGate::new(registry);

    // Enqueue lowest tier first.
    let bronze = gate.admit("free", ()).expect("admit free");
    let silver = gate.admit("std", ()).expect("admit std");
    let platinum = gate.admit("vip", ()).expect("admit vip");

    gate.enqueue(&bronze, "bronze-task");
    gate.enqueue(&silver, "silver-task");
    gate.enqueue(&platinum, "platinum-task");

    // Dispatcher should hand back platinum, silver, bronze in that order.
    let first = gate.next_dispatch().expect("first dequeue");
    assert_eq!(first.tenant_id, "vip");
    let second = gate.next_dispatch().expect("second dequeue");
    assert_eq!(second.tenant_id, "std");
    let third = gate.next_dispatch().expect("third dequeue");
    assert_eq!(third.tenant_id, "free");
    assert!(gate.next_dispatch().is_none());
}

#[test]
fn unregistered_tenant_returns_typed_error() {
    let gate = ArqSlaGate::new(TenantConfigRegistry::new());
    let err = gate.admit("nobody", ()).expect_err("must reject unknown");
    match err {
        ArqSlaError::TenantNotRegistered { tenant_id } => assert_eq!(tenant_id, "nobody"),
        other => panic!("expected TenantNotRegistered, got {other:?}"),
    }
}

#[test]
fn admit_carries_per_tenant_timeout() {
    let registry = TenantConfigRegistry::new();
    registry.register(
        TenantConfig::new("timed", SlaClass::Gold).with_query_timeout(Duration::from_millis(120)),
    );
    let gate = ArqSlaGate::new(registry);
    let admitted = gate.admit("timed", ()).expect("admit");
    assert_eq!(admitted.timeout, Some(Duration::from_millis(120)));
}

#[test]
fn registry_clone_shares_underlying_state() {
    let r1 = TenantConfigRegistry::new();
    let r2 = r1.clone();
    r1.register(TenantConfig::new("shared", SlaClass::Gold));
    assert_eq!(r2.sla_class("shared"), Some(SlaClass::Gold));
    assert_eq!(r1.len(), r2.len());
}

#[test]
fn gate_register_tenant_propagates_to_admission_controller() {
    let gate = ArqSlaGate::new(TenantConfigRegistry::new());
    gate.register_tenant(TenantConfig::new("hot-add", SlaClass::Platinum));

    let admitted = gate
        .admit("hot-add", ())
        .expect("hot-added tenant must admit");
    assert_eq!(admitted.sla_class, SlaClass::Platinum);
}

#[test]
fn pending_count_tracks_dispatcher_state() {
    let registry = TenantConfigRegistry::new();
    registry.register(TenantConfig::new("counter", SlaClass::Gold));
    let gate = ArqSlaGate::new(registry);
    let admitted = gate.admit("counter", ()).expect("admit");
    assert_eq!(gate.pending_count(), 0);
    gate.enqueue(&admitted, "task-1");
    gate.enqueue(&admitted, "task-2");
    gate.enqueue(&admitted, "task-3");
    assert_eq!(gate.pending_count(), 3);
    let _ = gate.next_dispatch();
    let _ = gate.next_dispatch();
    let _ = gate.next_dispatch();
    assert_eq!(gate.pending_count(), 0);
}
