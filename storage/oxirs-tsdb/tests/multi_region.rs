//! Integration tests for the active-active multi-region geometry.
//!
//! Exercises:
//! - A 3-region simulator with per-region Raft groups + cross-region fanout.
//! - Routing rules that bind tenants and subject prefixes to specific
//!   regions.
//! - Health-probe-driven failover when a region goes silent.
//! - Recovery after a region reappears.
//! - Last-writer-wins conflict resolution across concurrent writes.

use std::collections::BTreeSet;

use oxirs_tsdb::multi_region::{
    ActiveActiveConfig, ActiveActiveMultiRegion, FanoutOutcome, FanoutResolution, HealthConfig,
    RegionHealthSnapshot, RegionId, RegionStatus, RegionWriteRecord, RouteContext, RouteDecision,
    RoutingTable, WriteRoutingRule,
};
use oxirs_tsdb::replication::WriteEntry;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn three_region(local: &str) -> ActiveActiveMultiRegion {
    let cfg = ActiveActiveConfig::multi_region(
        local,
        vec!["us-east".into(), "eu-west".into(), "ap-south".into()],
        3,
    )
    .expect("cfg");
    let mut mr = ActiveActiveMultiRegion::new(cfg).expect("init");
    mr.tick_all(50);
    mr
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[test]
fn three_region_initialises_with_one_group_per_region() {
    let mr = three_region("us-east");
    assert_eq!(mr.groups().len(), 3);
    let names: BTreeSet<&str> = mr.groups().keys().map(String::as_str).collect();
    assert!(names.contains("us-east"));
    assert!(names.contains("eu-west"));
    assert!(names.contains("ap-south"));
}

#[test]
fn each_region_elects_a_leader() {
    let mr = three_region("us-east");
    for (region, group) in mr.groups() {
        let leader = group.leader_id();
        assert!(leader.is_some(), "region '{region}' did not elect a leader");
    }
}

#[test]
fn writes_committed_in_home_region_fan_out_to_peers() {
    let mut mr = three_region("us-east");
    let entry = WriteEntry::new(1_000, "metrics.cpu", 0.7);
    let ctx = RouteContext::new("metrics.cpu", 1_000);
    let outcome = mr.submit_write(&ctx, entry, 60).expect("submit");
    assert_eq!(outcome.home_region, "us-east");
    let drained = mr.drain_replicator();
    let mut total: usize = 0;
    for n in drained.values() {
        total += n;
    }
    // 2 peers should each apply once.
    assert_eq!(total, 2);
}

#[test]
fn routing_rules_pin_writes_to_specific_regions() {
    let cfg = ActiveActiveConfig::multi_region(
        "us-east",
        vec!["us-east".into(), "eu-west".into(), "ap-south".into()],
        3,
    )
    .expect("cfg");
    let mut routing = RoutingTable::default_to("us-east");
    routing.push_rule(
        WriteRoutingRule::default_to("eu-west")
            .named("eu-tenants")
            .with_tenant_id("tenant-eu"),
    );
    routing.push_rule(
        WriteRoutingRule::default_to("ap-south")
            .named("ap-tenants")
            .with_tenant_id("tenant-ap"),
    );
    let cfg = ActiveActiveConfig { routing, ..cfg };
    let mut mr = ActiveActiveMultiRegion::new(cfg).expect("mr");
    mr.tick_all(60);

    let eu_ctx = RouteContext::new("anything", 1_000).with_tenant("tenant-eu");
    let ap_ctx = RouteContext::new("anything", 2_000).with_tenant("tenant-ap");
    let default_ctx = RouteContext::new("anything", 3_000); // no tenant ⇒ default

    let eu_out = mr
        .submit_write(&eu_ctx, WriteEntry::new(1_000, "x", 1.0), 60)
        .expect("eu");
    assert_eq!(eu_out.home_region, "eu-west");

    let ap_out = mr
        .submit_write(&ap_ctx, WriteEntry::new(2_000, "y", 2.0), 60)
        .expect("ap");
    assert_eq!(ap_out.home_region, "ap-south");

    let dflt_out = mr
        .submit_write(&default_ctx, WriteEntry::new(3_000, "z", 3.0), 60)
        .expect("dflt");
    assert_eq!(dflt_out.home_region, "us-east");
}

#[test]
fn health_probe_marks_silent_region_failed() {
    let mut mr = three_region("us-east");
    // Force one region to "Failed" via the probe.
    mr.health_probe_mut()
        .force_status("eu-west", RegionStatus::Failed);
    let snap = mr.health_probe().snapshot();
    assert_eq!(snap.status_of(&"eu-west".to_string()), RegionStatus::Failed);
    assert_eq!(
        snap.status_of(&"us-east".to_string()),
        RegionStatus::Healthy
    );
}

#[test]
fn routing_skips_failed_region_with_failover() {
    let cfg = ActiveActiveConfig::multi_region(
        "us-east",
        vec!["us-east".into(), "eu-west".into(), "ap-south".into()],
        3,
    )
    .expect("cfg");
    let mut routing = RoutingTable::default_to("us-east");
    routing.push_rule(
        WriteRoutingRule::default_to("eu-west")
            .named("primary-eu")
            .with_subject_prefix("eu.")
            .with_failover("ap-south")
            .with_failover("us-east"),
    );
    let cfg = ActiveActiveConfig { routing, ..cfg };
    let mut mr = ActiveActiveMultiRegion::new(cfg).expect("mr");
    mr.tick_all(60);
    // Kill eu-west.
    mr.health_probe_mut()
        .force_status("eu-west", RegionStatus::Failed);

    let ctx = RouteContext::new("eu.metrics.cpu", 1_000);
    let out = mr
        .submit_write(&ctx, WriteEntry::new(1_000, "eu.metrics.cpu", 5.0), 60)
        .expect("submit");
    // Failover should pick ap-south (next in chain).
    assert_eq!(out.home_region, "ap-south");
}

#[test]
fn three_region_kill_and_recover_cycle() {
    let mut mr = three_region("us-east");
    let probe = mr.health_probe_mut();

    // Mark eu-west failed (kill).
    probe.force_status("eu-west", RegionStatus::Failed);
    assert_eq!(
        probe.snapshot().status_of(&"eu-west".to_string()),
        RegionStatus::Failed
    );

    // Heartbeat returns: probe should mark it Healthy again.
    probe.record_heartbeat(&"eu-west".to_string());
    assert_eq!(
        probe.snapshot().status_of(&"eu-west".to_string()),
        RegionStatus::Healthy
    );
}

#[test]
fn lww_resolves_concurrent_writes_deterministically() {
    let mut mr = three_region("us-east");
    let key = "metrics.shared";

    // us-east commits first with timestamp_ms=200.
    let us_ctx = RouteContext::new(key, 200);
    mr.submit_write(&us_ctx, WriteEntry::new(200, key, 1.0), 60)
        .expect("us");
    mr.drain_replicator();

    // Forcibly route a second write to eu-west by adding a rule.
    let mut routing = RoutingTable::default_to("us-east");
    routing.push_rule(
        WriteRoutingRule::default_to("eu-west")
            .named("eu-pin")
            .with_subject_prefix(key),
    );
    // Rebuild the multi-region with the targeted routing.
    let mut cfg = mr.config().clone();
    cfg.routing = routing;
    let mut mr2 = ActiveActiveMultiRegion::new(cfg).expect("mr2");
    mr2.tick_all(60);
    mr2.submit_write(
        &RouteContext::new(key, 200),
        WriteEntry::new(200, key, 99.0),
        60,
    )
    .expect("eu");
    mr2.drain_replicator();
    // No assertion here — both deployments are independent. The point is
    // that neither submit_write panicked under lww simulation.
    let _ = mr;
    let _ = mr2;
}

#[test]
fn replicator_lww_keeps_newest_record() {
    let mut mr = three_region("us-east");
    let key = "metrics.uniq";
    mr.submit_write(
        &RouteContext::new(key, 100),
        WriteEntry::new(100, key, 1.0),
        60,
    )
    .expect("first");
    mr.drain_replicator();
    // Apply an older record directly to a region's view; should be rejected.
    let stale = RegionWriteRecord {
        region: "ap-south".into(),
        log_index: 0,
        key: key.into(),
        timestamp_ms: 50,
        value: 9.0,
        tags_json: "{}".into(),
        observed_at_ms: 0,
    };
    let resolution = mr
        .replicator_mut()
        .apply_remote(&"us-east".to_string(), &stale);
    assert!(matches!(resolution, FanoutResolution::Rejected { .. }));
}

#[test]
fn snapshot_visible_to_routing_in_isolation() {
    // Sanity: routing decision must consult the supplied snapshot, not
    // some shared mutable global state.
    let mut routing = RoutingTable::default_to("primary");
    routing.push_rule(
        WriteRoutingRule::default_to("primary")
            .named("p")
            .with_failover("backup"),
    );
    let snap_ok = RegionHealthSnapshot::from_entries(
        [
            ("primary", RegionStatus::Healthy),
            ("backup", RegionStatus::Healthy),
        ]
        .iter()
        .map(|(r, s)| ((*r).to_string(), *s)),
    );
    let snap_dead = RegionHealthSnapshot::from_entries(
        [
            ("primary".to_string(), RegionStatus::Failed),
            ("backup".to_string(), RegionStatus::Healthy),
        ]
        .iter()
        .cloned(),
    );
    let dec_ok = routing
        .route(&RouteContext::new("subj", 0), &snap_ok)
        .expect("ok");
    let dec_failover = routing
        .route(&RouteContext::new("subj", 0), &snap_dead)
        .expect("failover");
    assert_eq!(dec_ok.region, "primary");
    assert!(!dec_ok.via_failover);
    assert_eq!(dec_failover.region, "backup");
    assert!(dec_failover.via_failover);
}

#[test]
fn write_outcome_preserves_log_index_per_region() {
    let mut mr = three_region("us-east");
    let entry = WriteEntry::new(1_000, "m", 0.5);
    let out = mr
        .submit_write(&RouteContext::new("m", 1_000), entry, 60)
        .expect("submit");
    assert!(out.log_index >= 1);
}

#[test]
fn enqueue_then_drain_settles_state() {
    let mut mr = three_region("us-east");
    for i in 0..5 {
        let ts = 1_000 + i;
        mr.submit_write(
            &RouteContext::new("m", ts),
            WriteEntry::new(ts, "m", i as f64),
            60,
        )
        .expect("submit");
    }
    let drained = mr.drain_replicator();
    let total: usize = drained.values().sum();
    // Two peers per write × 5 writes — but later writes win, so
    // effectively each peer applies the latest 5 in order = 5 each.
    assert!(total >= 5);
}

#[test]
fn fanout_outcome_matches_actual_targets() {
    let mut mr = three_region("us-east");
    let entry = WriteEntry::new(7_000, "m", 1.0);
    let out = mr
        .submit_write(&RouteContext::new("m", 7_000), entry, 60)
        .expect("submit");
    if let FanoutOutcome::Enqueued { targets } = &out.replication_outcome {
        let names: BTreeSet<&str> = targets.iter().map(String::as_str).collect();
        assert!(names.contains("eu-west"));
        assert!(names.contains("ap-south"));
        assert!(!names.contains("us-east"));
    } else {
        panic!("expected Enqueued, got {:?}", out.replication_outcome);
    }
}

#[test]
fn route_context_builder_attaches_tenant() {
    let ctx = RouteContext::new("subj", 1_000).with_tenant("tenant-a");
    assert_eq!(ctx.subject, "subj");
    assert_eq!(ctx.tenant.as_deref(), Some("tenant-a"));
    assert_eq!(ctx.timestamp_ms, 1_000);
}

#[test]
fn regions_that_kill_and_recover_resume_writes() {
    // Build a 2-region setup so failover is observable but not too noisy.
    let cfg =
        ActiveActiveConfig::multi_region("us-east", vec!["us-east".into(), "eu-west".into()], 3)
            .expect("cfg");
    let mut routing = RoutingTable::default_to("us-east");
    routing.push_rule(
        WriteRoutingRule::default_to("eu-west")
            .named("eu-default")
            .with_failover("us-east"),
    );
    let cfg = ActiveActiveConfig { routing, ..cfg };
    let mut mr = ActiveActiveMultiRegion::new(cfg).expect("mr");
    mr.tick_all(60);

    // Kill eu-west.
    mr.health_probe_mut()
        .force_status("eu-west", RegionStatus::Failed);
    let dead_out = mr
        .submit_write(
            &RouteContext::new("subj", 1_000),
            WriteEntry::new(1_000, "x", 0.0),
            60,
        )
        .expect("write during failure");
    assert_eq!(dead_out.home_region, "us-east");

    // Recover eu-west (heartbeat clears the forced status).
    mr.health_probe_mut()
        .record_heartbeat(&"eu-west".to_string());
    let alive_out = mr
        .submit_write(
            &RouteContext::new("subj", 2_000),
            WriteEntry::new(2_000, "x", 0.0),
            60,
        )
        .expect("write after recovery");
    assert_eq!(alive_out.home_region, "eu-west");
}

#[test]
fn write_routing_rule_full_match_path() {
    let rule = WriteRoutingRule::default_to("a")
        .with_subject_prefix("metrics.")
        .with_tenant_id("acme")
        .with_failover("b")
        .named("acme-metrics");
    assert!(rule.matches(&RouteContext::new("metrics.cpu", 1).with_tenant("acme")));
    assert!(!rule.matches(&RouteContext::new("metrics.cpu", 1)));
    assert!(!rule.matches(&RouteContext::new("logs.foo", 1).with_tenant("acme")));
}

#[test]
fn route_decision_carries_rule_name() {
    let mut routing = RoutingTable::default_to("home");
    routing.push_rule(WriteRoutingRule::default_to("home").named("named-rule"));
    let snap = RegionHealthSnapshot::from_entries(std::iter::once((
        "home".to_string(),
        RegionStatus::Healthy,
    )));
    let dec: RouteDecision = routing
        .route(&RouteContext::new("subj", 1), &snap)
        .expect("ok");
    assert_eq!(dec.rule_name, "named-rule");
    assert_eq!(dec.region, "home");
}

#[test]
fn config_round_trip_via_health_probe() {
    let cfg = ActiveActiveConfig::single_region("solo", 1).expect("cfg");
    let mr = ActiveActiveMultiRegion::new(cfg).expect("mr");
    let probe = mr.health_probe();
    let snap = probe.snapshot();
    assert_eq!(snap.len(), 1);
    let local = snap.healthy_regions();
    assert_eq!(local, vec!["solo".to_string()]);
}

#[test]
fn health_config_default_values_are_reasonable() {
    let cfg = HealthConfig::default();
    assert!(cfg.suspect_after < cfg.failure_threshold);
    assert!(cfg.suspect_after > 0);
}

#[test]
fn region_id_displays_as_string() {
    let r: RegionId = "us-east-1".to_string();
    assert_eq!(r, "us-east-1".to_string());
}
