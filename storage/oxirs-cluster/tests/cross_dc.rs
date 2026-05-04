//! Integration tests for the active-active geo replication geometry.
//!
//! Exercises the public API exposed by `oxirs_cluster::replication::*`:
//!
//! * `ActiveActiveGeoConfig` + `ActiveActiveReplicator`
//! * `RegionFailoverController` (failover state machine)
//! * `CrossRegionAntiEntropy` (Merkle-driven divergence detection)

use std::sync::Arc;
use std::time::Duration;

use oxirs_cluster::conflict_resolution::VectorClock;
use oxirs_cluster::cross_dc::ConsistencyLevel;
use oxirs_cluster::merkle_tree::MerkleTree;
use oxirs_cluster::replication::{
    active_active_geo::current_timestamp_ms, ActiveActiveGeoConfig, ActiveActiveReplicator,
    ConflictResolutionMode, CrossRegionAntiEntropy, GeoWriteOutcome, RegionFailoverController,
    RegionRaftGroup, RegionRoutingRule,
};

// ───────────────────────────────────────────────────────────────────────────
// Three-region cluster simulator
// ───────────────────────────────────────────────────────────────────────────

fn three_region_setup() -> (
    ActiveActiveReplicator,
    RegionFailoverController,
    CrossRegionAntiEntropy,
) {
    let regions = vec![
        "us-east-1".to_string(),
        "eu-west-1".to_string(),
        "ap-northeast-1".to_string(),
    ];
    let mut config = ActiveActiveGeoConfig::multi_region("us-east-1", regions.clone());
    // Add an extra rule that routes EU subjects to eu-west-1.
    config.routing_rules.insert(
        0,
        RegionRoutingRule {
            name: "eu-tenant".into(),
            subject_prefix: Some("https://eu.example.org/".into()),
            tenant_id: None,
            target_region: "eu-west-1".into(),
            consistency: ConsistencyLevel::EachQuorum,
        },
    );
    let groups = regions
        .iter()
        .enumerate()
        .map(|(i, r)| RegionRaftGroup::new(r.clone(), 7000 + i as u64, vec![1, 2, 3]))
        .collect();
    let replicator = ActiveActiveReplicator::new(config.clone(), groups).expect("replicator");
    let failover = RegionFailoverController::new(config);
    let ae = CrossRegionAntiEntropy::new("us-east-1");
    (replicator, failover, ae)
}

#[test]
fn multi_region_default_routing_uses_local_region() {
    let (replicator, _failover, _ae) = three_region_setup();
    let decision = replicator.config().route("http://example.org/s", None);
    assert_eq!(decision.primary_region, "us-east-1");
    assert_eq!(decision.fanout_regions.len(), 2);
}

#[test]
fn multi_region_subject_prefix_routes_to_eu() {
    let (replicator, _failover, _ae) = three_region_setup();
    let decision = replicator
        .config()
        .route("https://eu.example.org/foo", None);
    assert_eq!(decision.primary_region, "eu-west-1");
    assert_eq!(decision.consistency, ConsistencyLevel::EachQuorum);
}

#[test]
fn fanout_drains_pending_entries_per_target() {
    let (replicator, _failover, _ae) = three_region_setup();
    let us = "us-east-1".to_string();
    replicator
        .apply_local_write(&us, "k", "v", current_timestamp_ms(), VectorClock::new())
        .expect("apply");
    let drained_eu = replicator
        .drain_fanout(&us, &"eu-west-1".to_string())
        .expect("drain eu");
    assert_eq!(drained_eu.len(), 1);
    let drained_ap = replicator
        .drain_fanout(&us, &"ap-northeast-1".to_string())
        .expect("drain ap");
    assert_eq!(drained_ap.len(), 1);
    // After both targets drained, queue is empty.
    assert_eq!(replicator.pending_fanout_len(&us).expect("pending"), 0);
}

#[test]
fn lww_resolves_concurrent_regional_writes() {
    let (replicator, _failover, _ae) = three_region_setup();
    let us = "us-east-1".to_string();
    let eu = "eu-west-1".to_string();
    replicator
        .apply_local_write(&us, "k", "from-us", 100, VectorClock::new())
        .expect("local");
    let outcome = replicator
        .apply_remote_write(&eu, "k", "from-eu", 200, VectorClock::new())
        .expect("remote");
    assert!(matches!(outcome, GeoWriteOutcome::Committed { .. }));
    let rec = replicator.get_record("k").expect("get").expect("record");
    assert_eq!(rec.value, "from-eu");
    assert_eq!(rec.region, "eu-west-1");
}

#[test]
fn vector_clock_mode_accepts_strictly_later_writes() {
    let mut config = ActiveActiveGeoConfig::multi_region(
        "us-east-1",
        vec![
            "us-east-1".into(),
            "eu-west-1".into(),
            "ap-northeast-1".into(),
        ],
    );
    config.conflict_mode = ConflictResolutionMode::VectorClock;
    let groups = config
        .regions
        .iter()
        .enumerate()
        .map(|(i, r)| RegionRaftGroup::new(r.clone(), 9000 + i as u64, vec![1, 2, 3]))
        .collect();
    let replicator = ActiveActiveReplicator::new(config, groups).expect("replicator");

    let mut clk = VectorClock::new();
    clk.increment(1);
    replicator
        .apply_local_write(&"us-east-1".into(), "k", "v1", 100, clk.clone())
        .expect("first");

    let mut clk2 = clk.clone();
    clk2.increment(2);
    let outcome = replicator
        .apply_remote_write(&"eu-west-1".into(), "k", "v2", 50, clk2)
        .expect("second");
    // Strictly happens-after: accepted regardless of older timestamp.
    assert!(matches!(outcome, GeoWriteOutcome::Committed { .. }));
}

// ───────────────────────────────────────────────────────────────────────────
// Region failover scenario
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn kill_primary_region_promotes_secondary_and_replays_writes() {
    let (replicator, failover, _ae) = three_region_setup();

    // Apply a few local writes in us-east-1, capturing the fanout pending
    // for eu-west-1 — we will use those as the "outstanding writes" the
    // replicator never managed to ship before us-east-1 went down.
    let us = "us-east-1".to_string();
    let eu = "eu-west-1".to_string();
    for i in 0..5 {
        replicator
            .apply_local_write(
                &us,
                &format!("k{i}"),
                &format!("v{i}"),
                current_timestamp_ms() + i,
                VectorClock::new(),
            )
            .expect("apply");
    }
    let outstanding = replicator
        .drain_fanout(&us, &eu)
        .expect("drain eu fanout")
        .into_iter()
        .map(
            |(key, record, seq)| oxirs_cluster::replication::region_failover::OutstandingWrite {
                origin_region: us.clone(),
                seq,
                key,
                value: record.value,
            },
        )
        .collect::<Vec<_>>();
    assert_eq!(outstanding.len(), 5);

    // Buffer them for replay to eu-west-1 once it takes over.
    failover
        .buffer_replay_writes(&eu, outstanding.clone())
        .expect("buffer");

    // Kill us-east-1 — controller demotes it and promotes a secondary.
    let (failed, promoted) = failover.demote_and_promote(&us).expect("failover");
    assert_eq!(failed, "us-east-1");
    assert_ne!(promoted, "us-east-1");
    assert!(["eu-west-1", "ap-northeast-1"].contains(&promoted.as_str()));

    // Replay outstanding writes on the takeover region.
    let drained = failover.replay_outstanding(&eu).expect("replay");
    assert_eq!(drained.len(), 5);

    // Bring us-east-1 back online — readmit it as Secondary.
    failover.heartbeat(&us).expect("heartbeat");
    failover.readmit(&us).expect("readmit");
    assert_eq!(
        failover.role(&us).expect("role"),
        oxirs_cluster::replication::RegionRole::Secondary
    );
}

#[test]
fn failover_history_records_full_lifecycle() {
    let (_replicator, failover, _ae) = three_region_setup();
    failover
        .demote_and_promote(&"us-east-1".to_string())
        .expect("failover");
    failover
        .heartbeat(&"us-east-1".to_string())
        .expect("heartbeat");
    failover.readmit(&"us-east-1".to_string()).expect("readmit");
    let hist = failover.history().expect("history");
    use oxirs_cluster::replication::FailoverEvent;
    assert!(hist
        .iter()
        .any(|e| matches!(e, FailoverEvent::Demoted { .. })));
    assert!(hist
        .iter()
        .any(|e| matches!(e, FailoverEvent::Promoted { .. })));
    assert!(hist
        .iter()
        .any(|e| matches!(e, FailoverEvent::Readmitted { .. })));
}

// ───────────────────────────────────────────────────────────────────────────
// Cross-region anti-entropy
// ───────────────────────────────────────────────────────────────────────────

async fn populated_tree(pairs: &[(&str, &str)]) -> Arc<MerkleTree> {
    let tree = Arc::new(MerkleTree::new());
    for (k, v) in pairs {
        tree.insert(k.to_string(), v).await;
    }
    tree
}

#[tokio::test]
async fn cross_region_merkle_compares_three_regions() {
    let ae = CrossRegionAntiEntropy::new("us-east-1");
    ae.set_local_tree(populated_tree(&[("k1", "v1"), ("k2", "v2"), ("k3", "v3")]).await)
        .await;
    // EU has the same data as US — should be in sync.
    ae.upsert_peer_tree(
        "eu-west-1".into(),
        populated_tree(&[("k1", "v1"), ("k2", "v2"), ("k3", "v3")]).await,
    )
    .await
    .expect("upsert eu");
    // AP is missing k3 — should diverge with one missing key.
    ae.upsert_peer_tree(
        "ap-northeast-1".into(),
        populated_tree(&[("k1", "v1"), ("k2", "v2")]).await,
    )
    .await
    .expect("upsert ap");

    let report = ae.compare_all().await.expect("compare_all");
    assert_eq!(report.local_region, "us-east-1");
    assert!(report.per_peer[&"eu-west-1".to_string()].is_in_sync());
    assert!(!report.per_peer[&"ap-northeast-1".to_string()].is_in_sync());
    assert_eq!(
        report.per_peer[&"ap-northeast-1".to_string()].keys_missing_on_peer,
        ["k3".to_string()].into_iter().collect()
    );
    assert!(!report.is_in_sync());
    let divergent: Vec<_> = report.divergent_peers().cloned().collect();
    assert_eq!(divergent, vec!["ap-northeast-1".to_string()]);
}

#[tokio::test]
async fn cross_region_merkle_detects_conflicts() {
    let ae = CrossRegionAntiEntropy::new("us-east-1");
    ae.set_local_tree(populated_tree(&[("k1", "v-us")]).await)
        .await;
    // EU has k1 with a different value — must surface as a conflict.
    ae.upsert_peer_tree("eu-west-1".into(), populated_tree(&[("k1", "v-eu")]).await)
        .await
        .expect("upsert");
    let div = ae.compare_with(&"eu-west-1".into()).await.expect("compare");
    assert!(div.keys_with_conflicts.contains("k1"));
    assert!(div.keys_missing_on_peer.is_empty());
    assert!(div.keys_missing_locally.is_empty());
    assert_eq!(div.total_divergent_keys(), 1);
}

#[tokio::test]
async fn cross_region_merkle_after_full_sync_round_trip() {
    let ae = CrossRegionAntiEntropy::new("us-east-1");
    let local = Arc::new(MerkleTree::new());
    local.insert("a".into(), "1").await;
    ae.set_local_tree(local.clone()).await;

    // Initially the peer has nothing, so we're divergent.
    let peer = Arc::new(MerkleTree::new());
    ae.upsert_peer_tree("eu-west-1".into(), peer.clone())
        .await
        .expect("upsert");
    let div_before = ae.compare_with(&"eu-west-1".into()).await.expect("compare");
    assert_eq!(div_before.total_divergent_keys(), 1);

    // Simulate a sync: peer applies the missing key.
    peer.insert("a".into(), "1").await;
    let div_after = ae.compare_with(&"eu-west-1".into()).await.expect("compare");
    assert!(div_after.is_in_sync());
}

// ───────────────────────────────────────────────────────────────────────────
// Smoke test that the controller's tick / heartbeat semantics work
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn quiet_region_eventually_marked_suspect() {
    let cfg = ActiveActiveGeoConfig::multi_region(
        "us-east-1",
        vec!["us-east-1".into(), "eu-west-1".into()],
    );
    let failover = RegionFailoverController::with_options(cfg, Duration::from_millis(20), 32);
    std::thread::sleep(Duration::from_millis(40));
    let events = failover.tick().expect("tick");
    assert!(!events.is_empty());
}
