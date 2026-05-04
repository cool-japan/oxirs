//! Scalability integration tests: gossip fanout + hash-ring + sim cluster
//! (1000-node scaling phase A).
//!
//! These tests verify the correctness of:
//! - `GossipFanout` policy enum (`Bounded`, `Sqrt`, `Unbounded`)
//! - `SimCluster` construction at 10 and 1000 nodes
//! - `AntiEntropyThrottle` concurrency gate
//!
//! All tests run with `--all-features` (includes the `simulation` feature).

use oxirs_cluster::anti_entropy::AntiEntropyThrottle;
use oxirs_cluster::gossip::fanout::GossipFanout;
use oxirs_cluster::simulation::SimCluster;

// ── SimCluster tests ─────────────────────────────────────────────────────────

#[test]
fn test_sim_cluster_10_nodes_constructs() {
    let cluster = SimCluster::new(10);
    assert_eq!(cluster.size(), 10);
}

#[test]
fn test_sim_cluster_1000_nodes_constructs() {
    let cluster = SimCluster::new(1000);
    assert_eq!(cluster.size(), 1000);
}

#[tokio::test]
async fn test_sim_cluster_gossip_delivery_succeeds() {
    let cluster = SimCluster::new(5);
    let result = cluster.gossip(0, 4, b"test payload".to_vec()).await;
    assert!(result.is_ok(), "gossip should succeed: {:?}", result);
}

#[tokio::test]
async fn test_sim_cluster_gossip_unknown_node_fails() {
    let cluster = SimCluster::new(3);
    let result = cluster.gossip(0, 999, b"oops".to_vec()).await;
    assert!(result.is_err(), "gossip to unknown node should return Err");
}

// ── GossipFanout tests ───────────────────────────────────────────────────────

#[test]
fn test_gossip_fanout_bounded() {
    let f = GossipFanout::Bounded(5);
    assert_eq!(f.resolve(1000), 5);
}

#[test]
fn test_gossip_fanout_bounded_capped_by_n() {
    let f = GossipFanout::Bounded(100);
    // cluster smaller than bound → capped at cluster size
    assert_eq!(f.resolve(10), 10);
}

#[test]
fn test_gossip_fanout_sqrt() {
    let f = GossipFanout::Sqrt;
    assert_eq!(f.resolve(100), 10);
    // floor(sqrt(1000)) == 31
    assert_eq!(f.resolve(1000), 31);
}

#[test]
fn test_gossip_fanout_sqrt_small() {
    let f = GossipFanout::Sqrt;
    assert_eq!(f.resolve(4), 2);
    assert_eq!(f.resolve(1), 1);
}

#[test]
fn test_gossip_fanout_unbounded() {
    let f = GossipFanout::Unbounded;
    assert_eq!(f.resolve(1000), 1000);
    assert_eq!(f.resolve(0), 0);
}

#[test]
fn test_gossip_fanout_default_for_large_cluster() {
    assert_eq!(GossipFanout::default_for(1000), GossipFanout::Sqrt);
    assert_eq!(GossipFanout::default_for(33), GossipFanout::Sqrt);
}

#[test]
fn test_gossip_fanout_default_for_small_cluster() {
    assert_eq!(GossipFanout::default_for(10), GossipFanout::Unbounded);
    assert_eq!(GossipFanout::default_for(32), GossipFanout::Unbounded);
}

// ── AntiEntropyThrottle tests ────────────────────────────────────────────────

#[tokio::test]
async fn test_anti_entropy_throttle_default_limit() {
    let t = AntiEntropyThrottle::with_default_limit();
    assert_eq!(t.max_concurrent_syncs, 4);
}

#[tokio::test]
async fn test_anti_entropy_throttle_limits_concurrency() {
    let t = AntiEntropyThrottle::new(2);
    let _p1 = t.acquire().await.expect("first permit");
    let _p2 = t.acquire().await.expect("second permit");
    // Third should fail immediately via try_acquire
    assert!(t.try_acquire().is_none(), "no permit should be available");
}

#[tokio::test]
async fn test_anti_entropy_throttle_releases_on_drop() {
    let t = AntiEntropyThrottle::new(1);
    {
        let _p = t.acquire().await.expect("permit");
    } // permit drops here
      // Slot now free
    assert!(t.try_acquire().is_some());
}

#[tokio::test]
async fn test_anti_entropy_throttle_available_permits() {
    let t = AntiEntropyThrottle::new(3);
    assert_eq!(t.available_permits(), 3);
    let _p = t.try_acquire().expect("one permit");
    assert_eq!(t.available_permits(), 2);
}
