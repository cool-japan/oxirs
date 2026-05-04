//! Phase B integration tests: hierarchical log-replication topology + witness nodes.
//!
//! Topology tests (12): single-node, small cluster, 1000-node, hop distances,
//! downstream/upstream queries, message bound, AZ locality, witness marking.
//!
//! Witness node tests (8): vote grant/deny logic, stale term rejection,
//! append-entries success/rejection, tail eviction, commit-index advance.

use oxirs_cluster::{
    NodeDescriptor, ReplicationRole, ReplicationTopology, VoteRequest, WitnessAppendRequest,
    WitnessLogEntry, WitnessNode,
};

// ═════════════════════════════════════════════════════════════════════════════
// Helper builders
// ═════════════════════════════════════════════════════════════════════════════

fn make_members(count: usize) -> Vec<NodeDescriptor> {
    (0..count)
        .map(|i| {
            let az = format!("az-{}", i % 4);
            NodeDescriptor::full_member(&format!("node-{i}"), &az, "us-east-1")
        })
        .collect()
}

fn make_witness_entry(index: u64, term: u64) -> WitnessLogEntry {
    WitnessLogEntry {
        index,
        term,
        checksum: index ^ term,
    }
}

fn append_req(
    term: u64,
    prev_index: u64,
    prev_term: u64,
    entries: Vec<WitnessLogEntry>,
    leader_commit: u64,
) -> WitnessAppendRequest {
    WitnessAppendRequest {
        leader_id: "leader".to_owned(),
        term,
        prev_log_index: prev_index,
        prev_log_term: prev_term,
        entries,
        leader_commit,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Topology tests
// ═════════════════════════════════════════════════════════════════════════════

/// Single member → topology has primary + that member as the sole relay.
#[test]
fn test_build_single_node_topology() {
    let members = vec![NodeDescriptor::full_member("n1", "az-a", "r1")];
    let topo = ReplicationTopology::build("primary", &members);

    // Total nodes: primary + 1 member
    assert_eq!(topo.len(), 2);
    // With 1 member, relay_count = ceil(sqrt(1)) = 1
    assert_eq!(topo.relay_count(), 1);
    let primary_downstream = topo.downstream_of("primary");
    assert_eq!(primary_downstream.len(), 1);
    assert_eq!(primary_downstream[0], "n1");
}

/// 4 members → relay_count = ceil(sqrt(4)) = 2.
#[test]
fn test_build_small_cluster_topology() {
    // 4 members across 2 AZs — one relay per AZ.
    let members = vec![
        NodeDescriptor::full_member("n0", "az-a", "r1"),
        NodeDescriptor::full_member("n1", "az-a", "r1"),
        NodeDescriptor::full_member("n2", "az-b", "r1"),
        NodeDescriptor::full_member("n3", "az-b", "r1"),
    ];
    let topo = ReplicationTopology::build("primary", &members);

    assert_eq!(topo.relay_count(), 2, "expected 2 relays for 4 members");
    // Primary fans out to the 2 relays
    assert_eq!(topo.downstream_of("primary").len(), 2);
    // Total = primary + 4 members
    assert_eq!(topo.len(), 5);
}

/// 1000 members → relay_count ≈ √1000 ≈ 32 (accept 30–33).
#[test]
fn test_build_1000_node_topology() {
    let members = make_members(1000);
    let topo = ReplicationTopology::build("primary", &members);

    let r = topo.relay_count();
    assert!(
        (30..=33).contains(&r),
        "expected ~31-32 relays for 1000 nodes, got {r}"
    );
    // All 1001 nodes present
    assert_eq!(topo.len(), 1001);
}

/// Primary has hop distance 0.
#[test]
fn test_topology_hop_distance_primary_is_zero() {
    let members = make_members(5);
    let topo = ReplicationTopology::build("primary", &members);
    assert_eq!(topo.hop_distance("primary"), 0);
}

/// Relay nodes have hop distance 1.
#[test]
fn test_topology_hop_distance_relay_is_one() {
    let members = make_members(10);
    let topo = ReplicationTopology::build("primary", &members);

    for relay_id in topo.relay_ids() {
        assert_eq!(
            topo.hop_distance(relay_id),
            1,
            "relay {relay_id} should be 1 hop"
        );
    }
}

/// Leaf nodes have hop distance 2.
#[test]
fn test_topology_hop_distance_leaf_is_two() {
    let members = make_members(10);
    let topo = ReplicationTopology::build("primary", &members);
    let relay_set: std::collections::HashSet<&str> =
        topo.relay_ids().iter().map(|s| s.as_str()).collect();

    for member in &members {
        if !relay_set.contains(member.node_id.as_str()) {
            assert_eq!(
                topo.hop_distance(&member.node_id),
                2,
                "leaf {} should be 2 hops",
                member.node_id
            );
        }
    }
}

/// Primary's downstream list contains exactly the relay nodes.
#[test]
fn test_topology_downstream_of_primary_returns_relays() {
    let members = make_members(20);
    let topo = ReplicationTopology::build("primary", &members);

    let downstream: std::collections::HashSet<&str> = topo
        .downstream_of("primary")
        .iter()
        .map(|s| s.as_str())
        .collect();
    let relays: std::collections::HashSet<&str> =
        topo.relay_ids().iter().map(|s| s.as_str()).collect();
    assert_eq!(downstream, relays);
}

/// Each relay fans out to at least one leaf/witness (for non-trivial clusters).
#[test]
fn test_topology_downstream_of_relay_returns_leaves() {
    let members = make_members(20);
    let topo = ReplicationTopology::build("primary", &members);

    // At least some relays should have downstream nodes when n > relay_count
    let total_downstream: usize = topo
        .relay_ids()
        .iter()
        .map(|r| topo.downstream_of(r).len())
        .sum();
    assert!(
        total_downstream > 0,
        "expected relays to have downstream leaves for 20-node cluster"
    );
}

/// Leaf/witness nodes have no downstream.
#[test]
fn test_topology_downstream_of_leaf_is_empty() {
    let members = make_members(10);
    let topo = ReplicationTopology::build("primary", &members);
    let relay_set: std::collections::HashSet<&str> =
        topo.relay_ids().iter().map(|s| s.as_str()).collect();

    for member in &members {
        if !relay_set.contains(member.node_id.as_str()) {
            assert!(
                topo.downstream_of(&member.node_id).is_empty(),
                "leaf {} should have no downstream",
                member.node_id
            );
        }
    }
}

/// For 1000 nodes, max_messages_per_entry << 1000 (the whole point of Phase B).
#[test]
fn test_topology_max_messages_less_than_n() {
    let members = make_members(1000);
    let topo = ReplicationTopology::build("primary", &members);
    let max_msgs = topo.max_messages_per_entry();
    assert!(
        max_msgs < 1000,
        "max_messages_per_entry={max_msgs} should be < 1000 (O(N) bound violated)"
    );
}

/// Nodes in the same AZ should be assigned to the same relay.
#[test]
fn test_topology_az_locality() {
    // 100 nodes across 4 AZs — well within the relay budget.
    let members: Vec<NodeDescriptor> = (0..100)
        .map(|i| {
            let az = format!("az-{}", i % 4); // 4 AZs → 4 relays at most
            NodeDescriptor::full_member(&format!("node-{i}"), &az, "us-east-1")
        })
        .collect();
    let topo = ReplicationTopology::build("primary", &members);

    // For every pair of non-relay nodes in the same AZ, their upstream relay should match.
    let relay_set: std::collections::HashSet<&str> =
        topo.relay_ids().iter().map(|s| s.as_str()).collect();

    let mut az_to_relay: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for member in &members {
        if relay_set.contains(member.node_id.as_str()) {
            continue; // relays themselves
        }
        if let Some(upstream) = topo.upstream_of(&member.node_id) {
            let prev = az_to_relay.entry(member.az.as_str()).or_insert(upstream);
            assert_eq!(
                *prev, upstream,
                "nodes in AZ '{}' have different relays: {} vs {}",
                member.az, prev, upstream
            );
        }
    }
}

/// Witness descriptors produce nodes with `ReplicationRole::Witness`.
#[test]
fn test_topology_witness_nodes_are_witnesses() {
    let members = vec![
        NodeDescriptor::full_member("relay1", "az-a", "r1"),
        NodeDescriptor::full_member("leaf1", "az-a", "r1"),
        NodeDescriptor::witness("wit1", "az-a", "r1", 50),
        NodeDescriptor::witness("wit2", "az-b", "r1", 100),
    ];
    let topo = ReplicationTopology::build("primary", &members);

    let wit1 = topo.node("wit1").expect("wit1 should be in topology");
    let wit2 = topo.node("wit2").expect("wit2 should be in topology");

    assert!(
        matches!(wit1.role, ReplicationRole::Witness { tail_window: 50 }),
        "wit1 should be Witness{{tail_window:50}}, got {:?}",
        wit1.role
    );
    assert!(
        matches!(wit2.role, ReplicationRole::Witness { tail_window: 100 }),
        "wit2 should be Witness{{tail_window:100}}, got {:?}",
        wit2.role
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Witness node tests
// ═════════════════════════════════════════════════════════════════════════════

/// A fresh witness grants its first vote in term 1.
#[test]
fn test_witness_grants_vote_first_request() {
    let mut witness = WitnessNode::new("w0", 10);
    let req = VoteRequest {
        candidate_id: "c1".to_owned(),
        term: 1,
        last_log_index: 0,
        last_log_term: 0,
    };
    let resp = witness.handle_vote_request(&req);
    assert!(resp.vote_granted, "should grant first vote");
    assert_eq!(resp.term, 1);
}

/// A witness rejects a second vote request for a *different* candidate in the same term.
#[test]
fn test_witness_denies_duplicate_vote_same_term() {
    let mut witness = WitnessNode::new("w0", 10);

    let req1 = VoteRequest {
        candidate_id: "c1".to_owned(),
        term: 2,
        last_log_index: 0,
        last_log_term: 0,
    };
    let resp1 = witness.handle_vote_request(&req1);
    assert!(resp1.vote_granted);

    // Same term, different candidate → denied
    let req2 = VoteRequest {
        candidate_id: "c2".to_owned(),
        term: 2,
        last_log_index: 0,
        last_log_term: 0,
    };
    let resp2 = witness.handle_vote_request(&req2);
    assert!(
        !resp2.vote_granted,
        "second candidate in same term should be denied"
    );
}

/// A vote request with a stale term is rejected.
#[test]
fn test_witness_denies_stale_term() {
    let mut witness = WitnessNode::new("w0", 10);

    // Advance to term 5 via append entries
    let req = WitnessAppendRequest {
        leader_id: "leader".to_owned(),
        term: 5,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };
    witness.handle_append_entries(&req);
    assert_eq!(witness.current_term(), 5);

    // Now try to vote for term 3 (stale)
    let vote_req = VoteRequest {
        candidate_id: "c1".to_owned(),
        term: 3,
        last_log_index: 0,
        last_log_term: 0,
    };
    let resp = witness.handle_vote_request(&vote_req);
    assert!(!resp.vote_granted, "stale-term vote should be denied");
    assert_eq!(resp.term, 5);
}

/// A valid AppendEntries with entries is accepted.
#[test]
fn test_witness_append_entries_success() {
    let mut witness = WitnessNode::new("w0", 10);

    let req = append_req(
        1,
        0,
        0,
        vec![
            make_witness_entry(1, 1),
            make_witness_entry(2, 1),
            make_witness_entry(3, 1),
        ],
        2,
    );
    let resp = witness.handle_append_entries(&req);

    assert!(resp.success, "valid append should succeed");
    assert!(resp.in_window);
    assert_eq!(resp.match_index, 3);
    assert_eq!(witness.commit_index(), 2);
    assert_eq!(witness.tail_len(), 3);
}

/// An AppendEntries from a stale leader (term < current_term) is rejected.
#[test]
fn test_witness_append_entries_stale_term_rejected() {
    let mut witness = WitnessNode::new("w0", 10);

    // Bring to term 5
    witness.handle_append_entries(&append_req(5, 0, 0, vec![], 0));
    assert_eq!(witness.current_term(), 5);

    // Stale leader at term 3
    let resp = witness.handle_append_entries(&append_req(3, 0, 0, vec![], 0));
    assert!(!resp.success, "stale-leader append should fail");
    assert!(
        resp.in_window,
        "failure is from stale leader, not out-of-window"
    );
}

/// An AppendEntries with prev_log_index before the tail window returns in_window=false.
#[test]
fn test_witness_append_entries_out_of_window() {
    let mut witness = WitnessNode::new("w0", 5); // tiny window

    // Fill window with entries 1–5
    let fill_req = append_req(
        1,
        0,
        0,
        (1..=5).map(|i| make_witness_entry(i, 1)).collect(),
        5,
    );
    witness.handle_append_entries(&fill_req);
    assert_eq!(witness.tail_len(), 5);

    // Now add more to push the window forward (entries 6–10)
    let push_req = append_req(
        1,
        5,
        1,
        (6..=10).map(|i| make_witness_entry(i, 1)).collect(),
        10,
    );
    witness.handle_append_entries(&push_req);

    // Entry 1 has been evicted. A request with prev_log_index=1 should be out-of-window.
    let old_req = WitnessAppendRequest {
        leader_id: "leader".to_owned(),
        term: 1,
        prev_log_index: 1,
        prev_log_term: 1,
        entries: vec![make_witness_entry(2, 1)],
        leader_commit: 1,
    };
    let resp = witness.handle_append_entries(&old_req);
    assert!(
        !resp.in_window,
        "prev_log_index before tail window should return in_window=false"
    );
    assert!(!resp.success);
}

/// Tail window evicts oldest entries when capacity is exceeded.
#[test]
fn test_witness_tail_window_evicts_oldest() {
    let mut witness = WitnessNode::new("w0", 3); // window of 3

    let req = append_req(
        1,
        0,
        0,
        (1..=5).map(|i| make_witness_entry(i, 1)).collect(),
        3,
    );
    witness.handle_append_entries(&req);

    // Only the last 3 entries should remain
    assert_eq!(witness.tail_len(), 3);
    assert!(!witness.has_entry(1), "entry 1 should have been evicted");
    assert!(!witness.has_entry(2), "entry 2 should have been evicted");
    assert!(witness.has_entry(3), "entry 3 should be present");
    assert!(witness.has_entry(4), "entry 4 should be present");
    assert!(witness.has_entry(5), "entry 5 should be present");
}

/// commit_index advances to min(leader_commit, last_appended_index).
#[test]
fn test_witness_commit_index_advances() {
    let mut witness = WitnessNode::new("w0", 10);

    // Append entries 1–3, leader_commit = 2
    let resp = witness.handle_append_entries(&append_req(
        1,
        0,
        0,
        (1..=3).map(|i| make_witness_entry(i, 1)).collect(),
        2,
    ));
    assert!(resp.success);
    assert_eq!(witness.commit_index(), 2);

    // Append entries 4–6, leader_commit = 5 (beyond new entries is fine)
    let resp2 = witness.handle_append_entries(&append_req(
        1,
        3,
        1,
        (4..=6).map(|i| make_witness_entry(i, 1)).collect(),
        5,
    ));
    assert!(resp2.success);
    assert_eq!(witness.commit_index(), 5);
}
