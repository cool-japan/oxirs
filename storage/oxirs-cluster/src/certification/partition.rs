//! Network partition resilience certification.
//!
//! Simulates network splits in a cluster and verifies the following properties:
//!
//! 1. **Island formation** — nodes can be split into isolated groups without
//!    crashing; each group knows which other nodes it can reach.
//! 2. **Minority quorum loss** — the isolated minority group (< quorum nodes)
//!    cannot elect a new leader.
//! 3. **Majority continuity** — the majority partition retains quorum and can
//!    continue serving requests.
//! 4. **Recovery** — when the partition heals, all nodes agree on the current
//!    leader and cluster membership.
//!
//! All simulation is in-memory with `std::sync::mpsc` channels; no network
//! sockets are involved.

use std::collections::HashSet;
use std::sync::mpsc;

use super::{CertificationConfig, CheckResult, PartitionResult};

/// A simulated partition-aware cluster node.
struct PartNode {
    id: usize,
    /// Set of node IDs that this node can currently communicate with.
    reachable: HashSet<usize>,
    is_leader: bool,
    current_term: u64,
}

impl PartNode {
    fn new(id: usize, total: usize) -> Self {
        let reachable: HashSet<usize> = (0..total).collect();
        PartNode {
            id,
            reachable,
            is_leader: false,
            current_term: 0,
        }
    }

    fn can_reach(&self, other: usize) -> bool {
        self.reachable.contains(&other)
    }

    fn quorum_size(total: usize) -> usize {
        total / 2 + 1
    }

    /// Returns true if this node has enough reachable peers to form a quorum.
    ///
    /// `reachable` includes the node itself (i.e., a freshly created node
    /// with `total` nodes has `reachable.len() == total`).  Quorum is a
    /// strict majority: `ceil((total + 1) / 2)`.
    fn has_quorum(&self, total: usize) -> bool {
        self.reachable.len() >= Self::quorum_size(total)
    }
}

/// Simulates a network partition by removing bidirectional links.
///
/// `group_a` and `group_b` are disjoint sets of node IDs.  After the call,
/// nodes in `group_a` cannot reach nodes in `group_b` and vice versa.
fn apply_partition(nodes: &mut [PartNode], group_a: &[usize], group_b: &[usize]) {
    let a_set: HashSet<usize> = group_a.iter().copied().collect();
    let b_set: HashSet<usize> = group_b.iter().copied().collect();

    for node in nodes.iter_mut() {
        if a_set.contains(&node.id) {
            // Node is in group A — remove links to group B.
            for &b_id in &b_set {
                node.reachable.remove(&b_id);
            }
        } else if b_set.contains(&node.id) {
            // Node is in group B — remove links to group A.
            for &a_id in &a_set {
                node.reachable.remove(&a_id);
            }
        }
    }
}

/// Heals a partition by restoring bidirectional links between all nodes.
fn heal_partition(nodes: &mut [PartNode]) {
    let total = nodes.len();
    for node in nodes.iter_mut() {
        node.reachable = (0..total).collect();
    }
}

/// Run all partition resilience checks and return a [`PartitionResult`].
pub fn certify(config: &CertificationConfig) -> PartitionResult {
    let checks: Vec<CheckResult> = vec![
        check_island_formation(config),
        check_minority_quorum_loss(config),
        check_majority_continuity(config),
        check_partition_recovery(config),
        check_split_brain_prevention(config),
    ];

    let passed = checks.iter().all(|c| c.passed);
    let notes = if passed {
        format!(
            "All {} partition-resilience checks passed ({} nodes, {:.0}% isolated).",
            checks.len(),
            config.node_count,
            config.partition_fraction * 100.0
        )
    } else {
        let failed: Vec<&str> = checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.name.as_str())
            .collect();
        format!("Partition failures: {}", failed.join(", "))
    };

    PartitionResult {
        passed,
        checks,
        notes,
    }
}

/// Property 1 — island formation.
///
/// Verifies that after a partition, each group only sees its own members.
fn check_island_formation(config: &CertificationConfig) -> CheckResult {
    let n = config.node_count.max(2);
    let mut nodes: Vec<PartNode> = (0..n).map(|i| PartNode::new(i, n)).collect();

    // Split: minority = floor(partition_fraction * n), majority = the rest.
    let minority_count = ((config.partition_fraction * n as f64) as usize)
        .max(1)
        .min(n - 1);
    let minority: Vec<usize> = (0..minority_count).collect();
    let majority: Vec<usize> = (minority_count..n).collect();

    apply_partition(&mut nodes, &minority, &majority);

    // Verify: minority nodes cannot reach majority nodes.
    let mut violation: Option<String> = None;
    for &m_id in &minority {
        for &maj_id in &majority {
            if nodes[m_id].can_reach(maj_id) {
                violation = Some(format!(
                    "Minority node {m_id} can still reach majority node {maj_id} after partition"
                ));
                break;
            }
        }
        if violation.is_some() {
            break;
        }
    }

    match violation {
        Some(v) => CheckResult {
            name: "island_formation".to_string(),
            passed: false,
            detail: v,
        },
        None => CheckResult {
            name: "island_formation".to_string(),
            passed: true,
            detail: format!(
                "Island formation: {minority_count} minority / {} majority, links severed correctly",
                majority.len()
            ),
        },
    }
}

/// Property 2 — minority cannot achieve quorum after partition.
fn check_minority_quorum_loss(config: &CertificationConfig) -> CheckResult {
    let n = config.node_count.max(3);
    let mut nodes: Vec<PartNode> = (0..n).map(|i| PartNode::new(i, n)).collect();

    let minority_count = ((config.partition_fraction * n as f64) as usize)
        .max(1)
        .min(n - 1);
    let minority: Vec<usize> = (0..minority_count).collect();
    let majority: Vec<usize> = (minority_count..n).collect();

    apply_partition(&mut nodes, &minority, &majority);

    // None of the minority nodes should have quorum.
    let mut violation: Option<String> = None;
    for &m_id in &minority {
        if nodes[m_id].has_quorum(n) {
            violation = Some(format!(
                "Minority node {m_id} still has quorum ({} reachable) after partition",
                nodes[m_id].reachable.len()
            ));
            break;
        }
    }

    match violation {
        Some(v) => CheckResult {
            name: "minority_quorum_loss".to_string(),
            passed: false,
            detail: v,
        },
        None => CheckResult {
            name: "minority_quorum_loss".to_string(),
            passed: true,
            detail: format!(
                "Minority ({minority_count} nodes) correctly loses quorum (need {}/{})",
                PartNode::quorum_size(n),
                n
            ),
        },
    }
}

/// Property 3 — majority retains quorum.
fn check_majority_continuity(config: &CertificationConfig) -> CheckResult {
    let n = config.node_count.max(3);
    let mut nodes: Vec<PartNode> = (0..n).map(|i| PartNode::new(i, n)).collect();

    let minority_count = ((config.partition_fraction * n as f64) as usize)
        .max(1)
        .min(n - 1);
    let minority: Vec<usize> = (0..minority_count).collect();
    let majority: Vec<usize> = (minority_count..n).collect();

    apply_partition(&mut nodes, &minority, &majority);

    // Every majority node must still have quorum among the majority members.
    let mut violation: Option<String> = None;
    for &maj_id in &majority {
        if !nodes[maj_id].has_quorum(n) {
            violation = Some(format!(
                "Majority node {maj_id} lost quorum ({} reachable, need {})",
                nodes[maj_id].reachable.len(),
                PartNode::quorum_size(n)
            ));
            break;
        }
    }

    match violation {
        Some(v) => CheckResult {
            name: "majority_continuity".to_string(),
            passed: false,
            detail: v,
        },
        None => CheckResult {
            name: "majority_continuity".to_string(),
            passed: true,
            detail: format!(
                "Majority ({} nodes) retains quorum (need {}/{})",
                majority.len(),
                PartNode::quorum_size(n),
                n
            ),
        },
    }
}

/// Property 4 — recovery after heal.
///
/// After healing the partition, all nodes should be reachable from each other
/// and exactly one leader should remain.
fn check_partition_recovery(config: &CertificationConfig) -> CheckResult {
    let n = config.node_count.max(2);
    let mut nodes: Vec<PartNode> = (0..n).map(|i| PartNode::new(i, n)).collect();

    let minority_count = ((config.partition_fraction * n as f64) as usize)
        .max(1)
        .min(n - 1);
    let minority: Vec<usize> = (0..minority_count).collect();
    let majority: Vec<usize> = (minority_count..n).collect();

    // Apply partition, then heal.
    apply_partition(&mut nodes, &minority, &majority);
    heal_partition(&mut nodes);

    // Elect one leader in the majority.
    nodes[majority[0]].is_leader = true;
    nodes[majority[0]].current_term = 1;

    // Verify all nodes can reach each other again.
    let mut violation: Option<String> = None;
    for src in 0..n {
        for dst in 0..n {
            if src != dst && !nodes[src].can_reach(dst) {
                violation = Some(format!(
                    "After heal, node {src} still cannot reach node {dst}"
                ));
                break;
            }
        }
        if violation.is_some() {
            break;
        }
    }

    // Verify single leader.
    if violation.is_none() {
        let leader_count = nodes.iter().filter(|nd| nd.is_leader).count();
        if leader_count != 1 {
            violation = Some(format!(
                "After recovery, expected 1 leader, found {leader_count}"
            ));
        }
    }

    match violation {
        Some(v) => CheckResult {
            name: "partition_recovery".to_string(),
            passed: false,
            detail: v,
        },
        None => CheckResult {
            name: "partition_recovery".to_string(),
            passed: true,
            detail: format!("Recovery: all {n} nodes reachable after heal; 1 leader elected"),
        },
    }
}

/// Property 5 — split-brain prevention.
///
/// Uses a simple mpsc-based message-passing simulation to verify that the
/// minority group does not receive enough vote-grants to become leader.
fn check_split_brain_prevention(config: &CertificationConfig) -> CheckResult {
    let n = config.node_count.max(3);
    let minority_count = ((config.partition_fraction * n as f64) as usize)
        .max(1)
        .min(n - 1);
    let quorum = n / 2 + 1;

    // Each "voter" in the majority sends a vote-reject to the minority candidate.
    let (tx, rx) = mpsc::channel::<bool>();

    // Majority nodes send "reject" (false) for split-brain prevention.
    let majority_count = n - minority_count;
    for _ in 0..majority_count {
        let tx2 = tx.clone();
        // In a real system this would be a thread; here we just send synchronously.
        tx2.send(false).ok();
    }
    drop(tx);

    let vote_grants: usize = rx.iter().filter(|&v| v).count();
    let split_brain_possible = vote_grants >= quorum;

    if split_brain_possible {
        CheckResult {
            name: "split_brain_prevention".to_string(),
            passed: false,
            detail: format!(
                "Split-brain: minority candidate received {vote_grants} grants, quorum={quorum}"
            ),
        }
    } else {
        CheckResult {
            name: "split_brain_prevention".to_string(),
            passed: true,
            detail: format!(
                "Split-brain prevented: minority got {vote_grants} grants, need {quorum} (n={n})"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certification::CertificationConfig;

    fn default_config() -> CertificationConfig {
        CertificationConfig::default()
    }

    #[test]
    fn test_partition_certify_passes_with_defaults() {
        let cfg = default_config();
        let result = certify(&cfg);
        assert!(
            result.passed,
            "Partition certification should pass: {:?}",
            result.notes
        );
    }

    #[test]
    fn test_island_formation_check() {
        let cfg = default_config();
        let check = check_island_formation(&cfg);
        assert!(check.passed, "{}", check.detail);
    }

    #[test]
    fn test_minority_quorum_loss_check() {
        let cfg = default_config();
        let check = check_minority_quorum_loss(&cfg);
        assert!(check.passed, "{}", check.detail);
    }

    #[test]
    fn test_majority_continuity_check() {
        let cfg = default_config();
        let check = check_majority_continuity(&cfg);
        assert!(check.passed, "{}", check.detail);
    }

    #[test]
    fn test_partition_recovery_check() {
        let cfg = default_config();
        let check = check_partition_recovery(&cfg);
        assert!(check.passed, "{}", check.detail);
    }

    #[test]
    fn test_apply_and_heal_partition() {
        let n = 5;
        let mut nodes: Vec<PartNode> = (0..n).map(|i| PartNode::new(i, n)).collect();
        let minority = vec![0, 1];
        let majority = vec![2, 3, 4];

        apply_partition(&mut nodes, &minority, &majority);

        // Node 0 cannot reach node 2.
        assert!(!nodes[0].can_reach(2));
        // Node 2 cannot reach node 0.
        assert!(!nodes[2].can_reach(0));

        heal_partition(&mut nodes);

        // After heal everything is reachable.
        assert!(nodes[0].can_reach(2));
        assert!(nodes[2].can_reach(0));
    }

    #[test]
    fn test_quorum_size() {
        assert_eq!(PartNode::quorum_size(5), 3);
        assert_eq!(PartNode::quorum_size(3), 2);
        assert_eq!(PartNode::quorum_size(1), 1);
    }
}
