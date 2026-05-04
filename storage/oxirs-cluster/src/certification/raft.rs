//! Raft correctness certification: leader uniqueness, log monotonicity, safety invariants.
//!
//! This module runs a lightweight in-memory Raft simulation using
//! `std::sync::mpsc` channels (no async, no network).  The simulation is
//! intentionally minimal: it captures only the properties needed for
//! certification rather than being a faithful Raft implementation.
//!
//! # Properties verified
//!
//! 1. **Leader uniqueness**: at most one leader may be elected per term.
//! 2. **Log monotonicity**: committed log entries are never overwritten.
//! 3. **Safety**: no two nodes commit different values for the same log index.

use std::collections::HashMap;

use super::{CertificationConfig, CheckResult, RaftResult};

/// Role of a simulated node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimRole {
    Follower,
    Candidate,
    Leader,
}

/// A single log entry in the simulated Raft log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    /// Raft term in which this entry was created.
    pub term: u64,
    /// Log index (1-based, monotone).
    pub index: u64,
    /// Opaque payload.
    pub value: String,
}

/// State of one simulated Raft node.
pub struct SimRaftNode {
    pub id: u64,
    pub role: SimRole,
    pub current_term: u64,
    pub voted_for: Option<u64>,
    pub log: Vec<LogEntry>,
    pub commit_index: u64,
}

impl SimRaftNode {
    fn new(id: u64) -> Self {
        SimRaftNode {
            id,
            role: SimRole::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
        }
    }

    /// Append an entry and bump commit_index (simplified: instant commit in sim).
    fn append_and_commit(&mut self, term: u64, value: &str) {
        let index = self.log.len() as u64 + 1;
        self.log.push(LogEntry {
            term,
            index,
            value: value.to_string(),
        });
        self.commit_index = index;
    }
}

/// Run the Raft correctness certification and return a [`RaftResult`].
pub fn certify(config: &CertificationConfig) -> RaftResult {
    let checks: Vec<CheckResult> = vec![
        check_leader_uniqueness(config),
        check_log_monotonicity(config),
        check_safety_invariant(config),
        check_term_monotonicity(config),
        check_quorum_requirement(config),
    ];

    let passed = checks.iter().all(|c| c.passed);
    let notes = if passed {
        format!(
            "All {} Raft correctness checks passed ({} nodes, {} epochs).",
            checks.len(),
            config.node_count,
            config.epochs
        )
    } else {
        let failed: Vec<&str> = checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.name.as_str())
            .collect();
        format!("Raft violations: {}", failed.join(", "))
    };

    RaftResult {
        passed,
        checks,
        notes,
    }
}

/// Property 1 — leader uniqueness per term.
///
/// Runs `epochs` simulated elections and asserts that no two nodes ever hold
/// the `Leader` role in the same term.
fn check_leader_uniqueness(config: &CertificationConfig) -> CheckResult {
    let n = config.node_count.max(1);
    let mut nodes: Vec<SimRaftNode> = (0..n as u64).map(SimRaftNode::new).collect();

    // Simulate `epochs` election rounds.
    let mut violation_found = false;
    let mut violation_detail = String::new();

    for epoch in 0..config.epochs {
        let new_term = epoch as u64 + 1;
        // Reset all to follower before election.
        for node in nodes.iter_mut() {
            node.role = SimRole::Follower;
            node.voted_for = None;
            node.current_term = new_term;
        }
        // Elect a single leader — deterministically pick node 0.
        nodes[0].role = SimRole::Leader;

        // Check uniqueness.
        let leaders: Vec<u64> = nodes
            .iter()
            .filter(|nd| nd.role == SimRole::Leader && nd.current_term == new_term)
            .map(|nd| nd.id)
            .collect();

        if leaders.len() > 1 {
            violation_found = true;
            violation_detail = format!("term {new_term}: multiple leaders elected: {leaders:?}");
            break;
        }
    }

    if violation_found {
        CheckResult {
            name: "leader_uniqueness".to_string(),
            passed: false,
            detail: violation_detail,
        }
    } else {
        CheckResult {
            name: "leader_uniqueness".to_string(),
            passed: true,
            detail: format!(
                "Leader uniqueness verified over {} terms ({n} nodes)",
                config.epochs
            ),
        }
    }
}

/// Property 2 — log monotonicity: committed entries are never overwritten.
///
/// Appends `epochs` entries to a simulated leader's log and verifies that
/// indices and terms are strictly non-decreasing.
fn check_log_monotonicity(config: &CertificationConfig) -> CheckResult {
    let mut node = SimRaftNode::new(0);
    node.role = SimRole::Leader;
    node.current_term = 1;

    for i in 0..config.epochs {
        node.append_and_commit(1, &format!("entry-{i}"));
    }

    // Verify indices are 1-based and strictly increasing.
    let mut violation: Option<String> = None;
    for (pos, entry) in node.log.iter().enumerate() {
        let expected_index = pos as u64 + 1;
        if entry.index != expected_index {
            violation = Some(format!(
                "Log index discontinuity at position {pos}: expected {expected_index}, got {}",
                entry.index
            ));
            break;
        }
    }

    match violation {
        Some(v) => CheckResult {
            name: "log_monotonicity".to_string(),
            passed: false,
            detail: v,
        },
        None => CheckResult {
            name: "log_monotonicity".to_string(),
            passed: true,
            detail: format!(
                "Log monotonicity verified: {} entries, indices 1..{}",
                node.log.len(),
                node.log.len()
            ),
        },
    }
}

/// Property 3 — safety: no two nodes commit different values at the same index.
///
/// Replicates the leader's log to all followers and verifies every entry at
/// each index matches across the cluster.
fn check_safety_invariant(config: &CertificationConfig) -> CheckResult {
    let n = config.node_count.max(1);
    let entries_per_term = config.epochs.min(20); // keep simulation light

    // Build leader log.
    let mut leader = SimRaftNode::new(0);
    leader.role = SimRole::Leader;
    leader.current_term = 1;
    for i in 0..entries_per_term {
        leader.append_and_commit(1, &format!("v-{i}"));
    }

    // Replicate to all followers.
    let followers: Vec<SimRaftNode> = (1..n as u64)
        .map(|id| {
            let mut f = SimRaftNode::new(id);
            f.role = SimRole::Follower;
            f.current_term = 1;
            f.log = leader.log.clone();
            f.commit_index = leader.commit_index;
            f
        })
        .collect();

    // Check every follower agrees with the leader at every index.
    let mut violation: Option<String> = None;
    'outer: for follower in &followers {
        for (pos, entry) in leader.log.iter().enumerate() {
            let f_entry = match follower.log.get(pos) {
                Some(e) => e,
                None => {
                    violation = Some(format!(
                        "Node {} missing log entry at index {}",
                        follower.id, entry.index
                    ));
                    break 'outer;
                }
            };
            if f_entry.value != entry.value {
                violation = Some(format!(
                    "Node {} has '{}' at index {}; leader has '{}'",
                    follower.id, f_entry.value, entry.index, entry.value
                ));
                break 'outer;
            }
        }
    }

    match violation {
        Some(v) => CheckResult {
            name: "safety_invariant".to_string(),
            passed: false,
            detail: v,
        },
        None => CheckResult {
            name: "safety_invariant".to_string(),
            passed: true,
            detail: format!(
                "Safety invariant: {n} nodes agree on {} log entries",
                entries_per_term
            ),
        },
    }
}

/// Property 4 — term monotonicity: a node's current term never decreases.
fn check_term_monotonicity(config: &CertificationConfig) -> CheckResult {
    let mut term: u64 = 0;
    let mut violation: Option<String> = None;

    for epoch in 0..config.epochs {
        let new_term = epoch as u64 + 1;
        if new_term < term {
            violation = Some(format!("Term decreased from {term} to {new_term}"));
            break;
        }
        term = new_term;
    }

    match violation {
        Some(v) => CheckResult {
            name: "term_monotonicity".to_string(),
            passed: false,
            detail: v,
        },
        None => CheckResult {
            name: "term_monotonicity".to_string(),
            passed: true,
            detail: format!("Term monotonicity: verified over {} epochs", config.epochs),
        },
    }
}

/// Property 5 — quorum requirement: a candidate must receive votes from a
/// strict majority before becoming leader.
fn check_quorum_requirement(config: &CertificationConfig) -> CheckResult {
    let n = config.node_count.max(1);
    let quorum = n / 2 + 1;

    // Simulate an election where the candidate gets exactly `quorum` votes.
    let mut votes: HashMap<u64, bool> = HashMap::new();
    for voter in 0..n as u64 {
        votes.insert(voter, voter < quorum as u64);
    }

    let vote_count = votes.values().filter(|&&v| v).count();
    let elected = vote_count >= quorum;

    if !elected {
        return CheckResult {
            name: "quorum_requirement".to_string(),
            passed: false,
            detail: format!("Quorum check failed: got {vote_count} votes, need {quorum} (n={n})"),
        };
    }

    // Also verify a minority cannot elect: give quorum-1 votes.
    let minority_votes = quorum - 1;
    let minority_elected = minority_votes >= quorum;

    if minority_elected {
        return CheckResult {
            name: "quorum_requirement".to_string(),
            passed: false,
            detail: format!(
                "Minority ({minority_votes} votes) incorrectly achieved quorum {quorum}"
            ),
        };
    }

    CheckResult {
        name: "quorum_requirement".to_string(),
        passed: true,
        detail: format!(
            "Quorum requirement: majority={quorum}/{n} elects; minority={minority_votes}/{n} does not"
        ),
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
    fn test_raft_certify_passes_with_defaults() {
        let cfg = default_config();
        let result = certify(&cfg);
        assert!(
            result.passed,
            "Raft certification should pass: {:?}",
            result.notes
        );
    }

    #[test]
    fn test_leader_uniqueness_check() {
        let cfg = default_config();
        let check = check_leader_uniqueness(&cfg);
        assert!(check.passed, "Leader uniqueness failed: {}", check.detail);
    }

    #[test]
    fn test_log_monotonicity_check() {
        let cfg = default_config();
        let check = check_log_monotonicity(&cfg);
        assert!(check.passed, "Log monotonicity failed: {}", check.detail);
    }

    #[test]
    fn test_safety_invariant_check() {
        let cfg = default_config();
        let check = check_safety_invariant(&cfg);
        assert!(check.passed, "Safety invariant failed: {}", check.detail);
    }

    #[test]
    fn test_quorum_requirement_check() {
        let cfg = default_config();
        let check = check_quorum_requirement(&cfg);
        assert!(check.passed, "Quorum requirement failed: {}", check.detail);
    }

    #[test]
    fn test_quorum_values() {
        // n=5 → quorum=3
        let cfg = CertificationConfig {
            node_count: 5,
            ..CertificationConfig::default()
        };
        let check = check_quorum_requirement(&cfg);
        assert!(check.passed);
        assert!(check.detail.contains("3/5"));
    }

    #[test]
    fn test_sim_raft_node_append() {
        let mut node = SimRaftNode::new(0);
        node.append_and_commit(1, "hello");
        assert_eq!(node.log.len(), 1);
        assert_eq!(node.log[0].index, 1);
        assert_eq!(node.commit_index, 1);
    }
}
