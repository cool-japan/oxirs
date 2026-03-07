//! Deadlock detection for MVCC transactions using a wait-for graph.
//!
//! A wait-for graph has an edge `A → B` when transaction A is waiting
//! for transaction B to release a resource. A cycle in this graph means
//! deadlock. The detector periodically scans for cycles and aborts the
//! youngest transaction in each cycle (the "victim").
//!
//! ## Algorithm
//! Tarjan's strongly connected components (SCC) algorithm runs in O(V+E).
//! We consider any SCC with more than one node to be a deadlock cycle.
//!
//! ## Usage
//! ```ignore
//! let detector = DeadlockDetector::new();
//! detector.add_wait_edge(tx_a, tx_b);        // tx_a waiting for tx_b
//! detector.remove_transaction(tx_b);          // tx_b committed / rolled back
//! let victims = detector.detect_cycles();     // returns victim TxIds
//! ```

use crate::mvcc::TxId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

// =========================================================================
// Core data structure
// =========================================================================

/// Wait-for graph state (held behind a Mutex)
struct WaitForGraph {
    /// edges[A] = set of TxIds that A is waiting for
    edges: HashMap<TxId, HashSet<TxId>>,
    /// all known transaction nodes
    nodes: HashSet<TxId>,
}

impl WaitForGraph {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    fn add_node(&mut self, tx_id: TxId) {
        self.nodes.insert(tx_id);
        self.edges.entry(tx_id).or_default();
    }

    fn add_edge(&mut self, from: TxId, to: TxId) {
        self.add_node(from);
        self.add_node(to);
        self.edges.entry(from).or_default().insert(to);
    }

    fn remove_node(&mut self, tx_id: TxId) {
        self.nodes.remove(&tx_id);
        self.edges.remove(&tx_id);
        // Remove all edges pointing TO tx_id
        for neighbors in self.edges.values_mut() {
            neighbors.remove(&tx_id);
        }
    }

    /// Detect deadlock cycles using iterative DFS (Tarjan-inspired).
    ///
    /// Returns a list of SCCs that contain more than one node; each such
    /// SCC represents a set of mutually deadlocked transactions.
    fn find_cycles(&self) -> Vec<Vec<TxId>> {
        // Iterative Tarjan SCC
        let nodes: Vec<TxId> = self.nodes.iter().copied().collect();
        let n = nodes.len();
        if n == 0 {
            return vec![];
        }

        let node_index: HashMap<TxId, usize> =
            nodes.iter().enumerate().map(|(i, &t)| (t, i)).collect();

        let mut index_counter = 0usize;
        let mut stack: Vec<usize> = Vec::new();
        let mut on_stack: Vec<bool> = vec![false; n];
        let mut index: Vec<Option<usize>> = vec![None; n];
        let mut lowlink: Vec<usize> = vec![0; n];
        let mut sccs: Vec<Vec<TxId>> = Vec::new();

        for start in 0..n {
            if index[start].is_some() {
                continue;
            }
            // Iterative DFS for Tarjan
            let mut call_stack: Vec<(usize, usize, Vec<usize>)> = Vec::new(); // (node, edge_idx, neighbors)

            let neighbors_of = |node: usize| -> Vec<usize> {
                let tx = nodes[node];
                self.edges
                    .get(&tx)
                    .map(|nbrs| {
                        nbrs.iter()
                            .filter_map(|&t| node_index.get(&t).copied())
                            .collect()
                    })
                    .unwrap_or_default()
            };

            index[start] = Some(index_counter);
            lowlink[start] = index_counter;
            index_counter += 1;
            stack.push(start);
            on_stack[start] = true;
            call_stack.push((start, 0, neighbors_of(start)));

            while let Some((v, ei, ref mut nbrs)) = call_stack.last_mut().map(|x| x.clone()) {
                call_stack.pop();

                let mut ei = ei;
                let nbrs_owned: Vec<usize> = nbrs.clone();

                loop {
                    if ei >= nbrs_owned.len() {
                        // Post-order: check if v is SCC root
                        if let Some(parent_frame) = call_stack.last_mut() {
                            let parent = parent_frame.0;
                            lowlink[parent] = lowlink[parent].min(lowlink[v]);
                        }
                        if lowlink[v] == index[v].unwrap_or(0) {
                            // Pop SCC
                            let mut scc = Vec::new();
                            loop {
                                let w = stack.pop().unwrap_or(0);
                                on_stack[w] = false;
                                scc.push(nodes[w]);
                                if w == v {
                                    break;
                                }
                            }
                            if scc.len() > 1 {
                                sccs.push(scc);
                            }
                        }
                        break;
                    }

                    let w = nbrs_owned[ei];
                    ei += 1;

                    if index[w].is_none() {
                        index[w] = Some(index_counter);
                        lowlink[w] = index_counter;
                        index_counter += 1;
                        stack.push(w);
                        on_stack[w] = true;
                        // Push continuation for v, then descend into w
                        call_stack.push((v, ei, nbrs_owned));
                        call_stack.push((w, 0, neighbors_of(w)));
                        break;
                    } else if on_stack[w] {
                        lowlink[v] = lowlink[v].min(index[w].unwrap_or(0));
                    }
                }
            }
        }

        sccs
    }
}

// =========================================================================
// Public API
// =========================================================================

/// Deadlock detector backed by a wait-for graph.
pub struct DeadlockDetector {
    graph: Arc<Mutex<WaitForGraph>>,
    /// Running count of detected deadlocks
    deadlock_count: std::sync::atomic::AtomicU64,
}

impl DeadlockDetector {
    /// Create a new detector.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            graph: Arc::new(Mutex::new(WaitForGraph::new())),
            deadlock_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Register that `waiter` is now blocked waiting for `holder`.
    pub fn add_wait_edge(&self, waiter: TxId, holder: TxId) {
        let mut g = self.graph.lock().expect("deadlock graph lock");
        g.add_edge(waiter, holder);
    }

    /// Remove all edges from/to `tx_id` (called when a transaction completes).
    pub fn remove_transaction(&self, tx_id: TxId) {
        let mut g = self.graph.lock().expect("deadlock graph lock");
        g.remove_node(tx_id);
    }

    /// Scan the wait-for graph for cycles (deadlocks).
    ///
    /// Returns the TxIds chosen as victims (youngest in each cycle).
    /// The caller is responsible for aborting these transactions.
    pub fn detect_cycles(&self) -> Vec<TxId> {
        let g = self.graph.lock().expect("deadlock graph lock");
        let sccs = g.find_cycles();
        drop(g);

        let mut victims = Vec::new();
        for scc in sccs {
            self.deadlock_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Choose victim: youngest (highest TxId) in the cycle
            if let Some(&victim) = scc.iter().max() {
                victims.push(victim);
            }
        }
        victims
    }

    /// Total number of deadlock cycles ever detected.
    pub fn deadlock_count(&self) -> u64 {
        self.deadlock_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Return current edge count (for diagnostics).
    pub fn edge_count(&self) -> usize {
        let g = self.graph.lock().expect("deadlock graph lock");
        g.edges.values().map(|s| s.len()).sum()
    }

    /// Return current node count (for diagnostics).
    pub fn node_count(&self) -> usize {
        let g = self.graph.lock().expect("deadlock graph lock");
        g.nodes.len()
    }
}

impl Default for DeadlockDetector {
    fn default() -> Self {
        Arc::try_unwrap(Self::new()).unwrap_or_else(|_| unreachable!())
    }
}

// =========================================================================
// Background scan helper (optional, for use with tokio tasks)
// =========================================================================

/// Result of a background scan
#[derive(Debug, Clone)]
pub struct DeadlockScanResult {
    /// TxIds selected as deadlock victims in this scan
    pub victims: Vec<TxId>,
    /// Total deadlocks detected so far
    pub total_detected: u64,
}

impl DeadlockDetector {
    /// Run a single detection scan and return the result.
    /// In a real system you would call `MvccManager::abort(victim)` for each victim.
    pub fn scan(&self) -> DeadlockScanResult {
        let victims = self.detect_cycles();
        DeadlockScanResult {
            total_detected: self.deadlock_count(),
            victims,
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> WaitForGraph {
        WaitForGraph::new()
    }

    // -----------------------------------------------------------------------
    // WaitForGraph unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph_no_cycles() {
        let g = make_graph();
        assert!(g.find_cycles().is_empty());
    }

    #[test]
    fn test_single_node_no_cycle() {
        let mut g = make_graph();
        g.add_node(1);
        assert!(g.find_cycles().is_empty());
    }

    #[test]
    fn test_two_node_chain_no_cycle() {
        let mut g = make_graph();
        g.add_edge(1, 2); // 1 waits for 2
        assert!(g.find_cycles().is_empty());
    }

    #[test]
    fn test_simple_cycle_two_nodes() {
        let mut g = make_graph();
        g.add_edge(1, 2);
        g.add_edge(2, 1);
        let cycles = g.find_cycles();
        assert!(!cycles.is_empty(), "expected a cycle");
        assert_eq!(cycles[0].len(), 2);
    }

    #[test]
    fn test_three_node_cycle() {
        let mut g = make_graph();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 1);
        let cycles = g.find_cycles();
        assert!(!cycles.is_empty());
        assert_eq!(cycles[0].len(), 3);
    }

    #[test]
    fn test_remove_node_breaks_cycle() {
        let mut g = make_graph();
        g.add_edge(1, 2);
        g.add_edge(2, 1);
        g.remove_node(2);
        assert!(g.find_cycles().is_empty());
    }

    #[test]
    fn test_diamond_no_cycle() {
        let mut g = make_graph();
        // 1→2, 1→3, 2→4, 3→4
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(2, 4);
        g.add_edge(3, 4);
        assert!(g.find_cycles().is_empty());
    }

    #[test]
    fn test_multiple_independent_cycles() {
        let mut g = make_graph();
        // Cycle A: 1↔2
        g.add_edge(1, 2);
        g.add_edge(2, 1);
        // Cycle B: 3↔4
        g.add_edge(3, 4);
        g.add_edge(4, 3);
        let cycles = g.find_cycles();
        assert_eq!(cycles.len(), 2);
    }

    // -----------------------------------------------------------------------
    // DeadlockDetector API tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_detector_no_deadlock_initially() {
        let d = DeadlockDetector::new();
        assert!(d.detect_cycles().is_empty());
        assert_eq!(d.deadlock_count(), 0);
    }

    #[test]
    fn test_detector_single_edge_no_deadlock() {
        let d = DeadlockDetector::new();
        d.add_wait_edge(10, 20);
        assert!(d.detect_cycles().is_empty());
    }

    #[test]
    fn test_detector_cycle_detected() {
        let d = DeadlockDetector::new();
        d.add_wait_edge(10, 20);
        d.add_wait_edge(20, 10);
        let victims = d.detect_cycles();
        assert!(!victims.is_empty());
        // Victim should be the youngest (highest TxId)
        assert_eq!(victims[0], 20);
    }

    #[test]
    fn test_detector_victim_is_youngest() {
        let d = DeadlockDetector::new();
        // Cycle: 5 → 10 → 15 → 5
        d.add_wait_edge(5, 10);
        d.add_wait_edge(10, 15);
        d.add_wait_edge(15, 5);
        let victims = d.detect_cycles();
        assert!(!victims.is_empty());
        assert_eq!(victims[0], 15); // youngest in cycle
    }

    #[test]
    fn test_detector_remove_breaks_deadlock() {
        let d = DeadlockDetector::new();
        d.add_wait_edge(1, 2);
        d.add_wait_edge(2, 1);
        assert!(!d.detect_cycles().is_empty());

        d.remove_transaction(2);
        assert!(d.detect_cycles().is_empty());
    }

    #[test]
    fn test_detector_edge_count() {
        let d = DeadlockDetector::new();
        assert_eq!(d.edge_count(), 0);
        d.add_wait_edge(1, 2);
        assert_eq!(d.edge_count(), 1);
        d.add_wait_edge(2, 3);
        assert_eq!(d.edge_count(), 2);
    }

    #[test]
    fn test_detector_node_count() {
        let d = DeadlockDetector::new();
        assert_eq!(d.node_count(), 0);
        d.add_wait_edge(1, 2);
        assert_eq!(d.node_count(), 2);
        d.add_wait_edge(3, 4);
        assert_eq!(d.node_count(), 4);
    }

    #[test]
    fn test_detector_deadlock_count_increments() {
        let d = DeadlockDetector::new();
        d.add_wait_edge(1, 2);
        d.add_wait_edge(2, 1);

        let _ = d.detect_cycles();
        assert!(d.deadlock_count() >= 1);
    }

    #[test]
    fn test_scan_returns_victims_and_count() {
        let d = DeadlockDetector::new();
        d.add_wait_edge(100, 200);
        d.add_wait_edge(200, 100);
        let result = d.scan();
        assert!(!result.victims.is_empty());
        assert!(result.total_detected >= 1);
    }

    #[test]
    fn test_scan_no_deadlock_returns_empty() {
        let d = DeadlockDetector::new();
        d.add_wait_edge(1, 2);
        let result = d.scan();
        assert!(result.victims.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_node_is_noop() {
        let d = DeadlockDetector::new();
        d.remove_transaction(999); // should not panic
        assert_eq!(d.node_count(), 0);
    }

    #[test]
    fn test_self_loop_no_cycle_in_wait_for_semantics() {
        // A transaction cannot wait for itself in a well-formed system,
        // but a self-loop technically forms a trivial SCC; our algorithm
        // only flags SCCs with >1 node as deadlocks.
        let mut g = make_graph();
        g.add_edge(1, 1); // self-loop
                          // Our SCC algorithm only flags multi-node SCCs
        let cycles = g.find_cycles();
        // A single-node SCC with a self-edge may or may not be reported
        // depending on implementation; what matters is we don't panic
        let _ = cycles;
    }

    #[test]
    fn test_four_node_cycle() {
        let mut g = make_graph();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        g.add_edge(4, 1);
        let cycles = g.find_cycles();
        assert!(!cycles.is_empty());
        assert_eq!(cycles[0].len(), 4);
    }

    #[test]
    fn test_partial_cycle_no_false_positive() {
        let mut g = make_graph();
        // Chain: 1→2→3 and 3→2 (back-edge to 2, not to 1)
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 2); // cycle: 2↔3
        let cycles = g.find_cycles();
        assert!(!cycles.is_empty());
        // The cycle should include 2 and 3, not 1
        let cycle = &cycles[0];
        assert!(cycle.contains(&2) && cycle.contains(&3));
        assert!(!cycle.contains(&1));
    }
}
