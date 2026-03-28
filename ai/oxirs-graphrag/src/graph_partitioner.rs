//! Graph partitioning using greedy and label-propagation methods.
//!
//! Provides k-way graph partitioning for knowledge graphs, useful for
//! distributed SPARQL query planning, parallel community processing,
//! and load-balanced federated queries.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Assignment of a single node to a partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPartition {
    /// Node identifier (matches the input `nodes` slice).
    pub node_id: String,
    /// Zero-based partition index (0..k).
    pub partition: usize,
}

/// Result of a graph partitioning run.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// Per-node partition assignments.
    pub assignments: Vec<GraphPartition>,
    /// Number of partitions k.
    pub num_partitions: usize,
    /// Number of edges whose endpoints are in different partitions.
    pub cut_edges: usize,
    /// Balance score in [0, 1]. 1.0 = perfectly balanced.
    pub balance_score: f64,
}

/// Available partitioning algorithms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionMethod {
    /// Round-robin greedy assignment followed by a local-improvement step.
    Greedy,
    /// Iterative label propagation (majority vote of neighbour labels).
    LabelPropagation,
    /// Recursive bisection by BFS-cut (produces a binary-tree of cuts).
    Bisection,
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphPartitioner
// ─────────────────────────────────────────────────────────────────────────────

/// Configures and runs graph partitioning.
#[derive(Debug, Clone)]
pub struct GraphPartitioner {
    /// Number of partitions k.
    pub num_partitions: usize,
    /// Algorithm selection.
    pub method: PartitionMethod,
    /// Maximum iterations (used by LabelPropagation and improvement loops).
    pub max_iterations: usize,
}

impl GraphPartitioner {
    /// Create a partitioner with the default method (Greedy) and 20 iterations.
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions: num_partitions.max(1),
            method: PartitionMethod::Greedy,
            max_iterations: 20,
        }
    }

    /// Select the partitioning method.
    pub fn with_method(mut self, method: PartitionMethod) -> Self {
        self.method = method;
        self
    }

    /// Override the maximum iteration count.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Partition the graph and return a `PartitionResult`.
    ///
    /// # Arguments
    /// * `nodes` – list of node identifiers (must be unique).
    /// * `edges` – list of `(from, to)` string pairs; direction is ignored.
    pub fn partition(&self, nodes: &[String], edges: &[(String, String)]) -> PartitionResult {
        if nodes.is_empty() {
            return PartitionResult {
                assignments: vec![],
                num_partitions: self.num_partitions,
                cut_edges: 0,
                balance_score: 1.0,
            };
        }

        let k = self.num_partitions;

        let labels = match &self.method {
            PartitionMethod::Greedy => Self::greedy_partition(nodes, edges, k),
            PartitionMethod::LabelPropagation => {
                Self::label_propagation(nodes, edges, k, self.max_iterations)
            }
            PartitionMethod::Bisection => Self::bisection_partition(nodes, edges, k),
        };

        // Map node-string → index for cut-edge counting
        let node_idx: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        let int_edges: Vec<(usize, usize)> = edges
            .iter()
            .filter_map(|(a, b)| {
                let ai = node_idx.get(a.as_str())?;
                let bi = node_idx.get(b.as_str())?;
                Some((*ai, *bi))
            })
            .collect();

        let cut_edges = Self::count_cut_edges(&labels, &int_edges);
        let balance = Self::balance_score(&labels, k);

        let assignments = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| GraphPartition {
                node_id: n.clone(),
                partition: labels[i],
            })
            .collect();

        PartitionResult {
            assignments,
            num_partitions: k,
            cut_edges,
            balance_score: balance,
        }
    }

    // ── Algorithm implementations ─────────────────────────────────────────────

    /// Greedy round-robin assignment with a single local-improvement sweep.
    ///
    /// Returns a per-node partition vector (indices into `nodes`).
    pub fn greedy_partition(nodes: &[String], edges: &[(String, String)], k: usize) -> Vec<usize> {
        let k = k.max(1);
        let n = nodes.len();
        if n == 0 {
            return vec![];
        }

        // Initial round-robin assignment
        let mut labels: Vec<usize> = (0..n).map(|i| i % k).collect();

        // Build adjacency list
        let adj = Self::build_adjacency(nodes, edges);

        // One balance-constrained local-improvement pass: for each node, consider
        // moving it to the partition most common among its neighbours, but only
        // if the destination partition is not larger than a balanced target.
        let target_per_partition = (n + k - 1) / k; // ceiling division
        for i in 0..n {
            if adj[i].is_empty() {
                continue;
            }
            let mut counts = vec![0usize; k];
            for &nb in &adj[i] {
                counts[labels[nb]] += 1;
            }
            // Find best neighbour-majority partition that is not over-full
            let current_part = labels[i];
            let mut best_part = current_part;
            let mut best_count = counts[current_part];

            // Count current partition sizes
            let mut part_sizes = vec![0usize; k];
            for &l in labels.iter() {
                part_sizes[l.min(k - 1)] += 1;
            }

            for (p, &c) in counts.iter().enumerate() {
                if c > best_count && part_sizes[p] < target_per_partition {
                    best_part = p;
                    best_count = c;
                }
            }
            labels[i] = best_part;
        }
        labels
    }

    /// Iterative label propagation.
    ///
    /// Each node adopts the most frequent label among its neighbours.
    /// Labels are initialised in round-robin fashion and the propagation
    /// continues for at most `max_iter` rounds or until convergence.
    pub fn label_propagation(
        nodes: &[String],
        edges: &[(String, String)],
        k: usize,
        max_iter: usize,
    ) -> Vec<usize> {
        let k = k.max(1);
        let n = nodes.len();
        if n == 0 {
            return vec![];
        }

        let adj = Self::build_adjacency(nodes, edges);
        // Round-robin init
        let mut labels: Vec<usize> = (0..n).map(|i| i % k).collect();

        for _ in 0..max_iter {
            let mut changed = false;
            let prev = labels.clone();

            for i in 0..n {
                if adj[i].is_empty() {
                    continue;
                }
                let mut counts = vec![0usize; k];
                for &nb in &adj[i] {
                    counts[prev[nb]] += 1;
                }
                // Find label with max count; break ties by smallest label index
                let best = counts
                    .iter()
                    .enumerate()
                    .max_by(|(la, &ca), (lb, &cb)| ca.cmp(&cb).then(lb.cmp(la)))
                    .map(|(p, _)| p)
                    .unwrap_or(labels[i]);

                if best != labels[i] {
                    labels[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Ensure labels stay in [0, k)
        for l in &mut labels {
            *l = (*l).min(k - 1);
        }
        labels
    }

    /// Recursive bisection: repeatedly split the largest partition by BFS.
    pub fn bisection_partition(
        nodes: &[String],
        edges: &[(String, String)],
        k: usize,
    ) -> Vec<usize> {
        let k = k.max(1);
        let n = nodes.len();
        if n == 0 {
            return vec![];
        }

        let adj = Self::build_adjacency(nodes, edges);
        let mut labels = vec![0usize; n];

        // We produce up to k partitions by splitting iteratively
        let target_splits = k.saturating_sub(1);

        for (current_k, _) in (1usize..).zip(0..target_splits) {
            if current_k >= k {
                break;
            }
            // Find the partition with the most nodes
            let mut part_sizes = vec![0usize; current_k];
            for &l in &labels {
                part_sizes[l] += 1;
            }
            let largest_part = part_sizes
                .iter()
                .enumerate()
                .max_by_key(|(_, &s)| s)
                .map(|(p, _)| p)
                .unwrap_or(0);

            // Collect nodes in that partition
            let part_nodes: Vec<usize> = (0..n).filter(|&i| labels[i] == largest_part).collect();
            if part_nodes.len() < 2 {
                break;
            }

            // BFS from the first node; assign second half to new_part
            let half = part_nodes.len() / 2;
            let new_part = current_k;

            let mut bfs_order: Vec<usize> = Vec::with_capacity(part_nodes.len());
            let mut visited = vec![false; n];
            let mut queue = std::collections::VecDeque::new();
            let start = part_nodes[0];
            queue.push_back(start);
            visited[start] = true;

            while let Some(node) = queue.pop_front() {
                if labels[node] == largest_part {
                    bfs_order.push(node);
                }
                for &nb in &adj[node] {
                    if !visited[nb] && labels[nb] == largest_part {
                        visited[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }

            // Any nodes not reached by BFS are also in part_nodes
            for &pn in &part_nodes {
                if !bfs_order.contains(&pn) {
                    bfs_order.push(pn);
                }
            }

            // Move second half to new partition
            for &node in bfs_order.iter().skip(half) {
                labels[node] = new_part;
            }
        }

        labels
    }

    // ── Utility functions ─────────────────────────────────────────────────────

    /// Count edges whose endpoints are in different partitions.
    pub fn count_cut_edges(assignments: &[usize], edges: &[(usize, usize)]) -> usize {
        edges
            .iter()
            .filter(|&&(a, b)| {
                a < assignments.len() && b < assignments.len() && assignments[a] != assignments[b]
            })
            .count()
    }

    /// Balance score in [0, 1].  1.0 = perfectly balanced.
    ///
    /// Defined as `min_size / max_size` where sizes are partition cardinalities.
    /// Returns 1.0 for empty assignments or k=1.
    pub fn balance_score(assignments: &[usize], k: usize) -> f64 {
        if assignments.is_empty() || k <= 1 {
            return 1.0;
        }
        let mut counts = vec![0usize; k];
        for &l in assignments {
            let idx = l.min(k - 1);
            counts[idx] += 1;
        }
        let max = *counts.iter().max().unwrap_or(&0);
        let min = *counts.iter().min().unwrap_or(&0);
        if max == 0 {
            return 1.0;
        }
        min as f64 / max as f64
    }

    /// Build an undirected adjacency list indexed by position in `nodes`.
    pub fn build_adjacency(nodes: &[String], edges: &[(String, String)]) -> Vec<Vec<usize>> {
        let n = nodes.len();
        let node_idx: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        let mut adj = vec![vec![]; n];
        for (a, b) in edges {
            if let (Some(&ai), Some(&bi)) = (node_idx.get(a.as_str()), node_idx.get(b.as_str())) {
                if ai != bi {
                    adj[ai].push(bi);
                    adj[bi].push(ai);
                }
            }
        }
        adj
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nodes(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("node_{i}")).collect()
    }

    fn chain_edges(n: usize) -> Vec<(String, String)> {
        (0..n.saturating_sub(1))
            .map(|i| (format!("node_{i}"), format!("node_{}", i + 1)))
            .collect()
    }

    // ── GraphPartitioner::new ─────────────────────────────────────────────────

    #[test]
    fn test_new_default_method() {
        let gp = GraphPartitioner::new(4);
        assert_eq!(gp.num_partitions, 4);
        assert_eq!(gp.method, PartitionMethod::Greedy);
        assert_eq!(gp.max_iterations, 20);
    }

    #[test]
    fn test_new_zero_becomes_one() {
        let gp = GraphPartitioner::new(0);
        assert_eq!(gp.num_partitions, 1);
    }

    #[test]
    fn test_with_method_label_propagation() {
        let gp = GraphPartitioner::new(3).with_method(PartitionMethod::LabelPropagation);
        assert_eq!(gp.method, PartitionMethod::LabelPropagation);
    }

    #[test]
    fn test_with_max_iterations() {
        let gp = GraphPartitioner::new(3).with_max_iterations(50);
        assert_eq!(gp.max_iterations, 50);
    }

    // ── partition – empty input ────────────────────────────────────────────────

    #[test]
    fn test_partition_empty_nodes() {
        let gp = GraphPartitioner::new(3);
        let result = gp.partition(&[], &[]);
        assert!(result.assignments.is_empty());
        assert_eq!(result.cut_edges, 0);
        assert_eq!(result.balance_score, 1.0);
    }

    // ── partition – single node ───────────────────────────────────────────────

    #[test]
    fn test_partition_single_node() {
        let gp = GraphPartitioner::new(3);
        let nodes = vec!["A".to_string()];
        let result = gp.partition(&nodes, &[]);
        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.assignments[0].node_id, "A");
        assert_eq!(result.cut_edges, 0);
    }

    // ── partition – correct assignment count ──────────────────────────────────

    #[test]
    fn test_partition_returns_all_nodes() {
        let nodes = make_nodes(10);
        let edges = chain_edges(10);
        let gp = GraphPartitioner::new(3);
        let result = gp.partition(&nodes, &edges);
        assert_eq!(result.assignments.len(), 10);
    }

    #[test]
    fn test_partition_labels_in_range() {
        let nodes = make_nodes(12);
        let edges = chain_edges(12);
        let k = 4;
        let gp = GraphPartitioner::new(k);
        let result = gp.partition(&nodes, &edges);
        for a in &result.assignments {
            assert!(a.partition < k, "label {} out of range", a.partition);
        }
    }

    #[test]
    fn test_partition_num_partitions_field() {
        let nodes = make_nodes(6);
        let gp = GraphPartitioner::new(3);
        let result = gp.partition(&nodes, &[]);
        assert_eq!(result.num_partitions, 3);
    }

    // ── greedy_partition ──────────────────────────────────────────────────────

    #[test]
    fn test_greedy_partition_count() {
        let nodes = make_nodes(9);
        let edges = chain_edges(9);
        let labels = GraphPartitioner::greedy_partition(&nodes, &edges, 3);
        assert_eq!(labels.len(), 9);
    }

    #[test]
    fn test_greedy_partition_labels_valid() {
        let nodes = make_nodes(9);
        let edges = chain_edges(9);
        let labels = GraphPartitioner::greedy_partition(&nodes, &edges, 3);
        for &l in &labels {
            assert!(l < 3);
        }
    }

    #[test]
    fn test_greedy_partition_empty() {
        let labels = GraphPartitioner::greedy_partition(&[], &[], 3);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_greedy_partition_k1() {
        let nodes = make_nodes(5);
        let labels = GraphPartitioner::greedy_partition(&nodes, &[], 1);
        assert!(labels.iter().all(|&l| l == 0));
    }

    // ── label_propagation ─────────────────────────────────────────────────────

    #[test]
    fn test_label_propagation_count() {
        let nodes = make_nodes(8);
        let edges = chain_edges(8);
        let labels = GraphPartitioner::label_propagation(&nodes, &edges, 2, 10);
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_label_propagation_labels_valid() {
        let nodes = make_nodes(8);
        let edges = chain_edges(8);
        let labels = GraphPartitioner::label_propagation(&nodes, &edges, 4, 10);
        for &l in &labels {
            assert!(l < 4);
        }
    }

    #[test]
    fn test_label_propagation_empty() {
        let labels = GraphPartitioner::label_propagation(&[], &[], 3, 10);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_label_propagation_converges() {
        // With many iterations the result should still be valid
        let nodes = make_nodes(10);
        let edges = chain_edges(10);
        let labels = GraphPartitioner::label_propagation(&nodes, &edges, 3, 100);
        assert_eq!(labels.len(), 10);
        for &l in &labels {
            assert!(l < 3);
        }
    }

    // ── count_cut_edges ───────────────────────────────────────────────────────

    #[test]
    fn test_count_cut_edges_none() {
        // All in partition 0
        let assignments = vec![0, 0, 0];
        let edges = vec![(0, 1), (1, 2)];
        assert_eq!(GraphPartitioner::count_cut_edges(&assignments, &edges), 0);
    }

    #[test]
    fn test_count_cut_edges_all() {
        // Alternating partitions
        let assignments = vec![0, 1, 0, 1];
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        assert_eq!(GraphPartitioner::count_cut_edges(&assignments, &edges), 3);
    }

    #[test]
    fn test_count_cut_edges_empty() {
        assert_eq!(GraphPartitioner::count_cut_edges(&[], &[]), 0);
    }

    #[test]
    fn test_count_cut_edges_partial() {
        let assignments = vec![0, 0, 1, 1];
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        // edge (1,2) is cut
        assert_eq!(GraphPartitioner::count_cut_edges(&assignments, &edges), 1);
    }

    // ── balance_score ─────────────────────────────────────────────────────────

    #[test]
    fn test_balance_score_perfect() {
        let assignments = vec![0, 1, 0, 1];
        let score = GraphPartitioner::balance_score(&assignments, 2);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_balance_score_empty() {
        assert!((GraphPartitioner::balance_score(&[], 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_balance_score_k1() {
        let assignments = vec![0, 0, 0];
        assert!((GraphPartitioner::balance_score(&assignments, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_balance_score_imbalanced() {
        // 1 node in part 0, 3 in part 1 → min/max = 1/3
        let assignments = vec![0, 1, 1, 1];
        let score = GraphPartitioner::balance_score(&assignments, 2);
        assert!((score - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_balance_score_in_range() {
        let assignments = vec![0, 1, 2, 0, 1, 2, 0];
        let score = GraphPartitioner::balance_score(&assignments, 3);
        assert!((0.0..=1.0).contains(&score));
    }

    // ── build_adjacency ───────────────────────────────────────────────────────

    #[test]
    fn test_build_adjacency_empty() {
        let adj = GraphPartitioner::build_adjacency(&[], &[]);
        assert!(adj.is_empty());
    }

    #[test]
    fn test_build_adjacency_chain() {
        let nodes = make_nodes(3);
        let edges = chain_edges(3);
        let adj = GraphPartitioner::build_adjacency(&nodes, &edges);
        assert_eq!(adj.len(), 3);
        assert!(adj[0].contains(&1));
        assert!(adj[1].contains(&0));
        assert!(adj[1].contains(&2));
        assert!(adj[2].contains(&1));
    }

    #[test]
    fn test_build_adjacency_no_self_loops() {
        let nodes = vec!["A".to_string()];
        let edges = vec![("A".to_string(), "A".to_string())];
        let adj = GraphPartitioner::build_adjacency(&nodes, &edges);
        assert!(adj[0].is_empty());
    }

    #[test]
    fn test_build_adjacency_unknown_node_ignored() {
        let nodes = make_nodes(2);
        // edge references node_99 which does not exist
        let edges = vec![("node_0".to_string(), "node_99".to_string())];
        let adj = GraphPartitioner::build_adjacency(&nodes, &edges);
        assert!(adj[0].is_empty());
    }

    // ── bisection partition ───────────────────────────────────────────────────

    #[test]
    fn test_bisection_labels_valid() {
        let nodes = make_nodes(8);
        let edges = chain_edges(8);
        let labels = GraphPartitioner::bisection_partition(&nodes, &edges, 4);
        for &l in &labels {
            assert!(l < 4);
        }
    }

    #[test]
    fn test_bisection_count() {
        let nodes = make_nodes(6);
        let labels = GraphPartitioner::bisection_partition(&nodes, &[], 3);
        assert_eq!(labels.len(), 6);
    }

    // ── LabelPropagation method end-to-end ───────────────────────────────────

    #[test]
    fn test_partition_label_propagation_method() {
        let nodes = make_nodes(10);
        let edges = chain_edges(10);
        let gp = GraphPartitioner::new(2).with_method(PartitionMethod::LabelPropagation);
        let result = gp.partition(&nodes, &edges);
        assert_eq!(result.assignments.len(), 10);
        assert!(result.balance_score >= 0.0 && result.balance_score <= 1.0);
    }

    // ── Bisection method end-to-end ───────────────────────────────────────────

    #[test]
    fn test_partition_bisection_method() {
        let nodes = make_nodes(8);
        let edges = chain_edges(8);
        let gp = GraphPartitioner::new(4).with_method(PartitionMethod::Bisection);
        let result = gp.partition(&nodes, &edges);
        assert_eq!(result.assignments.len(), 8);
        for a in &result.assignments {
            assert!(a.partition < 4);
        }
    }

    // ── GraphPartition struct ─────────────────────────────────────────────────

    #[test]
    fn test_graph_partition_fields() {
        let gp = GraphPartition {
            node_id: "A".to_string(),
            partition: 2,
        };
        assert_eq!(gp.node_id, "A");
        assert_eq!(gp.partition, 2);
    }

    #[test]
    fn test_graph_partition_clone() {
        let gp = GraphPartition {
            node_id: "X".to_string(),
            partition: 1,
        };
        let gp2 = gp.clone();
        assert_eq!(gp, gp2);
    }

    // ── Dense fully-connected graph ───────────────────────────────────────────

    #[test]
    fn test_fully_connected_partition() {
        let nodes = make_nodes(4);
        let edges: Vec<(String, String)> = vec![
            ("node_0".to_string(), "node_1".to_string()),
            ("node_0".to_string(), "node_2".to_string()),
            ("node_0".to_string(), "node_3".to_string()),
            ("node_1".to_string(), "node_2".to_string()),
            ("node_1".to_string(), "node_3".to_string()),
            ("node_2".to_string(), "node_3".to_string()),
        ];
        let gp = GraphPartitioner::new(2);
        let result = gp.partition(&nodes, &edges);
        assert_eq!(result.assignments.len(), 4);
        // Some edges must be cut for k=2 in a fully connected graph
        assert!(result.cut_edges > 0);
    }

    // ── PartitionResult fields ────────────────────────────────────────────────

    #[test]
    fn test_partition_result_fields() {
        let nodes = make_nodes(6);
        let edges = chain_edges(6);
        let gp = GraphPartitioner::new(2);
        let result = gp.partition(&nodes, &edges);
        assert_eq!(result.num_partitions, 2);
        assert!(result.balance_score >= 0.0 && result.balance_score <= 1.0);
    }

    #[test]
    fn test_partition_result_assignment_node_ids() {
        let nodes = make_nodes(4);
        let gp = GraphPartitioner::new(2);
        let result = gp.partition(&nodes, &[]);
        let ids: Vec<&str> = result
            .assignments
            .iter()
            .map(|a| a.node_id.as_str())
            .collect();
        assert!(ids.contains(&"node_0"));
        assert!(ids.contains(&"node_3"));
    }

    // ── greedy with no edges ──────────────────────────────────────────────────

    #[test]
    fn test_greedy_no_edges() {
        let nodes = make_nodes(6);
        let labels = GraphPartitioner::greedy_partition(&nodes, &[], 3);
        assert_eq!(labels.len(), 6);
        for &l in &labels {
            assert!(l < 3);
        }
    }

    // ── label propagation with k > node count ────────────────────────────────

    #[test]
    fn test_label_propagation_k_larger_than_n() {
        let nodes = make_nodes(3);
        let edges = chain_edges(3);
        let labels = GraphPartitioner::label_propagation(&nodes, &edges, 10, 5);
        assert_eq!(labels.len(), 3);
        for &l in &labels {
            assert!(l < 10);
        }
    }

    // ── balance score all same partition ─────────────────────────────────────

    #[test]
    fn test_balance_score_all_same() {
        let assignments = vec![0, 0, 0, 0];
        // Only partition 0 has nodes; partition 1 has 0 → min=0, max=4 → 0.0
        let score = GraphPartitioner::balance_score(&assignments, 2);
        assert_eq!(score, 0.0);
    }

    // ── cut edges out-of-range index ─────────────────────────────────────────

    #[test]
    fn test_count_cut_edges_out_of_range() {
        let assignments = vec![0, 1];
        // edge (0, 5) — index 5 is out of range, should be ignored
        let edges = vec![(0usize, 5usize)];
        assert_eq!(GraphPartitioner::count_cut_edges(&assignments, &edges), 0);
    }

    // ── bisection single node ─────────────────────────────────────────────────

    #[test]
    fn test_bisection_single_node() {
        let nodes = vec!["A".to_string()];
        let labels = GraphPartitioner::bisection_partition(&nodes, &[], 2);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], 0);
    }

    // ── PartitionMethod debug ─────────────────────────────────────────────────

    #[test]
    fn test_partition_method_debug() {
        let m = PartitionMethod::LabelPropagation;
        let s = format!("{m:?}");
        assert!(s.contains("LabelPropagation"));
    }
}
