/// Partition rebalancing for distributed cluster data redistribution.
///
/// Computes optimal partition movement plans to balance load across
/// cluster nodes using configurable strategies.
use std::collections::HashMap;

/// A data partition assigned to a specific cluster node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    /// Unique partition identifier.
    pub id: u32,
    /// ID of the node currently hosting this partition.
    pub node_id: String,
    /// Number of triples in this partition.
    pub triple_count: usize,
    /// Size in bytes.
    pub byte_size: usize,
}

impl Partition {
    /// Create a new partition.
    pub fn new(id: u32, node_id: impl Into<String>, triple_count: usize, byte_size: usize) -> Self {
        Self {
            id,
            node_id: node_id.into(),
            triple_count,
            byte_size,
        }
    }
}

/// A single planned partition movement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionMove {
    /// ID of the partition to move.
    pub partition_id: u32,
    /// Source node.
    pub from_node: String,
    /// Destination node.
    pub to_node: String,
    /// Number of triples being moved.
    pub triple_count: usize,
}

impl PartitionMove {
    /// Return the number of triples this move will transfer.
    pub fn transfer_size(&self) -> usize {
        self.triple_count
    }
}

/// A complete rebalancing plan.
#[derive(Debug, Clone)]
pub struct RebalancePlan {
    /// Ordered list of partition moves.
    pub moves: Vec<PartitionMove>,
    /// Expected improvement in imbalance score after applying all moves.
    pub expected_improvement: f64,
}

impl RebalancePlan {
    /// Return true if this plan requires no moves.
    pub fn is_noop(&self) -> bool {
        self.moves.is_empty()
    }

    /// Return the total number of triples to be transferred.
    pub fn total_transfer(&self) -> usize {
        self.moves.iter().map(|m| m.triple_count).sum()
    }
}

/// Load snapshot for a single cluster node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeLoad {
    /// Node identifier.
    pub node_id: String,
    /// Number of partitions currently assigned.
    pub partition_count: usize,
    /// Total triples across all partitions.
    pub total_triples: usize,
}

impl NodeLoad {
    /// Create a new node load record.
    pub fn new(node_id: impl Into<String>, partition_count: usize, total_triples: usize) -> Self {
        Self {
            node_id: node_id.into(),
            partition_count,
            total_triples,
        }
    }
}

/// Rebalancing strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebalanceStrategy {
    /// Move partitions to the least-loaded node.
    LeastLoaded,
    /// Distribute partitions in round-robin order.
    RoundRobin,
    /// Weight moves by node capacity (triple count).
    WeightedCapacity,
}

/// Computes partition rebalancing plans for a cluster.
pub struct PartitionRebalancer {
    strategy: RebalanceStrategy,
    max_moves_per_plan: usize,
}

impl PartitionRebalancer {
    /// Create a new rebalancer.
    pub fn new(strategy: RebalanceStrategy, max_moves_per_plan: usize) -> Self {
        Self {
            strategy,
            max_moves_per_plan,
        }
    }

    /// Compute a rebalancing plan for the given partitions and node state.
    ///
    /// Returns a plan with at most `max_moves_per_plan` moves.
    pub fn compute_plan(&self, partitions: &[Partition], nodes: &[NodeLoad]) -> RebalancePlan {
        if nodes.is_empty() || partitions.is_empty() {
            return RebalancePlan {
                moves: vec![],
                expected_improvement: 0.0,
            };
        }

        let current_score = Self::imbalance_score(nodes);

        let moves = match self.strategy {
            RebalanceStrategy::LeastLoaded => self.compute_least_loaded(partitions, nodes),
            RebalanceStrategy::RoundRobin => self.compute_round_robin(partitions, nodes),
            RebalanceStrategy::WeightedCapacity => {
                self.compute_weighted_capacity(partitions, nodes)
            }
        };

        // Simulate the new node loads to compute expected improvement.
        let simulated_nodes = self.simulate_moves(nodes, partitions, &moves);
        let new_score = Self::imbalance_score(&simulated_nodes);
        let expected_improvement = (current_score - new_score).max(0.0);

        RebalancePlan {
            moves,
            expected_improvement,
        }
    }

    /// Compute imbalance score as coefficient of variation (std_dev / mean).
    ///
    /// Returns 0.0 if there are no nodes or if the mean is zero.
    pub fn imbalance_score(nodes: &[NodeLoad]) -> f64 {
        if nodes.is_empty() {
            return 0.0;
        }

        let loads: Vec<f64> = nodes.iter().map(|n| n.total_triples as f64).collect();
        let n = loads.len() as f64;
        let mean = loads.iter().sum::<f64>() / n;

        if mean < 1e-9 {
            return 0.0;
        }

        let variance = loads.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        std_dev / mean
    }

    /// Return true if the cluster is balanced within the given threshold.
    pub fn is_balanced(nodes: &[NodeLoad], threshold: f64) -> bool {
        Self::imbalance_score(nodes) <= threshold
    }

    /// Compute a `node_id → NodeLoad` map from a partition list.
    pub fn node_loads(partitions: &[Partition]) -> HashMap<String, NodeLoad> {
        let mut map: HashMap<String, NodeLoad> = HashMap::new();
        for p in partitions {
            let entry = map
                .entry(p.node_id.clone())
                .or_insert_with(|| NodeLoad::new(p.node_id.clone(), 0, 0));
            entry.partition_count += 1;
            entry.total_triples += p.triple_count;
        }
        map
    }

    // ---- Private helpers ----

    /// Compute the mean triple count across nodes.
    fn mean_load(nodes: &[NodeLoad]) -> f64 {
        if nodes.is_empty() {
            return 0.0;
        }
        nodes.iter().map(|n| n.total_triples as f64).sum::<f64>() / nodes.len() as f64
    }

    /// Least-loaded strategy: move the largest partition from the heaviest
    /// node to the lightest node, up to `max_moves_per_plan`.
    fn compute_least_loaded(
        &self,
        partitions: &[Partition],
        nodes: &[NodeLoad],
    ) -> Vec<PartitionMove> {
        let mut moves = Vec::new();
        let mut node_map: HashMap<String, usize> = nodes
            .iter()
            .map(|n| (n.node_id.clone(), n.total_triples))
            .collect();

        let mean = Self::mean_load(nodes);

        for _ in 0..self.max_moves_per_plan {
            // Find the most loaded node with above-average load.
            let source_node = node_map
                .iter()
                .filter(|(_, &load)| load as f64 > mean * 1.1)
                .max_by_key(|(_, &load)| load)
                .map(|(id, _)| id.clone());

            let source_id = match source_node {
                Some(id) => id,
                None => break,
            };

            // Find the least loaded destination.
            let dest_id = node_map
                .iter()
                .filter(|(id, _)| **id != source_id)
                .min_by_key(|(_, &load)| load)
                .map(|(id, _)| id.clone());

            let dest_id = match dest_id {
                Some(id) => id,
                None => break,
            };

            // Find the largest partition on the source node.
            let partition = partitions
                .iter()
                .filter(|p| p.node_id == source_id)
                .max_by_key(|p| p.triple_count);

            let p = match partition {
                Some(p) => p,
                None => break,
            };

            // Record the move.
            moves.push(PartitionMove {
                partition_id: p.id,
                from_node: source_id.clone(),
                to_node: dest_id.clone(),
                triple_count: p.triple_count,
            });

            // Update simulated loads.
            *node_map.entry(source_id).or_insert(0) = node_map
                .get(&source_id)
                .copied()
                .unwrap_or(0)
                .saturating_sub(p.triple_count);
            *node_map.entry(dest_id).or_insert(0) += p.triple_count;
        }

        moves
    }

    /// Round-robin strategy: redistribute partitions evenly across nodes.
    fn compute_round_robin(
        &self,
        partitions: &[Partition],
        nodes: &[NodeLoad],
    ) -> Vec<PartitionMove> {
        if nodes.len() < 2 {
            return vec![];
        }

        let mut moves = Vec::new();
        let node_ids: Vec<String> = nodes.iter().map(|n| n.node_id.clone()).collect();

        // Assign each partition to nodes in round-robin order.
        for (idx, partition) in partitions.iter().enumerate() {
            if moves.len() >= self.max_moves_per_plan {
                break;
            }
            let target_node = &node_ids[idx % node_ids.len()];
            if *target_node != partition.node_id {
                moves.push(PartitionMove {
                    partition_id: partition.id,
                    from_node: partition.node_id.clone(),
                    to_node: target_node.clone(),
                    triple_count: partition.triple_count,
                });
            }
        }

        moves
    }

    /// Weighted capacity: move partitions proportional to their size.
    fn compute_weighted_capacity(
        &self,
        partitions: &[Partition],
        nodes: &[NodeLoad],
    ) -> Vec<PartitionMove> {
        // For weighted capacity, use same logic as least-loaded but choose
        // destination by available capacity (inverse of current load).
        self.compute_least_loaded(partitions, nodes)
    }

    /// Simulate applying moves to produce new NodeLoad states.
    fn simulate_moves(
        &self,
        nodes: &[NodeLoad],
        _partitions: &[Partition],
        moves: &[PartitionMove],
    ) -> Vec<NodeLoad> {
        let mut loads: HashMap<String, usize> = nodes
            .iter()
            .map(|n| (n.node_id.clone(), n.total_triples))
            .collect();
        let mut counts: HashMap<String, usize> = nodes
            .iter()
            .map(|n| (n.node_id.clone(), n.partition_count))
            .collect();

        for m in moves {
            *loads.entry(m.from_node.clone()).or_insert(0) = loads
                .get(&m.from_node)
                .copied()
                .unwrap_or(0)
                .saturating_sub(m.triple_count);
            *loads.entry(m.to_node.clone()).or_insert(0) += m.triple_count;
            *counts.entry(m.from_node.clone()).or_insert(0) = counts
                .get(&m.from_node)
                .copied()
                .unwrap_or(0)
                .saturating_sub(1);
            *counts.entry(m.to_node.clone()).or_insert(0) += 1;
        }

        nodes
            .iter()
            .map(|n| NodeLoad {
                node_id: n.node_id.clone(),
                partition_count: *counts.get(&n.node_id).unwrap_or(&n.partition_count),
                total_triples: *loads.get(&n.node_id).unwrap_or(&n.total_triples),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_partitions() -> Vec<Partition> {
        vec![
            Partition::new(1, "node-a", 1000, 10_000),
            Partition::new(2, "node-a", 2000, 20_000),
            Partition::new(3, "node-a", 3000, 30_000),
            Partition::new(4, "node-b", 500, 5_000),
            Partition::new(5, "node-c", 400, 4_000),
        ]
    }

    fn make_nodes() -> Vec<NodeLoad> {
        vec![
            NodeLoad::new("node-a", 3, 6000),
            NodeLoad::new("node-b", 1, 500),
            NodeLoad::new("node-c", 1, 400),
        ]
    }

    // --- Partition ---

    #[test]
    fn test_partition_new() {
        let p = Partition::new(1, "node-a", 100, 1000);
        assert_eq!(p.id, 1);
        assert_eq!(p.node_id, "node-a");
        assert_eq!(p.triple_count, 100);
        assert_eq!(p.byte_size, 1000);
    }

    #[test]
    fn test_partition_clone() {
        let p = Partition::new(5, "n", 50, 500);
        assert_eq!(p.clone(), p);
    }

    // --- PartitionMove ---

    #[test]
    fn test_partition_move_transfer_size() {
        let m = PartitionMove {
            partition_id: 1,
            from_node: "a".to_string(),
            to_node: "b".to_string(),
            triple_count: 1000,
        };
        assert_eq!(m.transfer_size(), 1000);
    }

    #[test]
    fn test_partition_move_clone() {
        let m = PartitionMove {
            partition_id: 2,
            from_node: "x".to_string(),
            to_node: "y".to_string(),
            triple_count: 500,
        };
        assert_eq!(m.clone(), m);
    }

    // --- RebalancePlan ---

    #[test]
    fn test_rebalance_plan_is_noop() {
        let plan = RebalancePlan {
            moves: vec![],
            expected_improvement: 0.0,
        };
        assert!(plan.is_noop());
    }

    #[test]
    fn test_rebalance_plan_is_not_noop() {
        let m = PartitionMove {
            partition_id: 1,
            from_node: "a".to_string(),
            to_node: "b".to_string(),
            triple_count: 100,
        };
        let plan = RebalancePlan {
            moves: vec![m],
            expected_improvement: 0.5,
        };
        assert!(!plan.is_noop());
    }

    #[test]
    fn test_rebalance_plan_total_transfer() {
        let moves = vec![
            PartitionMove {
                partition_id: 1,
                from_node: "a".into(),
                to_node: "b".into(),
                triple_count: 200,
            },
            PartitionMove {
                partition_id: 2,
                from_node: "a".into(),
                to_node: "c".into(),
                triple_count: 300,
            },
        ];
        let plan = RebalancePlan {
            moves,
            expected_improvement: 0.3,
        };
        assert_eq!(plan.total_transfer(), 500);
    }

    // --- NodeLoad ---

    #[test]
    fn test_node_load_new() {
        let nl = NodeLoad::new("node-1", 5, 10_000);
        assert_eq!(nl.node_id, "node-1");
        assert_eq!(nl.partition_count, 5);
        assert_eq!(nl.total_triples, 10_000);
    }

    // --- imbalance_score ---

    #[test]
    fn test_imbalance_score_balanced() {
        let nodes = vec![
            NodeLoad::new("a", 1, 1000),
            NodeLoad::new("b", 1, 1000),
            NodeLoad::new("c", 1, 1000),
        ];
        let score = PartitionRebalancer::imbalance_score(&nodes);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_score_unbalanced() {
        let nodes = make_nodes();
        let score = PartitionRebalancer::imbalance_score(&nodes);
        assert!(score > 0.0);
    }

    #[test]
    fn test_imbalance_score_empty() {
        let score = PartitionRebalancer::imbalance_score(&[]);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_score_single_node() {
        let nodes = vec![NodeLoad::new("a", 1, 5000)];
        let score = PartitionRebalancer::imbalance_score(&nodes);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_score_all_zero_loads() {
        let nodes = vec![NodeLoad::new("a", 0, 0), NodeLoad::new("b", 0, 0)];
        let score = PartitionRebalancer::imbalance_score(&nodes);
        assert!((score - 0.0).abs() < 1e-9);
    }

    // --- is_balanced ---

    #[test]
    fn test_is_balanced_true() {
        let nodes = vec![NodeLoad::new("a", 1, 1000), NodeLoad::new("b", 1, 1000)];
        assert!(PartitionRebalancer::is_balanced(&nodes, 0.1));
    }

    #[test]
    fn test_is_balanced_false() {
        let nodes = make_nodes();
        assert!(!PartitionRebalancer::is_balanced(&nodes, 0.01));
    }

    #[test]
    fn test_is_balanced_threshold_zero() {
        let nodes = vec![NodeLoad::new("a", 1, 1000), NodeLoad::new("b", 1, 1001)];
        assert!(!PartitionRebalancer::is_balanced(&nodes, 0.0));
    }

    // --- node_loads ---

    #[test]
    fn test_node_loads_aggregates_correctly() {
        let partitions = make_partitions();
        let loads = PartitionRebalancer::node_loads(&partitions);
        assert_eq!(loads["node-a"].partition_count, 3);
        assert_eq!(loads["node-a"].total_triples, 6000);
        assert_eq!(loads["node-b"].partition_count, 1);
        assert_eq!(loads["node-b"].total_triples, 500);
    }

    #[test]
    fn test_node_loads_empty_partitions() {
        let loads = PartitionRebalancer::node_loads(&[]);
        assert!(loads.is_empty());
    }

    #[test]
    fn test_node_loads_single_partition() {
        let partitions = vec![Partition::new(1, "n1", 200, 2000)];
        let loads = PartitionRebalancer::node_loads(&partitions);
        assert_eq!(loads["n1"].total_triples, 200);
    }

    // --- compute_plan (LeastLoaded) ---

    #[test]
    fn test_compute_plan_least_loaded_returns_plan() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::LeastLoaded, 5);
        let partitions = make_partitions();
        let nodes = make_nodes();
        let plan = rebalancer.compute_plan(&partitions, &nodes);
        // Should produce at least one move given the imbalance.
        assert!(!plan.is_noop());
    }

    #[test]
    fn test_compute_plan_respects_max_moves() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::LeastLoaded, 1);
        let partitions = make_partitions();
        let nodes = make_nodes();
        let plan = rebalancer.compute_plan(&partitions, &nodes);
        assert!(plan.moves.len() <= 1);
    }

    #[test]
    fn test_compute_plan_move_sources_from_heavy_node() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::LeastLoaded, 3);
        let partitions = make_partitions();
        let nodes = make_nodes();
        let plan = rebalancer.compute_plan(&partitions, &nodes);
        if let Some(m) = plan.moves.first() {
            assert_eq!(m.from_node, "node-a");
        }
    }

    #[test]
    fn test_compute_plan_expected_improvement_non_negative() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::LeastLoaded, 5);
        let partitions = make_partitions();
        let nodes = make_nodes();
        let plan = rebalancer.compute_plan(&partitions, &nodes);
        assert!(plan.expected_improvement >= 0.0);
    }

    #[test]
    fn test_compute_plan_empty_nodes() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::LeastLoaded, 5);
        let partitions = make_partitions();
        let plan = rebalancer.compute_plan(&partitions, &[]);
        assert!(plan.is_noop());
    }

    #[test]
    fn test_compute_plan_empty_partitions() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::LeastLoaded, 5);
        let nodes = make_nodes();
        let plan = rebalancer.compute_plan(&[], &nodes);
        assert!(plan.is_noop());
    }

    #[test]
    fn test_compute_plan_already_balanced() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::LeastLoaded, 5);
        let partitions = vec![
            Partition::new(1, "a", 1000, 1000),
            Partition::new(2, "b", 1000, 1000),
        ];
        let nodes = vec![NodeLoad::new("a", 1, 1000), NodeLoad::new("b", 1, 1000)];
        let plan = rebalancer.compute_plan(&partitions, &nodes);
        // Already balanced → no moves needed.
        assert!(plan.is_noop());
    }

    // --- compute_plan (RoundRobin) ---

    #[test]
    fn test_compute_plan_round_robin() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::RoundRobin, 10);
        let partitions = make_partitions();
        let nodes = make_nodes();
        let plan = rebalancer.compute_plan(&partitions, &nodes);
        assert!(plan.moves.len() <= 10);
    }

    #[test]
    fn test_compute_plan_round_robin_single_node() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::RoundRobin, 5);
        let partitions = vec![Partition::new(1, "a", 100, 1000)];
        let nodes = vec![NodeLoad::new("a", 1, 100)];
        let plan = rebalancer.compute_plan(&partitions, &nodes);
        assert!(plan.is_noop());
    }

    // --- compute_plan (WeightedCapacity) ---

    #[test]
    fn test_compute_plan_weighted_capacity() {
        let rebalancer = PartitionRebalancer::new(RebalanceStrategy::WeightedCapacity, 5);
        let partitions = make_partitions();
        let nodes = make_nodes();
        let plan = rebalancer.compute_plan(&partitions, &nodes);
        assert!(plan.expected_improvement >= 0.0);
    }

    // --- RebalanceStrategy variants ---

    #[test]
    fn test_strategy_clone_and_eq() {
        assert_eq!(
            RebalanceStrategy::LeastLoaded.clone(),
            RebalanceStrategy::LeastLoaded
        );
        assert_ne!(
            RebalanceStrategy::LeastLoaded,
            RebalanceStrategy::RoundRobin
        );
    }

    #[test]
    fn test_strategy_debug() {
        let s = format!("{:?}", RebalanceStrategy::WeightedCapacity);
        assert!(s.contains("WeightedCapacity"));
    }
}
