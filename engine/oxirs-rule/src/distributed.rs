//! Distributed Reasoning Foundations
//!
//! Provides infrastructure for distributed rule-based reasoning across multiple nodes.
//! Enables horizontal scaling of reasoning workloads.
//!
//! # Features
//!
//! - **Node Management**: Register and manage reasoning nodes
//! - **Work Distribution**: Partition facts and rules across nodes
//! - **Result Aggregation**: Collect and merge results from distributed execution
//! - **Fault Tolerance**: Handle node failures gracefully
//! - **Load Balancing**: Distribute work based on node capacity
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::distributed::{DistributedReasoner, Node, PartitionStrategy};
//! use oxirs_rule::RuleEngine;
//!
//! let reasoner = DistributedReasoner::new(PartitionStrategy::RoundRobin);
//!
//! // Register nodes
//! // reasoner.register_node(Node::new("node1", "localhost:8001"))?;
//! // reasoner.register_node(Node::new("node2", "localhost:8002"))?;
//!
//! // Execute distributed reasoning
//! // let results = reasoner.execute_distributed(&rules, &facts)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, RuleEngine};
use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Reasoning node in the distributed system
#[derive(Debug, Clone)]
pub struct Node {
    /// Node identifier
    pub id: String,
    /// Node address (host:port)
    pub address: String,
    /// Node status
    pub status: NodeStatus,
    /// Node capacity (facts per second)
    pub capacity: usize,
    /// Current load (0.0 - 1.0)
    pub load: f64,
}

impl Node {
    /// Create a new node
    pub fn new(id: String, address: String) -> Self {
        Self {
            id,
            address,
            status: NodeStatus::Available,
            capacity: 1000,
            load: 0.0,
        }
    }

    /// Set node capacity
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Check if node is available
    pub fn is_available(&self) -> bool {
        matches!(self.status, NodeStatus::Available) && self.load < 0.9
    }

    /// Update node load
    pub fn update_load(&mut self, load: f64) {
        self.load = load.clamp(0.0, 1.0);

        // Update status based on load
        if self.load > 0.95 {
            self.status = NodeStatus::Overloaded;
        } else if self.status == NodeStatus::Overloaded && self.load < 0.8 {
            self.status = NodeStatus::Available;
        }
    }
}

/// Node status
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    /// Node is available for work
    Available,
    /// Node is processing work
    Busy,
    /// Node is overloaded
    Overloaded,
    /// Node is offline or failed
    Offline,
}

/// Partitioning strategy for distributing work
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Hash-based partitioning
    HashBased,
    /// Load-balanced distribution
    LoadBalanced,
    /// Random distribution
    Random,
}

/// Work partition for a node
#[derive(Debug, Clone)]
pub struct WorkPartition {
    /// Node ID this partition is assigned to
    pub node_id: String,
    /// Rules to execute
    pub rules: Vec<Rule>,
    /// Facts to process
    pub facts: Vec<RuleAtom>,
    /// Partition ID
    pub partition_id: usize,
}

impl WorkPartition {
    /// Create a new work partition
    pub fn new(node_id: String, partition_id: usize) -> Self {
        Self {
            node_id,
            rules: Vec::new(),
            facts: Vec::new(),
            partition_id,
        }
    }

    /// Add rules to partition
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        self.rules.extend(rules);
    }

    /// Add facts to partition
    pub fn add_facts(&mut self, facts: Vec<RuleAtom>) {
        self.facts.extend(facts);
    }

    /// Get estimated work size
    pub fn work_size(&self) -> usize {
        self.facts.len() * self.rules.len()
    }
}

/// Result from a distributed reasoning task
#[derive(Debug, Clone)]
pub struct DistributedResult {
    /// Node that produced this result
    pub node_id: String,
    /// Derived facts
    pub facts: Vec<RuleAtom>,
    /// Execution time in milliseconds
    pub execution_time_ms: u128,
    /// Whether execution succeeded
    pub success: bool,
}

impl DistributedResult {
    /// Create a new result
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            facts: Vec::new(),
            execution_time_ms: 0,
            success: true,
        }
    }
}

/// Distributed reasoner coordinator
pub struct DistributedReasoner {
    /// Registered nodes
    nodes: HashMap<String, Node>,
    /// Partitioning strategy
    strategy: PartitionStrategy,
    /// Statistics
    stats: DistributedStats,
    /// Cached rules (optimization to avoid cloning rules repeatedly)
    cached_rules: Option<Vec<Rule>>,
    /// Min facts per partition (to avoid over-partitioning)
    min_facts_per_partition: usize,
}

impl DistributedReasoner {
    /// Create a new distributed reasoner
    pub fn new(strategy: PartitionStrategy) -> Self {
        Self {
            nodes: HashMap::new(),
            strategy,
            stats: DistributedStats::default(),
            cached_rules: None,
            // For simulated local execution, require large datasets to justify overhead
            // Real distributed systems would use lower thresholds (e.g., 50)
            min_facts_per_partition: 500,
        }
    }

    /// Set minimum facts per partition (to prevent over-partitioning)
    pub fn set_min_facts_per_partition(&mut self, min: usize) {
        self.min_facts_per_partition = min;
    }

    /// Register a node
    pub fn register_node(&mut self, node: Node) -> Result<()> {
        let id = node.id.clone();
        if self.nodes.contains_key(&id) {
            return Err(anyhow::anyhow!("Node '{}' already registered", id));
        }

        info!("Registering node '{}' at {}", id, node.address);
        self.nodes.insert(id, node);
        Ok(())
    }

    /// Unregister a node
    pub fn unregister_node(&mut self, node_id: &str) -> Option<Node> {
        info!("Unregistering node '{}'", node_id);
        self.nodes.remove(node_id)
    }

    /// Get available nodes
    pub fn get_available_nodes(&self) -> Vec<&Node> {
        self.nodes.values().filter(|n| n.is_available()).collect()
    }

    /// Partition work across available nodes
    pub fn partition_work(
        &mut self,
        rules: &[Rule],
        facts: &[RuleAtom],
    ) -> Result<Vec<WorkPartition>> {
        let available_nodes = self.get_available_nodes();

        if available_nodes.is_empty() {
            return Err(anyhow::anyhow!(
                "No available nodes for distributed reasoning"
            ));
        }

        debug!(
            "Partitioning work across {} nodes using {:?} strategy",
            available_nodes.len(),
            self.strategy
        );

        match self.strategy {
            PartitionStrategy::RoundRobin => self.partition_round_robin(rules, facts),
            PartitionStrategy::HashBased => self.partition_hash_based(rules, facts),
            PartitionStrategy::LoadBalanced => self.partition_load_balanced(rules, facts),
            PartitionStrategy::Random => self.partition_random(rules, facts),
        }
    }

    /// Round-robin partitioning
    fn partition_round_robin(
        &mut self,
        rules: &[Rule],
        facts: &[RuleAtom],
    ) -> Result<Vec<WorkPartition>> {
        let available_nodes = self.get_available_nodes();
        let node_count = available_nodes.len();

        let facts_per_node = (facts.len() + node_count - 1) / node_count;
        let mut partitions = Vec::new();

        for (i, node) in available_nodes.iter().enumerate() {
            let start = i * facts_per_node;
            let end = ((i + 1) * facts_per_node).min(facts.len());

            let mut partition = WorkPartition::new(node.id.clone(), i);
            partition.add_rules(rules.to_vec());
            partition.add_facts(facts[start..end].to_vec());

            partitions.push(partition);
        }

        Ok(partitions)
    }

    /// Hash-based partitioning
    fn partition_hash_based(
        &self,
        rules: &[Rule],
        facts: &[RuleAtom],
    ) -> Result<Vec<WorkPartition>> {
        let available_nodes = self.get_available_nodes();
        let node_count = available_nodes.len();

        let mut partitions: Vec<WorkPartition> = available_nodes
            .iter()
            .enumerate()
            .map(|(i, node)| WorkPartition::new(node.id.clone(), i))
            .collect();

        // Hash each fact to a partition
        for fact in facts {
            let hash = self.hash_atom(fact);
            let partition_idx = hash % node_count;
            partitions[partition_idx].add_facts(vec![fact.clone()]);
        }

        // Add all rules to all partitions
        for partition in &mut partitions {
            partition.add_rules(rules.to_vec());
        }

        Ok(partitions)
    }

    /// Load-balanced partitioning
    fn partition_load_balanced(
        &self,
        rules: &[Rule],
        facts: &[RuleAtom],
    ) -> Result<Vec<WorkPartition>> {
        let available_nodes = self.get_available_nodes();
        let _node_count = available_nodes.len();

        // Calculate capacity-based weights
        let total_capacity: usize = available_nodes.iter().map(|n| n.capacity).sum();

        let mut partitions = Vec::new();
        let mut fact_idx = 0;

        for (i, node) in available_nodes.iter().enumerate() {
            let weight = node.capacity as f64 / total_capacity as f64;
            let facts_for_node = ((facts.len() as f64) * weight).ceil() as usize;

            let end = (fact_idx + facts_for_node).min(facts.len());

            let mut partition = WorkPartition::new(node.id.clone(), i);
            partition.add_rules(rules.to_vec());
            partition.add_facts(facts[fact_idx..end].to_vec());

            partitions.push(partition);
            fact_idx = end;

            if fact_idx >= facts.len() {
                break;
            }
        }

        Ok(partitions)
    }

    /// Random partitioning
    fn partition_random(
        &mut self,
        rules: &[Rule],
        facts: &[RuleAtom],
    ) -> Result<Vec<WorkPartition>> {
        // For simplicity, use round-robin as fallback
        // In a full implementation, we would use random assignment
        self.partition_round_robin(rules, facts)
    }

    /// Simple hash function for atoms
    fn hash_atom(&self, atom: &RuleAtom) -> usize {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let mut hash = 0;
                hash ^= self.hash_term(subject);
                hash ^= self.hash_term(predicate);
                hash ^= self.hash_term(object);
                hash
            }
            _ => 0,
        }
    }

    fn hash_term(&self, term: &crate::Term) -> usize {
        match term {
            crate::Term::Constant(s) | crate::Term::Literal(s) | crate::Term::Variable(s) => s
                .bytes()
                .fold(0, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize)),
            _ => 0,
        }
    }

    /// Execute distributed reasoning (optimized)
    pub fn execute_distributed(
        &mut self,
        rules: &[Rule],
        facts: &[RuleAtom],
    ) -> Result<Vec<RuleAtom>> {
        self.stats.total_executions += 1;
        let start = std::time::Instant::now();

        // Check if distributed execution is worthwhile
        // Calculate how many partitions we would actually create
        let ideal_partitions = (facts.len() / self.min_facts_per_partition)
            .max(1)
            .min(self.nodes.len());

        if ideal_partitions <= 1 || self.nodes.is_empty() {
            // Not enough work to justify distribution overhead - use single engine
            debug!(
                "Workload too small for distribution ({} facts -> {} partitions) - using single engine",
                facts.len(),
                ideal_partitions
            );
            return self.execute_single_engine(rules, facts);
        }

        // Cache rules once (optimization to avoid repeated cloning)
        self.cached_rules = Some(rules.to_vec());

        // Partition work with smart sizing
        let partitions = self.partition_work_smart(rules, facts)?;
        self.stats.partitions_created += partitions.len();

        // Execute partitions with cached rules
        let results = self.execute_partitions_optimized(partitions)?;

        // Aggregate results
        let aggregated = self.aggregate_results(results)?;

        self.stats.total_time_ms += start.elapsed().as_millis();
        info!(
            "Distributed execution complete: {} facts derived across {} partitions",
            aggregated.len(),
            self.stats.partitions_created
        );

        Ok(aggregated)
    }

    /// Execute on single engine (fallback for small workloads)
    fn execute_single_engine(&self, rules: &[Rule], facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();
        for rule in rules {
            engine.add_rule(rule.clone());
        }
        engine.forward_chain(facts)
    }

    /// Smart partitioning that avoids over-partitioning
    fn partition_work_smart(
        &mut self,
        rules: &[Rule],
        facts: &[RuleAtom],
    ) -> Result<Vec<WorkPartition>> {
        // Calculate optimal number of partitions
        let ideal_partitions = (facts.len() / self.min_facts_per_partition).min(self.nodes.len());
        let effective_nodes = ideal_partitions.max(1);

        debug!(
            "Smart partitioning: {} facts across {} nodes (min {} facts/partition)",
            facts.len(),
            effective_nodes,
            self.min_facts_per_partition
        );

        // Use round-robin but with effective node count
        let available_nodes: Vec<_> = self
            .get_available_nodes()
            .into_iter()
            .take(effective_nodes)
            .collect();
        let node_count = available_nodes.len();

        if node_count == 0 {
            return Err(anyhow::anyhow!("No available nodes"));
        }

        let facts_per_node = (facts.len() + node_count - 1) / node_count;
        let mut partitions = Vec::new();

        for (i, node) in available_nodes.iter().enumerate() {
            let start = i * facts_per_node;
            let end = ((i + 1) * facts_per_node).min(facts.len());

            if start >= facts.len() {
                break;
            }

            let mut partition = WorkPartition::new(node.id.clone(), i);
            partition.add_rules(rules.to_vec());
            partition.add_facts(facts[start..end].to_vec());

            debug!(
                "Partition {} -> node '{}': {} facts",
                i,
                node.id,
                end - start
            );

            partitions.push(partition);
        }

        Ok(partitions)
    }

    /// Execute partitions with optimized rule handling
    fn execute_partitions_optimized(
        &mut self,
        partitions: Vec<WorkPartition>,
    ) -> Result<Vec<DistributedResult>> {
        let mut results = Vec::new();

        // Get cached rules once
        let rules = self
            .cached_rules
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Rules not cached"))?;

        for partition in partitions {
            debug!(
                "Executing partition {} on node '{}' ({} facts)",
                partition.partition_id,
                partition.node_id,
                partition.facts.len()
            );

            let start = std::time::Instant::now();

            // Create engine and add cached rules efficiently
            let mut engine = RuleEngine::new();
            for rule in rules {
                engine.add_rule(rule.clone());
            }

            let derived = engine.forward_chain(&partition.facts)?;

            let mut result = DistributedResult::new(partition.node_id.clone());
            result.facts = derived;
            result.execution_time_ms = start.elapsed().as_millis();
            result.success = true;

            // Update node load (simulated)
            if let Some(node) = self.nodes.get_mut(&partition.node_id) {
                let load = partition.work_size() as f64 / node.capacity as f64;
                node.update_load(load);
            }

            results.push(result);
            self.stats.successful_executions += 1;
        }

        Ok(results)
    }

    /// Aggregate results from distributed execution
    fn aggregate_results(&mut self, results: Vec<DistributedResult>) -> Result<Vec<RuleAtom>> {
        let mut all_facts = Vec::new();

        for result in results {
            if result.success {
                all_facts.extend(result.facts);
                debug!(
                    "Aggregating {} facts from node '{}' ({}ms)",
                    all_facts.len(),
                    result.node_id,
                    result.execution_time_ms
                );
            } else {
                warn!("Node '{}' execution failed", result.node_id);
                self.stats.failed_executions += 1;
            }
        }

        // Deduplicate facts
        all_facts.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        all_facts.dedup_by(|a, b| format!("{:?}", a) == format!("{:?}", b));

        Ok(all_facts)
    }

    /// Get statistics
    pub fn get_stats(&self) -> &DistributedStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DistributedStats::default();
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Distributed reasoning statistics
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    /// Total distributed executions
    pub total_executions: usize,
    /// Successful executions
    pub successful_executions: usize,
    /// Failed executions
    pub failed_executions: usize,
    /// Total partitions created
    pub partitions_created: usize,
    /// Total execution time in milliseconds
    pub total_time_ms: u128,
}

impl std::fmt::Display for DistributedStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Executions: {} (success: {}, failed: {}), Partitions: {}, Time: {}ms",
            self.total_executions,
            self.successful_executions,
            self.failed_executions,
            self.partitions_created,
            self.total_time_ms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_node_creation() {
        let node = Node::new("node1".to_string(), "localhost:8001".to_string());

        assert_eq!(node.id, "node1");
        assert_eq!(node.address, "localhost:8001");
        assert!(node.is_available());
    }

    #[test]
    fn test_node_load_update() {
        let mut node = Node::new("node1".to_string(), "localhost:8001".to_string());

        node.update_load(0.5);
        assert_eq!(node.load, 0.5);
        assert_eq!(node.status, NodeStatus::Available);

        node.update_load(0.96);
        assert_eq!(node.status, NodeStatus::Overloaded);
    }

    #[test]
    fn test_work_partition_creation() {
        let mut partition = WorkPartition::new("node1".to_string(), 0);

        partition.add_facts(vec![RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        }]);

        assert_eq!(partition.facts.len(), 1);
        assert_eq!(partition.work_size(), 0); // 1 fact * 0 rules
    }

    #[test]
    fn test_distributed_reasoner_creation() {
        let reasoner = DistributedReasoner::new(PartitionStrategy::RoundRobin);

        assert_eq!(reasoner.node_count(), 0);
    }

    #[test]
    fn test_node_registration() {
        let mut reasoner = DistributedReasoner::new(PartitionStrategy::RoundRobin);

        let node = Node::new("node1".to_string(), "localhost:8001".to_string());
        reasoner.register_node(node).unwrap();

        assert_eq!(reasoner.node_count(), 1);
    }

    #[test]
    fn test_node_unregistration() {
        let mut reasoner = DistributedReasoner::new(PartitionStrategy::RoundRobin);

        let node = Node::new("node1".to_string(), "localhost:8001".to_string());
        reasoner.register_node(node).unwrap();

        let removed = reasoner.unregister_node("node1");
        assert!(removed.is_some());
        assert_eq!(reasoner.node_count(), 0);
    }

    #[test]
    fn test_work_partitioning() {
        let mut reasoner = DistributedReasoner::new(PartitionStrategy::RoundRobin);

        reasoner
            .register_node(Node::new("node1".to_string(), "localhost:8001".to_string()))
            .unwrap();
        reasoner
            .register_node(Node::new("node2".to_string(), "localhost:8002".to_string()))
            .unwrap();

        let rules = vec![];
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("a".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Constant("b".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("c".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Constant("d".to_string()),
            },
        ];

        let partitions = reasoner.partition_work(&rules, &facts).unwrap();

        assert_eq!(partitions.len(), 2);
    }

    #[test]
    fn test_distributed_execution() {
        let mut reasoner = DistributedReasoner::new(PartitionStrategy::RoundRobin);

        reasoner
            .register_node(Node::new("node1".to_string(), "localhost:8001".to_string()))
            .unwrap();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![],
            head: vec![],
        };

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        }];

        let results = reasoner.execute_distributed(&[rule], &facts).unwrap();

        assert!(!results.is_empty());
    }
}
