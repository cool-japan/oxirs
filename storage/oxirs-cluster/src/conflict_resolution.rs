//! # Advanced Conflict Resolution for Distributed Operations
//!
//! This module provides sophisticated conflict resolution mechanisms for distributed RDF operations,
//! including vector clocks, operational transforms, and semantic conflict detection.

use crate::raft::OxirsNodeId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Vector clock for tracking causality in distributed operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorClock {
    /// Clock values per node
    pub clocks: BTreeMap<OxirsNodeId, u64>,
}

impl VectorClock {
    /// Create a new vector clock
    pub fn new() -> Self {
        Self {
            clocks: BTreeMap::new(),
        }
    }

    /// Increment clock for a specific node
    pub fn increment(&mut self, node_id: OxirsNodeId) {
        let counter = self.clocks.entry(node_id).or_insert(0);
        *counter += 1;
    }

    /// Update clock based on received vector clock
    pub fn update(&mut self, other: &VectorClock) {
        for (node_id, other_time) in &other.clocks {
            let my_time = self.clocks.entry(*node_id).or_insert(0);
            *my_time = (*my_time).max(*other_time);
        }
    }

    /// Check if this clock happens before another
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut all_less_or_equal = true;
        let mut at_least_one_less = false;

        // Get all node IDs from both clocks
        let all_nodes: HashSet<_> = self.clocks.keys().chain(other.clocks.keys()).collect();

        for node_id in all_nodes {
            let my_time = self.clocks.get(node_id).unwrap_or(&0);
            let other_time = other.clocks.get(node_id).unwrap_or(&0);

            if my_time > other_time {
                all_less_or_equal = false;
                break;
            }
            if my_time < other_time {
                at_least_one_less = true;
            }
        }

        all_less_or_equal && at_least_one_less
    }

    /// Check if this clock is concurrent with another
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self != other
    }

    /// Merge two vector clocks
    pub fn merge(&self, other: &VectorClock) -> VectorClock {
        let mut result = self.clone();
        result.update(other);
        result
    }
}

/// Distributed operation with vector clock
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimestampedOperation {
    /// Unique operation identifier
    pub operation_id: String,
    /// Node that originated the operation
    pub origin_node: OxirsNodeId,
    /// Vector clock at time of operation
    pub vector_clock: VectorClock,
    /// Physical timestamp
    pub physical_time: SystemTime,
    /// The actual RDF operation
    pub operation: RdfOperation,
    /// Operation priority (higher values have priority)
    pub priority: u32,
}

/// RDF operation types for conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RdfOperation {
    /// Insert a triple
    Insert {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    },
    /// Delete a triple
    Delete {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    },
    /// Update triple (delete old, insert new)
    Update {
        old_triple: (String, String, String),
        new_triple: (String, String, String),
        graph: Option<String>,
    },
    /// Clear graph or entire store
    Clear { graph: Option<String> },
    /// Batch operation
    Batch { operations: Vec<RdfOperation> },
}

/// Conflict types in distributed RDF operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictType {
    /// Write-write conflict on same triple
    WriteWrite {
        operation1: TimestampedOperation,
        operation2: TimestampedOperation,
    },
    /// Delete-update conflict
    DeleteUpdate {
        delete_op: TimestampedOperation,
        update_op: TimestampedOperation,
    },
    /// Semantic conflict (violates constraints)
    Semantic {
        conflicting_ops: Vec<TimestampedOperation>,
        constraint_violation: String,
    },
    /// Clear conflicts with any other operation
    Clear {
        clear_op: TimestampedOperation,
        conflicting_ops: Vec<TimestampedOperation>,
    },
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResolutionStrategy {
    /// Last writer wins based on vector clock causality
    LastWriterWins,
    /// First writer wins (reject later conflicting operations)
    FirstWriterWins,
    /// Priority-based resolution
    PriorityBased,
    /// Node-based resolution (prefer specific nodes)
    NodePriority {
        node_priorities: HashMap<OxirsNodeId, u32>,
    },
    /// Semantic resolution with application-specific rules
    SemanticResolution { resolution_rules: Vec<SemanticRule> },
    /// Custom resolution function
    Custom { resolver_name: String },
    /// Manual resolution required
    Manual,
}

/// Semantic resolution rule
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule description
    pub description: String,
    /// Pattern to match operations
    pub pattern: OperationPattern,
    /// Resolution action
    pub action: ResolutionAction,
}

/// Pattern for matching operations in semantic rules
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OperationPattern {
    /// Subject pattern (supports wildcards)
    pub subject_pattern: Option<String>,
    /// Predicate pattern
    pub predicate_pattern: Option<String>,
    /// Object pattern
    pub object_pattern: Option<String>,
    /// Operation type filter
    pub operation_type: Option<OperationType>,
}

/// Simplified operation type for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OperationType {
    Insert,
    Delete,
    Update,
    Clear,
}

/// Resolution action for semantic rules
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResolutionAction {
    /// Accept the first operation, reject others
    AcceptFirst,
    /// Accept the last operation, reject others
    AcceptLast,
    /// Accept operation with highest priority
    AcceptHighestPriority,
    /// Merge operations if possible
    Merge,
    /// Reject all conflicting operations
    RejectAll,
    /// Custom action
    Custom { action_name: String },
}

/// Result of conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolutionResult {
    /// Original conflicting operations
    pub conflicting_operations: Vec<TimestampedOperation>,
    /// Resolved operations to apply
    pub resolved_operations: Vec<TimestampedOperation>,
    /// Operations that were rejected
    pub rejected_operations: Vec<TimestampedOperation>,
    /// Resolution strategy used
    pub strategy_used: ResolutionStrategy,
    /// Additional metadata about the resolution
    pub metadata: HashMap<String, String>,
}

/// Advanced conflict resolver
#[derive(Debug)]
pub struct ConflictResolver {
    /// Default resolution strategy
    default_strategy: ResolutionStrategy,
    /// Strategy overrides for specific patterns
    strategy_overrides: Vec<(OperationPattern, ResolutionStrategy)>,
    /// Semantic rules for conflict resolution
    semantic_rules: Vec<SemanticRule>,
    /// Node priorities for node-based resolution
    node_priorities: HashMap<OxirsNodeId, u32>,
    /// Statistics and metrics
    resolution_stats: Arc<RwLock<ResolutionStatistics>>,
}

/// Statistics for conflict resolution
#[derive(Debug, Default, Clone)]
pub struct ResolutionStatistics {
    /// Total conflicts resolved
    pub total_conflicts: u64,
    /// Conflicts by type
    pub conflicts_by_type: HashMap<String, u64>,
    /// Resolution strategies used
    pub strategies_used: HashMap<String, u64>,
    /// Average resolution time
    pub average_resolution_time: Duration,
    /// Success rate (resolved vs manual)
    pub success_rate: f64,
}

impl ConflictResolver {
    /// Create a new conflict resolver
    pub fn new(default_strategy: ResolutionStrategy) -> Self {
        Self {
            default_strategy,
            strategy_overrides: Vec::new(),
            semantic_rules: Vec::new(),
            node_priorities: HashMap::new(),
            resolution_stats: Arc::new(RwLock::new(ResolutionStatistics::default())),
        }
    }

    /// Add a strategy override for specific patterns
    pub fn add_strategy_override(
        &mut self,
        pattern: OperationPattern,
        strategy: ResolutionStrategy,
    ) {
        self.strategy_overrides.push((pattern, strategy));
    }

    /// Add a semantic rule
    pub fn add_semantic_rule(&mut self, rule: SemanticRule) {
        self.semantic_rules.push(rule);
    }

    /// Set node priority for node-based resolution
    pub fn set_node_priority(&mut self, node_id: OxirsNodeId, priority: u32) {
        self.node_priorities.insert(node_id, priority);
    }

    /// Detect conflicts between operations
    pub async fn detect_conflicts(
        &self,
        operations: &[TimestampedOperation],
    ) -> Result<Vec<ConflictType>> {
        let mut conflicts = Vec::new();

        // Check for write-write conflicts
        for i in 0..operations.len() {
            for j in (i + 1)..operations.len() {
                let op1 = &operations[i];
                let op2 = &operations[j];

                if let Some(conflict) = self.check_operation_conflict(op1, op2).await? {
                    conflicts.push(conflict);
                }
            }
        }

        // Check for semantic conflicts
        let semantic_conflicts = self.check_semantic_conflicts(operations).await?;
        conflicts.extend(semantic_conflicts);

        Ok(conflicts)
    }

    /// Resolve conflicts using configured strategies
    pub async fn resolve_conflicts(
        &self,
        conflicts: &[ConflictType],
    ) -> Result<Vec<ResolutionResult>> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();

        for conflict in conflicts {
            let result = self.resolve_single_conflict(conflict).await?;
            results.push(result);
        }

        // Update statistics
        let resolution_time = start_time.elapsed();
        self.update_statistics(&results, resolution_time).await;

        Ok(results)
    }

    /// Check for conflict between two operations
    async fn check_operation_conflict(
        &self,
        op1: &TimestampedOperation,
        op2: &TimestampedOperation,
    ) -> Result<Option<ConflictType>> {
        // Skip if operations are causally ordered
        if op1.vector_clock.happens_before(&op2.vector_clock)
            || op2.vector_clock.happens_before(&op1.vector_clock)
        {
            return Ok(None);
        }

        match (&op1.operation, &op2.operation) {
            // Write-write conflicts
            (
                RdfOperation::Insert {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                    graph: g1,
                },
                RdfOperation::Insert {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                    graph: g2,
                },
            ) => {
                if s1 == s2 && p1 == p2 && o1 != o2 && g1 == g2 {
                    Ok(Some(ConflictType::WriteWrite {
                        operation1: op1.clone(),
                        operation2: op2.clone(),
                    }))
                } else {
                    Ok(None)
                }
            }

            // Delete-update conflicts
            (
                RdfOperation::Delete {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                    graph: g1,
                },
                RdfOperation::Update {
                    old_triple: (s2, p2, o2),
                    graph: g2,
                    ..
                },
            ) => {
                if s1 == s2 && p1 == p2 && o1 == o2 && g1 == g2 {
                    Ok(Some(ConflictType::DeleteUpdate {
                        delete_op: op1.clone(),
                        update_op: op2.clone(),
                    }))
                } else {
                    Ok(None)
                }
            }

            // Clear conflicts
            (RdfOperation::Clear { graph: g1 }, _) | (_, RdfOperation::Clear { graph: g1 }) => {
                let clear_op = if matches!(op1.operation, RdfOperation::Clear { .. }) {
                    op1.clone()
                } else {
                    op2.clone()
                };
                let other_op = if matches!(op1.operation, RdfOperation::Clear { .. }) {
                    op2.clone()
                } else {
                    op1.clone()
                };

                Ok(Some(ConflictType::Clear {
                    clear_op,
                    conflicting_ops: vec![other_op],
                }))
            }

            _ => Ok(None),
        }
    }

    /// Check for semantic conflicts
    async fn check_semantic_conflicts(
        &self,
        operations: &[TimestampedOperation],
    ) -> Result<Vec<ConflictType>> {
        let mut conflicts = Vec::new();

        // Apply semantic rules
        for rule in &self.semantic_rules {
            let matching_ops: Vec<_> = operations
                .iter()
                .filter(|op| self.operation_matches_pattern(&op.operation, &rule.pattern))
                .cloned()
                .collect();

            if matching_ops.len() > 1 {
                conflicts.push(ConflictType::Semantic {
                    conflicting_ops: matching_ops,
                    constraint_violation: rule.description.clone(),
                });
            }
        }

        Ok(conflicts)
    }

    /// Resolve a single conflict
    async fn resolve_single_conflict(&self, conflict: &ConflictType) -> Result<ResolutionResult> {
        let strategy = self.select_resolution_strategy(conflict).await;

        match conflict {
            ConflictType::WriteWrite {
                operation1,
                operation2,
            } => {
                self.resolve_write_write_conflict(operation1, operation2, &strategy)
                    .await
            }
            ConflictType::DeleteUpdate {
                delete_op,
                update_op,
            } => {
                self.resolve_delete_update_conflict(delete_op, update_op, &strategy)
                    .await
            }
            ConflictType::Semantic {
                conflicting_ops,
                constraint_violation,
            } => {
                self.resolve_semantic_conflict(conflicting_ops, constraint_violation, &strategy)
                    .await
            }
            ConflictType::Clear {
                clear_op,
                conflicting_ops,
            } => {
                self.resolve_clear_conflict(clear_op, conflicting_ops, &strategy)
                    .await
            }
        }
    }

    /// Select appropriate resolution strategy
    async fn select_resolution_strategy(&self, conflict: &ConflictType) -> ResolutionStrategy {
        // Check for strategy overrides first
        let operations = match conflict {
            ConflictType::WriteWrite {
                operation1,
                operation2,
            } => vec![operation1, operation2],
            ConflictType::DeleteUpdate {
                delete_op,
                update_op,
            } => vec![delete_op, update_op],
            ConflictType::Semantic {
                conflicting_ops, ..
            } => conflicting_ops.iter().collect(),
            ConflictType::Clear {
                clear_op,
                conflicting_ops,
            } => {
                let mut ops = vec![clear_op];
                ops.extend(conflicting_ops.iter());
                ops
            }
        };

        // Check strategy overrides
        for op in &operations {
            for (pattern, strategy) in &self.strategy_overrides {
                if self.operation_matches_pattern(&op.operation, pattern) {
                    return strategy.clone();
                }
            }
        }

        // Use default strategy
        self.default_strategy.clone()
    }

    /// Resolve write-write conflict
    async fn resolve_write_write_conflict(
        &self,
        op1: &TimestampedOperation,
        op2: &TimestampedOperation,
        strategy: &ResolutionStrategy,
    ) -> Result<ResolutionResult> {
        let (resolved, rejected) = match strategy {
            ResolutionStrategy::LastWriterWins => {
                if op1.physical_time >= op2.physical_time {
                    (vec![op1.clone()], vec![op2.clone()])
                } else {
                    (vec![op2.clone()], vec![op1.clone()])
                }
            }
            ResolutionStrategy::FirstWriterWins => {
                if op1.physical_time <= op2.physical_time {
                    (vec![op1.clone()], vec![op2.clone()])
                } else {
                    (vec![op2.clone()], vec![op1.clone()])
                }
            }
            ResolutionStrategy::PriorityBased => {
                if op1.priority >= op2.priority {
                    (vec![op1.clone()], vec![op2.clone()])
                } else {
                    (vec![op2.clone()], vec![op1.clone()])
                }
            }
            ResolutionStrategy::NodePriority { node_priorities } => {
                let priority1 = node_priorities.get(&op1.origin_node).unwrap_or(&0);
                let priority2 = node_priorities.get(&op2.origin_node).unwrap_or(&0);

                if priority1 >= priority2 {
                    (vec![op1.clone()], vec![op2.clone()])
                } else {
                    (vec![op2.clone()], vec![op1.clone()])
                }
            }
            _ => {
                // Default to last writer wins
                if op1.physical_time >= op2.physical_time {
                    (vec![op1.clone()], vec![op2.clone()])
                } else {
                    (vec![op2.clone()], vec![op1.clone()])
                }
            }
        };

        Ok(ResolutionResult {
            conflicting_operations: vec![op1.clone(), op2.clone()],
            resolved_operations: resolved,
            rejected_operations: rejected,
            strategy_used: strategy.clone(),
            metadata: HashMap::new(),
        })
    }

    /// Resolve delete-update conflict
    async fn resolve_delete_update_conflict(
        &self,
        delete_op: &TimestampedOperation,
        update_op: &TimestampedOperation,
        strategy: &ResolutionStrategy,
    ) -> Result<ResolutionResult> {
        let (resolved, rejected) = match strategy {
            ResolutionStrategy::LastWriterWins => {
                if delete_op.physical_time >= update_op.physical_time {
                    (vec![delete_op.clone()], vec![update_op.clone()])
                } else {
                    (vec![update_op.clone()], vec![delete_op.clone()])
                }
            }
            _ => {
                // Default: delete wins
                (vec![delete_op.clone()], vec![update_op.clone()])
            }
        };

        Ok(ResolutionResult {
            conflicting_operations: vec![delete_op.clone(), update_op.clone()],
            resolved_operations: resolved,
            rejected_operations: rejected,
            strategy_used: strategy.clone(),
            metadata: HashMap::new(),
        })
    }

    /// Resolve semantic conflict
    async fn resolve_semantic_conflict(
        &self,
        conflicting_ops: &[TimestampedOperation],
        _constraint_violation: &str,
        strategy: &ResolutionStrategy,
    ) -> Result<ResolutionResult> {
        let (resolved, rejected) = match strategy {
            ResolutionStrategy::SemanticResolution { resolution_rules } => {
                // Apply semantic rules
                let mut resolved = Vec::new();
                let mut rejected = conflicting_ops.to_vec();

                for rule in resolution_rules {
                    match &rule.action {
                        ResolutionAction::AcceptFirst => {
                            if let Some(first_op) = conflicting_ops.first() {
                                resolved = vec![first_op.clone()];
                                rejected = conflicting_ops[1..].to_vec();
                            }
                            break;
                        }
                        ResolutionAction::AcceptLast => {
                            if let Some(last_op) = conflicting_ops.last() {
                                resolved = vec![last_op.clone()];
                                rejected = conflicting_ops[..conflicting_ops.len() - 1].to_vec();
                            }
                            break;
                        }
                        ResolutionAction::AcceptHighestPriority => {
                            if let Some(highest_priority_op) =
                                conflicting_ops.iter().max_by_key(|op| op.priority)
                            {
                                resolved = vec![highest_priority_op.clone()];
                                rejected = conflicting_ops
                                    .iter()
                                    .filter(|op| {
                                        op.operation_id != highest_priority_op.operation_id
                                    })
                                    .cloned()
                                    .collect();
                            }
                            break;
                        }
                        ResolutionAction::RejectAll => {
                            resolved = Vec::new();
                            rejected = conflicting_ops.to_vec();
                            break;
                        }
                        _ => continue,
                    }
                }

                (resolved, rejected)
            }
            _ => {
                // Default: reject all
                (Vec::new(), conflicting_ops.to_vec())
            }
        };

        Ok(ResolutionResult {
            conflicting_operations: conflicting_ops.to_vec(),
            resolved_operations: resolved,
            rejected_operations: rejected,
            strategy_used: strategy.clone(),
            metadata: HashMap::new(),
        })
    }

    /// Resolve clear conflict
    async fn resolve_clear_conflict(
        &self,
        clear_op: &TimestampedOperation,
        conflicting_ops: &[TimestampedOperation],
        _strategy: &ResolutionStrategy,
    ) -> Result<ResolutionResult> {
        // Clear operation typically wins
        Ok(ResolutionResult {
            conflicting_operations: {
                let mut ops = vec![clear_op.clone()];
                ops.extend(conflicting_ops.iter().cloned());
                ops
            },
            resolved_operations: vec![clear_op.clone()],
            rejected_operations: conflicting_ops.to_vec(),
            strategy_used: ResolutionStrategy::FirstWriterWins, // Clear always wins
            metadata: HashMap::new(),
        })
    }

    /// Check if operation matches pattern
    fn operation_matches_pattern(
        &self,
        operation: &RdfOperation,
        pattern: &OperationPattern,
    ) -> bool {
        // Check operation type
        if let Some(expected_type) = &pattern.operation_type {
            let actual_type = match operation {
                RdfOperation::Insert { .. } => OperationType::Insert,
                RdfOperation::Delete { .. } => OperationType::Delete,
                RdfOperation::Update { .. } => OperationType::Update,
                RdfOperation::Clear { .. } => OperationType::Clear,
                RdfOperation::Batch { .. } => return false, // Batch operations don't match simple patterns
            };

            if &actual_type != expected_type {
                return false;
            }
        }

        // Check triple patterns
        match operation {
            RdfOperation::Insert {
                subject,
                predicate,
                object,
                ..
            }
            | RdfOperation::Delete {
                subject,
                predicate,
                object,
                ..
            } => self.check_triple_pattern(subject, predicate, object, pattern),
            RdfOperation::Update {
                new_triple: (subject, predicate, object),
                ..
            } => self.check_triple_pattern(subject, predicate, object, pattern),
            _ => true, // Clear and batch operations match by default
        }
    }

    /// Check if triple matches pattern
    fn check_triple_pattern(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        pattern: &OperationPattern,
    ) -> bool {
        if let Some(subject_pattern) = &pattern.subject_pattern {
            if !self.matches_wildcard_pattern(subject, subject_pattern) {
                return false;
            }
        }

        if let Some(predicate_pattern) = &pattern.predicate_pattern {
            if !self.matches_wildcard_pattern(predicate, predicate_pattern) {
                return false;
            }
        }

        if let Some(object_pattern) = &pattern.object_pattern {
            if !self.matches_wildcard_pattern(object, object_pattern) {
                return false;
            }
        }

        true
    }

    /// Check if string matches wildcard pattern
    fn matches_wildcard_pattern(&self, value: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        // Simple wildcard matching (can be enhanced)
        if pattern.contains('*') {
            let parts: Vec<_> = pattern.split('*').collect();
            let mut value_pos = 0;

            for (i, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }

                if i == 0 {
                    // First part must match from the beginning
                    if !value[value_pos..].starts_with(part) {
                        return false;
                    }
                    value_pos += part.len();
                } else if i == parts.len() - 1 {
                    // Last part must match at the end
                    return value[value_pos..].ends_with(part);
                } else {
                    // Middle part must be found somewhere
                    if let Some(pos) = value[value_pos..].find(part) {
                        value_pos += pos + part.len();
                    } else {
                        return false;
                    }
                }
            }

            true
        } else {
            value == pattern
        }
    }

    /// Update resolution statistics
    async fn update_statistics(&self, results: &[ResolutionResult], resolution_time: Duration) {
        let mut stats = self.resolution_stats.write().await;

        stats.total_conflicts += results.len() as u64;

        // Update average resolution time
        let total_time = stats.average_resolution_time.as_nanos() * stats.total_conflicts as u128
            + resolution_time.as_nanos();
        stats.average_resolution_time = Duration::from_nanos(
            (total_time / (stats.total_conflicts + results.len() as u64) as u128) as u64,
        );

        // Update strategy usage
        for result in results {
            let strategy_name = format!("{:?}", result.strategy_used);
            *stats.strategies_used.entry(strategy_name).or_insert(0) += 1;
        }

        // Calculate success rate
        let manual_resolutions = results
            .iter()
            .filter(|r| matches!(r.strategy_used, ResolutionStrategy::Manual))
            .count();
        let total_resolutions = results.len();
        stats.success_rate = if total_resolutions > 0 {
            1.0 - (manual_resolutions as f64 / total_resolutions as f64)
        } else {
            1.0
        };
    }

    /// Get resolution statistics
    pub async fn get_statistics(&self) -> ResolutionStatistics {
        self.resolution_stats.read().await.clone()
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_operation(
        id: &str,
        node: OxirsNodeId,
        operation: RdfOperation,
    ) -> TimestampedOperation {
        TimestampedOperation {
            operation_id: id.to_string(),
            origin_node: node,
            vector_clock: VectorClock::new(),
            physical_time: UNIX_EPOCH,
            operation,
            priority: 0,
        }
    }

    #[test]
    fn test_vector_clock_operations() {
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment(1);
        clock1.increment(1);
        clock2.increment(2);

        assert!(!clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));
        assert!(clock1.is_concurrent(&clock2));

        clock2.update(&clock1);
        assert!(clock1.happens_before(&clock2));
    }

    #[tokio::test]
    async fn test_write_write_conflict_detection() {
        let resolver = ConflictResolver::new(ResolutionStrategy::LastWriterWins);

        let op1 = create_test_operation(
            "op1",
            1,
            RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
                graph: None,
            },
        );

        let op2 = create_test_operation(
            "op2",
            2,
            RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o2".to_string(),
                graph: None,
            },
        );

        let conflicts = resolver.detect_conflicts(&[op1, op2]).await.unwrap();
        assert_eq!(conflicts.len(), 1);
        assert!(matches!(conflicts[0], ConflictType::WriteWrite { .. }));
    }

    #[tokio::test]
    async fn test_conflict_resolution() {
        let resolver = ConflictResolver::new(ResolutionStrategy::LastWriterWins);

        let mut op1 = create_test_operation(
            "op1",
            1,
            RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
                graph: None,
            },
        );
        op1.physical_time = UNIX_EPOCH + Duration::from_secs(1);

        let mut op2 = create_test_operation(
            "op2",
            2,
            RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o2".to_string(),
                graph: None,
            },
        );
        op2.physical_time = UNIX_EPOCH + Duration::from_secs(2);

        let conflict = ConflictType::WriteWrite {
            operation1: op1.clone(),
            operation2: op2.clone(),
        };

        let results = resolver.resolve_conflicts(&[conflict]).await.unwrap();
        assert_eq!(results.len(), 1);

        let result = &results[0];
        assert_eq!(result.resolved_operations.len(), 1);
        assert_eq!(result.resolved_operations[0].operation_id, "op2"); // Later operation wins
        assert_eq!(result.rejected_operations.len(), 1);
        assert_eq!(result.rejected_operations[0].operation_id, "op1");
    }

    #[test]
    fn test_wildcard_pattern_matching() {
        let resolver = ConflictResolver::new(ResolutionStrategy::LastWriterWins);

        assert!(resolver.matches_wildcard_pattern("hello", "*"));
        assert!(resolver.matches_wildcard_pattern("hello", "hello"));
        assert!(resolver.matches_wildcard_pattern("hello", "h*o"));
        assert!(resolver.matches_wildcard_pattern("hello", "*lo"));
        assert!(resolver.matches_wildcard_pattern("hello", "he*"));
        assert!(!resolver.matches_wildcard_pattern("hello", "world"));
        assert!(!resolver.matches_wildcard_pattern("hello", "h*x"));
    }
}
