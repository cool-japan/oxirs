//! Conflict resolution for patches

use crate::{PatchOperation, RdfPatch};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tracing::info;

pub struct ConflictResolver {
    strategy: ConflictStrategy,
    priority_rules: Vec<PriorityRule>,
    merge_policies: HashMap<String, MergePolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictStrategy {
    FirstWins,
    LastWins,
    Merge,
    Manual,
    Priority,
    Temporal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityRule {
    pub operation_type: String,
    pub priority: i32,
    pub source_pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergePolicy {
    Union,
    Intersection,
    CustomLogic(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    pub conflicts_found: usize,
    pub conflicts_resolved: usize,
    pub resolution_strategy: ConflictStrategy,
    pub detailed_conflicts: Vec<DetailedConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedConflict {
    pub conflict_type: String,
    pub operation1: PatchOperation,
    pub operation2: PatchOperation,
    pub resolution: ConflictResolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    KeepFirst,
    KeepSecond,
    KeepBoth,
    Merged(PatchOperation),
    RequiresManualReview,
}

impl ConflictResolver {
    pub fn new(strategy: ConflictStrategy) -> Self {
        Self {
            strategy,
            priority_rules: Vec::new(),
            merge_policies: HashMap::new(),
        }
    }

    pub fn with_priority_rule(mut self, rule: PriorityRule) -> Self {
        self.priority_rules.push(rule);
        self
    }

    pub fn with_merge_policy(mut self, operation_type: String, policy: MergePolicy) -> Self {
        self.merge_policies.insert(operation_type, policy);
        self
    }

    /// Resolve conflicts between two patches
    pub fn resolve_conflicts(
        &self,
        patch1: &RdfPatch,
        patch2: &RdfPatch,
    ) -> Result<(RdfPatch, ConflictReport)> {
        let mut merged_patch = RdfPatch::new();
        merged_patch.id = format!("merged-{}-{}", patch1.id, patch2.id);

        let mut conflicts = Vec::new();
        let mut operation_map = BTreeMap::new();

        // Index operations from both patches
        for (idx, op) in patch1.operations.iter().enumerate() {
            let key = self.operation_key(op);
            operation_map.insert(format!("p1-{idx}-{key}"), (op, "patch1"));
        }

        for (idx, op) in patch2.operations.iter().enumerate() {
            let key = self.operation_key(op);
            let conflict_key = format!("p2-{idx}-{key}");

            // Check for conflicts
            if let Some(existing) = operation_map
                .iter()
                .find(|(k, _)| self.operations_conflict(op, k.split('-').nth(2).unwrap_or("")))
            {
                let conflict = DetailedConflict {
                    conflict_type: "operation_overlap".to_string(),
                    operation1: existing.1 .0.clone(),
                    operation2: op.clone(),
                    resolution: self.resolve_operation_conflict(existing.1 .0, op)?,
                };
                conflicts.push(conflict);
            } else {
                operation_map.insert(conflict_key, (op, "patch2"));
            }
        }

        // Apply resolution strategy
        for (_, (operation, _source)) in operation_map {
            merged_patch.add_operation(operation.clone());
        }

        // Apply conflict resolutions
        for conflict in &conflicts {
            match &conflict.resolution {
                ConflictResolution::KeepFirst => {
                    // Already in merged patch
                }
                ConflictResolution::KeepSecond => {
                    // Replace with second operation
                    merged_patch.add_operation(conflict.operation2.clone());
                }
                ConflictResolution::KeepBoth => {
                    merged_patch.add_operation(conflict.operation1.clone());
                    merged_patch.add_operation(conflict.operation2.clone());
                }
                ConflictResolution::Merged(merged_op) => {
                    merged_patch.add_operation(merged_op.clone());
                }
                ConflictResolution::RequiresManualReview => {
                    // Add as comment or metadata
                    merged_patch.add_operation(PatchOperation::Header {
                        key: "conflict".to_string(),
                        value: format!("Manual review required: {:?}", conflict.conflict_type),
                    });
                }
            }
        }

        let report = ConflictReport {
            conflicts_found: conflicts.len(),
            conflicts_resolved: conflicts
                .iter()
                .filter(|c| !matches!(c.resolution, ConflictResolution::RequiresManualReview))
                .count(),
            resolution_strategy: self.strategy.clone(),
            detailed_conflicts: conflicts,
        };

        info!(
            "Conflict resolution completed: {}/{} conflicts resolved",
            report.conflicts_resolved, report.conflicts_found
        );
        Ok((merged_patch, report))
    }

    fn operation_key(&self, operation: &PatchOperation) -> String {
        match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => {
                format!("add-{subject}-{predicate}-{object}")
            }
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => {
                format!("delete-{subject}-{predicate}-{object}")
            }
            PatchOperation::AddGraph { graph } => {
                format!("add-graph-{graph}")
            }
            PatchOperation::DeleteGraph { graph } => {
                format!("delete-graph-{graph}")
            }
            _ => format!("{operation:?}"),
        }
    }

    fn operations_conflict(&self, _op1: &PatchOperation, _op2_key: &str) -> bool {
        // Simplified conflict detection - in practice this would be more sophisticated
        false
    }

    fn resolve_operation_conflict(
        &self,
        op1: &PatchOperation,
        op2: &PatchOperation,
    ) -> Result<ConflictResolution> {
        match self.strategy {
            ConflictStrategy::FirstWins => Ok(ConflictResolution::KeepFirst),
            ConflictStrategy::LastWins => Ok(ConflictResolution::KeepSecond),
            ConflictStrategy::Merge => {
                // Attempt to merge operations
                self.attempt_merge(op1, op2)
            }
            ConflictStrategy::Priority => {
                // Use priority rules
                self.resolve_by_priority(op1, op2)
            }
            ConflictStrategy::Temporal => {
                // Use timestamps if available
                Ok(ConflictResolution::KeepSecond) // Default to later operation
            }
            ConflictStrategy::Manual => Ok(ConflictResolution::RequiresManualReview),
        }
    }

    fn attempt_merge(
        &self,
        op1: &PatchOperation,
        op2: &PatchOperation,
    ) -> Result<ConflictResolution> {
        match (op1, op2) {
            (
                PatchOperation::Add {
                    subject: s1,
                    predicate: p1,
                    object: _o1,
                },
                PatchOperation::Add {
                    subject: s2,
                    predicate: p2,
                    object: _o2,
                },
            ) => {
                if s1 == s2 && p1 == p2 {
                    // Different objects for same subject/predicate - keep both
                    Ok(ConflictResolution::KeepBoth)
                } else {
                    Ok(ConflictResolution::KeepBoth)
                }
            }
            _ => Ok(ConflictResolution::RequiresManualReview),
        }
    }

    fn resolve_by_priority(
        &self,
        op1: &PatchOperation,
        op2: &PatchOperation,
    ) -> Result<ConflictResolution> {
        let priority1 = self.get_operation_priority(op1);
        let priority2 = self.get_operation_priority(op2);

        if priority1 > priority2 {
            Ok(ConflictResolution::KeepFirst)
        } else if priority2 > priority1 {
            Ok(ConflictResolution::KeepSecond)
        } else {
            Ok(ConflictResolution::RequiresManualReview)
        }
    }

    fn get_operation_priority(&self, operation: &PatchOperation) -> i32 {
        let op_type = match operation {
            PatchOperation::Add { .. } => "add",
            PatchOperation::Delete { .. } => "delete",
            PatchOperation::AddGraph { .. } => "add_graph",
            PatchOperation::DeleteGraph { .. } => "delete_graph",
            _ => "other",
        };

        for rule in &self.priority_rules {
            if rule.operation_type == op_type {
                return rule.priority;
            }
        }

        0 // Default priority
    }
}
