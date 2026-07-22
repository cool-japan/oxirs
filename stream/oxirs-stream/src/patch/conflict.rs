//! Conflict resolution for patches

use crate::{PatchOperation, RdfPatch};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

        // Start from patch1's operations. patch2's non-conflicting operations are
        // appended; conflicting ones are resolved according to the strategy.
        let mut merged_ops: Vec<PatchOperation> = patch1.operations.clone();
        // Track which patch1 operations have already been resolved so the same
        // op isn't matched twice.
        let mut resolved_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        for op2 in &patch2.operations {
            // Find the first not-yet-resolved conflicting operation in patch1.
            let conflicting = patch1.operations.iter().enumerate().find(|(idx, op1)| {
                !resolved_indices.contains(idx) && self.operations_conflict(op1, op2)
            });

            if let Some((idx, op1)) = conflicting {
                resolved_indices.insert(idx);
                let resolution = self.resolve_operation_conflict(op1, op2)?;

                // Apply the resolution against `merged_ops`.
                match &resolution {
                    ConflictResolution::KeepFirst => {
                        // op1 already present; drop op2.
                    }
                    ConflictResolution::KeepSecond => {
                        // Replace op1 with op2 in place.
                        if let Some(slot) = merged_ops.get_mut(idx) {
                            *slot = op2.clone();
                        }
                    }
                    ConflictResolution::KeepBoth => {
                        merged_ops.push(op2.clone());
                    }
                    ConflictResolution::Merged(merged_op) => {
                        if let Some(slot) = merged_ops.get_mut(idx) {
                            *slot = merged_op.clone();
                        }
                    }
                    ConflictResolution::RequiresManualReview => {
                        merged_ops.push(PatchOperation::Header {
                            key: "conflict".to_string(),
                            value: format!("Manual review required for {:?} vs {:?}", op1, op2),
                        });
                    }
                }

                conflicts.push(DetailedConflict {
                    conflict_type: "operation_overlap".to_string(),
                    operation1: op1.clone(),
                    operation2: op2.clone(),
                    resolution,
                });
            } else if !merged_ops.contains(op2) {
                // No conflict and not a duplicate: keep it.
                merged_ops.push(op2.clone());
            }
        }

        for operation in merged_ops {
            merged_patch.add_operation(operation);
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

    /// Determine whether two operations conflict.
    ///
    /// Two triple operations conflict when they target the same subject and
    /// predicate but either assert a different object or have opposing Add/Delete
    /// semantics (e.g. one adds a triple another deletes). Two graph operations
    /// conflict when they target the same graph URI with opposing Add/Delete
    /// semantics.
    fn operations_conflict(&self, op1: &PatchOperation, op2: &PatchOperation) -> bool {
        if let (Some((add1, s1, p1, o1)), Some((add2, s2, p2, o2))) =
            (Self::triple_parts(op1), Self::triple_parts(op2))
        {
            if s1 == s2 && p1 == p2 {
                // Same subject+predicate: conflict if the object differs or the
                // Add/Delete direction differs. Identical operations (same
                // object, same direction) are duplicates, not conflicts.
                return o1 != o2 || add1 != add2;
            }
            return false;
        }

        if let (Some((add1, g1)), Some((add2, g2))) =
            (Self::graph_parts(op1), Self::graph_parts(op2))
        {
            // Opposing graph operations on the same graph conflict.
            return g1 == g2 && add1 != add2;
        }

        false
    }

    /// Return `(is_add, subject, predicate, object)` for triple operations.
    fn triple_parts(operation: &PatchOperation) -> Option<(bool, &str, &str, &str)> {
        match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => Some((true, subject, predicate, object)),
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => Some((false, subject, predicate, object)),
            _ => None,
        }
    }

    /// Return `(is_add, graph)` for graph operations.
    fn graph_parts(operation: &PatchOperation) -> Option<(bool, &str)> {
        match operation {
            PatchOperation::AddGraph { graph } => Some((true, graph)),
            PatchOperation::DeleteGraph { graph } => Some((false, graph)),
            _ => None,
        }
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
                    object: o1,
                },
                PatchOperation::Add {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                if s1 == s2 && p1 == p2 && o1 == o2 {
                    // Identical add on both sides: keep a single copy.
                    Ok(ConflictResolution::KeepFirst)
                } else {
                    // Same subject/predicate but different object (multi-valued
                    // property): retain both assertions.
                    Ok(ConflictResolution::KeepBoth)
                }
            }
            // An add opposing a delete of the same triple cannot be merged
            // automatically without a policy: keep the addition.
            (PatchOperation::Add { .. }, PatchOperation::Delete { .. })
            | (PatchOperation::Delete { .. }, PatchOperation::Add { .. }) => {
                Ok(ConflictResolution::KeepFirst)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn add(subject: &str, object: &str) -> PatchOperation {
        PatchOperation::Add {
            subject: subject.to_string(),
            predicate: "http://example.org/p".to_string(),
            object: object.to_string(),
        }
    }

    fn delete(subject: &str, object: &str) -> PatchOperation {
        PatchOperation::Delete {
            subject: subject.to_string(),
            predicate: "http://example.org/p".to_string(),
            object: object.to_string(),
        }
    }

    #[test]
    fn regression_add_vs_delete_conflict_detected() {
        let mut p1 = RdfPatch::new();
        p1.add_operation(add("http://example.org/s", "http://example.org/o"));

        let mut p2 = RdfPatch::new();
        p2.add_operation(delete("http://example.org/s", "http://example.org/o"));

        let resolver = ConflictResolver::new(ConflictStrategy::LastWins);
        let (merged, report) = resolver.resolve_conflicts(&p1, &p2).unwrap();

        // The opposing add/delete of the same triple MUST be detected as a
        // conflict (previously `operations_conflict` was hardcoded to false).
        assert_eq!(report.conflicts_found, 1);
        assert_eq!(report.conflicts_resolved, 1);

        // LastWins => the delete replaces the add.
        assert_eq!(merged.operations.len(), 1);
        assert!(matches!(
            merged.operations[0],
            PatchOperation::Delete { .. }
        ));
    }

    #[test]
    fn regression_non_conflicting_ops_concatenate() {
        let mut p1 = RdfPatch::new();
        p1.add_operation(add("http://example.org/s1", "http://example.org/o"));

        let mut p2 = RdfPatch::new();
        p2.add_operation(add("http://example.org/s2", "http://example.org/o"));

        let resolver = ConflictResolver::new(ConflictStrategy::LastWins);
        let (merged, report) = resolver.resolve_conflicts(&p1, &p2).unwrap();

        assert_eq!(report.conflicts_found, 0);
        assert_eq!(merged.operations.len(), 2);
    }
}
