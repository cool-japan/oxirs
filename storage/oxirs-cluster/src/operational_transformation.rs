//! # Enhanced Conflict Resolution with Operational Transformation
//!
//! Provides sophisticated conflict resolution for concurrent RDF triple modifications:
//! - Operational Transformation (OT) for RDF operations
//! - Three-way merge for concurrent updates
//! - Automatic conflict detection
//! - Resolution strategies (LWW, priority-based, semantic)
//! - Comprehensive conflict history

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::info;

use crate::raft::OxirsNodeId;

/// RDF operation type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RdfOperation {
    /// Insert triple
    Insert {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Delete triple
    Delete {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Update triple (delete + insert)
    Update {
        old_subject: String,
        old_predicate: String,
        old_object: String,
        new_subject: String,
        new_predicate: String,
        new_object: String,
    },
}

/// Conflict type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    /// Write-write conflict (same triple modified by different nodes)
    WriteWrite,
    /// Delete-update conflict (one deletes, another updates)
    DeleteUpdate,
    /// Update-update conflict (both update same triple differently)
    UpdateUpdate,
    /// Semantic conflict (violates RDF semantics)
    Semantic,
}

/// Operation with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationRecord {
    /// Operation ID
    pub operation_id: String,
    /// Node that originated this operation
    pub origin_node: OxirsNodeId,
    /// The operation
    pub operation: RdfOperation,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Vector clock for causality
    pub vector_clock: BTreeMap<OxirsNodeId, u64>,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Last Writer Wins (based on timestamp)
    LastWriterWins,
    /// Priority-based (higher priority wins)
    PriorityBased,
    /// Semantic resolution (preserve RDF semantics)
    Semantic,
    /// Manual resolution required
    Manual,
}

/// Conflict record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRecord {
    /// Conflict ID
    pub conflict_id: String,
    /// Conflict type
    pub conflict_type: ConflictType,
    /// Conflicting operations
    pub operations: Vec<OperationRecord>,
    /// Detected timestamp
    pub detected_at: SystemTime,
    /// Resolution strategy used
    pub resolution_strategy: ResolutionStrategy,
    /// Resolved operation (if resolved)
    pub resolved_operation: Option<RdfOperation>,
    /// Is resolved
    pub is_resolved: bool,
}

/// OT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OTConfig {
    /// Default resolution strategy
    pub default_strategy: ResolutionStrategy,
    /// Enable automatic resolution
    pub enable_auto_resolution: bool,
    /// Maximum conflict history size
    pub max_conflict_history: usize,
    /// Enable semantic validation
    pub enable_semantic_validation: bool,
    /// Conflict detection threshold (seconds)
    pub conflict_detection_window_secs: u64,
}

impl Default for OTConfig {
    fn default() -> Self {
        Self {
            default_strategy: ResolutionStrategy::LastWriterWins,
            enable_auto_resolution: true,
            max_conflict_history: 1000,
            enable_semantic_validation: true,
            conflict_detection_window_secs: 60,
        }
    }
}

/// OT statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OTStats {
    /// Total conflicts detected
    pub total_conflicts: u64,
    /// Auto-resolved conflicts
    pub auto_resolved: u64,
    /// Manually resolved conflicts
    pub manually_resolved: u64,
    /// Unresolved conflicts
    pub unresolved: u64,
    /// Write-write conflicts
    pub write_write_conflicts: u64,
    /// Delete-update conflicts
    pub delete_update_conflicts: u64,
    /// Update-update conflicts
    pub update_update_conflicts: u64,
    /// Semantic conflicts
    pub semantic_conflicts: u64,
}

impl Default for OTStats {
    fn default() -> Self {
        Self {
            total_conflicts: 0,
            auto_resolved: 0,
            manually_resolved: 0,
            unresolved: 0,
            write_write_conflicts: 0,
            delete_update_conflicts: 0,
            update_update_conflicts: 0,
            semantic_conflicts: 0,
        }
    }
}

/// Operational Transformation manager
pub struct OperationalTransformation {
    config: OTConfig,
    /// Pending operations
    pending_operations: Arc<RwLock<VecDeque<OperationRecord>>>,
    /// Conflict history
    conflict_history: Arc<RwLock<VecDeque<ConflictRecord>>>,
    /// Active conflicts
    active_conflicts: Arc<RwLock<HashMap<String, ConflictRecord>>>,
    /// Statistics
    stats: Arc<RwLock<OTStats>>,
    /// Local node ID
    local_node_id: OxirsNodeId,
}

impl OperationalTransformation {
    /// Create a new OT manager
    pub fn new(local_node_id: OxirsNodeId, config: OTConfig) -> Self {
        Self {
            config,
            pending_operations: Arc::new(RwLock::new(VecDeque::new())),
            conflict_history: Arc::new(RwLock::new(VecDeque::new())),
            active_conflicts: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(OTStats::default())),
            local_node_id,
        }
    }

    /// Submit an operation for conflict detection and resolution
    pub async fn submit_operation(
        &self,
        mut operation: OperationRecord,
    ) -> Result<RdfOperation, String> {
        // Ensure operation has metadata
        if operation.vector_clock.is_empty() {
            operation.vector_clock.insert(self.local_node_id, 1);
        }

        // Add to pending operations
        let mut pending = self.pending_operations.write().await;
        pending.push_back(operation.clone());

        // Detect conflicts
        let conflicts = self.detect_conflicts(&operation, &pending).await;

        if conflicts.is_empty() {
            // No conflicts, return operation as-is
            drop(pending);
            return Ok(operation.operation);
        }

        // Conflicts detected
        drop(pending);

        if self.config.enable_auto_resolution {
            self.resolve_conflicts(conflicts).await
        } else {
            Err("Conflicts detected, manual resolution required".to_string())
        }
    }

    /// Detect conflicts between operations
    async fn detect_conflicts(
        &self,
        operation: &OperationRecord,
        pending: &VecDeque<OperationRecord>,
    ) -> Vec<ConflictRecord> {
        let mut conflicts = Vec::new();

        // Check against recent operations within time window
        let window = std::time::Duration::from_secs(self.config.conflict_detection_window_secs);

        for pending_op in pending.iter() {
            if pending_op.operation_id == operation.operation_id {
                continue; // Skip self
            }

            // Check if within time window
            if let (Ok(op_elapsed), Ok(pending_elapsed)) = (
                operation.timestamp.elapsed(),
                pending_op.timestamp.elapsed(),
            ) {
                if op_elapsed > window && pending_elapsed > window {
                    continue; // Too old
                }
            }

            // Check for conflicts
            if let Some(conflict_type) =
                self.check_conflict(&operation.operation, &pending_op.operation)
            {
                let conflict_id = format!(
                    "conflict-{}-{}",
                    operation.operation_id, pending_op.operation_id
                );

                let conflict = ConflictRecord {
                    conflict_id,
                    conflict_type,
                    operations: vec![operation.clone(), pending_op.clone()],
                    detected_at: SystemTime::now(),
                    resolution_strategy: self.config.default_strategy,
                    resolved_operation: None,
                    is_resolved: false,
                };

                conflicts.push(conflict);
            }
        }

        conflicts
    }

    /// Check if two operations conflict
    fn check_conflict(&self, op1: &RdfOperation, op2: &RdfOperation) -> Option<ConflictType> {
        match (op1, op2) {
            // Insert vs Insert on same triple
            (
                RdfOperation::Insert {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RdfOperation::Insert {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                if s1 == s2 && p1 == p2 && o1 != o2 {
                    Some(ConflictType::WriteWrite)
                } else {
                    None
                }
            }

            // Delete vs Update
            (
                RdfOperation::Delete {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RdfOperation::Update {
                    old_subject: s2,
                    old_predicate: p2,
                    old_object: o2,
                    ..
                },
            )
            | (
                RdfOperation::Update {
                    old_subject: s2,
                    old_predicate: p2,
                    old_object: o2,
                    ..
                },
                RdfOperation::Delete {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
            ) => {
                if s1 == s2 && p1 == p2 && o1 == o2 {
                    Some(ConflictType::DeleteUpdate)
                } else {
                    None
                }
            }

            // Update vs Update
            (
                RdfOperation::Update {
                    old_subject: s1,
                    old_predicate: p1,
                    old_object: o1,
                    new_subject: ns1,
                    new_predicate: np1,
                    new_object: no1,
                },
                RdfOperation::Update {
                    old_subject: s2,
                    old_predicate: p2,
                    old_object: o2,
                    new_subject: ns2,
                    new_predicate: np2,
                    new_object: no2,
                },
            ) => {
                if s1 == s2 && p1 == p2 && o1 == o2 {
                    if ns1 != ns2 || np1 != np2 || no1 != no2 {
                        Some(ConflictType::UpdateUpdate)
                    } else {
                        None // Same update
                    }
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Resolve conflicts using configured strategy
    async fn resolve_conflicts(
        &self,
        mut conflicts: Vec<ConflictRecord>,
    ) -> Result<RdfOperation, String> {
        if conflicts.is_empty() {
            return Err("No conflicts to resolve".to_string());
        }

        // Take first conflict (simplification)
        let mut conflict = conflicts.remove(0);

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_conflicts += 1;

        match conflict.conflict_type {
            ConflictType::WriteWrite => stats.write_write_conflicts += 1,
            ConflictType::DeleteUpdate => stats.delete_update_conflicts += 1,
            ConflictType::UpdateUpdate => stats.update_update_conflicts += 1,
            ConflictType::Semantic => stats.semantic_conflicts += 1,
        }

        drop(stats);

        let resolved_op = match self.config.default_strategy {
            ResolutionStrategy::LastWriterWins => {
                self.resolve_last_writer_wins(&conflict.operations)
            }
            ResolutionStrategy::PriorityBased => self.resolve_priority_based(&conflict.operations),
            ResolutionStrategy::Semantic => self.resolve_semantic(&conflict.operations),
            ResolutionStrategy::Manual => {
                return Err("Manual resolution required".to_string());
            }
        };

        // Mark as resolved
        conflict.resolved_operation = Some(resolved_op.clone());
        conflict.is_resolved = true;

        // Store in history
        let mut history = self.conflict_history.write().await;
        history.push_back(conflict.clone());

        if history.len() > self.config.max_conflict_history {
            history.pop_front();
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.auto_resolved += 1;

        info!(
            "Resolved conflict {} using {:?} strategy",
            conflict.conflict_id, self.config.default_strategy
        );

        Ok(resolved_op)
    }

    /// Last Writer Wins resolution
    fn resolve_last_writer_wins(&self, operations: &[OperationRecord]) -> RdfOperation {
        operations
            .iter()
            .max_by_key(|op| op.timestamp)
            .map(|op| op.operation.clone())
            .expect("operations should not be empty for LWW resolution")
    }

    /// Priority-based resolution
    fn resolve_priority_based(&self, operations: &[OperationRecord]) -> RdfOperation {
        operations
            .iter()
            .max_by_key(|op| op.priority)
            .map(|op| op.operation.clone())
            .expect("operations should not be empty for priority resolution")
    }

    /// Semantic resolution (preserve RDF semantics)
    fn resolve_semantic(&self, operations: &[OperationRecord]) -> RdfOperation {
        // Semantic resolution prioritizes:
        // 1. Inserts over Deletes (preserve data)
        // 2. Updates over Deletes
        // 3. LWW as fallback

        let has_insert = operations
            .iter()
            .any(|op| matches!(op.operation, RdfOperation::Insert { .. }));
        let has_update = operations
            .iter()
            .any(|op| matches!(op.operation, RdfOperation::Update { .. }));

        if has_insert {
            // Prefer insert
            operations
                .iter()
                .find(|op| matches!(op.operation, RdfOperation::Insert { .. }))
                .map(|op| op.operation.clone())
                .expect("insert operation should exist after has_insert check")
        } else if has_update {
            // Prefer update
            operations
                .iter()
                .find(|op| matches!(op.operation, RdfOperation::Update { .. }))
                .map(|op| op.operation.clone())
                .expect("update operation should exist after has_update check")
        } else {
            // Fallback to LWW
            self.resolve_last_writer_wins(operations)
        }
    }

    /// Transform operation against another (OT algorithm)
    pub fn transform_operation(&self, op1: &RdfOperation, op2: &RdfOperation) -> RdfOperation {
        // Operational Transformation for RDF operations
        match (op1, op2) {
            // If both insert same triple, keep second one
            (
                RdfOperation::Insert {
                    subject: s1,
                    predicate: p1,
                    object: _o1,
                },
                RdfOperation::Insert {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                if s1 == s2 && p1 == p2 {
                    // Conflict: prefer o2 (second operation)
                    RdfOperation::Insert {
                        subject: s2.clone(),
                        predicate: p2.clone(),
                        object: o2.clone(),
                    }
                } else {
                    op1.clone()
                }
            }

            // Delete transformed against insert: becomes no-op if same triple
            (
                RdfOperation::Delete {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RdfOperation::Insert {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                if s1 == s2 && p1 == p2 && o1 == o2 {
                    // No-op: insert cancels delete
                    op2.clone()
                } else {
                    op1.clone()
                }
            }

            _ => op1.clone(),
        }
    }

    /// Get conflict history
    pub async fn get_conflict_history(&self) -> Vec<ConflictRecord> {
        self.conflict_history.read().await.iter().cloned().collect()
    }

    /// Get active conflicts
    pub async fn get_active_conflicts(&self) -> Vec<ConflictRecord> {
        self.active_conflicts
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> OTStats {
        self.stats.read().await.clone()
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.pending_operations.write().await.clear();
        self.conflict_history.write().await.clear();
        self.active_conflicts.write().await.clear();
        *self.stats.write().await = OTStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ot_creation() {
        let config = OTConfig::default();
        let ot = OperationalTransformation::new(1, config);

        let stats = ot.get_stats().await;
        assert_eq!(stats.total_conflicts, 0);
    }

    #[tokio::test]
    async fn test_no_conflict() {
        let config = OTConfig::default();
        let ot = OperationalTransformation::new(1, config);

        let op = OperationRecord {
            operation_id: "op-1".to_string(),
            origin_node: 1,
            operation: RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
            },
            timestamp: SystemTime::now(),
            priority: 0,
            vector_clock: BTreeMap::new(),
        };

        let result = ot.submit_operation(op).await;
        assert!(result.is_ok());

        let stats = ot.get_stats().await;
        assert_eq!(stats.total_conflicts, 0);
    }

    #[tokio::test]
    async fn test_write_write_conflict() {
        let config = OTConfig {
            default_strategy: ResolutionStrategy::LastWriterWins,
            enable_auto_resolution: true,
            ..Default::default()
        };
        let ot = OperationalTransformation::new(1, config);

        // First operation
        let op1 = OperationRecord {
            operation_id: "op-1".to_string(),
            origin_node: 1,
            operation: RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
            },
            timestamp: SystemTime::now(),
            priority: 0,
            vector_clock: BTreeMap::new(),
        };

        ot.submit_operation(op1).await.unwrap();

        // Conflicting operation (same subject/predicate, different object)
        let op2 = OperationRecord {
            operation_id: "op-2".to_string(),
            origin_node: 2,
            operation: RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o2".to_string(), // Different object
            },
            timestamp: SystemTime::now(),
            priority: 0,
            vector_clock: BTreeMap::new(),
        };

        let result = ot.submit_operation(op2).await;
        assert!(result.is_ok());

        let stats = ot.get_stats().await;
        assert_eq!(stats.total_conflicts, 1);
        assert_eq!(stats.write_write_conflicts, 1);
    }

    #[tokio::test]
    async fn test_priority_based_resolution() {
        let config = OTConfig {
            default_strategy: ResolutionStrategy::PriorityBased,
            enable_auto_resolution: true,
            ..Default::default()
        };
        let ot = OperationalTransformation::new(1, config);

        let op1 = OperationRecord {
            operation_id: "op-1".to_string(),
            origin_node: 1,
            operation: RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
            },
            timestamp: SystemTime::now(),
            priority: 1,
            vector_clock: BTreeMap::new(),
        };

        ot.submit_operation(op1).await.unwrap();

        let op2 = OperationRecord {
            operation_id: "op-2".to_string(),
            origin_node: 2,
            operation: RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o2".to_string(),
            },
            timestamp: SystemTime::now(),
            priority: 10, // Higher priority
            vector_clock: BTreeMap::new(),
        };

        let result = ot.submit_operation(op2.clone()).await;
        assert!(result.is_ok());

        // Should resolve to op2 due to higher priority
        if let Ok(RdfOperation::Insert { object, .. }) = result {
            assert_eq!(object, "o2");
        }
    }

    #[tokio::test]
    async fn test_semantic_resolution() {
        let config = OTConfig {
            default_strategy: ResolutionStrategy::Semantic,
            enable_auto_resolution: true,
            enable_semantic_validation: true,
            ..Default::default()
        };
        let ot = OperationalTransformation::new(1, config);

        // Delete operation
        let op1 = OperationRecord {
            operation_id: "op-1".to_string(),
            origin_node: 1,
            operation: RdfOperation::Delete {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
            },
            timestamp: SystemTime::now(),
            priority: 0,
            vector_clock: BTreeMap::new(),
        };

        ot.submit_operation(op1).await.unwrap();

        // Insert operation (conflicts with delete)
        let op2 = OperationRecord {
            operation_id: "op-2".to_string(),
            origin_node: 2,
            operation: RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
            },
            timestamp: SystemTime::now(),
            priority: 0,
            vector_clock: BTreeMap::new(),
        };

        // Semantic resolution should prefer insert (preserve data)
        // This would normally create a delete-update conflict
        let result = ot.submit_operation(op2).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_transform_operation() {
        let config = OTConfig::default();
        let ot = OperationalTransformation::new(1, config);

        let op1 = RdfOperation::Insert {
            subject: "s1".to_string(),
            predicate: "p1".to_string(),
            object: "o1".to_string(),
        };

        let op2 = RdfOperation::Insert {
            subject: "s1".to_string(),
            predicate: "p1".to_string(),
            object: "o2".to_string(),
        };

        let transformed = ot.transform_operation(&op1, &op2);

        // Should transform to op2
        if let RdfOperation::Insert { object, .. } = transformed {
            assert_eq!(object, "o2");
        }
    }

    #[tokio::test]
    async fn test_conflict_history() {
        let config = OTConfig {
            max_conflict_history: 10,
            enable_auto_resolution: true,
            ..Default::default()
        };
        let ot = OperationalTransformation::new(1, config);

        // Create conflicts
        for i in 0..5 {
            let op1 = OperationRecord {
                operation_id: format!("op-{}-1", i),
                origin_node: 1,
                operation: RdfOperation::Insert {
                    subject: format!("s{}", i),
                    predicate: "p1".to_string(),
                    object: "o1".to_string(),
                },
                timestamp: SystemTime::now(),
                priority: 0,
                vector_clock: BTreeMap::new(),
            };

            ot.submit_operation(op1).await.unwrap();

            let op2 = OperationRecord {
                operation_id: format!("op-{}-2", i),
                origin_node: 2,
                operation: RdfOperation::Insert {
                    subject: format!("s{}", i),
                    predicate: "p1".to_string(),
                    object: "o2".to_string(),
                },
                timestamp: SystemTime::now(),
                priority: 0,
                vector_clock: BTreeMap::new(),
            };

            let _ = ot.submit_operation(op2).await;
        }

        let history = ot.get_conflict_history().await;
        assert_eq!(history.len(), 5);
    }

    #[tokio::test]
    async fn test_clear() {
        let config = OTConfig::default();
        let ot = OperationalTransformation::new(1, config);

        let op = OperationRecord {
            operation_id: "op-1".to_string(),
            origin_node: 1,
            operation: RdfOperation::Insert {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
            },
            timestamp: SystemTime::now(),
            priority: 0,
            vector_clock: BTreeMap::new(),
        };

        ot.submit_operation(op).await.unwrap();

        ot.clear().await;

        let stats = ot.get_stats().await;
        assert_eq!(stats.total_conflicts, 0);
    }
}
