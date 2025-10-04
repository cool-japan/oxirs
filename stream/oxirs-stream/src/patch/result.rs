//! Patch result and reporting

use crate::{PatchOperation, RdfPatch};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::fmt;
use tracing::debug;

pub struct PatchResult {
    pub patch_id: String,
    pub total_operations: usize,
    pub operations_applied: usize,
    pub errors: Vec<String>,
    pub applied_at: DateTime<Utc>,
}

impl Default for PatchResult {
    fn default() -> Self {
        Self::new()
    }
}

impl PatchResult {
    pub fn new() -> Self {
        Self {
            patch_id: String::new(),
            total_operations: 0,
            operations_applied: 0,
            errors: Vec::new(),
            applied_at: Utc::now(),
        }
    }

    pub fn is_success(&self) -> bool {
        self.errors.is_empty() && self.operations_applied == self.total_operations
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            1.0
        } else {
            self.operations_applied as f64 / self.total_operations as f64
        }
    }
}

impl fmt::Display for PatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Patch {} applied: {}/{} operations ({:.1}% success)",
            self.patch_id,
            self.operations_applied,
            self.total_operations,
            self.success_rate() * 100.0
        )
    }
}

/// Create RDF Patch from SPARQL Update
pub fn create_patch_from_sparql(update: &str) -> Result<RdfPatch> {
    let mut delta_computer = crate::delta::DeltaComputer::new();
    delta_computer.sparql_to_patch(update)
}

/// Create transactional patch
pub fn create_transactional_patch(operations: Vec<PatchOperation>) -> RdfPatch {
    let mut patch = RdfPatch::new();
    let transaction_id = uuid::Uuid::new_v4().to_string();

    // Add transaction begin
    patch.add_operation(PatchOperation::TransactionBegin {
        transaction_id: Some(transaction_id.clone()),
    });
    patch.transaction_id = Some(transaction_id);

    // Add all operations
    for op in operations {
        patch.add_operation(op);
    }

    // Add transaction commit
    patch.add_operation(PatchOperation::TransactionCommit);

    patch
}

/// Create reverse patch from an existing patch
pub fn create_reverse_patch(patch: &RdfPatch) -> Result<RdfPatch> {
    let mut reverse_patch = RdfPatch::new();
    reverse_patch.id = format!("{}-reverse", patch.id);

    // Copy headers and prefixes
    reverse_patch.headers = patch.headers.clone();
    reverse_patch.prefixes = patch.prefixes.clone();

    // Track if we're reversing a transaction
    let mut reversing_transaction = false;
    let mut transaction_operations = Vec::new();

    // Reverse the operations and their order
    for operation in patch.operations.iter().rev() {
        let reverse_operation = match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => PatchOperation::Delete {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
            },
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => PatchOperation::Add {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
            },
            PatchOperation::AddGraph { graph } => PatchOperation::DeleteGraph {
                graph: graph.clone(),
            },
            PatchOperation::DeleteGraph { graph } => PatchOperation::AddGraph {
                graph: graph.clone(),
            },
            PatchOperation::AddPrefix {
                prefix,
                namespace: _namespace,
            } => PatchOperation::DeletePrefix {
                prefix: prefix.clone(),
            },
            PatchOperation::DeletePrefix { prefix } => {
                // Can't reverse a prefix deletion without knowing the namespace
                // Skip or add as header
                reverse_patch.add_operation(PatchOperation::Header {
                    key: "warning".to_string(),
                    value: format!("Cannot reverse prefix deletion for '{prefix}'"),
                });
                continue;
            }
            PatchOperation::TransactionBegin { transaction_id: _ } => {
                // End of transaction (we're reversing)
                reversing_transaction = false;
                // Add all collected operations
                for op in transaction_operations.drain(..) {
                    reverse_patch.add_operation(op);
                }
                PatchOperation::TransactionCommit
            }
            PatchOperation::TransactionCommit => {
                // Start of transaction (we're reversing)
                reversing_transaction = true;
                PatchOperation::TransactionBegin {
                    transaction_id: patch.transaction_id.clone(),
                }
            }
            PatchOperation::TransactionAbort => {
                // Transaction was aborted, no need to reverse
                continue;
            }
            PatchOperation::Header { key, value } => PatchOperation::Header {
                key: format!("reversed-{key}"),
                value: value.clone(),
            },
        };

        if reversing_transaction && !matches!(operation, PatchOperation::TransactionCommit) {
            transaction_operations.push(reverse_operation);
        } else {
            reverse_patch.add_operation(reverse_operation);
        }
    }

    debug!(
        "Created reverse patch with {} operations",
        reverse_patch.operations.len()
    );
    Ok(reverse_patch)
}

/// Merge multiple patches into one
pub fn merge_patches(patches: &[RdfPatch]) -> Result<RdfPatch> {
    let mut merged = RdfPatch::new();
    merged.id = format!("merged-{}", uuid::Uuid::new_v4());

    for patch in patches {
        for operation in &patch.operations {
            merged.add_operation(operation.clone());
        }
    }

    debug!(
        "Merged {} patches into {} operations",
        patches.len(),
        merged.operations.len()
    );
    Ok(merged)
}

/// Optimize patch by removing redundant operations
pub fn optimize_patch(patch: &RdfPatch) -> Result<RdfPatch> {
    let mut optimized = RdfPatch::new();
    optimized.id = format!("{}-optimized", patch.id);

    let mut seen_operations = std::collections::HashSet::new();

    for operation in &patch.operations {
        let operation_key = format!("{operation:?}");

        // Skip duplicate operations
        if seen_operations.contains(&operation_key) {
            continue;
        }

        seen_operations.insert(operation_key);
        optimized.add_operation(operation.clone());
    }

    debug!(
        "Optimized patch from {} to {} operations",
        patch.operations.len(),
        optimized.operations.len()
    );

    Ok(optimized)
}

/// Validate patch consistency
pub fn validate_patch(patch: &RdfPatch) -> Result<Vec<String>> {
    let mut warnings = Vec::new();

    // Check for conflicting operations
    let mut adds = std::collections::HashSet::new();
    let mut deletes = std::collections::HashSet::new();

    for operation in &patch.operations {
        match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => {
                let triple = (subject.clone(), predicate.clone(), object.clone());
                if deletes.contains(&triple) {
                    warnings.push(format!("Triple added after being deleted: {triple:?}"));
                }
                adds.insert(triple);
            }
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => {
                let triple = (subject.clone(), predicate.clone(), object.clone());
                if !adds.contains(&triple) {
                    warnings.push(format!("Triple deleted without prior addition: {triple:?}"));
                }
                deletes.insert(triple);
            }
            _ => {} // Graph operations don't conflict in the same way
        }
    }

    Ok(warnings)
}

