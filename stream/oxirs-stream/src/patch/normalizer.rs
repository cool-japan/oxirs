//! Patch normalizer

use crate::{PatchOperation, RdfPatch};
use anyhow::Result;
use std::collections::BTreeSet;
use tracing::info;

pub struct PatchNormalizer {
    canonical_ordering: bool,
    deduplicate_operations: bool,
    normalize_uris: bool,
    sort_by_subject: bool,
}

impl PatchNormalizer {
    pub fn new() -> Self {
        Self {
            canonical_ordering: true,
            deduplicate_operations: true,
            normalize_uris: true,
            sort_by_subject: true,
        }
    }

    pub fn with_canonical_ordering(mut self, enabled: bool) -> Self {
        self.canonical_ordering = enabled;
        self
    }

    pub fn with_deduplication(mut self, enabled: bool) -> Self {
        self.deduplicate_operations = enabled;
        self
    }

    pub fn with_uri_normalization(mut self, enabled: bool) -> Self {
        self.normalize_uris = enabled;
        self
    }

    /// Enable or disable sorting triple operations by subject.
    ///
    /// Sorting is applied only *within* contiguous runs of triple operations so
    /// that transaction boundaries and other structural operations are never
    /// reordered.
    pub fn with_subject_sort(mut self, enabled: bool) -> Self {
        self.sort_by_subject = enabled;
        self
    }

    /// Normalize a patch according to configured rules
    pub fn normalize(&self, patch: &RdfPatch) -> Result<RdfPatch> {
        let mut normalized = patch.clone();
        normalized.id = format!("{}-normalized", patch.id);

        // Step 1: Normalize URIs
        if self.normalize_uris {
            normalized = self.normalize_uris_in_patch(normalized)?;
        }

        // Step 2: Deduplicate operations
        if self.deduplicate_operations {
            normalized = self.deduplicate_operations_in_patch(normalized)?;
        }

        // Step 3: Canonical ordering
        if self.canonical_ordering {
            normalized = self.apply_canonical_ordering(normalized)?;
        }

        // Step 4: Sort by subject if enabled
        if self.sort_by_subject {
            normalized = self.sort_operations_by_subject(normalized)?;
        }

        info!(
            "Normalized patch: {} -> {} operations",
            patch.operations.len(),
            normalized.operations.len()
        );
        Ok(normalized)
    }

    fn normalize_uris_in_patch(&self, mut patch: RdfPatch) -> Result<RdfPatch> {
        for operation in &mut patch.operations {
            match operation {
                PatchOperation::Add {
                    subject,
                    predicate,
                    object,
                } => {
                    *subject = self.normalize_uri(subject);
                    *predicate = self.normalize_uri(predicate);
                    *object = self.normalize_uri(object);
                }
                PatchOperation::Delete {
                    subject,
                    predicate,
                    object,
                } => {
                    *subject = self.normalize_uri(subject);
                    *predicate = self.normalize_uri(predicate);
                    *object = self.normalize_uri(object);
                }
                PatchOperation::AddGraph { graph } => {
                    *graph = self.normalize_uri(graph);
                }
                PatchOperation::DeleteGraph { graph } => {
                    *graph = self.normalize_uri(graph);
                }
                _ => {} // Other operations don't have URIs to normalize
            }
        }
        Ok(patch)
    }

    fn normalize_uri(&self, uri: &str) -> String {
        // Remove trailing slashes, normalize case, etc.
        let mut normalized = uri.trim_end_matches('/').to_string();

        // Convert to lowercase for schemes
        if normalized.starts_with("http://") || normalized.starts_with("https://") {
            if let Some(pos) = normalized.find("://") {
                let (scheme, rest) = normalized.split_at(pos + 3);
                if let Some(domain_end) = rest.find('/') {
                    let (domain, path) = rest.split_at(domain_end);
                    normalized =
                        format!("{}{}{}", scheme.to_lowercase(), domain.to_lowercase(), path);
                } else {
                    normalized = format!("{}{}", scheme.to_lowercase(), rest.to_lowercase());
                }
            }
        }

        normalized
    }

    fn deduplicate_operations_in_patch(&self, mut patch: RdfPatch) -> Result<RdfPatch> {
        let mut seen = BTreeSet::new();
        patch.operations.retain(|op| {
            let key = format!("{op:?}");
            if seen.contains(&key) {
                false
            } else {
                seen.insert(key);
                true
            }
        });
        Ok(patch)
    }

    fn apply_canonical_ordering(&self, mut patch: RdfPatch) -> Result<RdfPatch> {
        // Group operations by type and apply canonical ordering within each group
        let mut headers = Vec::new();
        let mut prefixes = Vec::new();
        let mut tx_begin = Vec::new();
        let mut adds = Vec::new();
        let mut deletes = Vec::new();
        let mut graphs = Vec::new();
        let mut tx_end = Vec::new();

        for operation in &patch.operations {
            match operation {
                PatchOperation::Header { .. } => headers.push(operation.clone()),
                PatchOperation::AddPrefix { .. } | PatchOperation::DeletePrefix { .. } => {
                    prefixes.push(operation.clone())
                }
                PatchOperation::TransactionBegin { .. } => tx_begin.push(operation.clone()),
                PatchOperation::Add { .. } => adds.push(operation.clone()),
                PatchOperation::Delete { .. } => deletes.push(operation.clone()),
                PatchOperation::AddGraph { .. } | PatchOperation::DeleteGraph { .. } => {
                    graphs.push(operation.clone())
                }
                PatchOperation::TransactionCommit | PatchOperation::TransactionAbort => {
                    tx_end.push(operation.clone())
                }
            }
        }

        // Rebuild operations in canonical order
        patch.operations.clear();
        patch.operations.extend(headers);
        patch.operations.extend(prefixes);
        patch.operations.extend(tx_begin);
        patch.operations.extend(graphs);
        patch.operations.extend(deletes); // Deletes before adds
        patch.operations.extend(adds);
        patch.operations.extend(tx_end);

        Ok(patch)
    }

    fn sort_operations_by_subject(&self, mut patch: RdfPatch) -> Result<RdfPatch> {
        // Sort triple (Add/Delete) operations by subject, but ONLY within
        // contiguous runs of triple operations. A single global `sort_by` would
        // move structural operations such as TransactionCommit/TransactionAbort
        // (whose `extract_subject` is the empty string, which sorts before any
        // URI) ahead of the very operations they are meant to enclose. Sorting
        // within runs preserves the canonical grouping and all transaction
        // boundaries produced by `apply_canonical_ordering`.
        let ops = &mut patch.operations;
        let mut i = 0;
        while i < ops.len() {
            if Self::is_triple_operation(&ops[i]) {
                let run_start = i;
                while i < ops.len() && Self::is_triple_operation(&ops[i]) {
                    i += 1;
                }
                // Sort the contiguous run [run_start, i). Keep deletes before
                // adds (matching canonical ordering), then order by subject.
                ops[run_start..i].sort_by(|a, b| {
                    let key_a = (Self::triple_kind_order(a), Self::extract_subject(a));
                    let key_b = (Self::triple_kind_order(b), Self::extract_subject(b));
                    key_a.cmp(&key_b)
                });
            } else {
                i += 1;
            }
        }

        Ok(patch)
    }

    /// Whether an operation is a triple-level Add/Delete (the only operations
    /// that participate in subject sorting).
    fn is_triple_operation(operation: &PatchOperation) -> bool {
        matches!(
            operation,
            PatchOperation::Add { .. } | PatchOperation::Delete { .. }
        )
    }

    /// Ordering key that keeps deletes before adds within a run.
    fn triple_kind_order(operation: &PatchOperation) -> u8 {
        match operation {
            PatchOperation::Delete { .. } => 0,
            PatchOperation::Add { .. } => 1,
            _ => 2,
        }
    }

    fn extract_subject(operation: &PatchOperation) -> String {
        match operation {
            PatchOperation::Add { subject, .. } | PatchOperation::Delete { subject, .. } => {
                subject.clone()
            }
            _ => String::new(),
        }
    }
}

impl Default for PatchNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_normalize_keeps_transaction_commit_last() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::TransactionBegin {
            transaction_id: Some("tx-1".to_string()),
        });
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/s1".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });
        patch.add_operation(PatchOperation::TransactionCommit);

        let normalizer = PatchNormalizer::new();
        let normalized = normalizer.normalize(&patch).unwrap();

        // TransactionBegin must come first, TransactionCommit must come last,
        // and the Add must sit strictly between them (never before the begin or
        // after the commit).
        let ops = &normalized.operations;
        assert!(matches!(
            ops.first(),
            Some(PatchOperation::TransactionBegin { .. })
        ));
        assert!(matches!(
            ops.last(),
            Some(PatchOperation::TransactionCommit)
        ));
        let add_pos = ops
            .iter()
            .position(|op| matches!(op, PatchOperation::Add { .. }))
            .expect("Add operation should be present");
        assert!(add_pos > 0 && add_pos < ops.len() - 1);
    }

    #[test]
    fn regression_subject_sort_within_run() {
        let mut patch = RdfPatch::new();
        for subject in [
            "http://example.org/s3",
            "http://example.org/s1",
            "http://example.org/s2",
        ] {
            patch.add_operation(PatchOperation::Add {
                subject: subject.to_string(),
                predicate: "http://example.org/p".to_string(),
                object: "http://example.org/o".to_string(),
            });
        }

        let normalized = PatchNormalizer::new().normalize(&patch).unwrap();
        let subjects: Vec<String> = normalized
            .operations
            .iter()
            .filter_map(|op| match op {
                PatchOperation::Add { subject, .. } => Some(subject.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(
            subjects,
            vec![
                "http://example.org/s1".to_string(),
                "http://example.org/s2".to_string(),
                "http://example.org/s3".to_string(),
            ]
        );
    }
}
