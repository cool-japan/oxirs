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
        // Sort triple operations by subject
        patch.operations.sort_by(|a, b| {
            let subject_a = self.extract_subject(a);
            let subject_b = self.extract_subject(b);
            subject_a.cmp(&subject_b)
        });

        Ok(patch)
    }

    fn extract_subject(&self, operation: &PatchOperation) -> String {
        match operation {
            PatchOperation::Add { subject, .. } | PatchOperation::Delete { subject, .. } => {
                subject.clone()
            }
            _ => String::new(), // Non-triple operations sort first
        }
    }
}

impl Default for PatchNormalizer {
    fn default() -> Self {
        Self::new()
    }
}
