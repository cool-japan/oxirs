//! Patch context and application

use super::PatchResult;
use crate::{PatchOperation, RdfPatch};
use anyhow::{anyhow, Result};
use tracing::debug;

pub struct PatchContext {
    pub strict_mode: bool,
    pub validate_operations: bool,
    pub dry_run: bool,
}

impl Default for PatchContext {
    fn default() -> Self {
        Self {
            strict_mode: false,
            validate_operations: true,
            dry_run: false,
        }
    }
}

/// A destination that RDF Patch operations can be applied to.
///
/// This is the seam between the patch engine and a concrete RDF store. Callers
/// that want a patch to actually mutate a dataset implement this trait (or use
/// the real store integration in [`crate::store_integration`]) and pass it to
/// [`apply_patch_to_sink`]. Implementations should return `Err` when an
/// operation cannot be applied so that the error is surfaced rather than
/// silently swallowed.
pub trait PatchSink {
    fn add_triple(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()>;
    fn remove_triple(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()>;
    fn add_graph(&mut self, graph: &str) -> Result<()>;
    fn delete_graph(&mut self, graph: &str) -> Result<()>;
    fn add_prefix(&mut self, prefix: &str, namespace: &str) -> Result<()>;
    fn remove_prefix(&mut self, prefix: &str) -> Result<()>;
    fn begin_transaction(&mut self, transaction_id: Option<&str>) -> Result<()>;
    fn commit_transaction(&mut self) -> Result<()>;
    fn abort_transaction(&mut self) -> Result<()>;
    /// Handle a header operation. Headers are metadata and default to a no-op.
    fn header(&mut self, _key: &str, _value: &str) -> Result<()> {
        Ok(())
    }
}

/// Validate (and optionally dry-run) an RDF Patch **without** persisting it.
///
/// This function performs real work — it validates every operation — but it has
/// no RDF store to mutate. To avoid the fail-loud contract violation of
/// reporting a fabricated success while the dataset is untouched, it requires
/// `context.dry_run` to be set. To actually apply a patch, use
/// [`apply_patch_to_sink`] with a [`PatchSink`], or the store-backed
/// integration in [`crate::store_integration`].
pub fn apply_patch_with_context(patch: &RdfPatch, context: &PatchContext) -> Result<PatchResult> {
    if !context.dry_run {
        return Err(anyhow!(
            "apply_patch_with_context cannot persist changes: no RDF store is wired. \
             Use apply_patch_to_sink() with a PatchSink, or set context.dry_run \
             for validation-only processing."
        ));
    }

    debug!("Performing dry run / validation of patch {}", patch.id);

    let mut result = PatchResult::new();
    for operation in patch.operations.iter() {
        if context.validate_operations {
            validate_operation(operation)?;
        }
        result.operations_applied += 1; // Counted as validated for dry run.
    }

    result.patch_id = patch.id.clone();
    result.total_operations = patch.operations.len();
    Ok(result)
}

/// Validate an RDF Patch without applying it (convenience function).
///
/// Uses the default context which has `dry_run = false`, so this returns an
/// error directing callers to a real sink. It exists to keep the historical
/// signature; prefer [`apply_patch_to_sink`].
pub fn apply_patch(patch: &RdfPatch) -> Result<PatchResult> {
    apply_patch_with_context(patch, &PatchContext::default())
}

/// Apply RDF Patch operations to a concrete [`PatchSink`].
///
/// Every operation is validated (when configured) and then applied to `sink`.
/// Errors from the sink are recorded in [`PatchResult::errors`]; in
/// `strict_mode` the first failure aborts with an error. When `context.dry_run`
/// is set the sink is never touched and operations are only validated.
pub fn apply_patch_to_sink<S: PatchSink + ?Sized>(
    patch: &RdfPatch,
    context: &PatchContext,
    sink: &mut S,
) -> Result<PatchResult> {
    let mut result = PatchResult::new();

    for (i, operation) in patch.operations.iter().enumerate() {
        if context.validate_operations {
            validate_operation(operation)?;
        }

        if context.dry_run {
            result.operations_applied += 1;
            continue;
        }

        match apply_operation(operation, sink) {
            Ok(_) => {
                result.operations_applied += 1;
                debug!("Applied operation {}: {:?}", i, operation);
            }
            Err(e) => {
                result.errors.push(format!("Operation {i}: {e}"));
                if context.strict_mode {
                    return Err(anyhow!("Failed to apply operation {}: {}", i, e));
                }
            }
        }
    }

    result.patch_id = patch.id.clone();
    result.total_operations = patch.operations.len();
    Ok(result)
}

fn validate_operation(operation: &PatchOperation) -> Result<()> {
    match operation {
        PatchOperation::Add {
            subject,
            predicate,
            object,
        }
        | PatchOperation::Delete {
            subject,
            predicate,
            object,
        } => {
            if subject.is_empty() || predicate.is_empty() || object.is_empty() {
                return Err(anyhow!("Triple operation has empty components"));
            }
        }
        PatchOperation::AddGraph { graph } | PatchOperation::DeleteGraph { graph } => {
            if graph.is_empty() {
                return Err(anyhow!("Graph operation has empty graph URI"));
            }
        }
        PatchOperation::AddPrefix {
            prefix: _,
            namespace: _,
        } => {
            // Prefix operations are always valid
        }
        PatchOperation::DeletePrefix { prefix: _ } => {
            // Prefix operations are always valid
        }
        PatchOperation::TransactionBegin { .. } => {
            // Transaction operations are always valid
        }
        PatchOperation::TransactionCommit => {
            // Transaction operations are always valid
        }
        PatchOperation::TransactionAbort => {
            // Transaction operations are always valid
        }
        PatchOperation::Header { .. } => {
            // Header operations are always valid
        }
    }
    Ok(())
}

/// Apply a single patch operation to a [`PatchSink`].
///
/// Each triple/graph component is validated before being handed to the sink,
/// and any error returned by the sink is propagated to the caller.
fn apply_operation<S: PatchSink + ?Sized>(operation: &PatchOperation, sink: &mut S) -> Result<()> {
    use tracing::warn;

    match operation {
        PatchOperation::Add {
            subject,
            predicate,
            object,
        } => {
            validate_rdf_term(subject, "subject")?;
            validate_rdf_term(predicate, "predicate")?;
            validate_rdf_term(object, "object")?;
            sink.add_triple(subject, predicate, object)?;
        }

        PatchOperation::Delete {
            subject,
            predicate,
            object,
        } => {
            validate_rdf_term(subject, "subject")?;
            validate_rdf_term(predicate, "predicate")?;
            validate_rdf_term(object, "object")?;
            sink.remove_triple(subject, predicate, object)?;
        }

        PatchOperation::AddGraph { graph } => {
            validate_rdf_term(graph, "graph")?;
            sink.add_graph(graph)?;
        }

        PatchOperation::DeleteGraph { graph } => {
            validate_rdf_term(graph, "graph")?;
            sink.delete_graph(graph)?;
        }

        PatchOperation::AddPrefix { prefix, namespace } => {
            if prefix.is_empty() {
                return Err(anyhow!("Prefix name cannot be empty"));
            }
            if !namespace.starts_with("http://")
                && !namespace.starts_with("https://")
                && !namespace.starts_with("urn:")
            {
                warn!(
                    "Namespace '{}' doesn't follow standard URI scheme",
                    namespace
                );
            }
            sink.add_prefix(prefix, namespace)?;
        }

        PatchOperation::DeletePrefix { prefix } => {
            if prefix.is_empty() {
                return Err(anyhow!("Prefix name cannot be empty"));
            }
            sink.remove_prefix(prefix)?;
        }

        PatchOperation::TransactionBegin { transaction_id } => {
            sink.begin_transaction(transaction_id.as_deref())?;
        }

        PatchOperation::TransactionCommit => {
            sink.commit_transaction()?;
        }

        PatchOperation::TransactionAbort => {
            sink.abort_transaction()?;
        }

        PatchOperation::Header { key, value } => {
            if key == "timestamp" && chrono::DateTime::parse_from_rfc3339(value).is_err() {
                warn!("Invalid timestamp format in header: {}", value);
            }
            sink.header(key, value)?;
        }
    }

    Ok(())
}

/// Validate an RDF term (IRI, blank node, or literal)
fn validate_rdf_term(term: &str, term_type: &str) -> Result<()> {
    if term.is_empty() {
        return Err(anyhow!("{} cannot be empty", term_type));
    }

    // Check for IRI format
    if term.starts_with('<') && term.ends_with('>') {
        let iri = &term[1..term.len() - 1];
        if iri.is_empty() {
            return Err(anyhow!("Empty IRI in {}", term_type));
        }

        // Basic IRI validation - should contain valid characters
        if iri.contains(' ') || iri.contains('\n') || iri.contains('\t') {
            return Err(anyhow!("Invalid characters in IRI: {}", iri));
        }
    }
    // Check for blank node format
    else if term.starts_with('_') {
        if !term.starts_with("_:") {
            return Err(anyhow!("Invalid blank node format: {}", term));
        }

        let local_name = &term[2..];
        if local_name.is_empty() {
            return Err(anyhow!("Empty blank node local name"));
        }
    }
    // Check for literal format (quoted strings)
    else if term.starts_with('"') {
        if !term.ends_with('"') && !term.contains("\"@") && !term.contains("\"^^") {
            return Err(anyhow!("Invalid literal format: {}", term));
        }
    }
    // Check for prefixed name
    else if term.contains(':') {
        let parts: Vec<&str> = term.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(anyhow!("Invalid prefixed name format: {}", term));
        }

        let prefix = parts[0];
        let local_name = parts[1];

        // Prefix should not be empty (unless it's the default prefix)
        if prefix.is_empty() && local_name.is_empty() {
            return Err(anyhow!("Invalid prefixed name: {}", term));
        }
    }
    // If none of the above, it might be a relative IRI or invalid
    else if term_type == "predicate" {
        // Predicates should always be IRIs or prefixed names
        return Err(anyhow!(
            "Predicate must be an IRI or prefixed name: {}",
            term
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory sink used to prove operations really reach the store.
    #[derive(Default)]
    struct CollectingSink {
        triples: Vec<(String, String, String)>,
        removed: Vec<(String, String, String)>,
        transactions: Vec<String>,
    }

    impl PatchSink for CollectingSink {
        fn add_triple(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()> {
            self.triples.push((
                subject.to_string(),
                predicate.to_string(),
                object.to_string(),
            ));
            Ok(())
        }
        fn remove_triple(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()> {
            self.removed.push((
                subject.to_string(),
                predicate.to_string(),
                object.to_string(),
            ));
            Ok(())
        }
        fn add_graph(&mut self, _graph: &str) -> Result<()> {
            Ok(())
        }
        fn delete_graph(&mut self, _graph: &str) -> Result<()> {
            Ok(())
        }
        fn add_prefix(&mut self, _prefix: &str, _namespace: &str) -> Result<()> {
            Ok(())
        }
        fn remove_prefix(&mut self, _prefix: &str) -> Result<()> {
            Ok(())
        }
        fn begin_transaction(&mut self, transaction_id: Option<&str>) -> Result<()> {
            self.transactions
                .push(format!("begin:{}", transaction_id.unwrap_or("auto")));
            Ok(())
        }
        fn commit_transaction(&mut self) -> Result<()> {
            self.transactions.push("commit".to_string());
            Ok(())
        }
        fn abort_transaction(&mut self) -> Result<()> {
            self.transactions.push("abort".to_string());
            Ok(())
        }
    }

    #[test]
    fn regression_apply_patch_to_sink_actually_mutates() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });
        patch.add_operation(PatchOperation::Delete {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/old".to_string(),
        });

        let mut sink = CollectingSink::default();
        let result = apply_patch_to_sink(&patch, &PatchContext::default(), &mut sink).unwrap();

        assert_eq!(result.operations_applied, 2);
        assert_eq!(sink.triples.len(), 1);
        assert_eq!(sink.removed.len(), 1);
        assert_eq!(sink.triples[0].0, "http://example.org/s");
    }

    #[test]
    fn regression_storeless_apply_is_fail_loud() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });

        // Non-dry-run without a store must error rather than fabricate success.
        let ctx = PatchContext {
            strict_mode: false,
            validate_operations: true,
            dry_run: false,
        };
        assert!(apply_patch_with_context(&patch, &ctx).is_err());

        // Dry run validates and reports success without a store.
        let dry = PatchContext {
            dry_run: true,
            ..Default::default()
        };
        let result = apply_patch_with_context(&patch, &dry).unwrap();
        assert_eq!(result.operations_applied, 1);
    }
}
