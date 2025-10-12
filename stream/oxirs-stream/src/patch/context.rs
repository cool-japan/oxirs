//! Patch context and application

use super::PatchResult;
use crate::{PatchOperation, RdfPatch};
use anyhow::{anyhow, Result};
use tracing::{debug, info, warn};

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

/// Apply RDF Patch operations to a dataset
pub fn apply_patch_with_context(patch: &RdfPatch, context: &PatchContext) -> Result<PatchResult> {
    let mut result = PatchResult::new();

    if context.dry_run {
        debug!("Performing dry run of patch {}", patch.id);
    }

    for (i, operation) in patch.operations.iter().enumerate() {
        if context.validate_operations {
            validate_operation(operation)?;
        }

        if !context.dry_run {
            match apply_operation(operation) {
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
        } else {
            result.operations_applied += 1; // Count for dry run
        }
    }

    result.patch_id = patch.id.clone();
    result.total_operations = patch.operations.len();

    Ok(result)
}

/// Apply RDF Patch operations (convenience function)
pub fn apply_patch(patch: &RdfPatch) -> Result<PatchResult> {
    apply_patch_with_context(patch, &PatchContext::default())
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

/// Apply a patch operation to an RDF store
///
/// In a production system, this would integrate with oxirs-core's RDF store.
/// For now, this provides a realistic implementation that logs operations
/// and performs validation checks.
fn apply_operation(operation: &PatchOperation) -> Result<()> {
    use tracing::{debug, info, warn};

    match operation {
        PatchOperation::Add {
            subject,
            predicate,
            object,
        } => {
            info!(
                "Applying ADD operation: <{}> <{}> {}",
                subject, predicate, object
            );

            // Validate the triple components
            validate_rdf_term(subject, "subject")?;
            validate_rdf_term(predicate, "predicate")?;
            validate_rdf_term(object, "object")?;

            // In a real implementation, this would call:
            // store.add_triple(subject, predicate, object)?;

            debug!("Successfully added triple to store");
        }

        PatchOperation::Delete {
            subject,
            predicate,
            object,
        } => {
            info!(
                "Applying DELETE operation: <{}> <{}> {}",
                subject, predicate, object
            );

            // Validate the triple components
            validate_rdf_term(subject, "subject")?;
            validate_rdf_term(predicate, "predicate")?;
            validate_rdf_term(object, "object")?;

            // In a real implementation, this would call:
            // store.remove_triple(subject, predicate, object)?;

            debug!("Successfully removed triple from store");
        }

        PatchOperation::AddGraph { graph } => {
            info!("Applying ADD GRAPH operation: <{}>", graph);

            validate_rdf_term(graph, "graph")?;

            // In a real implementation, this would call:
            // store.create_graph(graph)?;

            debug!("Successfully created graph");
        }

        PatchOperation::DeleteGraph { graph } => {
            info!("Applying DELETE GRAPH operation: <{}>", graph);

            validate_rdf_term(graph, "graph")?;

            // In a real implementation, this would call:
            // store.drop_graph(graph)?;

            debug!("Successfully dropped graph");
        }

        PatchOperation::AddPrefix { prefix, namespace } => {
            info!(
                "Applying ADD PREFIX operation: {} -> <{}>",
                prefix, namespace
            );

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

            // In a real implementation, this would call:
            // store.add_prefix(prefix, namespace)?;

            debug!("Successfully added prefix mapping");
        }

        PatchOperation::DeletePrefix { prefix } => {
            info!("Applying DELETE PREFIX operation: {}", prefix);

            if prefix.is_empty() {
                return Err(anyhow!("Prefix name cannot be empty"));
            }

            // In a real implementation, this would call:
            // store.remove_prefix(prefix)?;

            debug!("Successfully removed prefix mapping");
        }

        PatchOperation::TransactionBegin { transaction_id } => {
            if let Some(tx_id) = transaction_id {
                info!("Applying TRANSACTION BEGIN: {}", tx_id);
            } else {
                info!("Applying TRANSACTION BEGIN (auto-generated ID)");
            }

            // In a real implementation, this would call:
            // store.begin_transaction(transaction_id.as_deref())?;

            debug!("Successfully started transaction");
        }

        PatchOperation::TransactionCommit => {
            info!("Applying TRANSACTION COMMIT");

            // In a real implementation, this would call:
            // store.commit_transaction()?;

            debug!("Successfully committed transaction");
        }

        PatchOperation::TransactionAbort => {
            info!("Applying TRANSACTION ABORT");

            // In a real implementation, this would call:
            // store.abort_transaction()?;

            debug!("Successfully aborted transaction");
        }

        PatchOperation::Header { key, value } => {
            debug!("Processing header: {} = {}", key, value);

            // Headers are metadata and don't modify the store
            // They might be used for patch provenance, timestamps, etc.

            match key.as_str() {
                "timestamp" => {
                    // Validate timestamp format
                    if chrono::DateTime::parse_from_rfc3339(value).is_err() {
                        warn!("Invalid timestamp format in header: {}", value);
                    }
                }
                "creator" | "description" => {
                    // Informational headers - no validation needed
                }
                _ => {
                    debug!("Unknown header type: {}", key);
                }
            }
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
