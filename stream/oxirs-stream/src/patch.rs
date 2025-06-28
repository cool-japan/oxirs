//! # RDF Patch Support
//!
//! RDF Patch format support for atomic updates.
//!
//! This module implements the RDF Patch specification for describing
//! atomic changes to RDF datasets. RDF Patch provides a standardized
//! way to represent additions, deletions, and graph operations.
//!
//! Reference: https://afs.github.io/rdf-patch/

use crate::{PatchOperation, RdfPatch};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use tracing::{debug, warn};

/// RDF Patch parser with comprehensive support for the specification
pub struct PatchParser {
    strict_mode: bool,
    current_line: usize,
    prefixes: HashMap<String, String>,
}

impl PatchParser {
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            current_line: 0,
            prefixes: HashMap::new(),
        }
    }

    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Parse RDF Patch from string
    pub fn parse(&mut self, input: &str) -> Result<RdfPatch> {
        let mut patch = RdfPatch::new();
        self.current_line = 0;
        self.prefixes.clear();

        // Add common prefixes
        self.prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        self.prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        self.prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        for line in input.lines() {
            self.current_line += 1;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse the line
            match self.parse_line(line) {
                Ok(Some(operation)) => {
                    // Handle special operations that affect patch state
                    match &operation {
                        PatchOperation::TransactionBegin { transaction_id } => {
                            patch.transaction_id = transaction_id.clone();
                        }
                        PatchOperation::Header { key, value } => {
                            patch.headers.insert(key.clone(), value.clone());
                        }
                        PatchOperation::AddPrefix { prefix, namespace } => {
                            patch.prefixes.insert(prefix.clone(), namespace.clone());
                        }
                        _ => {}
                    }
                    patch.add_operation(operation);
                }
                Ok(None) => {
                    // Line was a directive (like @prefix), not an operation
                    continue;
                }
                Err(e) => {
                    if self.strict_mode {
                        return Err(anyhow!("Parse error at line {}: {}", self.current_line, e));
                    } else {
                        warn!(
                            "Ignoring invalid line {}: {} ({})",
                            self.current_line, line, e
                        );
                    }
                }
            }
        }

        debug!(
            "Parsed RDF Patch with {} operations",
            patch.operations.len()
        );
        Ok(patch)
    }

    fn parse_line(&mut self, line: &str) -> Result<Option<PatchOperation>> {
        // Handle prefix declarations
        if line.starts_with("@prefix") {
            self.parse_prefix(line)?;
            return Ok(None);
        }

        // Transaction and header operations are now handled in the main match below

        // Parse operation lines
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Err(anyhow!("Empty operation line"));
        }

        let operation = parts[0];
        match operation {
            "A" => self.parse_add_operation(&parts[1..]),
            "D" => self.parse_delete_operation(&parts[1..]),
            "PA" => self.parse_prefix_add(&parts[1..]),
            "PD" => self.parse_prefix_delete(&parts[1..]),
            "GA" => self.parse_graph_add(&parts[1..]),
            "GD" => self.parse_graph_delete(&parts[1..]),
            "TX" => self.parse_transaction_begin(&parts[1..]),
            "TC" => Ok(Some(PatchOperation::TransactionCommit)),
            "TA" => Ok(Some(PatchOperation::TransactionAbort)),
            "H" => self.parse_header(&parts[1..]),
            _ => Err(anyhow!("Unknown operation: {}", operation)),
        }
    }

    fn parse_prefix(&mut self, line: &str) -> Result<()> {
        // Format: @prefix prefix: <uri>
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(anyhow!("Invalid prefix declaration"));
        }

        let prefix_with_colon = parts[1];
        let prefix = prefix_with_colon.trim_end_matches(':');
        let uri = parts[2].trim_matches('<').trim_matches('>');

        self.prefixes.insert(prefix.to_string(), uri.to_string());
        debug!("Added prefix: {} -> {}", prefix, uri);
        Ok(())
    }

    fn parse_add_operation(&self, parts: &[&str]) -> Result<Option<PatchOperation>> {
        if parts.len() < 3 {
            return Err(anyhow!(
                "Add operation requires subject, predicate, and object"
            ));
        }

        let subject = self.expand_term(parts[0])?;
        let predicate = self.expand_term(parts[1])?;
        let object = self.expand_term(parts[2])?;

        Ok(Some(PatchOperation::Add {
            subject,
            predicate,
            object,
        }))
    }

    fn parse_delete_operation(&self, parts: &[&str]) -> Result<Option<PatchOperation>> {
        if parts.len() < 3 {
            return Err(anyhow!(
                "Delete operation requires subject, predicate, and object"
            ));
        }

        let subject = self.expand_term(parts[0])?;
        let predicate = self.expand_term(parts[1])?;
        let object = self.expand_term(parts[2])?;

        Ok(Some(PatchOperation::Delete {
            subject,
            predicate,
            object,
        }))
    }

    fn parse_prefix_add(&self, parts: &[&str]) -> Result<Option<PatchOperation>> {
        if parts.len() < 2 {
            return Err(anyhow!("Prefix add requires prefix and namespace"));
        }
        
        let prefix = parts[0].trim_end_matches(':').to_string();
        let namespace = parts[1].trim_matches('<').trim_matches('>').to_string();
        
        Ok(Some(PatchOperation::AddPrefix {
            prefix,
            namespace,
        }))
    }

    fn parse_prefix_delete(&self, parts: &[&str]) -> Result<Option<PatchOperation>> {
        if parts.is_empty() {
            return Err(anyhow!("Prefix delete requires prefix name"));
        }
        
        let prefix = parts[0].trim_end_matches(':').to_string();
        
        Ok(Some(PatchOperation::DeletePrefix { prefix }))
    }

    fn parse_graph_add(&self, parts: &[&str]) -> Result<Option<PatchOperation>> {
        if parts.is_empty() {
            return Err(anyhow!("Graph add operation requires graph URI"));
        }

        let graph = self.expand_term(parts[0])?;
        Ok(Some(PatchOperation::AddGraph { graph }))
    }

    fn parse_graph_delete(&self, parts: &[&str]) -> Result<Option<PatchOperation>> {
        if parts.is_empty() {
            return Err(anyhow!("Graph delete operation requires graph URI"));
        }

        let graph = self.expand_term(parts[0])?;
        Ok(Some(PatchOperation::DeleteGraph { graph }))
    }

    fn parse_transaction_begin(&self, parts: &[&str]) -> Result<Option<PatchOperation>> {
        let transaction_id = if !parts.is_empty() {
            Some(parts[0].to_string())
        } else {
            None
        };
        
        Ok(Some(PatchOperation::TransactionBegin { transaction_id }))
    }
    
    fn parse_header(&self, parts: &[&str]) -> Result<Option<PatchOperation>> {
        if parts.len() < 2 {
            return Err(anyhow!("Header requires key and value"));
        }
        
        let key = parts[0].to_string();
        let value = parts[1..].join(" ");
        
        Ok(Some(PatchOperation::Header { key, value }))
    }

    fn expand_term(&self, term: &str) -> Result<String> {
        if term.starts_with('<') && term.ends_with('>') {
            // Full URI
            Ok(term[1..term.len() - 1].to_string())
        } else if term.starts_with('"') {
            // Literal
            Ok(term.to_string())
        } else if term.starts_with('_') {
            // Blank node
            Ok(term.to_string())
        } else if term.contains(':') {
            // Prefixed name
            let parts: Vec<&str> = term.splitn(2, ':').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let local = parts[1];

                if let Some(namespace) = self.prefixes.get(prefix) {
                    Ok(format!("{}{}", namespace, local))
                } else {
                    if self.strict_mode {
                        Err(anyhow!("Unknown prefix: {}", prefix))
                    } else {
                        // Return as-is in non-strict mode
                        Ok(term.to_string())
                    }
                }
            } else {
                Err(anyhow!("Invalid prefixed name: {}", term))
            }
        } else {
            // Assume it's a relative URI or local name
            Ok(term.to_string())
        }
    }
}

impl Default for PatchParser {
    fn default() -> Self {
        Self::new()
    }
}

/// RDF Patch serializer with full specification support
pub struct PatchSerializer {
    pretty_print: bool,
    include_metadata: bool,
    prefixes: HashMap<String, String>,
}

impl PatchSerializer {
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        Self {
            pretty_print: true,
            include_metadata: true,
            prefixes,
        }
    }

    pub fn with_pretty_print(mut self, pretty: bool) -> Self {
        self.pretty_print = pretty;
        self
    }

    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    pub fn add_prefix(&mut self, prefix: String, namespace: String) {
        self.prefixes.insert(prefix, namespace);
    }

    /// Serialize RDF Patch to string
    pub fn serialize(&self, patch: &RdfPatch) -> Result<String> {
        let mut output = String::new();

        // Write header
        if self.include_metadata {
            output.push_str("# RDF Patch\n");
            output.push_str(&format!(
                "# Generated: {}\n",
                patch.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
            ));
            output.push_str(&format!("# Patch ID: {}\n", patch.id));
            output.push_str(&format!("# Operations: {}\n", patch.operations.len()));
            output.push('\n');
        }

        // Write prefixes
        if self.pretty_print {
            for (prefix, namespace) in &self.prefixes {
                output.push_str(&format!("@prefix {}: <{}> .\n", prefix, namespace));
            }
            if !self.prefixes.is_empty() {
                output.push('\n');
            }
        }

        // Write operations
        for (i, operation) in patch.operations.iter().enumerate() {
            let op_str = self.serialize_operation(operation)?;
            output.push_str(&op_str);
            output.push('\n');

            // Add spacing for readability in pretty print mode
            if self.pretty_print && i > 0 && i % 10 == 0 {
                output.push('\n');
            }
        }

        Ok(output)
    }

    fn serialize_operation(&self, operation: &PatchOperation) -> Result<String> {
        match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => {
                let s = self.compact_term(subject);
                let p = self.compact_term(predicate);
                let o = self.compact_term(object);
                Ok(format!("A {} {} {} .", s, p, o))
            }
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => {
                let s = self.compact_term(subject);
                let p = self.compact_term(predicate);
                let o = self.compact_term(object);
                Ok(format!("D {} {} {} .", s, p, o))
            }
            PatchOperation::AddGraph { graph } => {
                let g = self.compact_term(graph);
                Ok(format!("GA {} .", g))
            }
            PatchOperation::DeleteGraph { graph } => {
                let g = self.compact_term(graph);
                Ok(format!("GD {} .", g))
            }
            PatchOperation::AddPrefix { prefix, namespace } => {
                Ok(format!("PA {}: <{}> .", prefix, namespace))
            }
            PatchOperation::DeletePrefix { prefix } => {
                Ok(format!("PD {}: .", prefix))
            }
            PatchOperation::TransactionBegin { transaction_id } => {
                if let Some(id) = transaction_id {
                    Ok(format!("TX {} .", id))
                } else {
                    Ok("TX .".to_string())
                }
            }
            PatchOperation::TransactionCommit => Ok("TC .".to_string()),
            PatchOperation::TransactionAbort => Ok("TA .".to_string()),
            PatchOperation::Header { key, value } => {
                Ok(format!("H {} {} .", key, value))
            }
        }
    }

    fn compact_term(&self, term: &str) -> String {
        // Try to find a prefix that matches
        for (prefix, namespace) in &self.prefixes {
            if term.starts_with(namespace) {
                let local = &term[namespace.len()..];
                return format!("{}:{}", prefix, local);
            }
        }

        // If no prefix matches, use full URI notation
        if term.starts_with('"') || term.starts_with('_') {
            // Literal or blank node - return as-is
            term.to_string()
        } else {
            // Wrap in angle brackets for full URI
            format!("<{}>", term)
        }
    }
}

impl Default for PatchSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Patch application context
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
                    result.errors.push(format!("Operation {}: {}", i, e));
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
        PatchOperation::AddPrefix { prefix: _, namespace: _ } => {
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

fn apply_operation(_operation: &PatchOperation) -> Result<()> {
    // TODO: Implement actual operation application
    // This would integrate with the RDF store to perform the operations
    Ok(())
}

/// Result of patch application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchResult {
    pub patch_id: String,
    pub total_operations: usize,
    pub operations_applied: usize,
    pub errors: Vec<String>,
    pub applied_at: DateTime<Utc>,
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
            PatchOperation::AddPrefix { prefix, namespace } => PatchOperation::DeletePrefix {
                prefix: prefix.clone(),
            },
            PatchOperation::DeletePrefix { prefix } => {
                // Can't reverse a prefix deletion without knowing the namespace
                // Skip or add as header
                reverse_patch.add_operation(PatchOperation::Header {
                    key: "warning".to_string(),
                    value: format!("Cannot reverse prefix deletion for '{}'", prefix),
                });
                continue;
            }
            PatchOperation::TransactionBegin { transaction_id } => {
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
                key: format!("reversed-{}", key),
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
        let operation_key = format!("{:?}", operation);

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
                    warnings.push(format!("Triple added after being deleted: {:?}", triple));
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
                    warnings.push(format!(
                        "Triple deleted without prior addition: {:?}",
                        triple
                    ));
                }
                deletes.insert(triple);
            }
            _ => {} // Graph operations don't conflict in the same way
        }
    }

    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_serialization() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Header {
            key: "creator".to_string(),
            value: "test-suite".to_string(),
        });
        patch.add_operation(PatchOperation::TransactionBegin {
            transaction_id: Some("tx-123".to_string()),
        });
        patch.add_operation(PatchOperation::AddPrefix {
            prefix: "ex".to_string(),
            namespace: "http://example.org/".to_string(),
        });
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "\"Object literal\"".to_string(),
        });
        patch.add_operation(PatchOperation::Delete {
            subject: "http://example.org/subject2".to_string(),
            predicate: "http://example.org/predicate2".to_string(),
            object: "http://example.org/object2".to_string(),
        });
        patch.add_operation(PatchOperation::TransactionCommit);

        let serializer = PatchSerializer::new();
        let result = serializer.serialize(&patch);
        
        assert!(result.is_ok());
        let serialized = result.unwrap();
        assert!(serialized.contains("H creator test-suite"));
        assert!(serialized.contains("TX tx-123"));
        assert!(serialized.contains("PA ex:"));
        assert!(serialized.contains("A "));
        assert!(serialized.contains("D "));
        assert!(serialized.contains("TC"));
        assert!(serialized.contains("@prefix"));
    }

    #[test]
    fn test_patch_parsing() {
        let patch_content = r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

H creator test-parser .
TX tx-456 .
PA ex2: <http://example2.org/> .
A ex:subject ex:predicate "Object literal" .
D ex:subject2 ex:predicate2 ex:object2 .
GA ex:graph1 .
GD ex:graph2 .
PD old: .
TC .
"#;

        let mut parser = PatchParser::new();
        let result = parser.parse(patch_content);
        
        assert!(result.is_ok());
        let patch = result.unwrap();
        assert_eq!(patch.operations.len(), 9);
        
        // Check header was captured
        assert_eq!(patch.headers.get("creator"), Some(&"test-parser".to_string()));
        
        // Check transaction ID was captured
        assert_eq!(patch.transaction_id, Some("tx-456".to_string()));
        
        // Check prefix was captured
        assert_eq!(patch.prefixes.get("ex2"), Some(&"http://example2.org/".to_string()));
        
        match &patch.operations[0] {
            PatchOperation::Header { key, value } => {
                assert_eq!(key, "creator");
                assert_eq!(value, "test-parser");
            }
            _ => panic!("Expected Header operation"),
        }
        
        match &patch.operations[1] {
            PatchOperation::TransactionBegin { transaction_id } => {
                assert_eq!(transaction_id, &Some("tx-456".to_string()));
            }
            _ => panic!("Expected TransactionBegin operation"),
        }
    }

    #[test]
    fn test_patch_round_trip() {
        let mut original_patch = RdfPatch::new();
        original_patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "\"Test object\"".to_string(),
        });

        // Serialize to string
        let serialized = original_patch.to_rdf_patch_format().unwrap();
        
        // Parse back from string
        let parsed_patch = RdfPatch::from_rdf_patch_format(&serialized).unwrap();
        
        // Check that we get the same operations
        assert_eq!(original_patch.operations.len(), parsed_patch.operations.len());
        
        match (&original_patch.operations[0], &parsed_patch.operations[0]) {
            (
                PatchOperation::Add { subject: s1, predicate: p1, object: o1 },
                PatchOperation::Add { subject: s2, predicate: p2, object: o2 }
            ) => {
                assert_eq!(s1, s2);
                assert_eq!(p1, p2);
                assert_eq!(o1, o2);
            }
            _ => panic!("Operations don't match"),
        }
    }

    #[test]
    fn test_reverse_patch() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::TransactionBegin {
            transaction_id: Some("tx-789".to_string()),
        });
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });
        patch.add_operation(PatchOperation::AddGraph {
            graph: "http://example.org/graph".to_string(),
        });
        patch.add_operation(PatchOperation::TransactionCommit);
        patch.transaction_id = Some("tx-789".to_string());

        let reverse = create_reverse_patch(&patch).unwrap();
        
        // Should have TX, two reversed operations, and TC
        assert!(reverse.operations.len() >= 4);
        
        // First should be transaction begin (reversing the commit)
        match &reverse.operations[0] {
            PatchOperation::TransactionBegin { .. } => {},
            _ => panic!("Expected TransactionBegin operation"),
        }
        
        // Find the reversed operations
        let has_delete_graph = reverse.operations.iter().any(|op| {
            matches!(op, PatchOperation::DeleteGraph { graph } if graph == "http://example.org/graph")
        });
        let has_delete_triple = reverse.operations.iter().any(|op| {
            matches!(op, PatchOperation::Delete { subject, .. } if subject == "http://example.org/s")
        });
        
        assert!(has_delete_graph);
        assert!(has_delete_triple);
        
        // Last should be transaction commit
        match reverse.operations.last() {
            Some(PatchOperation::TransactionCommit) => {},
            _ => panic!("Expected TransactionCommit as last operation"),
        }
    }

    #[test]
    fn test_patch_optimization() {
        let mut patch = RdfPatch::new();
        let operation = PatchOperation::Add {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        };
        
        // Add the same operation twice
        patch.add_operation(operation.clone());
        patch.add_operation(operation);
        
        let optimized = optimize_patch(&patch).unwrap();
        
        // Should remove duplicate
        assert_eq!(optimized.operations.len(), 1);
    }
    
    #[test]
    fn test_transactional_patch() {
        let operations = vec![
            PatchOperation::Add {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
            },
            PatchOperation::Delete {
                subject: "s2".to_string(),
                predicate: "p2".to_string(),
                object: "o2".to_string(),
            },
        ];
        
        let patch = create_transactional_patch(operations);
        
        // Should have TX + 2 operations + TC = 4 total
        assert_eq!(patch.operations.len(), 4);
        
        // First should be transaction begin
        assert!(matches!(&patch.operations[0], PatchOperation::TransactionBegin { .. }));
        
        // Last should be transaction commit
        assert!(matches!(&patch.operations[3], PatchOperation::TransactionCommit));
        
        // Should have transaction ID set
        assert!(patch.transaction_id.is_some());
    }

    #[test]
    fn test_patch_validation() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Delete {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });
        
        let warnings = validate_patch(&patch).unwrap();
        
        // Should warn about deleting without prior addition
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("deleted without prior addition"));
    }

    #[test]
    fn test_patch_application() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });

        let context = PatchContext {
            strict_mode: false,
            validate_operations: true,
            dry_run: true,
        };

        let result = apply_patch_with_context(&patch, &context).unwrap();
        
        assert_eq!(result.total_operations, 1);
        assert_eq!(result.operations_applied, 1);
        assert!(result.is_success());
        assert_eq!(result.success_rate(), 1.0);
    }
}
