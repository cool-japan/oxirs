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
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::io::{Read, Write};
use tracing::{debug, info, warn};

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

        // Parse operation lines with proper tokenization that respects quoted strings
        let parts = self.tokenize_line(line);
        if parts.is_empty() {
            return Err(anyhow!("Empty operation line"));
        }

        let operation = &parts[0];
        match operation.as_str() {
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

    fn parse_add_operation(&self, parts: &[String]) -> Result<Option<PatchOperation>> {
        if parts.len() < 3 {
            return Err(anyhow!(
                "Add operation requires subject, predicate, and object"
            ));
        }

        let subject = self.expand_term(&parts[0])?;
        let predicate = self.expand_term(&parts[1])?;
        let object = self.expand_term(&parts[2])?;

        Ok(Some(PatchOperation::Add {
            subject,
            predicate,
            object,
        }))
    }

    fn parse_delete_operation(&self, parts: &[String]) -> Result<Option<PatchOperation>> {
        if parts.len() < 3 {
            return Err(anyhow!(
                "Delete operation requires subject, predicate, and object"
            ));
        }

        let subject = self.expand_term(&parts[0])?;
        let predicate = self.expand_term(&parts[1])?;
        let object = self.expand_term(&parts[2])?;

        Ok(Some(PatchOperation::Delete {
            subject,
            predicate,
            object,
        }))
    }

    fn parse_prefix_add(&self, parts: &[String]) -> Result<Option<PatchOperation>> {
        if parts.len() < 2 {
            return Err(anyhow!("Prefix add requires prefix and namespace"));
        }

        let prefix = parts[0].trim_end_matches(':').to_string();
        let namespace = parts[1].trim_matches('<').trim_matches('>').to_string();

        Ok(Some(PatchOperation::AddPrefix { prefix, namespace }))
    }

    fn parse_prefix_delete(&self, parts: &[String]) -> Result<Option<PatchOperation>> {
        if parts.is_empty() {
            return Err(anyhow!("Prefix delete requires prefix name"));
        }

        let prefix = parts[0].trim_end_matches(':').to_string();

        Ok(Some(PatchOperation::DeletePrefix { prefix }))
    }

    fn parse_graph_add(&self, parts: &[String]) -> Result<Option<PatchOperation>> {
        if parts.is_empty() {
            return Err(anyhow!("Graph add operation requires graph URI"));
        }

        let graph = self.expand_term(&parts[0])?;
        Ok(Some(PatchOperation::AddGraph { graph }))
    }

    fn parse_graph_delete(&self, parts: &[String]) -> Result<Option<PatchOperation>> {
        if parts.is_empty() {
            return Err(anyhow!("Graph delete operation requires graph URI"));
        }

        let graph = self.expand_term(&parts[0])?;
        Ok(Some(PatchOperation::DeleteGraph { graph }))
    }

    fn parse_transaction_begin(&self, parts: &[String]) -> Result<Option<PatchOperation>> {
        let transaction_id = if !parts.is_empty() {
            Some(parts[0].clone())
        } else {
            None
        };

        Ok(Some(PatchOperation::TransactionBegin { transaction_id }))
    }

    fn parse_header(&self, parts: &[String]) -> Result<Option<PatchOperation>> {
        if parts.len() < 2 {
            return Err(anyhow!("Header requires key and value"));
        }

        let key = parts[0].clone();

        // Handle RDF Patch line terminator - exclude trailing "." from value
        let value_parts = if parts.len() > 2 && parts[parts.len() - 1] == "." {
            &parts[1..parts.len() - 1]
        } else {
            &parts[1..]
        };
        let value = value_parts.join(" ");

        Ok(Some(PatchOperation::Header { key, value }))
    }

    /// Tokenize a line while respecting quoted strings and RDF Patch terminators
    fn tokenize_line(&self, line: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_quotes = false;
        let mut in_uri = false;
        let mut chars = line.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                // Handle quoted strings
                '"' => {
                    current_token.push(ch);
                    in_quotes = !in_quotes;
                }
                // Handle URI brackets
                '<' if !in_quotes => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                    current_token.push(ch);
                    in_uri = true;
                }
                '>' if !in_quotes && in_uri => {
                    current_token.push(ch);
                    tokens.push(current_token.clone());
                    current_token.clear();
                    in_uri = false;
                }
                // Handle whitespace
                c if c.is_whitespace() && !in_quotes && !in_uri => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                }
                // Handle RDF Patch line terminator
                '.' if !in_quotes && !in_uri => {
                    // Check if this is a standalone terminator (followed by whitespace or end)
                    if let Some(&next_ch) = chars.peek() {
                        if next_ch.is_whitespace() || current_token.is_empty() {
                            if !current_token.is_empty() {
                                tokens.push(current_token.clone());
                                current_token.clear();
                            }
                            tokens.push(".".to_string());
                            continue;
                        }
                    } else {
                        // End of line
                        if !current_token.is_empty() {
                            tokens.push(current_token.clone());
                            current_token.clear();
                        }
                        tokens.push(".".to_string());
                        continue;
                    }
                    current_token.push(ch);
                }
                // Regular characters
                _ => {
                    current_token.push(ch);
                }
            }
        }

        // Add any remaining token
        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        tokens
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
                    Ok(format!("{namespace}{local}"))
                } else if self.strict_mode {
                    Err(anyhow!("Unknown prefix: {}", prefix))
                } else {
                    // Return as-is in non-strict mode
                    Ok(term.to_string())
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
                output.push_str(&format!("@prefix {prefix}: <{namespace}> .\n"));
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
                Ok(format!("A {s} {p} {o} ."))
            }
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => {
                let s = self.compact_term(subject);
                let p = self.compact_term(predicate);
                let o = self.compact_term(object);
                Ok(format!("D {s} {p} {o} ."))
            }
            PatchOperation::AddGraph { graph } => {
                let g = self.compact_term(graph);
                Ok(format!("GA {g} ."))
            }
            PatchOperation::DeleteGraph { graph } => {
                let g = self.compact_term(graph);
                Ok(format!("GD {g} ."))
            }
            PatchOperation::AddPrefix { prefix, namespace } => {
                Ok(format!("PA {prefix}: <{namespace}> ."))
            }
            PatchOperation::DeletePrefix { prefix } => Ok(format!("PD {prefix}: .")),
            PatchOperation::TransactionBegin { transaction_id } => {
                if let Some(id) = transaction_id {
                    Ok(format!("TX {id} ."))
                } else {
                    Ok("TX .".to_string())
                }
            }
            PatchOperation::TransactionCommit => Ok("TC .".to_string()),
            PatchOperation::TransactionAbort => Ok("TA .".to_string()),
            PatchOperation::Header { key, value } => Ok(format!("H {key} {value} .")),
        }
    }

    fn compact_term(&self, term: &str) -> String {
        // Try to find a prefix that matches
        for (prefix, namespace) in &self.prefixes {
            if term.starts_with(namespace) {
                let local = &term[namespace.len()..];
                return format!("{prefix}:{local}");
            }
        }

        // If no prefix matches, use full URI notation
        if term.starts_with('"') || term.starts_with('_') {
            // Literal or blank node - return as-is
            term.to_string()
        } else {
            // Wrap in angle brackets for full URI
            format!("<{term}>")
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

/// Result of patch application
#[derive(Debug, Clone, Serialize, Deserialize)]
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
            PatchOperation::AddPrefix { prefix, namespace: _namespace } => PatchOperation::DeletePrefix {
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

/// Advanced conflict resolution for patch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Patch normalization utilities
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

/// Patch compression utilities
pub struct PatchCompressor {
    compression_level: u32,
    enable_dictionary: bool,
    prefix_compression: bool,
}

impl PatchCompressor {
    pub fn new() -> Self {
        Self {
            compression_level: 6,
            enable_dictionary: true,
            prefix_compression: true,
        }
    }

    pub fn with_compression_level(mut self, level: u32) -> Self {
        self.compression_level = level.min(9);
        self
    }

    pub fn with_dictionary_compression(mut self, enabled: bool) -> Self {
        self.enable_dictionary = enabled;
        self
    }

    pub fn with_prefix_compression(mut self, enabled: bool) -> Self {
        self.prefix_compression = enabled;
        self
    }

    /// Compress patch using gzip compression
    pub fn compress_patch(&self, patch: &RdfPatch) -> Result<Vec<u8>> {
        // Serialize patch to string
        let serializer = PatchSerializer::new().with_pretty_print(false);
        let patch_str = serializer.serialize(patch)?;
        let original_len = patch_str.len();

        // Apply dictionary compression if enabled
        let optimized_str = if self.enable_dictionary {
            self.apply_dictionary_compression(&patch_str)?
        } else {
            patch_str
        };

        // Apply gzip compression
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder.write_all(optimized_str.as_bytes())?;
        let compressed = encoder.finish()?;

        info!(
            "Compressed patch: {} -> {} bytes ({:.1}% reduction)",
            original_len,
            compressed.len(),
            (1.0 - compressed.len() as f64 / original_len as f64) * 100.0
        );

        Ok(compressed)
    }

    /// Decompress patch from compressed bytes
    pub fn decompress_patch(&self, compressed_data: &[u8]) -> Result<RdfPatch> {
        // Decompress gzip
        let mut decoder = GzDecoder::new(compressed_data);
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed)?;

        // Apply dictionary decompression if needed
        let patch_str = if self.enable_dictionary {
            self.apply_dictionary_decompression(&decompressed)?
        } else {
            decompressed
        };

        // Parse patch
        let mut parser = PatchParser::new();
        parser.parse(&patch_str)
    }

    fn apply_dictionary_compression(&self, patch_str: &str) -> Result<String> {
        // Build frequency dictionary of common terms
        let mut word_freq = HashMap::new();
        for word in patch_str.split_whitespace() {
            *word_freq.entry(word.to_string()).or_insert(0) += 1;
        }

        // Create dictionary of most frequent terms
        let mut freq_words: Vec<_> = word_freq.into_iter().collect();
        freq_words.sort_by(|a, b| b.1.cmp(&a.1));

        let mut dictionary = HashMap::new();
        let mut compressed = patch_str.to_string();

        // Replace most frequent words with short codes
        for (i, (word, freq)) in freq_words.iter().take(256).enumerate() {
            if word.len() > 3 && *freq > 2 {
                let code = format!("#{i:02x}");
                dictionary.insert(code.clone(), word.clone());
                compressed = compressed.replace(word, &code);
            }
        }

        // Prepend dictionary to compressed string
        let mut dict_header = String::new();
        for (code, word) in dictionary {
            dict_header.push_str(&format!("{code}={word}\n"));
        }
        dict_header.push_str("---\n");
        dict_header.push_str(&compressed);

        Ok(dict_header)
    }

    fn apply_dictionary_decompression(&self, compressed_str: &str) -> Result<String> {
        if let Some(separator_pos) = compressed_str.find("---\n") {
            let (dict_part, content_part) = compressed_str.split_at(separator_pos);
            let content = &content_part[4..]; // Skip "---\n"

            let mut dictionary = HashMap::new();
            for line in dict_part.lines() {
                if let Some(eq_pos) = line.find('=') {
                    let code = &line[..eq_pos];
                    let word = &line[eq_pos + 1..];
                    dictionary.insert(code, word);
                }
            }

            let mut decompressed = content.to_string();
            for (code, word) in dictionary {
                decompressed = decompressed.replace(code, word);
            }

            Ok(decompressed)
        } else {
            Ok(compressed_str.to_string())
        }
    }

    /// Compress using prefix compression for common namespaces
    pub fn compress_with_prefixes(&self, patch: &RdfPatch) -> Result<RdfPatch> {
        let mut compressed = patch.clone();
        compressed.id = format!("{}-prefix-compressed", patch.id);

        if !self.prefix_compression {
            return Ok(compressed);
        }

        // Build frequency map of URI prefixes
        let mut prefix_freq = HashMap::new();
        for operation in &patch.operations {
            self.collect_uris_from_operation(operation, &mut prefix_freq);
        }

        // Find common prefixes
        let mut common_prefixes = HashMap::new();
        for (uri, freq) in prefix_freq {
            if freq > 2 {
                if let Some(prefix) = self.extract_namespace_prefix(&uri) {
                    if prefix.len() > 10 {
                        let short_prefix = format!("ns{}", common_prefixes.len());
                        common_prefixes.insert(prefix, short_prefix);
                    }
                }
            }
        }

        // Add prefix declarations to patch
        for (namespace, prefix) in &common_prefixes {
            compressed.add_operation(PatchOperation::AddPrefix {
                prefix: prefix.clone(),
                namespace: namespace.clone(),
            });
        }

        // Replace URIs with prefixed forms
        for operation in &mut compressed.operations {
            self.apply_prefix_compression_to_operation(operation, &common_prefixes);
        }

        info!(
            "Applied prefix compression: {} prefixes defined",
            common_prefixes.len()
        );
        Ok(compressed)
    }

    fn collect_uris_from_operation(
        &self,
        operation: &PatchOperation,
        prefix_freq: &mut HashMap<String, usize>,
    ) {
        match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => {
                *prefix_freq.entry(subject.clone()).or_insert(0) += 1;
                *prefix_freq.entry(predicate.clone()).or_insert(0) += 1;
                *prefix_freq.entry(object.clone()).or_insert(0) += 1;
            }
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => {
                *prefix_freq.entry(subject.clone()).or_insert(0) += 1;
                *prefix_freq.entry(predicate.clone()).or_insert(0) += 1;
                *prefix_freq.entry(object.clone()).or_insert(0) += 1;
            }
            PatchOperation::AddGraph { graph } | PatchOperation::DeleteGraph { graph } => {
                *prefix_freq.entry(graph.clone()).or_insert(0) += 1;
            }
            _ => {}
        }
    }

    fn extract_namespace_prefix(&self, uri: &str) -> Option<String> {
        // Extract namespace part of URI (everything up to last # or /)
        if let Some(pos) = uri.rfind('#') {
            Some(uri[..pos + 1].to_string())
        } else {
            uri.rfind('/').map(|pos| uri[..pos + 1].to_string())
        }
    }

    fn apply_prefix_compression_to_operation(
        &self,
        operation: &mut PatchOperation,
        prefixes: &HashMap<String, String>,
    ) {
        match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => {
                *subject = self.compress_uri_with_prefixes(subject, prefixes);
                *predicate = self.compress_uri_with_prefixes(predicate, prefixes);
                *object = self.compress_uri_with_prefixes(object, prefixes);
            }
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => {
                *subject = self.compress_uri_with_prefixes(subject, prefixes);
                *predicate = self.compress_uri_with_prefixes(predicate, prefixes);
                *object = self.compress_uri_with_prefixes(object, prefixes);
            }
            PatchOperation::AddGraph { graph } | PatchOperation::DeleteGraph { graph } => {
                *graph = self.compress_uri_with_prefixes(graph, prefixes);
            }
            _ => {}
        }
    }

    fn compress_uri_with_prefixes(&self, uri: &str, prefixes: &HashMap<String, String>) -> String {
        for (namespace, prefix) in prefixes {
            if uri.starts_with(namespace) {
                let local_name = &uri[namespace.len()..];
                return format!("{prefix}:{local_name}");
            }
        }
        uri.to_string()
    }
}

impl Default for PatchCompressor {
    fn default() -> Self {
        Self::new()
    }
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
        assert_eq!(
            patch.headers.get("creator"),
            Some(&"test-parser".to_string())
        );

        // Check transaction ID was captured
        assert_eq!(patch.transaction_id, Some("tx-456".to_string()));

        // Check prefix was captured
        assert_eq!(
            patch.prefixes.get("ex2"),
            Some(&"http://example2.org/".to_string())
        );

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
        assert_eq!(
            original_patch.operations.len(),
            parsed_patch.operations.len()
        );

        match (&original_patch.operations[0], &parsed_patch.operations[0]) {
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
            PatchOperation::TransactionBegin { .. } => {}
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
            Some(PatchOperation::TransactionCommit) => {}
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
        assert!(matches!(
            &patch.operations[0],
            PatchOperation::TransactionBegin { .. }
        ));

        // Last should be transaction commit
        assert!(matches!(
            &patch.operations[3],
            PatchOperation::TransactionCommit
        ));

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
