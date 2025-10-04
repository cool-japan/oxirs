//! RDF Patch parser

use crate::{PatchOperation, RdfPatch};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{debug, warn};

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
