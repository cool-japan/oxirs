//! RDF Patch serializer

use crate::{PatchOperation, RdfPatch};
use anyhow::Result;
use std::collections::HashMap;

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
