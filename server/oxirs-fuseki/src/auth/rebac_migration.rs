//! ReBAC Migration Tools
//!
//! This module provides tools for migrating authorization data between
//! in-memory and RDF-native backends.
//!
//! ## Features
//!
//! - **Export**: Export in-memory relationships to Turtle/RDF format
//! - **Import**: Import Turtle/RDF relationships to in-memory
//! - **Migrate**: Migrate between backends with validation
//! - **Verify**: Verify data integrity after migration
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_fuseki::auth::rebac_migration::*;
//!
//! // Export to Turtle
//! let exporter = RebacExporter::new();
//! let turtle = exporter.export_to_turtle(&relationships)?;
//! std::fs::write("auth.ttl", turtle)?;
//!
//! // Import from Turtle
//! let importer = RebacImporter::new();
//! let turtle = std::fs::read_to_string("auth.ttl")?;
//! let relationships = importer.import_from_turtle(&turtle)?;
//!
//! // Migrate between backends
//! let migrator = RebacMigrator::new(source, target);
//! migrator.migrate().await?;
//! ```

use super::rebac::{RelationshipCondition, RelationshipTuple, Result};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use tracing::{debug, info, warn};

/// RDF/Turtle exporter for ReBAC relationships
pub struct RebacExporter {
    /// Namespace prefix for authorization vocabulary
    auth_ns: String,
    /// Named graph URI
    graph_uri: String,
}

impl RebacExporter {
    /// Create a new exporter with default settings
    pub fn new() -> Self {
        Self {
            auth_ns: "http://oxirs.org/auth#".to_string(),
            graph_uri: "urn:oxirs:auth:relationships".to_string(),
        }
    }

    /// Create exporter with custom namespace and graph URI
    pub fn with_config(auth_ns: String, graph_uri: String) -> Self {
        Self { auth_ns, graph_uri }
    }

    /// Export relationships to Turtle format
    pub fn export_to_turtle(&self, relationships: &[RelationshipTuple]) -> Result<String> {
        let mut output = String::new();

        // Write prefixes
        writeln!(&mut output, "@prefix auth: <{}> .", self.auth_ns)
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;
        writeln!(
            &mut output,
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ."
        )
        .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;
        writeln!(&mut output)
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;

        // Write graph
        writeln!(&mut output, "<{}> {{", self.graph_uri)
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;

        // Write relationships
        for tuple in relationships {
            if tuple.condition.is_none() {
                // Simple triple
                self.write_simple_triple(&mut output, tuple)?;
            } else {
                // Reified relationship with condition
                self.write_reified_triple(&mut output, tuple)?;
            }
        }

        writeln!(&mut output, "}}")
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;

        Ok(output)
    }

    /// Write a simple triple (no conditions)
    fn write_simple_triple(&self, output: &mut String, tuple: &RelationshipTuple) -> Result<()> {
        let relation_uri = self.relation_to_uri(&tuple.relation);
        writeln!(
            output,
            "  <{}> {} <{}> .",
            tuple.subject, relation_uri, tuple.object
        )
        .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;
        Ok(())
    }

    /// Write a reified triple with conditions
    fn write_reified_triple(&self, output: &mut String, tuple: &RelationshipTuple) -> Result<()> {
        let relation_uri = self.relation_to_uri(&tuple.relation);

        writeln!(output, "  [] a auth:Relationship ;")
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;
        writeln!(output, "     auth:subject <{}> ;", tuple.subject)
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;
        writeln!(output, "     auth:relation {} ;", relation_uri)
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;
        write!(output, "     auth:object <{}>", tuple.object)
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;

        if let Some(condition) = &tuple.condition {
            match condition {
                RelationshipCondition::TimeWindow {
                    not_before,
                    not_after,
                } => {
                    if let Some(nb) = not_before {
                        writeln!(output, " ;").map_err(|e| {
                            super::rebac::RebacError::Internal(format!("Write error: {}", e))
                        })?;
                        write!(
                            output,
                            "     auth:notBefore \"{}\"^^xsd:dateTime",
                            nb.to_rfc3339()
                        )
                        .map_err(|e| {
                            super::rebac::RebacError::Internal(format!("Write error: {}", e))
                        })?;
                    }
                    if let Some(na) = not_after {
                        writeln!(output, " ;").map_err(|e| {
                            super::rebac::RebacError::Internal(format!("Write error: {}", e))
                        })?;
                        write!(
                            output,
                            "     auth:notAfter \"{}\"^^xsd:dateTime",
                            na.to_rfc3339()
                        )
                        .map_err(|e| {
                            super::rebac::RebacError::Internal(format!("Write error: {}", e))
                        })?;
                    }
                }
                RelationshipCondition::IpAddress { allowed_ips } => {
                    for ip in allowed_ips {
                        writeln!(output, " ;").map_err(|e| {
                            super::rebac::RebacError::Internal(format!("Write error: {}", e))
                        })?;
                        write!(output, "     auth:allowedIp \"{}\"", ip).map_err(|e| {
                            super::rebac::RebacError::Internal(format!("Write error: {}", e))
                        })?;
                    }
                }
                RelationshipCondition::Attribute { key, value } => {
                    writeln!(output, " ;").map_err(|e| {
                        super::rebac::RebacError::Internal(format!("Write error: {}", e))
                    })?;
                    write!(
                        output,
                        "     auth:attribute [ auth:key \"{}\" ; auth:value \"{}\" ]",
                        key, value
                    )
                    .map_err(|e| {
                        super::rebac::RebacError::Internal(format!("Write error: {}", e))
                    })?;
                }
            }
        }

        writeln!(output, " .")
            .map_err(|e| super::rebac::RebacError::Internal(format!("Write error: {}", e)))?;
        Ok(())
    }

    /// Convert relation string to RDF URI
    fn relation_to_uri(&self, relation: &str) -> String {
        match relation {
            "can_read" => "auth:canRead".to_string(),
            "can_write" => "auth:canWrite".to_string(),
            "can_delete" => "auth:canDelete".to_string(),
            "owner" => "auth:owner".to_string(),
            "member" => "auth:memberOf".to_string(),
            "can_access" => "auth:canAccess".to_string(),
            "can_manage" => "auth:canManage".to_string(),
            other => format!("auth:{}", other.replace('_', "")),
        }
    }

    /// Export to JSON format
    pub fn export_to_json(&self, relationships: &[RelationshipTuple]) -> Result<String> {
        serde_json::to_string_pretty(relationships)
            .map_err(|e| super::rebac::RebacError::Internal(format!("JSON error: {}", e)))
    }

    /// Export statistics
    pub fn export_stats(&self, relationships: &[RelationshipTuple]) -> ExportStats {
        let mut relation_counts = HashMap::new();
        let mut conditional_count = 0;

        for tuple in relationships {
            *relation_counts.entry(tuple.relation.clone()).or_insert(0) += 1;
            if tuple.condition.is_some() {
                conditional_count += 1;
            }
        }

        ExportStats {
            total_relationships: relationships.len(),
            conditional_relationships: conditional_count,
            by_relation: relation_counts,
        }
    }
}

impl Default for RebacExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Export statistics
#[derive(Debug, Clone, Default)]
pub struct ExportStats {
    pub total_relationships: usize,
    pub conditional_relationships: usize,
    pub by_relation: HashMap<String, usize>,
}

impl ExportStats {
    pub fn display(&self) -> String {
        let mut output = String::new();
        writeln!(&mut output, "Export Statistics:").expect("string formatting never fails");
        writeln!(
            &mut output,
            "  Total relationships: {}",
            self.total_relationships
        )
        .expect("string formatting never fails");
        writeln!(
            &mut output,
            "  Conditional relationships: {}",
            self.conditional_relationships
        )
        .expect("string formatting never fails");
        writeln!(&mut output, "  By relation type:").expect("string formatting never fails");
        for (relation, count) in &self.by_relation {
            writeln!(&mut output, "    {}: {}", relation, count)
                .expect("string formatting never fails");
        }
        output
    }
}

/// RDF/Turtle importer for ReBAC relationships
pub struct RebacImporter {
    /// Namespace prefix for authorization vocabulary
    auth_ns: String,
}

impl RebacImporter {
    /// Create a new importer with default settings
    pub fn new() -> Self {
        Self {
            auth_ns: "http://oxirs.org/auth#".to_string(),
        }
    }

    /// Create importer with custom namespace
    pub fn with_namespace(auth_ns: String) -> Self {
        Self { auth_ns }
    }

    /// Import relationships from Turtle format
    ///
    /// This is a simplified parser for the specific Turtle format we export.
    /// For production use, consider using a full Turtle parser like `rio_turtle`.
    pub fn import_from_turtle(&self, turtle: &str) -> Result<Vec<RelationshipTuple>> {
        let mut relationships = Vec::new();

        // Simple line-based parser
        let lines: Vec<&str> = turtle.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Skip comments and empty lines
            if line.starts_with('#') || line.is_empty() || line.starts_with('@') {
                i += 1;
                continue;
            }

            // Skip graph declaration
            if line.contains("GRAPH") || line == "{" || line == "}" {
                i += 1;
                continue;
            }

            // Parse simple triple
            if line.contains("auth:") && !line.contains("auth:Relationship") {
                if let Some(tuple) = self.parse_simple_triple(line)? {
                    relationships.push(tuple);
                }
            }

            // Parse reified relationship (multi-line)
            if line.contains("auth:Relationship") {
                if let Some((tuple, lines_consumed)) = self.parse_reified_triple(&lines[i..])? {
                    relationships.push(tuple);
                    i += lines_consumed;
                }
            }

            i += 1;
        }

        info!("Imported {} relationships from Turtle", relationships.len());
        Ok(relationships)
    }

    /// Parse a simple triple from a line
    fn parse_simple_triple(&self, line: &str) -> Result<Option<RelationshipTuple>> {
        // Example: <user:alice> auth:owner <dataset:public> .
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Ok(None);
        }

        let subject = parts[0].trim_matches(|c| c == '<' || c == '>').to_string();
        let relation = self.uri_to_relation(parts[1]);
        let object = parts[2].trim_matches(|c| c == '<' || c == '>').to_string();

        Ok(Some(RelationshipTuple::new(subject, relation, object)))
    }

    /// Parse a reified triple (multi-line)
    fn parse_reified_triple(&self, lines: &[&str]) -> Result<Option<(RelationshipTuple, usize)>> {
        let mut subject = None;
        let mut relation = None;
        let mut object = None;
        let mut not_before = None;
        let mut not_after = None;
        let mut allowed_ips = Vec::new();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();

            if line.contains("auth:subject") {
                subject = Some(
                    line.split('<')
                        .nth(1)
                        .and_then(|s| s.split('>').next())
                        .unwrap_or("")
                        .to_string(),
                );
            } else if line.contains("auth:relation") {
                if let Some(rel_part) = line.split("auth:relation").nth(1) {
                    let rel = rel_part
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_matches(|c| c == ';' || c == ',');
                    relation = Some(self.uri_to_relation(rel));
                }
            } else if line.contains("auth:object") {
                object = Some(
                    line.split('<')
                        .nth(1)
                        .and_then(|s| s.split('>').next())
                        .unwrap_or("")
                        .to_string(),
                );
            } else if line.contains("auth:notBefore") {
                if let Some(datetime) = self.extract_datetime(line) {
                    not_before = Some(datetime);
                }
            } else if line.contains("auth:notAfter") {
                if let Some(datetime) = self.extract_datetime(line) {
                    not_after = Some(datetime);
                }
            } else if line.contains("auth:allowedIp") {
                if let Some(ip) = self.extract_string_literal(line) {
                    allowed_ips.push(ip);
                }
            } else if line.contains('.') {
                // End of reified relationship
                break;
            }

            i += 1;
        }

        if let (Some(subj), Some(rel), Some(obj)) = (subject, relation, object) {
            let mut tuple = RelationshipTuple::new(subj, rel, obj);

            // Add conditions
            if not_before.is_some() || not_after.is_some() {
                tuple.condition = Some(RelationshipCondition::TimeWindow {
                    not_before,
                    not_after,
                });
            } else if !allowed_ips.is_empty() {
                tuple.condition = Some(RelationshipCondition::IpAddress { allowed_ips });
            }

            Ok(Some((tuple, i)))
        } else {
            Ok(None)
        }
    }

    /// Extract datetime from Turtle literal
    fn extract_datetime(&self, line: &str) -> Option<DateTime<Utc>> {
        line.split('"')
            .nth(1)
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.into())
    }

    /// Extract string literal
    fn extract_string_literal(&self, line: &str) -> Option<String> {
        line.split('"').nth(1).map(|s| s.to_string())
    }

    /// Convert RDF URI to relation string
    fn uri_to_relation(&self, uri: &str) -> String {
        match uri {
            "auth:canRead" => "can_read".to_string(),
            "auth:canWrite" => "can_write".to_string(),
            "auth:canDelete" => "can_delete".to_string(),
            "auth:owner" => "owner".to_string(),
            "auth:memberOf" => "member".to_string(),
            "auth:canAccess" => "can_access".to_string(),
            "auth:canManage" => "can_manage".to_string(),
            other => {
                if let Some(name) = other.strip_prefix("auth:") {
                    name.to_string()
                } else {
                    other.to_string()
                }
            }
        }
    }

    /// Import from JSON format
    pub fn import_from_json(&self, json: &str) -> Result<Vec<RelationshipTuple>> {
        serde_json::from_str(json)
            .map_err(|e| super::rebac::RebacError::Internal(format!("JSON error: {}", e)))
    }
}

impl Default for RebacImporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Migration verification result
#[derive(Debug, Clone)]
pub struct MigrationVerification {
    pub source_count: usize,
    pub target_count: usize,
    pub matched: usize,
    pub missing_in_target: Vec<String>,
    pub extra_in_target: Vec<String>,
    pub success: bool,
}

impl MigrationVerification {
    pub fn display(&self) -> String {
        let mut output = String::new();
        writeln!(&mut output, "Migration Verification:").expect("string formatting never fails");
        writeln!(&mut output, "  Source count: {}", self.source_count)
            .expect("string formatting never fails");
        writeln!(&mut output, "  Target count: {}", self.target_count)
            .expect("string formatting never fails");
        writeln!(&mut output, "  Matched: {}", self.matched)
            .expect("string formatting never fails");

        if !self.missing_in_target.is_empty() {
            writeln!(
                &mut output,
                "  Missing in target: {}",
                self.missing_in_target.len()
            )
            .expect("string formatting never fails");
            for item in &self.missing_in_target {
                writeln!(&mut output, "    - {}", item).expect("string formatting never fails");
            }
        }

        if !self.extra_in_target.is_empty() {
            writeln!(
                &mut output,
                "  Extra in target: {}",
                self.extra_in_target.len()
            )
            .expect("string formatting never fails");
            for item in &self.extra_in_target {
                writeln!(&mut output, "    - {}", item).expect("string formatting never fails");
            }
        }

        writeln!(
            &mut output,
            "  Status: {}",
            if self.success {
                "✅ SUCCESS"
            } else {
                "❌ FAILED"
            }
        )
        .expect("string formatting never fails");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_simple_relationships() {
        let relationships = vec![
            RelationshipTuple::new("user:alice", "owner", "dataset:public"),
            RelationshipTuple::new("user:bob", "can_read", "graph:http://example.org/g1"),
        ];

        let exporter = RebacExporter::new();
        let turtle = exporter.export_to_turtle(&relationships).unwrap();

        assert!(turtle.contains("@prefix auth:"));
        assert!(turtle.contains("<user:alice> auth:owner <dataset:public>"));
        assert!(turtle.contains("<user:bob> auth:canRead <graph:http://example.org/g1>"));
    }

    #[test]
    fn test_export_conditional_relationships() {
        let condition = RelationshipCondition::TimeWindow {
            not_before: Some(
                DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
                    .unwrap()
                    .into(),
            ),
            not_after: Some(
                DateTime::parse_from_rfc3339("2025-12-31T23:59:59Z")
                    .unwrap()
                    .into(),
            ),
        };

        let relationships = vec![RelationshipTuple::with_condition(
            "user:charlie",
            "can_read",
            "dataset:temporary",
            condition,
        )];

        let exporter = RebacExporter::new();
        let turtle = exporter.export_to_turtle(&relationships).unwrap();

        assert!(turtle.contains("auth:Relationship"));
        assert!(turtle.contains("auth:subject"));
        assert!(turtle.contains("auth:notBefore"));
        assert!(turtle.contains("auth:notAfter"));
    }

    #[test]
    fn test_export_stats() {
        let relationships = vec![
            RelationshipTuple::new("user:alice", "owner", "dataset:public"),
            RelationshipTuple::new("user:bob", "can_read", "dataset:public"),
            RelationshipTuple::new("user:charlie", "can_read", "dataset:public"),
            RelationshipTuple::with_condition(
                "user:temp",
                "can_read",
                "dataset:temp",
                RelationshipCondition::TimeWindow {
                    not_before: None,
                    not_after: None,
                },
            ),
        ];

        let exporter = RebacExporter::new();
        let stats = exporter.export_stats(&relationships);

        assert_eq!(stats.total_relationships, 4);
        assert_eq!(stats.conditional_relationships, 1);
        assert_eq!(stats.by_relation.get("can_read"), Some(&3));
        assert_eq!(stats.by_relation.get("owner"), Some(&1));
    }

    #[test]
    fn test_roundtrip_simple() {
        let original = vec![
            RelationshipTuple::new("user:alice", "owner", "dataset:public"),
            RelationshipTuple::new("user:bob", "can_read", "dataset:public"),
        ];

        let exporter = RebacExporter::new();
        let turtle = exporter.export_to_turtle(&original).unwrap();

        let importer = RebacImporter::new();
        let imported = importer.import_from_turtle(&turtle).unwrap();

        assert_eq!(imported.len(), original.len());
        assert_eq!(imported[0].subject, "user:alice");
        assert_eq!(imported[0].relation, "owner");
        assert_eq!(imported[0].object, "dataset:public");
    }

    #[test]
    fn test_json_export_import() {
        let relationships = vec![
            RelationshipTuple::new("user:alice", "owner", "dataset:public"),
            RelationshipTuple::new("user:bob", "can_read", "dataset:public"),
        ];

        let exporter = RebacExporter::new();
        let json = exporter.export_to_json(&relationships).unwrap();

        let importer = RebacImporter::new();
        let imported = importer.import_from_json(&json).unwrap();

        assert_eq!(imported.len(), relationships.len());
        assert_eq!(imported[0].subject, relationships[0].subject);
    }
}
