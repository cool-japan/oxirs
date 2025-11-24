//! ReBAC Manager - Core relationship management for authorization
//!
//! This module provides the core functionality for managing ReBAC (Relationship-Based Access Control)
//! relationships using RDF as the underlying storage format.

use crate::cli::error::{CliError, CliResult};
use oxirs_core::model::{BlankNode, Literal, NamedNode, Quad, Subject, Term};
use oxirs_core::store::Store;
use scirs2_core::random::{rng, Random};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tracing::{debug, info, warn};

/// Authorization namespace for ReBAC relationships
pub const AUTH_NAMESPACE: &str = "http://oxirs.org/auth#";

/// Default graph URI for ReBAC relationships
pub const DEFAULT_GRAPH: &str = "urn:oxirs:auth:relationships";

/// ReBAC relationship tuple (subject, relation, object)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RelationshipTuple {
    /// Subject (e.g., "user:alice")
    pub subject: String,
    /// Relation (e.g., "owner", "can_read", "can_write")
    pub relation: String,
    /// Object (e.g., "dataset:public", "graph:http://example.org/g1")
    pub object: String,
    /// Optional condition (e.g., temporal constraints, context)
    pub condition: Option<String>,
}

impl RelationshipTuple {
    /// Create a new relationship tuple
    pub fn new(
        subject: impl Into<String>,
        relation: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            relation: relation.into(),
            object: object.into(),
            condition: None,
        }
    }

    /// Create a new conditional relationship tuple
    pub fn with_condition(
        subject: impl Into<String>,
        relation: impl Into<String>,
        object: impl Into<String>,
        condition: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            relation: relation.into(),
            object: object.into(),
            condition: Some(condition.into()),
        }
    }

    /// Convert to RDF quad
    pub fn to_quad(&self, namespace: &str, graph_uri: &str) -> CliResult<Quad> {
        let subject = self.parse_subject(&self.subject)?;
        let predicate = NamedNode::new(format!("{}{}", namespace, self.relation))
            .map_err(|e| CliError::execution_error(format!("Invalid relation: {}", e)))?;
        let object = self.parse_term(&self.object)?;
        let graph = NamedNode::new(graph_uri)
            .map_err(|e| CliError::execution_error(format!("Invalid graph URI: {}", e)))?;

        Ok(Quad::new(subject, predicate, object, graph))
    }

    /// Parse subject string to RDF Subject
    fn parse_subject(&self, s: &str) -> CliResult<Subject> {
        if s.starts_with("_:") {
            Ok(Subject::BlankNode(BlankNode::new(
                s.trim_start_matches("_:"),
            )?))
        } else if s.contains(':') {
            Ok(Subject::NamedNode(NamedNode::new(s).map_err(|e| {
                CliError::execution_error(format!("Invalid subject: {}", e))
            })?))
        } else {
            // Assume user/dataset/graph prefix
            Ok(Subject::NamedNode(
                NamedNode::new(format!("urn:oxirs:auth:{}", s))
                    .map_err(|e| CliError::execution_error(format!("Invalid subject: {}", e)))?,
            ))
        }
    }

    /// Parse term string to RDF Term
    fn parse_term(&self, s: &str) -> CliResult<Term> {
        if s.starts_with("_:") {
            Ok(Term::BlankNode(BlankNode::new(s.trim_start_matches("_:"))?))
        } else if s.starts_with('"') {
            // Literal value
            Ok(Term::Literal(Literal::new_simple_literal(
                s.trim_matches('"'),
            )))
        } else if s.contains(':') {
            Ok(Term::NamedNode(NamedNode::new(s).map_err(|e| {
                CliError::execution_error(format!("Invalid object: {}", e))
            })?))
        } else {
            // Assume resource prefix
            Ok(Term::NamedNode(
                NamedNode::new(format!("urn:oxirs:auth:{}", s))
                    .map_err(|e| CliError::execution_error(format!("Invalid object: {}", e)))?,
            ))
        }
    }

    /// Create from RDF quad
    pub fn from_quad(quad: &Quad, namespace: &str) -> Option<Self> {
        let subject = match quad.subject {
            Subject::NamedNode(ref n) => n.as_str().to_string(),
            Subject::BlankNode(ref b) => format!("_:{}", b.as_str()),
            _ => return None,
        };

        let predicate = quad.predicate.as_str();
        let relation = if let Some(stripped) = predicate.strip_prefix(namespace) {
            stripped.to_string()
        } else {
            predicate.to_string()
        };

        let object = match &quad.object {
            Term::NamedNode(n) => n.as_str().to_string(),
            Term::BlankNode(b) => format!("_:{}", b.as_str()),
            Term::Literal(l) => format!("\"{}\"", l.value()),
        };

        Some(Self {
            subject,
            relation,
            object,
            condition: None,
        })
    }
}

/// Backend storage type for ReBAC
#[derive(Debug, Clone, Copy)]
pub enum StorageBackend {
    /// In-memory storage (fast, non-persistent)
    InMemory,
    /// RDF-native storage using oxirs-tdb (persistent)
    RdfNative,
}

/// ReBAC Manager for managing authorization relationships
pub struct RebacManager {
    /// RDF store for relationship storage
    store: Store,
    /// Authorization namespace
    namespace: String,
    /// Default graph URI
    graph_uri: String,
    /// Backend type
    backend: StorageBackend,
}

impl RebacManager {
    /// Create a new in-memory ReBAC manager
    pub fn new_in_memory() -> CliResult<Self> {
        let store = Store::new()
            .map_err(|e| CliError::execution_error(format!("Failed to create store: {}", e)))?;
        Ok(Self {
            store,
            namespace: AUTH_NAMESPACE.to_string(),
            graph_uri: DEFAULT_GRAPH.to_string(),
            backend: StorageBackend::InMemory,
        })
    }

    /// Create a new persistent ReBAC manager with RDF-native storage
    pub fn new_persistent(path: &Path) -> CliResult<Self> {
        let store = Store::open(path)
            .map_err(|e| CliError::execution_error(format!("Failed to open store: {}", e)))?;
        Ok(Self {
            store,
            namespace: AUTH_NAMESPACE.to_string(),
            graph_uri: DEFAULT_GRAPH.to_string(),
            backend: StorageBackend::RdfNative,
        })
    }

    /// Set custom namespace
    pub fn with_namespace(mut self, namespace: String) -> Self {
        self.namespace = namespace;
        self
    }

    /// Set custom graph URI
    pub fn with_graph(mut self, graph_uri: String) -> Self {
        self.graph_uri = graph_uri;
        self
    }

    /// Get backend type
    pub fn backend(&self) -> StorageBackend {
        self.backend
    }

    /// Add a relationship tuple
    pub fn add_relationship(&mut self, tuple: &RelationshipTuple) -> CliResult<()> {
        let quad = tuple.to_quad(&self.namespace, &self.graph_uri)?;
        self.store.insert(&quad).map_err(|e| {
            CliError::execution_error(format!("Failed to insert relationship: {}", e))
        })?;
        debug!("Added relationship: {:?}", tuple);
        Ok(())
    }

    /// Add multiple relationships in batch
    pub fn add_relationships(&mut self, tuples: &[RelationshipTuple]) -> CliResult<usize> {
        let mut count = 0;
        for tuple in tuples {
            self.add_relationship(tuple)?;
            count += 1;
        }
        info!("Added {} relationships", count);
        Ok(count)
    }

    /// Remove a relationship tuple
    pub fn remove_relationship(&mut self, tuple: &RelationshipTuple) -> CliResult<bool> {
        let quad = tuple.to_quad(&self.namespace, &self.graph_uri)?;
        let removed = self.store.remove(&quad).map_err(|e| {
            CliError::execution_error(format!("Failed to remove relationship: {}", e))
        })?;
        if removed {
            debug!("Removed relationship: {:?}", tuple);
        }
        Ok(removed)
    }

    /// Check if a relationship exists
    pub fn has_relationship(&self, tuple: &RelationshipTuple) -> CliResult<bool> {
        let quad = tuple.to_quad(&self.namespace, &self.graph_uri)?;
        Ok(self.store.contains(&quad))
    }

    /// Query relationships with optional filters
    pub fn query_relationships(
        &self,
        subject: Option<&str>,
        relation: Option<&str>,
        object: Option<&str>,
    ) -> CliResult<Vec<RelationshipTuple>> {
        let graph = NamedNode::new(&self.graph_uri)
            .map_err(|e| CliError::execution_error(format!("Invalid graph URI: {}", e)))?;

        // Query all quads in the ReBAC graph
        let quads = self.store.quads_for_graph(&graph);

        let mut results = Vec::new();
        for quad in quads {
            if let Some(tuple) = RelationshipTuple::from_quad(&quad, &self.namespace) {
                // Apply filters
                if let Some(s) = subject {
                    if !tuple.subject.contains(s) {
                        continue;
                    }
                }
                if let Some(r) = relation {
                    if !tuple.relation.contains(r) {
                        continue;
                    }
                }
                if let Some(o) = object {
                    if !tuple.object.contains(o) {
                        continue;
                    }
                }
                results.push(tuple);
            }
        }

        Ok(results)
    }

    /// Get all relationships
    pub fn get_all_relationships(&self) -> CliResult<Vec<RelationshipTuple>> {
        self.query_relationships(None, None, None)
    }

    /// Get statistics about stored relationships
    pub fn get_statistics(&self) -> CliResult<RebacStatistics> {
        let all_tuples = self.get_all_relationships()?;

        let mut stats = RebacStatistics::default();
        stats.total_relationships = all_tuples.len();

        // Count by relation type
        for tuple in &all_tuples {
            *stats.by_relation.entry(tuple.relation.clone()).or_insert(0) += 1;
            *stats.by_subject.entry(tuple.subject.clone()).or_insert(0) += 1;
            *stats.by_object.entry(tuple.object.clone()).or_insert(0) += 1;

            if tuple.condition.is_some() {
                stats.conditional_relationships += 1;
            }
        }

        Ok(stats)
    }

    /// Check for duplicate relationships
    pub fn find_duplicates(&self) -> CliResult<Vec<RelationshipTuple>> {
        let all_tuples = self.get_all_relationships()?;
        let mut seen = HashSet::new();
        let mut duplicates = Vec::new();

        for tuple in all_tuples {
            let key = (
                tuple.subject.clone(),
                tuple.relation.clone(),
                tuple.object.clone(),
            );
            if !seen.insert(key) {
                duplicates.push(tuple);
            }
        }

        Ok(duplicates)
    }

    /// Check for orphaned relationships (subjects or objects that don't exist as entities)
    pub fn find_orphans(&self) -> CliResult<Vec<RelationshipTuple>> {
        let all_tuples = self.get_all_relationships()?;

        // Collect all subjects and objects
        let mut entities = HashSet::new();
        for tuple in &all_tuples {
            entities.insert(tuple.subject.clone());
            entities.insert(tuple.object.clone());
        }

        // Find orphans (relationships pointing to non-existent entities)
        let mut orphans = Vec::new();
        for tuple in all_tuples {
            // Check if object exists as a subject somewhere
            let object_exists = entities.contains(&tuple.object);
            if !object_exists
                && !tuple.object.starts_with("dataset:")
                && !tuple.object.starts_with("graph:")
            {
                orphans.push(tuple);
            }
        }

        Ok(orphans)
    }

    /// Clear all relationships
    pub fn clear_all(&mut self) -> CliResult<usize> {
        let all_tuples = self.get_all_relationships()?;
        let count = all_tuples.len();

        for tuple in all_tuples {
            self.remove_relationship(&tuple)?;
        }

        info!("Cleared {} relationships", count);
        Ok(count)
    }

    /// Export relationships to Turtle format
    pub fn export_turtle(&self) -> CliResult<String> {
        let tuples = self.get_all_relationships()?;
        let mut turtle = String::new();

        // Prefixes
        turtle.push_str("@prefix auth: <http://oxirs.org/auth#> .\n");
        turtle.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

        // Named graph
        turtle.push_str(&format!("<{}> {{\n", self.graph_uri));

        // Relationships
        for tuple in tuples {
            turtle.push_str(&format!(
                "  <{}> auth:{} <{}> .\n",
                tuple.subject, tuple.relation, tuple.object
            ));
        }

        turtle.push_str("}\n");

        Ok(turtle)
    }

    /// Export relationships to JSON format
    pub fn export_json(&self) -> CliResult<String> {
        let tuples = self.get_all_relationships()?;
        serde_json::to_string_pretty(&tuples)
            .map_err(|e| CliError::execution_error(format!("Failed to serialize to JSON: {}", e)))
    }

    /// Import relationships from Turtle format
    pub fn import_turtle(&mut self, turtle: &str) -> CliResult<usize> {
        // Simple parsing for demo purposes
        // In production, use a proper Turtle parser
        let mut count = 0;
        for line in turtle.lines() {
            let line = line.trim();
            if line.is_empty()
                || line.starts_with('@')
                || line.starts_with('{')
                || line.starts_with('}')
            {
                continue;
            }

            // Parse triple: <subject> predicate <object> .
            if let Some(parts) = self.parse_turtle_triple(line) {
                let tuple = RelationshipTuple::new(parts.0, parts.1, parts.2);
                self.add_relationship(&tuple)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Simple Turtle triple parser (for demo purposes)
    fn parse_turtle_triple(&self, line: &str) -> Option<(String, String, String)> {
        let line = line.trim_end_matches(" .").trim();
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return None;
        }

        let subject = parts[0].trim_matches('<').trim_matches('>').to_string();
        let predicate = parts[1].strip_prefix("auth:")?.to_string();
        let object = parts[2].trim_matches('<').trim_matches('>').to_string();

        Some((subject, predicate, object))
    }

    /// Import relationships from JSON format
    pub fn import_json(&mut self, json: &str) -> CliResult<usize> {
        let tuples: Vec<RelationshipTuple> = serde_json::from_str(json)
            .map_err(|e| CliError::execution_error(format!("Failed to parse JSON: {}", e)))?;

        self.add_relationships(&tuples)
    }

    /// Verify data integrity
    pub fn verify_integrity(&self) -> CliResult<IntegrityReport> {
        let duplicates = self.find_duplicates()?;
        let orphans = self.find_orphans()?;
        let total = self.get_all_relationships()?.len();

        Ok(IntegrityReport {
            total_relationships: total,
            duplicates: duplicates.len(),
            orphans: orphans.len(),
            is_valid: duplicates.is_empty() && orphans.is_empty(),
        })
    }
}

/// Statistics about ReBAC relationships
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RebacStatistics {
    pub total_relationships: usize,
    pub conditional_relationships: usize,
    pub by_relation: HashMap<String, usize>,
    pub by_subject: HashMap<String, usize>,
    pub by_object: HashMap<String, usize>,
}

/// Integrity verification report
#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub total_relationships: usize,
    pub duplicates: usize,
    pub orphans: usize,
    pub is_valid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_relationship_tuple_creation() {
        let tuple = RelationshipTuple::new("user:alice", "owner", "dataset:public");
        assert_eq!(tuple.subject, "user:alice");
        assert_eq!(tuple.relation, "owner");
        assert_eq!(tuple.object, "dataset:public");
        assert!(tuple.condition.is_none());
    }

    #[test]
    fn test_relationship_tuple_with_condition() {
        let tuple =
            RelationshipTuple::with_condition("user:bob", "can_read", "dataset:secret", "time>9am");
        assert_eq!(tuple.subject, "user:bob");
        assert_eq!(tuple.condition, Some("time>9am".to_string()));
    }

    #[test]
    fn test_rebac_manager_creation() {
        let manager = RebacManager::new_in_memory();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_add_and_query_relationships() {
        let mut manager = RebacManager::new_in_memory().unwrap();

        let tuple1 = RelationshipTuple::new("user:alice", "owner", "dataset:public");
        let tuple2 = RelationshipTuple::new("user:bob", "can_read", "dataset:public");

        assert!(manager.add_relationship(&tuple1).is_ok());
        assert!(manager.add_relationship(&tuple2).is_ok());

        let all = manager.get_all_relationships().unwrap();
        assert_eq!(all.len(), 2);

        // Query by subject
        let alice_rels = manager
            .query_relationships(Some("alice"), None, None)
            .unwrap();
        assert_eq!(alice_rels.len(), 1);
        assert_eq!(alice_rels[0].subject, "user:alice");
    }

    #[test]
    fn test_remove_relationship() {
        let mut manager = RebacManager::new_in_memory().unwrap();

        let tuple = RelationshipTuple::new("user:alice", "owner", "dataset:public");
        manager.add_relationship(&tuple).unwrap();

        assert!(manager.has_relationship(&tuple).unwrap());

        let removed = manager.remove_relationship(&tuple).unwrap();
        assert!(removed);
        assert!(!manager.has_relationship(&tuple).unwrap());
    }

    #[test]
    fn test_statistics() {
        let mut manager = RebacManager::new_in_memory().unwrap();

        manager
            .add_relationship(&RelationshipTuple::new(
                "user:alice",
                "owner",
                "dataset:public",
            ))
            .unwrap();
        manager
            .add_relationship(&RelationshipTuple::new(
                "user:bob",
                "can_read",
                "dataset:public",
            ))
            .unwrap();
        manager
            .add_relationship(&RelationshipTuple::new(
                "user:charlie",
                "can_read",
                "dataset:internal",
            ))
            .unwrap();

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_relationships, 3);
        assert_eq!(stats.by_relation.get("owner"), Some(&1));
        assert_eq!(stats.by_relation.get("can_read"), Some(&2));
    }

    #[test]
    fn test_find_duplicates() {
        let mut manager = RebacManager::new_in_memory().unwrap();

        let tuple = RelationshipTuple::new("user:alice", "owner", "dataset:public");
        manager.add_relationship(&tuple).unwrap();
        manager.add_relationship(&tuple).unwrap(); // Duplicate

        let duplicates = manager.find_duplicates().unwrap();
        assert_eq!(duplicates.len(), 1);
    }

    #[test]
    fn test_export_import_turtle() {
        let mut manager = RebacManager::new_in_memory().unwrap();

        manager
            .add_relationship(&RelationshipTuple::new(
                "user:alice",
                "owner",
                "dataset:public",
            ))
            .unwrap();
        manager
            .add_relationship(&RelationshipTuple::new(
                "user:bob",
                "can_read",
                "dataset:public",
            ))
            .unwrap();

        let turtle = manager.export_turtle().unwrap();
        assert!(turtle.contains("user:alice"));
        assert!(turtle.contains("auth:owner"));

        let mut new_manager = RebacManager::new_in_memory().unwrap();
        let count = new_manager.import_turtle(&turtle).unwrap();
        assert_eq!(count, 2);

        let all = new_manager.get_all_relationships().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_export_import_json() {
        let mut manager = RebacManager::new_in_memory().unwrap();

        manager
            .add_relationship(&RelationshipTuple::new(
                "user:alice",
                "owner",
                "dataset:public",
            ))
            .unwrap();

        let json = manager.export_json().unwrap();
        assert!(json.contains("user:alice"));

        let mut new_manager = RebacManager::new_in_memory().unwrap();
        let count = new_manager.import_json(&json).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_verify_integrity() {
        let mut manager = RebacManager::new_in_memory().unwrap();

        manager
            .add_relationship(&RelationshipTuple::new(
                "user:alice",
                "owner",
                "dataset:public",
            ))
            .unwrap();
        manager
            .add_relationship(&RelationshipTuple::new(
                "user:bob",
                "can_read",
                "dataset:public",
            ))
            .unwrap();

        let report = manager.verify_integrity().unwrap();
        assert_eq!(report.total_relationships, 2);
        assert!(report.is_valid);
    }

    #[test]
    fn test_clear_all() {
        let mut manager = RebacManager::new_in_memory().unwrap();

        manager
            .add_relationship(&RelationshipTuple::new(
                "user:alice",
                "owner",
                "dataset:public",
            ))
            .unwrap();
        manager
            .add_relationship(&RelationshipTuple::new(
                "user:bob",
                "can_read",
                "dataset:public",
            ))
            .unwrap();

        let count = manager.clear_all().unwrap();
        assert_eq!(count, 2);

        let all = manager.get_all_relationships().unwrap();
        assert_eq!(all.len(), 0);
    }

    #[test]
    fn test_persistent_storage() {
        let temp_dir = env::temp_dir().join(format!("rebac_test_{}", rng().gen::<u64>()));
        std::fs::create_dir_all(&temp_dir).unwrap();

        {
            let mut manager = RebacManager::new_persistent(&temp_dir).unwrap();
            manager
                .add_relationship(&RelationshipTuple::new(
                    "user:alice",
                    "owner",
                    "dataset:public",
                ))
                .unwrap();
        }

        // Reopen and verify persistence
        {
            let manager = RebacManager::new_persistent(&temp_dir).unwrap();
            let all = manager.get_all_relationships().unwrap();
            assert_eq!(all.len(), 1);
        }

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }
}
