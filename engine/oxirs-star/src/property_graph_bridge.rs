//! # Property Graph Bridge for RDF-star
//!
//! Bidirectional conversion between RDF-star and Labeled Property Graph (LPG) models.
//!
//! This module provides:
//! - **RDF-star → LPG**: Convert RDF-star to Neo4j/Cypher-compatible property graphs
//! - **LPG → RDF-star**: Import property graphs as RDF-star with quoted triples
//! - **Cypher Query Translation**: Translate SPARQL-star ↔ Cypher queries
//! - **Schema Mapping**: Flexible mapping between RDF vocabularies and graph schemas
//! - **Provenance Preservation**: Maintain RDF-star annotations as graph properties
//!
//! ## Overview
//!
//! Property graphs (like Neo4j) and RDF-star both support rich metadata, but use different
//! models. This bridge enables:
//!
//! - Migrating from Neo4j to OxiRS or vice versa
//! - Hybrid architectures using both models
//! - Leveraging Neo4j's graph algorithms on RDF-star data
//! - Using SPARQL-star queries on property graph data
//!
//! ## Mapping Model
//!
//! ### RDF-star → Property Graph
//!
//! ```text
//! RDF Triple: <alice> <knows> <bob>
//! → Node(id: "alice") -[:knows]-> Node(id: "bob")
//!
//! RDF-star Quoted Triple: << <alice> <age> 30 >> <certainty> 0.9
//! → Node(alice) -[:age {value: 30, certainty: 0.9}]-> Node(30)
//! ```
//!
//! ### Property Graph → RDF-star
//!
//! ```text
//! (alice)-[:knows {since: 2020}]->(bob)
//! → <alice> <knows> <bob> .
//!   << <alice> <knows> <bob> >> <since> "2020" .
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_star::property_graph_bridge::{PropertyGraphBridge, ConversionConfig, LabeledPropertyGraph};
//! use oxirs_star::StarStore;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut store = StarStore::new();
//! // ... populate store with RDF-star data
//!
//! // Convert to property graph
//! let config = ConversionConfig::default();
//! let bridge = PropertyGraphBridge::new(config);
//! let lpg = bridge.rdf_to_lpg(&store)?;
//!
//! // Export to Neo4j Cypher format
//! let cypher = lpg.to_cypher_script()?;
//! println!("{}", cypher);
//!
//! // Convert back to RDF-star
//! let restored = bridge.lpg_to_rdf(&lpg)?;
//! # Ok(())
//! # }
//! ```

use crate::{StarResult, StarStore, StarTerm, StarTriple};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{info, instrument, warn};

// Import SciRS2 components (SCIRS2 POLICY)
use scirs2_core::profiling::Profiler;

/// Conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionConfig {
    /// Use URIs as node IDs (vs. generating numeric IDs)
    pub use_uri_ids: bool,
    /// Preserve blank nodes
    pub preserve_blank_nodes: bool,
    /// Convert literals to property values (vs. separate nodes)
    pub literals_as_properties: bool,
    /// Maximum property size before creating separate node
    pub max_property_size: usize,
    /// Namespace prefix mappings
    pub namespace_prefixes: HashMap<String, String>,
    /// Default edge label for unlabeled relationships
    pub default_edge_label: String,
    /// Property name for RDF types
    pub type_property_name: String,
    /// Convert quoted triples to edge properties
    pub quoted_as_edge_properties: bool,
    /// Enable bidirectional edges (for symmetric properties)
    pub bidirectional_edges: bool,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
            "rdf".to_string(),
        );
        prefixes.insert(
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
            "rdfs".to_string(),
        );
        prefixes.insert(
            "http://www.w3.org/2002/07/owl#".to_string(),
            "owl".to_string(),
        );
        prefixes.insert("http://xmlns.com/foaf/0.1/".to_string(), "foaf".to_string());
        prefixes.insert(
            "http://purl.org/dc/elements/1.1/".to_string(),
            "dc".to_string(),
        );

        Self {
            use_uri_ids: true,
            preserve_blank_nodes: true,
            literals_as_properties: true,
            max_property_size: 10_000,
            namespace_prefixes: prefixes,
            default_edge_label: "related".to_string(),
            type_property_name: "rdf_type".to_string(),
            quoted_as_edge_properties: true,
            bidirectional_edges: false,
        }
    }
}

/// Node in a labeled property graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpgNode {
    /// Unique node ID
    pub id: String,
    /// Node labels (types)
    pub labels: Vec<String>,
    /// Node properties
    pub properties: HashMap<String, PropertyValue>,
}

impl LpgNode {
    /// Create a new node
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            labels: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Add a label
    pub fn add_label(&mut self, label: impl Into<String>) {
        self.labels.push(label.into());
    }

    /// Set a property
    pub fn set_property(&mut self, key: impl Into<String>, value: PropertyValue) {
        self.properties.insert(key.into(), value);
    }

    /// Get a property
    pub fn get_property(&self, key: &str) -> Option<&PropertyValue> {
        self.properties.get(key)
    }
}

/// Edge in a labeled property graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpgEdge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Edge type/label
    pub label: String,
    /// Edge properties
    pub properties: HashMap<String, PropertyValue>,
}

impl LpgEdge {
    /// Create a new edge
    pub fn new(from: impl Into<String>, label: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            label: label.into(),
            to: to.into(),
            properties: HashMap::new(),
        }
    }

    /// Set a property
    pub fn set_property(&mut self, key: impl Into<String>, value: PropertyValue) {
        self.properties.insert(key.into(), value);
    }
}

/// Property value in a property graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<PropertyValue>),
    Map(HashMap<String, PropertyValue>),
}

impl PropertyValue {
    /// Convert to string representation
    pub fn to_string_repr(&self) -> String {
        match self {
            PropertyValue::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            PropertyValue::Integer(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::Boolean(b) => b.to_string(),
            PropertyValue::List(items) => {
                let inner: Vec<_> = items.iter().map(|v| v.to_string_repr()).collect();
                format!("[{}]", inner.join(", "))
            }
            PropertyValue::Map(map) => {
                let inner: Vec<_> = map
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_string_repr()))
                    .collect();
                format!("{{{}}}", inner.join(", "))
            }
        }
    }
}

/// Labeled property graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LabeledPropertyGraph {
    /// All nodes in the graph
    pub nodes: HashMap<String, LpgNode>,
    /// All edges in the graph
    pub edges: Vec<LpgEdge>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl LabeledPropertyGraph {
    /// Create a new empty property graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node
    pub fn add_node(&mut self, node: LpgNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<&LpgNode> {
        self.nodes.get(id)
    }

    /// Get mutable node
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut LpgNode> {
        self.nodes.get_mut(id)
    }

    /// Add an edge
    pub fn add_edge(&mut self, edge: LpgEdge) {
        self.edges.push(edge);
    }

    /// Get edges from a node
    pub fn get_edges_from(&self, node_id: &str) -> Vec<&LpgEdge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Get edges to a node
    pub fn get_edges_to(&self, node_id: &str) -> Vec<&LpgEdge> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Get all edges between two nodes
    pub fn get_edges_between(&self, from: &str, to: &str) -> Vec<&LpgEdge> {
        self.edges
            .iter()
            .filter(|e| e.from == from && e.to == to)
            .collect()
    }

    /// Generate Cypher CREATE statements
    #[instrument(skip(self))]
    pub fn to_cypher_script(&self) -> StarResult<String> {
        let mut script = String::new();
        script.push_str("// Neo4j Cypher Script - Generated from RDF-star\n");
        script.push_str(
            "// WARNING: This will create nodes and edges. Run in an empty database.\n\n",
        );

        // Create nodes
        script.push_str("// Create Nodes\n");
        for node in self.nodes.values() {
            let labels = if node.labels.is_empty() {
                String::new()
            } else {
                format!(":{}", node.labels.join(":"))
            };

            let props = if node.properties.is_empty() {
                String::new()
            } else {
                let props: Vec<_> = node
                    .properties
                    .iter()
                    .map(|(k, v)| format!("{}: {}", Self::escape_cypher_id(k), v.to_string_repr()))
                    .collect();
                format!(" {{{}}}", props.join(", "))
            };

            script.push_str(&format!("CREATE (n{}{})\n", labels, props));
        }

        script.push_str("\n// Create Edges\n");
        for edge in &self.edges {
            let props = if edge.properties.is_empty() {
                String::new()
            } else {
                let props: Vec<_> = edge
                    .properties
                    .iter()
                    .map(|(k, v)| format!("{}: {}", Self::escape_cypher_id(k), v.to_string_repr()))
                    .collect();
                format!(" {{{}}}", props.join(", "))
            };

            script.push_str(&format!(
                "MATCH (a {{id: \"{}\"}}), (b {{id: \"{}\"}})\n",
                edge.from.replace('"', "\\\""),
                edge.to.replace('"', "\\\"")
            ));
            script.push_str(&format!(
                "CREATE (a)-[:{}{}]->(b)\n\n",
                Self::escape_cypher_label(&edge.label),
                props
            ));
        }

        Ok(script)
    }

    /// Escape a Cypher identifier
    fn escape_cypher_id(id: &str) -> String {
        if id.chars().all(|c| c.is_alphanumeric() || c == '_') {
            id.to_string()
        } else {
            format!("`{}`", id.replace('`', "``"))
        }
    }

    /// Escape a Cypher label
    fn escape_cypher_label(label: &str) -> String {
        label
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }

    /// Get statistics
    pub fn statistics(&self) -> LpgStatistics {
        let mut label_counts: HashMap<String, usize> = HashMap::new();
        for node in self.nodes.values() {
            for label in &node.labels {
                *label_counts.entry(label.clone()).or_insert(0) += 1;
            }
        }

        let mut edge_type_counts: HashMap<String, usize> = HashMap::new();
        for edge in &self.edges {
            *edge_type_counts.entry(edge.label.clone()).or_insert(0) += 1;
        }

        LpgStatistics {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            label_counts,
            edge_type_counts,
        }
    }
}

/// LPG statistics
#[derive(Debug, Clone)]
pub struct LpgStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub label_counts: HashMap<String, usize>,
    pub edge_type_counts: HashMap<String, usize>,
}

/// Property graph bridge
pub struct PropertyGraphBridge {
    config: ConversionConfig,
    #[allow(dead_code)]
    profiler: Profiler,
}

impl PropertyGraphBridge {
    /// Create a new property graph bridge
    pub fn new(config: ConversionConfig) -> Self {
        Self {
            config,
            profiler: Profiler::new(),
        }
    }

    /// Convert RDF-star store to labeled property graph
    #[instrument(skip(self, store), fields(triple_count = store.len()))]
    pub fn rdf_to_lpg(&self, store: &StarStore) -> StarResult<LabeledPropertyGraph> {
        info!("Converting RDF-star to property graph");

        let mut lpg = LabeledPropertyGraph::new();
        let mut node_ids: HashSet<String> = HashSet::new();

        // First pass: collect all subjects and objects as potential nodes
        for triple in store.iter() {
            let subj_id = self.term_to_node_id(&triple.subject);
            let obj_id = self.term_to_node_id(&triple.object);

            node_ids.insert(subj_id);
            if !self.config.literals_as_properties || !matches!(triple.object, StarTerm::Literal(_))
            {
                node_ids.insert(obj_id);
            }
        }

        // Create nodes
        for node_id in &node_ids {
            let mut node = LpgNode::new(node_id.clone());
            node.set_property("id", PropertyValue::String(node_id.clone()));
            lpg.add_node(node);
        }

        // Second pass: create edges and properties
        for triple in store.iter() {
            let subj_id = self.term_to_node_id(&triple.subject);
            let pred_label = self.term_to_edge_label(&triple.predicate);

            match &triple.object {
                StarTerm::Literal(lit) if self.config.literals_as_properties => {
                    // Add as node property
                    if let Some(node) = lpg.get_node_mut(&subj_id) {
                        let value = self.literal_to_property_value(lit)?;
                        node.set_property(pred_label, value);
                    }
                }
                _ => {
                    // Create edge
                    let obj_id = self.term_to_node_id(&triple.object);
                    let edge = LpgEdge::new(subj_id, pred_label, obj_id);
                    lpg.add_edge(edge);
                }
            }

            // Handle quoted triple annotations
            if let StarTerm::QuotedTriple(qt) = &triple.subject {
                // Add metadata to the edge representing the quoted triple
                self.add_quoted_triple_metadata(&mut lpg, qt, &triple)?;
            }
        }

        // Add metadata
        lpg.metadata
            .insert("source".to_string(), "rdf-star".to_string());
        lpg.metadata
            .insert("conversion_config".to_string(), "default".to_string());

        info!(
            "Conversion complete: {} nodes, {} edges",
            lpg.nodes.len(),
            lpg.edges.len()
        );
        Ok(lpg)
    }

    /// Convert labeled property graph to RDF-star
    #[instrument(skip(self, lpg), fields(node_count = lpg.nodes.len(), edge_count = lpg.edges.len()))]
    pub fn lpg_to_rdf(&self, lpg: &LabeledPropertyGraph) -> StarResult<StarStore> {
        info!("Converting property graph to RDF-star");

        let store = StarStore::new();

        // Convert node properties to triples
        for (node_id, node) in &lpg.nodes {
            let subject = StarTerm::iri(node_id)?;

            // Add type triples for labels
            for label in &node.labels {
                let type_triple = StarTriple::new(
                    subject.clone(),
                    StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
                    StarTerm::iri(label)?,
                );
                store.insert(&type_triple)?;
            }

            // Add property triples
            for (key, value) in &node.properties {
                if key != "id" {
                    // Skip the 'id' property as it's the subject URI
                    let predicate = StarTerm::iri(&self.expand_namespace(key))?;
                    let object = self.property_value_to_term(value)?;

                    let triple = StarTriple::new(subject.clone(), predicate, object);
                    store.insert(&triple)?;
                }
            }
        }

        // Convert edges to triples
        for edge in &lpg.edges {
            let subject = StarTerm::iri(&edge.from)?;
            let predicate = StarTerm::iri(&self.expand_namespace(&edge.label))?;
            let object = StarTerm::iri(&edge.to)?;

            let base_triple = StarTriple::new(subject.clone(), predicate.clone(), object.clone());
            store.insert(&base_triple)?;

            // If edge has properties, create quoted triple with annotations
            if !edge.properties.is_empty() {
                for (key, value) in &edge.properties {
                    let meta_pred = StarTerm::iri(&self.expand_namespace(key))?;
                    let meta_obj = self.property_value_to_term(value)?;

                    let quoted_triple = StarTriple::new(
                        StarTerm::quoted_triple(base_triple.clone()),
                        meta_pred,
                        meta_obj,
                    );
                    store.insert(&quoted_triple)?;
                }
            }
        }

        info!("Conversion complete: {} triples", store.len());
        Ok(store)
    }

    /// Convert StarTerm to node ID
    fn term_to_node_id(&self, term: &StarTerm) -> String {
        match term {
            StarTerm::NamedNode(nn) => {
                if self.config.use_uri_ids {
                    nn.iri.clone()
                } else {
                    self.compact_uri(&nn.iri)
                }
            }
            StarTerm::BlankNode(bn) => format!("_:{}", bn.id),
            StarTerm::Literal(lit) => format!("literal:{}", lit.value),
            StarTerm::Variable(var) => format!("?{}", var.name),
            StarTerm::QuotedTriple(qt) => {
                format!(
                    "<<{} {} {}>>",
                    self.term_to_node_id(&qt.subject),
                    self.term_to_node_id(&qt.predicate),
                    self.term_to_node_id(&qt.object)
                )
            }
        }
    }

    /// Convert StarTerm to edge label
    fn term_to_edge_label(&self, term: &StarTerm) -> String {
        match term {
            StarTerm::NamedNode(nn) => self.compact_uri(&nn.iri),
            _ => self.config.default_edge_label.clone(),
        }
    }

    /// Compact a URI using namespace prefixes
    fn compact_uri(&self, uri: &str) -> String {
        for (namespace, prefix) in &self.config.namespace_prefixes {
            if uri.starts_with(namespace) {
                let local = &uri[namespace.len()..];
                return format!("{}_{}", prefix, local);
            }
        }
        // No prefix found, use last path segment or whole URI
        uri.rsplit('/').next().unwrap_or(uri).to_string()
    }

    /// Expand a namespace-prefixed name to full URI
    fn expand_namespace(&self, name: &str) -> String {
        for (namespace, prefix) in &self.config.namespace_prefixes {
            let prefix_with_underscore = format!("{}_", prefix);
            if name.starts_with(&prefix_with_underscore) {
                let local = &name[prefix_with_underscore.len()..];
                return format!("{}{}", namespace, local);
            }
        }
        // No prefix match, assume it's already a full URI or create one
        if name.starts_with("http://") || name.starts_with("https://") {
            name.to_string()
        } else {
            format!("http://example.org/{}", name)
        }
    }

    /// Convert RDF literal to property value
    fn literal_to_property_value(&self, lit: &crate::model::Literal) -> StarResult<PropertyValue> {
        // Try to parse as number
        if let Ok(i) = lit.value.parse::<i64>() {
            return Ok(PropertyValue::Integer(i));
        }
        if let Ok(f) = lit.value.parse::<f64>() {
            return Ok(PropertyValue::Float(f));
        }
        // Try to parse as boolean
        if let Ok(b) = lit.value.parse::<bool>() {
            return Ok(PropertyValue::Boolean(b));
        }

        // Default to string
        Ok(PropertyValue::String(lit.value.clone()))
    }

    /// Convert property value to RDF term
    fn property_value_to_term(&self, value: &PropertyValue) -> StarResult<StarTerm> {
        match value {
            PropertyValue::String(s) => StarTerm::literal(s),
            PropertyValue::Integer(i) => StarTerm::literal(&i.to_string()),
            PropertyValue::Float(f) => StarTerm::literal(&f.to_string()),
            PropertyValue::Boolean(b) => StarTerm::literal(&b.to_string()),
            PropertyValue::List(items) => {
                // Convert list to string representation
                let str_items: Vec<_> = items.iter().map(|v| v.to_string_repr()).collect();
                StarTerm::literal(&format!("[{}]", str_items.join(", ")))
            }
            PropertyValue::Map(map) => {
                // Convert map to string representation
                let str_items: Vec<_> = map
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_string_repr()))
                    .collect();
                StarTerm::literal(&format!("{{{}}}", str_items.join(", ")))
            }
        }
    }

    /// Add metadata from quoted triple to corresponding edge
    fn add_quoted_triple_metadata(
        &self,
        lpg: &mut LabeledPropertyGraph,
        quoted_triple: &StarTriple,
        meta_triple: &StarTriple,
    ) -> StarResult<()> {
        let from = self.term_to_node_id(&quoted_triple.subject);
        let to = self.term_to_node_id(&quoted_triple.object);

        // Find the corresponding edge
        for edge in &mut lpg.edges {
            if edge.from == from && edge.to == to {
                // Add the metadata as edge property
                let prop_key = self.term_to_edge_label(&meta_triple.predicate);
                if let StarTerm::Literal(lit) = &meta_triple.object {
                    let prop_value = self.literal_to_property_value(lit)?;
                    edge.set_property(prop_key, prop_value);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Translate SPARQL-star query to Cypher (simplified)
    pub fn sparql_to_cypher(&self, sparql: &str) -> StarResult<String> {
        // Simplified translation - production would need full parser
        let mut cypher = String::new();

        if sparql.to_lowercase().contains("select") {
            cypher.push_str("MATCH ");

            // Extract pattern (very simplified)
            if sparql.contains("?s ?p ?o") {
                cypher.push_str("(s)-[r]->(o) RETURN s, r, o");
            } else if sparql.contains("<< ?s ?p ?o >>") {
                cypher.push_str("(s)-[r]->(o) RETURN s, r, o, properties(r)");
            } else {
                cypher.push_str("(n) RETURN n LIMIT 100");
            }
        }

        Ok(cypher)
    }

    /// Translate Cypher query to SPARQL-star (simplified)
    pub fn cypher_to_sparql(&self, cypher: &str) -> StarResult<String> {
        // Simplified translation
        let mut sparql = String::new();

        if cypher.to_lowercase().contains("match") {
            sparql.push_str("SELECT * WHERE { ");

            // Extract pattern (very simplified)
            if cypher.contains("(a)-[r]->(b)") {
                sparql.push_str("?a ?r ?b . ");
            } else {
                sparql.push_str("?s ?p ?o . ");
            }

            sparql.push('}');
        }

        Ok(sparql)
    }

    /// Get conversion statistics
    pub fn conversion_statistics(&self) -> ConversionStatistics {
        ConversionStatistics {
            conversions_performed: 0, // Placeholder
        }
    }
}

/// Conversion statistics
#[derive(Debug, Clone)]
pub struct ConversionStatistics {
    pub conversions_performed: u64,
}

/// Cypher query builder for common graph patterns
pub struct CypherQueryBuilder {
    query: String,
}

impl CypherQueryBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            query: String::new(),
        }
    }

    /// Add MATCH clause
    pub fn match_pattern(mut self, pattern: &str) -> Self {
        if !self.query.is_empty() {
            self.query.push(' ');
        }
        self.query.push_str("MATCH ");
        self.query.push_str(pattern);
        self
    }

    /// Add WHERE clause
    pub fn where_clause(mut self, condition: &str) -> Self {
        self.query.push_str(" WHERE ");
        self.query.push_str(condition);
        self
    }

    /// Add RETURN clause
    pub fn return_clause(mut self, items: &str) -> Self {
        self.query.push_str(" RETURN ");
        self.query.push_str(items);
        self
    }

    /// Add LIMIT
    pub fn limit(mut self, count: usize) -> Self {
        self.query.push_str(&format!(" LIMIT {}", count));
        self
    }

    /// Build the query
    pub fn build(self) -> String {
        self.query
    }
}

impl Default for CypherQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_config_default() {
        let config = ConversionConfig::default();
        assert!(config.use_uri_ids);
        assert!(config.literals_as_properties);
        assert_eq!(config.default_edge_label, "related");
    }

    #[test]
    fn test_lpg_node_creation() {
        let mut node = LpgNode::new("n1");
        node.add_label("Person");
        node.set_property("name", PropertyValue::String("Alice".to_string()));
        node.set_property("age", PropertyValue::Integer(30));

        assert_eq!(node.id, "n1");
        assert_eq!(node.labels.len(), 1);
        assert_eq!(node.properties.len(), 2);
        assert_eq!(
            node.get_property("name"),
            Some(&PropertyValue::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_lpg_edge_creation() {
        let mut edge = LpgEdge::new("n1", "knows", "n2");
        edge.set_property("since", PropertyValue::Integer(2020));

        assert_eq!(edge.from, "n1");
        assert_eq!(edge.to, "n2");
        assert_eq!(edge.label, "knows");
        assert_eq!(edge.properties.len(), 1);
    }

    #[test]
    fn test_property_value_string_repr() {
        assert_eq!(
            PropertyValue::String("test".to_string()).to_string_repr(),
            "\"test\""
        );
        assert_eq!(PropertyValue::Integer(42).to_string_repr(), "42");
        assert_eq!(PropertyValue::Float(2.5).to_string_repr(), "2.5");
        assert_eq!(PropertyValue::Boolean(true).to_string_repr(), "true");

        let list = PropertyValue::List(vec![PropertyValue::Integer(1), PropertyValue::Integer(2)]);
        assert_eq!(list.to_string_repr(), "[1, 2]");
    }

    #[test]
    fn test_labeled_property_graph() {
        let mut lpg = LabeledPropertyGraph::new();

        let mut node1 = LpgNode::new("n1");
        node1.add_label("Person");
        node1.set_property("name", PropertyValue::String("Alice".to_string()));

        let mut node2 = LpgNode::new("n2");
        node2.add_label("Person");
        node2.set_property("name", PropertyValue::String("Bob".to_string()));

        lpg.add_node(node1);
        lpg.add_node(node2);

        let edge = LpgEdge::new("n1", "knows", "n2");
        lpg.add_edge(edge);

        assert_eq!(lpg.nodes.len(), 2);
        assert_eq!(lpg.edges.len(), 1);
        assert!(lpg.get_node("n1").is_some());
        assert_eq!(lpg.get_edges_from("n1").len(), 1);
    }

    #[test]
    fn test_rdf_to_lpg_simple() {
        let store = StarStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        store.insert(&triple).unwrap();

        let config = ConversionConfig::default();
        let bridge = PropertyGraphBridge::new(config);
        let lpg = bridge.rdf_to_lpg(&store).unwrap();

        assert!(lpg.nodes.len() >= 2);
        assert_eq!(lpg.edges.len(), 1);
    }

    #[test]
    fn test_rdf_to_lpg_with_literal() {
        let store = StarStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        );
        store.insert(&triple).unwrap();

        let config = ConversionConfig::default();
        let bridge = PropertyGraphBridge::new(config);
        let lpg = bridge.rdf_to_lpg(&store).unwrap();

        // With literals_as_properties=true, literal becomes a property
        let alice_node = lpg.get_node("http://example.org/alice").unwrap();
        assert!(
            alice_node.properties.contains_key("foaf_age")
                || alice_node.properties.contains_key("age")
                || !lpg.edges.is_empty()
        ); // Or becomes an edge
    }

    #[test]
    fn test_lpg_to_rdf() {
        let mut lpg = LabeledPropertyGraph::new();

        let mut node = LpgNode::new("http://example.org/alice");
        node.add_label("http://example.org/Person");
        node.set_property("name", PropertyValue::String("Alice".to_string()));
        lpg.add_node(node);

        let config = ConversionConfig::default();
        let bridge = PropertyGraphBridge::new(config);
        let store = bridge.lpg_to_rdf(&lpg).unwrap();

        // Should have at least a type triple and a name triple
        assert!(store.len() >= 2);
    }

    #[test]
    fn test_lpg_to_cypher() {
        let mut lpg = LabeledPropertyGraph::new();

        let mut node1 = LpgNode::new("n1");
        node1.add_label("Person");
        node1.set_property("name", PropertyValue::String("Alice".to_string()));
        lpg.add_node(node1);

        let cypher = lpg.to_cypher_script().unwrap();
        assert!(cypher.contains("CREATE"));
        assert!(cypher.contains("Person"));
        assert!(cypher.contains("Alice"));
    }

    #[test]
    fn test_cypher_query_builder() {
        let query = CypherQueryBuilder::new()
            .match_pattern("(a:Person)-[:knows]->(b:Person)")
            .where_clause("a.age > 25")
            .return_clause("a, b")
            .limit(10)
            .build();

        assert!(query.contains("MATCH (a:Person)-[:knows]->(b:Person)"));
        assert!(query.contains("WHERE a.age > 25"));
        assert!(query.contains("RETURN a, b"));
        assert!(query.contains("LIMIT 10"));
    }

    #[test]
    fn test_compact_uri() {
        let config = ConversionConfig::default();
        let bridge = PropertyGraphBridge::new(config);

        let compacted = bridge.compact_uri("http://xmlns.com/foaf/0.1/name");
        assert_eq!(compacted, "foaf_name");

        let compacted2 = bridge.compact_uri("http://example.org/unknown");
        assert_eq!(compacted2, "unknown");
    }

    #[test]
    fn test_roundtrip_conversion() {
        let store = StarStore::new();

        // Add some triples
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://xmlns.com/foaf/0.1/name").unwrap(),
            StarTerm::literal("Alice").unwrap(),
        );
        store.insert(&triple1).unwrap();

        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://xmlns.com/foaf/0.1/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        store.insert(&triple2).unwrap();

        let config = ConversionConfig::default();
        let bridge = PropertyGraphBridge::new(config.clone());

        // RDF -> LPG
        let lpg = bridge.rdf_to_lpg(&store).unwrap();
        assert!(!lpg.nodes.is_empty());

        // LPG -> RDF
        let restored = bridge.lpg_to_rdf(&lpg).unwrap();

        // Should have roughly the same number of triples (may have additional type triples)
        assert!(restored.len() >= store.len());
    }

    #[test]
    fn test_lpg_statistics() {
        let mut lpg = LabeledPropertyGraph::new();

        let mut node1 = LpgNode::new("n1");
        node1.add_label("Person");
        lpg.add_node(node1);

        let mut node2 = LpgNode::new("n2");
        node2.add_label("Person");
        lpg.add_node(node2);

        let mut node3 = LpgNode::new("n3");
        node3.add_label("Organization");
        lpg.add_node(node3);

        lpg.add_edge(LpgEdge::new("n1", "knows", "n2"));
        lpg.add_edge(LpgEdge::new("n1", "worksFor", "n3"));

        let stats = lpg.statistics();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.label_counts.get("Person"), Some(&2));
        assert_eq!(stats.label_counts.get("Organization"), Some(&1));
        assert_eq!(stats.edge_type_counts.get("knows"), Some(&1));
    }
}
