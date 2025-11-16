//! Schema Introspection Module
//!
//! Automatically discovers RDF schema patterns from the knowledge graph to improve
//! natural language to SPARQL translation and provide better query suggestions.

use anyhow::{Context as AnyhowContext, Result};
use oxirs_core::{rdf_store::SolutionMapping, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Discovered RDF class information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfClass {
    /// Class URI
    pub uri: String,
    /// Human-readable label (from rdfs:label)
    pub label: Option<String>,
    /// Number of instances
    pub instance_count: usize,
    /// Properties used by instances of this class
    pub properties: Vec<RdfProperty>,
    /// Comment/description (from rdfs:comment)
    pub comment: Option<String>,
}

/// RDF property information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfProperty {
    /// Property URI
    pub uri: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Domain (class that has this property)
    pub domain: Option<String>,
    /// Range (expected value type)
    pub range: Option<String>,
    /// Usage frequency
    pub usage_count: usize,
    /// Sample values (for generation examples)
    pub sample_values: Vec<String>,
}

/// Discovered schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredSchema {
    /// All discovered classes
    pub classes: Vec<RdfClass>,
    /// All discovered properties
    pub properties: Vec<RdfProperty>,
    /// Common prefixes used in the dataset
    pub prefixes: HashMap<String, String>,
    /// Triple count
    pub total_triples: usize,
}

impl DiscoveredSchema {
    /// Find a class by URI or label
    pub fn find_class(&self, query: &str) -> Option<&RdfClass> {
        let query_lower = query.to_lowercase();
        self.classes.iter().find(|c| {
            c.uri.to_lowercase().contains(&query_lower)
                || c.label
                    .as_ref()
                    .map(|l| l.to_lowercase().contains(&query_lower))
                    .unwrap_or(false)
        })
    }

    /// Find a property by URI or label
    pub fn find_property(&self, query: &str) -> Option<&RdfProperty> {
        let query_lower = query.to_lowercase();
        self.properties.iter().find(|p| {
            p.uri.to_lowercase().contains(&query_lower)
                || p.label
                    .as_ref()
                    .map(|l| l.to_lowercase().contains(&query_lower))
                    .unwrap_or(false)
        })
    }

    /// Get properties for a specific class
    pub fn get_class_properties(&self, class_uri: &str) -> Vec<&RdfProperty> {
        self.properties
            .iter()
            .filter(|p| p.domain.as_ref().map(|d| d == class_uri).unwrap_or(false))
            .collect()
    }

    /// Generate SPARQL prefix declarations
    pub fn generate_prefix_declarations(&self) -> String {
        let mut prefixes = String::new();
        for (prefix, uri) in &self.prefixes {
            prefixes.push_str(&format!("PREFIX {prefix}: <{uri}>\n"));
        }
        prefixes
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Schema: {} classes, {} properties, {} triples",
            self.classes.len(),
            self.properties.len(),
            self.total_triples
        )
    }
}

/// Schema introspection engine
pub struct SchemaIntrospector {
    store: Arc<dyn Store>,
    config: IntrospectionConfig,
}

/// Configuration for schema introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionConfig {
    /// Maximum number of sample values to collect per property
    pub max_sample_values: usize,
    /// Minimum instance count to include a class
    pub min_class_instances: usize,
    /// Minimum usage count to include a property
    pub min_property_usage: usize,
    /// Whether to discover implicit classes (entities without rdf:type)
    pub discover_implicit_classes: bool,
    /// Timeout for introspection queries (seconds)
    pub query_timeout_secs: u64,
}

impl Default for IntrospectionConfig {
    fn default() -> Self {
        Self {
            max_sample_values: 10,
            min_class_instances: 1,
            min_property_usage: 1,
            discover_implicit_classes: true,
            query_timeout_secs: 30,
        }
    }
}

impl SchemaIntrospector {
    /// Create a new schema introspector
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self {
            store,
            config: IntrospectionConfig::default(),
        }
    }

    /// Helper to extract a value from a solution mapping
    fn get_solution_value(solution: &SolutionMapping, var_name: &str) -> Option<String> {
        for (var, term) in solution.iter() {
            if var == var_name {
                return Some(term.to_string());
            }
        }
        None
    }

    /// Create with custom configuration
    pub fn with_config(store: Arc<dyn Store>, config: IntrospectionConfig) -> Self {
        Self { store, config }
    }

    /// Discover schema from the knowledge graph
    pub async fn discover_schema(&self) -> Result<DiscoveredSchema> {
        info!("Starting schema introspection");

        let total_triples = self.count_total_triples().await?;
        let classes = self.discover_classes().await?;
        let properties = self.discover_properties().await?;
        let prefixes = self.discover_common_prefixes(&classes, &properties);

        let schema = DiscoveredSchema {
            classes,
            properties,
            prefixes,
            total_triples,
        };

        info!("{}", schema.summary());
        Ok(schema)
    }

    /// Count total triples in the store
    async fn count_total_triples(&self) -> Result<usize> {
        let query = "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }";
        let prepared = self
            .store
            .prepare_query(query)
            .context("Failed to prepare triple count query")?;

        let results = prepared
            .exec()
            .context("Failed to execute triple count query")?;

        for solution in results {
            if let Some(count_str) = Self::get_solution_value(&solution, "count") {
                if let Ok(count) = count_str.parse::<usize>() {
                    return Ok(count);
                }
            }
        }

        Ok(0)
    }

    /// Discover all classes in the knowledge graph
    async fn discover_classes(&self) -> Result<Vec<RdfClass>> {
        debug!("Discovering RDF classes");

        let query = r#"
            SELECT ?class (COUNT(DISTINCT ?instance) AS ?count) ?label ?comment
            WHERE {
                ?instance a ?class .
                OPTIONAL { ?class rdfs:label ?label }
                OPTIONAL { ?class rdfs:comment ?comment }
            }
            GROUP BY ?class ?label ?comment
            HAVING (COUNT(DISTINCT ?instance) >= 1)
            ORDER BY DESC(?count)
        "#;

        let prepared = self
            .store
            .prepare_query(query)
            .context("Failed to prepare class discovery query")?;

        let results = prepared
            .exec()
            .context("Failed to execute class discovery query")?;

        let mut classes = Vec::new();

        for solution in results {
            if let Some(uri) = Self::get_solution_value(&solution, "class") {
                let label = Self::get_solution_value(&solution, "label");
                let comment = Self::get_solution_value(&solution, "comment");
                let instance_count = Self::get_solution_value(&solution, "count")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);

                if instance_count >= self.config.min_class_instances {
                    // Discover properties for this class
                    let properties = self.discover_class_properties(&uri).await?;

                    classes.push(RdfClass {
                        uri,
                        label,
                        instance_count,
                        properties,
                        comment,
                    });
                }
            }
        }

        info!("Discovered {} classes", classes.len());
        Ok(classes)
    }

    /// Discover properties for a specific class
    async fn discover_class_properties(&self, class_uri: &str) -> Result<Vec<RdfProperty>> {
        let query = format!(
            r#"
            SELECT ?property (COUNT(*) AS ?count) ?label
            WHERE {{
                ?instance a <{class_uri}> .
                ?instance ?property ?value .
                OPTIONAL {{ ?property rdfs:label ?label }}
            }}
            GROUP BY ?property ?label
            ORDER BY DESC(?count)
            LIMIT 50
        "#
        );

        let prepared = self
            .store
            .prepare_query(&query)
            .context("Failed to prepare property discovery query")?;

        let results = prepared
            .exec()
            .context("Failed to execute property discovery query")?;

        let mut properties = Vec::new();

        for solution in results {
            if let Some(uri) = Self::get_solution_value(&solution, "property") {
                let label = Self::get_solution_value(&solution, "label");
                let usage_count = Self::get_solution_value(&solution, "count")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);

                if usage_count >= self.config.min_property_usage {
                    // Get sample values for this property
                    let sample_values = self.get_property_samples(class_uri, &uri).await?;

                    properties.push(RdfProperty {
                        uri,
                        label,
                        domain: Some(class_uri.to_string()),
                        range: None, // Could be inferred from sample values
                        usage_count,
                        sample_values,
                    });
                }
            }
        }

        Ok(properties)
    }

    /// Get sample values for a property
    async fn get_property_samples(
        &self,
        class_uri: &str,
        property_uri: &str,
    ) -> Result<Vec<String>> {
        let query = format!(
            r#"
            SELECT DISTINCT ?value
            WHERE {{
                ?instance a <{class_uri}> .
                ?instance <{property_uri}> ?value .
            }}
            LIMIT {}
        "#,
            self.config.max_sample_values
        );

        let prepared = self
            .store
            .prepare_query(&query)
            .context("Failed to prepare sample values query")?;

        let results = prepared
            .exec()
            .context("Failed to execute sample values query")?;

        let mut samples = Vec::new();
        for solution in results {
            if let Some(value) = Self::get_solution_value(&solution, "value") {
                samples.push(value);
            }
        }

        Ok(samples)
    }

    /// Discover all properties (global view)
    async fn discover_properties(&self) -> Result<Vec<RdfProperty>> {
        debug!("Discovering RDF properties");

        let query = r#"
            SELECT ?property (COUNT(*) AS ?count) ?label
            WHERE {
                ?s ?property ?o .
                OPTIONAL { ?property rdfs:label ?label }
            }
            GROUP BY ?property ?label
            HAVING (COUNT(*) >= 1)
            ORDER BY DESC(?count)
            LIMIT 200
        "#;

        let prepared = self
            .store
            .prepare_query(query)
            .context("Failed to prepare property discovery query")?;

        let results = prepared
            .exec()
            .context("Failed to execute property discovery query")?;

        let mut properties = Vec::new();

        for solution in results {
            if let Some(uri) = Self::get_solution_value(&solution, "property") {
                let label = Self::get_solution_value(&solution, "label");
                let usage_count = Self::get_solution_value(&solution, "count")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);

                if usage_count >= self.config.min_property_usage {
                    properties.push(RdfProperty {
                        uri,
                        label,
                        domain: None,
                        range: None,
                        usage_count,
                        sample_values: Vec::new(),
                    });
                }
            }
        }

        info!("Discovered {} properties", properties.len());
        Ok(properties)
    }

    /// Discover common prefixes from URIs
    fn discover_common_prefixes(
        &self,
        classes: &[RdfClass],
        properties: &[RdfProperty],
    ) -> HashMap<String, String> {
        let mut prefix_map = HashMap::new();

        // Standard prefixes
        prefix_map.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefix_map.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefix_map.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefix_map.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        prefix_map.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        prefix_map.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );
        prefix_map.insert(
            "dcterms".to_string(),
            "http://purl.org/dc/terms/".to_string(),
        );

        // Extract namespace from URIs
        let mut namespaces: HashMap<String, usize> = HashMap::new();

        for class in classes {
            if let Some(ns) = Self::extract_namespace(&class.uri) {
                *namespaces.entry(ns).or_insert(0) += 1;
            }
        }

        for property in properties {
            if let Some(ns) = Self::extract_namespace(&property.uri) {
                *namespaces.entry(ns).or_insert(0) += 1;
            }
        }

        // Add most common namespaces
        let mut ns_vec: Vec<_> = namespaces.into_iter().collect();
        ns_vec.sort_by(|a, b| b.1.cmp(&a.1));

        for (i, (ns, _count)) in ns_vec.iter().take(10).enumerate() {
            let prefix = format!("ns{}", i + 1);
            if !prefix_map.values().any(|v| v == ns) {
                prefix_map.insert(prefix, ns.clone());
            }
        }

        prefix_map
    }

    /// Extract namespace from a URI
    fn extract_namespace(uri: &str) -> Option<String> {
        if let Some(hash_pos) = uri.rfind('#') {
            Some(uri[..=hash_pos].to_string())
        } else {
            uri.rfind('/')
                .map(|slash_pos| uri[..=slash_pos].to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_namespace() {
        assert_eq!(
            SchemaIntrospector::extract_namespace("http://example.org/ns#Class"),
            Some("http://example.org/ns#".to_string())
        );
        assert_eq!(
            SchemaIntrospector::extract_namespace("http://example.org/ns/property"),
            Some("http://example.org/ns/".to_string())
        );
    }

    #[test]
    fn test_discovered_schema_find_class() {
        let schema = DiscoveredSchema {
            classes: vec![RdfClass {
                uri: "http://example.org/Person".to_string(),
                label: Some("Person".to_string()),
                instance_count: 10,
                properties: Vec::new(),
                comment: None,
            }],
            properties: Vec::new(),
            prefixes: HashMap::new(),
            total_triples: 0,
        };

        assert!(schema.find_class("Person").is_some());
        assert!(schema.find_class("person").is_some());
        assert!(schema.find_class("Nonexistent").is_none());
    }
}
