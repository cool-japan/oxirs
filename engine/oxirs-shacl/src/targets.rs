//! SHACL target selection implementation
//!
//! This module handles target node selection according to SHACL specification.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use oxirs_core::{
    model::{BlankNode, NamedNode, RdfTerm, Term, Triple},
    OxirsError, Store,
};

use crate::{Result, ShaclError, SHACL_VOCAB};

/// SHACL target types for selecting nodes to validate
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Target {
    /// sh:targetClass - selects all instances of a class
    Class(NamedNode),

    /// sh:targetNode - selects specific nodes
    Node(Term),

    /// sh:targetObjectsOf - selects objects of a property
    ObjectsOf(NamedNode),

    /// sh:targetSubjectsOf - selects subjects of a property  
    SubjectsOf(NamedNode),

    /// sh:target with SPARQL SELECT query
    Sparql(SparqlTarget),

    /// Implicit target (shape IRI used as class)
    Implicit(NamedNode),
}

/// SPARQL-based target definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlTarget {
    /// SPARQL SELECT query that returns target nodes
    pub query: String,

    /// Optional prefixes for the query
    pub prefixes: Option<String>,
}

impl Target {
    /// Create a class target
    pub fn class(class_iri: NamedNode) -> Self {
        Target::Class(class_iri)
    }

    /// Create a node target
    pub fn node(node: Term) -> Self {
        Target::Node(node)
    }

    /// Create an objects-of target
    pub fn objects_of(property: NamedNode) -> Self {
        Target::ObjectsOf(property)
    }

    /// Create a subjects-of target
    pub fn subjects_of(property: NamedNode) -> Self {
        Target::SubjectsOf(property)
    }

    /// Create a SPARQL target
    pub fn sparql(query: String, prefixes: Option<String>) -> Self {
        Target::Sparql(SparqlTarget { query, prefixes })
    }

    /// Create an implicit target
    pub fn implicit(class_iri: NamedNode) -> Self {
        Target::Implicit(class_iri)
    }
}

/// Target selector for finding nodes that match target definitions
#[derive(Debug)]
pub struct TargetSelector {
    /// Cache for target results to improve performance
    cache: std::collections::HashMap<String, Vec<Term>>,

    /// Optimization settings
    optimization_config: TargetOptimizationConfig,
}

/// Configuration for target selection optimization
#[derive(Debug, Clone)]
pub struct TargetOptimizationConfig {
    /// Maximum number of results to return per target (0 = unlimited)
    pub max_results_per_target: usize,

    /// Enable query batching for multiple targets
    pub enable_batching: bool,

    /// Batch size for paginated queries
    pub batch_size: usize,

    /// Enable index-aware query optimization
    pub use_indexes: bool,

    /// Timeout for target queries in milliseconds
    pub query_timeout_ms: Option<u64>,
}

impl Default for TargetOptimizationConfig {
    fn default() -> Self {
        Self {
            max_results_per_target: 0, // unlimited
            enable_batching: true,
            batch_size: 10000,
            use_indexes: true,
            query_timeout_ms: Some(30000), // 30 seconds
        }
    }
}

impl TargetSelector {
    /// Create a new target selector
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            optimization_config: TargetOptimizationConfig::default(),
        }
    }

    /// Create a new target selector with custom optimization config
    pub fn with_config(config: TargetOptimizationConfig) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            optimization_config: config,
        }
    }

    /// Update optimization configuration
    pub fn set_optimization_config(&mut self, config: TargetOptimizationConfig) {
        self.optimization_config = config;
    }

    /// Select all target nodes for a given target definition
    pub fn select_targets(
        &mut self,
        store: &Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let cache_key = self.create_cache_key(target, graph_name);

        // Check cache first
        if let Some(cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        let result = self.select_targets_impl(store, target, graph_name)?;

        // Cache the result
        self.cache.insert(cache_key, result.clone());

        Ok(result)
    }

    /// Select target nodes for multiple targets
    pub fn select_multiple_targets(
        &mut self,
        store: &Store,
        targets: &[Target],
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        if targets.len() == 1 {
            return self.select_targets(store, &targets[0], graph_name);
        }

        // Use optimized UNION query if batching is enabled and targets support it
        if self.optimization_config.enable_batching && self.can_use_union_query(targets) {
            return self.select_targets_with_union_query(store, targets, graph_name);
        }

        // Fallback to individual target selection
        let mut all_targets = HashSet::new();

        for target in targets {
            let target_nodes = self.select_targets(store, target, graph_name)?;
            all_targets.extend(target_nodes);
        }

        Ok(all_targets.into_iter().collect())
    }

    /// Check if targets can be combined into a UNION query
    fn can_use_union_query(&self, targets: &[Target]) -> bool {
        // Only combine Class and Implicit targets for now
        targets
            .iter()
            .all(|target| matches!(target, Target::Class(_) | Target::Implicit(_)))
    }

    /// Select targets using optimized UNION query
    fn select_targets_with_union_query(
        &mut self,
        store: &Store,
        targets: &[Target],
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let cache_key = format!(
            "union_{}",
            targets
                .iter()
                .map(|t| format!("{:?}", t))
                .collect::<Vec<_>>()
                .join(",")
        );

        // Check cache first
        if let Some(cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        let query = self.build_union_query(targets, graph_name)?;

        let mut results = Vec::new();

        match self.execute_target_query(store, &query) {
            Ok(query_result) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = query_result
                {
                    for binding in bindings {
                        if let Some(instance) = binding.get("instance") {
                            results.push(instance.clone());
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Union query failed, falling back to individual queries: {}",
                    e
                );
                // Fallback to individual target selection
                let mut all_targets = HashSet::new();
                for target in targets {
                    let target_nodes = self.select_targets_impl(store, target, graph_name)?;
                    all_targets.extend(target_nodes);
                }
                results = all_targets.into_iter().collect();
            }
        }

        // Cache the result
        self.cache.insert(cache_key, results.clone());

        tracing::debug!("Union query found {} total targets", results.len());
        Ok(results)
    }

    /// Build a UNION query for multiple class targets
    fn build_union_query(&self, targets: &[Target], graph_name: Option<&str>) -> Result<String> {
        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

        let mut union_parts = Vec::new();

        for target in targets {
            let class_iri = match target {
                Target::Class(class) => class.as_str(),
                Target::Implicit(class) => class.as_str(),
                _ => continue, // Skip non-class targets
            };

            let pattern = if let Some(graph) = graph_name {
                format!(
                    "{{ GRAPH <{}> {{ ?instance <{}> <{}> . }} }}",
                    graph, rdf_type, class_iri
                )
            } else {
                format!("{{ ?instance <{}> <{}> . }}", rdf_type, class_iri)
            };

            union_parts.push(pattern);
        }

        if union_parts.is_empty() {
            return Err(ShaclError::TargetSelection(
                "No valid class targets for UNION query".to_string(),
            ));
        }

        let limit_clause = if self.optimization_config.max_results_per_target > 0 {
            format!(
                " LIMIT {}",
                self.optimization_config.max_results_per_target * targets.len()
            )
        } else {
            String::new()
        };

        let query = format!(
            "SELECT DISTINCT ?instance WHERE {{\n  {}\n}}{}",
            union_parts.join("\n  UNION\n  "),
            limit_clause
        );

        Ok(query)
    }

    /// Implementation of target selection without caching
    fn select_targets_impl(
        &self,
        store: &Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        match target {
            Target::Class(class_iri) => self.select_class_instances(store, class_iri, graph_name),
            Target::Node(node) => Ok(vec![node.clone()]),
            Target::ObjectsOf(property) => {
                self.select_objects_of_property(store, property, graph_name)
            }
            Target::SubjectsOf(property) => {
                self.select_subjects_of_property(store, property, graph_name)
            }
            Target::Sparql(sparql_target) => {
                self.select_sparql_targets(store, sparql_target, graph_name)
            }
            Target::Implicit(class_iri) => {
                self.select_class_instances(store, class_iri, graph_name)
            }
        }
    }

    /// Select all instances of a class using rdf:type relationships
    fn select_class_instances(
        &self,
        store: &Store,
        class_iri: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .map_err(|e| ShaclError::Core(OxirsError::Parse(e.to_string())))?;

        let mut instances = Vec::new();

        // Create an optimized SPARQL query to find all instances of the class
        let limit_clause = if self.optimization_config.max_results_per_target > 0 {
            format!(" LIMIT {}", self.optimization_config.max_results_per_target)
        } else {
            String::new()
        };

        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?instance WHERE {{
                    GRAPH <{}> {{
                        ?instance <{}> <{}> .
                    }}
                }}
                ORDER BY ?instance{}
            "#,
                graph,
                rdf_type.as_str(),
                class_iri.as_str(),
                limit_clause
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?instance WHERE {{
                    ?instance <{}> <{}> .
                }}
                ORDER BY ?instance{}
            "#,
                rdf_type.as_str(),
                class_iri.as_str(),
                limit_clause
            )
        };

        // Execute the query using oxirs-core query engine
        match self.execute_target_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(instance) = binding.get("instance") {
                            instances.push(instance.clone());
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to execute class instance query: {}", e);
                // Fallback to direct store querying
                instances = self.select_class_instances_direct(store, class_iri, graph_name)?;
            }
        }

        tracing::debug!(
            "Found {} instances of class {}",
            instances.len(),
            class_iri.as_str()
        );
        Ok(instances)
    }

    /// Select objects of a specific property
    fn select_objects_of_property(
        &self,
        store: &Store,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut objects = Vec::new();

        // Create an optimized SPARQL query to find all objects of the property
        let limit_clause = if self.optimization_config.max_results_per_target > 0 {
            format!(" LIMIT {}", self.optimization_config.max_results_per_target)
        } else {
            String::new()
        };

        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?object WHERE {{
                    GRAPH <{}> {{
                        ?subject <{}> ?object .
                    }}
                }}
                ORDER BY ?object{}
            "#,
                graph,
                property.as_str(),
                limit_clause
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?object WHERE {{
                    ?subject <{}> ?object .
                }}
                ORDER BY ?object{}
            "#,
                property.as_str(),
                limit_clause
            )
        };

        // Execute the query using oxirs-core query engine
        match self.execute_target_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(object) = binding.get("object") {
                            objects.push(object.clone());
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to execute objects query: {}", e);
                // Fallback to direct store querying
                objects = self.select_objects_direct(store, property, graph_name)?;
            }
        }

        tracing::debug!(
            "Found {} objects of property {}",
            objects.len(),
            property.as_str()
        );
        Ok(objects)
    }

    /// Select subjects of a specific property
    fn select_subjects_of_property(
        &self,
        store: &Store,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut subjects = Vec::new();

        // Create an optimized SPARQL query to find all subjects of the property
        let limit_clause = if self.optimization_config.max_results_per_target > 0 {
            format!(" LIMIT {}", self.optimization_config.max_results_per_target)
        } else {
            String::new()
        };

        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?subject WHERE {{
                    GRAPH <{}> {{
                        ?subject <{}> ?object .
                    }}
                }}
                ORDER BY ?subject{}
            "#,
                graph,
                property.as_str(),
                limit_clause
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?subject WHERE {{
                    ?subject <{}> ?object .
                }}
                ORDER BY ?subject{}
            "#,
                property.as_str(),
                limit_clause
            )
        };

        // Execute the query using oxirs-core query engine
        match self.execute_target_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(subject) = binding.get("subject") {
                            subjects.push(subject.clone());
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to execute subjects query: {}", e);
                // Fallback to direct store querying
                subjects = self.select_subjects_direct(store, property, graph_name)?;
            }
        }

        tracing::debug!(
            "Found {} subjects of property {}",
            subjects.len(),
            property.as_str()
        );
        Ok(subjects)
    }

    /// Select targets using SPARQL query
    fn select_sparql_targets(
        &self,
        store: &Store,
        sparql_target: &SparqlTarget,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut complete_query = String::new();

        // Add prefixes if provided
        if let Some(ref prefixes) = sparql_target.prefixes {
            complete_query.push_str(prefixes);
            complete_query.push('\n');
        }

        // Add the main query
        let query = if let Some(graph) = graph_name {
            // Wrap the query in a GRAPH clause if graph is specified
            format!(
                "SELECT ?this WHERE {{ GRAPH <{}> {{ {} }} }}",
                graph, sparql_target.query
            )
        } else {
            sparql_target.query.clone()
        };

        complete_query.push_str(&query);

        let mut targets = Vec::new();

        // Execute the SPARQL query using oxirs-core query engine
        match self.execute_target_query(store, &complete_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        // Look for the standard ?this variable
                        if let Some(this_value) = binding.get("this") {
                            targets.push(this_value.clone());
                        }
                        // Also check for other common target variable names
                        else if let Some(target_value) = binding.get("target") {
                            targets.push(target_value.clone());
                        }
                        // If no standard variable found, take the first binding value
                        else if let Some(first_value) = binding.values().next() {
                            targets.push(first_value.clone());
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to execute SPARQL target query: {}", e);
                return Err(ShaclError::TargetSelection(format!(
                    "SPARQL target query execution failed: {}",
                    e
                )));
            }
        }

        tracing::debug!("Found {} targets from SPARQL query", targets.len());
        Ok(targets)
    }

    /// Execute a target selection query using oxirs-core query engine
    fn execute_target_query(
        &self,
        store: &Store,
        query: &str,
    ) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;

        let query_engine = QueryEngine::new();

        tracing::debug!("Executing target query: {}", query);

        let result = query_engine.query(query, store).map_err(|e| {
            ShaclError::TargetSelection(format!("Target query execution failed: {}", e))
        })?;

        Ok(result)
    }

    /// Fallback method to select class instances using direct store queries
    fn select_class_instances_direct(
        &self,
        store: &Store,
        class_iri: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, Object, Predicate, Subject};

        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .map_err(|e| ShaclError::Core(OxirsError::Parse(e.to_string())))?;

        let predicate = Predicate::NamedNode(rdf_type);
        let object = Object::NamedNode(class_iri.clone());

        let graph_filter = if let Some(g) = graph_name {
            Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                ShaclError::Core(OxirsError::Parse(e.to_string()))
            })?))
        } else {
            None
        };

        let quads = store
            .query_quads(
                None, // Any subject
                Some(&predicate),
                Some(&object),
                graph_filter.as_ref(),
            )
            .map_err(|e| ShaclError::Core(e))?;

        let instances: Vec<Term> = quads
            .into_iter()
            .map(|quad| Term::from(quad.subject().clone()))
            .collect();

        Ok(instances)
    }

    /// Fallback method to select objects using direct store queries
    fn select_objects_direct(
        &self,
        store: &Store,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, Predicate};

        let predicate = Predicate::NamedNode(property.clone());

        let graph_filter = if let Some(g) = graph_name {
            Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                ShaclError::Core(OxirsError::Parse(e.to_string()))
            })?))
        } else {
            None
        };

        let quads = store
            .query_quads(
                None, // Any subject
                Some(&predicate),
                None, // Any object
                graph_filter.as_ref(),
            )
            .map_err(|e| ShaclError::Core(e))?;

        let objects: Vec<Term> = quads
            .into_iter()
            .map(|quad| Term::from(quad.object().clone()))
            .collect();

        Ok(objects)
    }

    /// Fallback method to select subjects using direct store queries
    fn select_subjects_direct(
        &self,
        store: &Store,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, Predicate};

        let predicate = Predicate::NamedNode(property.clone());

        let graph_filter = if let Some(g) = graph_name {
            Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                ShaclError::Core(OxirsError::Parse(e.to_string()))
            })?))
        } else {
            None
        };

        let quads = store
            .query_quads(
                None, // Any subject
                Some(&predicate),
                None, // Any object
                graph_filter.as_ref(),
            )
            .map_err(|e| ShaclError::Core(e))?;

        let subjects: Vec<Term> = quads
            .into_iter()
            .map(|quad| Term::from(quad.subject().clone()))
            .collect();

        Ok(subjects)
    }

    /// Create a cache key for target results
    fn create_cache_key(&self, target: &Target, graph_name: Option<&str>) -> String {
        let graph_suffix = graph_name.unwrap_or("default");

        match target {
            Target::Class(class_iri) => format!("class:{}:{}", class_iri.as_str(), graph_suffix),
            Target::Node(node) => format!("node:{}:{}", node.as_str(), graph_suffix),
            Target::ObjectsOf(property) => {
                format!("objects_of:{}:{}", property.as_str(), graph_suffix)
            }
            Target::SubjectsOf(property) => {
                format!("subjects_of:{}:{}", property.as_str(), graph_suffix)
            }
            Target::Sparql(sparql_target) => {
                // Use a hash of the query for caching
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                sparql_target.query.hash(&mut hasher);
                let query_hash = hasher.finish();

                format!("sparql:{}:{}", query_hash, graph_suffix)
            }
            Target::Implicit(class_iri) => {
                format!("implicit:{}:{}", class_iri.as_str(), graph_suffix)
            }
        }
    }

    /// Clear the target cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> TargetCacheStats {
        TargetCacheStats {
            entries: self.cache.len(),
            total_targets: self.cache.values().map(|v| v.len()).sum(),
        }
    }
}

impl Default for TargetSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about target cache performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetCacheStats {
    /// Number of cache entries
    pub entries: usize,

    /// Total number of cached targets across all entries
    pub total_targets: usize,
}

/// Performance statistics for target selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSelectionStats {
    /// Cache statistics
    pub cache: TargetCacheStats,

    /// Number of queries executed
    pub queries_executed: usize,

    /// Number of cache hits
    pub cache_hits: usize,

    /// Number of fallback operations
    pub fallback_operations: usize,

    /// Number of union queries used
    pub union_queries_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_creation() {
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let target = Target::class(class_iri.clone());

        match target {
            Target::Class(iri) => assert_eq!(iri, class_iri),
            _ => panic!("Expected class target"),
        }
    }

    #[test]
    fn test_node_target() {
        let node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
        let target = Target::node(node.clone());

        match target {
            Target::Node(n) => assert_eq!(n, node),
            _ => panic!("Expected node target"),
        }
    }

    #[test]
    fn test_sparql_target() {
        let query = "SELECT ?this WHERE { ?this a ex:Person }".to_string();
        let prefixes = Some("PREFIX ex: <http://example.org/>".to_string());
        let target = Target::sparql(query.clone(), prefixes.clone());

        match target {
            Target::Sparql(sparql_target) => {
                assert_eq!(sparql_target.query, query);
                assert_eq!(sparql_target.prefixes, prefixes);
            }
            _ => panic!("Expected SPARQL target"),
        }
    }

    #[test]
    fn test_cache_key_generation() {
        let selector = TargetSelector::new();
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let target = Target::class(class_iri);

        let key1 = selector.create_cache_key(&target, None);
        let key2 = selector.create_cache_key(&target, None);
        let key3 = selector.create_cache_key(&target, Some("graph1"));

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_target_optimization_config() {
        let config = TargetOptimizationConfig::default();

        assert_eq!(config.max_results_per_target, 0); // unlimited
        assert!(config.enable_batching);
        assert_eq!(config.batch_size, 10000);
        assert!(config.use_indexes);
        assert_eq!(config.query_timeout_ms, Some(30000));
    }

    #[test]
    fn test_target_selector_with_config() {
        let mut config = TargetOptimizationConfig::default();
        config.max_results_per_target = 100;
        config.enable_batching = false;

        let selector = TargetSelector::with_config(config.clone());
        assert_eq!(selector.optimization_config.max_results_per_target, 100);
        assert!(!selector.optimization_config.enable_batching);
    }
}
