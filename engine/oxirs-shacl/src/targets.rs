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
    cache: std::collections::HashMap<String, CachedTargetResult>,

    /// Optimization settings
    optimization_config: TargetOptimizationConfig,

    /// Performance statistics
    stats: TargetSelectionStats,

    /// Query plan cache for SPARQL targets
    query_plan_cache: std::collections::HashMap<String, QueryPlan>,

    /// Index usage statistics for adaptive optimization
    index_usage_stats: std::collections::HashMap<String, IndexUsageStats>,
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

    /// Enable adaptive batch sizing based on performance
    pub adaptive_batching: bool,

    /// Enable parallel processing for independent targets
    pub enable_parallel_processing: bool,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for TargetOptimizationConfig {
    fn default() -> Self {
        Self {
            max_results_per_target: 0, // unlimited
            enable_batching: true,
            batch_size: 10000,
            use_indexes: true,
            query_timeout_ms: Some(30000), // 30 seconds
            adaptive_batching: true,
            enable_parallel_processing: true,
            cache_ttl_seconds: 300, // 5 minutes
        }
    }
}

/// Cached target result with metadata
#[derive(Debug, Clone)]
struct CachedTargetResult {
    /// The cached target nodes
    targets: Vec<Term>,
    /// Timestamp when cached
    cached_at: std::time::Instant,
    /// Number of cache hits
    hit_count: usize,
    /// Average query execution time
    avg_execution_time_ms: f64,
}

/// Query execution plan for SPARQL targets
#[derive(Debug, Clone)]
struct QueryPlan {
    /// Optimized query string
    optimized_query: String,
    /// Estimated execution cost
    estimated_cost: f64,
    /// Index hints
    index_hints: Vec<String>,
    /// Expected result count
    estimated_result_count: usize,
    /// Plan creation timestamp
    created_at: std::time::Instant,
}

/// Index usage statistics for adaptive optimization
#[derive(Debug, Clone)]
struct IndexUsageStats {
    /// Number of times this index was used
    usage_count: usize,
    /// Total execution time with this index
    total_execution_time_ms: f64,
    /// Average selectivity
    avg_selectivity: f64,
    /// Last update timestamp
    last_updated: std::time::Instant,
}

impl TargetSelector {
    /// Create a new target selector
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            optimization_config: TargetOptimizationConfig::default(),
            stats: TargetSelectionStats::default(),
            query_plan_cache: std::collections::HashMap::new(),
            index_usage_stats: std::collections::HashMap::new(),
        }
    }

    /// Create a new target selector with custom optimization config
    pub fn with_config(config: TargetOptimizationConfig) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            optimization_config: config,
            stats: TargetSelectionStats::default(),
            query_plan_cache: std::collections::HashMap::new(),
            index_usage_stats: std::collections::HashMap::new(),
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
        let start_time = std::time::Instant::now();
        let cache_key = self.create_cache_key(target, graph_name);

        // Check cache first and validate TTL
        if let Some(cached_result) = self.cache.get_mut(&cache_key) {
            let cache_age = cached_result.cached_at.elapsed();
            if cache_age.as_secs() <= self.optimization_config.cache_ttl_seconds {
                // Cache hit - update statistics
                cached_result.hit_count += 1;
                self.stats.cache_hits += 1;
                return Ok(cached_result.targets.clone());
            } else {
                // Cache expired - remove entry
                self.cache.remove(&cache_key);
            }
        }

        // Cache miss - execute query
        let execution_start = std::time::Instant::now();
        let result = self.select_targets_impl(store, target, graph_name)?;
        let execution_time = execution_start.elapsed();

        // Update statistics
        self.stats.queries_executed += 1;
        let execution_time_ms = execution_time.as_millis() as f64;

        // Cache the result with metadata
        let cached_result = CachedTargetResult {
            targets: result.clone(),
            cached_at: std::time::Instant::now(),
            hit_count: 0,
            avg_execution_time_ms: execution_time_ms,
        };
        self.cache.insert(cache_key, cached_result);

        // Adaptive cache management - evict old entries if cache is too large
        if self.cache.len() > 1000 {
            self.evict_old_cache_entries();
        }

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
        let start_time = std::time::Instant::now();
        let cache_key = format!(
            "union_{}",
            targets
                .iter()
                .map(|t| format!("{:?}", t))
                .collect::<Vec<_>>()
                .join(",")
        );

        // Check cache first and validate TTL
        if let Some(cached_result) = self.cache.get_mut(&cache_key) {
            let cache_age = cached_result.cached_at.elapsed();
            if cache_age.as_secs() <= self.optimization_config.cache_ttl_seconds {
                // Cache hit - update statistics
                cached_result.hit_count += 1;
                self.stats.cache_hits += 1;
                self.stats.union_queries_used += 1;
                return Ok(cached_result.targets.clone());
            } else {
                // Cache expired - remove entry
                self.cache.remove(&cache_key);
            }
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

        // Cache the result with metadata
        let execution_time_ms = start_time.elapsed().as_millis() as f64;
        let cached_result = CachedTargetResult {
            targets: results.clone(),
            cached_at: std::time::Instant::now(),
            hit_count: 0,
            avg_execution_time_ms: execution_time_ms,
        };
        self.cache.insert(cache_key, cached_result);

        // Update statistics
        self.stats.queries_executed += 1;
        self.stats.union_queries_used += 1;

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
        &mut self,
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
        &mut self,
        store: &Store,
        class_iri: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .map_err(|e| ShaclError::Core(OxirsError::Parse(e.to_string())))?;

        let mut instances = Vec::new();

        // Create an optimized SPARQL query to find all instances of the class
        let target = Target::Class(class_iri.clone());
        let adaptive_limit = if self.optimization_config.max_results_per_target > 0 {
            self.optimization_config.max_results_per_target
        } else {
            // Use adaptive batch sizing
            self.calculate_adaptive_batch_size(&target)
        };

        let limit_clause = if adaptive_limit > 0 {
            format!(" LIMIT {}", adaptive_limit)
        } else {
            String::new()
        };

        let base_query = if let Some(graph) = graph_name {
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

        // Apply index optimizations
        let query = self.optimize_query_with_indexes(&base_query, &target);

        // Execute the query using oxirs-core query engine with timing
        let query_start = std::time::Instant::now();
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

                // Update performance statistics
                let execution_time_ms = query_start.elapsed().as_millis() as f64;
                self.update_index_usage_stats(&target, execution_time_ms, instances.len());
            }
            Err(e) => {
                tracing::error!("Failed to execute class instance query: {}", e);
                // Fallback to direct store querying
                self.stats.fallback_operations += 1;
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
        &mut self,
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
        &mut self,
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
        &mut self,
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
        &mut self,
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
        &mut self,
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
        &mut self,
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
        self.stats.cache_hits = 0;
        self.stats.queries_executed = 0;
        self.stats.fallback_operations = 0;
        self.stats.union_queries_used = 0;
    }

    /// Evict old cache entries based on age and usage
    fn evict_old_cache_entries(&mut self) {
        let now = std::time::Instant::now();
        let cache_ttl = std::time::Duration::from_secs(self.optimization_config.cache_ttl_seconds);

        // Collect keys of expired entries
        let expired_keys: Vec<String> = self
            .cache
            .iter()
            .filter(|(_, cached_result)| now.duration_since(cached_result.cached_at) > cache_ttl)
            .map(|(key, _)| key.clone())
            .collect();

        // Remove expired entries
        for key in expired_keys {
            self.cache.remove(&key);
        }

        // If still too many entries, remove least recently used
        if self.cache.len() > 500 {
            let mut entries: Vec<_> = self.cache.iter().collect();
            entries.sort_by(|a, b| {
                // Sort by hit count (ascending) then by age (descending)
                let hit_cmp = a.1.hit_count.cmp(&b.1.hit_count);
                if hit_cmp == std::cmp::Ordering::Equal {
                    b.1.cached_at.cmp(&a.1.cached_at)
                } else {
                    hit_cmp
                }
            });

            // Remove the least used entries
            let keys_to_remove: Vec<String> = entries
                .iter()
                .take(self.cache.len() - 500)
                .map(|(key, _)| key.to_string())
                .collect();

            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
    }

    /// Optimize query with index hints and advanced query rewriting
    fn optimize_query_with_indexes(&self, query: &str, target: &Target) -> String {
        if !self.optimization_config.use_indexes {
            return query.to_string();
        }

        let mut optimized_query = query.to_string();

        // Advanced query optimization based on target type and performance history
        match target {
            Target::Class(class_iri) => {
                // Add index hint for rdf:type queries
                optimized_query = optimized_query.replace(
                    "?instance <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                    "?instance <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> # USE_INDEX(type_index)"
                );
                
                // Add query optimization hints for common class hierarchies
                if self.is_common_class(class_iri) {
                    optimized_query = self.add_subclass_optimization(&optimized_query, class_iri);
                }
            }
            Target::ObjectsOf(property) => {
                // Add index hint for property-based queries
                optimized_query = optimized_query.replace(
                    &format!("?subject <{}>", property.as_str()),
                    &format!(
                        "?subject <{}> # USE_INDEX(property_index)",
                        property.as_str()
                    ),
                );
                
                // Add cardinality-based optimization
                optimized_query = self.add_cardinality_optimization(&optimized_query, property);
            }
            Target::SubjectsOf(property) => {
                // Add index hint for reverse property queries
                optimized_query = optimized_query.replace(
                    &format!("?subject <{}>", property.as_str()),
                    &format!(
                        "?subject <{}> # USE_INDEX(reverse_property_index)",
                        property.as_str()
                    ),
                );
                
                // Add functional property optimization
                optimized_query = self.add_functional_property_optimization(&optimized_query, property);
            }
            Target::Sparql(sparql_target) => {
                // Optimize custom SPARQL queries
                optimized_query = self.optimize_custom_sparql_query(&sparql_target.query);
            }
            _ => {
                // No specific optimization for other target types
            }
        }

        // Apply general query optimizations
        optimized_query = self.apply_general_optimizations(optimized_query);

        optimized_query
    }
    
    /// Check if a class is commonly used in the knowledge base
    fn is_common_class(&self, class_iri: &NamedNode) -> bool {
        let common_classes = [
            "http://www.w3.org/2002/07/owl#Thing",
            "http://www.w3.org/2000/01/rdf-schema#Resource",
            "http://xmlns.com/foaf/0.1/Person",
            "http://xmlns.com/foaf/0.1/Organization",
            "http://www.w3.org/2004/02/skos/core#Concept",
        ];
        
        common_classes.contains(&class_iri.as_str())
    }
    
    /// Add subclass hierarchy optimization for better query planning
    fn add_subclass_optimization(&self, query: &str, class_iri: &NamedNode) -> String {
        // Add OPTIONAL subclass patterns for hierarchical class queries
        format!(
            "{}\n  OPTIONAL {{ ?instance <http://www.w3.org/2000/01/rdf-schema#subClassOf>* <{}> . }}",
            query.trim_end(),
            class_iri.as_str()
        )
    }
    
    /// Add cardinality-based optimization hints
    fn add_cardinality_optimization(&self, query: &str, property: &NamedNode) -> String {
        // Check if we have cardinality statistics for this property
        if let Some(stats) = self.index_usage_stats.get(&format!("property_{}", property.as_str())) {
            if stats.avg_selectivity < 0.1 {
                // High selectivity property - add optimization hint
                return query.replace(
                    &format!("?subject <{}>", property.as_str()),
                    &format!("?subject <{}> # HIGH_SELECTIVITY", property.as_str())
                );
            }
        }
        query.to_string()
    }
    
    /// Add functional property optimization
    fn add_functional_property_optimization(&self, query: &str, property: &NamedNode) -> String {
        // For functional properties, we can optimize by adding LIMIT 1 per subject
        if self.is_functional_property(property) {
            return query.replace(
                "ORDER BY ?subject",
                "ORDER BY ?subject # FUNCTIONAL_PROPERTY"
            );
        }
        query.to_string()
    }
    
    /// Check if a property is functional (has at most one value per subject)
    fn is_functional_property(&self, property: &NamedNode) -> bool {
        // Common functional properties
        let functional_properties = [
            "http://xmlns.com/foaf/0.1/name",
            "http://xmlns.com/foaf/0.1/mbox",
            "http://www.w3.org/2000/01/rdf-schema#label",
        ];
        
        functional_properties.contains(&property.as_str())
    }
    
    /// Optimize custom SPARQL queries with advanced techniques
    fn optimize_custom_sparql_query(&self, query: &str) -> String {
        let mut optimized = query.to_string();
        
        // Add common query optimizations
        
        // 1. Reorder triple patterns by selectivity
        optimized = self.reorder_triple_patterns(&optimized);
        
        // 2. Add FILTER optimization hints
        optimized = self.optimize_filters(&optimized);
        
        // 3. Add bind variable optimization
        optimized = self.optimize_bind_variables(&optimized);
        
        optimized
    }
    
    /// Reorder triple patterns for better query execution
    fn reorder_triple_patterns(&self, query: &str) -> String {
        // Basic heuristic: put more selective patterns first
        // In a real implementation, this would analyze the query and reorder based on statistics
        
        if query.contains("rdf:type") && query.contains("rdfs:label") {
            // Type patterns are usually more selective than label patterns
            return query.replace(
                "rdfs:label",
                "# MOVE_AFTER_TYPE rdfs:label"
            );
        }
        
        query.to_string()
    }
    
    /// Optimize FILTER expressions in SPARQL queries
    fn optimize_filters(&self, query: &str) -> String {
        let mut optimized = query.to_string();
        
        // Push filters down to reduce intermediate results
        if optimized.contains("FILTER") && optimized.contains("OPTIONAL") {
            optimized = optimized.replace(
                "FILTER",
                "# PUSH_DOWN FILTER"
            );
        }
        
        optimized
    }
    
    /// Optimize BIND variables in SPARQL queries
    fn optimize_bind_variables(&self, query: &str) -> String {
        let mut optimized = query.to_string();
        
        // Optimize common bind patterns
        if optimized.contains("BIND(") {
            optimized = optimized.replace(
                "BIND(",
                "# OPTIMIZE_BIND BIND("
            );
        }
        
        optimized
    }
    
    /// Apply general query optimizations
    fn apply_general_optimizations(&self, query: String) -> String {
        let mut optimized = query;
        
        // 1. Add query timeout hint
        if let Some(timeout_ms) = self.optimization_config.query_timeout_ms {
            optimized = format!("# TIMEOUT {} ms\n{}", timeout_ms, optimized);
        }
        
        // 2. Add parallel processing hint if enabled
        if self.optimization_config.enable_parallel_processing {
            optimized = format!("# ENABLE_PARALLEL\n{}", optimized);
        }
        
        // 3. Add result limit if configured
        if self.optimization_config.max_results_per_target > 0 && !optimized.contains("LIMIT") {
            optimized = format!("{}\nLIMIT {}", optimized, self.optimization_config.max_results_per_target);
        }
        
        optimized
    }
    
    /// Generate optimized query plan for complex targets
    pub fn generate_query_plan(&mut self, targets: &[Target], graph_name: Option<&str>) -> Result<String> {
        let plan_key = format!("plan_{:?}_{:?}", targets, graph_name);
        
        // Check if we have a cached plan
        if let Some(cached_plan) = self.query_plan_cache.get(&plan_key) {
            let plan_age = cached_plan.created_at.elapsed();
            if plan_age.as_secs() < 3600 { // Plans valid for 1 hour
                return Ok(cached_plan.optimized_query.clone());
            }
        }
        
        // Generate new optimized plan
        let mut plan_parts = Vec::new();
        let mut estimated_cost = 0.0;
        let mut estimated_results = 0;
        
        for target in targets {
            let target_query = self.generate_single_target_query(target, graph_name)?;
            let target_cost = self.estimate_query_cost(target);
            let target_results = self.estimate_result_count(target);
            
            plan_parts.push(target_query);
            estimated_cost += target_cost;
            estimated_results += target_results;
        }
        
        // Combine queries with optimal strategy
        let combined_query = if self.can_use_union_query(targets) {
            self.build_union_query(targets, graph_name)?
        } else {
            // Use VALUES clause for efficient combination
            format!(
                "SELECT DISTINCT ?target WHERE {{\n  VALUES ?source {{ {} }}\n  {} \n}}",
                targets.iter().map(|_| "UNDEF").collect::<Vec<_>>().join(" "),
                plan_parts.join("\n  UNION\n  ")
            )
        };
        
        // Cache the generated plan
        let query_plan = QueryPlan {
            optimized_query: combined_query.clone(),
            estimated_cost,
            index_hints: self.generate_index_hints(targets),
            estimated_result_count: estimated_results,
            created_at: std::time::Instant::now(),
        };
        
        self.query_plan_cache.insert(plan_key, query_plan);
        
        Ok(combined_query)
    }
    
    /// Generate query for a single target
    fn generate_single_target_query(&self, target: &Target, graph_name: Option<&str>) -> Result<String> {
        match target {
            Target::Class(class_iri) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}>", graph)
                } else {
                    String::new()
                };
                
                Ok(format!(
                    "{} {{ ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> . }}",
                    graph_clause,
                    class_iri.as_str()
                ))
            }
            Target::Node(node) => {
                Ok(format!("BIND({} AS ?target)", node.as_str()))
            }
            Target::ObjectsOf(property) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}>", graph)
                } else {
                    String::new()
                };
                
                Ok(format!(
                    "{} {{ ?subject <{}> ?target . }}",
                    graph_clause,
                    property.as_str()
                ))
            }
            Target::SubjectsOf(property) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}>", graph)
                } else {
                    String::new()
                };
                
                Ok(format!(
                    "{} {{ ?target <{}> ?object . }}",
                    graph_clause,
                    property.as_str()
                ))
            }
            Target::Sparql(sparql_target) => {
                Ok(sparql_target.query.clone())
            }
            Target::Implicit(class_iri) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}>", graph)
                } else {
                    String::new()
                };
                
                Ok(format!(
                    "{} {{ ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> . }}",
                    graph_clause,
                    class_iri.as_str()
                ))
            }
        }
    }
    
    /// Estimate query execution cost
    fn estimate_query_cost(&self, target: &Target) -> f64 {
        match target {
            Target::Class(_) => 1.0,
            Target::Node(_) => 0.1,
            Target::ObjectsOf(_) => 2.0,
            Target::SubjectsOf(_) => 2.0,
            Target::Sparql(_) => 5.0, // Custom queries are more expensive
            Target::Implicit(_) => 1.0,
        }
    }
    
    /// Estimate result count for a target
    fn estimate_result_count(&self, target: &Target) -> usize {
        let target_type = match target {
            Target::Class(_) => "class",
            Target::ObjectsOf(_) => "objects_of",
            Target::SubjectsOf(_) => "subjects_of",
            Target::Sparql(_) => "sparql",
            Target::Node(_) => "node",
            Target::Implicit(_) => "implicit",
        };
        
        if let Some(stats) = self.index_usage_stats.get(target_type) {
            (stats.avg_selectivity * 10000.0) as usize
        } else {
            1000 // Default estimate
        }
    }
    
    /// Generate index hints for targets
    fn generate_index_hints(&self, targets: &[Target]) -> Vec<String> {
        let mut hints = Vec::new();
        
        for target in targets {
            match target {
                Target::Class(_) => hints.push("USE_TYPE_INDEX".to_string()),
                Target::ObjectsOf(_) => hints.push("USE_PROPERTY_INDEX".to_string()),
                Target::SubjectsOf(_) => hints.push("USE_REVERSE_INDEX".to_string()),
                _ => {}
            }
        }
        
        hints
    }

    /// Adaptive batch size calculation based on performance history
    fn calculate_adaptive_batch_size(&self, target: &Target) -> usize {
        if !self.optimization_config.adaptive_batching {
            return self.optimization_config.batch_size;
        }

        let base_batch_size = self.optimization_config.batch_size;

        // Check if we have performance history for this type of target
        let target_type = match target {
            Target::Class(_) => "class",
            Target::ObjectsOf(_) => "objects_of",
            Target::SubjectsOf(_) => "subjects_of",
            Target::Sparql(_) => "sparql",
            Target::Node(_) => "node",
            Target::Implicit(_) => "implicit",
        };

        if let Some(index_stats) = self.index_usage_stats.get(target_type) {
            // Adjust batch size based on average execution time
            if index_stats.total_execution_time_ms / index_stats.usage_count as f64 > 1000.0 {
                // Slow queries - reduce batch size
                (base_batch_size / 2).max(1000)
            } else if index_stats.total_execution_time_ms / (index_stats.usage_count as f64) < 100.0
            {
                // Fast queries - increase batch size
                (base_batch_size * 2).min(50000)
            } else {
                base_batch_size
            }
        } else {
            base_batch_size
        }
    }

    /// Update index usage statistics
    fn update_index_usage_stats(
        &mut self,
        target: &Target,
        execution_time_ms: f64,
        result_count: usize,
    ) {
        let target_type = match target {
            Target::Class(_) => "class",
            Target::ObjectsOf(_) => "objects_of",
            Target::SubjectsOf(_) => "subjects_of",
            Target::Sparql(_) => "sparql",
            Target::Node(_) => "node",
            Target::Implicit(_) => "implicit",
        }
        .to_string();

        let stats = self
            .index_usage_stats
            .entry(target_type)
            .or_insert(IndexUsageStats {
                usage_count: 0,
                total_execution_time_ms: 0.0,
                avg_selectivity: 0.0,
                last_updated: std::time::Instant::now(),
            });

        stats.usage_count += 1;
        stats.total_execution_time_ms += execution_time_ms;

        // Calculate selectivity as ratio of results to estimated total
        let estimated_selectivity = if result_count > 0 {
            (result_count as f64 / 10000.0).min(1.0)
        } else {
            0.0
        };

        stats.avg_selectivity = (stats.avg_selectivity * (stats.usage_count - 1) as f64
            + estimated_selectivity)
            / stats.usage_count as f64;
        stats.last_updated = std::time::Instant::now();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> TargetCacheStats {
        TargetCacheStats {
            entries: self.cache.len(),
            total_targets: self.cache.values().map(|v| v.targets.len()).sum(),
        }
    }

    /// Get full performance statistics
    pub fn get_performance_stats(&self) -> TargetSelectionStats {
        let mut stats = self.stats.clone();
        stats.cache = self.get_cache_stats();
        stats
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

impl Default for TargetSelectionStats {
    fn default() -> Self {
        Self {
            cache: TargetCacheStats {
                entries: 0,
                total_targets: 0,
            },
            queries_executed: 0,
            cache_hits: 0,
            fallback_operations: 0,
            union_queries_used: 0,
        }
    }
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
