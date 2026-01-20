//! Target selector implementation
//!
//! This module contains the TargetSelector struct and its implementation for efficient
//! target node selection according to SHACL specification.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use oxirs_core::{model::Term, Store};

use super::optimization::*;
use super::types::*;
use crate::{Result, ShaclError};

/// Format a term for use in SPARQL queries
fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(format!("_:{}", node.as_str())),
        Term::Literal(literal) => {
            // Format literal with proper escaping and datatype/language tags
            let value = literal.value().replace('\\', "\\\\").replace('"', "\\\"");

            let datatype = literal.datatype();
            if datatype.as_str() == "http://www.w3.org/2001/XMLSchema#string" {
                // Simple string literals don't need datatype annotation
                Ok(format!("\"{value}\""))
            } else {
                Ok(format!("\"{}\"^^<{}>", value, datatype.as_str()))
            }
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
        Term::QuotedTriple(_) => Err(ShaclError::TargetSelection(
            "Quoted triples not supported in target selection queries".to_string(),
        )),
    }
}

/// Cached target result
#[derive(Debug, Clone)]
struct CachedTargetResult {
    /// Target nodes
    nodes: HashSet<Term>,
    /// Cache timestamp
    cached_at: Instant,
    /// Cache statistics
    stats: CacheStats,
}

/// Target selector for finding nodes that match target definitions
#[derive(Debug)]
pub struct TargetSelector {
    /// Cache for target results to improve performance
    cache: HashMap<String, CachedTargetResult>,
    /// Optimization settings
    optimization_config: TargetOptimizationConfig,
    /// Performance statistics
    stats: TargetSelectionStats,
    /// Query plan cache for SPARQL targets
    query_plan_cache: HashMap<String, QueryPlan>,
    /// Index usage statistics for adaptive optimization
    index_usage_stats: HashMap<String, IndexUsageStats>,
}

impl Default for TargetSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl TargetSelector {
    /// Create a new target selector
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            optimization_config: TargetOptimizationConfig::default(),
            stats: TargetSelectionStats::default(),
            query_plan_cache: HashMap::new(),
            index_usage_stats: HashMap::new(),
        }
    }

    /// Create a new target selector with custom optimization config
    pub fn with_config(config: TargetOptimizationConfig) -> Self {
        Self {
            cache: HashMap::new(),
            optimization_config: config,
            stats: TargetSelectionStats::default(),
            query_plan_cache: HashMap::new(),
            index_usage_stats: HashMap::new(),
        }
    }

    /// Generate efficient SPARQL query for target selection
    pub fn generate_target_query(
        &self,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<String> {
        self.generate_optimized_target_query(
            target,
            graph_name,
            &QueryOptimizationOptions::default(),
        )
    }

    /// Generate optimized SPARQL query for target selection with custom optimization options
    pub fn generate_optimized_target_query(
        &self,
        target: &Target,
        graph_name: Option<&str>,
        options: &QueryOptimizationOptions,
    ) -> Result<String> {
        let mut query = self.generate_basic_target_query(target, graph_name)?;

        // Apply optimizations
        if options.use_index_hints {
            query = self.add_index_hints(query, target)?;
        }

        if let Some(limit) = options.limit {
            query = format!("{query} LIMIT {limit}");
        }

        if options.deterministic_results {
            query = self.add_deterministic_ordering(query)?;
        }

        if options.include_performance_hints {
            query = self.add_performance_hints(query, target)?;
        }

        Ok(query)
    }

    /// Generate basic SPARQL query for target selection (internal method)
    fn generate_basic_target_query(
        &self,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<String> {
        match target {
            Target::Class(class_iri) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{graph}> {{")
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .{} }}",
                    graph_clause,
                    class_iri.as_str(),
                    close_clause
                ))
            }
            Target::Node(node) => {
                // For specific nodes, bind the node to ?target
                let node_str = format_term_for_sparql(node)?;
                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ BIND({node_str} AS ?target) }}"
                ))
            }
            Target::ObjectsOf(property) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{graph}> {{")
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?s <{}> ?target .{} }}",
                    graph_clause,
                    property.as_str(),
                    close_clause
                ))
            }
            Target::SubjectsOf(property) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{graph}> {{")
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <{}> ?o .{} }}",
                    graph_clause,
                    property.as_str(),
                    close_clause
                ))
            }
            Target::Sparql(sparql_target) => {
                // Return the user-provided SPARQL query with optional prefixes
                let mut query = String::new();
                if let Some(prefixes) = &sparql_target.prefixes {
                    query.push_str(prefixes);
                    query.push('\n');
                }
                query.push_str(&sparql_target.query);
                Ok(query)
            }
            Target::Implicit(class_iri) => {
                // Same as class target
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{graph}> {{")
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .{} }}",
                    graph_clause,
                    class_iri.as_str(),
                    close_clause
                ))
            }

            // Complex target combinations
            Target::Union(union_target) => {
                self.generate_union_query(&union_target.targets, graph_name)
            }

            Target::Intersection(intersection_target) => {
                self.generate_intersection_query(&intersection_target.targets, graph_name)
            }

            Target::Difference(difference_target) => self.generate_difference_query(
                &difference_target.primary_target,
                &difference_target.exclusion_target,
                graph_name,
            ),

            Target::Conditional(conditional_target) => self.generate_conditional_query(
                &conditional_target.base_target,
                &conditional_target.condition,
                conditional_target.context.as_ref(),
                graph_name,
            ),

            Target::Hierarchical(hierarchical_target) => self.generate_hierarchical_query(
                &hierarchical_target.root_target,
                &hierarchical_target.relationship,
                hierarchical_target.max_depth,
                hierarchical_target.include_intermediate,
                graph_name,
            ),

            Target::PathBased(path_target) => self.generate_path_based_query(
                &path_target.start_target,
                &path_target.path,
                &path_target.direction,
                &path_target.filters,
                graph_name,
            ),
        }
    }

    /// Select all target nodes for a given target definition
    pub fn select_targets(
        &mut self,
        store: &dyn Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let start_time = Instant::now();
        let cache_key = self.create_cache_key(target, graph_name);

        // Check cache first
        if self.optimization_config.enable_caching {
            if let Some(cached_result) = self.cache.get(&cache_key) {
                let cache_age = cached_result.cached_at.elapsed();
                if cache_age.as_secs() <= self.optimization_config.cache_ttl {
                    // Clone the nodes before updating statistics to avoid borrow conflict
                    let nodes: Vec<_> = cached_result.nodes.iter().cloned().collect();

                    // Update statistics
                    self.stats.total_evaluations += 1;
                    self.record_cache_hit();

                    return Ok(nodes);
                }
            }
        }

        // Record cache miss
        self.record_cache_miss();

        // Execute target selection
        let result = self.execute_target_selection(store, target, graph_name)?;

        // Cache the result if beneficial
        if self.optimization_config.enable_caching && self.should_cache_result(&result) {
            let cached_result = CachedTargetResult {
                nodes: result.iter().cloned().collect(),
                cached_at: Instant::now(),
                stats: CacheStats {
                    hits: 0,
                    misses: 1,
                    avg_query_time: start_time.elapsed(),
                },
            };

            // Manage cache size with intelligent eviction
            self.manage_cache_size(&cache_key, cached_result);
        }

        // Update statistics
        self.update_execution_statistics(start_time.elapsed(), result.len());

        Ok(result)
    }

    // Helper methods (these would contain the actual implementation logic)

    fn create_cache_key(&self, target: &Target, graph_name: Option<&str>) -> String {
        format!("{target:?}_{graph_name:?}")
    }

    fn record_cache_hit(&mut self) {
        self.stats.total_evaluations += 1;
        // Update cache hit rate
        let total_cache_operations = self
            .cache
            .values()
            .map(|c| c.stats.hits + c.stats.misses)
            .sum::<usize>();
        let total_hits = self.cache.values().map(|c| c.stats.hits).sum::<usize>() + 1;

        if total_cache_operations > 0 {
            self.stats.cache_hit_rate = total_hits as f64 / total_cache_operations as f64;
        }
    }

    fn record_cache_miss(&mut self) {
        self.stats.total_evaluations += 1;
        // Update cache hit rate
        let total_cache_operations = self
            .cache
            .values()
            .map(|c| c.stats.hits + c.stats.misses)
            .sum::<usize>();
        let total_hits = self.cache.values().map(|c| c.stats.hits).sum::<usize>();

        if total_cache_operations > 0 {
            self.stats.cache_hit_rate = total_hits as f64 / total_cache_operations as f64;
        }
    }

    fn should_cache_result(&self, result: &[Term]) -> bool {
        // Implementation for determining if a result should be cached
        result.len() < 10000 // Simple heuristic
    }

    fn manage_cache_size(&mut self, cache_key: &str, cached_result: CachedTargetResult) {
        self.cache.insert(cache_key.to_string(), cached_result);

        // Simple LRU eviction if cache is too large
        if self.cache.len() > self.optimization_config.max_cache_size {
            // Find oldest entry and remove it
            if let Some(oldest_key) = self
                .cache
                .iter()
                .min_by_key(|(_, v)| v.cached_at)
                .map(|(k, _)| k.clone())
            {
                self.cache.remove(&oldest_key);
            }
        }
    }

    fn update_execution_statistics(&mut self, duration: std::time::Duration, _result_count: usize) {
        self.stats.total_evaluations += 1;
        self.stats.total_time += duration;

        if self.stats.total_evaluations > 0 {
            self.stats.avg_evaluation_time =
                self.stats.total_time / self.stats.total_evaluations as u32;
        }
    }

    fn execute_target_selection(
        &mut self,
        store: &dyn Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::NamedNode;
        use oxirs_core::{Object, Predicate, Subject};

        tracing::trace!("execute_target_selection: processing target {target:?}");

        match target {
            Target::Class(class_iri) => {
                eprintln!(
                    "DEBUG execute_target_selection: finding instances of class {}",
                    class_iri.as_str()
                );

                // Find all nodes that have rdf:type <class>
                let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid rdf:type IRI: {e}"))
                    })?;

                let predicate: Predicate = rdf_type.into();
                let object: Object = class_iri.clone().into();

                // Query for all triples: ?subject rdf:type <class>
                let quads = if let Some(graph) = graph_name {
                    let graph_name = NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {e}"))
                    })?;
                    let graph_name = oxirs_core::model::GraphName::NamedNode(graph_name);
                    store.find_quads(None, Some(&predicate), Some(&object), Some(&graph_name))?
                } else {
                    store.find_quads(None, Some(&predicate), Some(&object), None)?
                };

                let mut target_nodes = Vec::new();
                for quad in quads {
                    match quad.subject() {
                        Subject::NamedNode(nn) => target_nodes.push(Term::NamedNode(nn.clone())),
                        Subject::BlankNode(bn) => target_nodes.push(Term::BlankNode(bn.clone())),
                        _ => {
                            // Skip other subject types (e.g., RDF-star triples if supported)
                        }
                    }
                }

                eprintln!(
                    "DEBUG execute_target_selection: found {} instances of class {}",
                    target_nodes.len(),
                    class_iri.as_str()
                );
                for (i, node) in target_nodes.iter().enumerate() {
                    tracing::trace!("execute_target_selection: instance[{i}] = {node:?}");
                }

                Ok(target_nodes)
            }
            Target::Node(node) => {
                tracing::trace!("execute_target_selection: specific node target: {node:?}");
                // For specific nodes, just return the node itself
                Ok(vec![node.clone()])
            }
            Target::ObjectsOf(property) => {
                eprintln!(
                    "DEBUG execute_target_selection: finding objects of property {}",
                    property.as_str()
                );

                let predicate: Predicate = property.clone().into();

                let quads = if let Some(graph) = graph_name {
                    let graph_name = NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {e}"))
                    })?;
                    let graph_name = oxirs_core::model::GraphName::NamedNode(graph_name);
                    store.find_quads(None, Some(&predicate), None, Some(&graph_name))?
                } else {
                    store.find_quads(None, Some(&predicate), None, None)?
                };

                let mut target_nodes = Vec::new();
                for quad in quads {
                    match quad.object() {
                        Object::NamedNode(nn) => target_nodes.push(Term::NamedNode(nn.clone())),
                        Object::BlankNode(bn) => target_nodes.push(Term::BlankNode(bn.clone())),
                        Object::Literal(lit) => target_nodes.push(Term::Literal(lit.clone())),
                        _ => {
                            // Skip other object types (e.g., RDF-star triples if supported)
                        }
                    }
                }

                eprintln!(
                    "DEBUG execute_target_selection: found {} objects of property {}",
                    target_nodes.len(),
                    property.as_str()
                );
                Ok(target_nodes)
            }
            Target::SubjectsOf(property) => {
                eprintln!(
                    "DEBUG execute_target_selection: finding subjects of property {}",
                    property.as_str()
                );

                let predicate: Predicate = property.clone().into();

                let quads = if let Some(graph) = graph_name {
                    let graph_name = NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {e}"))
                    })?;
                    let graph_name = oxirs_core::model::GraphName::NamedNode(graph_name);
                    store.find_quads(None, Some(&predicate), None, Some(&graph_name))?
                } else {
                    store.find_quads(None, Some(&predicate), None, None)?
                };

                let mut target_nodes = Vec::new();
                for quad in quads {
                    match quad.subject() {
                        Subject::NamedNode(nn) => target_nodes.push(Term::NamedNode(nn.clone())),
                        Subject::BlankNode(bn) => target_nodes.push(Term::BlankNode(bn.clone())),
                        _ => {
                            // Skip other subject types (e.g., RDF-star triples if supported)
                        }
                    }
                }

                eprintln!(
                    "DEBUG execute_target_selection: found {} subjects of property {}",
                    target_nodes.len(),
                    property.as_str()
                );
                Ok(target_nodes)
            }
            Target::Implicit(class_iri) => {
                eprintln!(
                    "DEBUG execute_target_selection: implicit class target for {}",
                    class_iri.as_str()
                );
                // Implicit targets work like class targets
                self.execute_target_selection(store, &Target::Class(class_iri.clone()), graph_name)
            }
            Target::Sparql(_sparql_target) => {
                tracing::trace!("execute_target_selection: executing SPARQL target query");
                // For SPARQL targets, we need to execute the query against the store
                // This is a placeholder - actual SPARQL execution would be needed
                Ok(Vec::new())
            }
            Target::Union(union_target) => {
                eprintln!(
                    "DEBUG execute_target_selection: executing union target with {} targets",
                    union_target.targets.len()
                );
                let mut all_nodes = HashSet::new();
                for target in &union_target.targets {
                    let nodes = self.execute_target_selection(store, target, graph_name)?;
                    all_nodes.extend(nodes);
                }
                Ok(all_nodes.into_iter().collect())
            }
            Target::Intersection(intersection_target) => {
                eprintln!(
                    "DEBUG execute_target_selection: executing intersection target with {} targets",
                    intersection_target.targets.len()
                );
                if intersection_target.targets.is_empty() {
                    return Ok(Vec::new());
                }

                // Start with first target's results
                let mut result_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &intersection_target.targets[0], graph_name)?
                    .into_iter()
                    .collect();

                // Intersect with remaining targets
                for target in &intersection_target.targets[1..] {
                    let nodes: HashSet<Term> = self
                        .execute_target_selection(store, target, graph_name)?
                        .into_iter()
                        .collect();
                    result_nodes.retain(|node| nodes.contains(node));
                    if result_nodes.is_empty() {
                        break; // Early exit if intersection is empty
                    }
                }

                Ok(result_nodes.into_iter().collect())
            }
            Target::Difference(difference_target) => {
                tracing::trace!("execute_target_selection: executing difference target");
                let primary_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &difference_target.primary_target, graph_name)?
                    .into_iter()
                    .collect();
                let exclusion_nodes: HashSet<Term> = self
                    .execute_target_selection(
                        store,
                        &difference_target.exclusion_target,
                        graph_name,
                    )?
                    .into_iter()
                    .collect();

                Ok(primary_nodes
                    .difference(&exclusion_nodes)
                    .cloned()
                    .collect())
            }
            Target::Conditional(conditional_target) => {
                tracing::trace!("execute_target_selection: executing conditional target");

                // First get nodes from base target
                let base_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &conditional_target.base_target, graph_name)?
                    .into_iter()
                    .collect();

                // Filter nodes based on condition
                let mut filtered_nodes = Vec::new();
                for node in base_nodes {
                    if self.evaluate_target_condition(
                        store,
                        &node,
                        &conditional_target.condition,
                        graph_name,
                    )? {
                        filtered_nodes.push(node);
                    }
                }

                Ok(filtered_nodes)
            }
            Target::Hierarchical(hierarchical_target) => {
                tracing::trace!("execute_target_selection: executing hierarchical target");

                // Get root nodes from root target
                let root_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &hierarchical_target.root_target, graph_name)?
                    .into_iter()
                    .collect();

                // Traverse hierarchy from root nodes
                let hierarchical_nodes = self.traverse_hierarchy(
                    store,
                    &root_nodes,
                    &hierarchical_target.relationship,
                    hierarchical_target.max_depth,
                    hierarchical_target.include_intermediate,
                    graph_name,
                )?;

                Ok(hierarchical_nodes)
            }
            Target::PathBased(path_target) => {
                tracing::trace!("execute_target_selection: executing path-based target");

                // Get starting nodes from start target
                let start_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &path_target.start_target, graph_name)?
                    .into_iter()
                    .collect();

                // Follow property path from start nodes
                let path_nodes = self.follow_property_path(
                    store,
                    &start_nodes,
                    &path_target.path,
                    &path_target.direction,
                    &path_target.filters,
                    graph_name,
                )?;

                Ok(path_nodes)
            }
        }
    }

    // Complex query generation methods

    fn generate_union_query(&self, targets: &[Target], graph_name: Option<&str>) -> Result<String> {
        if targets.is_empty() {
            return Ok("SELECT DISTINCT ?target WHERE { }".to_string());
        }

        let mut union_parts = Vec::new();

        for target in targets {
            let individual_query = self.generate_target_query(target, graph_name)?;

            // Extract the WHERE clause from the individual query
            if let Some(where_start) = individual_query.find("WHERE {") {
                let where_clause = &individual_query[where_start + 7..]; // Skip "WHERE {"
                if let Some(where_end) = where_clause.rfind('}') {
                    let where_content = &where_clause[..where_end].trim();
                    if !where_content.is_empty() {
                        union_parts.push(format!("  {{ {where_content} }}"));
                    }
                }
            }
        }

        if union_parts.is_empty() {
            return Ok("SELECT DISTINCT ?target WHERE { }".to_string());
        }

        Ok(format!(
            "SELECT DISTINCT ?target WHERE {{\n{}\n}}",
            union_parts.join("\n  UNION\n")
        ))
    }

    fn generate_intersection_query(
        &self,
        targets: &[Target],
        graph_name: Option<&str>,
    ) -> Result<String> {
        if targets.is_empty() {
            return Ok("SELECT DISTINCT ?target WHERE { }".to_string());
        }

        if targets.len() == 1 {
            return self.generate_target_query(&targets[0], graph_name);
        }

        let mut constraints = Vec::new();

        for (index, target) in targets.iter().enumerate() {
            let subquery_var = format!("?target_{index}");
            let individual_query = self.generate_target_query(target, graph_name)?;

            // Extract the WHERE clause and adapt it for intersection
            if let Some(where_start) = individual_query.find("WHERE {") {
                let where_clause = &individual_query[where_start + 7..];
                if let Some(where_end) = where_clause.rfind('}') {
                    let where_content = &where_clause[..where_end].trim();
                    if !where_content.is_empty() {
                        // Replace ?target with our indexed variable
                        let adapted_content = where_content.replace("?target", &subquery_var);
                        constraints.push(adapted_content.to_string());

                        // Add equality constraint (except for the first one)
                        if index > 0 {
                            constraints.push(format!("FILTER(?target = {subquery_var})"));
                        }
                    }
                }
            }
        }

        // Bind the first target variable to ?target
        constraints.push("BIND(?target_0 AS ?target)".to_string());

        Ok(format!(
            "SELECT DISTINCT ?target WHERE {{\n  {}\n}}",
            constraints.join("\n  ")
        ))
    }

    fn generate_difference_query(
        &self,
        primary: &Target,
        exclusion: &Target,
        graph_name: Option<&str>,
    ) -> Result<String> {
        let primary_query = self.generate_target_query(primary, graph_name)?;
        let exclusion_query = self.generate_target_query(exclusion, graph_name)?;

        // Extract WHERE clauses
        let primary_where = self.extract_where_clause(&primary_query)?;
        let exclusion_where = self.extract_where_clause(&exclusion_query)?;

        let query = format!(
            r#"SELECT DISTINCT ?target WHERE {{
  {primary_where}
  FILTER NOT EXISTS {{
    {exclusion_where}
  }}
}}"#
        );

        Ok(query)
    }

    fn generate_conditional_query(
        &self,
        base: &Target,
        condition: &TargetCondition,
        context: Option<&TargetContext>,
        graph_name: Option<&str>,
    ) -> Result<String> {
        let base_query = self.generate_target_query(base, graph_name)?;
        let base_where = self.extract_where_clause(&base_query)?;

        let condition_clause = match condition {
            TargetCondition::SparqlAsk { query, prefixes } => {
                let prefixes_str = prefixes.as_deref().unwrap_or("");
                format!("{prefixes_str}\nFILTER EXISTS {{ {query} }}")
            }
            TargetCondition::PropertyExists {
                property,
                direction,
            } => match direction {
                PropertyDirection::Subject => {
                    format!("?target <{}> ?conditionValue", property.as_str())
                }
                PropertyDirection::Object => {
                    format!("?conditionValue <{}> ?target", property.as_str())
                }
                PropertyDirection::Either => format!(
                    "{{ ?target <{}> ?conditionValue }} UNION {{ ?conditionValue <{}> ?target }}",
                    property.as_str(),
                    property.as_str()
                ),
            },
            TargetCondition::PropertyValue {
                property,
                value,
                direction,
            } => {
                let value_str = format_term_for_sparql(value)?;
                match direction {
                    PropertyDirection::Subject => {
                        format!("?target <{}> {}", property.as_str(), value_str)
                    }
                    PropertyDirection::Object => {
                        format!("{} <{}> ?target", value_str, property.as_str())
                    }
                    PropertyDirection::Either => format!(
                        "{{ ?target <{}> {} }} UNION {{ {} <{}> ?target }}",
                        property.as_str(),
                        value_str,
                        value_str,
                        property.as_str()
                    ),
                }
            }
            TargetCondition::HasType { class_iri } => {
                format!(
                    "?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}>",
                    class_iri.as_str()
                )
            }
            TargetCondition::Cardinality {
                property,
                min_count,
                max_count,
                direction,
            } => {
                let property_pattern = match direction {
                    PropertyDirection::Subject => format!("?target <{}> ?cardinalityValue", property.as_str()),
                    PropertyDirection::Object => format!("?cardinalityValue <{}> ?target", property.as_str()),
                    PropertyDirection::Either => format!(
                        "{{ ?target <{}> ?cardinalityValue }} UNION {{ ?cardinalityValue <{}> ?target }}",
                        property.as_str(),
                        property.as_str()
                    ),
                };

                let mut constraints = vec![property_pattern];

                if let Some(min) = min_count {
                    constraints.push(format!(
                        "HAVING(COUNT(DISTINCT ?cardinalityValue) >= {min})"
                    ));
                }
                if let Some(max) = max_count {
                    constraints.push(format!(
                        "HAVING(COUNT(DISTINCT ?cardinalityValue) <= {max})"
                    ));
                }

                format!(
                    "{{ SELECT ?target WHERE {{ {} }} GROUP BY ?target {} }}",
                    constraints[0],
                    constraints[1..].join(" ")
                )
            }
        };

        let context_clause = if let Some(ctx) = context {
            // Add any custom bindings from the context
            let mut bindings_clauses = Vec::new();
            for (var, value) in &ctx.bindings {
                let value_str = format_term_for_sparql(value)?;
                bindings_clauses.push(format!("BIND({value_str} AS ?{var})"));
            }

            let binding_section = if bindings_clauses.is_empty() {
                String::new()
            } else {
                format!("{}\n  ", bindings_clauses.join("\n  "))
            };

            format!("{binding_section}{condition_clause}")
        } else {
            condition_clause
        };

        let query = format!(
            r#"SELECT DISTINCT ?target WHERE {{
  {base_where}
  {context_clause}
}}"#
        );

        Ok(query)
    }

    fn generate_hierarchical_query(
        &self,
        root: &Target,
        relationship: &HierarchicalRelationship,
        max_depth: i32,
        include_intermediate: bool,
        graph_name: Option<&str>,
    ) -> Result<String> {
        // For simple cases where the root is a direct node/class, we can extract the IRI directly
        let root_iri = match root {
            Target::Class(class_iri) => Some(class_iri.as_str()),
            Target::Node(Term::NamedNode(node)) => Some(node.as_str()),
            _ => None,
        };

        let relationship_pattern = match relationship {
            HierarchicalRelationship::Property(property) => {
                format!("<{}>", property.as_str())
            }
            HierarchicalRelationship::InverseProperty(property) => {
                format!("^<{}>", property.as_str())
            }
            HierarchicalRelationship::SubclassOf => {
                "<http://www.w3.org/2000/01/rdf-schema#subClassOf>".to_string()
            }
            HierarchicalRelationship::SuperclassOf => {
                "^<http://www.w3.org/2000/01/rdf-schema#subClassOf>".to_string()
            }
            HierarchicalRelationship::SubpropertyOf => {
                "<http://www.w3.org/2000/01/rdf-schema#subPropertyOf>".to_string()
            }
            HierarchicalRelationship::SuperpropertyOf => {
                "^<http://www.w3.org/2000/01/rdf-schema#subPropertyOf>".to_string()
            }
            HierarchicalRelationship::TypeOf => {
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>".to_string()
            }
            HierarchicalRelationship::CustomPath(sparql_path) => sparql_path.clone(),
        };

        let depth_limit = if max_depth > 0 {
            max_depth.to_string()
        } else {
            "50".to_string() // Default reasonable limit
        };

        let path_pattern = if include_intermediate {
            format!("{relationship_pattern}*") // Include root and all descendants
        } else {
            format!("{relationship_pattern}+") // Only descendants, not root
        };

        let query = if let Some(iri) = root_iri {
            // Simple case: direct IRI binding
            let graph_wrapper = if let Some(graph) = graph_name {
                format!("GRAPH <{graph}> {{ ?target {path_pattern} ?root }}")
            } else {
                format!("?target {path_pattern} ?root")
            };

            format!(
                r#"SELECT DISTINCT ?target WHERE {{
  BIND(<{iri}> AS ?root)
  {graph_wrapper}
  # Recursive depth limited to {depth_limit}
}}"#
            )
        } else {
            // Complex case: use subquery for root resolution
            let root_query = self.generate_target_query(root, graph_name)?;
            let root_where = self.extract_where_clause(&root_query)?;

            let graph_wrapper = if let Some(graph) = graph_name {
                format!("GRAPH <{graph}> {{ ?target {path_pattern} ?root }}")
            } else {
                format!("?target {path_pattern} ?root")
            };

            format!(
                r#"SELECT DISTINCT ?target WHERE {{
  {{
    {}
    BIND(?target AS ?root)
  }}
  {}
  # Recursive depth limited to {}
}}"#,
                root_where.replace("?target", "?rootCandidate"),
                graph_wrapper,
                depth_limit
            )
        };

        Ok(query)
    }

    fn generate_path_based_query(
        &self,
        start: &Target,
        path: &crate::paths::PropertyPath,
        direction: &PathDirection,
        filters: &[PathFilter],
        graph_name: Option<&str>,
    ) -> Result<String> {
        let start_query = self.generate_target_query(start, graph_name)?;
        let start_where = self.extract_where_clause(&start_query)?;

        // Generate SPARQL property path syntax
        let sparql_path = path.to_sparql_path()?;

        let path_pattern = match direction {
            PathDirection::Forward => format!("?startNode {sparql_path} ?target"),
            PathDirection::Backward => format!("?target {sparql_path} ?startNode"),
            PathDirection::Both => format!(
                "{{ ?startNode {sparql_path} ?target }} UNION {{ ?target {sparql_path} ?startNode }}"
            ),
        };

        let graph_wrapper = if let Some(graph) = graph_name {
            format!("GRAPH <{graph}> {{ {path_pattern} }}")
        } else {
            path_pattern
        };

        let mut filter_clauses = Vec::new();
        for filter in filters {
            match filter {
                PathFilter::NodeType(node_type_filter) => match node_type_filter {
                    NodeTypeFilter::IriOnly => {
                        filter_clauses.push("FILTER(isIRI(?target))".to_string());
                    }
                    NodeTypeFilter::BlankNodeOnly => {
                        filter_clauses.push("FILTER(isBlank(?target))".to_string());
                    }
                    NodeTypeFilter::LiteralOnly => {
                        filter_clauses.push("FILTER(isLiteral(?target))".to_string());
                    }
                    NodeTypeFilter::InstanceOf(class_iri) => {
                        filter_clauses.push(format!(
                            "?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}>",
                            class_iri.as_str()
                        ));
                    }
                },
                PathFilter::PropertyValue { property, value } => {
                    let value_str = format_term_for_sparql(value)?;
                    filter_clauses.push(format!("?target <{}> {}", property.as_str(), value_str));
                }
                PathFilter::SparqlCondition {
                    condition,
                    prefixes,
                } => {
                    let prefixes_str = prefixes.as_deref().unwrap_or("");
                    if !prefixes_str.is_empty() {
                        filter_clauses.push(prefixes_str.to_string());
                    }
                    filter_clauses.push(condition.clone());
                }
            }
        }

        let filter_section = if filter_clauses.is_empty() {
            String::new()
        } else {
            format!("  {}", filter_clauses.join("\n  "))
        };

        let query = format!(
            r#"SELECT DISTINCT ?target WHERE {{
  {{
    {}
  }}
  {}
{}
}}"#,
            start_where.replace("?target", "?startNode"),
            graph_wrapper,
            filter_section
        );

        Ok(query)
    }

    /// Helper method to extract WHERE clause content from a SPARQL query
    fn extract_where_clause(&self, query: &str) -> Result<String> {
        if let Some(where_start) = query.find("WHERE {") {
            let where_clause = &query[where_start + 7..]; // Skip "WHERE {"
            if let Some(where_end) = where_clause.rfind('}') {
                let where_content = where_clause[..where_end].trim();
                return Ok(where_content.to_string());
            }
        }
        Err(ShaclError::TargetSelection(
            "Could not extract WHERE clause from query".to_string(),
        ))
    }

    /// Get statistics about target selection performance
    pub fn get_stats(&self) -> &TargetSelectionStats {
        &self.stats
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> TargetCacheStats {
        let total_hits: usize = self.cache.values().map(|c| c.stats.hits).sum();
        let total_misses: usize = self.cache.values().map(|c| c.stats.misses).sum();
        let total_operations = total_hits + total_misses;

        TargetCacheStats {
            hits: total_hits,
            misses: total_misses,
            hit_rate: if total_operations > 0 {
                total_hits as f64 / total_operations as f64
            } else {
                0.0
            },
            cache_size: self.cache.len(),
            memory_usage_bytes: self.cache.len() * std::mem::size_of::<CachedTargetResult>(),
        }
    }

    /// Clear all cached results
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Reset all statistics
    pub fn reset_stats(&mut self) {
        self.stats = TargetSelectionStats::default();
    }

    /// Add index hints to optimize query performance for large datasets
    fn add_index_hints(&self, query: String, target: &Target) -> Result<String> {
        // Add index hints based on target type
        match target {
            Target::Class(_) => {
                // For class targets, suggest using RDF type index
                Ok(format!("# Use RDF type index for class targets\n{query}"))
            }
            Target::ObjectsOf(property) | Target::SubjectsOf(property) => {
                // For property-based targets, suggest using property index
                Ok(format!(
                    "# Use property index for {}\n{}",
                    property.as_str(),
                    query
                ))
            }
            _ => Ok(query),
        }
    }

    /// Add deterministic ordering to ensure consistent results
    fn add_deterministic_ordering(&self, query: String) -> Result<String> {
        // Add ORDER BY clause if not already present
        if !query.contains("ORDER BY") {
            Ok(format!("{query} ORDER BY ?target"))
        } else {
            Ok(query)
        }
    }

    /// Add performance hints for query optimization
    fn add_performance_hints(&self, query: String, target: &Target) -> Result<String> {
        let mut hints = Vec::new();

        match target {
            Target::Class(_) => {
                hints
                    .push("# Consider using RDFS reasoning for subclass relationships".to_string());
            }
            Target::Union(union_target) => {
                if union_target.targets.len() > 5 {
                    hints.push("# Large union - consider query rewriting".to_string());
                }
            }
            Target::Intersection(intersection_target) => {
                if intersection_target.targets.len() > 3 {
                    hints
                        .push("# Complex intersection - consider selectivity ordering".to_string());
                }
            }
            _ => {}
        }

        if hints.is_empty() {
            Ok(query)
        } else {
            Ok(format!("{}\n{}", hints.join("\n"), query))
        }
    }

    /// Generate optimized batch query for multiple targets
    pub fn generate_batch_query(
        &self,
        targets: &[Target],
        graph_name: Option<&str>,
    ) -> Result<BatchQueryResult> {
        let start_time = Instant::now();
        let mut individual_queries = Vec::new();
        let mut total_estimated_cardinality = 0;

        // Generate individual queries
        for target in targets {
            let query = self.generate_optimized_target_query(
                target,
                graph_name,
                &QueryOptimizationOptions::default(),
            )?;
            let estimated_cardinality = self.estimate_target_cardinality(target);
            total_estimated_cardinality += estimated_cardinality;

            individual_queries.push(OptimizedQuery {
                sparql: query,
                estimated_cardinality,
                execution_strategy: ExecutionStrategy::Sequential,
                index_hints: vec![],
                optimization_time: Duration::from_millis(0),
            });
        }

        // Generate union query if beneficial
        let union_query = if targets.len() > 1 && self.optimization_config.use_union_optimization {
            Some(self.generate_union_query(targets, graph_name)?)
        } else {
            None
        };

        Ok(BatchQueryResult {
            individual_queries,
            union_query,
            total_estimated_cardinality,
            batch_optimization_time: start_time.elapsed(),
        })
    }

    /// Estimate cardinality for a target (simple heuristic)
    #[allow(clippy::only_used_in_recursion)]
    fn estimate_target_cardinality(&self, target: &Target) -> usize {
        match target {
            Target::Class(_) => 1000,     // Assume moderate class size
            Target::Node(_) => 1,         // Single node
            Target::ObjectsOf(_) => 500,  // Moderate property usage
            Target::SubjectsOf(_) => 500, // Moderate property usage
            Target::Union(union_target) => union_target
                .targets
                .iter()
                .map(|t| self.estimate_target_cardinality(t))
                .sum(),
            Target::Intersection(intersection_target) => {
                intersection_target
                    .targets
                    .iter()
                    .map(|t| self.estimate_target_cardinality(t))
                    .min()
                    .unwrap_or(0)
                    / 2
            }
            _ => 100, // Default estimate
        }
    }

    /// Evaluate a target condition for a specific node
    fn evaluate_target_condition(
        &self,
        store: &dyn Store,
        node: &Term,
        condition: &TargetCondition,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        match condition {
            TargetCondition::SparqlAsk { query, prefixes } => {
                // Execute SPARQL ASK query with node binding
                let mut final_query = query.clone();

                // Replace $this with the actual node
                let node_sparql = format_term_for_sparql(node)?;
                final_query = final_query.replace("$this", &node_sparql);

                // Add prefixes if provided
                if let Some(prefixes) = prefixes {
                    final_query = format!("{prefixes}\n{final_query}");
                }

                // For now, return true as we'd need SPARQL execution capability
                // TODO: Integrate with SPARQL engine when available
                tracing::warn!("SPARQL ASK condition evaluation not fully implemented yet");
                tracing::debug!("Would execute SPARQL query: {}", final_query);
                Ok(true)
            }
            TargetCondition::PropertyExists {
                property,
                direction,
            } => {
                // Check if the node has the specified property
                match direction {
                    PropertyDirection::Subject => {
                        // Check if node has property as subject
                        self.check_property_exists(store, node, property, true, graph_name)
                    }
                    PropertyDirection::Object => {
                        // Check if node has property as object
                        self.check_property_exists(store, node, property, false, graph_name)
                    }
                    PropertyDirection::Either => {
                        // Check both directions
                        let forward =
                            self.check_property_exists(store, node, property, true, graph_name)?;
                        let backward =
                            self.check_property_exists(store, node, property, false, graph_name)?;
                        Ok(forward || backward)
                    }
                }
            }
            TargetCondition::PropertyValue {
                property,
                value,
                direction,
            } => {
                // Check if the node has the specified property with the specified value
                match direction {
                    PropertyDirection::Subject => {
                        self.check_property_value(store, node, property, value, true, graph_name)
                    }
                    PropertyDirection::Object => {
                        self.check_property_value(store, node, property, value, false, graph_name)
                    }
                    PropertyDirection::Either => {
                        let forward = self
                            .check_property_value(store, node, property, value, true, graph_name)?;
                        let backward = self.check_property_value(
                            store, node, property, value, false, graph_name,
                        )?;
                        Ok(forward || backward)
                    }
                }
            }
            TargetCondition::HasType { class_iri } => {
                // Check if node is an instance of the specified class
                self.check_node_type(store, node, class_iri, graph_name)
            }
            TargetCondition::Cardinality {
                property,
                min_count,
                max_count,
                direction,
            } => {
                // Check if node satisfies cardinality constraints for the property
                let count =
                    self.count_property_values(store, node, property, direction, graph_name)?;

                let min_satisfied = min_count.map_or(true, |min| count >= min);
                let max_satisfied = max_count.map_or(true, |max| count <= max);

                Ok(min_satisfied && max_satisfied)
            }
        }
    }

    /// Traverse hierarchy from root nodes following the specified relationship
    fn traverse_hierarchy(
        &self,
        store: &dyn Store,
        root_nodes: &HashSet<Term>,
        relationship: &HierarchicalRelationship,
        max_depth: i32,
        include_intermediate: bool,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut result_nodes = Vec::new();
        let mut visited = HashSet::new();
        let mut current_level: HashSet<Term> = root_nodes.clone();
        let mut depth = 0;

        // Include root nodes if requested
        if include_intermediate || depth == max_depth || max_depth == 0 {
            result_nodes.extend(root_nodes.iter().cloned());
        }

        while !current_level.is_empty() && (max_depth == -1 || depth < max_depth) {
            let mut next_level = HashSet::new();

            for node in &current_level {
                if visited.contains(node) {
                    continue; // Avoid cycles
                }
                visited.insert(node.clone());

                let related_nodes =
                    self.get_related_nodes(store, node, relationship, graph_name)?;

                for related_node in related_nodes {
                    if !visited.contains(&related_node) {
                        next_level.insert(related_node.clone());

                        // Include this node if we want intermediate nodes or this is the final level
                        if include_intermediate || depth + 1 == max_depth || max_depth == -1 {
                            result_nodes.push(related_node);
                        }
                    }
                }
            }

            current_level = next_level;
            depth += 1;
        }

        Ok(result_nodes)
    }

    /// Follow property path from start nodes
    fn follow_property_path(
        &self,
        store: &dyn Store,
        start_nodes: &HashSet<Term>,
        path: &crate::paths::PropertyPath,
        direction: &PathDirection,
        filters: &[PathFilter],
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut result_nodes = Vec::new();

        for start_node in start_nodes {
            let path_results =
                self.evaluate_property_path(store, start_node, path, direction, graph_name)?;

            // Apply filters
            for result_node in path_results {
                if self.apply_path_filters(&result_node, filters, store, graph_name)? {
                    result_nodes.push(result_node);
                }
            }
        }

        Ok(result_nodes)
    }

    /// Helper methods for condition evaluation
    fn check_property_exists(
        &self,
        store: &dyn Store,
        node: &Term,
        property: &oxirs_core::model::NamedNode,
        forward: bool,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        use oxirs_core::{Object, Predicate, Subject};

        let predicate: Predicate = property.clone().into();

        // Check property existence using Store queries
        let quads = if forward {
            // Check if: node property ?o (node is subject)
            let subject_ref = match node {
                Term::NamedNode(n) => Subject::from(n.clone()),
                Term::BlankNode(n) => Subject::from(n.clone()),
                _ => return Ok(false),
            };

            if let Some(graph) = graph_name {
                let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                    ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                })?;
                let gn = Some(oxirs_core::GraphName::from(graph_node));
                store.find_quads(Some(&subject_ref), Some(&predicate), None, gn.as_ref())?
            } else {
                store.find_quads(Some(&subject_ref), Some(&predicate), None, None)?
            }
        } else {
            // Check if: ?s property node (node is object)
            let object_ref = match node {
                Term::NamedNode(n) => Object::from(n.clone()),
                Term::BlankNode(n) => Object::from(n.clone()),
                Term::Literal(lit) => Object::from(lit.clone()),
                _ => return Ok(false),
            };

            if let Some(graph) = graph_name {
                let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                    ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                })?;
                let gn = Some(oxirs_core::GraphName::from(graph_node));
                store.find_quads(None, Some(&predicate), Some(&object_ref), gn.as_ref())?
            } else {
                store.find_quads(None, Some(&predicate), Some(&object_ref), None)?
            }
        };

        Ok(!quads.is_empty())
    }

    fn check_property_value(
        &self,
        store: &dyn Store,
        node: &Term,
        property: &oxirs_core::model::NamedNode,
        value: &Term,
        forward: bool,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        use oxirs_core::{Object, Predicate, Subject};

        let predicate: Predicate = property.clone().into();

        // Check if the specific triple exists: node property value (or value property node)
        let quads = if forward {
            // Check if: node property value
            let subject_ref = match node {
                Term::NamedNode(n) => Subject::from(n.clone()),
                Term::BlankNode(n) => Subject::from(n.clone()),
                _ => return Ok(false),
            };

            let object_ref = match value {
                Term::NamedNode(n) => Object::from(n.clone()),
                Term::BlankNode(n) => Object::from(n.clone()),
                Term::Literal(lit) => Object::from(lit.clone()),
                _ => return Ok(false),
            };

            if let Some(graph) = graph_name {
                let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                    ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                })?;
                let gn = Some(oxirs_core::GraphName::from(graph_node));
                store.find_quads(
                    Some(&subject_ref),
                    Some(&predicate),
                    Some(&object_ref),
                    gn.as_ref(),
                )?
            } else {
                store.find_quads(
                    Some(&subject_ref),
                    Some(&predicate),
                    Some(&object_ref),
                    None,
                )?
            }
        } else {
            // Check if: value property node
            let subject_ref = match value {
                Term::NamedNode(n) => Subject::from(n.clone()),
                Term::BlankNode(n) => Subject::from(n.clone()),
                _ => return Ok(false),
            };

            let object_ref = match node {
                Term::NamedNode(n) => Object::from(n.clone()),
                Term::BlankNode(n) => Object::from(n.clone()),
                Term::Literal(lit) => Object::from(lit.clone()),
                _ => return Ok(false),
            };

            if let Some(graph) = graph_name {
                let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                    ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                })?;
                let gn = Some(oxirs_core::GraphName::from(graph_node));
                store.find_quads(
                    Some(&subject_ref),
                    Some(&predicate),
                    Some(&object_ref),
                    gn.as_ref(),
                )?
            } else {
                store.find_quads(
                    Some(&subject_ref),
                    Some(&predicate),
                    Some(&object_ref),
                    None,
                )?
            }
        };

        Ok(!quads.is_empty())
    }

    fn check_node_type(
        &self,
        store: &dyn Store,
        node: &Term,
        class_iri: &oxirs_core::model::NamedNode,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        use oxirs_core::{Object, Predicate, Subject};

        // Check if: node rdf:type class
        let subject_ref = match node {
            Term::NamedNode(n) => Subject::from(n.clone()),
            Term::BlankNode(n) => Subject::from(n.clone()),
            _ => return Ok(false),
        };

        let rdf_type =
            oxirs_core::model::NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid IRI: {}", e)))?;
        let predicate: Predicate = rdf_type.into();
        let object: Object = class_iri.clone().into();

        let quads = if let Some(graph) = graph_name {
            let graph_node = oxirs_core::model::NamedNode::new(graph)
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e)))?;
            let gn = Some(oxirs_core::GraphName::from(graph_node));
            store.find_quads(
                Some(&subject_ref),
                Some(&predicate),
                Some(&object),
                gn.as_ref(),
            )?
        } else {
            store.find_quads(Some(&subject_ref), Some(&predicate), Some(&object), None)?
        };

        Ok(!quads.is_empty())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn count_property_values(
        &self,
        store: &dyn Store,
        node: &Term,
        property: &oxirs_core::model::NamedNode,
        direction: &PropertyDirection,
        graph_name: Option<&str>,
    ) -> Result<usize> {
        use oxirs_core::{Object, Predicate, Subject};

        let predicate: Predicate = property.clone().into();

        let quads = match direction {
            PropertyDirection::Subject => {
                // node -> property -> ?object
                // Count values where the node has this property
                let subject_ref = match node {
                    Term::NamedNode(n) => Subject::from(n.clone()),
                    Term::BlankNode(n) => Subject::from(n.clone()),
                    _ => return Ok(0), // Literals can't be subjects
                };

                if let Some(graph) = graph_name {
                    let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                    })?;
                    let gn = Some(oxirs_core::GraphName::from(graph_node));
                    store.find_quads(Some(&subject_ref), Some(&predicate), None, gn.as_ref())?
                } else {
                    store.find_quads(Some(&subject_ref), Some(&predicate), None, None)?
                }
            }
            PropertyDirection::Object => {
                // ?subject -> property -> node
                // Count subjects that have this property pointing to our node
                let object_ref: Object = match node {
                    Term::NamedNode(n) => n.clone().into(),
                    Term::BlankNode(n) => n.clone().into(),
                    Term::Literal(lit) => lit.clone().into(),
                    _ => return Ok(0),
                };

                if let Some(graph) = graph_name {
                    let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                    })?;
                    let gn = Some(oxirs_core::GraphName::from(graph_node));
                    store.find_quads(None, Some(&predicate), Some(&object_ref), gn.as_ref())?
                } else {
                    store.find_quads(None, Some(&predicate), Some(&object_ref), None)?
                }
            }
            PropertyDirection::Either => {
                // Count both as subject and as object
                let as_subject = self.count_property_values(
                    store,
                    node,
                    property,
                    &PropertyDirection::Subject,
                    graph_name,
                )?;
                let as_object = self.count_property_values(
                    store,
                    node,
                    property,
                    &PropertyDirection::Object,
                    graph_name,
                )?;
                return Ok(as_subject + as_object);
            }
        };

        Ok(quads.len())
    }

    fn get_related_nodes(
        &self,
        store: &dyn Store,
        node: &Term,
        relationship: &HierarchicalRelationship,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::{Object, Predicate, Subject};

        match relationship {
            HierarchicalRelationship::Property(prop) => {
                // Follow property in forward direction: node -> prop -> ?related
                let subject_ref = match node {
                    Term::NamedNode(n) => Subject::from(n.clone()),
                    Term::BlankNode(n) => Subject::from(n.clone()),
                    _ => return Ok(Vec::new()), // Literals can't be subjects
                };

                let predicate: Predicate = prop.clone().into();

                let quads = if let Some(graph) = graph_name {
                    let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                    })?;
                    let gn = Some(oxirs_core::GraphName::from(graph_node));
                    store.find_quads(Some(&subject_ref), Some(&predicate), None, gn.as_ref())?
                } else {
                    store.find_quads(Some(&subject_ref), Some(&predicate), None, None)?
                };

                Ok(quads
                    .into_iter()
                    .map(|q| q.object().clone().into())
                    .collect())
            }
            HierarchicalRelationship::InverseProperty(prop) => {
                // Follow property in inverse direction: ?related -> prop -> node
                let object_ref: Object = match node {
                    Term::NamedNode(n) => n.clone().into(),
                    Term::BlankNode(n) => n.clone().into(),
                    Term::Literal(lit) => lit.clone().into(),
                    _ => return Ok(Vec::new()),
                };

                let predicate: Predicate = prop.clone().into();

                let quads = if let Some(graph) = graph_name {
                    let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                    })?;
                    let gn = Some(oxirs_core::GraphName::from(graph_node));
                    store.find_quads(None, Some(&predicate), Some(&object_ref), gn.as_ref())?
                } else {
                    store.find_quads(None, Some(&predicate), Some(&object_ref), None)?
                };

                Ok(quads
                    .into_iter()
                    .map(|q| q.subject().clone().into())
                    .collect())
            }
            HierarchicalRelationship::SubclassOf => {
                // Follow rdfs:subClassOf
                let subject_ref = match node {
                    Term::NamedNode(n) => Subject::from(n.clone()),
                    Term::BlankNode(n) => Subject::from(n.clone()),
                    _ => return Ok(Vec::new()),
                };

                let subclass_of = oxirs_core::model::NamedNode::new(
                    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                )
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid IRI: {}", e)))?;
                let predicate: Predicate = subclass_of.into();

                let quads = if let Some(graph) = graph_name {
                    let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                    })?;
                    let gn = Some(oxirs_core::GraphName::from(graph_node));
                    store.find_quads(Some(&subject_ref), Some(&predicate), None, gn.as_ref())?
                } else {
                    store.find_quads(Some(&subject_ref), Some(&predicate), None, None)?
                };

                Ok(quads
                    .into_iter()
                    .map(|q| q.object().clone().into())
                    .collect())
            }
            HierarchicalRelationship::TypeOf => {
                // Follow rdf:type
                let subject_ref = match node {
                    Term::NamedNode(n) => Subject::from(n.clone()),
                    Term::BlankNode(n) => Subject::from(n.clone()),
                    _ => return Ok(Vec::new()),
                };

                let rdf_type = oxirs_core::model::NamedNode::new(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                )
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid IRI: {}", e)))?;
                let predicate: Predicate = rdf_type.into();

                let quads = if let Some(graph) = graph_name {
                    let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                    })?;
                    let gn = Some(oxirs_core::GraphName::from(graph_node));
                    store.find_quads(Some(&subject_ref), Some(&predicate), None, gn.as_ref())?
                } else {
                    store.find_quads(Some(&subject_ref), Some(&predicate), None, None)?
                };

                Ok(quads
                    .into_iter()
                    .map(|q| q.object().clone().into())
                    .collect())
            }
            _ => {
                tracing::debug!(
                    "Hierarchical relationship type not yet supported: {:?}",
                    relationship
                );
                Ok(Vec::new())
            }
        }
    }

    fn evaluate_property_path(
        &self,
        _store: &dyn Store,
        _start_node: &Term,
        _path: &crate::paths::PropertyPath,
        _direction: &PathDirection,
        _graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // TODO: Integrate with property path evaluator
        tracing::warn!("Property path evaluation not fully implemented yet");
        Ok(Vec::new())
    }

    fn apply_path_filters(
        &self,
        node: &Term,
        filters: &[PathFilter],
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        // Apply all filters - node must pass all filters
        for filter in filters {
            if !self.evaluate_path_filter(node, filter, store, graph_name)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn evaluate_path_filter(
        &self,
        node: &Term,
        filter: &PathFilter,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        use oxirs_core::{Object, Predicate, Subject};

        match filter {
            PathFilter::NodeType(node_type_filter) => {
                // Filter based on node type
                match node_type_filter {
                    NodeTypeFilter::IriOnly => {
                        // Only accept IRI nodes (named nodes)
                        Ok(matches!(node, Term::NamedNode(_)))
                    }
                    NodeTypeFilter::BlankNodeOnly => {
                        // Only accept blank nodes
                        Ok(matches!(node, Term::BlankNode(_)))
                    }
                    NodeTypeFilter::LiteralOnly => {
                        // Only accept literals
                        Ok(matches!(node, Term::Literal(_)))
                    }
                    NodeTypeFilter::InstanceOf(class) => {
                        // Check if node is an instance of the specified class
                        // i.e., check if: node rdf:type class
                        let subject_ref = match node {
                            Term::NamedNode(n) => Subject::from(n.clone()),
                            Term::BlankNode(n) => Subject::from(n.clone()),
                            _ => return Ok(false), // Literals can't have types
                        };

                        let rdf_type = oxirs_core::model::NamedNode::new(
                            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        )
                        .map_err(|e| ShaclError::TargetSelection(format!("Invalid IRI: {}", e)))?;
                        let predicate: Predicate = rdf_type.into();
                        let object: Object = class.clone().into();

                        let quads = if let Some(graph) = graph_name {
                            let graph_node =
                                oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                                    ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                                })?;
                            let gn = Some(oxirs_core::GraphName::from(graph_node));
                            store.find_quads(
                                Some(&subject_ref),
                                Some(&predicate),
                                Some(&object),
                                gn.as_ref(),
                            )?
                        } else {
                            store.find_quads(
                                Some(&subject_ref),
                                Some(&predicate),
                                Some(&object),
                                None,
                            )?
                        };

                        Ok(!quads.is_empty())
                    }
                }
            }
            PathFilter::PropertyValue { property, value } => {
                // Check if node has specified property with specified value
                // i.e., check if: node property value
                let subject_ref = match node {
                    Term::NamedNode(n) => Subject::from(n.clone()),
                    Term::BlankNode(n) => Subject::from(n.clone()),
                    _ => return Ok(false), // Literals can't be subjects
                };

                let predicate: Predicate = property.clone().into();
                let object: Object = match value {
                    Term::NamedNode(n) => n.clone().into(),
                    Term::BlankNode(n) => n.clone().into(),
                    Term::Literal(lit) => lit.clone().into(),
                    _ => return Ok(false),
                };

                let quads = if let Some(graph) = graph_name {
                    let graph_node = oxirs_core::model::NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e))
                    })?;
                    let gn = Some(oxirs_core::GraphName::from(graph_node));
                    store.find_quads(
                        Some(&subject_ref),
                        Some(&predicate),
                        Some(&object),
                        gn.as_ref(),
                    )?
                } else {
                    store.find_quads(Some(&subject_ref), Some(&predicate), Some(&object), None)?
                };

                Ok(!quads.is_empty())
            }
            PathFilter::SparqlCondition {
                condition: _,
                prefixes: _,
            } => {
                // SPARQL condition filtering requires query execution
                // For now, log a debug message and return true (don't filter)
                // This would require integration with SPARQL query engine
                tracing::debug!(
                    "SPARQL condition path filter requires query execution - not yet implemented"
                );
                Ok(true)
            }
        }
    }
}
