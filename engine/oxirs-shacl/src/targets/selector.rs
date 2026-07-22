//! Target selector implementation
//!
//! This module contains the TargetSelector struct and its core public API.
//! Implementation is split across sibling modules:
//! - `selector_query.rs`  — SPARQL query generation
//! - `selector_eval.rs`   — Condition evaluation and hierarchy traversal

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use oxirs_core::{model::Term, Store};

use super::optimization::*;
use super::types::*;
use crate::{Result, ShaclError};

/// Format a term for use in SPARQL queries
pub(super) fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(format!("_:{}", node.as_str())),
        Term::Literal(literal) => {
            let value = literal.value().replace('\\', "\\\\").replace('"', "\\\"");

            let datatype = literal.datatype();
            if datatype.as_str() == "http://www.w3.org/2001/XMLSchema#string" {
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
    nodes: HashSet<Term>,
    cached_at: Instant,
    stats: CacheStats,
}

/// Target selector for finding nodes that match target definitions
#[derive(Debug)]
pub struct TargetSelector {
    cache: HashMap<String, CachedTargetResult>,
    optimization_config: TargetOptimizationConfig,
    stats: TargetSelectionStats,
    query_plan_cache: HashMap<String, QueryPlan>,
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

        if options.use_index_hints {
            query = super::selector_query::add_index_hints(query, target)?;
        }

        if let Some(limit) = options.limit {
            query = format!("{query} LIMIT {limit}");
        }

        if options.deterministic_results {
            query = super::selector_query::add_deterministic_ordering(query)?;
        }

        if options.include_performance_hints {
            query = super::selector_query::add_performance_hints(query, target)?;
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
                let mut query = String::new();
                if let Some(prefixes) = &sparql_target.prefixes {
                    query.push_str(prefixes);
                    query.push('\n');
                }
                query.push_str(&sparql_target.query);
                Ok(query)
            }
            Target::Implicit(class_iri) => {
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

            Target::Union(union_target) => {
                let query_fn = |t: &Target, gn: Option<&str>| self.generate_target_query(t, gn);
                super::selector_query::generate_union_query(
                    &union_target.targets,
                    graph_name,
                    &query_fn,
                )
            }

            Target::Intersection(intersection_target) => {
                let query_fn = |t: &Target, gn: Option<&str>| self.generate_target_query(t, gn);
                super::selector_query::generate_intersection_query(
                    &intersection_target.targets,
                    graph_name,
                    &query_fn,
                )
            }

            Target::Difference(difference_target) => {
                let query_fn = |t: &Target, gn: Option<&str>| self.generate_target_query(t, gn);
                super::selector_query::generate_difference_query(
                    &difference_target.primary_target,
                    &difference_target.exclusion_target,
                    graph_name,
                    &query_fn,
                )
            }

            Target::Conditional(conditional_target) => {
                let query_fn = |t: &Target, gn: Option<&str>| self.generate_target_query(t, gn);
                super::selector_query::generate_conditional_query(
                    &conditional_target.base_target,
                    &conditional_target.condition,
                    conditional_target.context.as_ref(),
                    graph_name,
                    &query_fn,
                )
            }

            Target::Hierarchical(hierarchical_target) => {
                let query_fn = |t: &Target, gn: Option<&str>| self.generate_target_query(t, gn);
                super::selector_query::generate_hierarchical_query(
                    &hierarchical_target.root_target,
                    &hierarchical_target.relationship,
                    hierarchical_target.max_depth,
                    hierarchical_target.include_intermediate,
                    graph_name,
                    &query_fn,
                )
            }

            Target::PathBased(path_target) => {
                let query_fn = |t: &Target, gn: Option<&str>| self.generate_target_query(t, gn);
                super::selector_query::generate_path_based_query(
                    &path_target.start_target,
                    &path_target.path,
                    &path_target.direction,
                    &path_target.filters,
                    graph_name,
                    &query_fn,
                )
            }
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

        if self.optimization_config.enable_caching {
            if let Some(cached_result) = self.cache.get(&cache_key) {
                let cache_age = cached_result.cached_at.elapsed();
                if cache_age.as_secs() <= self.optimization_config.cache_ttl {
                    let nodes: Vec<_> = cached_result.nodes.iter().cloned().collect();

                    self.stats.total_evaluations += 1;
                    self.record_cache_hit();

                    return Ok(nodes);
                }
            }
        }

        self.record_cache_miss();

        let result = self.execute_target_selection(store, target, graph_name)?;

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

            self.manage_cache_size(&cache_key, cached_result);
        }

        self.update_execution_statistics(start_time.elapsed(), result.len());

        Ok(result)
    }

    fn create_cache_key(&self, target: &Target, graph_name: Option<&str>) -> String {
        format!("{target:?}_{graph_name:?}")
    }

    fn record_cache_hit(&mut self) {
        self.stats.total_evaluations += 1;
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
        result.len() < 10000
    }

    fn manage_cache_size(&mut self, cache_key: &str, cached_result: CachedTargetResult) {
        self.cache.insert(cache_key.to_string(), cached_result);

        if self.cache.len() > self.optimization_config.max_cache_size {
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
                tracing::trace!(
                    "execute_target_selection: finding SHACL instances of class {}",
                    class_iri.as_str()
                );

                let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid rdf:type IRI: {e}"))
                    })?;

                let predicate: Predicate = rdf_type.into();

                let optional_graph = if let Some(graph) = graph_name {
                    let graph_name = NamedNode::new(graph).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid graph IRI: {e}"))
                    })?;
                    Some(oxirs_core::model::GraphName::NamedNode(graph_name))
                } else {
                    None
                };

                // A node is a "SHACL instance" of class_iri if it is rdf:type class_iri
                // directly, OR rdf:type of any class C such that C rdfs:subClassOf* class_iri
                // (transitive). This matches ClassConstraint::check_class_membership so that
                // sh:targetClass and sh:class agree on what counts as "instance of C".
                let closure =
                    crate::advanced_features::subclass_closure::build_rdfs_subclass_closure(store)?;

                let mut classes_to_match: HashSet<NamedNode> = HashSet::new();
                classes_to_match.insert(class_iri.clone());
                for (sub_class, superclasses) in &closure {
                    if superclasses.contains(class_iri) {
                        classes_to_match.insert(sub_class.clone());
                    }
                }

                let mut target_nodes = Vec::new();
                let mut seen: HashSet<Term> = HashSet::new();
                for class in &classes_to_match {
                    let object: Object = class.clone().into();

                    let quads = if let Some(graph) = &optional_graph {
                        store.find_quads(None, Some(&predicate), Some(&object), Some(graph))?
                    } else {
                        store.find_quads(None, Some(&predicate), Some(&object), None)?
                    };

                    for quad in quads {
                        let term = match quad.subject() {
                            Subject::NamedNode(nn) => Some(Term::NamedNode(nn.clone())),
                            Subject::BlankNode(bn) => Some(Term::BlankNode(bn.clone())),
                            _ => None,
                        };
                        if let Some(term) = term {
                            if seen.insert(term.clone()) {
                                target_nodes.push(term);
                            }
                        }
                    }
                }

                tracing::trace!(
                    "execute_target_selection: found {} instances of class {} ({} classes considered including subclasses)",
                    target_nodes.len(),
                    class_iri.as_str(),
                    classes_to_match.len()
                );
                for (i, node) in target_nodes.iter().enumerate() {
                    tracing::trace!("execute_target_selection: instance[{i}] = {node:?}");
                }

                Ok(target_nodes)
            }
            Target::Node(node) => {
                tracing::trace!("execute_target_selection: specific node target: {node:?}");
                Ok(vec![node.clone()])
            }
            Target::ObjectsOf(property) => {
                tracing::trace!(
                    "execute_target_selection: finding objects of property {}",
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
                        _ => {}
                    }
                }

                tracing::trace!(
                    "execute_target_selection: found {} objects of property {}",
                    target_nodes.len(),
                    property.as_str()
                );
                Ok(target_nodes)
            }
            Target::SubjectsOf(property) => {
                tracing::trace!(
                    "execute_target_selection: finding subjects of property {}",
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
                        _ => {}
                    }
                }

                tracing::trace!(
                    "execute_target_selection: found {} subjects of property {}",
                    target_nodes.len(),
                    property.as_str()
                );
                Ok(target_nodes)
            }
            Target::Implicit(class_iri) => {
                tracing::trace!(
                    "execute_target_selection: implicit class target for {}",
                    class_iri.as_str()
                );
                self.execute_target_selection(store, &Target::Class(class_iri.clone()), graph_name)
            }
            Target::Sparql(_sparql_target) => {
                tracing::trace!("execute_target_selection: executing SPARQL target query");

                let query = self.generate_basic_target_query(target, graph_name)?;

                let query_results = store.query(&query).map_err(|e| {
                    ShaclError::TargetSelection(format!(
                        "SPARQL target query execution failed: {e}"
                    ))
                })?;

                let mut target_nodes = Vec::new();
                match query_results.results() {
                    oxirs_core::rdf_store::QueryResults::Bindings(bindings) => {
                        for binding in bindings {
                            let bound_term = binding
                                .get("this")
                                .or_else(|| binding.get("target"))
                                .or_else(|| {
                                    binding.variables().next().and_then(|v| binding.get(v))
                                });

                            match bound_term {
                                Some(term) => target_nodes.push(term.clone()),
                                None => {
                                    return Err(ShaclError::TargetSelection(
                                        "SPARQL target query returned a solution with no bound variable"
                                            .to_string(),
                                    ));
                                }
                            }
                        }
                    }
                    oxirs_core::rdf_store::QueryResults::Boolean(_) => {
                        return Err(ShaclError::TargetSelection(
                            "sh:SPARQLTarget query must be a SELECT query returning focus nodes, not an ASK query"
                                .to_string(),
                        ));
                    }
                    oxirs_core::rdf_store::QueryResults::Graph(_) => {
                        return Err(ShaclError::TargetSelection(
                            "sh:SPARQLTarget query must be a SELECT query returning focus nodes, not a CONSTRUCT/DESCRIBE query"
                                .to_string(),
                        ));
                    }
                }

                tracing::trace!(
                    "execute_target_selection: SPARQL target query returned {} nodes",
                    target_nodes.len()
                );

                Ok(target_nodes)
            }
            Target::Union(union_target) => {
                tracing::trace!(
                    "execute_target_selection: executing union target with {} targets",
                    union_target.targets.len()
                );
                let mut all_nodes = HashSet::new();
                for t in &union_target.targets {
                    let nodes = self.execute_target_selection(store, t, graph_name)?;
                    all_nodes.extend(nodes);
                }
                Ok(all_nodes.into_iter().collect())
            }
            Target::Intersection(intersection_target) => {
                tracing::trace!(
                    "execute_target_selection: executing intersection target with {} targets",
                    intersection_target.targets.len()
                );
                if intersection_target.targets.is_empty() {
                    return Ok(Vec::new());
                }

                let mut result_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &intersection_target.targets[0], graph_name)?
                    .into_iter()
                    .collect();

                for t in &intersection_target.targets[1..] {
                    let nodes: HashSet<Term> = self
                        .execute_target_selection(store, t, graph_name)?
                        .into_iter()
                        .collect();
                    result_nodes.retain(|node| nodes.contains(node));
                    if result_nodes.is_empty() {
                        break;
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

                let base_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &conditional_target.base_target, graph_name)?
                    .into_iter()
                    .collect();

                let mut filtered_nodes = Vec::new();
                for node in base_nodes {
                    if super::selector_eval::evaluate_target_condition(
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

                let root_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &hierarchical_target.root_target, graph_name)?
                    .into_iter()
                    .collect();

                let hierarchical_nodes = super::selector_eval::traverse_hierarchy(
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

                let start_nodes: HashSet<Term> = self
                    .execute_target_selection(store, &path_target.start_target, graph_name)?
                    .into_iter()
                    .collect();

                let path_nodes = super::selector_eval::follow_property_path(
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

    /// Generate optimized batch query for multiple targets
    pub fn generate_batch_query(
        &self,
        targets: &[Target],
        graph_name: Option<&str>,
    ) -> Result<BatchQueryResult> {
        let optimized_query_fn = |t: &Target, gn: Option<&str>, opts: &QueryOptimizationOptions| {
            self.generate_optimized_target_query(t, gn, opts)
        };
        let union_query_fn = |ts: &[Target], gn: Option<&str>| {
            let query_fn = |t: &Target, g: Option<&str>| self.generate_target_query(t, g);
            super::selector_query::generate_union_query(ts, gn, &query_fn)
        };
        super::selector_query::generate_batch_query_impl(
            targets,
            graph_name,
            &self.optimization_config,
            &optimized_query_fn,
            &union_query_fn,
        )
    }

    /// Helper method to extract WHERE clause content from a SPARQL query
    fn extract_where_clause(&self, query: &str) -> Result<String> {
        super::selector_query::extract_where_clause(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{GraphName, NamedNode, Quad};
    use oxirs_core::ConcreteStore;

    fn nn(iri: &str) -> NamedNode {
        NamedNode::new(iri).expect("valid IRI")
    }

    /// Target::Sparql previously always returned an empty Vec (see
    /// `execute_target_selection`'s Sparql arm) instead of actually running
    /// the SPARQL SELECT query against the store.
    #[test]
    fn regression_sparql_target_selects_real_focus_nodes() {
        let store = ConcreteStore::new().expect("store");
        let invoice1 = nn("http://example.org/invoice1");
        let invoice2 = nn("http://example.org/invoice2");
        let not_invoice = nn("http://example.org/notAnInvoice");
        let rdf_type = nn("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        let invoice_class = nn("http://example.org/Invoice");

        for node in [&invoice1, &invoice2] {
            store
                .insert_quad(Quad::new(
                    node.clone(),
                    rdf_type.clone(),
                    invoice_class.clone(),
                    GraphName::DefaultGraph,
                ))
                .expect("insert invoice type");
        }
        store
            .insert_quad(Quad::new(
                not_invoice.clone(),
                rdf_type.clone(),
                nn("http://example.org/Other"),
                GraphName::DefaultGraph,
            ))
            .expect("insert other type");

        // Use the expanded rdf:type IRI rather than the `a` shorthand: this
        // exercises the same target-selection fix (a real, executed SPARQL
        // query) without depending on `a`-shorthand support in the store's
        // SPARQL parser.
        let target = Target::sparql(
            "SELECT ?this WHERE { ?this <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Invoice> }".to_string(),
            None,
        );

        let mut selector = TargetSelector::new();
        let nodes = selector
            .select_targets(&store, &target, None)
            .expect("SPARQL target selection should succeed");

        assert_eq!(
            nodes.len(),
            2,
            "SPARQL target must select exactly the two Invoice instances, got {nodes:?}"
        );
        assert!(nodes.contains(&Term::NamedNode(invoice1)));
        assert!(nodes.contains(&Term::NamedNode(invoice2)));
        assert!(!nodes.contains(&Term::NamedNode(not_invoice)));
    }

    /// A SPARQL target query that is not a SELECT (e.g. ASK) must fail loudly
    /// rather than silently returning zero focus nodes.
    #[test]
    fn regression_sparql_target_ask_query_is_fail_loud() {
        let store = ConcreteStore::new().expect("store");
        let target = Target::sparql("ASK { ?s ?p ?o }".to_string(), None);

        let mut selector = TargetSelector::new();
        let result = selector.select_targets(&store, &target, None);

        assert!(
            result.is_err(),
            "an ASK query used as a SPARQLTarget must be rejected, not silently return no nodes"
        );
    }

    /// Target::Class (and Target::Implicit, which delegates to it) must
    /// select "SHACL instances" of the class: nodes typed with the class
    /// itself OR with any rdfs:subClassOf-transitive subclass, matching
    /// ClassConstraint::check_class_membership's semantics.
    #[test]
    fn regression_target_class_includes_rdfs_subclass_instances() {
        let store = ConcreteStore::new().expect("store");
        let rex = nn("http://example.org/Rex");
        let dog_class = nn("http://example.org/Dog");
        let animal_class = nn("http://example.org/Animal");
        let rdf_type = nn("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        let subclass_of = nn("http://www.w3.org/2000/01/rdf-schema#subClassOf");

        // Rex is a Dog, and Dog is a rdfs:subClassOf Animal.
        store
            .insert_quad(Quad::new(
                rex.clone(),
                rdf_type,
                dog_class.clone(),
                GraphName::DefaultGraph,
            ))
            .expect("insert rex type");
        store
            .insert_quad(Quad::new(
                dog_class,
                subclass_of,
                animal_class.clone(),
                GraphName::DefaultGraph,
            ))
            .expect("insert subclass triple");

        let target = Target::Class(animal_class);
        let mut selector = TargetSelector::new();
        let nodes = selector
            .select_targets(&store, &target, None)
            .expect("class target selection should succeed");

        assert!(
            nodes.contains(&Term::NamedNode(rex)),
            "Rex is a SHACL instance of Animal via rdfs:subClassOf and must be selected, got {nodes:?}"
        );
    }

    /// Direct rdf:type instances of the exact target class must still be
    /// selected (baseline behavior, no regression from the subclass fix).
    #[test]
    fn regression_target_class_direct_instance_still_selected() {
        let store = ConcreteStore::new().expect("store");
        let alice = nn("http://example.org/alice");
        let person_class = nn("http://example.org/Person");
        let rdf_type = nn("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");

        store
            .insert_quad(Quad::new(
                alice.clone(),
                rdf_type,
                person_class.clone(),
                GraphName::DefaultGraph,
            ))
            .expect("insert alice type");

        let target = Target::Class(person_class);
        let mut selector = TargetSelector::new();
        let nodes = selector
            .select_targets(&store, &target, None)
            .expect("class target selection should succeed");

        assert!(nodes.contains(&Term::NamedNode(alice)));
    }
}
