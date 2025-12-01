//! # PropertyPathEvaluator - evaluate_predicate_direct_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Fallback method to evaluate predicate using direct store queries
    pub(super) fn evaluate_predicate_direct(
        &self,
        store: &dyn Store,
        start_node: &Term,
        predicate: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, Predicate, Subject};
        let subject = match start_node {
            Term::NamedNode(node) => Subject::NamedNode(node.clone()),
            Term::BlankNode(node) => Subject::BlankNode(node.clone()),
            _ => {
                return Err(ShaclError::PropertyPath(
                    "Invalid subject term for predicate path".to_string(),
                ));
            }
        };
        let predicate_term = Predicate::NamedNode(predicate.clone());
        let graph_filter = match graph_name {
            Some(g) => {
                Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                    ShaclError::Core(OxirsError::Parse(e.to_string()))
                })?))
            }
            None => None,
        };
        let quads = store
            .find_quads(
                Some(&subject),
                Some(&predicate_term),
                None,
                graph_filter.as_ref(),
            )
            .map_err(ShaclError::Core)?;
        let values: Vec<Term> = quads
            .into_iter()
            .map(|quad| Term::from(quad.object().clone()))
            .collect();
        tracing::debug!(
            "Direct store query found {} values for predicate path",
            values.len()
        );
        Ok(values)
    }
    /// Fallback method to evaluate inverse predicate using direct store queries
    pub(super) fn evaluate_inverse_predicate_direct(
        &self,
        store: &dyn Store,
        start_node: &Term,
        predicate: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, Object, Predicate};
        let object = match start_node {
            Term::NamedNode(node) => Object::NamedNode(node.clone()),
            Term::BlankNode(node) => Object::BlankNode(node.clone()),
            Term::Literal(literal) => Object::Literal(literal.clone()),
            _ => {
                return Err(ShaclError::PropertyPath(
                    "Invalid object term for inverse predicate path".to_string(),
                ));
            }
        };
        let predicate_term = Predicate::NamedNode(predicate.clone());
        let graph_filter = match graph_name {
            Some(g) => {
                Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                    ShaclError::Core(OxirsError::Parse(e.to_string()))
                })?))
            }
            None => None,
        };
        let quads = store
            .find_quads(
                None,
                Some(&predicate_term),
                Some(&object),
                graph_filter.as_ref(),
            )
            .map_err(ShaclError::Core)?;
        let values: Vec<Term> = quads
            .into_iter()
            .map(|quad| Term::from(quad.subject().clone()))
            .collect();
        tracing::debug!(
            "Direct store query found {} values for inverse predicate path",
            values.len()
        );
        Ok(values)
    }
    /// Fallback method to get candidate nodes using direct store queries
    pub(super) fn get_candidates_direct(
        &self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<HashSet<Term>> {
        use oxirs_core::model::GraphName;
        let mut candidates = HashSet::new();
        let graph_filter = match graph_name {
            Some(g) => {
                Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                    ShaclError::Core(OxirsError::Parse(e.to_string()))
                })?))
            }
            None => None,
        };
        let quads = store
            .find_quads(None, None, None, graph_filter.as_ref())
            .map_err(ShaclError::Core)?;
        for quad in quads.into_iter().take(1000) {
            candidates.insert(Term::from(quad.subject().clone()));
        }
        tracing::debug!(
            "Direct store query found {} candidate nodes",
            candidates.len()
        );
        Ok(candidates)
    }
    /// Create a cache key for path evaluation
    pub(super) fn create_cache_key(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        start_node.as_str().hash(&mut hasher);
        path.hash(&mut hasher);
        graph_name.hash(&mut hasher);
        format!("path_eval_{}", hasher.finish())
    }
    /// Generate native SPARQL property path query
    pub(super) fn generate_native_sparql_path_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;
        let mut query_parts = Vec::new();
        query_parts.push(self.generate_common_prefixes());
        let select_clause = if optimization_hints.parallel_threshold > 0 {
            "SELECT DISTINCT ?value # HINT: PARALLEL"
        } else {
            "SELECT DISTINCT ?value"
        };
        query_parts.push(select_clause.to_string());
        let where_clause = if let PropertyPath::Sequence(paths) = path {
            return self.generate_sequence_sparql_query(
                start_node,
                paths,
                graph_name,
                optimization_hints,
            );
        } else if matches!(path, PropertyPath::Alternative(_)) {
            self.generate_union_based_alternative_query(start_node, path, graph_name)?
        } else {
            let sparql_path = path.to_sparql_path()?;
            if let Some(graph) = graph_name {
                format!(
                    "WHERE {{\n  GRAPH <{graph}> {{\n    {start_term} {sparql_path} ?value .\n  }}\n}}"
                )
            } else {
                format!("WHERE {{\n  {start_term} {sparql_path} ?value .\n}}")
            }
        };
        query_parts.push(where_clause);
        if path.complexity() > 10 {
            query_parts.push("# HINT: USE_INDEX(property_path_index)".to_string());
        }
        query_parts.push("ORDER BY ?value".to_string());
        if optimization_hints.max_intermediate_results < usize::MAX {
            query_parts.push(format!(
                "LIMIT {}",
                optimization_hints.max_intermediate_results
            ));
        }
        Ok(query_parts.join("\n"))
    }
    /// Generate UNION-based query for alternative paths
    fn generate_union_based_alternative_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> Result<String> {
        if let PropertyPath::Alternative(paths) = path {
            let start_term = format_term_for_sparql(start_node)?;
            let mut union_parts = Vec::new();
            for alternative_path in paths {
                let path_sparql = alternative_path.to_sparql_path()?;
                let triple_pattern = if let Some(graph) = graph_name {
                    format!("    GRAPH <{graph}> {{ {start_term} {path_sparql} ?value . }}")
                } else {
                    format!("    {start_term} {path_sparql} ?value .")
                };
                union_parts.push(format!("  {{\n{triple_pattern}\n  }}"));
            }
            Ok(format!("WHERE {{\n{}\n}}", union_parts.join("\n  UNION\n")))
        } else {
            Err(ShaclError::PropertyPath(
                "Expected alternative path for UNION query generation".to_string(),
            ))
        }
    }
    /// Generate optimized SPARQL query for sequence paths
    pub(super) fn generate_sequence_sparql_query(
        &self,
        start_node: &Term,
        paths: &[PropertyPath],
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;
        let mut query_parts = Vec::new();
        query_parts.push(self.generate_common_prefixes());
        query_parts.push("SELECT DISTINCT ?value".to_string());
        let mut variables = vec!["?start".to_string()];
        for i in 1..paths.len() {
            variables.push(format!("?inter{i}"));
        }
        variables.push("?value".to_string());
        let mut where_parts = Vec::new();
        where_parts.push(format!("BIND({start_term} AS ?start)"));
        for (i, path) in paths.iter().enumerate() {
            let from_var = &variables[i];
            let to_var = &variables[i + 1];
            let sparql_path = path.to_sparql_path()?;
            let triple_pattern = if let Some(graph) = graph_name {
                format!("GRAPH <{graph}> {{ {from_var} {sparql_path} {to_var} . }}")
            } else {
                format!("{from_var} {sparql_path} {to_var} .")
            };
            where_parts.push(triple_pattern);
        }
        query_parts.push(format!("WHERE {{\n  {}\n}}", where_parts.join("\n  ")));
        if paths.len() > 3 {
            query_parts.push("# HINT: OPTIMIZE_JOIN_ORDER".to_string());
        }
        query_parts.push("ORDER BY ?value".to_string());
        if optimization_hints.max_intermediate_results < usize::MAX {
            query_parts.push(format!(
                "LIMIT {}",
                optimization_hints.max_intermediate_results
            ));
        }
        Ok(query_parts.join("\n"))
    }
    /// Generate optimized SPARQL query for alternative paths using UNION
    pub(super) fn generate_union_sparql_query(
        &self,
        start_node: &Term,
        paths: &[PropertyPath],
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;
        let mut query_parts = Vec::new();
        query_parts.push(self.generate_common_prefixes());
        query_parts.push("SELECT DISTINCT ?value".to_string());
        let mut union_parts = Vec::new();
        for path in paths {
            let sparql_path = path.to_sparql_path()?;
            let union_part = if let Some(graph) = graph_name {
                format!("{{ GRAPH <{graph}> {{ {start_term} {sparql_path} ?value . }} }}")
            } else {
                format!("{{ {start_term} {sparql_path} ?value . }}")
            };
            union_parts.push(union_part);
        }
        query_parts.push(format!(
            "WHERE {{\n  {}\n}}",
            union_parts.join("\n  UNION\n  ")
        ));
        if paths.len() > 5 {
            query_parts.push("# HINT: OPTIMIZE_UNION".to_string());
        }
        query_parts.push("ORDER BY ?value".to_string());
        if optimization_hints.max_intermediate_results < usize::MAX {
            query_parts.push(format!(
                "LIMIT {}",
                optimization_hints.max_intermediate_results
            ));
        }
        Ok(query_parts.join("\n"))
    }
    /// Generate specialized SPARQL query for recursive paths
    pub(super) fn generate_recursive_sparql_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let start_term = format_term_for_sparql(start_node)?;
        let mut query_parts = Vec::new();
        query_parts.push(self.generate_common_prefixes());
        query_parts.push("SELECT DISTINCT ?value".to_string());
        let where_clause = match path {
            PropertyPath::ZeroOrMore(inner_path) => {
                let inner_sparql = inner_path.to_sparql_path()?;
                if let Some(graph) = graph_name {
                    format!(
                        "WHERE {{\n  {{\n    BIND({start_term} AS ?value)\n  }}\n  UNION\n  {{\n    GRAPH <{graph}> {{\n      {start_term} {inner_sparql}+ ?value .\n    }}\n  }}\n}}"
                    )
                } else {
                    format!(
                        "WHERE {{\n  {{\n    BIND({start_term} AS ?value)\n  }}\n  UNION\n  {{\n    {start_term} {inner_sparql}+ ?value .\n  }}\n}}"
                    )
                }
            }
            PropertyPath::OneOrMore(inner_path) => {
                let inner_sparql = inner_path.to_sparql_path()?;
                if let Some(graph) = graph_name {
                    format!(
                        "WHERE {{\n  GRAPH <{graph}> {{\n    {start_term} {inner_sparql}+ ?value .\n  }}\n}}"
                    )
                } else {
                    format!("WHERE {{\n  {start_term} {inner_sparql}+ ?value .\n}}")
                }
            }
            _ => {
                return Err(ShaclError::PropertyPath(
                    "Invalid recursive path type".to_string(),
                ));
            }
        };
        query_parts.push(where_clause);
        query_parts.push(format!(
            "# HINT: MAX_RECURSION_DEPTH {}",
            optimization_hints.max_recursion_depth
        ));
        query_parts.push("ORDER BY ?value".to_string());
        let recursive_limit = optimization_hints.max_intermediate_results.min(1000);
        query_parts.push(format!("LIMIT {recursive_limit}"));
        Ok(query_parts.join("\n"))
    }
    /// Validate a property path for correctness and performance
    pub fn validate_property_path(&self, path: &PropertyPath) -> PathValidationResult {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let complexity = path.complexity();
        if complexity > 100 {
            warnings.push(format!(
                "High complexity path ({complexity}), consider simplification"
            ));
        }
        match path {
            PropertyPath::ZeroOrMore(_) => {
                warnings.push("Zero-or-more paths can be expensive on large datasets".to_string());
            }
            PropertyPath::OneOrMore(_) => {
                warnings.push("One-or-more paths can be expensive on large datasets".to_string());
            }
            PropertyPath::Sequence(paths) => {
                if paths.len() > 10 {
                    warnings.push(format!(
                        "Long sequence path ({} steps), consider breaking into smaller parts",
                        paths.len()
                    ));
                }
                if paths.is_empty() {
                    errors.push("Empty sequence path is invalid".to_string());
                }
            }
            PropertyPath::Alternative(paths) => {
                if paths.len() > 20 {
                    warnings.push(format!(
                        "Many alternatives ({}), consider grouping",
                        paths.len()
                    ));
                }
                if paths.len() < 2 {
                    errors.push("Alternative path must have at least 2 alternatives".to_string());
                }
            }
            _ => {}
        }
        if !path.can_use_sparql_path() {
            warnings.push(
                "Path cannot use native SPARQL property paths, will use programmatic evaluation"
                    .to_string(),
            );
        }
        if self.has_potential_cycles(path) {
            warnings.push(
                "Path has potential for infinite recursion, ensure proper depth limits".to_string(),
            );
        }
        PathValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            complexity,
            can_use_sparql: path.can_use_sparql_path(),
            estimated_cost: self.estimate_path_cost(path),
        }
    }
}
