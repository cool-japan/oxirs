//! SPARQL query generation for SHACL target selection
//!
//! Extracted from selector.rs to keep file sizes under the 2000-line policy.

use std::time::{Duration, Instant};

use super::optimization::*;
use super::types::*;
use crate::{Result, ShaclError};

use super::selector::format_term_for_sparql;

/// Type alias for a callable that generates an optimized SPARQL query for a single target.
pub(super) type OptimizedQueryFn<'a> =
    dyn Fn(&Target, Option<&str>, &QueryOptimizationOptions) -> Result<String> + 'a;

/// Type alias for a callable that generates a union SPARQL query across multiple targets.
pub(super) type UnionQueryFn<'a> = dyn Fn(&[Target], Option<&str>) -> Result<String> + 'a;

/// Generate union SPARQL query from multiple targets
pub(super) fn generate_union_query(
    targets: &[Target],
    graph_name: Option<&str>,
    query_fn: &dyn Fn(&Target, Option<&str>) -> Result<String>,
) -> Result<String> {
    if targets.is_empty() {
        return Ok("SELECT DISTINCT ?target WHERE { }".to_string());
    }

    let mut union_parts = Vec::new();

    for target in targets {
        let individual_query = query_fn(target, graph_name)?;

        if let Some(where_start) = individual_query.find("WHERE {") {
            let where_clause = &individual_query[where_start + 7..];
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

/// Generate intersection SPARQL query from multiple targets
pub(super) fn generate_intersection_query(
    targets: &[Target],
    graph_name: Option<&str>,
    query_fn: &dyn Fn(&Target, Option<&str>) -> Result<String>,
) -> Result<String> {
    if targets.is_empty() {
        return Ok("SELECT DISTINCT ?target WHERE { }".to_string());
    }

    if targets.len() == 1 {
        return query_fn(&targets[0], graph_name);
    }

    let mut constraints = Vec::new();

    for (index, target) in targets.iter().enumerate() {
        let subquery_var = format!("?target_{index}");
        let individual_query = query_fn(target, graph_name)?;

        if let Some(where_start) = individual_query.find("WHERE {") {
            let where_clause = &individual_query[where_start + 7..];
            if let Some(where_end) = where_clause.rfind('}') {
                let where_content = &where_clause[..where_end].trim();
                if !where_content.is_empty() {
                    let adapted_content = where_content.replace("?target", &subquery_var);
                    constraints.push(adapted_content.to_string());

                    if index > 0 {
                        constraints.push(format!("FILTER(?target = {subquery_var})"));
                    }
                }
            }
        }
    }

    constraints.push("BIND(?target_0 AS ?target)".to_string());

    Ok(format!(
        "SELECT DISTINCT ?target WHERE {{\n  {}\n}}",
        constraints.join("\n  ")
    ))
}

/// Generate difference SPARQL query
pub(super) fn generate_difference_query(
    primary: &Target,
    exclusion: &Target,
    graph_name: Option<&str>,
    query_fn: &dyn Fn(&Target, Option<&str>) -> Result<String>,
) -> Result<String> {
    let primary_query = query_fn(primary, graph_name)?;
    let exclusion_query = query_fn(exclusion, graph_name)?;

    let primary_where = extract_where_clause(&primary_query)?;
    let exclusion_where = extract_where_clause(&exclusion_query)?;

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

/// Generate conditional SPARQL query
pub(super) fn generate_conditional_query(
    base: &Target,
    condition: &TargetCondition,
    context: Option<&TargetContext>,
    graph_name: Option<&str>,
    query_fn: &dyn Fn(&Target, Option<&str>) -> Result<String>,
) -> Result<String> {
    let base_query = query_fn(base, graph_name)?;
    let base_where = extract_where_clause(&base_query)?;

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

/// Generate hierarchical SPARQL query
pub(super) fn generate_hierarchical_query(
    root: &Target,
    relationship: &HierarchicalRelationship,
    max_depth: i32,
    include_intermediate: bool,
    graph_name: Option<&str>,
    query_fn: &dyn Fn(&Target, Option<&str>) -> Result<String>,
) -> Result<String> {
    let root_iri = match root {
        Target::Class(class_iri) => Some(class_iri.as_str()),
        Target::Node(oxirs_core::model::Term::NamedNode(node)) => Some(node.as_str()),
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
        "50".to_string()
    };

    let path_pattern = if include_intermediate {
        format!("{relationship_pattern}*")
    } else {
        format!("{relationship_pattern}+")
    };

    let query = if let Some(iri) = root_iri {
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
        let root_query = query_fn(root, graph_name)?;
        let root_where = extract_where_clause(&root_query)?;

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

/// Generate path-based SPARQL query
pub(super) fn generate_path_based_query(
    start: &Target,
    path: &crate::paths::PropertyPath,
    direction: &PathDirection,
    filters: &[PathFilter],
    graph_name: Option<&str>,
    query_fn: &dyn Fn(&Target, Option<&str>) -> Result<String>,
) -> Result<String> {
    let start_query = query_fn(start, graph_name)?;
    let start_where = extract_where_clause(&start_query)?;

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

/// Extract WHERE clause content from a SPARQL query
pub(super) fn extract_where_clause(query: &str) -> Result<String> {
    if let Some(where_start) = query.find("WHERE {") {
        let where_clause = &query[where_start + 7..];
        if let Some(where_end) = where_clause.rfind('}') {
            let where_content = where_clause[..where_end].trim();
            return Ok(where_content.to_string());
        }
    }
    Err(ShaclError::TargetSelection(
        "Could not extract WHERE clause from query".to_string(),
    ))
}

/// Add index hints to optimize query performance
pub(super) fn add_index_hints(query: String, target: &Target) -> Result<String> {
    match target {
        Target::Class(_) => Ok(format!("# Use RDF type index for class targets\n{query}")),
        Target::ObjectsOf(property) | Target::SubjectsOf(property) => Ok(format!(
            "# Use property index for {}\n{}",
            property.as_str(),
            query
        )),
        _ => Ok(query),
    }
}

/// Add deterministic ordering to ensure consistent results
pub(super) fn add_deterministic_ordering(query: String) -> Result<String> {
    if !query.contains("ORDER BY") {
        Ok(format!("{query} ORDER BY ?target"))
    } else {
        Ok(query)
    }
}

/// Add performance hints for query optimization
pub(super) fn add_performance_hints(query: String, target: &Target) -> Result<String> {
    let mut hints = Vec::new();

    match target {
        Target::Class(_) => {
            hints.push("# Consider using RDFS reasoning for subclass relationships".to_string());
        }
        Target::Union(union_target) if union_target.targets.len() > 5 => {
            hints.push("# Large union - consider query rewriting".to_string());
        }
        Target::Intersection(intersection_target) if intersection_target.targets.len() > 3 => {
            hints.push("# Complex intersection - consider selectivity ordering".to_string());
        }
        _ => {}
    }

    if hints.is_empty() {
        Ok(query)
    } else {
        Ok(format!("{}\n{}", hints.join("\n"), query))
    }
}

/// Estimate cardinality for a target (simple heuristic)
#[allow(clippy::only_used_in_recursion)]
pub(super) fn estimate_target_cardinality(target: &Target) -> usize {
    match target {
        Target::Class(_) => 1000,
        Target::Node(_) => 1,
        Target::ObjectsOf(_) => 500,
        Target::SubjectsOf(_) => 500,
        Target::Union(union_target) => union_target
            .targets
            .iter()
            .map(estimate_target_cardinality)
            .sum(),
        Target::Intersection(intersection_target) => {
            intersection_target
                .targets
                .iter()
                .map(estimate_target_cardinality)
                .min()
                .unwrap_or(0)
                / 2
        }
        _ => 100,
    }
}

/// Generate optimized batch query result for multiple targets
pub(super) fn generate_batch_query_impl(
    targets: &[Target],
    graph_name: Option<&str>,
    optimization_config: &TargetOptimizationConfig,
    optimized_query_fn: &OptimizedQueryFn<'_>,
    union_query_fn: &UnionQueryFn<'_>,
) -> Result<BatchQueryResult> {
    let start_time = Instant::now();
    let mut individual_queries = Vec::new();
    let mut total_estimated_cardinality = 0;

    for target in targets {
        let query = optimized_query_fn(target, graph_name, &QueryOptimizationOptions::default())?;
        let estimated_cardinality = estimate_target_cardinality(target);
        total_estimated_cardinality += estimated_cardinality;

        individual_queries.push(OptimizedQuery {
            sparql: query,
            estimated_cardinality,
            execution_strategy: ExecutionStrategy::Sequential,
            index_hints: vec![],
            optimization_time: Duration::from_millis(0),
        });
    }

    let union_query = if targets.len() > 1 && optimization_config.use_union_optimization {
        Some(union_query_fn(targets, graph_name)?)
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
