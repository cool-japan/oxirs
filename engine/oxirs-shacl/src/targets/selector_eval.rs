//! Target condition evaluation and hierarchy traversal for SHACL target selection
//!
//! Extracted from selector.rs to keep file sizes under the 2000-line policy.

use std::collections::HashSet;

use oxirs_core::{model::Term, Store};

use super::types::*;
use crate::{Result, ShaclError};

use super::selector::format_term_for_sparql;

/// Evaluate a target condition for a specific node
pub(super) fn evaluate_target_condition(
    store: &dyn Store,
    node: &Term,
    condition: &TargetCondition,
    graph_name: Option<&str>,
) -> Result<bool> {
    match condition {
        TargetCondition::SparqlAsk { query, prefixes } => {
            let mut final_query = query.clone();

            let node_sparql = format_term_for_sparql(node)?;
            final_query = final_query.replace("$this", &node_sparql);

            if let Some(prefixes) = prefixes {
                final_query = format!("{prefixes}\n{final_query}");
            }

            tracing::warn!("SPARQL ASK condition evaluation not fully implemented yet");
            tracing::debug!("Would execute SPARQL query: {}", final_query);
            Ok(true)
        }
        TargetCondition::PropertyExists {
            property,
            direction,
        } => match direction {
            PropertyDirection::Subject => {
                check_property_exists(store, node, property, true, graph_name)
            }
            PropertyDirection::Object => {
                check_property_exists(store, node, property, false, graph_name)
            }
            PropertyDirection::Either => {
                let forward = check_property_exists(store, node, property, true, graph_name)?;
                let backward = check_property_exists(store, node, property, false, graph_name)?;
                Ok(forward || backward)
            }
        },
        TargetCondition::PropertyValue {
            property,
            value,
            direction,
        } => match direction {
            PropertyDirection::Subject => {
                check_property_value(store, node, property, value, true, graph_name)
            }
            PropertyDirection::Object => {
                check_property_value(store, node, property, value, false, graph_name)
            }
            PropertyDirection::Either => {
                let forward = check_property_value(store, node, property, value, true, graph_name)?;
                let backward =
                    check_property_value(store, node, property, value, false, graph_name)?;
                Ok(forward || backward)
            }
        },
        TargetCondition::HasType { class_iri } => {
            check_node_type(store, node, class_iri, graph_name)
        }
        TargetCondition::Cardinality {
            property,
            min_count,
            max_count,
            direction,
        } => {
            let count = count_property_values(store, node, property, direction, graph_name)?;

            let min_satisfied = min_count.map_or(true, |min| count >= min);
            let max_satisfied = max_count.map_or(true, |max| count <= max);

            Ok(min_satisfied && max_satisfied)
        }
    }
}

/// Traverse hierarchy from root nodes following the specified relationship
pub(super) fn traverse_hierarchy(
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

    if include_intermediate || depth == max_depth || max_depth == 0 {
        result_nodes.extend(root_nodes.iter().cloned());
    }

    while !current_level.is_empty() && (max_depth == -1 || depth < max_depth) {
        let mut next_level = HashSet::new();

        for node in &current_level {
            if visited.contains(node) {
                continue;
            }
            visited.insert(node.clone());

            let related_nodes = get_related_nodes(store, node, relationship, graph_name)?;

            for related_node in related_nodes {
                if !visited.contains(&related_node) {
                    next_level.insert(related_node.clone());

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
pub(super) fn follow_property_path(
    store: &dyn Store,
    start_nodes: &HashSet<Term>,
    path: &crate::paths::PropertyPath,
    direction: &PathDirection,
    filters: &[PathFilter],
    graph_name: Option<&str>,
) -> Result<Vec<Term>> {
    let mut result_nodes = Vec::new();

    for start_node in start_nodes {
        let path_results = evaluate_property_path(store, start_node, path, direction, graph_name)?;

        for result_node in path_results {
            if apply_path_filters(&result_node, filters, store, graph_name)? {
                result_nodes.push(result_node);
            }
        }
    }

    Ok(result_nodes)
}

/// Check if a property exists for a node
pub(super) fn check_property_exists(
    store: &dyn Store,
    node: &Term,
    property: &oxirs_core::model::NamedNode,
    forward: bool,
    graph_name: Option<&str>,
) -> Result<bool> {
    use oxirs_core::{Object, Predicate, Subject};

    let predicate: Predicate = property.clone().into();

    let quads = if forward {
        let subject_ref = match node {
            Term::NamedNode(n) => Subject::from(n.clone()),
            Term::BlankNode(n) => Subject::from(n.clone()),
            _ => return Ok(false),
        };

        if let Some(graph) = graph_name {
            let graph_node = oxirs_core::model::NamedNode::new(graph)
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e)))?;
            let gn = Some(oxirs_core::GraphName::from(graph_node));
            store.find_quads(Some(&subject_ref), Some(&predicate), None, gn.as_ref())?
        } else {
            store.find_quads(Some(&subject_ref), Some(&predicate), None, None)?
        }
    } else {
        let object_ref = match node {
            Term::NamedNode(n) => Object::from(n.clone()),
            Term::BlankNode(n) => Object::from(n.clone()),
            Term::Literal(lit) => Object::from(lit.clone()),
            _ => return Ok(false),
        };

        if let Some(graph) = graph_name {
            let graph_node = oxirs_core::model::NamedNode::new(graph)
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e)))?;
            let gn = Some(oxirs_core::GraphName::from(graph_node));
            store.find_quads(None, Some(&predicate), Some(&object_ref), gn.as_ref())?
        } else {
            store.find_quads(None, Some(&predicate), Some(&object_ref), None)?
        }
    };

    Ok(!quads.is_empty())
}

/// Check if a property has a specific value for a node
pub(super) fn check_property_value(
    store: &dyn Store,
    node: &Term,
    property: &oxirs_core::model::NamedNode,
    value: &Term,
    forward: bool,
    graph_name: Option<&str>,
) -> Result<bool> {
    use oxirs_core::{Object, Predicate, Subject};

    let predicate: Predicate = property.clone().into();

    let quads = if forward {
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
            let graph_node = oxirs_core::model::NamedNode::new(graph)
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e)))?;
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
            let graph_node = oxirs_core::model::NamedNode::new(graph)
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid graph IRI: {}", e)))?;
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

/// Check if a node is an instance of the specified class
pub(super) fn check_node_type(
    store: &dyn Store,
    node: &Term,
    class_iri: &oxirs_core::model::NamedNode,
    graph_name: Option<&str>,
) -> Result<bool> {
    use oxirs_core::{Object, Predicate, Subject};

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

/// Count property values for a node in a given direction
#[allow(clippy::only_used_in_recursion)]
pub(super) fn count_property_values(
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
            let subject_ref = match node {
                Term::NamedNode(n) => Subject::from(n.clone()),
                Term::BlankNode(n) => Subject::from(n.clone()),
                _ => return Ok(0),
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
            let as_subject = count_property_values(
                store,
                node,
                property,
                &PropertyDirection::Subject,
                graph_name,
            )?;
            let as_object = count_property_values(
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

/// Get nodes related to the given node via a hierarchical relationship
pub(super) fn get_related_nodes(
    store: &dyn Store,
    node: &Term,
    relationship: &HierarchicalRelationship,
    graph_name: Option<&str>,
) -> Result<Vec<Term>> {
    use oxirs_core::{Object, Predicate, Subject};

    match relationship {
        HierarchicalRelationship::Property(prop) => {
            let subject_ref = match node {
                Term::NamedNode(n) => Subject::from(n.clone()),
                Term::BlankNode(n) => Subject::from(n.clone()),
                _ => return Ok(Vec::new()),
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

/// Evaluate a property path from a start node
pub(super) fn evaluate_property_path(
    _store: &dyn Store,
    _start_node: &Term,
    _path: &crate::paths::PropertyPath,
    _direction: &PathDirection,
    _graph_name: Option<&str>,
) -> Result<Vec<Term>> {
    tracing::warn!("Property path evaluation not fully implemented yet");
    Ok(Vec::new())
}

/// Apply all path filters to a node
pub(super) fn apply_path_filters(
    node: &Term,
    filters: &[PathFilter],
    store: &dyn Store,
    graph_name: Option<&str>,
) -> Result<bool> {
    for filter in filters {
        if !evaluate_path_filter(node, filter, store, graph_name)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Evaluate a single path filter for a node
pub(super) fn evaluate_path_filter(
    node: &Term,
    filter: &PathFilter,
    store: &dyn Store,
    graph_name: Option<&str>,
) -> Result<bool> {
    use oxirs_core::{Object, Predicate, Subject};

    match filter {
        PathFilter::NodeType(node_type_filter) => match node_type_filter {
            NodeTypeFilter::IriOnly => Ok(matches!(node, Term::NamedNode(_))),
            NodeTypeFilter::BlankNodeOnly => Ok(matches!(node, Term::BlankNode(_))),
            NodeTypeFilter::LiteralOnly => Ok(matches!(node, Term::Literal(_))),
            NodeTypeFilter::InstanceOf(class) => {
                let subject_ref = match node {
                    Term::NamedNode(n) => Subject::from(n.clone()),
                    Term::BlankNode(n) => Subject::from(n.clone()),
                    _ => return Ok(false),
                };

                let rdf_type = oxirs_core::model::NamedNode::new(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                )
                .map_err(|e| ShaclError::TargetSelection(format!("Invalid IRI: {}", e)))?;
                let predicate: Predicate = rdf_type.into();
                let object: Object = class.clone().into();

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
        },
        PathFilter::PropertyValue { property, value } => {
            let subject_ref = match node {
                Term::NamedNode(n) => Subject::from(n.clone()),
                Term::BlankNode(n) => Subject::from(n.clone()),
                _ => return Ok(false),
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
            tracing::debug!(
                "SPARQL condition path filter requires query execution - not yet implemented"
            );
            Ok(true)
        }
    }
}
