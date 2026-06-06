//! Core flattening algorithm for JSON-LD 1.1.
//!
//! This module implements:
//!
//! * [`flatten_internal`] — produces `{"@graph": [...]}` from an expanded document.
//! * [`merge_value`]      — helper to merge without duplicating `@value` literals.
//! * [`clone_node_recursively`] — deep clone a node, inlining sub-graph content.
//! * Type helpers [`is_node_object`], [`is_value_object`], [`is_list_object`].
//!
//! Spec reference:
//! <https://www.w3.org/TR/json-ld11-api/#flattening-algorithm>

use super::node_map::{
    generate_node_map, merge_value as merge_value_inner, node_map_to_flat_array, NodeMap,
    NodeObject,
};
use super::{FlatteningError, FlatteningOptions, JsonLdValue};
use indexmap::IndexMap;
use std::collections::HashSet;

// ============================================================================
// Type predicates
// ============================================================================

/// Returns `true` if `value` is a JSON-LD node object.
///
/// A node object is a JSON object that is neither a value object nor a list
/// object (i.e., has no `@value` or `@list` key).  Scalars and arrays are not
/// node objects.
#[inline]
pub fn is_node_object(value: &JsonLdValue) -> bool {
    match value {
        JsonLdValue::Object(m) => !m.contains_key("@value") && !m.contains_key("@list"),
        _ => false,
    }
}

/// Returns `true` if `value` is a JSON-LD value object (`{"@value": …}`).
#[inline]
pub fn is_value_object(value: &JsonLdValue) -> bool {
    match value {
        JsonLdValue::Object(m) => m.contains_key("@value"),
        _ => false,
    }
}

/// Returns `true` if `value` is a JSON-LD list object (`{"@list": …}`).
#[inline]
pub fn is_list_object(value: &JsonLdValue) -> bool {
    match value {
        JsonLdValue::Object(m) => m.contains_key("@list"),
        _ => false,
    }
}

// ============================================================================
// merge_value — public re-export of the inner helper
// ============================================================================

/// Merge `value` into `into` without inserting structural duplicates.
///
/// This delegates to [`merge_value_inner`] in `node_map`.
pub fn merge_value(into: &mut Vec<JsonLdValue>, value: JsonLdValue) {
    merge_value_inner(into, value);
}

// ============================================================================
// flatten_internal
// ============================================================================

/// Core JSON-LD flattening algorithm.
///
/// Accepts an already-expanded array of node objects and returns a
/// `{"@graph": [...]}` document where every node appears at the top level.
///
/// # Steps
///
/// 1. Generate the node map from the expanded input.
/// 2. Retrieve the `@default` graph.
/// 3. For each node in sorted order, build a flat JSON-LD object and add it
///    to the result array (with any sub-graph content inlined).
/// 4. Return `{"@graph": result_array}`.
///
/// # Arguments
///
/// * `expanded` — Expanded JSON-LD document array.
/// * `options`  — Flattening options.
pub fn flatten_internal(
    expanded: Vec<JsonLdValue>,
    options: &FlatteningOptions,
) -> Result<JsonLdValue, FlatteningError> {
    // Step 1 – build the node map.
    let node_map = generate_node_map(&expanded, options)?;

    // Step 2 – serialise the default graph to a flat array.
    let flat_array = node_map_to_flat_array(&node_map, options.ordered);

    // Step 3 – wrap in @graph.
    let mut result_map: IndexMap<String, JsonLdValue> = IndexMap::new();
    result_map.insert("@graph".to_string(), JsonLdValue::Array(flat_array));

    Ok(JsonLdValue::Object(result_map))
}

// ============================================================================
// clone_node_recursively
// ============================================================================

/// Deep-clone a [`NodeObject`], inlining any sub-graph content it has.
///
/// This is used to produce fully self-contained node representations when
/// building the flat output.  Cycle detection is performed via `visited`; a
/// [`FlatteningError::CyclicNodeReference`] is returned if a cycle is found.
///
/// # Arguments
///
/// * `node`    — The node to clone.
/// * `node_map` — The full node map (used to look up sub-graphs).
/// * `visited` — Set of subject IDs currently on the traversal stack.
pub fn clone_node_recursively(
    node: &NodeObject,
    node_map: &NodeMap,
    visited: &mut HashSet<String>,
) -> Result<JsonLdValue, FlatteningError> {
    if visited.contains(&node.id) {
        return Err(FlatteningError::CyclicNodeReference(node.id.clone()));
    }
    visited.insert(node.id.clone());

    let mut map: IndexMap<String, JsonLdValue> = IndexMap::new();

    map.insert("@id".to_string(), JsonLdValue::Str(node.id.clone()));

    if !node.types.is_empty() {
        let types: Vec<JsonLdValue> = node
            .types
            .iter()
            .map(|t| JsonLdValue::Str(t.clone()))
            .collect();
        map.insert("@type".to_string(), JsonLdValue::Array(types));
    }

    // Properties (skip @type — already handled).
    for (prop, values) in &node.properties {
        if prop == "@type" {
            continue;
        }
        let mut cloned_values: Vec<JsonLdValue> = Vec::with_capacity(values.len());
        for val in values {
            cloned_values.push(clone_value_recursively(val, node_map, visited)?);
        }
        if !cloned_values.is_empty() {
            map.insert(prop.clone(), JsonLdValue::Array(cloned_values));
        }
    }

    // Inline sub-graph if present.
    if let Some(inner_graph) = node_map.graphs.get(&node.id) {
        let mut inner_nodes: Vec<&str> = inner_graph.nodes.keys().map(String::as_str).collect();
        inner_nodes.sort_unstable();
        let mut inner_array: Vec<JsonLdValue> = Vec::with_capacity(inner_nodes.len());
        for subj in inner_nodes {
            let inner_node = &inner_graph.nodes[subj];
            inner_array.push(clone_node_recursively(inner_node, node_map, visited)?);
        }
        map.insert("@graph".to_string(), JsonLdValue::Array(inner_array));
    }

    visited.remove(&node.id);
    Ok(JsonLdValue::Object(map))
}

/// Recursively clone a single value, descending into node references.
fn clone_value_recursively(
    value: &JsonLdValue,
    node_map: &NodeMap,
    visited: &mut HashSet<String>,
) -> Result<JsonLdValue, FlatteningError> {
    match value {
        JsonLdValue::Object(m) => {
            if m.contains_key("@value") || m.contains_key("@list") {
                // Value or list object — return as-is.
                return Ok(value.clone());
            }
            // May be a node reference `{"@id": "..."}`.
            if let Some(JsonLdValue::Str(id)) = m.get("@id") {
                // Check if this ID has a full entry in the node map.
                if let Some(node) = node_map.default_graph().nodes.get(id.as_str()) {
                    return clone_node_recursively(node, node_map, visited);
                }
            }
            Ok(value.clone())
        }
        JsonLdValue::Array(items) => {
            let mut out: Vec<JsonLdValue> = Vec::with_capacity(items.len());
            for item in items {
                out.push(clone_value_recursively(item, node_map, visited)?);
            }
            Ok(JsonLdValue::Array(out))
        }
        other => Ok(other.clone()),
    }
}
