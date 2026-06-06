//! Node map generation for JSON-LD 1.1 Flattening.
//!
//! The node map is the intermediate representation produced during flattening.
//! Every named-graph-scoped subject appears as an entry in the node map for
//! its graph (`@default` for the top-level graph).
//!
//! Spec reference:
//! <https://www.w3.org/TR/json-ld11-api/#node-map-generation>

use super::{FlatteningError, FlatteningOptions, JsonLdValue};
use indexmap::IndexMap;

// ============================================================================
// BlankNodeIdMapper
// ============================================================================

/// Canonically renames blank node identifiers in encounter order.
///
/// The first blank node seen gets `_:b0`, the second `_:b1`, and so on.
/// Mapping is idempotent: mapping the same original ID twice returns the
/// same canonical name.
#[derive(Debug, Default)]
pub struct BlankNodeIdMapper {
    mapping: IndexMap<String, String>,
    counter: u64,
}

impl BlankNodeIdMapper {
    /// Create a new, empty mapper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Map `original` to a canonical blank-node ID.
    ///
    /// If `original` has been seen before the previously assigned name is
    /// returned (idempotent).  Otherwise a new `_:bN` name is assigned.
    pub fn map(&mut self, original: &str) -> String {
        if let Some(mapped) = self.mapping.get(original) {
            return mapped.clone();
        }
        let new_id = format!("_:b{}", self.counter);
        self.counter += 1;
        self.mapping.insert(original.to_string(), new_id.clone());
        new_id
    }

    /// Reset the mapper so it starts again from `_:b0`.
    ///
    /// All previous mappings are discarded.
    pub fn reset(&mut self) {
        self.mapping.clear();
        self.counter = 0;
    }
}

// ============================================================================
// NodeObject
// ============================================================================

/// A single RDF subject collected during node-map generation.
#[derive(Debug, Clone, Default)]
pub struct NodeObject {
    /// Canonical subject identifier (may be a blank node `_:bN`).
    pub id: String,
    /// List of `@type` IRIs for this subject.
    pub types: Vec<String>,
    /// Property → array-of-values map.
    pub properties: IndexMap<String, Vec<JsonLdValue>>,
}

impl NodeObject {
    /// Create a new, empty `NodeObject` with the given identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Merge a value into one of this node's property arrays.
    ///
    /// Duplicate `{"@value": …}` literals are not inserted twice.
    pub fn merge_property(&mut self, property: &str, value: JsonLdValue) {
        let values = self.properties.entry(property.to_string()).or_default();
        merge_value(values, value);
    }
}

// ============================================================================
// GraphNodeMap
// ============================================================================

/// A subject-keyed map for one named graph (or the `@default` graph).
#[derive(Debug, Clone, Default)]
pub struct GraphNodeMap {
    /// Subject ID → node object.
    pub nodes: IndexMap<String, NodeObject>,
}

impl GraphNodeMap {
    /// Create an empty graph node map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create the [`NodeObject`] for `id`.
    pub fn get_or_create(&mut self, id: &str) -> &mut NodeObject {
        self.nodes
            .entry(id.to_string())
            .or_insert_with(|| NodeObject::new(id))
    }
}

// ============================================================================
// NodeMap
// ============================================================================

/// The complete node map: one [`GraphNodeMap`] per named graph plus `@default`.
#[derive(Debug, Clone, Default)]
pub struct NodeMap {
    /// Graph name → per-graph node map.  Always contains at least `@default`.
    pub graphs: IndexMap<String, GraphNodeMap>,
}

impl NodeMap {
    /// Create a new node map, pre-seeded with a `@default` graph.
    pub fn new() -> Self {
        let mut nm = Self::default();
        nm.graphs
            .insert("@default".to_string(), GraphNodeMap::new());
        nm
    }

    /// Get the mutable reference to a graph, creating it if absent.
    pub fn get_or_create_graph(&mut self, graph_name: &str) -> &mut GraphNodeMap {
        self.graphs.entry(graph_name.to_string()).or_default()
    }

    /// Shorthand accessor for the `@default` graph.
    pub fn default_graph(&self) -> &GraphNodeMap {
        self.graphs
            .get("@default")
            .expect("@default graph always present")
    }

    /// Mutable shorthand accessor for the `@default` graph.
    pub fn default_graph_mut(&mut self) -> &mut GraphNodeMap {
        self.graphs
            .get_mut("@default")
            .expect("@default graph always present")
    }
}

// ============================================================================
// generate_node_map — public API
// ============================================================================

/// Generate a node map from an array of expanded JSON-LD node objects.
///
/// This implements the W3C JSON-LD 1.1 Node-Map Generation algorithm.
///
/// # Arguments
///
/// * `expanded`        — The expanded JSON-LD document (top-level array).
/// * `options`         — Flattening options (used for processing mode).
///
/// # Returns
///
/// A [`NodeMap`] where every node has been collected by subject and graph.
pub fn generate_node_map(
    expanded: &[JsonLdValue],
    options: &FlatteningOptions,
) -> Result<NodeMap, FlatteningError> {
    let mut node_map = NodeMap::new();
    let mut blank_mapper = BlankNodeIdMapper::new();

    for element in expanded {
        generate_node_map_element(
            element,
            &mut node_map,
            "@default",
            None,
            None,
            &mut blank_mapper,
            options,
        )?;
    }

    Ok(node_map)
}

// ============================================================================
// generate_node_map_element — recursive traversal
// ============================================================================

/// Process one JSON-LD element during node-map generation.
///
/// This is the core recursive step of the algorithm.  It handles:
///
/// * **Scalar values** — ignored at top level; stored as property values when
///   nested.
/// * **`@graph` objects** — recurse into the named graph.
/// * **`@value` objects** — literal value; stored directly on the property.
/// * **`@list` objects** — list node; each element is processed recursively.
/// * **Node objects** — collected into the node map; properties are recursed.
/// * **`@reverse` properties** — recorded as reverse links.
#[allow(clippy::too_many_arguments)]
pub fn generate_node_map_element(
    element: &JsonLdValue,
    node_map: &mut NodeMap,
    active_graph: &str,
    active_subject: Option<&str>,
    active_property: Option<&str>,
    blank_mapper: &mut BlankNodeIdMapper,
    _options: &FlatteningOptions,
) -> Result<(), FlatteningError> {
    match element {
        // ── Array ────────────────────────────────────────────────────────────
        JsonLdValue::Array(items) => {
            for item in items {
                generate_node_map_element(
                    item,
                    node_map,
                    active_graph,
                    active_subject,
                    active_property,
                    blank_mapper,
                    _options,
                )?;
            }
            return Ok(());
        }

        // ── Non-object scalars ────────────────────────────────────────────────
        JsonLdValue::Null | JsonLdValue::Bool(_) | JsonLdValue::Number(_) | JsonLdValue::Str(_) => {
            // Scalars can appear as property values; store them if we have context.
            if let (Some(subj), Some(prop)) = (active_subject, active_property) {
                let graph = node_map.get_or_create_graph(active_graph);
                graph
                    .get_or_create(subj)
                    .merge_property(prop, element.clone());
            }
            return Ok(());
        }

        JsonLdValue::Object(_) => {} // handled below
    }

    let obj = match element {
        JsonLdValue::Object(m) => m,
        _ => unreachable!("handled above"),
    };

    // ── @value object (literal) ───────────────────────────────────────────────
    if obj.contains_key("@value") {
        if let (Some(subj), Some(prop)) = (active_subject, active_property) {
            let graph = node_map.get_or_create_graph(active_graph);
            graph
                .get_or_create(subj)
                .merge_property(prop, element.clone());
        }
        return Ok(());
    }

    // ── @list object ─────────────────────────────────────────────────────────
    if obj.contains_key("@list") {
        // Process each item in the list, then store the whole list as a value.
        let list_items = match obj.get("@list") {
            Some(JsonLdValue::Array(items)) => items.clone(),
            _ => Vec::new(),
        };

        // Collect processed list items.
        let mut processed: Vec<JsonLdValue> = Vec::new();
        for item in &list_items {
            let item_copy = resolve_node_reference(item, node_map, active_graph, blank_mapper);
            processed.push(item_copy);
        }

        let list_obj = {
            let mut m: IndexMap<String, JsonLdValue> = IndexMap::new();
            m.insert("@list".to_string(), JsonLdValue::Array(processed));
            JsonLdValue::Object(m)
        };

        if let (Some(subj), Some(prop)) = (active_subject, active_property) {
            let graph = node_map.get_or_create_graph(active_graph);
            graph.get_or_create(subj).merge_property(prop, list_obj);
        }
        return Ok(());
    }

    // ── Determine subject ID ──────────────────────────────────────────────────
    let id = get_or_assign_id(obj, blank_mapper);

    // Ensure the node exists in the current graph.
    node_map
        .get_or_create_graph(active_graph)
        .get_or_create(&id);

    // If we are nested under a subject/property, link the parent to this node.
    if let (Some(subj), Some(prop)) = (active_subject, active_property) {
        let ref_obj = {
            let mut m: IndexMap<String, JsonLdValue> = IndexMap::new();
            m.insert("@id".to_string(), JsonLdValue::Str(id.clone()));
            JsonLdValue::Object(m)
        };
        let graph = node_map.get_or_create_graph(active_graph);
        graph.get_or_create(subj).merge_property(prop, ref_obj);
    }

    // ── Named graph (`@graph`) ────────────────────────────────────────────────
    if let Some(graph_val) = obj.get("@graph") {
        // Ensure the graph is registered in @default.
        {
            let def_graph = node_map.default_graph_mut();
            def_graph.get_or_create(&id);
        }
        // Recurse into the named graph.
        let graph_items = match graph_val {
            JsonLdValue::Array(items) => items.clone(),
            single => vec![single.clone()],
        };
        for item in &graph_items {
            generate_node_map_element(item, node_map, &id, None, None, blank_mapper, _options)?;
        }
    }

    // ── @type ─────────────────────────────────────────────────────────────────
    if let Some(type_val) = obj.get("@type") {
        let type_iris: Vec<String> = match type_val {
            JsonLdValue::Array(items) => items
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| {
                    if s.starts_with("_:") {
                        blank_mapper.map(s)
                    } else {
                        s.to_string()
                    }
                })
                .collect(),
            JsonLdValue::Str(s) => {
                let mapped = if s.starts_with("_:") {
                    blank_mapper.map(s)
                } else {
                    s.to_string()
                };
                vec![mapped]
            }
            _ => Vec::new(),
        };

        let graph = node_map.get_or_create_graph(active_graph);
        let node = graph.get_or_create(&id);
        for t in type_iris {
            if !node.types.contains(&t) {
                node.types.push(t.clone());
            }
            merge_value(
                node.properties.entry("@type".to_string()).or_default(),
                JsonLdValue::Str(t),
            );
        }
    }

    // ── @reverse properties ───────────────────────────────────────────────────
    if let Some(JsonLdValue::Object(rev_obj)) = obj.get("@reverse") {
        for (rev_prop, rev_val) in rev_obj {
            let rev_items: Vec<JsonLdValue> = match rev_val {
                JsonLdValue::Array(a) => a.clone(),
                single => vec![single.clone()],
            };
            for rev_item in &rev_items {
                // The reverse item is the *subject*; `id` is the object.
                let rev_item_id = match rev_item {
                    JsonLdValue::Object(m) => get_or_assign_id(m, blank_mapper),
                    _ => continue,
                };
                // Ensure rev_item_id exists.
                node_map
                    .get_or_create_graph(active_graph)
                    .get_or_create(&rev_item_id);
                // Record the forward property on rev_item_id pointing to id.
                let ref_obj = {
                    let mut m: IndexMap<String, JsonLdValue> = IndexMap::new();
                    m.insert("@id".to_string(), JsonLdValue::Str(id.clone()));
                    JsonLdValue::Object(m)
                };
                let graph = node_map.get_or_create_graph(active_graph);
                graph
                    .get_or_create(&rev_item_id)
                    .merge_property(rev_prop, ref_obj);
            }
        }
    }

    // ── Other properties ──────────────────────────────────────────────────────
    for (prop, value) in obj {
        // Skip already-handled keywords.
        match prop.as_str() {
            "@id" | "@type" | "@graph" | "@reverse" | "@context" => continue,
            _ => {}
        }

        let values: Vec<JsonLdValue> = match value {
            JsonLdValue::Array(a) => a.clone(),
            single => vec![single.clone()],
        };

        for v in values {
            generate_node_map_element(
                &v,
                node_map,
                active_graph,
                Some(&id),
                Some(prop),
                blank_mapper,
                _options,
            )?;
        }
    }

    Ok(())
}

// ============================================================================
// node_map_to_flat_array — serialise back to JSON-LD
// ============================================================================

/// Serialise the `@default` graph of a node map to a flat array of node objects.
///
/// Each [`NodeObject`] in the default graph is converted to a JSON-LD object
/// with `@id`, optionally `@type`, and all recorded properties.
///
/// # Arguments
///
/// * `node_map` — The node map to serialise.
/// * `ordered`  — If `true`, sort subjects lexicographically.
pub fn node_map_to_flat_array(node_map: &NodeMap, ordered: bool) -> Vec<JsonLdValue> {
    let default_graph = node_map.default_graph();

    let mut subjects: Vec<&str> = default_graph.nodes.keys().map(String::as_str).collect();
    if ordered {
        subjects.sort_unstable();
    }

    let mut result: Vec<JsonLdValue> = Vec::with_capacity(subjects.len());

    for subj_id in subjects {
        let node_obj = &default_graph.nodes[subj_id];
        let mut map: IndexMap<String, JsonLdValue> = IndexMap::new();

        map.insert("@id".to_string(), JsonLdValue::Str(subj_id.to_string()));

        // Include @type if present.
        if !node_obj.types.is_empty() {
            let types: Vec<JsonLdValue> = node_obj
                .types
                .iter()
                .map(|t| JsonLdValue::Str(t.clone()))
                .collect();
            map.insert("@type".to_string(), JsonLdValue::Array(types));
        }

        // Include all other properties (skip @type — already handled above).
        let mut props: Vec<&str> = node_obj.properties.keys().map(String::as_str).collect();
        if ordered {
            props.sort_unstable();
        }
        for prop in props {
            if prop == "@type" {
                continue;
            }
            let values = &node_obj.properties[prop];
            if !values.is_empty() {
                map.insert(prop.to_string(), JsonLdValue::Array(values.clone()));
            }
        }

        // If this subject is a named graph, include its `@graph` member.
        if node_map.graphs.contains_key(subj_id) && subj_id != "@default" {
            let inner_graph = &node_map.graphs[subj_id];
            let mut inner_subjects: Vec<&str> =
                inner_graph.nodes.keys().map(String::as_str).collect();
            if ordered {
                inner_subjects.sort_unstable();
            }
            let inner_array: Vec<JsonLdValue> = inner_subjects
                .iter()
                .map(|s| {
                    let inner_node = &inner_graph.nodes[*s];
                    serialise_node_object(inner_node, ordered)
                })
                .collect();
            map.insert("@graph".to_string(), JsonLdValue::Array(inner_array));
        }

        result.push(JsonLdValue::Object(map));
    }

    result
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Serialise a [`NodeObject`] to a JSON-LD object value.
fn serialise_node_object(node: &NodeObject, ordered: bool) -> JsonLdValue {
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

    let mut props: Vec<&str> = node.properties.keys().map(String::as_str).collect();
    if ordered {
        props.sort_unstable();
    }
    for prop in props {
        if prop == "@type" {
            continue;
        }
        let values = &node.properties[prop];
        if !values.is_empty() {
            map.insert(prop.to_string(), JsonLdValue::Array(values.clone()));
        }
    }

    JsonLdValue::Object(map)
}

/// Get `@id` from a node object, or assign a blank-node ID via the mapper.
///
/// If the object already has a string `@id` the value is returned (with
/// blank-node renaming applied if it starts with `_:`).  If there is no `@id`
/// a fresh blank-node ID is generated.
fn get_or_assign_id(
    obj: &IndexMap<String, JsonLdValue>,
    blank_mapper: &mut BlankNodeIdMapper,
) -> String {
    match obj.get("@id") {
        Some(JsonLdValue::Str(id)) => {
            if id.starts_with("_:") {
                blank_mapper.map(id)
            } else {
                id.clone()
            }
        }
        _ => {
            // Generate an anonymous blank-node ID.
            let anon = format!("_:anon_{}", blank_mapper.counter);
            blank_mapper.map(&anon)
        }
    }
}

/// Resolve a value that may be a node reference, ensuring the referenced node
/// exists and returning the canonical `{"@id": "..."}` reference.
///
/// For non-node values (value objects, scalars) the original value is returned
/// unchanged.
fn resolve_node_reference(
    value: &JsonLdValue,
    node_map: &mut NodeMap,
    active_graph: &str,
    blank_mapper: &mut BlankNodeIdMapper,
) -> JsonLdValue {
    match value {
        JsonLdValue::Object(m) if !m.contains_key("@value") && !m.contains_key("@list") => {
            let id = get_or_assign_id(m, blank_mapper);
            node_map
                .get_or_create_graph(active_graph)
                .get_or_create(&id);
            let mut ref_map: IndexMap<String, JsonLdValue> = IndexMap::new();
            ref_map.insert("@id".to_string(), JsonLdValue::Str(id));
            JsonLdValue::Object(ref_map)
        }
        other => other.clone(),
    }
}

/// Merge a value into a property array without duplicating identical `@value` literals.
///
/// Duplicate detection rules:
/// - Two `{"@value": v}` objects are duplicates if all their entries are equal.
/// - Any other two values are compared by structural equality (`PartialEq`).
pub fn merge_value(into: &mut Vec<JsonLdValue>, value: JsonLdValue) {
    // Check for exact structural duplicates.
    if into.contains(&value) {
        return;
    }
    into.push(value);
}
