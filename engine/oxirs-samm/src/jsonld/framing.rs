//! JSON-LD 1.1 Framing Algorithm
//!
//! Implements the framing step of the JSON-LD processing pipeline described
//! in the [W3C JSON-LD 1.1 Framing specification][spec] and the
//! [SAMM specification §mapping-to-json-ld][samm-spec].
//!
//! Framing allows a JSON-LD document to be reshaped into a specific tree
//! structure (the *frame*) that matches an application's expected shape.
//!
//! ## Algorithm summary
//!
//! 1. Collect all nodes from the input document's `@graph` array (or the
//!    root object if no `@graph` is present).
//! 2. For each node, check whether it *matches* the frame:
//!    - If the frame has a `@type` key, the node's `@type` must contain all
//!      types listed in the frame.
//!    - If the frame specifies other properties, those keys must exist in the
//!      node with the requested shape.
//! 3. For each matching node, build the output subtree by:
//!    - Retaining only the properties mentioned in the frame (property
//!      filtering).
//!    - Recursively applying the frame to nested objects.
//! 4. Wrap the result set in `{"@graph": [...]}`.
//!
//! [spec]: https://www.w3.org/TR/json-ld11-framing/
//! [samm-spec]: https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#mapping-to-json-ld

use serde_json::{Map, Value};

use super::compaction::JsonLdError;

// ------------------------------------------------------------------ //
//  Public API                                                         //
// ------------------------------------------------------------------ //

/// JSON-LD 1.1 document framer.
///
/// Reshapes a JSON-LD document to match the structure defined by a *frame*
/// object.  The resulting document is always wrapped in a top-level
/// `{"@graph": [...]}` envelope.
///
/// # Example
///
/// ```rust
/// use oxirs_samm::jsonld::framing::JsonLdFramer;
/// use serde_json::json;
///
/// let doc = json!({
///     "@graph": [
///         { "@id": "ex:sensor1", "@type": ["ex:Sensor"], "ex:reading": 42 },
///         { "@id": "ex:other",  "@type": ["ex:Device"]                   }
///     ]
/// });
///
/// let frame = json!({ "@type": ["ex:Sensor"] });
///
/// let framer = JsonLdFramer;
/// let result = framer.frame(&doc, &frame).expect("framing should succeed");
///
/// let graph = result["@graph"].as_array().expect("@graph must be array");
/// assert_eq!(graph.len(), 1, "only the Sensor node should match");
/// assert_eq!(graph[0]["@id"], "ex:sensor1");
/// ```
#[derive(Debug, Clone, Default)]
pub struct JsonLdFramer;

impl JsonLdFramer {
    /// Create a new framer instance.
    pub fn new() -> Self {
        Self
    }

    /// Apply `frame` to `document` and return a shaped JSON-LD document.
    ///
    /// The returned value always has the form:
    ///
    /// ```json
    /// { "@graph": [ ...matching nodes... ] }
    /// ```
    ///
    /// If no node in the document matches the frame, the `@graph` array is
    /// empty.
    pub fn frame(&self, document: &Value, frame: &Value) -> Result<Value, JsonLdError> {
        let nodes = collect_nodes(document);
        let frame_obj = frame
            .as_object()
            .ok_or_else(|| JsonLdError::Processing("frame must be a JSON object".to_string()))?;

        let mut graph: Vec<Value> = Vec::new();
        for node in nodes {
            if matches_frame(node, frame_obj) {
                let shaped = shape_node(node, frame_obj)?;
                graph.push(shaped);
            }
        }

        Ok(Value::Object({
            let mut root = Map::new();
            root.insert("@graph".to_string(), Value::Array(graph));
            root
        }))
    }
}

// ------------------------------------------------------------------ //
//  Node collection                                                    //
// ------------------------------------------------------------------ //

/// Collect all top-level node objects from a JSON-LD document.
///
/// The function handles two layouts:
/// - A document with a `@graph` array: returns the elements of that array.
/// - A plain object (no `@graph`): wraps it in a single-element slice.
/// - An array at the document root: returns each element that is an object.
fn collect_nodes(document: &Value) -> Vec<&Value> {
    match document {
        Value::Object(obj) => {
            if let Some(Value::Array(graph)) = obj.get("@graph") {
                graph.iter().filter(|v| v.is_object()).collect()
            } else {
                vec![document]
            }
        }
        Value::Array(arr) => arr.iter().filter(|v| v.is_object()).collect(),
        _ => vec![],
    }
}

// ------------------------------------------------------------------ //
//  Frame matching                                                     //
// ------------------------------------------------------------------ //

/// Return `true` if `node` satisfies all constraints expressed in `frame`.
///
/// Matching rules (applied in order):
/// 1. **`@type` matching** – every type listed in the frame's `@type` array
///    must be present in the node's `@type` value.
/// 2. **Property existence** – for every non-keyword key in the frame, the
///    corresponding key must exist in the node.  A frame value of `{}` (empty
///    object) acts as a wildcard: the key just needs to be present.
fn matches_frame(node: &Value, frame: &Map<String, Value>) -> bool {
    let node_obj = match node.as_object() {
        Some(o) => o,
        None => return false,
    };

    for (frame_key, frame_val) in frame {
        match frame_key.as_str() {
            "@type" => {
                if !matches_type(node_obj, frame_val) {
                    return false;
                }
            }
            // Other JSON-LD keywords are ignored during node matching
            k if k.starts_with('@') => {}
            // Non-keyword: the node must have this property
            key => {
                if !node_obj.contains_key(key) {
                    return false;
                }
                // If the frame value is a non-empty object, recurse to check
                // the sub-shape.
                if let Some(sub_frame) = frame_val.as_object() {
                    if !sub_frame.is_empty() {
                        let node_sub = match node_obj.get(key) {
                            Some(v) => v,
                            None => return false,
                        };
                        let sub_nodes = collect_nodes(node_sub);
                        let any_match = sub_nodes.iter().any(|sn| matches_frame(sn, sub_frame));
                        if !any_match {
                            return false;
                        }
                    }
                }
            }
        }
    }

    true
}

/// Check whether the `@type` constraint expressed in `frame_type` is satisfied
/// by the node object.
///
/// The frame may express `@type` as:
/// - A JSON array (all entries must match the node).
/// - A bare string (equivalent to a single-element array).
fn matches_type(node_obj: &Map<String, Value>, frame_type: &Value) -> bool {
    let node_types = collect_type_strings(node_obj.get("@type"));

    match frame_type {
        Value::Array(required_types) => required_types
            .iter()
            .filter_map(|t| t.as_str())
            .all(|t| node_types.contains(&t.to_string())),
        Value::String(s) => node_types.contains(s),
        _ => true, // non-constraining frame @type value
    }
}

/// Collect the type strings from a `@type` value which may be an array or a
/// bare string.
fn collect_type_strings(type_val: Option<&Value>) -> Vec<String> {
    match type_val {
        None => vec![],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => vec![],
    }
}

// ------------------------------------------------------------------ //
//  Node shaping                                                       //
// ------------------------------------------------------------------ //

/// Produce the output node by filtering and recursively shaping `node`
/// according to `frame`.
///
/// If `frame` has no non-keyword property constraints (i.e. the frame is
/// effectively `{}` or only contains `@type`), all properties of the node are
/// retained.  Otherwise only the properties listed in the frame are kept.
fn shape_node(node: &Value, frame: &Map<String, Value>) -> Result<Value, JsonLdError> {
    let node_obj = match node.as_object() {
        Some(o) => o,
        None => return Ok(node.clone()),
    };

    // Determine which non-keyword property keys the frame constrains.
    let frame_props: Vec<&str> = frame
        .keys()
        .filter(|k| !k.starts_with('@'))
        .map(|k| k.as_str())
        .collect();

    let mut out = Map::new();

    // Always copy over standard JSON-LD keywords from the source node.
    for kw in &["@id", "@type", "@graph"] {
        if let Some(v) = node_obj.get(*kw) {
            out.insert((*kw).to_string(), v.clone());
        }
    }

    if frame_props.is_empty() {
        // No property filtering: copy all non-keyword properties.
        for (k, v) in node_obj {
            if !k.starts_with('@') {
                out.insert(k.clone(), v.clone());
            }
        }
    } else {
        // Property filtering: only retain what the frame asks for.
        for prop in frame_props {
            if let Some(val) = node_obj.get(prop) {
                // Recurse if the frame provides a sub-frame for this property.
                let shaped_val =
                    if let Some(sub_frame) = frame.get(prop).and_then(|fv| fv.as_object()) {
                        if sub_frame.is_empty() {
                            val.clone()
                        } else {
                            shape_nested(val, sub_frame)?
                        }
                    } else {
                        val.clone()
                    };
                out.insert(prop.to_string(), shaped_val);
            }
        }
    }

    Ok(Value::Object(out))
}

/// Recursively apply a sub-frame to a nested value (which may itself be an
/// object, an array of objects, or a scalar).
fn shape_nested(val: &Value, sub_frame: &Map<String, Value>) -> Result<Value, JsonLdError> {
    match val {
        Value::Object(_) => shape_node(val, sub_frame),
        Value::Array(arr) => {
            let shaped: Result<Vec<Value>, JsonLdError> = arr
                .iter()
                .map(|v| {
                    if v.is_object() {
                        shape_node(v, sub_frame)
                    } else {
                        Ok(v.clone())
                    }
                })
                .collect();
            Ok(Value::Array(shaped?))
        }
        _ => Ok(val.clone()),
    }
}

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sensor_doc() -> Value {
        json!({
            "@graph": [
                {
                    "@id":   "ex:sensor1",
                    "@type": ["ex:Sensor"],
                    "ex:reading": 42,
                    "ex:label":   "Alpha"
                },
                {
                    "@id":   "ex:sensor2",
                    "@type": ["ex:Sensor", "ex:Device"],
                    "ex:reading": 77,
                    "ex:label":   "Beta"
                },
                {
                    "@id":   "ex:device1",
                    "@type": ["ex:Device"],
                    "ex:model": "ModelX"
                }
            ]
        })
    }

    // ---------------------------------------------------------------- //
    // Test 1: Frame matches nodes with matching @type                   //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_frame_matches_by_type() {
        let doc = sensor_doc();
        let frame = json!({ "@type": ["ex:Sensor"] });
        let framer = JsonLdFramer;
        let result = framer.frame(&doc, &frame).expect("framing should succeed");
        let graph = result["@graph"].as_array().expect("@graph must be array");
        // Both sensor1 and sensor2 carry ex:Sensor
        assert_eq!(graph.len(), 2, "both ex:Sensor nodes should match");
        let ids: Vec<&str> = graph.iter().filter_map(|n| n["@id"].as_str()).collect();
        assert!(ids.contains(&"ex:sensor1"));
        assert!(ids.contains(&"ex:sensor2"));
    }

    // ---------------------------------------------------------------- //
    // Test 2: Frame extracts specified properties only                  //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_frame_extracts_specified_properties() {
        let doc = sensor_doc();
        // Only request "ex:reading" — "ex:label" must be absent.
        let frame = json!({
            "@type": ["ex:Sensor"],
            "ex:reading": {}
        });
        let framer = JsonLdFramer;
        let result = framer.frame(&doc, &frame).expect("framing should succeed");
        let graph = result["@graph"].as_array().expect("@graph must be array");
        assert_eq!(graph.len(), 2);
        for node in graph {
            assert!(
                node.get("ex:reading").is_some(),
                "ex:reading must be present in shaped node"
            );
            assert!(
                node.get("ex:label").is_none(),
                "ex:label must NOT be present when not in frame"
            );
        }
    }

    // ---------------------------------------------------------------- //
    // Test 3: Frame handles nested objects recursively                  //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_frame_handles_nested_objects() {
        let doc = json!({
            "@graph": [
                {
                    "@id":  "ex:complex1",
                    "@type": ["ex:Complex"],
                    "ex:child": {
                        "@id":  "ex:child1",
                        "@type": ["ex:Child"],
                        "ex:childProp": "value",
                        "ex:extra":     "should-be-dropped"
                    },
                    "ex:other": "top-level-other"
                }
            ]
        });

        // Frame requests complex with its child's childProp only
        let frame = json!({
            "@type": ["ex:Complex"],
            "ex:child": {
                "ex:childProp": {}
            }
        });

        let framer = JsonLdFramer;
        let result = framer.frame(&doc, &frame).expect("framing should succeed");
        let graph = result["@graph"].as_array().expect("@graph");
        assert_eq!(graph.len(), 1);

        let child = &graph[0]["ex:child"];
        assert!(child.is_object(), "ex:child must be an object");
        assert!(
            child.get("ex:childProp").is_some(),
            "ex:childProp must be preserved in shaped child"
        );
        assert!(
            child.get("ex:extra").is_none(),
            "ex:extra must be dropped by property filtering"
        );
    }

    // ---------------------------------------------------------------- //
    // Test 4: Frame returns empty @graph for no-match document          //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_frame_empty_graph_no_match() {
        let doc = sensor_doc();
        let frame = json!({ "@type": ["ex:NonExistent"] });
        let framer = JsonLdFramer;
        let result = framer.frame(&doc, &frame).expect("framing should succeed");
        let graph = result["@graph"].as_array().expect("@graph must be array");
        assert!(
            graph.is_empty(),
            "no nodes should match a non-existent @type"
        );
    }

    // ---------------------------------------------------------------- //
    // Test 5: Document without @graph array still works                 //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_frame_plain_object_no_graph() {
        let doc = json!({
            "@id":   "ex:single",
            "@type": ["ex:Single"],
            "ex:prop": "hello"
        });
        let frame = json!({ "@type": ["ex:Single"] });
        let framer = JsonLdFramer;
        let result = framer.frame(&doc, &frame).expect("framing should succeed");
        let graph = result["@graph"].as_array().expect("@graph");
        assert_eq!(graph.len(), 1);
        assert_eq!(graph[0]["@id"], "ex:single");
    }

    // ---------------------------------------------------------------- //
    // Test 6: Multi-type node matches frame requiring subset of types   //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_frame_matches_subset_of_types() {
        let doc = sensor_doc();
        // sensor2 has both ex:Sensor and ex:Device — it must match a Device frame.
        let frame = json!({ "@type": ["ex:Device"] });
        let framer = JsonLdFramer;
        let result = framer.frame(&doc, &frame).expect("framing should succeed");
        let graph = result["@graph"].as_array().expect("@graph");
        // sensor2 and device1 both have ex:Device
        assert_eq!(graph.len(), 2);
    }

    // ---------------------------------------------------------------- //
    // Test 7: Frame with no type constraint matches all nodes           //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_frame_no_type_matches_all() {
        let doc = sensor_doc();
        // Empty frame: every node matches.
        let frame = json!({});
        let framer = JsonLdFramer;
        let result = framer.frame(&doc, &frame).expect("framing should succeed");
        let graph = result["@graph"].as_array().expect("@graph");
        assert_eq!(
            graph.len(),
            3,
            "empty frame must match all nodes in the document"
        );
    }

    // ---------------------------------------------------------------- //
    // Test 8: Result is always wrapped in @graph envelope               //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_result_always_has_graph_envelope() {
        let framer = JsonLdFramer;
        let doc = json!({ "@id": "ex:x", "@type": ["ex:X"] });
        let frame = json!({ "@type": ["ex:X"] });
        let result = framer.frame(&doc, &frame).expect("framing should succeed");
        assert!(
            result.get("@graph").is_some(),
            "framing result must always have a @graph key"
        );
        assert!(result["@graph"].is_array(), "@graph value must be an array");
    }
}
