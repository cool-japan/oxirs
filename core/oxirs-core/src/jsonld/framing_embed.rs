//! Embed-policy application and recursive subject embedding for JSON-LD 1.1 Framing.
//!
//! Implements §4.5 *Framing Algorithm* — the parts that concern:
//! - Deciding whether to embed, skip, or link a subject (`apply_embed_policy`).
//! - Recursively building an embedded node object (`embed_subject`).
//! - Applying `@explicit` property pruning (`apply_explicit`).
//! - Injecting `@default` values for missing properties (`apply_defaults`).

use serde_json::{Map, Value};
use std::collections::HashMap;

use super::framing::{EmbedPolicy, FramingError, FramingOptions, FramingState};
use super::framing_match::matches_frame;

// ──────────────────────────────────────────────────────────────────────────────
// EmbedDecision
// ──────────────────────────────────────────────────────────────────────────────

/// Decision returned by [`apply_embed_policy`] that controls how a subject is
/// rendered in the framed output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedDecision {
    /// Fully embed the subject, recursively embedding its property values.
    Full,
    /// Emit a `{"@id": "...", "@link": true}` reference node.
    Link,
    /// Emit a bare `{"@id": "..."}` reference node (no recursion).
    Skip,
}

// ──────────────────────────────────────────────────────────────────────────────
// apply_embed_policy
// ──────────────────────────────────────────────────────────────────────────────

/// Decide the embed action for subject with IRI `subject_id` given the current
/// framing state and the configured embed policy.
///
/// | Policy  | First visit | Subsequent visits |
/// |---------|-------------|-------------------|
/// | First   | Full        | Skip              |
/// | Last    | Skip        | Full (overwrite)  |
/// | Always  | Full        | Full (cycle → Skip)|
/// | Never   | Skip        | Skip              |
/// | Link    | Full        | Link              |
pub fn apply_embed_policy(
    state: &mut FramingState,
    subject_id: &str,
    policy: EmbedPolicy,
) -> EmbedDecision {
    let already_embedded = state.embedded.contains(subject_id);

    match policy {
        EmbedPolicy::First => {
            if already_embedded {
                EmbedDecision::Skip
            } else {
                EmbedDecision::Full
            }
        }
        EmbedPolicy::Last => {
            // Always re-embed (the caller updates state.embedded after the call)
            EmbedDecision::Full
        }
        EmbedPolicy::Always => {
            if already_embedded {
                // Cycle detected — emit @id only to break the loop
                EmbedDecision::Skip
            } else {
                EmbedDecision::Full
            }
        }
        EmbedPolicy::Never => EmbedDecision::Skip,
        EmbedPolicy::Link => {
            if already_embedded {
                EmbedDecision::Link
            } else {
                EmbedDecision::Full
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// embed_subject
// ──────────────────────────────────────────────────────────────────────────────

/// Recursively embed `subject` according to the constraints and embed-policy
/// expressed in `frame`.
///
/// Algorithm:
/// 1. Copy `@id` and `@type` from the subject.
/// 2. For each property in the subject (non-keywords), check whether the frame
///    has a sub-frame for that property.
///    - If it does, recurse into the nested frame for each value that is a
///      node object (has `@id`).  Non-node values (literals) are copied as-is.
///    - If it does not, copy the property values verbatim.
/// 3. If `@explicit` is set, only the properties present in the frame survive
///    (that pruning happens in the caller via `apply_explicit`).
pub fn embed_subject(
    state: &mut FramingState,
    subject: &Value,
    frame: &Value,
    subjects: &HashMap<String, Value>,
    options: &FramingOptions,
) -> Result<Value, FramingError> {
    let subj_obj = match subject {
        Value::Object(o) => o,
        _ => return Ok(subject.clone()),
    };

    let frame_obj = match frame {
        Value::Object(o) => o,
        _ => return Ok(subject.clone()),
    };

    let mut output: Map<String, Value> = Map::new();

    // ── Copy @id ────────────────────────────────────────────────────────────
    if let Some(id) = subj_obj.get("@id") {
        output.insert("@id".to_string(), id.clone());
    }

    // ── Copy @type ───────────────────────────────────────────────────────────
    if let Some(types) = subj_obj.get("@type") {
        output.insert("@type".to_string(), types.clone());
    }

    // ── Copy other keywords (@language, @value, @graph, …) ───────────────────
    for (key, val) in subj_obj.iter() {
        if key.starts_with('@') && key != "@id" && key != "@type" {
            output.insert(key.clone(), val.clone());
        }
    }

    // ── Process properties ───────────────────────────────────────────────────
    for (prop, prop_values) in subj_obj.iter() {
        if prop.starts_with('@') {
            continue; // keywords already handled
        }

        let values_arr = match prop_values {
            Value::Array(arr) => arr,
            _ => {
                output.insert(prop.clone(), prop_values.clone());
                continue;
            }
        };

        // Check if the frame has a sub-frame for this property
        let sub_frame: Option<&Value> = frame_obj.get(prop.as_str());

        let mut embedded_values: Vec<Value> = Vec::new();

        for val in values_arr {
            // Is this value a reference to another subject (a node object with @id)?
            if let Some(ref_id) = get_node_id(val) {
                // Try to look up the full subject object
                if let Some(ref_subject) = subjects.get(ref_id) {
                    let default_frame = Value::Object(Map::new());
                    let nested_frame = sub_frame.unwrap_or(&default_frame);

                    // Check if the referenced subject matches the nested frame
                    if matches_frame(ref_subject, nested_frame, options) {
                        let decision = apply_embed_policy(state, ref_id, options.embed.clone());

                        match decision {
                            EmbedDecision::Skip => {
                                embedded_values.push(serde_json::json!({"@id": ref_id}));
                            }
                            EmbedDecision::Link => {
                                if let Some(linked) = state.link.get(ref_id) {
                                    embedded_values.push(linked.clone());
                                } else {
                                    embedded_values.push(serde_json::json!({"@id": ref_id}));
                                }
                            }
                            EmbedDecision::Full => {
                                // Mark as embedded before recursing to detect cycles
                                state.embedded.insert(ref_id.to_string());
                                let mut embedded = embed_subject(
                                    state,
                                    ref_subject,
                                    nested_frame,
                                    subjects,
                                    options,
                                )?;
                                embedded = apply_explicit(&embedded, nested_frame, options);
                                apply_defaults(&mut embedded, nested_frame, options);

                                if options.embed == EmbedPolicy::Link {
                                    state.link.insert(ref_id.to_string(), embedded.clone());
                                }
                                embedded_values.push(embedded);
                            }
                        }
                    } else {
                        // Subject doesn't match the sub-frame — emit as @id reference
                        embedded_values.push(serde_json::json!({"@id": ref_id}));
                    }
                } else {
                    // No full subject in the map — just copy the reference
                    embedded_values.push(val.clone());
                }
            } else {
                // Literal value or value object — copy directly
                embedded_values.push(val.clone());
            }
        }

        output.insert(prop.clone(), Value::Array(embedded_values));
    }

    Ok(Value::Object(output))
}

// ──────────────────────────────────────────────────────────────────────────────
// apply_explicit
// ──────────────────────────────────────────────────────────────────────────────

/// When `@explicit` is `true`, prune all properties from `subject` that are
/// not listed (as keys) in the `frame`.  Keywords (`@id`, `@type`, etc.) are
/// always kept.
pub fn apply_explicit(subject: &Value, frame: &Value, options: &FramingOptions) -> Value {
    if !options.explicit {
        return subject.clone();
    }

    let subj_obj = match subject {
        Value::Object(o) => o,
        _ => return subject.clone(),
    };

    let frame_obj = match frame {
        Value::Object(o) => o,
        _ => return subject.clone(),
    };

    let mut pruned: Map<String, Value> = Map::new();

    for (key, val) in subj_obj {
        // Always keep keywords
        if key.starts_with('@') {
            pruned.insert(key.clone(), val.clone());
            continue;
        }
        // Keep property if it appears in the frame
        if frame_obj.contains_key(key.as_str()) {
            pruned.insert(key.clone(), val.clone());
        }
    }

    Value::Object(pruned)
}

// ──────────────────────────────────────────────────────────────────────────────
// apply_defaults
// ──────────────────────────────────────────────────────────────────────────────

/// When `@omitDefault` is `false` (the default), inject `@default` values from
/// the frame for any property that is missing in the output node.
///
/// Frame properties may have a `@default` keyword at the value level, e.g.:
///
/// ```json
/// {
///   "http://example.org/name": [{"@default": "Unknown"}]
/// }
/// ```
///
/// If the output node does not have `http://example.org/name`, this function
/// adds `[{"@value": "Unknown"}]` for it.
pub fn apply_defaults(output: &mut Value, frame: &Value, options: &FramingOptions) {
    if options.omit_default {
        return; // @omitDefault=true: do not inject defaults
    }

    let output_obj = match output {
        Value::Object(o) => o,
        _ => return,
    };

    let frame_obj = match frame {
        Value::Object(o) => o,
        _ => return,
    };

    for (prop, frame_val) in frame_obj {
        if prop.starts_with('@') {
            continue;
        }

        // Only inject if the property is missing from the output
        if output_obj.contains_key(prop.as_str()) {
            continue;
        }

        // Look for @default in the frame value array
        if let Some(default_val) = find_default_value(frame_val) {
            output_obj.insert(prop.clone(), serde_json::json!([{"@value": default_val}]));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// If `value` is a node object (not a `@value` literal) that has an `@id`,
/// return that IRI.  Otherwise return `None`.
fn get_node_id(value: &Value) -> Option<&str> {
    match value {
        Value::Object(obj) => {
            if obj.contains_key("@value") {
                None // literal value object
            } else {
                obj.get("@id").and_then(|v| v.as_str())
            }
        }
        _ => None,
    }
}

/// Search a frame-property value for a `@default` annotation, returning
/// the default as a raw `Value` reference if found.
fn find_default_value(frame_val: &Value) -> Option<&Value> {
    match frame_val {
        Value::Array(arr) => {
            for item in arr {
                if let Value::Object(obj) = item {
                    if let Some(default) = obj.get("@default") {
                        return Some(default);
                    }
                }
            }
            None
        }
        Value::Object(obj) => obj.get("@default"),
        _ => None,
    }
}
