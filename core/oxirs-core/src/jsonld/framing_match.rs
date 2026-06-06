//! Subject-frame matching for JSON-LD 1.1 Framing.
//!
//! Implements §4.5 *Framing Algorithm* — the matching predicates that decide
//! whether a given subject node satisfies the constraints expressed in a frame.

use serde_json::Value;

use super::framing::FramingOptions;

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Check whether `subject` satisfies all frame constraints.
///
/// The algorithm:
/// 1. If the frame has `@type`, check `matches_type`.
/// 2. If the frame has `@id`, check `matches_id`.
/// 3. For every non-keyword property in the frame, check `matches_property`.
///    When `options.require_all` is `true`, **all** properties must match;
///    otherwise **any** matching property is sufficient.
///
/// An empty frame `{}` matches every subject.
pub fn matches_frame(subject: &Value, frame: &Value, options: &FramingOptions) -> bool {
    let frame_obj = match frame {
        Value::Object(o) => o,
        // An array frame: match if any element of the frame array matches
        Value::Array(arr) => {
            return arr.iter().any(|f| matches_frame(subject, f, options));
        }
        // A non-object, non-array frame matches nothing
        _ => return false,
    };

    // ── @type matching ─────────────────────────────────────────────────────────
    if let Some(frame_types) = frame_obj.get("@type") {
        if !matches_type(subject, frame_types) {
            return false;
        }
    }

    // ── @id matching ───────────────────────────────────────────────────────────
    if let Some(frame_id) = frame_obj.get("@id") {
        if !matches_id(subject, frame_id) {
            return false;
        }
    }

    // ── property matching ──────────────────────────────────────────────────────
    let prop_entries: Vec<(&String, &Value)> = frame_obj
        .iter()
        .filter(|(k, _)| !k.starts_with('@'))
        .collect();

    if prop_entries.is_empty() {
        return true; // no property constraints → matches everything
    }

    if options.require_all {
        // ALL non-keyword frame properties must match
        prop_entries
            .iter()
            .all(|(prop, fval)| matches_property(subject, prop, fval, options))
    } else {
        // ANY non-keyword frame property match is sufficient
        prop_entries
            .iter()
            .any(|(prop, fval)| matches_property(subject, prop, fval, options))
    }
}

/// Check that the subject's `@type` satisfies the frame's `@type` constraint.
///
/// The frame's `@type` may be:
/// - A JSON string (single type IRI)
/// - An array of type IRIs
///
/// The subject must have **at least one** type in common with the frame's type
/// array.  If `require_all` is set on the caller's options, the call-site
/// (`matches_frame`) already handles that at the property level; here we only
/// need to satisfy the "any overlap" rule for `@type` matching as specified in
/// §4.4.1.
pub fn matches_type(subject: &Value, frame_types: &Value) -> bool {
    let subj_types = collect_type_strings(subject);
    let frame_type_set = collect_frame_type_strings(frame_types);

    if frame_type_set.is_empty() {
        return true; // empty frame @type → wildcard
    }

    // At least one type in the frame must appear in the subject
    frame_type_set
        .iter()
        .any(|ft| subj_types.contains(&ft.as_str()))
}

/// Check that the subject's `@id` satisfies the frame's `@id` constraint.
///
/// The frame's `@id` may be a string, an array of strings, or an empty object
/// `{}` (wildcard — match any subject with an `@id`).
pub fn matches_id(subject: &Value, frame_id: &Value) -> bool {
    let subj_id = match subject.get("@id") {
        Some(Value::String(s)) => s.as_str(),
        _ => return false, // subject has no @id
    };

    match frame_id {
        Value::String(fid) => subj_id == fid.as_str(),
        Value::Array(ids) => ids
            .iter()
            .any(|v| matches!(v, Value::String(s) if s == subj_id)),
        Value::Object(o) if o.is_empty() => true, // wildcard: `{"@id": {}}`
        _ => false,
    }
}

/// Check that `subject` has the given property `prop` and that at least one of
/// the subject's property values matches the frame value pattern `frame_value`.
///
/// Frame values are arrays in expanded JSON-LD.  Each element of the subject's
/// property value array is tested against each element of the frame value
/// array; a match occurs when `matches_value_pattern` returns `true` for any pair.
pub fn matches_property(
    subject: &Value,
    prop: &str,
    frame_value: &Value,
    _options: &FramingOptions,
) -> bool {
    let subj_values = match subject.get(prop) {
        Some(Value::Array(arr)) => arr,
        Some(single) => {
            // Non-array property value — treat as single element
            return matches_value_pattern(single, frame_value);
        }
        None => return false,
    };

    // Frame value is an array of patterns in expanded JSON-LD
    let frame_patterns: Vec<&Value> = match frame_value {
        Value::Array(arr) => arr.iter().collect(),
        other => vec![other],
    };

    // Subject must have at least one value that matches at least one pattern
    subj_values.iter().any(|sv| {
        frame_patterns
            .iter()
            .any(|fp| matches_value_pattern(sv, fp))
    })
}

/// Test whether a single subject value matches a frame value pattern.
///
/// Rules (per §4.4.4 of the W3C spec):
/// - An empty object `{}` is a wildcard: it matches any value.
/// - A `{"@value": ...}` pattern matches `@value` objects with equal `@value`,
///   `@type`, and `@language` (if specified in the pattern).
/// - A node object (with `@id`) matches another node object if the IDs are equal.
/// - A primitive JSON scalar matches equal scalars.
pub fn matches_value_pattern(value: &Value, pattern: &Value) -> bool {
    match (value, pattern) {
        // Wildcard: empty object matches anything
        (_, Value::Object(po)) if po.is_empty() => true,

        // @value object matching
        (Value::Object(vo), Value::Object(po)) if po.contains_key("@value") => {
            // The pattern @value must equal the subject's @value
            let pv = po.get("@value");
            let vv = vo.get("@value");
            if pv != vv {
                return false;
            }
            // @type must match if specified
            if let Some(pt) = po.get("@type") {
                if vo.get("@type") != Some(pt) {
                    return false;
                }
            }
            // @language must match if specified
            if let Some(pl) = po.get("@language") {
                if vo.get("@language") != Some(pl) {
                    return false;
                }
            }
            true
        }

        // Node object matching by @id
        (Value::Object(vo), Value::Object(po))
            if po.contains_key("@id") && !po.contains_key("@value") =>
        {
            vo.get("@id") == po.get("@id")
        }

        // Any node object pattern without @id or @value: wildcard
        (Value::Object(_), Value::Object(po)) if !po.contains_key("@id") => true,

        // Primitive scalar equality
        (Value::String(vs), Value::String(ps)) => vs == ps,
        (Value::Number(vn), Value::Number(pn)) => vn == pn,
        (Value::Bool(vb), Value::Bool(pb)) => vb == pb,
        (Value::Null, Value::Null) => true,

        _ => false,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Collect the `@type` IRI strings from a subject node.
fn collect_type_strings(subject: &Value) -> Vec<&str> {
    match subject.get("@type") {
        Some(Value::Array(arr)) => arr.iter().filter_map(|v| v.as_str()).collect(),
        Some(Value::String(s)) => vec![s.as_str()],
        _ => vec![],
    }
}

/// Collect the type IRI strings from a frame's `@type` value (may be string or
/// array).
fn collect_frame_type_strings(frame_types: &Value) -> Vec<String> {
    match frame_types {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        _ => vec![],
    }
}
