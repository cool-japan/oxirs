//! Core compaction algorithm steps for JSON-LD 1.1.
//!
//! This module implements:
//! - Value object compaction (`compact_value`)
//! - Node object compaction (`compact_node`)
//! - Array compaction (`compact_array`)
//!
//! All functions follow the W3C JSON-LD 1.1 Compaction Algorithm specification:
//! <https://www.w3.org/TR/json-ld11-api/#compaction-algorithm>

use super::context::{compact_iri, find_term, is_keyword};
use super::{
    CompactionError, CompactionOptions, ContainerType, JsonLdContext, JsonLdValue, TermDefinition,
};
use indexmap::IndexMap;

// ============================================================================
// compact_value — §4.1 Value Object Compaction
// ============================================================================

/// Compact a JSON-LD value object.
///
/// Applies the W3C JSON-LD 1.1 Value Compaction algorithm:
/// <https://www.w3.org/TR/json-ld11-api/#value-compaction>
///
/// # Compaction rules
///
/// 1. If the value is `{"@value": v, "@type": "@json"}` — keep as-is.
/// 2. If the value has `@type` and the active property's term has a matching
///    type mapping — drop the `@type` and `@value` wrapper.
/// 3. If the value has `@language` and the active property's term has a
///    matching language mapping — drop the wrapper.
/// 4. Plain string values without type/language — drop `@value` wrapper.
/// 5. Numeric and boolean values — return native JSON representation.
///
/// # Arguments
///
/// * `active_ctx` — the active context
/// * `active_property` — the property whose value is being compacted
/// * `value` — the expanded value object
pub fn compact_value(
    active_ctx: &JsonLdContext,
    active_property: Option<&str>,
    value: &JsonLdValue,
) -> Result<JsonLdValue, CompactionError> {
    let obj = match value {
        JsonLdValue::Object(m) => m,
        other => return Ok(other.clone()),
    };

    // Get the active term definition (if any).
    let term_def = active_property.and_then(|p| active_ctx.terms.get(p));

    let val = obj.get("@value");
    let typ = obj.get("@type");
    let lang = obj.get("@language");

    // 1. @json type — keep as-is.
    if matches!(typ, Some(JsonLdValue::Str(t)) if t == "@json") {
        return Ok(value.clone());
    }

    // 2. Type mapping match.
    if let Some(JsonLdValue::Str(type_iri)) = typ {
        if let Some(def) = term_def {
            if def.type_mapping.as_deref() == Some(type_iri.as_str()) {
                // Compaction: unwrap @value.
                return Ok(val.cloned().unwrap_or(JsonLdValue::Null));
            }
            if def.type_mapping.as_deref() == Some("@id") {
                // Value is an IRI; compact it.
                if let Some(JsonLdValue::Str(v)) = val {
                    let compacted = compact_iri(active_ctx, v, None, false, false);
                    return Ok(JsonLdValue::Str(compacted));
                }
            }
        }
        // Compact the type IRI.
        let compact_type = compact_iri(active_ctx, type_iri, None, true, false);
        let mut out: IndexMap<String, JsonLdValue> = IndexMap::new();
        out.insert(
            "@value".to_string(),
            val.cloned().unwrap_or(JsonLdValue::Null),
        );
        out.insert("@type".to_string(), JsonLdValue::Str(compact_type));
        return Ok(JsonLdValue::Object(out));
    }

    // 3. Language mapping match.
    if let Some(JsonLdValue::Str(lang_tag)) = lang {
        if let Some(def) = term_def {
            if def.language.as_deref() == Some(lang_tag.as_str()) {
                // Compaction: unwrap @value.
                return Ok(val.cloned().unwrap_or(JsonLdValue::Null));
            }
        }
        // Keep @language but compact if possible.
        let mut out: IndexMap<String, JsonLdValue> = IndexMap::new();
        if let Some(v) = val {
            out.insert("@value".to_string(), v.clone());
        }
        out.insert("@language".to_string(), JsonLdValue::Str(lang_tag.clone()));
        return Ok(JsonLdValue::Object(out));
    }

    // 4. Plain string — can drop @value wrapper.
    if let Some(JsonLdValue::Str(s)) = val {
        if typ.is_none() && lang.is_none() {
            return Ok(JsonLdValue::Str(s.clone()));
        }
    }

    // 5. Numeric value.
    if let Some(JsonLdValue::Number(n)) = val {
        if typ.is_none() && lang.is_none() {
            return Ok(JsonLdValue::Number(*n));
        }
    }

    // 6. Boolean value.
    if let Some(JsonLdValue::Bool(b)) = val {
        if typ.is_none() && lang.is_none() {
            return Ok(JsonLdValue::Bool(*b));
        }
    }

    // Fall back to returning the value object as-is (with compacted keys).
    let mut out: IndexMap<String, JsonLdValue> = IndexMap::new();
    if let Some(v) = val {
        out.insert("@value".to_string(), v.clone());
    }
    if let Some(t) = typ {
        out.insert("@type".to_string(), t.clone());
    }
    if let Some(l) = lang {
        out.insert("@language".to_string(), l.clone());
    }
    Ok(JsonLdValue::Object(out))
}

// ============================================================================
// compact_node — §4.2 Node Object Compaction
// ============================================================================

/// Compact a node object.
///
/// This implements the node object compaction portion of the W3C JSON-LD 1.1
/// Compaction Algorithm.
///
/// # Steps (abbreviated)
///
/// 1. Compact `@id`, `@type` using context term definitions.
/// 2. Handle `@reverse` properties.
/// 3. Handle container types (`@index`, `@graph`, `@set`, `@list`, `@language`).
/// 4. Recursively compact all property values.
///
/// # Arguments
///
/// * `active_ctx` — the current active context
/// * `type_scoped_ctx` — type-scoped context (may differ from `active_ctx` when types change it)
/// * `active_property` — the property under which this node appears
/// * `node` — the expanded node object map
/// * `options` — compaction options
pub fn compact_node(
    active_ctx: &JsonLdContext,
    _type_scoped_ctx: &JsonLdContext,
    _active_property: Option<&str>,
    node: &IndexMap<String, JsonLdValue>,
    options: &CompactionOptions,
) -> Result<JsonLdValue, CompactionError> {
    let mut result: IndexMap<String, JsonLdValue> = IndexMap::new();

    // Step 1: Handle @id.
    if let Some(id_val) = node.get("@id") {
        if let Some(id_str) = id_val.as_str() {
            let compact = compact_iri(active_ctx, id_str, None, false, false);
            result.insert("@id".to_string(), JsonLdValue::Str(compact));
        }
    }

    // Step 2: Handle @type.
    if let Some(type_val) = node.get("@type") {
        let types: Vec<&JsonLdValue> = match type_val {
            JsonLdValue::Array(a) => a.iter().collect(),
            single => vec![single],
        };
        let mut compact_types: Vec<JsonLdValue> = Vec::new();
        for t in types {
            if let Some(type_iri) = t.as_str() {
                let compacted = compact_iri(active_ctx, type_iri, None, true, false);
                compact_types.push(JsonLdValue::Str(compacted));
            }
        }
        let mut type_iter = compact_types.into_iter();
        match (options.compact_arrays, type_iter.next()) {
            (true, Some(single)) if type_iter.len() == 0 => {
                result.insert("@type".to_string(), single);
            }
            (_, first) => {
                let values: Vec<JsonLdValue> = first.into_iter().chain(type_iter).collect();
                result.insert("@type".to_string(), JsonLdValue::Array(values));
            }
        }
    }

    // Step 3: Handle @graph.
    if let Some(graph_val) = node.get("@graph") {
        let compact_key = compact_iri(active_ctx, "@graph", None, true, false);
        let compacted = compact_element_dispatch(graph_val, active_ctx, Some("@graph"), options)?;
        result.insert(compact_key, compacted);
    }

    // Step 4: Handle @reverse properties.
    if let Some(rev_val) = node.get("@reverse") {
        if let Some(rev_obj) = rev_val.as_object() {
            let mut rev_result: IndexMap<String, JsonLdValue> = IndexMap::new();
            for (prop, val) in rev_obj {
                let compact_prop = compact_iri(active_ctx, prop, Some(val), true, true);
                // Check if there is a reverse term for this property.
                let rev_term = find_term(active_ctx, prop, Some(val), &[], "@type", "");
                let compact_val = compact_element_dispatch(val, active_ctx, Some(prop), options)?;
                if let Some(_term) = rev_term {
                    // Map directly into the result object as a reverse-mapped term.
                    result.insert(compact_prop.clone(), compact_val);
                } else {
                    rev_result.insert(compact_prop, compact_val);
                }
            }
            if !rev_result.is_empty() {
                result.insert("@reverse".to_string(), JsonLdValue::Object(rev_result));
            }
        }
    }

    // Step 5: Handle all other properties.
    for (expanded_prop, value) in node {
        // Skip already-handled keywords.
        if matches!(
            expanded_prop.as_str(),
            "@id" | "@type" | "@graph" | "@reverse" | "@context"
        ) {
            continue;
        }

        if is_keyword(expanded_prop) {
            // Map JSON-LD keyword to compact form.
            let compact_key = compact_iri(active_ctx, expanded_prop, None, true, false);
            let compact_val =
                compact_element_dispatch(value, active_ctx, Some(expanded_prop), options)?;
            result.insert(compact_key, compact_val);
            continue;
        }

        // Find the compact property IRI.
        let compact_prop = compact_iri(active_ctx, expanded_prop, Some(value), true, false);

        // Look up the term definition for container typing.
        let term_def = active_ctx.terms.get(&compact_prop);

        // Compact the value(s).
        let values: &[JsonLdValue] = match value {
            JsonLdValue::Array(a) => a.as_slice(),
            single => std::slice::from_ref(single),
        };

        if let Some(def) = term_def {
            // Handle @language container.
            if def.container.contains(&ContainerType::Language) {
                let lang_map = build_language_map(values, active_ctx, expanded_prop, options)?;
                result.insert(compact_prop, lang_map);
                continue;
            }

            // Handle @type container.
            if def.container.contains(&ContainerType::Type) {
                let type_map = build_type_map(values, active_ctx, expanded_prop, options)?;
                result.insert(compact_prop, type_map);
                continue;
            }

            // Handle @index container.
            if def.container.contains(&ContainerType::Index) {
                let idx_map = build_index_map(values, active_ctx, expanded_prop, options)?;
                result.insert(compact_prop, idx_map);
                continue;
            }

            // Handle @list container.
            if def.container.contains(&ContainerType::List) {
                // Unwrap the @list array.
                let list_items = collect_list_items(values);
                let mut compact_items: Vec<JsonLdValue> = Vec::new();
                for item in &list_items {
                    let c =
                        compact_element_dispatch(item, active_ctx, Some(expanded_prop), options)?;
                    compact_items.push(c);
                }
                result.insert(compact_prop, JsonLdValue::Array(compact_items));
                continue;
            }

            // Handle @set container — always keep as array.
            if def.container.contains(&ContainerType::Set) {
                let compact_arr = compact_array(active_ctx, expanded_prop, values, options)?;
                // Force array even for single element.
                let arr = match compact_arr {
                    JsonLdValue::Array(a) => JsonLdValue::Array(a),
                    single => JsonLdValue::Array(vec![single]),
                };
                result.insert(compact_prop, arr);
                continue;
            }
        }

        // Default: compact value(s).
        let compact_val = if values.len() == 1 && options.compact_arrays {
            // Try to compact single value.
            let v = compact_element_dispatch(&values[0], active_ctx, Some(expanded_prop), options)?;
            // But if the term requires an array container, keep it.
            let needs_array = term_def
                .map(|d| {
                    d.container.contains(&ContainerType::Set)
                        || d.container.contains(&ContainerType::List)
                })
                .unwrap_or(false);
            if needs_array {
                JsonLdValue::Array(vec![v])
            } else {
                v
            }
        } else {
            compact_array(active_ctx, expanded_prop, values, options)?
        };

        // Merge into result (handle duplicate keys by collecting into array).
        insert_or_merge(&mut result, compact_prop, compact_val);
    }

    // Optionally sort keys for deterministic output.
    if options.ordered {
        result.sort_keys();
    }

    Ok(JsonLdValue::Object(result))
}

// ============================================================================
// compact_array — §4.3 Array Compaction
// ============================================================================

/// Compact an array of JSON-LD values.
///
/// If `compact_arrays=true` and the array has exactly one element, and the
/// active property's container does not require an array, the element is
/// returned unwrapped (as a scalar).
///
/// # Arguments
///
/// * `active_ctx` — the active context
/// * `active_property` — the property under which this array appears
/// * `array` — the array elements to compact
/// * `options` — compaction options
pub fn compact_array(
    active_ctx: &JsonLdContext,
    active_property: &str,
    array: &[JsonLdValue],
    options: &CompactionOptions,
) -> Result<JsonLdValue, CompactionError> {
    let mut out: Vec<JsonLdValue> = Vec::with_capacity(array.len());

    for item in array {
        let compacted = compact_element_dispatch(item, active_ctx, Some(active_property), options)?;
        out.push(compacted);
    }

    // Determine if we can collapse to a single value.
    let term_def = active_ctx.terms.get(active_property);
    let requires_array = term_def
        .map(|d| {
            d.container.contains(&ContainerType::Set) || d.container.contains(&ContainerType::List)
        })
        .unwrap_or(false);

    if options.compact_arrays && out.len() == 1 && !requires_array {
        Ok(out.into_iter().next().unwrap_or(JsonLdValue::Null))
    } else {
        Ok(JsonLdValue::Array(out))
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Dispatch compaction for any value type (node, value, array, scalar).
fn compact_element_dispatch(
    value: &JsonLdValue,
    ctx: &JsonLdContext,
    active_property: Option<&str>,
    options: &CompactionOptions,
) -> Result<JsonLdValue, CompactionError> {
    match value {
        JsonLdValue::Object(map) => {
            if map.contains_key("@value") {
                compact_value(ctx, active_property, value)
            } else if map.contains_key("@list") {
                let items = match map.get("@list") {
                    Some(JsonLdValue::Array(a)) => a.as_slice(),
                    _ => &[],
                };
                compact_array(ctx, active_property.unwrap_or("@list"), items, options)
            } else {
                compact_node(ctx, ctx, active_property, map, options)
            }
        }
        JsonLdValue::Array(items) => {
            compact_array(ctx, active_property.unwrap_or("@graph"), items, options)
        }
        other => Ok(other.clone()),
    }
}

/// Build a language-keyed map from an array of value objects.
///
/// Example: `[{"@value": "hello", "@language": "en"}, {"@value": "bonjour", "@language": "fr"}]`
/// becomes `{"en": "hello", "fr": "bonjour"}`.
fn build_language_map(
    values: &[JsonLdValue],
    _ctx: &JsonLdContext,
    _active_property: &str,
    options: &CompactionOptions,
) -> Result<JsonLdValue, CompactionError> {
    let mut lang_map: IndexMap<String, JsonLdValue> = IndexMap::new();
    for val in values {
        // For a language container, the value object `{"@value": v, "@language": lang}`
        // compacts to the key=lang, value=v form.
        let (lang_key, compacted) = match val {
            JsonLdValue::Object(obj) => {
                let lang = obj
                    .get("@language")
                    .and_then(|l| l.as_str())
                    .unwrap_or("@none")
                    .to_string();
                // The compacted value is the plain @value string.
                let v = obj.get("@value").cloned().unwrap_or(JsonLdValue::Null);
                (lang, v)
            }
            other => ("@none".to_string(), other.clone()),
        };
        insert_or_merge(&mut lang_map, lang_key, compacted);
    }
    // Optionally sort.
    if options.ordered {
        lang_map.sort_keys();
    }
    Ok(JsonLdValue::Object(lang_map))
}

/// Build a type-keyed map from an array of value objects.
fn build_type_map(
    values: &[JsonLdValue],
    ctx: &JsonLdContext,
    active_property: &str,
    options: &CompactionOptions,
) -> Result<JsonLdValue, CompactionError> {
    let mut type_map: IndexMap<String, JsonLdValue> = IndexMap::new();
    for val in values {
        let compacted = compact_value(ctx, Some(active_property), val)?;
        let type_key = match val {
            JsonLdValue::Object(obj) => obj
                .get("@type")
                .and_then(|t| t.as_str())
                .map(|t| compact_iri(ctx, t, None, true, false))
                .unwrap_or_else(|| "@none".to_string()),
            _ => "@none".to_string(),
        };
        insert_or_merge(&mut type_map, type_key, compacted);
    }
    if options.ordered {
        type_map.sort_keys();
    }
    Ok(JsonLdValue::Object(type_map))
}

/// Build an index-keyed map from an array of node/value objects with `@index`.
fn build_index_map(
    values: &[JsonLdValue],
    ctx: &JsonLdContext,
    active_property: &str,
    options: &CompactionOptions,
) -> Result<JsonLdValue, CompactionError> {
    let mut idx_map: IndexMap<String, JsonLdValue> = IndexMap::new();
    for val in values {
        let idx_key = match val {
            JsonLdValue::Object(obj) => obj
                .get("@index")
                .and_then(|i| i.as_str())
                .unwrap_or("@none")
                .to_string(),
            _ => "@none".to_string(),
        };
        let compacted = compact_element_dispatch(val, ctx, Some(active_property), options)?;
        insert_or_merge(&mut idx_map, idx_key, compacted);
    }
    if options.ordered {
        idx_map.sort_keys();
    }
    Ok(JsonLdValue::Object(idx_map))
}

/// Collect items from `@list` wrappers.
fn collect_list_items(values: &[JsonLdValue]) -> Vec<&JsonLdValue> {
    let mut items: Vec<&JsonLdValue> = Vec::new();
    for val in values {
        match val {
            JsonLdValue::Object(obj) => {
                if let Some(JsonLdValue::Array(list)) = obj.get("@list") {
                    items.extend(list.iter());
                } else {
                    items.push(val);
                }
            }
            _ => items.push(val),
        }
    }
    items
}

/// Insert a value into a map, merging with existing values via an array.
fn insert_or_merge(map: &mut IndexMap<String, JsonLdValue>, key: String, value: JsonLdValue) {
    if let Some(existing) = map.get_mut(&key) {
        // First check if it's already an array so we can push without reconstruction.
        if let JsonLdValue::Array(arr) = existing {
            arr.push(value);
            return;
        }
        // Not an array — replace with a two-element array.
        let prev = std::mem::replace(existing, JsonLdValue::Null);
        *existing = JsonLdValue::Array(vec![prev, value]);
    } else {
        map.insert(key, value);
    }
}

/// Add a simple term definition (IRI prefix) to the context for compaction use.
pub fn add_prefix_term(ctx: &mut JsonLdContext, prefix: impl Into<String>, iri: impl Into<String>) {
    ctx.terms
        .insert(prefix.into(), TermDefinition::prefix(iri.into()));
}
