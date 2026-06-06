//! Context manipulation for JSON-LD compaction.
//!
//! This module implements IRI compaction, term lookup, and context serialization
//! as required by the W3C JSON-LD 1.1 Compaction Algorithm.

use super::{ContainerType, JsonLdContext, JsonLdValue};
use indexmap::IndexMap;

// ============================================================================
// compact_iri — §4.2.5 IRI Compaction Algorithm
// ============================================================================

/// Compact an IRI to its shortest form using the active context.
///
/// This implements the W3C JSON-LD 1.1 IRI Compaction Algorithm.
/// See <https://www.w3.org/TR/json-ld11-api/#iri-compaction>.
///
/// # Arguments
///
/// * `ctx` — the active context
/// * `iri` — the IRI to compact
/// * `value` — optional value object for container matching
/// * `vocab` — if `true`, attempt vocabulary mapping
/// * `reverse` — if `true`, look for reverse property terms
///
/// # Returns
///
/// The shortest compact form for the IRI.
pub fn compact_iri(
    ctx: &JsonLdContext,
    iri: &str,
    _value: Option<&JsonLdValue>,
    vocab: bool,
    reverse: bool,
) -> String {
    // Step 1: If IRI is null/empty, return as-is.
    if iri.is_empty() {
        return iri.to_string();
    }

    // Step 2: Check for keyword mapping.
    if is_keyword(iri) {
        return iri.to_string();
    }

    // Step 3: If vocab=true and there is a vocabulary mapping, attempt vocab-relative compaction.
    if vocab {
        if let Some(vocab_iri) = &ctx.vocab {
            if let Some(suffix) = iri.strip_prefix(vocab_iri.as_str()) {
                // Check that no term already has this suffix as its name with a different IRI.
                if !suffix.is_empty() && !suffix.contains(':') {
                    // Verify there is no conflicting term definition.
                    let conflicts = ctx
                        .terms
                        .get(suffix)
                        .is_some_and(|def| def.iri_mapping.as_deref() != Some(iri));
                    if !conflicts {
                        return suffix.to_string();
                    }
                }
            }
        }
    }

    // Step 4: Search for an exact term match.
    let mut best_term: Option<String> = None;
    let mut best_len = usize::MAX;

    for (term, def) in &ctx.terms {
        if def.reverse_property != reverse {
            continue;
        }
        if let Some(mapping) = &def.iri_mapping {
            if mapping == iri {
                // Exact match — pick shortest term.
                if term.len() < best_len {
                    best_len = term.len();
                    best_term = Some(term.clone());
                }
            }
        }
    }

    if let Some(term) = best_term {
        return term;
    }

    // Step 5: Try prefix matching (CURIE compaction).
    let mut best_curie: Option<String> = None;
    let mut best_curie_len = usize::MAX;

    for (term, def) in &ctx.terms {
        if !def.prefix_flag {
            continue;
        }
        if let Some(mapping) = &def.iri_mapping {
            if let Some(suffix) = iri.strip_prefix(mapping.as_str()) {
                if suffix.is_empty() {
                    continue;
                }
                // suffix must not be absolute IRI
                if !suffix.contains("://") {
                    let curie = format!("{}:{}", term, suffix);
                    if curie.len() < best_curie_len {
                        best_curie_len = curie.len();
                        best_curie = Some(curie);
                    }
                }
            }
        }
    }

    if let Some(curie) = best_curie {
        return curie;
    }

    // Step 6: Try base IRI compaction (only if vocab=false).
    if !vocab {
        if let Some(base) = &ctx.base {
            if let Some(relative) = iri.strip_prefix(base.as_str()) {
                if !relative.is_empty() && !relative.starts_with('/') {
                    return relative.to_string();
                }
            }
        }
    }

    // Step 7: Return the absolute IRI.
    iri.to_string()
}

// ============================================================================
// find_term — find best matching term for IRI + container + type/language
// ============================================================================

/// Find the best matching term for an IRI given value and container constraints.
///
/// This is a helper used internally by the compaction algorithm to locate
/// the most specific term definition that matches the given IRI,  value
/// type/language constraints, and container types.
///
/// # Arguments
///
/// * `ctx` — the active context
/// * `iri` — the expanded IRI to match
/// * `value` — optional value object for type/language introspection
/// * `containers` — required container types
/// * `type_language` — `"@type"` or `"@language"` (for value object matching)
/// * `type_language_value` — the concrete type IRI or language tag
///
/// # Returns
///
/// The best matching term, or `None` if no term matches.
pub fn find_term(
    ctx: &JsonLdContext,
    iri: &str,
    value: Option<&JsonLdValue>,
    containers: &[ContainerType],
    type_language: &str,
    type_language_value: &str,
) -> Option<String> {
    let mut best_term: Option<String> = None;
    let mut best_score: i32 = -1;

    for (term, def) in &ctx.terms {
        // Term must map to the given IRI.
        let mapped_iri = def.iri_mapping.as_deref().unwrap_or("");
        if mapped_iri != iri {
            continue;
        }

        // Score: higher is better.
        let mut score: i32 = 0;

        // Container match.
        let has_all_containers = containers.iter().all(|c| def.container.contains(c));
        if !containers.is_empty() && has_all_containers {
            score += 10;
        }

        // Type/language match.
        if type_language == "@type" {
            if let Some(tm) = &def.type_mapping {
                if tm == type_language_value {
                    score += 5;
                }
            }
        } else if type_language == "@language" {
            if let Some(lm) = &def.language {
                if lm == type_language_value {
                    score += 5;
                }
            }
        }

        // Prefer value-matching terms.
        if let Some(val) = value {
            if let Some(obj) = val.as_object() {
                // Check @type match.
                if let Some(JsonLdValue::Str(vtype)) = obj.get("@type") {
                    if def.type_mapping.as_deref() == Some(vtype.as_str()) {
                        score += 3;
                    }
                }
                // Check @language match.
                if let Some(JsonLdValue::Str(vlang)) = obj.get("@language") {
                    if def.language.as_deref() == Some(vlang.as_str()) {
                        score += 3;
                    }
                }
            }
        }

        if score > best_score
            || (score == best_score && best_term.as_deref().unwrap_or("") > term.as_str())
        {
            best_score = score;
            best_term = Some(term.clone());
        }
    }

    best_term
}

// ============================================================================
// create_compact_context — serialize context to JSON-LD @context value
// ============================================================================

/// Serialize the active context back to a JSON-LD `@context` object.
///
/// This produces the `@context` value that should be placed at the top of
/// the compacted JSON-LD document so that consumers can interpret it.
///
/// # Returns
///
/// A [`JsonLdValue::Object`] suitable for placement as `"@context"` in the
/// output document.  Returns [`JsonLdValue::Null`] if the context is empty.
pub fn create_compact_context(ctx: &JsonLdContext) -> JsonLdValue {
    let mut obj: IndexMap<String, JsonLdValue> = IndexMap::new();

    // @base
    if let Some(base) = &ctx.base {
        obj.insert("@base".to_string(), JsonLdValue::Str(base.clone()));
    }

    // @vocab
    if let Some(vocab) = &ctx.vocab {
        obj.insert("@vocab".to_string(), JsonLdValue::Str(vocab.clone()));
    }

    // @language
    if let Some(lang) = &ctx.language {
        obj.insert("@language".to_string(), JsonLdValue::Str(lang.clone()));
    }

    // Term definitions (simple prefix mappings first, then complex ones).
    let mut simple_terms: Vec<(&String, &str)> = Vec::new();
    let mut complex_terms: Vec<(&String, &super::TermDefinition)> = Vec::new();

    for (term, def) in &ctx.terms {
        let iri = def.iri_mapping.as_deref().unwrap_or("");
        let is_simple = def.container.is_empty()
            && def.language.is_none()
            && def.direction.is_none()
            && def.type_mapping.is_none()
            && def.nest.is_none()
            && !def.reverse_property
            && !def.protected;
        if is_simple {
            simple_terms.push((term, iri));
        } else {
            complex_terms.push((term, def));
        }
    }

    // Sort for deterministic output.
    simple_terms.sort_by_key(|(t, _)| t.as_str());
    complex_terms.sort_by_key(|(t, _)| t.as_str());

    for (term, iri) in simple_terms {
        obj.insert(term.clone(), JsonLdValue::Str(iri.to_string()));
    }

    for (term, def) in complex_terms {
        let mut def_obj: IndexMap<String, JsonLdValue> = IndexMap::new();

        if let Some(iri) = &def.iri_mapping {
            def_obj.insert("@id".to_string(), JsonLdValue::Str(iri.clone()));
        }

        if def.reverse_property {
            def_obj.insert("@reverse".to_string(), JsonLdValue::Bool(true));
        }

        if let Some(tm) = &def.type_mapping {
            def_obj.insert("@type".to_string(), JsonLdValue::Str(tm.clone()));
        }

        if let Some(lang) = &def.language {
            def_obj.insert("@language".to_string(), JsonLdValue::Str(lang.clone()));
        }

        if !def.container.is_empty() {
            let containers: Vec<JsonLdValue> = def
                .container
                .iter()
                .map(|c| JsonLdValue::Str(container_type_keyword(c).to_string()))
                .collect();
            let mut container_iter = containers.into_iter();
            match container_iter.next() {
                Some(single) if container_iter.len() == 0 => {
                    def_obj.insert("@container".to_string(), single);
                }
                first => {
                    let values: Vec<JsonLdValue> =
                        first.into_iter().chain(container_iter).collect();
                    def_obj.insert("@container".to_string(), JsonLdValue::Array(values));
                }
            }
        }

        if def.protected {
            def_obj.insert("@protected".to_string(), JsonLdValue::Bool(true));
        }

        obj.insert(term.clone(), JsonLdValue::Object(def_obj));
    }

    if obj.is_empty() {
        JsonLdValue::Null
    } else {
        JsonLdValue::Object(obj)
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Returns the JSON-LD keyword string for a [`ContainerType`].
fn container_type_keyword(ct: &ContainerType) -> &'static str {
    match ct {
        ContainerType::List => "@list",
        ContainerType::Set => "@set",
        ContainerType::Language => "@language",
        ContainerType::Index => "@index",
        ContainerType::Id => "@id",
        ContainerType::Type => "@type",
        ContainerType::Graph => "@graph",
    }
}

/// Returns `true` if the string is a JSON-LD keyword (`@`-prefixed).
pub(crate) fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "@base"
            | "@container"
            | "@context"
            | "@direction"
            | "@graph"
            | "@id"
            | "@import"
            | "@included"
            | "@index"
            | "@json"
            | "@language"
            | "@list"
            | "@nest"
            | "@none"
            | "@prefix"
            | "@propagate"
            | "@protected"
            | "@reverse"
            | "@set"
            | "@type"
            | "@value"
            | "@version"
            | "@vocab"
    )
}

/// Compact an absolute IRI against a base IRI to produce a relative reference.
pub fn relativize_iri(iri: &str, base: &str) -> Option<String> {
    if let Some(relative) = iri.strip_prefix(base) {
        if !relative.is_empty() {
            return Some(relative.to_string());
        }
    }
    None
}
