//! JSON-LD 1.1 Compaction Algorithm
//!
//! Implements the compaction step of the JSON-LD processing pipeline
//! described in the [W3C JSON-LD 1.1 specification][spec] §4.2 and the
//! [SAMM specification §mapping-to-json-ld][samm-spec].
//!
//! The compactor takes an *expanded* or semi-expanded JSON-LD document
//! (where terms are spelled as full IRIs) and replaces them with the
//! compact representations defined in a `@context` object.
//!
//! ## Compact-IRI resolution
//!
//! Given a context entry `"ex": "http://example.org/"` the IRI
//! `"http://example.org/foo"` is compacted to `"ex:foo"`.
//!
//! ## Special keyword handling
//!
//! - `@type` arrays with a single element are flattened to a bare string.
//! - `@value` objects with only a `@value` key are unwrapped to the bare
//!   primitive value.
//! - `@id` values are subject to IRI compaction.
//!
//! [spec]: https://www.w3.org/TR/json-ld11/
//! [samm-spec]: https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#mapping-to-json-ld

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::error::SammError;

// ------------------------------------------------------------------ //
//  Error type                                                         //
// ------------------------------------------------------------------ //

/// Error produced by JSON-LD compaction or framing operations.
#[derive(Debug, thiserror::Error)]
pub enum JsonLdError {
    /// The supplied `@context` has an invalid shape.
    #[error("Invalid @context: {0}")]
    InvalidContext(String),

    /// A value in the document could not be processed.
    #[error("Processing error: {0}")]
    Processing(String),
}

impl From<JsonLdError> for SammError {
    fn from(e: JsonLdError) -> Self {
        SammError::Generation(e.to_string())
    }
}

// ------------------------------------------------------------------ //
//  Term map                                                           //
// ------------------------------------------------------------------ //

/// A resolved prefix → base-IRI map derived from a JSON-LD `@context`.
///
/// The map also handles *term* mappings: entries whose value is a full IRI
/// (not ending with `/` or `#`) define an exact term mapping, e.g.
/// `"name": "http://schema.org/name"`.
#[derive(Debug, Clone)]
struct TermMap {
    /// Prefix mappings: short prefix (e.g. `"ex"`) → base IRI (e.g. `"http://example.org/"`).
    prefixes: HashMap<String, String>,
    /// Exact term mappings: compact term (e.g. `"name"`) → full IRI.
    terms: HashMap<String, String>,
    /// Reverse map for compaction: full IRI → compact term.
    iri_to_term: HashMap<String, String>,
    /// Reverse map for compaction: base IRI → prefix label.
    iri_to_prefix: HashMap<String, String>,
}

impl TermMap {
    /// Parse a JSON-LD `@context` value into a [`TermMap`].
    ///
    /// The context must be a JSON object (`Value::Object`).  Nested or
    /// array contexts are not supported (returns `Err`).
    fn from_context(ctx: &Value) -> Result<Self, JsonLdError> {
        let obj = ctx.as_object().ok_or_else(|| {
            JsonLdError::InvalidContext(
                "@context must be a JSON object (array / string contexts not yet supported)"
                    .to_string(),
            )
        })?;

        let mut prefixes: HashMap<String, String> = HashMap::new();
        let mut terms: HashMap<String, String> = HashMap::new();
        let mut iri_to_prefix: HashMap<String, String> = HashMap::new();
        let mut iri_to_term: HashMap<String, String> = HashMap::new();

        for (key, val) in obj {
            // Skip JSON-LD keyword entries (e.g. "@vocab", "@base", "@language")
            if key.starts_with('@') {
                continue;
            }

            match val {
                Value::String(iri) => {
                    if iri.ends_with('/') || iri.ends_with('#') {
                        // Prefix mapping
                        iri_to_prefix.insert(iri.clone(), key.clone());
                        prefixes.insert(key.clone(), iri.clone());
                    } else {
                        // Exact term mapping
                        iri_to_term.insert(iri.clone(), key.clone());
                        terms.insert(key.clone(), iri.clone());
                    }
                }
                Value::Object(expanded_def) => {
                    // Expanded term definition: {"@id": "...", ...}
                    if let Some(Value::String(iri)) = expanded_def.get("@id") {
                        let iri = iri.clone();
                        if iri.ends_with('/') || iri.ends_with('#') {
                            iri_to_prefix.insert(iri.clone(), key.clone());
                            prefixes.insert(key.clone(), iri);
                        } else {
                            iri_to_term.insert(iri.clone(), key.clone());
                            terms.insert(key.clone(), iri);
                        }
                    }
                }
                _ => {
                    return Err(JsonLdError::InvalidContext(format!(
                        "Unexpected value type for context key '{}': expected string or object",
                        key
                    )));
                }
            }
        }

        Ok(Self {
            prefixes,
            terms,
            iri_to_term,
            iri_to_prefix,
        })
    }

    /// Compact an IRI string using the term map.
    ///
    /// Resolution order:
    /// 1. Exact term match → compact term (e.g. `"name"`).
    /// 2. Longest-prefix match → `prefix:localname`.
    /// 3. No match → original IRI unchanged.
    fn compact_iri(&self, iri: &str) -> String {
        // 1. Exact term match
        if let Some(term) = self.iri_to_term.get(iri) {
            return term.clone();
        }

        // 2. Longest-prefix match
        let mut best_len = 0usize;
        let mut best_compact: Option<String> = None;

        for (base, prefix) in &self.iri_to_prefix {
            if iri.starts_with(base.as_str()) && base.len() > best_len {
                let local = &iri[base.len()..];
                best_len = base.len();
                best_compact = Some(format!("{}:{}", prefix, local));
            }
        }

        best_compact.unwrap_or_else(|| iri.to_string())
    }
}

// ------------------------------------------------------------------ //
//  Public API                                                         //
// ------------------------------------------------------------------ //

/// JSON-LD 1.1 document compactor.
///
/// Replaces expanded IRI strings in a JSON-LD document with compact term
/// representations from the provided `@context`.
///
/// # Example
///
/// ```rust
/// use oxirs_samm::jsonld::compaction::JsonLdCompactor;
/// use serde_json::json;
///
/// let ctx = json!({ "ex": "http://example.org/" });
/// let doc = json!({
///     "@id": "http://example.org/resource",
///     "@type": ["http://example.org/Thing"]
/// });
///
/// let compactor = JsonLdCompactor::new(ctx.clone());
/// let compacted = compactor.compact(&doc, &ctx).expect("compaction should succeed");
/// assert_eq!(compacted["@id"], "ex:resource");
/// assert_eq!(compacted["@type"], "ex:Thing"); // single-element array flattened
/// ```
#[derive(Debug, Clone)]
pub struct JsonLdCompactor {
    context: Value,
}

impl JsonLdCompactor {
    /// Create a new compactor backed by the given `@context` value.
    pub fn new(context: Value) -> Self {
        Self { context }
    }

    /// Compact `document` using `context`.
    ///
    /// The returned document contains:
    /// - An `@context` key with the supplied context.
    /// - All IRI strings in key positions (like `@id`, `@type`) compacted
    ///   according to the term map.
    /// - `@value` objects with no language/type annotation unwrapped to bare
    ///   primitive scalars.
    /// - Single-element `@type` arrays flattened to a bare string.
    ///
    /// The function recurses into nested objects and arrays.
    pub fn compact(&self, document: &Value, context: &Value) -> Result<Value, JsonLdError> {
        let term_map = TermMap::from_context(context)?;
        let mut compacted = compact_value(document, &term_map)?;

        // Inject @context at the top level if the compacted root is an object
        if let Some(obj) = compacted.as_object_mut() {
            // Only add context if it isn't already there
            if !obj.contains_key("@context") {
                obj.insert("@context".to_string(), context.clone());
            }
        }

        Ok(compacted)
    }
}

// ------------------------------------------------------------------ //
//  Core compaction algorithm                                          //
// ------------------------------------------------------------------ //

/// Recursively compact a JSON-LD value using the provided term map.
fn compact_value(value: &Value, term_map: &TermMap) -> Result<Value, JsonLdError> {
    match value {
        Value::Object(obj) => compact_object(obj, term_map),
        Value::Array(arr) => {
            let compacted: Result<Vec<Value>, JsonLdError> =
                arr.iter().map(|v| compact_value(v, term_map)).collect();
            Ok(Value::Array(compacted?))
        }
        // Primitives pass through unchanged
        other => Ok(other.clone()),
    }
}

/// Compact a JSON object, handling JSON-LD keywords specially.
fn compact_object(obj: &Map<String, Value>, term_map: &TermMap) -> Result<Value, JsonLdError> {
    // --- @value unwrapping ----------------------------------------------------
    // A `{"@value": <scalar>}` node with no other qualifying keys (no @type,
    // no @language) is unwrapped to the bare scalar.
    if let Some(raw_val) = obj.get("@value") {
        let has_type = obj.contains_key("@type");
        let has_lang = obj.contains_key("@language");
        if !has_type && !has_lang && obj.len() == 1 {
            return Ok(raw_val.clone());
        }
    }

    let mut out = Map::with_capacity(obj.len());

    for (key, val) in obj {
        match key.as_str() {
            // ---- @id: compact the IRI ----------------------------------------
            "@id" => {
                let compacted_iri = if let Some(iri) = val.as_str() {
                    Value::String(term_map.compact_iri(iri))
                } else {
                    val.clone()
                };
                out.insert("@id".to_string(), compacted_iri);
            }

            // ---- @type: compact elements; flatten single-element array -------
            "@type" => {
                let compacted_type = compact_type_value(val, term_map);
                out.insert("@type".to_string(), compacted_type);
            }

            // ---- @context: pass through unchanged ----------------------------
            "@context" => {
                out.insert(key.clone(), val.clone());
            }

            // ---- @graph: recurse into its array ------------------------------
            "@graph" => {
                let inner = compact_value(val, term_map)?;
                out.insert(key.clone(), inner);
            }

            // ---- Regular keys: try to compact the key as an IRI, then recurse
            _ => {
                let compact_key = term_map.compact_iri(key);
                let compact_val = compact_value(val, term_map)?;
                out.insert(compact_key, compact_val);
            }
        }
    }

    Ok(Value::Object(out))
}

/// Compact the value of a `@type` entry.
///
/// - Array with a single element → bare string (flattened).
/// - Array with multiple elements → array of compacted strings.
/// - Bare string → compacted string.
fn compact_type_value(val: &Value, term_map: &TermMap) -> Value {
    match val {
        Value::Array(arr) if arr.len() == 1 => {
            let compacted = arr[0]
                .as_str()
                .map(|s| term_map.compact_iri(s))
                .unwrap_or_else(|| arr[0].to_string());
            Value::String(compacted)
        }
        Value::Array(arr) => {
            let compacted: Vec<Value> = arr
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| Value::String(term_map.compact_iri(s)))
                        .unwrap_or_else(|| v.clone())
                })
                .collect();
            Value::Array(compacted)
        }
        Value::String(s) => Value::String(term_map.compact_iri(s)),
        other => other.clone(),
    }
}

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn simple_context() -> Value {
        json!({
            "ex":   "http://example.org/",
            "samm": "urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#",
            "xsd":  "http://www.w3.org/2001/XMLSchema#",
            "name": "http://schema.org/name"
        })
    }

    // ---------------------------------------------------------------- //
    // Test 1: Compact IRI with prefix                                   //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_compact_iri_with_prefix() {
        let ctx = simple_context();
        let term_map = TermMap::from_context(&ctx).expect("context should parse");
        let result = term_map.compact_iri("http://example.org/foo");
        assert_eq!(
            result, "ex:foo",
            "IRI with matching prefix must be compacted"
        );
    }

    // ---------------------------------------------------------------- //
    // Test 2: Compact @type array with single item → bare string        //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_compact_type_single_element_flattened() {
        let ctx = simple_context();
        let term_map = TermMap::from_context(&ctx).expect("context should parse");

        let type_val = json!(["http://example.org/Thing"]);
        let result = compact_type_value(&type_val, &term_map);
        assert_eq!(
            result,
            Value::String("ex:Thing".to_string()),
            "@type array with one element must be flattened to a bare string"
        );
    }

    // ---------------------------------------------------------------- //
    // Test 3: Compact @value scalar → bare primitive                    //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_compact_value_scalar_unwrapped() {
        let ctx = simple_context();
        let term_map = TermMap::from_context(&ctx).expect("context should parse");

        let val_node = json!({ "@value": 42 });
        let result = compact_value(&val_node, &term_map).expect("compact should succeed");
        assert_eq!(
            result,
            Value::Number(42.into()),
            "@value node with bare integer must unwrap to scalar"
        );
    }

    // ---------------------------------------------------------------- //
    // Test 4: Round-trip — expand IRIs then compact preserves semantics //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_roundtrip_preserves_semantics() {
        let ctx = simple_context();
        let compactor = JsonLdCompactor::new(ctx.clone());

        // "Expanded" doc (full IRIs)
        let expanded = json!({
            "@id":   "http://example.org/sensor1",
            "@type": ["http://example.org/Sensor"],
            "http://example.org/reading": { "@value": 99.5 }
        });

        let compacted = compactor
            .compact(&expanded, &ctx)
            .expect("compaction should succeed");

        // @id and @type must be compacted
        assert_eq!(compacted["@id"], "ex:sensor1");
        assert_eq!(compacted["@type"], "ex:Sensor"); // flattened

        // Nested IRI key must be compacted
        let reading_key = "ex:reading";
        assert!(
            compacted.get(reading_key).is_some(),
            "key '{}' should be present in compacted doc; got: {}",
            reading_key,
            compacted
        );

        // @value scalar must be unwrapped
        assert_eq!(
            compacted[reading_key],
            json!(99.5),
            "@value scalar must be unwrapped after compaction"
        );
    }

    // ---------------------------------------------------------------- //
    // Test 5: Exact term mapping                                        //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_exact_term_mapping() {
        let ctx = simple_context();
        let term_map = TermMap::from_context(&ctx).expect("context should parse");
        // "name" is an exact term for "http://schema.org/name"
        let result = term_map.compact_iri("http://schema.org/name");
        assert_eq!(result, "name", "exact term mapping must be used");
    }

    // ---------------------------------------------------------------- //
    // Test 6: Unknown IRI passes through unchanged                      //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_unknown_iri_unchanged() {
        let ctx = simple_context();
        let term_map = TermMap::from_context(&ctx).expect("context should parse");
        let unknown = "http://totally-unknown.example/foo";
        assert_eq!(
            term_map.compact_iri(unknown),
            unknown,
            "unresolvable IRI must remain unchanged"
        );
    }

    // ---------------------------------------------------------------- //
    // Test 7: Multi-element @type stays as array                        //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_compact_type_multi_element_stays_array() {
        let ctx = simple_context();
        let term_map = TermMap::from_context(&ctx).expect("context should parse");

        let type_val = json!(["http://example.org/Thing", "http://example.org/Device"]);
        let result = compact_type_value(&type_val, &term_map);
        assert!(
            result.is_array(),
            "multi-element @type must remain an array"
        );
        let arr = result.as_array().expect("array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], "ex:Thing");
        assert_eq!(arr[1], "ex:Device");
    }

    // ---------------------------------------------------------------- //
    // Test 8: @context injected at top level                            //
    // ---------------------------------------------------------------- //
    #[test]
    fn test_context_injected_at_top_level() {
        let ctx = simple_context();
        let compactor = JsonLdCompactor::new(ctx.clone());
        let doc = json!({
            "@id": "http://example.org/x"
        });
        let compacted = compactor
            .compact(&doc, &ctx)
            .expect("compact should succeed");
        assert!(
            compacted.get("@context").is_some(),
            "compacted document must contain @context"
        );
    }
}
