//! WASM/JS interop bridge type conversions.
//!
//! Provides pure-Rust representations of JavaScript values and conversion
//! utilities for transferring SPARQL results, RDF terms, and query parameters
//! across the WASM boundary without requiring `wasm_bindgen` at the type level.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// JsValue
// ─────────────────────────────────────────────────────────────────────────────

/// A Rust-side representation of a JavaScript value.
///
/// Covers the six primitive JS types plus composite Object and Array.
#[derive(Debug, Clone, PartialEq)]
pub enum JsValue {
    /// JavaScript `undefined`.
    Undefined,
    /// JavaScript `null`.
    Null,
    /// JavaScript boolean.
    Boolean(bool),
    /// JavaScript number (always f64 in JS).
    Number(f64),
    /// JavaScript string.
    String(String),
    /// JavaScript array.
    Array(Vec<JsValue>),
    /// JavaScript plain object (string-keyed).
    Object(HashMap<String, JsValue>),
}

impl JsValue {
    /// Return the inner `bool`, or `None` if this is not a `Boolean`.
    pub fn as_bool(&self) -> Option<bool> {
        if let JsValue::Boolean(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    /// Return the inner `f64`, or `None` if this is not a `Number`.
    pub fn as_f64(&self) -> Option<f64> {
        if let JsValue::Number(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    /// Return a reference to the inner string slice, or `None` if not a `String`.
    pub fn as_str(&self) -> Option<&str> {
        if let JsValue::String(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Return a slice of the inner array, or `None` if not an `Array`.
    pub fn as_array(&self) -> Option<&[JsValue]> {
        if let JsValue::Array(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Get a value from an `Object` variant by key, or `None` otherwise.
    pub fn get(&self, key: &str) -> Option<&JsValue> {
        if let JsValue::Object(map) = self {
            map.get(key)
        } else {
            None
        }
    }

    /// Return `true` if this value is `Null` or `Undefined`.
    pub fn is_null_or_undefined(&self) -> bool {
        matches!(self, JsValue::Null | JsValue::Undefined)
    }

    /// Return the JS type name string (mirrors `typeof` except for null).
    pub fn type_name(&self) -> &str {
        match self {
            JsValue::Undefined => "undefined",
            JsValue::Null => "null",
            JsValue::Boolean(_) => "boolean",
            JsValue::Number(_) => "number",
            JsValue::String(_) => "string",
            JsValue::Array(_) => "array",
            JsValue::Object(_) => "object",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BridgeError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the WASM bridge converters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// The JS value had an unexpected type.
    TypeMismatch(String),
    /// A required key was absent from an Object value.
    KeyNotFound(String),
    /// An array index was out of bounds.
    IndexOutOfBounds(usize),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::TypeMismatch(msg) => write!(f, "Type mismatch: {msg}"),
            BridgeError::KeyNotFound(key) => write!(f, "Key not found: {key}"),
            BridgeError::IndexOutOfBounds(idx) => write!(f, "Index out of bounds: {idx}"),
        }
    }
}

impl std::error::Error for BridgeError {}

// ─────────────────────────────────────────────────────────────────────────────
// WasmBridge
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless bridge helper for WASM↔JS type conversions.
pub struct WasmBridge;

impl WasmBridge {
    /// Convert SPARQL SELECT results to a JS-friendly `JsValue`.
    ///
    /// Produces:
    /// ```json
    /// {
    ///   "head": { "vars": ["x", "y"] },
    ///   "results": {
    ///     "bindings": [
    ///       { "x": { "type": "literal", "value": "hello" }, ... },
    ///       ...
    ///     ]
    ///   }
    /// }
    /// ```
    pub fn sparql_results_to_js(vars: &[String], bindings: &[HashMap<String, String>]) -> JsValue {
        // head.vars
        let vars_js = JsValue::Array(vars.iter().map(|v| JsValue::String(v.clone())).collect());
        let mut head = HashMap::new();
        head.insert("vars".to_string(), vars_js);

        // results.bindings
        let binding_rows: Vec<JsValue> = bindings
            .iter()
            .map(|row| {
                let mut obj = HashMap::new();
                for var in vars {
                    if let Some(val) = row.get(var) {
                        let term = Self::rdf_term_to_js(val);
                        obj.insert(var.clone(), term);
                    }
                }
                JsValue::Object(obj)
            })
            .collect();

        let mut results = HashMap::new();
        results.insert("bindings".to_string(), JsValue::Array(binding_rows));

        let mut root = HashMap::new();
        root.insert("head".to_string(), JsValue::Object(head));
        root.insert("results".to_string(), JsValue::Object(results));

        JsValue::Object(root)
    }

    /// Extract a SPARQL query string from a JS value.
    ///
    /// Accepts either a `String` value directly or an `Object` with a `"query"` key.
    ///
    /// # Errors
    /// - `TypeMismatch` if the value type is unsupported.
    /// - `KeyNotFound` if an Object is provided but has no `"query"` key.
    pub fn js_to_sparql_query(val: &JsValue) -> Result<String, BridgeError> {
        match val {
            JsValue::String(s) => Ok(s.clone()),
            JsValue::Object(map) => map
                .get("query")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| BridgeError::KeyNotFound("query".to_string())),
            other => Err(BridgeError::TypeMismatch(format!(
                "expected string or object, got {}",
                other.type_name()
            ))),
        }
    }

    /// Convert an RDF term string to a JS object with `type` and `value`.
    ///
    /// Detection rules:
    /// - starts with `<` and ends with `>` → IRI
    /// - starts with `_:` → blank node
    /// - otherwise → literal
    pub fn rdf_term_to_js(term: &str) -> JsValue {
        let (term_type, value) = if term.starts_with('<') && term.ends_with('>') {
            ("uri", &term[1..term.len() - 1])
        } else if let Some(stripped) = term.strip_prefix("_:") {
            ("bnode", stripped)
        } else {
            ("literal", term)
        };

        let mut map = HashMap::new();
        map.insert("type".to_string(), JsValue::String(term_type.to_string()));
        map.insert("value".to_string(), JsValue::String(value.to_string()));
        JsValue::Object(map)
    }

    /// Convert a JS array of string values to `Vec<String>`.
    ///
    /// # Errors
    /// - `TypeMismatch` if the value is not an Array.
    /// - `TypeMismatch` if any element is not a String.
    pub fn js_array_to_strings(val: &JsValue) -> Result<Vec<String>, BridgeError> {
        let items = val.as_array().ok_or_else(|| {
            BridgeError::TypeMismatch(format!("expected array, got {}", val.type_name()))
        })?;

        items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                item.as_str().map(str::to_string).ok_or_else(|| {
                    BridgeError::TypeMismatch(format!(
                        "element [{}] expected string, got {}",
                        i,
                        item.type_name()
                    ))
                })
            })
            .collect()
    }

    /// Convert a JS object with string values to `HashMap<String, String>`.
    ///
    /// # Errors
    /// - `TypeMismatch` if the value is not an Object.
    /// - `TypeMismatch` if any value is not a String.
    pub fn js_object_to_map(val: &JsValue) -> Result<HashMap<String, String>, BridgeError> {
        let obj = match val {
            JsValue::Object(m) => m,
            other => {
                return Err(BridgeError::TypeMismatch(format!(
                    "expected object, got {}",
                    other.type_name()
                )))
            }
        };

        let mut result = HashMap::new();
        for (k, v) in obj {
            let s = v.as_str().ok_or_else(|| {
                BridgeError::TypeMismatch(format!(
                    "value for key '{}' expected string, got {}",
                    k,
                    v.type_name()
                ))
            })?;
            result.insert(k.clone(), s.to_string());
        }
        Ok(result)
    }

    /// Serialise a `JsValue` to a JSON-like string.
    ///
    /// Produces valid JSON for all variants.
    pub fn serialize_js_value(val: &JsValue) -> String {
        match val {
            JsValue::Undefined => "undefined".to_string(),
            JsValue::Null => "null".to_string(),
            JsValue::Boolean(b) => b.to_string(),
            JsValue::Number(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            JsValue::String(s) => {
                // Simple JSON-string escaping
                let escaped = s
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                format!("\"{escaped}\"")
            }
            JsValue::Array(items) => {
                let inner: Vec<String> = items.iter().map(Self::serialize_js_value).collect();
                format!("[{}]", inner.join(","))
            }
            JsValue::Object(map) => {
                let mut pairs: Vec<String> = map
                    .iter()
                    .map(|(k, v)| {
                        let ek = k.replace('\\', "\\\\").replace('"', "\\\"");
                        format!("\"{}\":{}", ek, Self::serialize_js_value(v))
                    })
                    .collect();
                pairs.sort(); // deterministic output for tests
                format!("{{{}}}", pairs.join(","))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── JsValue::as_bool ─────────────────────────────────────────────────────

    #[test]
    fn test_as_bool_true() {
        assert_eq!(JsValue::Boolean(true).as_bool(), Some(true));
    }

    #[test]
    fn test_as_bool_false() {
        assert_eq!(JsValue::Boolean(false).as_bool(), Some(false));
    }

    #[test]
    fn test_as_bool_non_bool() {
        assert_eq!(JsValue::Number(1.0).as_bool(), None);
    }

    // ── JsValue::as_f64 ──────────────────────────────────────────────────────

    #[test]
    fn test_as_f64_present() {
        assert_eq!(JsValue::Number(2.71).as_f64(), Some(2.71));
    }

    #[test]
    fn test_as_f64_absent() {
        assert_eq!(JsValue::String("x".to_string()).as_f64(), None);
    }

    // ── JsValue::as_str ──────────────────────────────────────────────────────

    #[test]
    fn test_as_str_present() {
        assert_eq!(JsValue::String("hello".to_string()).as_str(), Some("hello"));
    }

    #[test]
    fn test_as_str_absent() {
        assert_eq!(JsValue::Null.as_str(), None);
    }

    // ── JsValue::as_array ────────────────────────────────────────────────────

    #[test]
    fn test_as_array_present() {
        let arr = JsValue::Array(vec![JsValue::Number(1.0)]);
        assert_eq!(arr.as_array().map(|a| a.len()), Some(1));
    }

    #[test]
    fn test_as_array_absent() {
        assert!(JsValue::Null.as_array().is_none());
    }

    // ── JsValue::get ─────────────────────────────────────────────────────────

    #[test]
    fn test_get_existing_key() {
        let mut map = HashMap::new();
        map.insert("foo".to_string(), JsValue::Boolean(true));
        let obj = JsValue::Object(map);
        assert_eq!(obj.get("foo"), Some(&JsValue::Boolean(true)));
    }

    #[test]
    fn test_get_missing_key() {
        let obj = JsValue::Object(HashMap::new());
        assert!(obj.get("bar").is_none());
    }

    #[test]
    fn test_get_non_object() {
        assert!(JsValue::Null.get("key").is_none());
    }

    // ── JsValue::is_null_or_undefined ────────────────────────────────────────

    #[test]
    fn test_null_is_null_or_undefined() {
        assert!(JsValue::Null.is_null_or_undefined());
    }

    #[test]
    fn test_undefined_is_null_or_undefined() {
        assert!(JsValue::Undefined.is_null_or_undefined());
    }

    #[test]
    fn test_number_is_not_null_or_undefined() {
        assert!(!JsValue::Number(0.0).is_null_or_undefined());
    }

    // ── JsValue::type_name ───────────────────────────────────────────────────

    #[test]
    fn test_type_name_undefined() {
        assert_eq!(JsValue::Undefined.type_name(), "undefined");
    }

    #[test]
    fn test_type_name_null() {
        assert_eq!(JsValue::Null.type_name(), "null");
    }

    #[test]
    fn test_type_name_boolean() {
        assert_eq!(JsValue::Boolean(false).type_name(), "boolean");
    }

    #[test]
    fn test_type_name_number() {
        assert_eq!(JsValue::Number(0.0).type_name(), "number");
    }

    #[test]
    fn test_type_name_string() {
        assert_eq!(JsValue::String("x".to_string()).type_name(), "string");
    }

    #[test]
    fn test_type_name_array() {
        assert_eq!(JsValue::Array(vec![]).type_name(), "array");
    }

    #[test]
    fn test_type_name_object() {
        assert_eq!(JsValue::Object(HashMap::new()).type_name(), "object");
    }

    // ── sparql_results_to_js ─────────────────────────────────────────────────

    #[test]
    fn test_sparql_results_has_head() {
        let vars = vec!["x".to_string()];
        let bindings = vec![];
        let result = WasmBridge::sparql_results_to_js(&vars, &bindings);
        assert!(result.get("head").is_some());
    }

    #[test]
    fn test_sparql_results_has_results() {
        let vars = vec!["x".to_string()];
        let result = WasmBridge::sparql_results_to_js(&vars, &[]);
        assert!(result.get("results").is_some());
    }

    #[test]
    fn test_sparql_results_vars_count() {
        let vars = vec!["s".to_string(), "p".to_string(), "o".to_string()];
        let result = WasmBridge::sparql_results_to_js(&vars, &[]);
        let head = result.get("head").unwrap();
        let head_vars = head.get("vars").unwrap().as_array().unwrap();
        assert_eq!(head_vars.len(), 3);
    }

    #[test]
    fn test_sparql_results_binding_row() {
        let vars = vec!["x".to_string()];
        let mut row = HashMap::new();
        row.insert("x".to_string(), "<http://example.org/a>".to_string());
        let result = WasmBridge::sparql_results_to_js(&vars, &[row]);
        let results = result.get("results").unwrap();
        let rows = results.get("bindings").unwrap().as_array().unwrap();
        assert_eq!(rows.len(), 1);
    }

    // ── js_to_sparql_query ───────────────────────────────────────────────────

    #[test]
    fn test_js_to_sparql_query_string() {
        let val = JsValue::String("SELECT * WHERE { ?s ?p ?o }".to_string());
        let q = WasmBridge::js_to_sparql_query(&val).unwrap();
        assert!(q.contains("SELECT"));
    }

    #[test]
    fn test_js_to_sparql_query_object() {
        let mut map = HashMap::new();
        map.insert("query".to_string(), JsValue::String("ASK { }".to_string()));
        let val = JsValue::Object(map);
        let q = WasmBridge::js_to_sparql_query(&val).unwrap();
        assert_eq!(q, "ASK { }");
    }

    #[test]
    fn test_js_to_sparql_query_object_missing_key() {
        let val = JsValue::Object(HashMap::new());
        assert!(matches!(
            WasmBridge::js_to_sparql_query(&val),
            Err(BridgeError::KeyNotFound(_))
        ));
    }

    #[test]
    fn test_js_to_sparql_query_type_mismatch() {
        let val = JsValue::Number(42.0);
        assert!(matches!(
            WasmBridge::js_to_sparql_query(&val),
            Err(BridgeError::TypeMismatch(_))
        ));
    }

    // ── rdf_term_to_js ───────────────────────────────────────────────────────

    #[test]
    fn test_rdf_term_iri() {
        let js = WasmBridge::rdf_term_to_js("<http://example.org/a>");
        let ty = js.get("type").unwrap().as_str().unwrap();
        assert_eq!(ty, "uri");
    }

    #[test]
    fn test_rdf_term_blank() {
        let js = WasmBridge::rdf_term_to_js("_:b0");
        let ty = js.get("type").unwrap().as_str().unwrap();
        assert_eq!(ty, "bnode");
        let val = js.get("value").unwrap().as_str().unwrap();
        assert_eq!(val, "b0");
    }

    #[test]
    fn test_rdf_term_literal() {
        let js = WasmBridge::rdf_term_to_js("hello world");
        let ty = js.get("type").unwrap().as_str().unwrap();
        assert_eq!(ty, "literal");
    }

    #[test]
    fn test_rdf_term_iri_strips_brackets() {
        let js = WasmBridge::rdf_term_to_js("<http://x.org/>");
        let val = js.get("value").unwrap().as_str().unwrap();
        assert_eq!(val, "http://x.org/");
    }

    // ── js_array_to_strings ──────────────────────────────────────────────────

    #[test]
    fn test_js_array_to_strings_ok() {
        let val = JsValue::Array(vec![
            JsValue::String("a".to_string()),
            JsValue::String("b".to_string()),
        ]);
        let result = WasmBridge::js_array_to_strings(&val).unwrap();
        assert_eq!(result, vec!["a", "b"]);
    }

    #[test]
    fn test_js_array_to_strings_not_array() {
        let val = JsValue::Null;
        assert!(matches!(
            WasmBridge::js_array_to_strings(&val),
            Err(BridgeError::TypeMismatch(_))
        ));
    }

    #[test]
    fn test_js_array_to_strings_non_string_element() {
        let val = JsValue::Array(vec![JsValue::Number(1.0)]);
        assert!(matches!(
            WasmBridge::js_array_to_strings(&val),
            Err(BridgeError::TypeMismatch(_))
        ));
    }

    #[test]
    fn test_js_array_to_strings_empty() {
        let val = JsValue::Array(vec![]);
        assert_eq!(
            WasmBridge::js_array_to_strings(&val).unwrap(),
            Vec::<String>::new()
        );
    }

    // ── js_object_to_map ─────────────────────────────────────────────────────

    #[test]
    fn test_js_object_to_map_ok() {
        let mut map = HashMap::new();
        map.insert("k".to_string(), JsValue::String("v".to_string()));
        let val = JsValue::Object(map);
        let result = WasmBridge::js_object_to_map(&val).unwrap();
        assert_eq!(result.get("k").map(String::as_str), Some("v"));
    }

    #[test]
    fn test_js_object_to_map_not_object() {
        let val = JsValue::Array(vec![]);
        assert!(matches!(
            WasmBridge::js_object_to_map(&val),
            Err(BridgeError::TypeMismatch(_))
        ));
    }

    #[test]
    fn test_js_object_to_map_non_string_value() {
        let mut map = HashMap::new();
        map.insert("k".to_string(), JsValue::Number(1.0));
        let val = JsValue::Object(map);
        assert!(matches!(
            WasmBridge::js_object_to_map(&val),
            Err(BridgeError::TypeMismatch(_))
        ));
    }

    // ── serialize_js_value ───────────────────────────────────────────────────

    #[test]
    fn test_serialize_undefined() {
        assert_eq!(
            WasmBridge::serialize_js_value(&JsValue::Undefined),
            "undefined"
        );
    }

    #[test]
    fn test_serialize_null() {
        assert_eq!(WasmBridge::serialize_js_value(&JsValue::Null), "null");
    }

    #[test]
    fn test_serialize_boolean_true() {
        assert_eq!(
            WasmBridge::serialize_js_value(&JsValue::Boolean(true)),
            "true"
        );
    }

    #[test]
    fn test_serialize_boolean_false() {
        assert_eq!(
            WasmBridge::serialize_js_value(&JsValue::Boolean(false)),
            "false"
        );
    }

    #[test]
    fn test_serialize_integer_number() {
        assert_eq!(WasmBridge::serialize_js_value(&JsValue::Number(42.0)), "42");
    }

    #[test]
    fn test_serialize_float_number() {
        let s = WasmBridge::serialize_js_value(&JsValue::Number(2.71));
        assert!(s.contains("2.71"));
    }

    #[test]
    fn test_serialize_string() {
        assert_eq!(
            WasmBridge::serialize_js_value(&JsValue::String("hello".to_string())),
            "\"hello\""
        );
    }

    #[test]
    fn test_serialize_string_with_quotes() {
        let s = WasmBridge::serialize_js_value(&JsValue::String("say \"hi\"".to_string()));
        assert!(s.contains("\\\""));
    }

    #[test]
    fn test_serialize_empty_array() {
        assert_eq!(
            WasmBridge::serialize_js_value(&JsValue::Array(vec![])),
            "[]"
        );
    }

    #[test]
    fn test_serialize_array() {
        let val = JsValue::Array(vec![JsValue::Number(1.0), JsValue::Number(2.0)]);
        let s = WasmBridge::serialize_js_value(&val);
        assert!(s.starts_with('[') && s.ends_with(']'));
    }

    #[test]
    fn test_serialize_empty_object() {
        assert_eq!(
            WasmBridge::serialize_js_value(&JsValue::Object(HashMap::new())),
            "{}"
        );
    }

    #[test]
    fn test_serialize_object_with_field() {
        let mut map = HashMap::new();
        map.insert("x".to_string(), JsValue::Number(1.0));
        let val = JsValue::Object(map);
        let s = WasmBridge::serialize_js_value(&val);
        assert!(s.contains("\"x\""));
        assert!(s.contains('1'));
    }

    // ── BridgeError display ──────────────────────────────────────────────────

    #[test]
    fn test_bridge_error_type_mismatch_display() {
        let e = BridgeError::TypeMismatch("bad type".to_string());
        assert!(e.to_string().contains("Type mismatch"));
    }

    #[test]
    fn test_bridge_error_key_not_found_display() {
        let e = BridgeError::KeyNotFound("query".to_string());
        assert!(e.to_string().contains("Key not found"));
    }

    #[test]
    fn test_bridge_error_index_out_of_bounds_display() {
        let e = BridgeError::IndexOutOfBounds(5);
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_bridge_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(BridgeError::TypeMismatch("x".to_string()));
        assert!(!e.to_string().is_empty());
    }
}
