//! Enhanced String Functions for SPARQL 1.1+
//!
//! This module implements extended string manipulation and literal construction
//! functions as specified in SPARQL 1.1 and RDF 1.2.
//!
//! Based on Apache Jena ARQ implementation.

use crate::extensions::{CustomFunction, ExecutionContext, Value, ValueType};
use anyhow::{bail, Result};

// ============================================================================
// STRBEFORE Function - Substring before separator
// ============================================================================

/// STRBEFORE(str, separator) - Returns the substring before the first occurrence of separator
///
/// Returns the part of the string before the first occurrence of the separator.
/// If separator is not found, returns empty string "".
/// If separator is empty string, returns empty string "".
///
/// # Examples
/// ```sparql
/// STRBEFORE("abc@example.org", "@") → "abc"
/// STRBEFORE("foobar", "bar") → "foo"
/// STRBEFORE("foobar", "xyz") → ""
/// ```
#[derive(Debug, Clone)]
pub struct StrBeforeFunction;

impl CustomFunction for StrBeforeFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/XMLSchema#strBefore"
    }

    fn arity(&self) -> Option<usize> {
        Some(2)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }

    fn return_type(&self) -> ValueType {
        ValueType::String
    }

    fn documentation(&self) -> &str {
        "Returns the substring before the first occurrence of separator"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("STRBEFORE() requires exactly 2 arguments");
        }

        let (string, lang, _dt) = extract_string_value(&args[0])?;
        let (separator, sep_lang, _sep_dt) = extract_string_value(&args[1])?;

        // Language tags must be compatible
        if !compatible_languages(&lang, &sep_lang) {
            bail!("STRBEFORE: incompatible language tags");
        }

        if separator.is_empty() {
            return Ok(create_string_value("", lang));
        }

        let result = if let Some(pos) = string.find(&separator) {
            &string[..pos]
        } else {
            ""
        };

        Ok(create_string_value(result, lang))
    }
}

// ============================================================================
// STRAFTER Function - Substring after separator
// ============================================================================

/// STRAFTER(str, separator) - Returns the substring after the first occurrence of separator
///
/// Returns the part of the string after the first occurrence of the separator.
/// If separator is not found, returns empty string "".
/// If separator is empty string, returns the original string.
///
/// # Examples
/// ```sparql
/// STRAFTER("abc@example.org", "@") → "example.org"
/// STRAFTER("foobar", "foo") → "bar"
/// STRAFTER("foobar", "xyz") → ""
/// ```
#[derive(Debug, Clone)]
pub struct StrAfterFunction;

impl CustomFunction for StrAfterFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/XMLSchema#strAfter"
    }

    fn arity(&self) -> Option<usize> {
        Some(2)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }

    fn return_type(&self) -> ValueType {
        ValueType::String
    }

    fn documentation(&self) -> &str {
        "Returns the substring after the first occurrence of separator"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("STRAFTER() requires exactly 2 arguments");
        }

        let (string, lang, _dt) = extract_string_value(&args[0])?;
        let (separator, sep_lang, _sep_dt) = extract_string_value(&args[1])?;

        // Language tags must be compatible
        if !compatible_languages(&lang, &sep_lang) {
            bail!("STRAFTER: incompatible language tags");
        }

        if separator.is_empty() {
            return Ok(create_string_value(&string, lang));
        }

        let result = if let Some(pos) = string.find(&separator) {
            &string[pos + separator.len()..]
        } else {
            ""
        };

        Ok(create_string_value(result, lang))
    }
}

// ============================================================================
// STRLANG Function - Create language-tagged literal
// ============================================================================

/// STRLANG(str, lang) - Creates a language-tagged literal
///
/// Creates a new literal with the specified language tag.
/// The language tag must be a valid BCP47 language tag (lowercased).
///
/// # Examples
/// ```sparql
/// STRLANG("chat", "fr") → "chat"@fr
/// STRLANG("Hello", "en") → "Hello"@en
/// ```
#[derive(Debug, Clone)]
pub struct StrLangFunction;

impl CustomFunction for StrLangFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/XMLSchema#strLang"
    }

    fn arity(&self) -> Option<usize> {
        Some(2)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String]
    }

    fn return_type(&self) -> ValueType {
        ValueType::Literal
    }

    fn documentation(&self) -> &str {
        "Creates a language-tagged literal from a string and language tag"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("STRLANG() requires exactly 2 arguments");
        }

        // First argument must be a simple literal or string
        let lexical_form = match &args[0] {
            Value::String(s) => s.clone(),
            Value::Literal {
                value,
                language: None,
                datatype: None,
            } => value.clone(),
            _ => bail!("STRLANG: first argument must be a simple literal or string"),
        };

        // Second argument must be a simple literal or string (the language tag)
        let lang_tag = match &args[1] {
            Value::String(s) => s.clone(),
            Value::Literal {
                value,
                language: None,
                datatype: None,
            } => value.clone(),
            _ => bail!("STRLANG: second argument must be a simple literal or string"),
        };

        // Validate and normalize language tag (should be lowercase)
        let lang_tag = validate_language_tag(&lang_tag)?;

        Ok(Value::Literal {
            value: lexical_form,
            language: Some(lang_tag),
            datatype: None,
        })
    }
}

// ============================================================================
// STRLANGDIR Function - Create language and direction-tagged literal (RDF 1.2)
// ============================================================================

/// STRLANGDIR(str, lang, dir) - Creates a language and direction-tagged literal
///
/// Creates a new literal with specified language tag and text direction.
/// Part of RDF 1.2 specification for bidirectional text support.
///
/// # Examples
/// ```sparql
/// STRLANGDIR("Hello", "en", "ltr") → "Hello"@en--ltr
/// STRLANGDIR("مرحبا", "ar", "rtl") → "مرحبا"@ar--rtl
/// ```
#[derive(Debug, Clone)]
pub struct StrLangDirFunction;

impl CustomFunction for StrLangDirFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/ns/rdf#langString"
    }

    fn arity(&self) -> Option<usize> {
        Some(3)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::String, ValueType::String]
    }

    fn return_type(&self) -> ValueType {
        ValueType::Literal
    }

    fn documentation(&self) -> &str {
        "Creates a language and direction-tagged literal (RDF 1.2)"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 3 {
            bail!("STRLANGDIR() requires exactly 3 arguments");
        }

        // First argument: lexical form
        let lexical_form = match &args[0] {
            Value::String(s) => s.clone(),
            Value::Literal {
                value,
                language: None,
                datatype: None,
            } => value.clone(),
            _ => bail!("STRLANGDIR: first argument must be a simple literal or string"),
        };

        // Second argument: language tag
        let lang_tag = match &args[1] {
            Value::String(s) => s.clone(),
            Value::Literal {
                value,
                language: None,
                datatype: None,
            } => value.clone(),
            _ => bail!("STRLANGDIR: second argument must be a simple literal or string"),
        };

        // Third argument: direction (ltr or rtl)
        let direction = match &args[2] {
            Value::String(s) => s.clone(),
            Value::Literal {
                value,
                language: None,
                datatype: None,
            } => value.clone(),
            _ => bail!("STRLANGDIR: third argument must be a simple literal or string"),
        };

        // Validate direction
        if direction != "ltr" && direction != "rtl" {
            bail!("STRLANGDIR: direction must be 'ltr' or 'rtl'");
        }

        // Validate and normalize language tag
        let lang_tag = validate_language_tag(&lang_tag)?;

        // Combine language and direction using RDF 1.2 format: lang--dir
        let lang_with_dir = format!("{}--{}", lang_tag, direction);

        Ok(Value::Literal {
            value: lexical_form,
            language: Some(lang_with_dir),
            datatype: None,
        })
    }
}

// ============================================================================
// STRDT Function - Create datatyped literal
// ============================================================================

/// STRDT(str, datatype) - Creates a typed literal
///
/// Creates a new literal with the specified datatype IRI.
///
/// # Examples
/// ```sparql
/// STRDT("123", xsd:integer) → "123"^^xsd:integer
/// STRDT("true", xsd:boolean) → "true"^^xsd:boolean
/// ```
#[derive(Debug, Clone)]
pub struct StrDtFunction;

impl CustomFunction for StrDtFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/XMLSchema#strDt"
    }

    fn arity(&self) -> Option<usize> {
        Some(2)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::String, ValueType::Iri]
    }

    fn return_type(&self) -> ValueType {
        ValueType::Literal
    }

    fn documentation(&self) -> &str {
        "Creates a typed literal from a string and datatype IRI"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 2 {
            bail!("STRDT() requires exactly 2 arguments");
        }

        // First argument: lexical form (must be simple literal or string)
        let lexical_form = match &args[0] {
            Value::String(s) => s.clone(),
            Value::Literal {
                value,
                language: None,
                datatype: None,
            } => value.clone(),
            _ => bail!("STRDT: first argument must be a simple literal or string"),
        };

        // Second argument: datatype IRI
        let datatype = match &args[1] {
            Value::Iri(iri) => iri.clone(),
            _ => bail!("STRDT: second argument must be an IRI"),
        };

        Ok(Value::Literal {
            value: lexical_form,
            language: None,
            datatype: Some(datatype),
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract string value and metadata from a Value
fn extract_string_value(value: &Value) -> Result<(String, Option<String>, Option<String>)> {
    match value {
        Value::String(s) => Ok((s.clone(), None, None)),
        Value::Literal {
            value,
            language,
            datatype,
        } => Ok((value.clone(), language.clone(), datatype.clone())),
        Value::Iri(iri) => Ok((iri.clone(), None, None)),
        _ => bail!("Expected string or literal value"),
    }
}

/// Check if two language tags are compatible
fn compatible_languages(lang1: &Option<String>, lang2: &Option<String>) -> bool {
    match (lang1, lang2) {
        (None, None) => true,
        (Some(l1), Some(l2)) => l1 == l2,
        (Some(_), None) | (None, Some(_)) => false,
    }
}

/// Create a string value with optional language tag
fn create_string_value(s: &str, lang: Option<String>) -> Value {
    if let Some(lang_tag) = lang {
        Value::Literal {
            value: s.to_string(),
            language: Some(lang_tag),
            datatype: None,
        }
    } else {
        Value::String(s.to_string())
    }
}

/// Validate and normalize a language tag (BCP47)
fn validate_language_tag(tag: &str) -> Result<String> {
    // Basic validation: must not be empty, contains only ASCII alphanumeric and hyphens
    if tag.is_empty() {
        bail!("Language tag cannot be empty");
    }

    // Language tags should be ASCII
    if !tag.is_ascii() {
        bail!("Language tag must be ASCII");
    }

    // Normalize to lowercase (BCP47 recommends lowercase)
    let normalized = tag.to_lowercase();

    // Basic format check: starts with letter
    if !normalized
        .chars()
        .next()
        .expect("normalized string validated to be non-empty")
        .is_ascii_alphabetic()
    {
        bail!("Language tag must start with a letter");
    }

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_context() -> ExecutionContext {
        ExecutionContext {
            variables: HashMap::new(),
            namespaces: HashMap::new(),
            base_iri: None,
            dataset_context: None,
            query_time: chrono::Utc::now(),
            optimization_level: crate::extensions::OptimizationLevel::None,
            memory_limit: None,
            time_limit: None,
        }
    }

    #[test]
    fn test_strbefore_basic() {
        let func = StrBeforeFunction;
        let ctx = create_test_context();

        let string = Value::String("abc@example.org".to_string());
        let separator = Value::String("@".to_string());

        let result = func.execute(&[string, separator], &ctx).unwrap();
        assert_eq!(result, Value::String("abc".to_string()));
    }

    #[test]
    fn test_strbefore_not_found() {
        let func = StrBeforeFunction;
        let ctx = create_test_context();

        let string = Value::String("foobar".to_string());
        let separator = Value::String("xyz".to_string());

        let result = func.execute(&[string, separator], &ctx).unwrap();
        assert_eq!(result, Value::String("".to_string()));
    }

    #[test]
    fn test_strafter_basic() {
        let func = StrAfterFunction;
        let ctx = create_test_context();

        let string = Value::String("abc@example.org".to_string());
        let separator = Value::String("@".to_string());

        let result = func.execute(&[string, separator], &ctx).unwrap();
        assert_eq!(result, Value::String("example.org".to_string()));
    }

    #[test]
    fn test_strafter_not_found() {
        let func = StrAfterFunction;
        let ctx = create_test_context();

        let string = Value::String("foobar".to_string());
        let separator = Value::String("xyz".to_string());

        let result = func.execute(&[string, separator], &ctx).unwrap();
        assert_eq!(result, Value::String("".to_string()));
    }

    #[test]
    fn test_strlang() {
        let func = StrLangFunction;
        let ctx = create_test_context();

        let string = Value::String("chat".to_string());
        let lang = Value::String("fr".to_string());

        let result = func.execute(&[string, lang], &ctx).unwrap();
        match result {
            Value::Literal {
                value,
                language,
                datatype,
            } => {
                assert_eq!(value, "chat");
                assert_eq!(language, Some("fr".to_string()));
                assert_eq!(datatype, None);
            }
            _ => panic!("Expected Literal"),
        }
    }

    #[test]
    fn test_strlang_uppercase_normalized() {
        let func = StrLangFunction;
        let ctx = create_test_context();

        let string = Value::String("Hello".to_string());
        let lang = Value::String("EN".to_string());

        let result = func.execute(&[string, lang], &ctx).unwrap();
        match result {
            Value::Literal { language, .. } => {
                assert_eq!(language, Some("en".to_string()));
            }
            _ => panic!("Expected Literal"),
        }
    }

    #[test]
    fn test_strlangdir() {
        let func = StrLangDirFunction;
        let ctx = create_test_context();

        let string = Value::String("Hello".to_string());
        let lang = Value::String("en".to_string());
        let dir = Value::String("ltr".to_string());

        let result = func.execute(&[string, lang, dir], &ctx).unwrap();
        match result {
            Value::Literal { language, .. } => {
                assert_eq!(language, Some("en--ltr".to_string()));
            }
            _ => panic!("Expected Literal"),
        }
    }

    #[test]
    fn test_strlangdir_rtl() {
        let func = StrLangDirFunction;
        let ctx = create_test_context();

        let string = Value::String("مرحبا".to_string());
        let lang = Value::String("ar".to_string());
        let dir = Value::String("rtl".to_string());

        let result = func.execute(&[string, lang, dir], &ctx).unwrap();
        match result {
            Value::Literal { language, .. } => {
                assert_eq!(language, Some("ar--rtl".to_string()));
            }
            _ => panic!("Expected Literal"),
        }
    }

    #[test]
    fn test_strlangdir_invalid_direction() {
        let func = StrLangDirFunction;
        let ctx = create_test_context();

        let string = Value::String("Hello".to_string());
        let lang = Value::String("en".to_string());
        let dir = Value::String("invalid".to_string());

        let result = func.execute(&[string, lang, dir], &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_strdt() {
        let func = StrDtFunction;
        let ctx = create_test_context();

        let string = Value::String("123".to_string());
        let datatype = Value::Iri("http://www.w3.org/2001/XMLSchema#integer".to_string());

        let result = func.execute(&[string, datatype], &ctx).unwrap();
        match result {
            Value::Literal {
                value,
                language,
                datatype,
            } => {
                assert_eq!(value, "123");
                assert_eq!(language, None);
                assert_eq!(
                    datatype,
                    Some("http://www.w3.org/2001/XMLSchema#integer".to_string())
                );
            }
            _ => panic!("Expected Literal"),
        }
    }

    #[test]
    fn test_strbefore_with_language() {
        let func = StrBeforeFunction;
        let ctx = create_test_context();

        let string = Value::Literal {
            value: "abc@def".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        };
        let separator = Value::Literal {
            value: "@".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        };

        let result = func.execute(&[string, separator], &ctx).unwrap();
        match result {
            Value::Literal {
                value, language, ..
            } => {
                assert_eq!(value, "abc");
                assert_eq!(language, Some("en".to_string()));
            }
            _ => panic!("Expected Literal with language"),
        }
    }

    #[test]
    fn test_function_arities() {
        assert_eq!(StrBeforeFunction.arity(), Some(2));
        assert_eq!(StrAfterFunction.arity(), Some(2));
        assert_eq!(StrLangFunction.arity(), Some(2));
        assert_eq!(StrLangDirFunction.arity(), Some(3));
        assert_eq!(StrDtFunction.arity(), Some(2));
    }
}
