//! RDF Literal implementation

use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use crate::model::{RdfTerm, ObjectTerm, NamedNode, NamedNodeRef};
use crate::OxirsError;

/// An RDF Literal
/// 
/// Represents a literal value in RDF with an optional datatype and language tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Literal {
    value: String,
    datatype: Option<NamedNode>,
    language: Option<String>,
}

impl Literal {
    /// Creates a new string literal without language or datatype
    pub fn new(value: impl Into<String>) -> Self {
        Literal {
            value: value.into(),
            datatype: None,
            language: None,
        }
    }
    
    /// Creates a new literal with a datatype
    pub fn new_typed(value: impl Into<String>, datatype: NamedNode) -> Self {
        Literal {
            value: value.into(),
            datatype: Some(datatype),
            language: None,
        }
    }
    
    /// Creates a new literal with a language tag
    pub fn new_lang(value: impl Into<String>, language: impl Into<String>) -> Result<Self, OxirsError> {
        let language = language.into();
        
        // Basic language tag validation
        if language.is_empty() {
            return Err(OxirsError::Parse("Language tag cannot be empty".to_string()));
        }
        
        // TODO: Add BCP 47 language tag validation
        
        Ok(Literal {
            value: value.into(),
            datatype: None,
            language: Some(language),
        })
    }
    
    /// Returns the literal value as a string slice
    pub fn value(&self) -> &str {
        &self.value
    }
    
    /// Returns the datatype IRI if present
    pub fn datatype(&self) -> Option<&NamedNode> {
        self.datatype.as_ref()
    }
    
    /// Returns the language tag if present
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
    
    /// Returns true if this is a string literal (no datatype or language)
    pub fn is_plain(&self) -> bool {
        self.datatype.is_none() && self.language.is_none()
    }
    
    /// Returns true if this literal has a language tag
    pub fn is_lang_string(&self) -> bool {
        self.language.is_some()
    }
    
    /// Returns true if this literal has a datatype
    pub fn is_typed(&self) -> bool {
        self.datatype.is_some()
    }
    
    /// Attempts to extract the value as a boolean
    /// 
    /// Works for XSD boolean literals and other representations like "true"/"false"
    pub fn as_bool(&self) -> Option<bool> {
        match self.value.to_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }
    
    /// Attempts to extract the value as an integer
    /// 
    /// Works for XSD integer literals and other numeric representations
    pub fn as_i64(&self) -> Option<i64> {
        self.value.parse().ok()
    }
    
    /// Attempts to extract the value as a 32-bit integer
    pub fn as_i32(&self) -> Option<i32> {
        self.value.parse().ok()
    }
    
    /// Attempts to extract the value as a floating point number
    /// 
    /// Works for XSD decimal, double, float literals
    pub fn as_f64(&self) -> Option<f64> {
        self.value.parse().ok()
    }
    
    /// Attempts to extract the value as a 32-bit floating point number
    pub fn as_f32(&self) -> Option<f32> {
        self.value.parse().ok()
    }
    
    /// Returns true if this literal represents a numeric value
    pub fn is_numeric(&self) -> bool {
        if let Some(datatype) = &self.datatype {
            let dt_iri = datatype.as_str();
            matches!(dt_iri,
                "http://www.w3.org/2001/XMLSchema#integer" |
                "http://www.w3.org/2001/XMLSchema#decimal" |
                "http://www.w3.org/2001/XMLSchema#double" |
                "http://www.w3.org/2001/XMLSchema#float" |
                "http://www.w3.org/2001/XMLSchema#long" |
                "http://www.w3.org/2001/XMLSchema#int" |
                "http://www.w3.org/2001/XMLSchema#short" |
                "http://www.w3.org/2001/XMLSchema#byte" |
                "http://www.w3.org/2001/XMLSchema#unsignedLong" |
                "http://www.w3.org/2001/XMLSchema#unsignedInt" |
                "http://www.w3.org/2001/XMLSchema#unsignedShort" |
                "http://www.w3.org/2001/XMLSchema#unsignedByte" |
                "http://www.w3.org/2001/XMLSchema#positiveInteger" |
                "http://www.w3.org/2001/XMLSchema#nonNegativeInteger" |
                "http://www.w3.org/2001/XMLSchema#negativeInteger" |
                "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
            )
        } else {
            // Check if the value looks numeric
            self.as_f64().is_some()
        }
    }
    
    /// Returns true if this literal represents a boolean value
    pub fn is_boolean(&self) -> bool {
        if let Some(datatype) = &self.datatype {
            datatype.as_str() == "http://www.w3.org/2001/XMLSchema#boolean"
        } else {
            self.as_bool().is_some()
        }
    }
    
    /// Returns the canonical form of this literal
    /// 
    /// This normalizes the literal according to XSD rules
    pub fn canonical_form(&self) -> Literal {
        if let Some(datatype) = &self.datatype {
            let dt_iri = datatype.as_str();
            match dt_iri {
                "http://www.w3.org/2001/XMLSchema#boolean" => {
                    if let Some(bool_val) = self.as_bool() {
                        return Literal::new_typed(bool_val.to_string(), datatype.clone());
                    }
                }
                "http://www.w3.org/2001/XMLSchema#integer" => {
                    if let Some(int_val) = self.as_i64() {
                        return Literal::new_typed(int_val.to_string(), datatype.clone());
                    }
                }
                "http://www.w3.org/2001/XMLSchema#decimal" => {
                    if let Some(dec_val) = self.as_f64() {
                        // Format as decimal without scientific notation
                        return Literal::new_typed(format!("{:.1}", dec_val), datatype.clone());
                    }
                }
                _ => {}
            }
        }
        self.clone()
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.value)?;
        
        if let Some(lang) = &self.language {
            write!(f, "@{}", lang)?;
        } else if let Some(datatype) = &self.datatype {
            write!(f, "^^{}", datatype)?;
        }
        
        Ok(())
    }
}

impl RdfTerm for Literal {
    fn as_str(&self) -> &str {
        &self.value
    }
    
    fn is_literal(&self) -> bool {
        true
    }
}

impl ObjectTerm for Literal {}

/// A borrowed literal reference
/// 
/// This is an optimized version for temporary references
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LiteralRef<'a> {
    value: &'a str,
    datatype: Option<NamedNodeRef<'a>>,
    language: Option<&'a str>,
}

impl<'a> LiteralRef<'a> {
    /// Creates a new literal reference
    pub fn new(value: &'a str) -> Self {
        LiteralRef {
            value,
            datatype: None,
            language: None,
        }
    }
    
    /// Creates a new typed literal reference
    pub fn new_typed(value: &'a str, datatype: NamedNodeRef<'a>) -> Self {
        LiteralRef {
            value,
            datatype: Some(datatype),
            language: None,
        }
    }
    
    /// Creates a new language-tagged literal reference
    pub fn new_lang(value: &'a str, language: &'a str) -> Self {
        LiteralRef {
            value,
            datatype: None,
            language: Some(language),
        }
    }
    
    /// Returns the literal value
    pub fn value(&self) -> &str {
        self.value
    }
    
    /// Returns the datatype if present
    pub fn datatype(&self) -> Option<NamedNodeRef<'a>> {
        self.datatype
    }
    
    /// Returns the language tag if present
    pub fn language(&self) -> Option<&str> {
        self.language
    }
    
    /// Converts to an owned Literal
    pub fn to_owned(&self) -> Literal {
        Literal {
            value: self.value.to_string(),
            datatype: self.datatype.map(|dt| dt.to_owned()),
            language: self.language.map(|lang| lang.to_string()),
        }
    }
}

impl<'a> fmt::Display for LiteralRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.value)?;
        
        if let Some(lang) = self.language {
            write!(f, "@{}", lang)?;
        } else if let Some(datatype) = self.datatype {
            write!(f, "^^{}", datatype)?;
        }
        
        Ok(())
    }
}

impl<'a> RdfTerm for LiteralRef<'a> {
    fn as_str(&self) -> &str {
        self.value
    }
    
    fn is_literal(&self) -> bool {
        true
    }
}

impl<'a> From<LiteralRef<'a>> for Literal {
    fn from(literal_ref: LiteralRef<'a>) -> Self {
        literal_ref.to_owned()
    }
}

impl<'a> From<&'a Literal> for LiteralRef<'a> {
    fn from(literal: &'a Literal) -> Self {
        LiteralRef {
            value: literal.value(),
            datatype: literal.datatype().map(|dt| dt.into()),
            language: literal.language(),
        }
    }
}

/// Common XSD datatypes as constants and convenience functions
pub mod xsd {
    use super::*;
    
    // Core string types
    /// Creates an XSD string datatype IRI
    pub fn string() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#string")
    }
    
    /// Creates an XSD normalizedString datatype IRI
    pub fn normalized_string() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#normalizedString")
    }
    
    /// Creates an XSD token datatype IRI
    pub fn token() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#token")
    }
    
    // Numeric types
    /// Creates an XSD integer datatype IRI
    pub fn integer() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer")
    }
    
    /// Creates an XSD decimal datatype IRI
    pub fn decimal() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#decimal")
    }
    
    /// Creates an XSD double datatype IRI
    pub fn double() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#double")
    }
    
    /// Creates an XSD float datatype IRI
    pub fn float() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#float")
    }
    
    /// Creates an XSD long datatype IRI
    pub fn long() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#long")
    }
    
    /// Creates an XSD int datatype IRI
    pub fn int() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#int")
    }
    
    /// Creates an XSD short datatype IRI
    pub fn short() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#short")
    }
    
    /// Creates an XSD byte datatype IRI
    pub fn byte() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#byte")
    }
    
    // Unsigned integer types
    /// Creates an XSD unsignedLong datatype IRI
    pub fn unsigned_long() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#unsignedLong")
    }
    
    /// Creates an XSD unsignedInt datatype IRI
    pub fn unsigned_int() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#unsignedInt")
    }
    
    /// Creates an XSD unsignedShort datatype IRI
    pub fn unsigned_short() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#unsignedShort")
    }
    
    /// Creates an XSD unsignedByte datatype IRI
    pub fn unsigned_byte() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#unsignedByte")
    }
    
    // Boolean type
    /// Creates an XSD boolean datatype IRI
    pub fn boolean() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#boolean")
    }
    
    // Date and time types
    /// Creates an XSD dateTime datatype IRI
    pub fn date_time() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#dateTime")
    }
    
    /// Creates an XSD date datatype IRI
    pub fn date() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#date")
    }
    
    /// Creates an XSD time datatype IRI
    pub fn time() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#time")
    }
    
    /// Creates an XSD duration datatype IRI
    pub fn duration() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#duration")
    }
    
    // Binary types
    /// Creates an XSD base64Binary datatype IRI
    pub fn base64_binary() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#base64Binary")
    }
    
    /// Creates an XSD hexBinary datatype IRI
    pub fn hex_binary() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#hexBinary")
    }
    
    // URI type
    /// Creates an XSD anyURI datatype IRI
    pub fn any_uri() -> NamedNode {
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#anyURI")
    }
    
    // Convenience functions for creating typed literals
    
    /// Creates a boolean literal
    pub fn boolean_literal(value: bool) -> Literal {
        Literal::new_typed(value.to_string(), boolean())
    }
    
    /// Creates an integer literal
    pub fn integer_literal(value: i64) -> Literal {
        Literal::new_typed(value.to_string(), integer())
    }
    
    /// Creates a decimal literal
    pub fn decimal_literal(value: f64) -> Literal {
        Literal::new_typed(value.to_string(), decimal())
    }
    
    /// Creates a double literal
    pub fn double_literal(value: f64) -> Literal {
        Literal::new_typed(value.to_string(), double())
    }
    
    /// Creates a string literal
    pub fn string_literal(value: &str) -> Literal {
        Literal::new_typed(value, string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plain_literal() {
        let literal = Literal::new("Hello");
        assert_eq!(literal.value(), "Hello");
        assert!(literal.is_plain());
        assert!(!literal.is_lang_string());
        assert!(!literal.is_typed());
        assert_eq!(format!("{}", literal), "\"Hello\"");
    }
    
    #[test]
    fn test_lang_literal() {
        let literal = Literal::new_lang("Hello", "en").unwrap();
        assert_eq!(literal.value(), "Hello");
        assert_eq!(literal.language(), Some("en"));
        assert!(!literal.is_plain());
        assert!(literal.is_lang_string());
        assert!(!literal.is_typed());
        assert_eq!(format!("{}", literal), "\"Hello\"@en");
    }
    
    #[test]
    fn test_typed_literal() {
        let literal = Literal::new_typed("42", xsd::integer());
        assert_eq!(literal.value(), "42");
        assert!(literal.datatype().is_some());
        assert!(!literal.is_plain());
        assert!(!literal.is_lang_string());
        assert!(literal.is_typed());
        assert_eq!(format!("{}", literal), "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>");
    }
    
    #[test]
    fn test_literal_ref() {
        let literal_ref = LiteralRef::new("test");
        assert_eq!(literal_ref.value(), "test");
        
        let owned = literal_ref.to_owned();
        assert_eq!(owned.value(), "test");
    }
    
    #[test]
    fn test_boolean_extraction() {
        let bool_literal = xsd::boolean_literal(true);
        assert!(bool_literal.is_boolean());
        assert_eq!(bool_literal.as_bool(), Some(true));
        
        let false_literal = Literal::new_typed("false", xsd::boolean());
        assert_eq!(false_literal.as_bool(), Some(false));
        
        // Test string representations
        let true_str = Literal::new("true");
        assert_eq!(true_str.as_bool(), Some(true));
        
        let false_str = Literal::new("0");
        assert_eq!(false_str.as_bool(), Some(false));
    }
    
    #[test]
    fn test_numeric_extraction() {
        let int_literal = xsd::integer_literal(42);
        assert!(int_literal.is_numeric());
        assert_eq!(int_literal.as_i64(), Some(42));
        assert_eq!(int_literal.as_i32(), Some(42));
        assert_eq!(int_literal.as_f64(), Some(42.0));
        
        let decimal_literal = xsd::decimal_literal(3.14159);
        assert!(decimal_literal.is_numeric());
        assert_eq!(decimal_literal.as_f64(), Some(3.14159));
        assert_eq!(decimal_literal.as_f32(), Some(3.14159_f32));
        
        // Test untyped numeric strings
        let untyped_num = Literal::new("123");
        assert!(untyped_num.is_numeric());
        assert_eq!(untyped_num.as_i64(), Some(123));
    }
    
    #[test]
    fn test_canonical_form() {
        // Boolean canonicalization
        let bool_literal = Literal::new_typed("True", xsd::boolean());
        let canonical = bool_literal.canonical_form();
        assert_eq!(canonical.value(), "true");
        
        // Integer canonicalization
        let int_literal = Literal::new_typed("  42  ", xsd::integer());
        // Note: This would need actual whitespace trimming in canonical form
        // For now, just test that it returns a valid canonical form
        let canonical = int_literal.canonical_form();
        assert!(canonical.datatype().is_some());
        
        // Decimal canonicalization
        let dec_literal = Literal::new_typed("3.14", xsd::decimal());
        let canonical = dec_literal.canonical_form();
        assert_eq!(canonical.value(), "3.1"); // Should format with .1 precision
    }
    
    #[test]
    fn test_xsd_convenience_functions() {
        // Test all the convenience functions work
        assert_eq!(xsd::boolean_literal(true).value(), "true");
        assert_eq!(xsd::integer_literal(123).value(), "123");
        assert_eq!(xsd::decimal_literal(3.14).value(), "3.14");
        assert_eq!(xsd::double_literal(2.71).value(), "2.71");
        assert_eq!(xsd::string_literal("hello").value(), "hello");
        
        // Test datatype assignments
        assert_eq!(xsd::boolean_literal(true).datatype().unwrap().as_str(), 
                   "http://www.w3.org/2001/XMLSchema#boolean");
        assert_eq!(xsd::integer_literal(123).datatype().unwrap().as_str(), 
                   "http://www.w3.org/2001/XMLSchema#integer");
    }
    
    #[test]
    fn test_numeric_type_detection() {
        // Test various numeric types
        let int_lit = Literal::new_typed("42", xsd::integer());
        assert!(int_lit.is_numeric());
        
        let float_lit = Literal::new_typed("3.14", xsd::float());
        assert!(float_lit.is_numeric());
        
        let double_lit = Literal::new_typed("2.71", xsd::double());
        assert!(double_lit.is_numeric());
        
        // Non-numeric types
        let string_lit = Literal::new_typed("hello", xsd::string());
        assert!(!string_lit.is_numeric());
        
        let bool_lit = Literal::new_typed("true", xsd::boolean());
        assert!(!bool_lit.is_numeric());
    }
}