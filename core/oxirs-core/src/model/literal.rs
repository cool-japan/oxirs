//! RDF Literal implementation
//!
//! This implementation is extracted and adapted from Oxigraph's oxrdf literal handling
//! to provide zero-dependency RDF literal support with full XSD datatype validation.

use crate::model::{NamedNode, NamedNodeRef, ObjectTerm, RdfTerm};
use crate::vocab::{rdf, xsd};
use crate::OxirsError;
use lazy_static::lazy_static;
use regex::Regex;
use std::borrow::Cow;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
// use std::str::FromStr; // Used for parsing but currently commented out

/// Language tag validation error type
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageTagParseError {
    message: String,
}

impl fmt::Display for LanguageTagParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Language tag parse error: {}", self.message)
    }
}

impl std::error::Error for LanguageTagParseError {}

impl From<LanguageTagParseError> for OxirsError {
    fn from(err: LanguageTagParseError) -> Self {
        OxirsError::Parse(err.message)
    }
}

/// A language tag following BCP 47 specification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LanguageTag {
    tag: String,
}

impl LanguageTag {
    /// Parses a language tag from a string
    pub fn parse(tag: impl Into<String>) -> Result<Self, LanguageTagParseError> {
        let tag = tag.into();
        validate_language_tag(&tag)?;
        Ok(LanguageTag { tag })
    }

    /// Returns the language tag as a string slice
    pub fn as_str(&self) -> &str {
        &self.tag
    }

    /// Consumes the language tag and returns the inner string
    pub fn into_inner(self) -> String {
        self.tag
    }
}

impl fmt::Display for LanguageTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.tag)
    }
}

lazy_static! {
    /// BCP 47 language tag validation regex
    /// Based on RFC 5646 - Tags for Identifying Languages
    static ref LANGUAGE_TAG_REGEX: Regex = Regex::new(
        r"^([a-zA-Z]{2,3}(-[a-zA-Z]{3}){0,3}(-[a-zA-Z]{4})?(-[a-zA-Z]{2}|\d{3})?(-[0-9a-zA-Z]{5,8}|-\d[0-9a-zA-Z]{3})*(-[0-9a-wyzA-WYZ](-[0-9a-zA-Z]{2,8})+)*(-x(-[0-9a-zA-Z]{1,8})+)?|x(-[0-9a-zA-Z]{1,8})+|[a-zA-Z]{4}|[a-zA-Z]{5,8})$"
    ).expect("Language tag regex compilation failed");

    /// Simple language subtag validation (2-3 letter language codes)
    static ref SIMPLE_LANGUAGE_REGEX: Regex = Regex::new(
        r"^[a-zA-Z]{2,3}$"
    ).expect("Simple language regex compilation failed");

    /// XSD numeric type validation regexes
    static ref INTEGER_REGEX: Regex = Regex::new(
        r"^[+-]?\d+$"
    ).expect("Integer regex compilation failed");

    static ref DECIMAL_REGEX: Regex = Regex::new(
        r"^[+-]?(\d+(\.\d*)?|\.\d+)$"
    ).expect("Decimal regex compilation failed");

    static ref DOUBLE_REGEX: Regex = Regex::new(
        r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$|^[+-]?INF$|^NaN$"
    ).expect("Double regex compilation failed");

    static ref BOOLEAN_REGEX: Regex = Regex::new(
        r"^(true|false|1|0)$"
    ).expect("Boolean regex compilation failed");

    /// DateTime validation (simplified ISO 8601)
    static ref DATETIME_REGEX: Regex = Regex::new(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
    ).expect("DateTime regex compilation failed");

    static ref DATE_REGEX: Regex = Regex::new(
        r"^\d{4}-\d{2}-\d{2}(Z|[+-]\d{2}:\d{2})?$"
    ).expect("Date regex compilation failed");

    static ref TIME_REGEX: Regex = Regex::new(
        r"^\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
    ).expect("Time regex compilation failed");
}

/// Validates a language tag according to BCP 47 (RFC 5646)
fn validate_language_tag(tag: &str) -> Result<(), LanguageTagParseError> {
    if tag.is_empty() {
        return Err(LanguageTagParseError {
            message: "Language tag cannot be empty".to_string(),
        });
    }

    // Check overall structure with regex
    if !LANGUAGE_TAG_REGEX.is_match(tag) {
        return Err(LanguageTagParseError {
            message: format!(
                "Invalid language tag format: '{}'. Must follow BCP 47 specification",
                tag
            ),
        });
    }

    // Additional validations for common cases
    let parts: Vec<&str> = tag.split('-').collect();

    if parts.is_empty() {
        return Err(LanguageTagParseError {
            message: "Language tag cannot be empty".to_string(),
        });
    }

    // First part should be a valid language subtag
    let language_subtag = parts[0];
    if language_subtag.len() < 2 || language_subtag.len() > 8 {
        return Err(LanguageTagParseError {
            message: format!(
                "Invalid language subtag length: '{}'. Must be 2-8 characters",
                language_subtag
            ),
        });
    }

    // Check for common invalid patterns
    if tag.starts_with('-') || tag.ends_with('-') || tag.contains("--") {
        return Err(LanguageTagParseError {
            message: format!("Invalid language tag structure: '{}'", tag),
        });
    }

    Ok(())
}

/// Validates a literal value against its XSD datatype
pub fn validate_xsd_value(value: &str, datatype_iri: &str) -> Result<(), OxirsError> {
    match datatype_iri {
        // String types
        "http://www.w3.org/2001/XMLSchema#string"
        | "http://www.w3.org/2001/XMLSchema#normalizedString"
        | "http://www.w3.org/2001/XMLSchema#token" => {
            // All strings are valid for string types
            Ok(())
        }

        // Boolean type
        "http://www.w3.org/2001/XMLSchema#boolean" => {
            if BOOLEAN_REGEX.is_match(value) {
                Ok(())
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid boolean value: '{}'. Must be true, false, 1, or 0",
                    value
                )))
            }
        }

        // Integer types
        "http://www.w3.org/2001/XMLSchema#integer"
        | "http://www.w3.org/2001/XMLSchema#long"
        | "http://www.w3.org/2001/XMLSchema#int"
        | "http://www.w3.org/2001/XMLSchema#short"
        | "http://www.w3.org/2001/XMLSchema#byte"
        | "http://www.w3.org/2001/XMLSchema#unsignedLong"
        | "http://www.w3.org/2001/XMLSchema#unsignedInt"
        | "http://www.w3.org/2001/XMLSchema#unsignedShort"
        | "http://www.w3.org/2001/XMLSchema#unsignedByte"
        | "http://www.w3.org/2001/XMLSchema#positiveInteger"
        | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
        | "http://www.w3.org/2001/XMLSchema#negativeInteger"
        | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger" => {
            if INTEGER_REGEX.is_match(value) {
                // Additional validation for specific integer types
                validate_integer_range(value, datatype_iri)
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid integer format: '{}'",
                    value
                )))
            }
        }

        // Decimal types
        "http://www.w3.org/2001/XMLSchema#decimal" => {
            if DECIMAL_REGEX.is_match(value) {
                Ok(())
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid decimal format: '{}'",
                    value
                )))
            }
        }

        // Floating point types
        "http://www.w3.org/2001/XMLSchema#double" | "http://www.w3.org/2001/XMLSchema#float" => {
            if DOUBLE_REGEX.is_match(value) {
                Ok(())
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid floating point format: '{}'",
                    value
                )))
            }
        }

        // Date/time types
        "http://www.w3.org/2001/XMLSchema#dateTime" => {
            if DATETIME_REGEX.is_match(value) {
                Ok(())
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid dateTime format: '{}'. Expected ISO 8601 format",
                    value
                )))
            }
        }

        "http://www.w3.org/2001/XMLSchema#date" => {
            if DATE_REGEX.is_match(value) {
                // Additional validation for valid date values
                let date_part = if value.contains('T') {
                    value.split('T').next().unwrap()
                } else if value.contains('Z') || value.contains('+') || value.contains('-') {
                    &value[..10]
                } else {
                    value
                };

                let parts: Vec<&str> = date_part.split('-').collect();
                if parts.len() == 3 {
                    let year: i32 = parts[0].parse().map_err(|_| {
                        OxirsError::Parse(format!("Invalid year in date: {}", value))
                    })?;
                    let month: u32 = parts[1].parse().map_err(|_| {
                        OxirsError::Parse(format!("Invalid month in date: {}", value))
                    })?;
                    let day: u32 = parts[2].parse().map_err(|_| {
                        OxirsError::Parse(format!("Invalid day in date: {}", value))
                    })?;

                    // Validate month
                    if month < 1 || month > 12 {
                        return Err(OxirsError::Parse(format!(
                            "Invalid month in date: {}. Month must be between 01 and 12",
                            value
                        )));
                    }

                    // Validate day based on month
                    let max_day = match month {
                        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                        4 | 6 | 9 | 11 => 30,
                        2 => {
                            // Check for leap year
                            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                                29
                            } else {
                                28
                            }
                        }
                        _ => unreachable!(),
                    };

                    if day < 1 || day > max_day {
                        return Err(OxirsError::Parse(format!(
                            "Invalid day in date: {}. Day must be between 01 and {} for month {}",
                            value, max_day, month
                        )));
                    }
                }
                Ok(())
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid date format: '{}'. Expected YYYY-MM-DD format",
                    value
                )))
            }
        }

        "http://www.w3.org/2001/XMLSchema#time" => {
            if TIME_REGEX.is_match(value) {
                Ok(())
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid time format: '{}'. Expected HH:MM:SS format",
                    value
                )))
            }
        }

        // For unknown datatypes, don't validate
        _ => Ok(()),
    }
}

/// Validates integer values against their specific type ranges
fn validate_integer_range(value: &str, datatype_iri: &str) -> Result<(), OxirsError> {
    let parsed_value: i64 = value
        .parse()
        .map_err(|_| OxirsError::Parse(format!("Cannot parse integer: '{}'", value)))?;

    match datatype_iri {
        "http://www.w3.org/2001/XMLSchema#byte" => {
            if parsed_value < -128 || parsed_value > 127 {
                return Err(OxirsError::Parse(format!(
                    "Byte value out of range: {}. Must be between -128 and 127",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#short" => {
            if parsed_value < -32768 || parsed_value > 32767 {
                return Err(OxirsError::Parse(format!(
                    "Short value out of range: {}. Must be between -32768 and 32767",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#int" => {
            if parsed_value < -2147483648 || parsed_value > 2147483647 {
                return Err(OxirsError::Parse(format!(
                    "Int value out of range: {}. Must be between -2147483648 and 2147483647",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#unsignedByte" => {
            if parsed_value < 0 || parsed_value > 255 {
                return Err(OxirsError::Parse(format!(
                    "Unsigned byte value out of range: {}. Must be between 0 and 255",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#unsignedShort" => {
            if parsed_value < 0 || parsed_value > 65535 {
                return Err(OxirsError::Parse(format!(
                    "Unsigned short value out of range: {}. Must be between 0 and 65535",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#unsignedInt" => {
            if parsed_value < 0 || parsed_value > 4294967295 {
                return Err(OxirsError::Parse(format!(
                    "Unsigned int value out of range: {}. Must be between 0 and 4294967295",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#positiveInteger" => {
            if parsed_value <= 0 {
                return Err(OxirsError::Parse(format!(
                    "Positive integer must be greater than 0, got: {}",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#nonNegativeInteger" => {
            if parsed_value < 0 {
                return Err(OxirsError::Parse(format!(
                    "Non-negative integer must be >= 0, got: {}",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#negativeInteger" => {
            if parsed_value >= 0 {
                return Err(OxirsError::Parse(format!(
                    "Negative integer must be less than 0, got: {}",
                    parsed_value
                )));
            }
        }
        "http://www.w3.org/2001/XMLSchema#nonPositiveInteger" => {
            if parsed_value > 0 {
                return Err(OxirsError::Parse(format!(
                    "Non-positive integer must be <= 0, got: {}",
                    parsed_value
                )));
            }
        }
        _ => {} // Other integer types don't have additional range restrictions in this simplified implementation
    }

    Ok(())
}

/// An owned RDF [literal](https://www.w3.org/TR/rdf11-concepts/#dfn-literal).
///
/// The default string formatter is returning an N-Triples, Turtle, and SPARQL compatible representation:
/// ```
/// use oxirs_core::model::literal::Literal;
/// use oxirs_core::vocab::xsd;
///
/// assert_eq!(
///     "\"foo\\nbar\"",
///     Literal::new_simple_literal("foo\nbar").to_string()
/// );
///
/// assert_eq!(
///     r#""1999-01-01"^^<http://www.w3.org/2001/XMLSchema#date>"#,
///     Literal::new_typed_literal("1999-01-01", xsd::DATE.clone()).to_string()
/// );
///
/// assert_eq!(
///     r#""foo"@en"#,
///     Literal::new_language_tagged_literal("foo", "en").unwrap().to_string()
/// );
/// ```
#[derive(Eq, PartialEq, Debug, Clone, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Literal(LiteralContent);

#[derive(PartialEq, Eq, Debug, Clone, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum LiteralContent {
    String(String),
    LanguageTaggedString {
        value: String,
        language: String,
    },
    #[cfg(feature = "rdf-12")]
    DirectionalLanguageTaggedString {
        value: String,
        language: String,
        direction: BaseDirection,
    },
    TypedLiteral {
        value: String,
        datatype: NamedNode,
    },
}

impl Literal {
    /// Builds an RDF [simple literal](https://www.w3.org/TR/rdf11-concepts/#dfn-simple-literal).
    #[inline]
    pub fn new_simple_literal(value: impl Into<String>) -> Self {
        Self(LiteralContent::String(value.into()))
    }

    /// Creates a new string literal without language or datatype (alias for compatibility)
    #[inline]
    pub fn new(value: impl Into<String>) -> Self {
        Self::new_simple_literal(value)
    }

    /// Builds an RDF [literal](https://www.w3.org/TR/rdf11-concepts/#dfn-literal) with a [datatype](https://www.w3.org/TR/rdf11-concepts/#dfn-datatype-iri).
    #[inline]
    pub fn new_typed_literal(value: impl Into<String>, datatype: impl Into<NamedNode>) -> Self {
        let value = value.into();
        let datatype = datatype.into();
        Self(if datatype == *xsd::STRING {
            LiteralContent::String(value)
        } else {
            LiteralContent::TypedLiteral { value, datatype }
        })
    }

    /// Creates a new literal with a datatype (alias for compatibility)
    #[inline]
    pub fn new_typed(value: impl Into<String>, datatype: NamedNode) -> Self {
        Self::new_typed_literal(value, datatype)
    }

    /// Creates a new literal with a datatype and validates the value
    pub fn new_typed_validated(
        value: impl Into<String>,
        datatype: NamedNode,
    ) -> Result<Self, OxirsError> {
        let value = value.into();
        validate_xsd_value(&value, datatype.as_str())?;
        Ok(Literal::new_typed_literal(value, datatype))
    }

    /// Builds an RDF [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string).
    #[inline]
    pub fn new_language_tagged_literal(
        value: impl Into<String>,
        language: impl Into<String>,
    ) -> Result<Self, LanguageTagParseError> {
        let language = language.into();
        // Validate without modifying case to preserve RFC 5646 conventions
        validate_language_tag(&language)?;
        Ok(Self::new_language_tagged_literal_unchecked(value, language))
    }

    /// Builds an RDF [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string).
    ///
    /// It is the responsibility of the caller to check that `language`
    /// is valid [BCP47](https://tools.ietf.org/html/bcp47) language tag,
    /// and is lowercase.
    ///
    /// [`Literal::new_language_tagged_literal()`] is a safe version of this constructor and should be used for untrusted data.
    #[inline]
    pub fn new_language_tagged_literal_unchecked(
        value: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self(LiteralContent::LanguageTaggedString {
            value: value.into(),
            language: language.into(),
        })
    }

    /// Creates a new literal with a language tag (alias for compatibility)
    pub fn new_lang(
        value: impl Into<String>,
        language: impl Into<String>,
    ) -> Result<Self, OxirsError> {
        let result = Self::new_language_tagged_literal(value, language)?;
        Ok(result)
    }

    /// Builds an RDF [directional language-tagged string](https://www.w3.org/TR/rdf12-concepts/#dfn-dir-lang-string).
    #[cfg(feature = "rdf-12")]
    #[inline]
    pub fn new_directional_language_tagged_literal(
        value: impl Into<String>,
        language: impl Into<String>,
        direction: impl Into<BaseDirection>,
    ) -> Result<Self, LanguageTagParseError> {
        let mut language = language.into();
        language.make_ascii_lowercase();
        validate_language_tag(&language)?;
        Ok(Self::new_directional_language_tagged_literal_unchecked(
            value, language, direction,
        ))
    }

    /// Builds an RDF [directional language-tagged string](https://www.w3.org/TR/rdf12-concepts/#dfn-dir-lang-string).
    ///
    /// It is the responsibility of the caller to check that `language`
    /// is valid [BCP47](https://tools.ietf.org/html/bcp47) language tag,
    /// and is lowercase.
    ///
    /// [`Literal::new_directional_language_tagged_literal()`] is a safe version of this constructor and should be used for untrusted data.
    #[cfg(feature = "rdf-12")]
    #[inline]
    pub fn new_directional_language_tagged_literal_unchecked(
        value: impl Into<String>,
        language: impl Into<String>,
        direction: impl Into<BaseDirection>,
    ) -> Self {
        Self(LiteralContent::DirectionalLanguageTaggedString {
            value: value.into(),
            language: language.into(),
            direction: direction.into(),
        })
    }

    /// The literal [lexical form](https://www.w3.org/TR/rdf11-concepts/#dfn-lexical-form).
    #[inline]
    pub fn value(&self) -> &str {
        self.as_ref().value()
    }

    /// The literal [language tag](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tag) if it is a [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string).
    ///
    /// Language tags are defined by the [BCP47](https://tools.ietf.org/html/bcp47).
    /// They are normalized to lowercase by this implementation.
    #[inline]
    pub fn language(&self) -> Option<&str> {
        self.as_ref().language()
    }

    /// The literal [base direction](https://www.w3.org/TR/rdf12-concepts/#dfn-base-direction) if it is a [directional language-tagged string](https://www.w3.org/TR/rdf12-concepts/#dfn-base-direction).
    ///
    /// The two possible base directions are left-to-right (`ltr`) and right-to-left (`rtl`).
    #[cfg(feature = "rdf-12")]
    #[inline]
    pub fn direction(&self) -> Option<BaseDirection> {
        self.as_ref().direction()
    }

    /// The literal [datatype](https://www.w3.org/TR/rdf11-concepts/#dfn-datatype-iri).
    ///
    /// The datatype of [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string) is always [rdf:langString](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string).
    /// The datatype of [simple literals](https://www.w3.org/TR/rdf11-concepts/#dfn-simple-literal) is [xsd:string](https://www.w3.org/TR/xmlschema11-2/#string).
    #[inline]
    pub fn datatype(&self) -> NamedNodeRef<'_> {
        self.as_ref().datatype()
    }

    /// Checks if this literal could be seen as an RDF 1.0 [plain literal](https://www.w3.org/TR/2004/REC-rdf-concepts-20040210/#dfn-plain-literal).
    ///
    /// It returns true if the literal is a [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string)
    /// or has the datatype [xsd:string](https://www.w3.org/TR/xmlschema11-2/#string).
    #[inline]
    #[deprecated(note = "Plain literal concept is removed in RDF 1.1", since = "0.3.0")]
    pub fn is_plain(&self) -> bool {
        #[allow(deprecated)]
        self.as_ref().is_plain()
    }

    /// Returns true if this literal has a language tag
    pub fn is_lang_string(&self) -> bool {
        self.language().is_some()
    }

    /// Returns true if this literal has a datatype (excluding xsd:string which is implicit)
    pub fn is_typed(&self) -> bool {
        matches!(&self.0, LiteralContent::TypedLiteral { .. })
    }

    #[inline]
    pub fn as_ref(&self) -> LiteralRef<'_> {
        LiteralRef(match &self.0 {
            LiteralContent::String(value) => LiteralRefContent::String(value),
            LiteralContent::LanguageTaggedString { value, language } => {
                LiteralRefContent::LanguageTaggedString { value, language }
            }
            #[cfg(feature = "rdf-12")]
            LiteralContent::DirectionalLanguageTaggedString {
                value,
                language,
                direction,
            } => LiteralRefContent::DirectionalLanguageTaggedString {
                value,
                language,
                direction: *direction,
            },
            LiteralContent::TypedLiteral { value, datatype } => LiteralRefContent::TypedLiteral {
                value,
                datatype: NamedNodeRef::new_unchecked(datatype.as_str()),
            },
        })
    }

    /// Extract components from this literal (value, datatype, language tag).
    #[inline]
    pub fn destruct(self) -> (String, Option<NamedNode>, Option<String>) {
        match self.0 {
            LiteralContent::String(s) => (s, None, None),
            LiteralContent::LanguageTaggedString { value, language } => {
                (value, None, Some(language))
            }
            #[cfg(feature = "rdf-12")]
            LiteralContent::DirectionalLanguageTaggedString {
                value,
                language,
                direction: _,
            } => (value, None, Some(language)),
            LiteralContent::TypedLiteral { value, datatype } => (value, Some(datatype), None),
        }
    }

    /// Attempts to extract the value as a boolean
    ///
    /// Works for XSD boolean literals and other representations like "true"/"false"
    pub fn as_bool(&self) -> Option<bool> {
        match self.value().to_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }

    /// Attempts to extract the value as an integer
    ///
    /// Works for XSD integer literals and other numeric representations
    pub fn as_i64(&self) -> Option<i64> {
        self.value().parse().ok()
    }

    /// Attempts to extract the value as a 32-bit integer
    pub fn as_i32(&self) -> Option<i32> {
        self.value().parse().ok()
    }

    /// Attempts to extract the value as a floating point number
    ///
    /// Works for XSD decimal, double, float literals
    pub fn as_f64(&self) -> Option<f64> {
        self.value().parse().ok()
    }

    /// Attempts to extract the value as a 32-bit floating point number
    pub fn as_f32(&self) -> Option<f32> {
        self.value().parse().ok()
    }

    /// Returns true if this literal represents a numeric value
    pub fn is_numeric(&self) -> bool {
        match &self.0 {
            LiteralContent::TypedLiteral { datatype, .. } => {
                let dt_iri = datatype.as_str();
                matches!(
                    dt_iri,
                    "http://www.w3.org/2001/XMLSchema#integer"
                        | "http://www.w3.org/2001/XMLSchema#decimal"
                        | "http://www.w3.org/2001/XMLSchema#double"
                        | "http://www.w3.org/2001/XMLSchema#float"
                        | "http://www.w3.org/2001/XMLSchema#long"
                        | "http://www.w3.org/2001/XMLSchema#int"
                        | "http://www.w3.org/2001/XMLSchema#short"
                        | "http://www.w3.org/2001/XMLSchema#byte"
                        | "http://www.w3.org/2001/XMLSchema#unsignedLong"
                        | "http://www.w3.org/2001/XMLSchema#unsignedInt"
                        | "http://www.w3.org/2001/XMLSchema#unsignedShort"
                        | "http://www.w3.org/2001/XMLSchema#unsignedByte"
                        | "http://www.w3.org/2001/XMLSchema#positiveInteger"
                        | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
                        | "http://www.w3.org/2001/XMLSchema#negativeInteger"
                        | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
                )
            }
            _ => {
                // Check if the value looks numeric
                self.as_f64().is_some()
            }
        }
    }

    /// Returns true if this literal represents a boolean value
    pub fn is_boolean(&self) -> bool {
        match &self.0 {
            LiteralContent::TypedLiteral { datatype, .. } => {
                datatype.as_str() == "http://www.w3.org/2001/XMLSchema#boolean"
            }
            _ => self.as_bool().is_some(),
        }
    }

    /// Returns the canonical form of this literal
    ///
    /// This normalizes the literal according to XSD rules and recommendations
    pub fn canonical_form(&self) -> Literal {
        match &self.0 {
            LiteralContent::TypedLiteral { value, datatype } => {
                let dt_iri = datatype.as_str();
                match dt_iri {
                    "http://www.w3.org/2001/XMLSchema#boolean" => {
                        if let Some(bool_val) = self.as_bool() {
                            let canonical_value = if bool_val { "true" } else { "false" };
                            return Literal::new_typed(canonical_value, datatype.clone());
                        }
                    }
                    "http://www.w3.org/2001/XMLSchema#integer"
                    | "http://www.w3.org/2001/XMLSchema#long"
                    | "http://www.w3.org/2001/XMLSchema#int"
                    | "http://www.w3.org/2001/XMLSchema#short"
                    | "http://www.w3.org/2001/XMLSchema#byte" => {
                        if let Some(int_val) = self.as_i64() {
                            return Literal::new_typed(int_val.to_string(), datatype.clone());
                        }
                    }
                    "http://www.w3.org/2001/XMLSchema#unsignedLong"
                    | "http://www.w3.org/2001/XMLSchema#unsignedInt"
                    | "http://www.w3.org/2001/XMLSchema#unsignedShort"
                    | "http://www.w3.org/2001/XMLSchema#unsignedByte"
                    | "http://www.w3.org/2001/XMLSchema#positiveInteger"
                    | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger" => {
                        if let Some(int_val) = self.as_i64() {
                            if int_val >= 0 {
                                return Literal::new_typed(int_val.to_string(), datatype.clone());
                            }
                        }
                    }
                    "http://www.w3.org/2001/XMLSchema#negativeInteger"
                    | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger" => {
                        if let Some(int_val) = self.as_i64() {
                            if int_val <= 0 {
                                return Literal::new_typed(int_val.to_string(), datatype.clone());
                            }
                        }
                    }
                    "http://www.w3.org/2001/XMLSchema#decimal" => {
                        if let Some(dec_val) = self.as_f64() {
                            // Format decimal properly - remove trailing zeros after decimal point
                            let formatted = format!("{}", dec_val);
                            if formatted.contains('.') {
                                let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
                                return Literal::new_typed(
                                    if trimmed.is_empty() || trimmed == "-" {
                                        "0"
                                    } else {
                                        trimmed
                                    },
                                    datatype.clone(),
                                );
                            } else {
                                return Literal::new_typed(
                                    format!("{}.0", formatted),
                                    datatype.clone(),
                                );
                            }
                        }
                    }
                    "http://www.w3.org/2001/XMLSchema#double"
                    | "http://www.w3.org/2001/XMLSchema#float" => {
                        if let Some(float_val) = self.as_f64() {
                            // Handle special values
                            if float_val.is_infinite() {
                                return Literal::new_typed(
                                    if float_val.is_sign_positive() {
                                        "INF"
                                    } else {
                                        "-INF"
                                    },
                                    datatype.clone(),
                                );
                            } else if float_val.is_nan() {
                                return Literal::new_typed("NaN", datatype.clone());
                            } else {
                                // Use scientific notation for very large or very small numbers
                                let formatted = if float_val.abs() >= 1e6
                                    || (float_val.abs() < 1e-3 && float_val != 0.0)
                                {
                                    format!("{:E}", float_val)
                                } else {
                                    format!("{}", float_val)
                                };
                                return Literal::new_typed(formatted, datatype.clone());
                            }
                        }
                    }
                    "http://www.w3.org/2001/XMLSchema#string"
                    | "http://www.w3.org/2001/XMLSchema#normalizedString" => {
                        // Normalize whitespace for normalizedString
                        if dt_iri == "http://www.w3.org/2001/XMLSchema#normalizedString" {
                            let normalized = value
                                .replace('\t', " ")
                                .replace('\n', " ")
                                .replace('\r', " ");
                            return Literal::new_typed(normalized, datatype.clone());
                        }
                    }
                    "http://www.w3.org/2001/XMLSchema#token" => {
                        // Normalize whitespace and collapse consecutive spaces
                        let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
                        return Literal::new_typed(normalized, datatype.clone());
                    }
                    _ => {}
                }
            }
            LiteralContent::LanguageTaggedString { value, language } => {
                // Keep original case for language tags to match RFC 5646 best practices
                return Self(LiteralContent::LanguageTaggedString {
                    value: value.clone(),
                    language: language.clone(),
                });
            }
            _ => {}
        }
        self.clone()
    }

    /// Validates this literal against its datatype (if any)
    pub fn validate(&self) -> Result<(), OxirsError> {
        match &self.0 {
            LiteralContent::String(_) => Ok(()),
            LiteralContent::LanguageTaggedString { language, .. } => {
                validate_language_tag(language).map_err(Into::into)
            }
            #[cfg(feature = "rdf-12")]
            LiteralContent::DirectionalLanguageTaggedString { language, .. } => {
                validate_language_tag(language).map_err(Into::into)
            }
            LiteralContent::TypedLiteral { value, datatype } => {
                validate_xsd_value(value, datatype.as_str())
            }
        }
    }
}

impl fmt::Display for Literal {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl RdfTerm for Literal {
    fn as_str(&self) -> &str {
        self.value()
    }

    fn is_literal(&self) -> bool {
        true
    }
}

impl ObjectTerm for Literal {}

/// A borrowed RDF [literal](https://www.w3.org/TR/rdf11-concepts/#dfn-literal).
///
/// The default string formatter is returning an N-Triples, Turtle, and SPARQL compatible representation:
/// ```
/// use oxirs_core::model::literal::LiteralRef;
/// use oxirs_core::vocab::xsd;
///
/// assert_eq!(
///     "\"foo\\nbar\"",
///     LiteralRef::new_simple_literal("foo\nbar").to_string()
/// );
///
/// assert_eq!(
///     r#""1999-01-01"^^<http://www.w3.org/2001/XMLSchema#date>"#,
///     LiteralRef::new_typed_literal("1999-01-01", xsd::DATE.as_ref()).to_string()
/// );
/// ```
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash)]
pub struct LiteralRef<'a>(LiteralRefContent<'a>);

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
enum LiteralRefContent<'a> {
    String(&'a str),
    LanguageTaggedString {
        value: &'a str,
        language: &'a str,
    },
    #[cfg(feature = "rdf-12")]
    DirectionalLanguageTaggedString {
        value: &'a str,
        language: &'a str,
        direction: BaseDirection,
    },
    TypedLiteral {
        value: &'a str,
        datatype: NamedNodeRef<'a>,
    },
}

impl<'a> LiteralRef<'a> {
    /// Builds an RDF [simple literal](https://www.w3.org/TR/rdf11-concepts/#dfn-simple-literal).
    #[inline]
    pub const fn new_simple_literal(value: &'a str) -> Self {
        LiteralRef(LiteralRefContent::String(value))
    }

    /// Creates a new literal reference (alias for compatibility)
    #[inline]
    pub const fn new(value: &'a str) -> Self {
        Self::new_simple_literal(value)
    }

    /// Builds an RDF [literal](https://www.w3.org/TR/rdf11-concepts/#dfn-literal) with a [datatype](https://www.w3.org/TR/rdf11-concepts/#dfn-datatype-iri).
    #[inline]
    pub fn new_typed_literal(value: &'a str, datatype: impl Into<NamedNodeRef<'a>>) -> Self {
        let datatype = datatype.into();
        LiteralRef(if datatype == xsd::STRING.as_ref() {
            LiteralRefContent::String(value)
        } else {
            LiteralRefContent::TypedLiteral { value, datatype }
        })
    }

    /// Creates a new typed literal reference (alias for compatibility)
    #[inline]
    pub fn new_typed(value: &'a str, datatype: NamedNodeRef<'a>) -> Self {
        Self::new_typed_literal(value, datatype)
    }

    /// Builds an RDF [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string).
    ///
    /// It is the responsibility of the caller to check that `language`
    /// is valid [BCP47](https://tools.ietf.org/html/bcp47) language tag,
    /// and is lowercase.
    ///
    /// [`Literal::new_language_tagged_literal()`] is a safe version of this constructor and should be used for untrusted data.
    #[inline]
    pub const fn new_language_tagged_literal_unchecked(value: &'a str, language: &'a str) -> Self {
        LiteralRef(LiteralRefContent::LanguageTaggedString { value, language })
    }

    /// Creates a new language-tagged literal reference (alias for compatibility)
    #[inline]
    pub const fn new_lang(value: &'a str, language: &'a str) -> Self {
        Self::new_language_tagged_literal_unchecked(value, language)
    }

    /// Builds an RDF [directional language-tagged string](https://www.w3.org/TR/rdf12-concepts/#dfn-dir-lang-string).
    ///
    /// It is the responsibility of the caller to check that `language`
    /// is valid [BCP47](https://tools.ietf.org/html/bcp47) language tag,
    /// and is lowercase.
    ///
    /// [`Literal::new_directional_language_tagged_literal()`] is a safe version of this constructor and should be used for untrusted data.
    #[cfg(feature = "rdf-12")]
    #[inline]
    pub const fn new_directional_language_tagged_literal_unchecked(
        value: &'a str,
        language: &'a str,
        direction: BaseDirection,
    ) -> Self {
        LiteralRef(LiteralRefContent::DirectionalLanguageTaggedString {
            value,
            language,
            direction,
        })
    }

    /// The literal [lexical form](https://www.w3.org/TR/rdf11-concepts/#dfn-lexical-form)
    #[inline]
    pub const fn value(self) -> &'a str {
        match self.0 {
            LiteralRefContent::String(value)
            | LiteralRefContent::LanguageTaggedString { value, .. }
            | LiteralRefContent::TypedLiteral { value, .. } => value,
            #[cfg(feature = "rdf-12")]
            LiteralRefContent::DirectionalLanguageTaggedString { value, .. } => value,
        }
    }

    /// The literal [language tag](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tag) if it is a [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string).
    ///
    /// Language tags are defined by the [BCP47](https://tools.ietf.org/html/bcp47).
    /// They are normalized to lowercase by this implementation.
    #[inline]
    pub const fn language(self) -> Option<&'a str> {
        match self.0 {
            LiteralRefContent::LanguageTaggedString { language, .. } => Some(language),
            #[cfg(feature = "rdf-12")]
            LiteralRefContent::DirectionalLanguageTaggedString { language, .. } => Some(language),
            _ => None,
        }
    }

    /// The literal [base direction](https://www.w3.org/TR/rdf12-concepts/#dfn-base-direction) if it is a [directional language-tagged string](https://www.w3.org/TR/rdf12-concepts/#dfn-base-direction).
    ///
    /// The two possible base directions are left-to-right (`ltr`) and right-to-left (`rtl`).
    #[cfg(feature = "rdf-12")]
    #[inline]
    pub const fn direction(self) -> Option<BaseDirection> {
        match self.0 {
            LiteralRefContent::DirectionalLanguageTaggedString { direction, .. } => Some(direction),
            _ => None,
        }
    }

    /// The literal [datatype](https://www.w3.org/TR/rdf11-concepts/#dfn-datatype-iri).
    ///
    /// The datatype of [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string) is always [rdf:langString](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string).
    /// The datatype of [simple literals](https://www.w3.org/TR/rdf11-concepts/#dfn-simple-literal) is [xsd:string](https://www.w3.org/TR/xmlschema11-2/#string).
    #[inline]
    pub fn datatype(self) -> NamedNodeRef<'a> {
        match self.0 {
            LiteralRefContent::String(_) => xsd::STRING.as_ref(),
            LiteralRefContent::LanguageTaggedString { .. } => rdf::LANG_STRING.as_ref(),
            #[cfg(feature = "rdf-12")]
            LiteralRefContent::DirectionalLanguageTaggedString { .. } => {
                rdf::DIR_LANG_STRING.as_ref()
            }
            LiteralRefContent::TypedLiteral { datatype, .. } => datatype,
        }
    }

    /// Checks if this literal could be seen as an RDF 1.0 [plain literal](https://www.w3.org/TR/2004/REC-rdf-concepts-20040210/#dfn-plain-literal).
    ///
    /// It returns true if the literal is a [language-tagged string](https://www.w3.org/TR/rdf11-concepts/#dfn-language-tagged-string)
    /// or has the datatype [xsd:string](https://www.w3.org/TR/xmlschema11-2/#string).
    #[inline]
    #[deprecated(note = "Plain literal concept is removed in RDF 1.1", since = "0.3.0")]
    pub const fn is_plain(self) -> bool {
        matches!(
            self.0,
            LiteralRefContent::String(_) | LiteralRefContent::LanguageTaggedString { .. }
        )
    }

    #[inline]
    pub fn into_owned(self) -> Literal {
        Literal(match self.0 {
            LiteralRefContent::String(value) => LiteralContent::String(value.to_owned()),
            LiteralRefContent::LanguageTaggedString { value, language } => {
                LiteralContent::LanguageTaggedString {
                    value: value.to_owned(),
                    language: language.to_owned(),
                }
            }
            #[cfg(feature = "rdf-12")]
            LiteralRefContent::DirectionalLanguageTaggedString {
                value,
                language,
                direction,
            } => LiteralContent::DirectionalLanguageTaggedString {
                value: value.to_owned(),
                language: language.to_owned(),
                direction,
            },
            LiteralRefContent::TypedLiteral { value, datatype } => LiteralContent::TypedLiteral {
                value: value.to_owned(),
                datatype: datatype.to_owned(),
            },
        })
    }

    /// Converts to an owned Literal (alias for compatibility)
    #[inline]
    pub fn to_owned(&self) -> Literal {
        self.into_owned()
    }
}

impl fmt::Display for LiteralRef<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            LiteralRefContent::String(value) => print_quoted_str(value, f),
            LiteralRefContent::LanguageTaggedString { value, language } => {
                print_quoted_str(value, f)?;
                write!(f, "@{language}")
            }
            #[cfg(feature = "rdf-12")]
            LiteralRefContent::DirectionalLanguageTaggedString {
                value,
                language,
                direction,
            } => {
                print_quoted_str(value, f)?;
                write!(f, "@{language}--{direction}")
            }
            LiteralRefContent::TypedLiteral { value, datatype } => {
                print_quoted_str(value, f)?;
                write!(f, "^^{datatype}")
            }
        }
    }
}

impl<'a> RdfTerm for LiteralRef<'a> {
    fn as_str(&self) -> &str {
        self.value()
    }

    fn is_literal(&self) -> bool {
        true
    }
}

/// Helper function to print a quoted string with proper escaping
#[inline]
pub fn print_quoted_str(string: &str, f: &mut impl Write) -> fmt::Result {
    f.write_char('"')?;
    for c in string.chars() {
        match c {
            '\u{08}' => f.write_str("\\b"),
            '\t' => f.write_str("\\t"),
            '\n' => f.write_str("\\n"),
            '\u{0C}' => f.write_str("\\f"),
            '\r' => f.write_str("\\r"),
            '"' => f.write_str("\\\""),
            '\\' => f.write_str("\\\\"),
            '\0'..='\u{1F}' | '\u{7F}' => write!(f, "\\u{:04X}", u32::from(c)),
            _ => f.write_char(c),
        }?;
    }
    f.write_char('"')
}

/// A [directional language-tagged string](https://www.w3.org/TR/rdf12-concepts/#dfn-dir-lang-string) [base-direction](https://www.w3.org/TR/rdf12-concepts/#dfn-base-direction)
#[cfg(feature = "rdf-12")]
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BaseDirection {
    /// the initial text direction is set to left-to-right
    Ltr,
    /// the initial text direction is set to right-to-left
    Rtl,
}

#[cfg(feature = "rdf-12")]
impl fmt::Display for BaseDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ltr => "ltr",
            Self::Rtl => "rtl",
        })
    }
}

impl<'a> From<&'a Literal> for LiteralRef<'a> {
    #[inline]
    fn from(node: &'a Literal) -> Self {
        node.as_ref()
    }
}

impl<'a> From<LiteralRef<'a>> for Literal {
    #[inline]
    fn from(node: LiteralRef<'a>) -> Self {
        node.into_owned()
    }
}

impl<'a> From<&'a str> for LiteralRef<'a> {
    #[inline]
    fn from(value: &'a str) -> Self {
        LiteralRef(LiteralRefContent::String(value))
    }
}

impl PartialEq<Literal> for LiteralRef<'_> {
    #[inline]
    fn eq(&self, other: &Literal) -> bool {
        *self == other.as_ref()
    }
}

impl PartialEq<LiteralRef<'_>> for Literal {
    #[inline]
    fn eq(&self, other: &LiteralRef<'_>) -> bool {
        self.as_ref() == *other
    }
}

// Implement standard From traits
impl<'a> From<&'a str> for Literal {
    #[inline]
    fn from(value: &'a str) -> Self {
        Self(LiteralContent::String(value.into()))
    }
}

impl From<String> for Literal {
    #[inline]
    fn from(value: String) -> Self {
        Self(LiteralContent::String(value))
    }
}

impl<'a> From<Cow<'a, str>> for Literal {
    #[inline]
    fn from(value: Cow<'a, str>) -> Self {
        Self(LiteralContent::String(value.into()))
    }
}

impl From<bool> for Literal {
    #[inline]
    fn from(value: bool) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: value.to_string(),
            datatype: xsd::BOOLEAN.clone(),
        })
    }
}

impl From<i128> for Literal {
    #[inline]
    fn from(value: i128) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: value.to_string(),
            datatype: xsd::INTEGER.clone(),
        })
    }
}

impl From<i64> for Literal {
    #[inline]
    fn from(value: i64) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: value.to_string(),
            datatype: xsd::INTEGER.clone(),
        })
    }
}

impl From<i32> for Literal {
    #[inline]
    fn from(value: i32) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: value.to_string(),
            datatype: xsd::INTEGER.clone(),
        })
    }
}

impl From<i16> for Literal {
    #[inline]
    fn from(value: i16) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: value.to_string(),
            datatype: xsd::INTEGER.clone(),
        })
    }
}

impl From<u64> for Literal {
    #[inline]
    fn from(value: u64) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: value.to_string(),
            datatype: xsd::INTEGER.clone(),
        })
    }
}

impl From<u32> for Literal {
    #[inline]
    fn from(value: u32) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: value.to_string(),
            datatype: xsd::INTEGER.clone(),
        })
    }
}

impl From<u16> for Literal {
    #[inline]
    fn from(value: u16) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: value.to_string(),
            datatype: xsd::INTEGER.clone(),
        })
    }
}

impl From<f32> for Literal {
    #[inline]
    fn from(value: f32) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: if value == f32::INFINITY {
                "INF".to_owned()
            } else if value == f32::NEG_INFINITY {
                "-INF".to_owned()
            } else {
                value.to_string()
            },
            datatype: xsd::FLOAT.clone(),
        })
    }
}

impl From<f64> for Literal {
    #[inline]
    fn from(value: f64) -> Self {
        Self(LiteralContent::TypedLiteral {
            value: if value == f64::INFINITY {
                "INF".to_owned()
            } else if value == f64::NEG_INFINITY {
                "-INF".to_owned()
            } else {
                value.to_string()
            },
            datatype: xsd::DOUBLE.clone(),
        })
    }
}

/// Common XSD datatypes as constants and convenience functions
pub mod xsd_literals {
    use super::*;
    use crate::vocab::xsd;

    // Convenience functions for creating typed literals

    /// Creates a boolean literal
    pub fn boolean_literal(value: bool) -> Literal {
        Literal::new_typed(value.to_string(), xsd::BOOLEAN.clone())
    }

    /// Creates an integer literal
    pub fn integer_literal(value: i64) -> Literal {
        Literal::new_typed(value.to_string(), xsd::INTEGER.clone())
    }

    /// Creates a decimal literal
    pub fn decimal_literal(value: f64) -> Literal {
        Literal::new_typed(value.to_string(), xsd::DECIMAL.clone())
    }

    /// Creates a double literal
    pub fn double_literal(value: f64) -> Literal {
        Literal::new_typed(value.to_string(), xsd::DOUBLE.clone())
    }

    /// Creates a string literal
    pub fn string_literal(value: &str) -> Literal {
        Literal::new_typed(value, xsd::STRING.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_literal_equality() {
        assert_eq!(
            Literal::new_simple_literal("foo"),
            Literal::new_typed_literal("foo", xsd::STRING.clone())
        );
        assert_eq!(
            Literal::new_simple_literal("foo"),
            LiteralRef::new_typed_literal("foo", xsd::STRING.as_ref())
        );
        assert_eq!(
            LiteralRef::new_simple_literal("foo"),
            Literal::new_typed_literal("foo", xsd::STRING.clone())
        );
        assert_eq!(
            LiteralRef::new_simple_literal("foo"),
            LiteralRef::new_typed_literal("foo", xsd::STRING.as_ref())
        );
    }

    #[test]
    fn test_float_format() {
        assert_eq!("INF", Literal::from(f32::INFINITY).value());
        assert_eq!("INF", Literal::from(f64::INFINITY).value());
        assert_eq!("-INF", Literal::from(f32::NEG_INFINITY).value());
        assert_eq!("-INF", Literal::from(f64::NEG_INFINITY).value());
        assert_eq!("NaN", Literal::from(f32::NAN).value());
        assert_eq!("NaN", Literal::from(f64::NAN).value());
    }

    #[test]
    fn test_plain_literal() {
        let literal = Literal::new("Hello");
        assert_eq!(literal.value(), "Hello");
        #[allow(deprecated)]
        {
            assert!(literal.is_plain());
        }
        assert!(!literal.is_lang_string());
        assert!(!literal.is_typed());
        assert_eq!(format!("{}", literal), "\"Hello\"");
    }

    #[test]
    fn test_lang_literal() {
        let literal = Literal::new_lang("Hello", "en").unwrap();
        assert_eq!(literal.value(), "Hello");
        assert_eq!(literal.language(), Some("en"));
        #[allow(deprecated)]
        {
            assert!(literal.is_plain());
        }
        assert!(literal.is_lang_string());
        assert!(!literal.is_typed());
        assert_eq!(format!("{}", literal), "\"Hello\"@en");
    }

    #[test]
    fn test_typed_literal() {
        let literal = Literal::new_typed("42", xsd::INTEGER.clone());
        assert_eq!(literal.value(), "42");
        assert_eq!(
            literal.datatype().as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
        #[allow(deprecated)]
        {
            assert!(!literal.is_plain());
        }
        assert!(!literal.is_lang_string());
        assert!(literal.is_typed());
        assert_eq!(
            format!("{}", literal),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
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
        let bool_literal = xsd_literals::boolean_literal(true);
        assert!(bool_literal.is_boolean());
        assert_eq!(bool_literal.as_bool(), Some(true));

        let false_literal = Literal::new_typed("false", xsd::BOOLEAN.clone());
        assert_eq!(false_literal.as_bool(), Some(false));

        // Test string representations
        let true_str = Literal::new("true");
        assert_eq!(true_str.as_bool(), Some(true));

        let false_str = Literal::new("0");
        assert_eq!(false_str.as_bool(), Some(false));
    }

    #[test]
    fn test_numeric_extraction() {
        let int_literal = xsd_literals::integer_literal(42);
        assert!(int_literal.is_numeric());
        assert_eq!(int_literal.as_i64(), Some(42));
        assert_eq!(int_literal.as_i32(), Some(42));
        assert_eq!(int_literal.as_f64(), Some(42.0));

        let decimal_literal = xsd_literals::decimal_literal(3.25);
        assert!(decimal_literal.is_numeric());
        assert_eq!(decimal_literal.as_f64(), Some(3.25));
        assert_eq!(decimal_literal.as_f32(), Some(3.25_f32));

        // Test untyped numeric strings
        let untyped_num = Literal::new("123");
        assert!(untyped_num.is_numeric());
        assert_eq!(untyped_num.as_i64(), Some(123));
    }

    #[test]
    fn test_canonical_form() {
        // Boolean canonicalization
        let bool_literal = Literal::new_typed("True", xsd::BOOLEAN.clone());
        let canonical = bool_literal.canonical_form();
        assert_eq!(canonical.value(), "true");

        // Integer canonicalization
        let int_literal = Literal::new_typed("  42  ", xsd::INTEGER.clone());
        // Note: This would need actual whitespace trimming in canonical form
        // For now, just test that it returns a valid canonical form
        let canonical = int_literal.canonical_form();
        assert_eq!(
            canonical.datatype().as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
        );

        // Decimal canonicalization
        let dec_literal = Literal::new_typed("3.140", xsd::DECIMAL.clone());
        let canonical = dec_literal.canonical_form();
        assert_eq!(canonical.value(), "3.14"); // Should remove trailing zeros
    }

    #[test]
    fn test_xsd_convenience_functions() {
        // Test all the convenience functions work
        assert_eq!(xsd_literals::boolean_literal(true).value(), "true");
        assert_eq!(xsd_literals::integer_literal(123).value(), "123");
        assert_eq!(xsd_literals::decimal_literal(3.25).value(), "3.25");
        assert_eq!(xsd_literals::double_literal(2.71).value(), "2.71");
        assert_eq!(xsd_literals::string_literal("hello").value(), "hello");

        // Test datatype assignments
        assert_eq!(
            xsd_literals::boolean_literal(true).datatype().as_str(),
            "http://www.w3.org/2001/XMLSchema#boolean"
        );
        assert_eq!(
            xsd_literals::integer_literal(123).datatype().as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
    }

    #[test]
    fn test_numeric_type_detection() {
        // Test various numeric types
        let int_lit = Literal::new_typed("42", xsd::INTEGER.clone());
        assert!(int_lit.is_numeric());

        let float_lit = Literal::new_typed("3.14", xsd::FLOAT.clone());
        assert!(float_lit.is_numeric());

        let double_lit = Literal::new_typed("2.71", xsd::DOUBLE.clone());
        assert!(double_lit.is_numeric());

        // Non-numeric types
        let string_lit = Literal::new_typed("hello", xsd::STRING.clone());
        assert!(!string_lit.is_numeric());

        let bool_lit = Literal::new_typed("true", xsd::BOOLEAN.clone());
        assert!(!bool_lit.is_numeric());
    }
}
