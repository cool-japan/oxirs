//! RDF Literal implementation
//!
//! This implementation is extracted and adapted from Oxigraph's oxrdf literal handling
//! to provide zero-dependency RDF literal support with full XSD datatype validation.

use crate::model::{NamedNode, NamedNodeRef, ObjectTerm, RdfTerm};
use crate::vocab::{rdf, xsd};
use crate::OxirsError;
use oxilangtag::LanguageTag as OxiLanguageTag;
use oxsdatatypes::{Boolean, Date, DateTime, Decimal, Double, Float, Integer, Time};
use std::borrow::Cow;
use std::fmt::{self, Write};
use std::hash::Hash;
use std::str::FromStr;

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

/// Validates a language tag according to BCP 47 (RFC 5646) using oxilangtag
fn validate_language_tag(tag: &str) -> Result<(), LanguageTagParseError> {
    OxiLanguageTag::parse(tag)
        .map(|_| ())
        .map_err(|e| LanguageTagParseError {
            message: format!("Invalid language tag '{tag}': {e}"),
        })
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

        // Boolean type - use oxsdatatypes Boolean parsing
        "http://www.w3.org/2001/XMLSchema#boolean" => Boolean::from_str(value)
            .map(|_| ())
            .map_err(|e| OxirsError::Parse(format!("Invalid boolean value '{value}': {e}"))),

        // Integer types - use oxsdatatypes Integer parsing with range validation
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
        | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger" => Integer::from_str(value)
            .map_err(|e| OxirsError::Parse(format!("Invalid integer value '{value}': {e}")))
            .and_then(|integer| validate_integer_range_oxs(integer, datatype_iri)),

        // Decimal type - use oxsdatatypes Decimal parsing
        "http://www.w3.org/2001/XMLSchema#decimal" => Decimal::from_str(value)
            .map(|_| ())
            .map_err(|e| OxirsError::Parse(format!("Invalid decimal value '{value}': {e}"))),

        // Floating point types - use oxsdatatypes Float/Double parsing
        "http://www.w3.org/2001/XMLSchema#float" => Float::from_str(value)
            .map(|_| ())
            .map_err(|e| OxirsError::Parse(format!("Invalid float value '{value}': {e}"))),
        "http://www.w3.org/2001/XMLSchema#double" => Double::from_str(value)
            .map(|_| ())
            .map_err(|e| OxirsError::Parse(format!("Invalid double value '{value}': {e}"))),

        // Date/time types - use oxsdatatypes parsing
        "http://www.w3.org/2001/XMLSchema#dateTime" => DateTime::from_str(value)
            .map(|_| ())
            .map_err(|e| OxirsError::Parse(format!("Invalid dateTime value '{value}': {e}"))),

        "http://www.w3.org/2001/XMLSchema#date" => Date::from_str(value)
            .map(|_| ())
            .map_err(|e| OxirsError::Parse(format!("Invalid date value '{value}': {e}"))),

        "http://www.w3.org/2001/XMLSchema#time" => Time::from_str(value)
            .map(|_| ())
            .map_err(|e| OxirsError::Parse(format!("Invalid time value '{value}': {e}"))),

        // For unknown datatypes, don't validate
        _ => Ok(()),
    }
}

/// Validates integer values against their specific type ranges
#[allow(dead_code)]
fn validate_integer_range(value: &str, datatype_iri: &str) -> Result<(), OxirsError> {
    let parsed_value: i64 = value
        .parse()
        .map_err(|_| OxirsError::Parse(format!("Cannot parse integer: '{value}'")))?;

    match datatype_iri {
        "http://www.w3.org/2001/XMLSchema#byte" if !(-128..=127).contains(&parsed_value) => {
            return Err(OxirsError::Parse(format!(
                "Byte value out of range: {parsed_value}. Must be between -128 and 127"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#short" if !(-32768..=32767).contains(&parsed_value) => {
            return Err(OxirsError::Parse(format!(
                "Short value out of range: {parsed_value}. Must be between -32768 and 32767"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#int"
            if !(-2147483648..=2147483647).contains(&parsed_value) =>
        {
            return Err(OxirsError::Parse(format!(
                    "Int value out of range: {parsed_value}. Must be between -2147483648 and 2147483647"
                )));
        }
        "http://www.w3.org/2001/XMLSchema#unsignedByte" if !(0..=255).contains(&parsed_value) => {
            return Err(OxirsError::Parse(format!(
                "Unsigned byte value out of range: {parsed_value}. Must be between 0 and 255"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#unsignedShort"
            if !(0..=65535).contains(&parsed_value) =>
        {
            return Err(OxirsError::Parse(format!(
                "Unsigned short value out of range: {parsed_value}. Must be between 0 and 65535"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#unsignedInt"
            if !(0..=4294967295).contains(&parsed_value) =>
        {
            return Err(OxirsError::Parse(format!(
                "Unsigned int value out of range: {parsed_value}. Must be between 0 and 4294967295"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#positiveInteger" if parsed_value <= 0 => {
            return Err(OxirsError::Parse(format!(
                "Positive integer must be greater than 0, got: {parsed_value}"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#nonNegativeInteger" if parsed_value < 0 => {
            return Err(OxirsError::Parse(format!(
                "Non-negative integer must be >= 0, got: {parsed_value}"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#negativeInteger" if parsed_value >= 0 => {
            return Err(OxirsError::Parse(format!(
                "Negative integer must be less than 0, got: {parsed_value}"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#nonPositiveInteger" if parsed_value > 0 => {
            return Err(OxirsError::Parse(format!(
                "Non-positive integer must be <= 0, got: {parsed_value}"
            )));
        }
        _ => {} // Other integer types don't have additional range restrictions in this simplified implementation
    }

    Ok(())
}

/// Validates integer values against their specific type ranges using oxsdatatypes Integer
fn validate_integer_range_oxs(integer: Integer, datatype_iri: &str) -> Result<(), OxirsError> {
    // Convert oxsdatatypes Integer to i64 for range checking
    let parsed_value: i64 = integer.to_string().parse().map_err(|_| {
        OxirsError::Parse("Cannot convert integer to i64 for range validation".to_string())
    })?;

    match datatype_iri {
        "http://www.w3.org/2001/XMLSchema#byte" if !(-128..=127).contains(&parsed_value) => {
            return Err(OxirsError::Parse(format!(
                "Byte value out of range: {parsed_value}. Must be between -128 and 127"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#short" if !(-32768..=32767).contains(&parsed_value) => {
            return Err(OxirsError::Parse(format!(
                "Short value out of range: {parsed_value}. Must be between -32768 and 32767"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#int"
            if !(-2147483648..=2147483647).contains(&parsed_value) =>
        {
            return Err(OxirsError::Parse(format!(
                    "Int value out of range: {parsed_value}. Must be between -2147483648 and 2147483647"
                )));
        }
        "http://www.w3.org/2001/XMLSchema#unsignedByte" if !(0..=255).contains(&parsed_value) => {
            return Err(OxirsError::Parse(format!(
                "Unsigned byte value out of range: {parsed_value}. Must be between 0 and 255"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#unsignedShort"
            if !(0..=65535).contains(&parsed_value) =>
        {
            return Err(OxirsError::Parse(format!(
                "Unsigned short value out of range: {parsed_value}. Must be between 0 and 65535"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#unsignedInt"
            if !(0..=4294967295).contains(&parsed_value) =>
        {
            return Err(OxirsError::Parse(format!(
                "Unsigned int value out of range: {parsed_value}. Must be between 0 and 4294967295"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#positiveInteger" if parsed_value <= 0 => {
            return Err(OxirsError::Parse(format!(
                "Positive integer must be greater than 0, got: {parsed_value}"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#nonNegativeInteger" if parsed_value < 0 => {
            return Err(OxirsError::Parse(format!(
                "Non-negative integer must be >= 0, got: {parsed_value}"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#negativeInteger" if parsed_value >= 0 => {
            return Err(OxirsError::Parse(format!(
                "Negative integer must be less than 0, got: {parsed_value}"
            )));
        }
        "http://www.w3.org/2001/XMLSchema#nonPositiveInteger" if parsed_value > 0 => {
            return Err(OxirsError::Parse(format!(
                "Non-positive integer must be <= 0, got: {parsed_value}"
            )));
        }
        _ => {} // Other integer types don't have additional range restrictions
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
///     Literal::new_language_tagged_literal("foo", "en").expect("valid language literal").to_string()
/// );
/// ```
#[derive(Eq, PartialEq, Debug, Clone, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Literal(LiteralContent);

#[derive(Debug, Clone)]
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

/// Ordinal used to order/compare `LiteralContent` variants (mirrors the
/// enum's declaration order, matching what `#[derive(PartialOrd, Ord)]`
/// would have produced).
fn literal_content_variant_rank(content: &LiteralContent) -> u8 {
    match content {
        LiteralContent::String(_) => 0,
        LiteralContent::LanguageTaggedString { .. } => 1,
        #[cfg(feature = "rdf-12")]
        LiteralContent::DirectionalLanguageTaggedString { .. } => 2,
        LiteralContent::TypedLiteral { .. } => 3,
    }
}

// `LiteralContent`'s `PartialEq`/`Eq`/`Hash`/`PartialOrd`/`Ord` are hand-written
// rather than derived so that a language tag on `LanguageTaggedString`/
// `DirectionalLanguageTaggedString` compares and hashes *case-insensitively*,
// per RDF 1.1 (language tags are compared case-insensitively -- `"foo"@en-US`
// and `"foo"@en-us` denote the same literal) -- while the *stored* lexical
// form still preserves whatever case the caller originally supplied (see
// `Literal::new_language_tagged_literal`, which used to destructively
// lowercase the tag before storing it). A naive `#[derive]` here would make
// two RDF-equal literals compare unequal whenever their tags differ only in
// case.
impl PartialEq for LiteralContent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LiteralContent::String(a), LiteralContent::String(b)) => a == b,
            (
                LiteralContent::LanguageTaggedString {
                    value: v1,
                    language: l1,
                },
                LiteralContent::LanguageTaggedString {
                    value: v2,
                    language: l2,
                },
            ) => v1 == v2 && l1.eq_ignore_ascii_case(l2),
            #[cfg(feature = "rdf-12")]
            (
                LiteralContent::DirectionalLanguageTaggedString {
                    value: v1,
                    language: l1,
                    direction: d1,
                },
                LiteralContent::DirectionalLanguageTaggedString {
                    value: v2,
                    language: l2,
                    direction: d2,
                },
            ) => v1 == v2 && l1.eq_ignore_ascii_case(l2) && d1 == d2,
            (
                LiteralContent::TypedLiteral {
                    value: v1,
                    datatype: d1,
                },
                LiteralContent::TypedLiteral {
                    value: v2,
                    datatype: d2,
                },
            ) => v1 == v2 && d1 == d2,
            _ => false,
        }
    }
}

impl Eq for LiteralContent {}

impl std::hash::Hash for LiteralContent {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        literal_content_variant_rank(self).hash(state);
        match self {
            LiteralContent::String(value) => value.hash(state),
            LiteralContent::LanguageTaggedString { value, language } => {
                value.hash(state);
                // Hash the case-folded tag so tags that are `eq_ignore_ascii_case`
                // (and thus `PartialEq`-equal above) always hash equal.
                for b in language.bytes() {
                    b.to_ascii_lowercase().hash(state);
                }
            }
            #[cfg(feature = "rdf-12")]
            LiteralContent::DirectionalLanguageTaggedString {
                value,
                language,
                direction,
            } => {
                value.hash(state);
                for b in language.bytes() {
                    b.to_ascii_lowercase().hash(state);
                }
                direction.hash(state);
            }
            LiteralContent::TypedLiteral { value, datatype } => {
                value.hash(state);
                datatype.hash(state);
            }
        }
    }
}

impl PartialOrd for LiteralContent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LiteralContent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (LiteralContent::String(a), LiteralContent::String(b)) => a.cmp(b),
            (
                LiteralContent::LanguageTaggedString {
                    value: v1,
                    language: l1,
                },
                LiteralContent::LanguageTaggedString {
                    value: v2,
                    language: l2,
                },
            ) => v1
                .cmp(v2)
                .then_with(|| l1.to_ascii_lowercase().cmp(&l2.to_ascii_lowercase())),
            #[cfg(feature = "rdf-12")]
            (
                LiteralContent::DirectionalLanguageTaggedString {
                    value: v1,
                    language: l1,
                    direction: d1,
                },
                LiteralContent::DirectionalLanguageTaggedString {
                    value: v2,
                    language: l2,
                    direction: d2,
                },
            ) => v1
                .cmp(v2)
                .then_with(|| l1.to_ascii_lowercase().cmp(&l2.to_ascii_lowercase()))
                .then_with(|| d1.cmp(d2)),
            (
                LiteralContent::TypedLiteral {
                    value: v1,
                    datatype: d1,
                },
                LiteralContent::TypedLiteral {
                    value: v2,
                    datatype: d2,
                },
            ) => v1.cmp(v2).then_with(|| d1.cmp(d2)),
            _ => literal_content_variant_rank(self).cmp(&literal_content_variant_rank(other)),
        }
    }
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
        // RDF 1.1/1.2 language tags are compared case-insensitively, but the
        // *lexical form* must be preserved as authored (e.g. SPARQL `LANG()`
        // returns the tag exactly as written, and a parse -> serialize round
        // trip must not mutate the term). Validate the tag without mutating
        // it; `LiteralContent`'s `PartialEq`/`Eq`/`Hash`/`Ord` fold case for
        // language tags so equality/lookup semantics stay RDF-1.1-correct
        // even though the stored string keeps its original case.
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
        // See `new_language_tagged_literal`: preserve the tag's original
        // case for round-tripping; case-insensitive comparison is handled by
        // `LiteralContent`'s `PartialEq`/`Eq`/`Hash`/`Ord` impls.
        let language = language.into();
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
                            let formatted = format!("{dec_val}");
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
                                    format!("{formatted}.0"),
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
                                    format!("{float_val:E}")
                                } else {
                                    format!("{float_val}")
                                };
                                return Literal::new_typed(formatted, datatype.clone());
                            }
                        }
                    }
                    "http://www.w3.org/2001/XMLSchema#normalizedString" => {
                        // Normalize whitespace for normalizedString
                        let normalized = value.replace(['\t', '\n', '\r'], " ");
                        return Literal::new_typed(normalized, datatype.clone());
                    }
                    "http://www.w3.org/2001/XMLSchema#string" => {
                        // No normalization needed for string
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

#[derive(Debug, Clone, Copy)]
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

// Hand-written to match `LiteralContent`'s case-insensitive language-tag
// `PartialEq`/`Eq`/`Hash` (see the comment there): `LiteralRef == Literal`
// comparisons (below) go through this borrowed variant's equality, so it
// must agree with the owned `LiteralContent`'s semantics or the two
// directions of the cross-type `PartialEq` impls would disagree with each
// other.
impl PartialEq for LiteralRefContent<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LiteralRefContent::String(a), LiteralRefContent::String(b)) => a == b,
            (
                LiteralRefContent::LanguageTaggedString {
                    value: v1,
                    language: l1,
                },
                LiteralRefContent::LanguageTaggedString {
                    value: v2,
                    language: l2,
                },
            ) => v1 == v2 && l1.eq_ignore_ascii_case(l2),
            #[cfg(feature = "rdf-12")]
            (
                LiteralRefContent::DirectionalLanguageTaggedString {
                    value: v1,
                    language: l1,
                    direction: d1,
                },
                LiteralRefContent::DirectionalLanguageTaggedString {
                    value: v2,
                    language: l2,
                    direction: d2,
                },
            ) => v1 == v2 && l1.eq_ignore_ascii_case(l2) && d1 == d2,
            (
                LiteralRefContent::TypedLiteral {
                    value: v1,
                    datatype: d1,
                },
                LiteralRefContent::TypedLiteral {
                    value: v2,
                    datatype: d2,
                },
            ) => v1 == v2 && d1 == d2,
            _ => false,
        }
    }
}

impl Eq for LiteralRefContent<'_> {}

impl std::hash::Hash for LiteralRefContent<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            LiteralRefContent::String(value) => {
                0u8.hash(state);
                value.hash(state);
            }
            LiteralRefContent::LanguageTaggedString { value, language } => {
                1u8.hash(state);
                value.hash(state);
                for b in language.bytes() {
                    b.to_ascii_lowercase().hash(state);
                }
            }
            #[cfg(feature = "rdf-12")]
            LiteralRefContent::DirectionalLanguageTaggedString {
                value,
                language,
                direction,
            } => {
                2u8.hash(state);
                value.hash(state);
                for b in language.bytes() {
                    b.to_ascii_lowercase().hash(state);
                }
                direction.hash(state);
            }
            LiteralRefContent::TypedLiteral { value, datatype } => {
                3u8.hash(state);
                value.hash(state);
                datatype.hash(state);
            }
        }
    }
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
                datatype: datatype.into_owned(),
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
        assert_eq!(format!("{literal}"), "\"Hello\"");
    }

    #[test]
    fn test_lang_literal() {
        let literal = Literal::new_lang("Hello", "en").expect("construction should succeed");
        assert_eq!(literal.value(), "Hello");
        assert_eq!(literal.language(), Some("en"));
        #[allow(deprecated)]
        {
            assert!(literal.is_plain());
        }
        assert!(literal.is_lang_string());
        assert!(!literal.is_typed());
        assert_eq!(format!("{literal}"), "\"Hello\"@en");
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
            format!("{literal}"),
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

    /// Regression test: `new_language_tagged_literal` must preserve the
    /// language tag's original case (round-trip fidelity / SPARQL `LANG()`
    /// contract) rather than destructively lowercasing it.
    #[test]
    fn regression_language_tag_preserves_original_case() {
        let literal =
            Literal::new_language_tagged_literal("foo", "en-US").expect("valid language literal");
        assert_eq!(
            literal.language(),
            Some("en-US"),
            "the stored language tag must keep its original case"
        );
        assert_eq!(format!("{literal}"), "\"foo\"@en-US");
    }

    /// Regression test: two language-tagged literals whose tags differ only
    /// in case are still RDF-1.1 equal (and hash equal), even though the
    /// lexical form of the tag is no longer normalized to lowercase at
    /// construction time.
    #[test]
    fn regression_language_tag_case_insensitive_equality_and_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let upper =
            Literal::new_language_tagged_literal("foo", "en-US").expect("valid language literal");
        let lower =
            Literal::new_language_tagged_literal("foo", "en-us").expect("valid language literal");
        let different_value =
            Literal::new_language_tagged_literal("bar", "en-US").expect("valid language literal");

        assert_eq!(upper, lower, "tags differing only in case must be equal");
        assert_ne!(upper, different_value);

        // Case-insensitively-equal tags must hash equal too, so they behave
        // correctly as `HashMap`/`HashSet` keys.
        let hash_of = |l: &Literal| {
            let mut hasher = DefaultHasher::new();
            l.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash_of(&upper), hash_of(&lower));

        // And they must compare `Equal` under `Ord`, consistent with `PartialEq`.
        assert_eq!(upper.cmp(&lower), std::cmp::Ordering::Equal);

        // The cross-type `Literal == LiteralRef` comparison must agree too.
        let lower_ref = LiteralRef::new_language_tagged_literal_unchecked("foo", "en-us");
        assert_eq!(upper, lower_ref);
        assert_eq!(lower_ref, upper);
    }

    /// Regression test: the directional (rdf-12) language-tagged literal
    /// constructor must also preserve tag case, matching
    /// `new_language_tagged_literal`.
    #[cfg(feature = "rdf-12")]
    #[test]
    fn regression_directional_language_tag_preserves_original_case() {
        let literal =
            Literal::new_directional_language_tagged_literal("foo", "en-US", BaseDirection::Ltr)
                .expect("valid directional language literal");
        assert_eq!(literal.language(), Some("en-US"));
    }
}
