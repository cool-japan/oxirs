//! SHACL `sh:datatype` constraint checker.
//!
//! This module validates RDF literal values against XSD datatype constraints
//! as required by the SHACL specification (W3C Recommendation, July 2017,
//! section 4.3 — `sh:datatype`).
//!
//! # Supported datatypes
//!
//! - `xsd:string`
//! - `xsd:integer` (`xsd:int`, `xsd:long`, `xsd:short`, `xsd:byte`)
//! - `xsd:decimal`
//! - `xsd:float`
//! - `xsd:double`
//! - `xsd:boolean`
//! - `xsd:date` (YYYY-MM-DD)
//! - `xsd:dateTime` (YYYY-MM-DDThh:mm:ss with optional timezone)
//! - `xsd:time` (hh:mm:ss)
//! - `xsd:duration` (PnYnMnDTnHnMnS)
//! - `xsd:anyURI`
//! - `xsd:base64Binary`
//! - `xsd:hexBinary`
//! - Custom (pass-through)

use std::fmt;

/// Known XSD datatypes relevant to SHACL validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum XsdDatatype {
    XsdString,
    XsdInteger,
    XsdDecimal,
    XsdFloat,
    XsdDouble,
    XsdBoolean,
    XsdDate,
    XsdDateTime,
    XsdTime,
    XsdDuration,
    XsdAnyUri,
    XsdBase64Binary,
    XsdHexBinary,
    /// A custom / unsupported datatype — stored as its full IRI.
    Custom(String),
}

impl fmt::Display for XsdDatatype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::XsdString => write!(f, "xsd:string"),
            Self::XsdInteger => write!(f, "xsd:integer"),
            Self::XsdDecimal => write!(f, "xsd:decimal"),
            Self::XsdFloat => write!(f, "xsd:float"),
            Self::XsdDouble => write!(f, "xsd:double"),
            Self::XsdBoolean => write!(f, "xsd:boolean"),
            Self::XsdDate => write!(f, "xsd:date"),
            Self::XsdDateTime => write!(f, "xsd:dateTime"),
            Self::XsdTime => write!(f, "xsd:time"),
            Self::XsdDuration => write!(f, "xsd:duration"),
            Self::XsdAnyUri => write!(f, "xsd:anyURI"),
            Self::XsdBase64Binary => write!(f, "xsd:base64Binary"),
            Self::XsdHexBinary => write!(f, "xsd:hexBinary"),
            Self::Custom(iri) => write!(f, "{iri}"),
        }
    }
}

/// An RDF literal value with its datatype IRI and optional language tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralValue {
    /// The lexical form of the literal (as it appears in the RDF document).
    pub lexical: String,
    /// The datatype IRI (e.g. `http://www.w3.org/2001/XMLSchema#integer`).
    pub datatype: String,
    /// Optional BCP-47 language tag (for `rdf:langString`).
    pub lang: Option<String>,
}

impl LiteralValue {
    pub fn new(lexical: impl Into<String>, datatype: impl Into<String>) -> Self {
        Self {
            lexical: lexical.into(),
            datatype: datatype.into(),
            lang: None,
        }
    }

    pub fn with_lang(
        lexical: impl Into<String>,
        datatype: impl Into<String>,
        lang: impl Into<String>,
    ) -> Self {
        Self {
            lexical: lexical.into(),
            datatype: datatype.into(),
            lang: Some(lang.into()),
        }
    }
}

/// Errors produced by `DatatypeChecker`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatatypeError {
    /// The lexical form is not valid for the given datatype.
    InvalidLexicalForm { value: String, datatype: String },
    /// The datatype IRI is not recognised or supported.
    UnsupportedDatatype(String),
    /// The literal's datatype does not match the expected one.
    TypeMismatch { expected: String, found: String },
}

impl fmt::Display for DatatypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLexicalForm { value, datatype } => {
                write!(
                    f,
                    "Invalid lexical form '{}' for datatype {}",
                    value, datatype
                )
            }
            Self::UnsupportedDatatype(dt) => write!(f, "Unsupported datatype: {}", dt),
            Self::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

// ─── Constants ───────────────────────────────────────────────────────────────

const XSD_PREFIX: &str = "http://www.w3.org/2001/XMLSchema#";

// ─── DatatypeChecker ─────────────────────────────────────────────────────────

/// SHACL `sh:datatype` constraint checker.
#[derive(Debug, Default)]
pub struct DatatypeChecker;

impl DatatypeChecker {
    /// Create a new `DatatypeChecker`.
    pub fn new() -> Self {
        Self
    }

    /// Validate `literal` against the `expected` datatype.
    ///
    /// # Errors
    ///
    /// - `TypeMismatch` if the literal's `datatype` field doesn't match the
    ///   expected datatype (for known XSD types).
    /// - `InvalidLexicalForm` if the lexical form is syntactically invalid.
    /// - `UnsupportedDatatype` is NOT returned for `Custom` — custom types
    ///   are treated as a pass-through.
    pub fn check(
        &self,
        literal: &LiteralValue,
        expected: &XsdDatatype,
    ) -> Result<(), DatatypeError> {
        // For custom datatypes we accept any literal that declares the same IRI.
        if let XsdDatatype::Custom(expected_iri) = expected {
            if literal.datatype != *expected_iri {
                return Err(DatatypeError::TypeMismatch {
                    expected: expected_iri.clone(),
                    found: literal.datatype.clone(),
                });
            }
            return Ok(());
        }

        // Verify declared datatype matches expected
        let expected_iri = self.datatype_to_iri(expected);
        if !literal.datatype.is_empty() && literal.datatype != expected_iri {
            return Err(DatatypeError::TypeMismatch {
                expected: expected_iri,
                found: literal.datatype.clone(),
            });
        }

        // Validate lexical form
        let valid = match expected {
            XsdDatatype::XsdString => true, // all strings are valid
            XsdDatatype::XsdInteger => Self::validate_integer(&literal.lexical),
            XsdDatatype::XsdDecimal => Self::validate_decimal(&literal.lexical),
            XsdDatatype::XsdFloat => Self::validate_float(&literal.lexical),
            XsdDatatype::XsdDouble => Self::validate_float(&literal.lexical),
            XsdDatatype::XsdBoolean => Self::validate_boolean(&literal.lexical),
            XsdDatatype::XsdDate => Self::validate_date(&literal.lexical),
            XsdDatatype::XsdDateTime => Self::validate_datetime(&literal.lexical),
            XsdDatatype::XsdTime => Self::validate_time(&literal.lexical),
            XsdDatatype::XsdDuration => Self::validate_duration(&literal.lexical),
            XsdDatatype::XsdAnyUri => Self::validate_any_uri(&literal.lexical),
            XsdDatatype::XsdBase64Binary => Self::validate_base64_binary(&literal.lexical),
            XsdDatatype::XsdHexBinary => Self::validate_hex_binary(&literal.lexical),
            XsdDatatype::Custom(_) => unreachable!("handled above"),
        };

        if valid {
            Ok(())
        } else {
            Err(DatatypeError::InvalidLexicalForm {
                value: literal.lexical.clone(),
                datatype: expected.to_string(),
            })
        }
    }

    /// Convert an XSD IRI string to an `XsdDatatype` enum variant.
    pub fn parse_datatype(iri: &str) -> XsdDatatype {
        let local = if let Some(stripped) = iri.strip_prefix(XSD_PREFIX) {
            stripped
        } else if let Some(stripped) = iri.strip_prefix("xsd:") {
            stripped
        } else {
            return XsdDatatype::Custom(iri.to_string());
        };

        match local {
            "string" => XsdDatatype::XsdString,
            "integer" | "int" | "long" | "short" | "byte" | "nonNegativeInteger"
            | "positiveInteger" | "nonPositiveInteger" | "negativeInteger" | "unsignedByte"
            | "unsignedShort" | "unsignedInt" | "unsignedLong" => XsdDatatype::XsdInteger,
            "decimal" => XsdDatatype::XsdDecimal,
            "float" => XsdDatatype::XsdFloat,
            "double" => XsdDatatype::XsdDouble,
            "boolean" => XsdDatatype::XsdBoolean,
            "date" => XsdDatatype::XsdDate,
            "dateTime" => XsdDatatype::XsdDateTime,
            "time" => XsdDatatype::XsdTime,
            "duration" => XsdDatatype::XsdDuration,
            "anyURI" => XsdDatatype::XsdAnyUri,
            "base64Binary" => XsdDatatype::XsdBase64Binary,
            "hexBinary" => XsdDatatype::XsdHexBinary,
            other => XsdDatatype::Custom(format!("{XSD_PREFIX}{other}")),
        }
    }

    // ── Lexical validators ───────────────────────────────────────────────────

    /// Validate `xsd:integer` — optional sign, one or more digits.
    pub fn validate_integer(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
        !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
    }

    /// Validate `xsd:decimal` — optional sign, digits with optional decimal point.
    pub fn validate_decimal(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        let rest = s.strip_prefix(['+', '-']).unwrap_or(s);
        if rest.is_empty() {
            return false;
        }
        // Allow digits with at most one decimal point
        let parts: Vec<&str> = rest.splitn(2, '.').collect();
        match parts.as_slice() {
            [integer_part] => {
                !integer_part.is_empty() && integer_part.chars().all(|c| c.is_ascii_digit())
            }
            [integer_part, fractional_part] => {
                let int_ok =
                    !integer_part.is_empty() && integer_part.chars().all(|c| c.is_ascii_digit());
                let frac_ok = !fractional_part.is_empty()
                    && fractional_part.chars().all(|c| c.is_ascii_digit());
                int_ok && frac_ok
            }
            _ => false,
        }
    }

    /// Validate `xsd:float` / `xsd:double` — IEEE 754 textual forms including
    /// `INF`, `-INF`, `NaN`, and scientific notation.
    pub fn validate_float(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        match s {
            "INF" | "-INF" | "+INF" | "NaN" => return true,
            _ => {}
        }
        // Try parsing as f64
        s.parse::<f64>().is_ok()
    }

    /// Validate `xsd:boolean` — must be `"true"`, `"false"`, `"1"`, or `"0"`.
    pub fn validate_boolean(s: &str) -> bool {
        matches!(s, "true" | "false" | "1" | "0")
    }

    /// Validate `xsd:date` — YYYY-MM-DD with optional timezone.
    pub fn validate_date(s: &str) -> bool {
        // Strip optional timezone
        let base = Self::strip_timezone(s);
        Self::validate_date_core(base)
    }

    fn validate_date_core(s: &str) -> bool {
        if s.len() != 10 {
            return false;
        }
        let parts: Vec<&str> = s.splitn(3, '-').collect();
        if parts.len() != 3 {
            return false;
        }
        let year_ok = parts[0].len() == 4 && parts[0].chars().all(|c| c.is_ascii_digit());
        let month_ok =
            parts[1].len() == 2 && Self::parse_u8(parts[1]).is_some_and(|m| (1..=12).contains(&m));
        let day_ok =
            parts[2].len() == 2 && Self::parse_u8(parts[2]).is_some_and(|d| (1..=31).contains(&d));
        year_ok && month_ok && day_ok
    }

    /// Validate `xsd:dateTime` — YYYY-MM-DDThh:mm:ss with optional fractional
    /// seconds and optional timezone.
    pub fn validate_datetime(s: &str) -> bool {
        // Split on 'T'
        if let Some(t_pos) = s.find('T') {
            let date_part = &s[..t_pos];
            let time_rest = &s[t_pos + 1..];
            let time_part = Self::strip_timezone(time_rest);
            Self::validate_date_core(date_part) && Self::validate_time_core(time_part)
        } else {
            false
        }
    }

    /// Validate `xsd:time` — hh:mm:ss with optional fractional seconds and timezone.
    pub fn validate_time(s: &str) -> bool {
        let base = Self::strip_timezone(s);
        Self::validate_time_core(base)
    }

    fn validate_time_core(s: &str) -> bool {
        // hh:mm:ss or hh:mm:ss.sss
        let base = if let Some(dot_pos) = s.find('.') {
            let frac = &s[dot_pos + 1..];
            if frac.is_empty() || !frac.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
            &s[..dot_pos]
        } else {
            s
        };

        if base.len() != 8 {
            return false;
        }
        let parts: Vec<&str> = base.splitn(3, ':').collect();
        if parts.len() != 3 {
            return false;
        }
        let h_ok = parts[0].len() == 2 && Self::parse_u8(parts[0]).is_some_and(|h| h <= 23);
        let m_ok = parts[1].len() == 2 && Self::parse_u8(parts[1]).is_some_and(|m| m <= 59);
        let s_ok = parts[2].len() == 2 && Self::parse_u8(parts[2]).is_some_and(|s| s <= 60); // 60 for leap second
        h_ok && m_ok && s_ok
    }

    /// Validate `xsd:duration` — ISO 8601 duration (PnYnMnDTnHnMnS).
    fn validate_duration(s: &str) -> bool {
        if s.is_empty() || !s.starts_with('P') {
            return false;
        }
        // Minimal check: must start with P, rest must be non-empty
        s.len() > 1
    }

    /// Validate `xsd:anyURI` — non-empty, looks like a URI.
    pub fn validate_any_uri(s: &str) -> bool {
        !s.is_empty() && !s.contains(' ') && s.chars().all(|c| !c.is_control())
    }

    /// Validate `xsd:base64Binary` — valid Base64 characters (with padding).
    fn validate_base64_binary(s: &str) -> bool {
        if s.is_empty() {
            return true; // empty is valid
        }
        let no_whitespace: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        if no_whitespace.len() % 4 != 0 {
            return false;
        }
        no_whitespace
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
    }

    /// Validate `xsd:hexBinary` — even-length string of hex digits.
    pub fn validate_hex_binary(s: &str) -> bool {
        if s.is_empty() {
            return true; // empty is valid (zero bytes)
        }
        s.len() % 2 == 0 && s.chars().all(|c| c.is_ascii_hexdigit())
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Strip an optional timezone suffix (`Z`, `+hh:mm`, `-hh:mm`).
    fn strip_timezone(s: &str) -> &str {
        if let Some(stripped) = s.strip_suffix('Z') {
            stripped
        } else if let Some(pos) = s.rfind(['+', '-']) {
            // Check if what follows looks like hh:mm
            let tz_part = &s[pos..];
            if tz_part.len() == 6 && tz_part.chars().nth(3) == Some(':') {
                &s[..pos]
            } else {
                s
            }
        } else {
            s
        }
    }

    fn parse_u8(s: &str) -> Option<u8> {
        s.parse::<u8>().ok()
    }

    /// Convert an `XsdDatatype` back to its canonical IRI.
    fn datatype_to_iri(&self, dt: &XsdDatatype) -> String {
        match dt {
            XsdDatatype::XsdString => format!("{XSD_PREFIX}string"),
            XsdDatatype::XsdInteger => format!("{XSD_PREFIX}integer"),
            XsdDatatype::XsdDecimal => format!("{XSD_PREFIX}decimal"),
            XsdDatatype::XsdFloat => format!("{XSD_PREFIX}float"),
            XsdDatatype::XsdDouble => format!("{XSD_PREFIX}double"),
            XsdDatatype::XsdBoolean => format!("{XSD_PREFIX}boolean"),
            XsdDatatype::XsdDate => format!("{XSD_PREFIX}date"),
            XsdDatatype::XsdDateTime => format!("{XSD_PREFIX}dateTime"),
            XsdDatatype::XsdTime => format!("{XSD_PREFIX}time"),
            XsdDatatype::XsdDuration => format!("{XSD_PREFIX}duration"),
            XsdDatatype::XsdAnyUri => format!("{XSD_PREFIX}anyURI"),
            XsdDatatype::XsdBase64Binary => format!("{XSD_PREFIX}base64Binary"),
            XsdDatatype::XsdHexBinary => format!("{XSD_PREFIX}hexBinary"),
            XsdDatatype::Custom(iri) => iri.clone(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> DatatypeChecker {
        DatatypeChecker::new()
    }

    fn xsd_lit(lexical: &str, local: &str) -> LiteralValue {
        LiteralValue::new(lexical, format!("{XSD_PREFIX}{local}"))
    }

    // ── parse_datatype ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_xsd_string() {
        assert_eq!(
            DatatypeChecker::parse_datatype("http://www.w3.org/2001/XMLSchema#string"),
            XsdDatatype::XsdString
        );
    }

    #[test]
    fn test_parse_xsd_integer() {
        assert_eq!(
            DatatypeChecker::parse_datatype("http://www.w3.org/2001/XMLSchema#integer"),
            XsdDatatype::XsdInteger
        );
    }

    #[test]
    fn test_parse_xsd_int_alias() {
        assert_eq!(
            DatatypeChecker::parse_datatype("http://www.w3.org/2001/XMLSchema#int"),
            XsdDatatype::XsdInteger
        );
    }

    #[test]
    fn test_parse_xsd_decimal() {
        assert_eq!(
            DatatypeChecker::parse_datatype("http://www.w3.org/2001/XMLSchema#decimal"),
            XsdDatatype::XsdDecimal
        );
    }

    #[test]
    fn test_parse_xsd_boolean() {
        assert_eq!(
            DatatypeChecker::parse_datatype("http://www.w3.org/2001/XMLSchema#boolean"),
            XsdDatatype::XsdBoolean
        );
    }

    #[test]
    fn test_parse_xsd_date() {
        assert_eq!(
            DatatypeChecker::parse_datatype("http://www.w3.org/2001/XMLSchema#date"),
            XsdDatatype::XsdDate
        );
    }

    #[test]
    fn test_parse_xsd_datetime() {
        assert_eq!(
            DatatypeChecker::parse_datatype("http://www.w3.org/2001/XMLSchema#dateTime"),
            XsdDatatype::XsdDateTime
        );
    }

    #[test]
    fn test_parse_xsd_hex_binary() {
        assert_eq!(
            DatatypeChecker::parse_datatype("http://www.w3.org/2001/XMLSchema#hexBinary"),
            XsdDatatype::XsdHexBinary
        );
    }

    #[test]
    fn test_parse_xsd_prefix_shorthand() {
        assert_eq!(
            DatatypeChecker::parse_datatype("xsd:string"),
            XsdDatatype::XsdString
        );
        assert_eq!(
            DatatypeChecker::parse_datatype("xsd:integer"),
            XsdDatatype::XsdInteger
        );
    }

    #[test]
    fn test_parse_custom_iri() {
        let result = DatatypeChecker::parse_datatype("http://example.org/mytype");
        assert!(matches!(result, XsdDatatype::Custom(_)));
    }

    // ── validate_integer ──────────────────────────────────────────────────────

    #[test]
    fn test_integer_valid_positive() {
        assert!(DatatypeChecker::validate_integer("42"));
    }

    #[test]
    fn test_integer_valid_negative() {
        assert!(DatatypeChecker::validate_integer("-42"));
    }

    #[test]
    fn test_integer_valid_plus_prefix() {
        assert!(DatatypeChecker::validate_integer("+100"));
    }

    #[test]
    fn test_integer_invalid_decimal() {
        assert!(!DatatypeChecker::validate_integer("3.14"));
    }

    #[test]
    fn test_integer_invalid_empty() {
        assert!(!DatatypeChecker::validate_integer(""));
    }

    #[test]
    fn test_integer_invalid_letters() {
        assert!(!DatatypeChecker::validate_integer("abc"));
    }

    // ── validate_decimal ─────────────────────────────────────────────────────

    #[test]
    fn test_decimal_valid_with_point() {
        assert!(DatatypeChecker::validate_decimal("3.14"));
    }

    #[test]
    fn test_decimal_valid_integer_form() {
        assert!(DatatypeChecker::validate_decimal("100"));
    }

    #[test]
    fn test_decimal_valid_negative() {
        assert!(DatatypeChecker::validate_decimal("-2.718"));
    }

    #[test]
    fn test_decimal_invalid_multiple_points() {
        assert!(!DatatypeChecker::validate_decimal("1.2.3"));
    }

    #[test]
    fn test_decimal_invalid_empty() {
        assert!(!DatatypeChecker::validate_decimal(""));
    }

    // ── validate_float ───────────────────────────────────────────────────────

    #[test]
    fn test_float_valid_scientific() {
        assert!(DatatypeChecker::validate_float("1.5e10"));
    }

    #[test]
    fn test_float_valid_inf() {
        assert!(DatatypeChecker::validate_float("INF"));
    }

    #[test]
    fn test_float_valid_neg_inf() {
        assert!(DatatypeChecker::validate_float("-INF"));
    }

    #[test]
    fn test_float_valid_nan() {
        assert!(DatatypeChecker::validate_float("NaN"));
    }

    #[test]
    fn test_float_invalid_word() {
        assert!(!DatatypeChecker::validate_float("hello"));
    }

    // ── validate_boolean ─────────────────────────────────────────────────────

    #[test]
    fn test_boolean_true() {
        assert!(DatatypeChecker::validate_boolean("true"));
    }

    #[test]
    fn test_boolean_false() {
        assert!(DatatypeChecker::validate_boolean("false"));
    }

    #[test]
    fn test_boolean_one() {
        assert!(DatatypeChecker::validate_boolean("1"));
    }

    #[test]
    fn test_boolean_zero() {
        assert!(DatatypeChecker::validate_boolean("0"));
    }

    #[test]
    fn test_boolean_invalid_yes() {
        assert!(!DatatypeChecker::validate_boolean("yes"));
    }

    #[test]
    fn test_boolean_invalid_uppercase() {
        assert!(!DatatypeChecker::validate_boolean("True"));
    }

    // ── validate_date ─────────────────────────────────────────────────────────

    #[test]
    fn test_date_valid() {
        assert!(DatatypeChecker::validate_date("2024-03-15"));
    }

    #[test]
    fn test_date_valid_with_z() {
        assert!(DatatypeChecker::validate_date("2024-03-15Z"));
    }

    #[test]
    fn test_date_invalid_format() {
        assert!(!DatatypeChecker::validate_date("15-03-2024"));
    }

    #[test]
    fn test_date_invalid_month() {
        assert!(!DatatypeChecker::validate_date("2024-13-01"));
    }

    #[test]
    fn test_date_invalid_empty() {
        assert!(!DatatypeChecker::validate_date(""));
    }

    // ── validate_datetime ─────────────────────────────────────────────────────

    #[test]
    fn test_datetime_valid() {
        assert!(DatatypeChecker::validate_datetime("2024-03-15T10:30:00"));
    }

    #[test]
    fn test_datetime_valid_with_z() {
        assert!(DatatypeChecker::validate_datetime("2024-03-15T10:30:00Z"));
    }

    #[test]
    fn test_datetime_invalid_no_t() {
        assert!(!DatatypeChecker::validate_datetime("2024-03-15 10:30:00"));
    }

    #[test]
    fn test_datetime_invalid_bad_date() {
        assert!(!DatatypeChecker::validate_datetime("2024-13-15T10:30:00"));
    }

    // ── validate_hex_binary ───────────────────────────────────────────────────

    #[test]
    fn test_hex_binary_valid_even() {
        assert!(DatatypeChecker::validate_hex_binary("0FA3"));
    }

    #[test]
    fn test_hex_binary_valid_empty() {
        assert!(DatatypeChecker::validate_hex_binary(""));
    }

    #[test]
    fn test_hex_binary_invalid_odd_length() {
        assert!(!DatatypeChecker::validate_hex_binary("0FA"));
    }

    #[test]
    fn test_hex_binary_invalid_non_hex() {
        assert!(!DatatypeChecker::validate_hex_binary("ZZZZ"));
    }

    #[test]
    fn test_hex_binary_lowercase_valid() {
        assert!(DatatypeChecker::validate_hex_binary("deadbeef"));
    }

    // ── validate_any_uri ─────────────────────────────────────────────────────

    #[test]
    fn test_any_uri_valid_http() {
        assert!(DatatypeChecker::validate_any_uri(
            "http://example.org/resource"
        ));
    }

    #[test]
    fn test_any_uri_valid_relative() {
        assert!(DatatypeChecker::validate_any_uri("resource/path"));
    }

    #[test]
    fn test_any_uri_invalid_empty() {
        assert!(!DatatypeChecker::validate_any_uri(""));
    }

    #[test]
    fn test_any_uri_invalid_with_space() {
        assert!(!DatatypeChecker::validate_any_uri(
            "http://example.org/re source"
        ));
    }

    // ── check (full constraint check) ─────────────────────────────────────────

    #[test]
    fn test_check_integer_valid() {
        let c = checker();
        let lit = xsd_lit("42", "integer");
        assert!(c.check(&lit, &XsdDatatype::XsdInteger).is_ok());
    }

    #[test]
    fn test_check_integer_invalid_form() {
        let c = checker();
        let lit = xsd_lit("not-a-number", "integer");
        assert!(matches!(
            c.check(&lit, &XsdDatatype::XsdInteger),
            Err(DatatypeError::InvalidLexicalForm { .. })
        ));
    }

    #[test]
    fn test_check_boolean_valid() {
        let c = checker();
        let lit = xsd_lit("true", "boolean");
        assert!(c.check(&lit, &XsdDatatype::XsdBoolean).is_ok());
    }

    #[test]
    fn test_check_string_always_valid() {
        let c = checker();
        let lit = xsd_lit("anything goes here!", "string");
        assert!(c.check(&lit, &XsdDatatype::XsdString).is_ok());
    }

    #[test]
    fn test_check_date_valid() {
        let c = checker();
        let lit = xsd_lit("2024-01-01", "date");
        assert!(c.check(&lit, &XsdDatatype::XsdDate).is_ok());
    }

    #[test]
    fn test_check_type_mismatch() {
        let c = checker();
        let lit = xsd_lit("42", "string"); // declared as string
                                           // Expect TypeMismatch because declared type ≠ expected type
        assert!(matches!(
            c.check(&lit, &XsdDatatype::XsdInteger),
            Err(DatatypeError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_check_custom_datatype_pass_through() {
        let c = checker();
        let iri = "http://example.org/mytype";
        let lit = LiteralValue::new("some_value", iri);
        let expected = XsdDatatype::Custom(iri.to_string());
        assert!(c.check(&lit, &expected).is_ok());
    }

    #[test]
    fn test_check_custom_datatype_mismatch() {
        let c = checker();
        let lit = LiteralValue::new("val", "http://example.org/other");
        let expected = XsdDatatype::Custom("http://example.org/mytype".to_string());
        assert!(matches!(
            c.check(&lit, &expected),
            Err(DatatypeError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_check_hex_binary_valid() {
        let c = checker();
        let lit = xsd_lit("DEADBEEF", "hexBinary");
        assert!(c.check(&lit, &XsdDatatype::XsdHexBinary).is_ok());
    }

    #[test]
    fn test_check_hex_binary_invalid() {
        let c = checker();
        let lit = xsd_lit("XYZ", "hexBinary");
        assert!(matches!(
            c.check(&lit, &XsdDatatype::XsdHexBinary),
            Err(DatatypeError::InvalidLexicalForm { .. })
        ));
    }

    #[test]
    fn test_check_any_uri_valid() {
        let c = checker();
        let lit = xsd_lit("https://example.org/", "anyURI");
        assert!(c.check(&lit, &XsdDatatype::XsdAnyUri).is_ok());
    }

    #[test]
    fn test_check_decimal_valid() {
        let c = checker();
        let lit = xsd_lit("3.14159", "decimal");
        assert!(c.check(&lit, &XsdDatatype::XsdDecimal).is_ok());
    }

    #[test]
    fn test_check_float_valid_nan() {
        let c = checker();
        let lit = xsd_lit("NaN", "float");
        assert!(c.check(&lit, &XsdDatatype::XsdFloat).is_ok());
    }

    #[test]
    fn test_check_datetime_valid() {
        let c = checker();
        let lit = xsd_lit("2025-06-01T12:00:00Z", "dateTime");
        assert!(c.check(&lit, &XsdDatatype::XsdDateTime).is_ok());
    }
}
