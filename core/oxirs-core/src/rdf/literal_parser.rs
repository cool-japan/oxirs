//! RDF literal value parser and validator.
//!
//! Supports all common XSD datatypes with normalization and validation.

use std::fmt;

/// XSD datatype enumeration for RDF literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XsdType {
    Boolean,
    Integer,
    Decimal,
    Double,
    Float,
    String,
    Date,
    DateTime,
    Duration,
    AnyUri,
    HexBinary,
    Base64Binary,
    Language,
    NonNegativeInteger,
    PositiveInteger,
}

/// A successfully parsed and normalized RDF literal.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedLiteral {
    /// Original input value.
    pub value: std::string::String,
    /// Detected or specified datatype.
    pub datatype: XsdType,
    /// Canonical normalized form.
    pub normalized: std::string::String,
}

/// Errors that can occur when parsing or validating RDF literals.
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralError {
    InvalidBoolean(std::string::String),
    InvalidInteger(std::string::String),
    InvalidDecimal(std::string::String),
    InvalidDate(std::string::String),
    InvalidDateTime(std::string::String),
    UnknownDatatype(std::string::String),
    InvalidHexBinary(std::string::String),
    InvalidDouble(std::string::String),
    InvalidFloat(std::string::String),
    InvalidDuration(std::string::String),
    InvalidBase64Binary(std::string::String),
    InvalidNonNegativeInteger(std::string::String),
    InvalidPositiveInteger(std::string::String),
}

impl fmt::Display for LiteralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiteralError::InvalidBoolean(v) => write!(f, "Invalid boolean: {v}"),
            LiteralError::InvalidInteger(v) => write!(f, "Invalid integer: {v}"),
            LiteralError::InvalidDecimal(v) => write!(f, "Invalid decimal: {v}"),
            LiteralError::InvalidDate(v) => write!(f, "Invalid date: {v}"),
            LiteralError::InvalidDateTime(v) => write!(f, "Invalid dateTime: {v}"),
            LiteralError::UnknownDatatype(v) => write!(f, "Unknown datatype: {v}"),
            LiteralError::InvalidHexBinary(v) => write!(f, "Invalid hex binary: {v}"),
            LiteralError::InvalidDouble(v) => write!(f, "Invalid double: {v}"),
            LiteralError::InvalidFloat(v) => write!(f, "Invalid float: {v}"),
            LiteralError::InvalidDuration(v) => write!(f, "Invalid duration: {v}"),
            LiteralError::InvalidBase64Binary(v) => write!(f, "Invalid base64 binary: {v}"),
            LiteralError::InvalidNonNegativeInteger(v) => {
                write!(f, "Invalid non-negative integer: {v}")
            }
            LiteralError::InvalidPositiveInteger(v) => write!(f, "Invalid positive integer: {v}"),
        }
    }
}

impl std::error::Error for LiteralError {}

/// RDF literal parser and normalizer.
pub struct LiteralParser;

impl LiteralParser {
    /// Parse a literal value given its XSD datatype IRI.
    pub fn parse(
        value: &str,
        datatype_iri: &str,
    ) -> Result<ParsedLiteral, LiteralError> {
        let xsd_type = Self::from_xsd_iri(datatype_iri)
            .ok_or_else(|| LiteralError::UnknownDatatype(datatype_iri.to_string()))?;

        let normalized = match xsd_type {
            XsdType::Boolean => Self::normalize_boolean(value)?,
            XsdType::Integer => Self::normalize_integer(value)?,
            XsdType::Decimal => Self::normalize_decimal(value)?,
            XsdType::Double => Self::normalize_double(value)?,
            XsdType::Float => Self::normalize_float(value)?,
            XsdType::String => value.to_string(),
            XsdType::Date => Self::normalize_date(value)?,
            XsdType::DateTime => Self::normalize_datetime(value)?,
            XsdType::Duration => Self::normalize_duration(value)?,
            XsdType::AnyUri => value.to_string(),
            XsdType::HexBinary => Self::validate_hex_binary(value)?,
            XsdType::Base64Binary => Self::validate_base64_binary(value)?,
            XsdType::Language => value.to_lowercase(),
            XsdType::NonNegativeInteger => Self::normalize_non_negative_integer(value)?,
            XsdType::PositiveInteger => Self::normalize_positive_integer(value)?,
        };

        Ok(ParsedLiteral {
            value: value.to_string(),
            datatype: xsd_type,
            normalized,
        })
    }

    /// Heuristically detect XSD type by attempting to parse in order.
    pub fn detect_type(value: &str) -> XsdType {
        // Try boolean first
        if matches!(value, "true" | "false" | "1" | "0") {
            return XsdType::Boolean;
        }

        // Try integer (possibly signed, no decimal point)
        if Self::normalize_integer(value).is_ok() {
            return XsdType::Integer;
        }

        // Try decimal (has decimal point, no exponent)
        if value.contains('.') && !value.to_lowercase().contains('e')
            && Self::normalize_decimal(value).is_ok()
        {
            return XsdType::Decimal;
        }

        // Try double (scientific notation)
        if value.to_lowercase().contains('e') && Self::normalize_double(value).is_ok() {
            return XsdType::Double;
        }

        // Try date (YYYY-MM-DD)
        if value.len() >= 10 && value.chars().nth(4) == Some('-') && value.chars().nth(7) == Some('-') {
            if value.len() == 10 && Self::normalize_date(value).is_ok() {
                return XsdType::Date;
            } else if value.contains('T') && Self::normalize_datetime(value).is_ok() {
                return XsdType::DateTime;
            }
        }

        // Try duration (starts with P)
        if (value.starts_with('P') || value.starts_with("-P"))
            && Self::normalize_duration(value).is_ok()
        {
            return XsdType::Duration;
        }

        // Try AnyURI (contains ://)
        if value.contains("://") {
            return XsdType::AnyUri;
        }

        // Try hex binary (all hex chars, even length)
        if !value.is_empty()
            && value.len() % 2 == 0
            && value.chars().all(|c| c.is_ascii_hexdigit())
        {
            return XsdType::HexBinary;
        }

        // Default to string
        XsdType::String
    }

    /// Normalize a boolean literal. "1"/"true" → "true", "0"/"false" → "false".
    pub fn normalize_boolean(v: &str) -> Result<std::string::String, LiteralError> {
        match v.trim() {
            "true" | "1" => Ok("true".to_string()),
            "false" | "0" => Ok("false".to_string()),
            other => Err(LiteralError::InvalidBoolean(other.to_string())),
        }
    }

    /// Normalize an integer literal: strip leading zeros, handle sign.
    pub fn normalize_integer(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            return Err(LiteralError::InvalidInteger(v.to_string()));
        }

        let (sign, digits) = if let Some(stripped) = trimmed.strip_prefix('-') {
            ("-", stripped)
        } else if let Some(stripped) = trimmed.strip_prefix('+') {
            ("", stripped)
        } else {
            ("", trimmed)
        };

        if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
            return Err(LiteralError::InvalidInteger(v.to_string()));
        }

        // Strip leading zeros
        let stripped = digits.trim_start_matches('0');
        let canonical_digits = if stripped.is_empty() { "0" } else { stripped };

        // If the value is zero, no sign
        if canonical_digits == "0" {
            Ok("0".to_string())
        } else {
            Ok(format!("{sign}{canonical_digits}"))
        }
    }

    /// Normalize a decimal literal to canonical form.
    pub fn normalize_decimal(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            return Err(LiteralError::InvalidDecimal(v.to_string()));
        }

        let (sign, rest) = if let Some(stripped) = trimmed.strip_prefix('-') {
            ("-", stripped)
        } else if let Some(stripped) = trimmed.strip_prefix('+') {
            ("", stripped)
        } else {
            ("", trimmed)
        };

        // Must contain a dot
        let dot_pos = rest.find('.').ok_or_else(|| LiteralError::InvalidDecimal(v.to_string()))?;
        let integer_part = &rest[..dot_pos];
        let fraction_part = &rest[dot_pos + 1..];

        // Validate all chars are digits
        if !integer_part.chars().all(|c| c.is_ascii_digit())
            || !fraction_part.chars().all(|c| c.is_ascii_digit())
        {
            return Err(LiteralError::InvalidDecimal(v.to_string()));
        }

        // Strip leading zeros from integer part
        let int_stripped = integer_part.trim_start_matches('0');
        let canonical_int = if int_stripped.is_empty() { "0" } else { int_stripped };

        // Strip trailing zeros from fraction part
        let frac_stripped = fraction_part.trim_end_matches('0');
        let canonical_frac = if frac_stripped.is_empty() { "0" } else { frac_stripped };

        let result = format!("{sign}{canonical_int}.{canonical_frac}");

        // Avoid "-0.0"
        if result == "-0.0" {
            return Ok("0.0".to_string());
        }

        Ok(result)
    }

    /// Normalize a double literal to scientific notation canonical form.
    pub fn normalize_double(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim();
        // Try to parse as f64
        let parsed: f64 = trimmed.parse().map_err(|_| LiteralError::InvalidDouble(v.to_string()))?;

        if parsed.is_nan() {
            return Ok("NaN".to_string());
        }
        if parsed.is_infinite() {
            return if parsed.is_sign_positive() {
                Ok("INF".to_string())
            } else {
                Ok("-INF".to_string())
            };
        }

        // Format in scientific notation
        Ok(format!("{parsed:E}"))
    }

    /// Normalize a float literal.
    pub fn normalize_float(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim();
        let parsed: f32 = trimmed.parse().map_err(|_| LiteralError::InvalidFloat(v.to_string()))?;

        if parsed.is_nan() {
            return Ok("NaN".to_string());
        }
        if parsed.is_infinite() {
            return if parsed.is_sign_positive() {
                Ok("INF".to_string())
            } else {
                Ok("-INF".to_string())
            };
        }

        Ok(format!("{parsed:E}"))
    }

    /// Validate and normalize a date literal: YYYY-MM-DD.
    pub fn normalize_date(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim();

        // Handle optional timezone suffix
        let (date_part, _tz) = split_date_timezone(trimmed);

        if date_part.len() != 10 {
            return Err(LiteralError::InvalidDate(v.to_string()));
        }

        let parts: Vec<&str> = date_part.splitn(3, '-').collect();
        if parts.len() != 3 {
            return Err(LiteralError::InvalidDate(v.to_string()));
        }

        let year: u32 = parts[0]
            .parse()
            .map_err(|_| LiteralError::InvalidDate(v.to_string()))?;
        let month: u32 = parts[1]
            .parse()
            .map_err(|_| LiteralError::InvalidDate(v.to_string()))?;
        let day: u32 = parts[2]
            .parse()
            .map_err(|_| LiteralError::InvalidDate(v.to_string()))?;

        if parts[0].len() < 4 || parts[1].len() != 2 || parts[2].len() != 2 {
            return Err(LiteralError::InvalidDate(v.to_string()));
        }

        if !(1..=12).contains(&month) {
            return Err(LiteralError::InvalidDate(v.to_string()));
        }

        let max_day = days_in_month(year, month);
        if day < 1 || day > max_day {
            return Err(LiteralError::InvalidDate(v.to_string()));
        }

        Ok(trimmed.to_string())
    }

    /// Validate and normalize a dateTime literal: YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM].
    pub fn normalize_datetime(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim();

        let t_pos = trimmed.find('T').ok_or_else(|| LiteralError::InvalidDateTime(v.to_string()))?;

        let date_str = &trimmed[..t_pos];
        let rest = &trimmed[t_pos + 1..];

        // Validate date part
        Self::normalize_date(date_str)
            .map_err(|_| LiteralError::InvalidDateTime(v.to_string()))?;

        // Strip timezone from time part
        let (time_part, _tz) = split_time_timezone(rest);

        // Split seconds from fractional
        let (time_no_frac, _frac) = if let Some(dot_pos) = time_part.find('.') {
            (&time_part[..dot_pos], Some(&time_part[dot_pos + 1..]))
        } else {
            (time_part, None)
        };

        let time_parts: Vec<&str> = time_no_frac.splitn(3, ':').collect();
        if time_parts.len() != 3 {
            return Err(LiteralError::InvalidDateTime(v.to_string()));
        }

        let hour: u32 = time_parts[0]
            .parse()
            .map_err(|_| LiteralError::InvalidDateTime(v.to_string()))?;
        let minute: u32 = time_parts[1]
            .parse()
            .map_err(|_| LiteralError::InvalidDateTime(v.to_string()))?;
        let second: u32 = time_parts[2]
            .parse()
            .map_err(|_| LiteralError::InvalidDateTime(v.to_string()))?;

        if time_parts[0].len() != 2 || time_parts[1].len() != 2 || time_parts[2].len() != 2 {
            return Err(LiteralError::InvalidDateTime(v.to_string()));
        }

        if hour > 23 || minute > 59 || second > 59 {
            return Err(LiteralError::InvalidDateTime(v.to_string()));
        }

        Ok(trimmed.to_string())
    }

    /// Validate and normalize a duration literal: P[nY][nM][nD][T[nH][nM][nS]].
    pub fn normalize_duration(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim();

        let s = if let Some(stripped) = trimmed.strip_prefix('-') {
            stripped
        } else {
            trimmed
        };

        if !s.starts_with('P') {
            return Err(LiteralError::InvalidDuration(v.to_string()));
        }

        // Basic validation: rest must contain valid duration chars
        let rest = &s[1..];
        if rest.is_empty() {
            return Err(LiteralError::InvalidDuration(v.to_string()));
        }

        // Check all characters are valid duration chars
        for c in rest.chars() {
            if !matches!(c, '0'..='9' | 'Y' | 'M' | 'D' | 'T' | 'H' | 'S' | '.') {
                return Err(LiteralError::InvalidDuration(v.to_string()));
            }
        }

        Ok(trimmed.to_string())
    }

    /// Validate a hexBinary literal: even-length string of hex chars.
    pub fn validate_hex_binary(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim();

        if trimmed.len() % 2 != 0 {
            return Err(LiteralError::InvalidHexBinary(format!(
                "odd length: {}",
                trimmed.len()
            )));
        }

        if !trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(LiteralError::InvalidHexBinary(format!(
                "non-hex characters in: {trimmed}"
            )));
        }

        Ok(trimmed.to_uppercase())
    }

    /// Validate a base64Binary literal.
    pub fn validate_base64_binary(v: &str) -> Result<std::string::String, LiteralError> {
        let trimmed = v.trim().replace(' ', "");

        for c in trimmed.chars() {
            if !matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '+' | '/' | '=') {
                return Err(LiteralError::InvalidBase64Binary(format!(
                    "invalid char '{c}'"
                )));
            }
        }

        Ok(trimmed)
    }

    /// Normalize a non-negative integer (>= 0).
    pub fn normalize_non_negative_integer(v: &str) -> Result<std::string::String, LiteralError> {
        let normalized = Self::normalize_integer(v)
            .map_err(|_| LiteralError::InvalidNonNegativeInteger(v.to_string()))?;

        // Check it's not negative
        if normalized.starts_with('-') {
            return Err(LiteralError::InvalidNonNegativeInteger(v.to_string()));
        }

        Ok(normalized)
    }

    /// Normalize a positive integer (> 0).
    pub fn normalize_positive_integer(v: &str) -> Result<std::string::String, LiteralError> {
        let normalized = Self::normalize_non_negative_integer(v)
            .map_err(|_| LiteralError::InvalidPositiveInteger(v.to_string()))?;

        if normalized == "0" {
            return Err(LiteralError::InvalidPositiveInteger(
                "0 is not positive".to_string(),
            ));
        }

        Ok(normalized)
    }

    /// Get XSD IRI for a given type.
    pub fn xsd_iri(t: XsdType) -> &'static str {
        match t {
            XsdType::Boolean => "http://www.w3.org/2001/XMLSchema#boolean",
            XsdType::Integer => "http://www.w3.org/2001/XMLSchema#integer",
            XsdType::Decimal => "http://www.w3.org/2001/XMLSchema#decimal",
            XsdType::Double => "http://www.w3.org/2001/XMLSchema#double",
            XsdType::Float => "http://www.w3.org/2001/XMLSchema#float",
            XsdType::String => "http://www.w3.org/2001/XMLSchema#string",
            XsdType::Date => "http://www.w3.org/2001/XMLSchema#date",
            XsdType::DateTime => "http://www.w3.org/2001/XMLSchema#dateTime",
            XsdType::Duration => "http://www.w3.org/2001/XMLSchema#duration",
            XsdType::AnyUri => "http://www.w3.org/2001/XMLSchema#anyURI",
            XsdType::HexBinary => "http://www.w3.org/2001/XMLSchema#hexBinary",
            XsdType::Base64Binary => "http://www.w3.org/2001/XMLSchema#base64Binary",
            XsdType::Language => "http://www.w3.org/2001/XMLSchema#language",
            XsdType::NonNegativeInteger => {
                "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
            }
            XsdType::PositiveInteger => "http://www.w3.org/2001/XMLSchema#positiveInteger",
        }
    }

    /// Get XsdType from an XSD IRI.
    pub fn from_xsd_iri(iri: &str) -> Option<XsdType> {
        match iri {
            "http://www.w3.org/2001/XMLSchema#boolean" | "xsd:boolean" => {
                Some(XsdType::Boolean)
            }
            "http://www.w3.org/2001/XMLSchema#integer" | "xsd:integer" => {
                Some(XsdType::Integer)
            }
            "http://www.w3.org/2001/XMLSchema#decimal" | "xsd:decimal" => {
                Some(XsdType::Decimal)
            }
            "http://www.w3.org/2001/XMLSchema#double" | "xsd:double" => Some(XsdType::Double),
            "http://www.w3.org/2001/XMLSchema#float" | "xsd:float" => Some(XsdType::Float),
            "http://www.w3.org/2001/XMLSchema#string" | "xsd:string" => Some(XsdType::String),
            "http://www.w3.org/2001/XMLSchema#date" | "xsd:date" => Some(XsdType::Date),
            "http://www.w3.org/2001/XMLSchema#dateTime" | "xsd:dateTime" => {
                Some(XsdType::DateTime)
            }
            "http://www.w3.org/2001/XMLSchema#duration" | "xsd:duration" => {
                Some(XsdType::Duration)
            }
            "http://www.w3.org/2001/XMLSchema#anyURI" | "xsd:anyURI" => Some(XsdType::AnyUri),
            "http://www.w3.org/2001/XMLSchema#hexBinary" | "xsd:hexBinary" => {
                Some(XsdType::HexBinary)
            }
            "http://www.w3.org/2001/XMLSchema#base64Binary" | "xsd:base64Binary" => {
                Some(XsdType::Base64Binary)
            }
            "http://www.w3.org/2001/XMLSchema#language" | "xsd:language" => {
                Some(XsdType::Language)
            }
            "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
            | "xsd:nonNegativeInteger" => Some(XsdType::NonNegativeInteger),
            "http://www.w3.org/2001/XMLSchema#positiveInteger" | "xsd:positiveInteger" => {
                Some(XsdType::PositiveInteger)
            }
            _ => None,
        }
    }
}

/// Split a date string from its timezone suffix.
fn split_date_timezone(s: &str) -> (&str, &str) {
    // timezone can be 'Z', '+HH:MM', '-HH:MM'
    // date is YYYY-MM-DD (10 chars)
    if s.len() > 10 {
        (&s[..10], &s[10..])
    } else {
        (s, "")
    }
}

/// Split a time string from its timezone suffix.
fn split_time_timezone(s: &str) -> (&str, &str) {
    // Find 'Z' or '+'/'-' that indicates timezone (after at least HH:MM:SS = 8 chars)
    if let Some(z_pos) = s.find('Z') {
        return (&s[..z_pos], &s[z_pos..]);
    }

    // Look for +/- that is timezone (not part of the time itself)
    // Time part is at least HH:MM:SS = 8 chars
    for (i, c) in s.char_indices().skip(6) {
        if c == '+' || c == '-' {
            return (&s[..i], &s[i..]);
        }
    }

    (s, "")
}

/// Return number of days in a given month of a given year.
fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Check if a year is a leap year.
fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- XsdType round-trip tests ---

    #[test]
    fn test_xsd_iri_boolean() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Boolean),
            "http://www.w3.org/2001/XMLSchema#boolean"
        );
    }

    #[test]
    fn test_xsd_iri_integer() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Integer),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
    }

    #[test]
    fn test_xsd_iri_decimal() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Decimal),
            "http://www.w3.org/2001/XMLSchema#decimal"
        );
    }

    #[test]
    fn test_xsd_iri_double() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Double),
            "http://www.w3.org/2001/XMLSchema#double"
        );
    }

    #[test]
    fn test_xsd_iri_float() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Float),
            "http://www.w3.org/2001/XMLSchema#float"
        );
    }

    #[test]
    fn test_xsd_iri_string() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::String),
            "http://www.w3.org/2001/XMLSchema#string"
        );
    }

    #[test]
    fn test_xsd_iri_date() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Date),
            "http://www.w3.org/2001/XMLSchema#date"
        );
    }

    #[test]
    fn test_xsd_iri_datetime() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::DateTime),
            "http://www.w3.org/2001/XMLSchema#dateTime"
        );
    }

    #[test]
    fn test_xsd_iri_duration() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Duration),
            "http://www.w3.org/2001/XMLSchema#duration"
        );
    }

    #[test]
    fn test_xsd_iri_anyuri() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::AnyUri),
            "http://www.w3.org/2001/XMLSchema#anyURI"
        );
    }

    #[test]
    fn test_xsd_iri_hexbinary() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::HexBinary),
            "http://www.w3.org/2001/XMLSchema#hexBinary"
        );
    }

    #[test]
    fn test_xsd_iri_base64binary() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Base64Binary),
            "http://www.w3.org/2001/XMLSchema#base64Binary"
        );
    }

    #[test]
    fn test_xsd_iri_language() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::Language),
            "http://www.w3.org/2001/XMLSchema#language"
        );
    }

    #[test]
    fn test_xsd_iri_nonnegativeinteger() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::NonNegativeInteger),
            "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
        );
    }

    #[test]
    fn test_xsd_iri_positiveinteger() {
        assert_eq!(
            LiteralParser::xsd_iri(XsdType::PositiveInteger),
            "http://www.w3.org/2001/XMLSchema#positiveInteger"
        );
    }

    // --- from_xsd_iri round-trip ---

    #[test]
    fn test_from_xsd_iri_roundtrip_all_types() {
        let types = [
            XsdType::Boolean,
            XsdType::Integer,
            XsdType::Decimal,
            XsdType::Double,
            XsdType::Float,
            XsdType::String,
            XsdType::Date,
            XsdType::DateTime,
            XsdType::Duration,
            XsdType::AnyUri,
            XsdType::HexBinary,
            XsdType::Base64Binary,
            XsdType::Language,
            XsdType::NonNegativeInteger,
            XsdType::PositiveInteger,
        ];
        for t in types {
            let iri = LiteralParser::xsd_iri(t);
            let recovered = LiteralParser::from_xsd_iri(iri);
            assert_eq!(recovered, Some(t), "round-trip failed for {t:?}");
        }
    }

    #[test]
    fn test_from_xsd_iri_unknown() {
        assert_eq!(
            LiteralParser::from_xsd_iri("http://example.org/unknown"),
            None
        );
    }

    // --- Boolean normalization ---

    #[test]
    fn test_normalize_boolean_true() {
        assert_eq!(LiteralParser::normalize_boolean("true").expect("normalization should succeed"), "true");
    }

    #[test]
    fn test_normalize_boolean_false() {
        assert_eq!(LiteralParser::normalize_boolean("false").expect("normalization should succeed"), "false");
    }

    #[test]
    fn test_normalize_boolean_one() {
        assert_eq!(LiteralParser::normalize_boolean("1").expect("normalization should succeed"), "true");
    }

    #[test]
    fn test_normalize_boolean_zero() {
        assert_eq!(LiteralParser::normalize_boolean("0").expect("normalization should succeed"), "false");
    }

    #[test]
    fn test_normalize_boolean_invalid() {
        assert!(LiteralParser::normalize_boolean("yes").is_err());
        assert!(LiteralParser::normalize_boolean("True").is_err());
        assert!(LiteralParser::normalize_boolean("2").is_err());
    }

    // --- Integer normalization ---

    #[test]
    fn test_normalize_integer_simple() {
        assert_eq!(LiteralParser::normalize_integer("42").expect("normalization should succeed"), "42");
    }

    #[test]
    fn test_normalize_integer_leading_zeros() {
        assert_eq!(LiteralParser::normalize_integer("007").expect("normalization should succeed"), "7");
    }

    #[test]
    fn test_normalize_integer_negative() {
        assert_eq!(LiteralParser::normalize_integer("-5").expect("normalization should succeed"), "-5");
    }

    #[test]
    fn test_normalize_integer_plus_sign() {
        assert_eq!(LiteralParser::normalize_integer("+10").expect("normalization should succeed"), "10");
    }

    #[test]
    fn test_normalize_integer_zero() {
        assert_eq!(LiteralParser::normalize_integer("0").expect("normalization should succeed"), "0");
    }

    #[test]
    fn test_normalize_integer_negative_leading_zeros() {
        assert_eq!(LiteralParser::normalize_integer("-007").expect("normalization should succeed"), "-7");
    }

    #[test]
    fn test_normalize_integer_invalid() {
        assert!(LiteralParser::normalize_integer("3.14").is_err());
        assert!(LiteralParser::normalize_integer("abc").is_err());
        assert!(LiteralParser::normalize_integer("").is_err());
    }

    // --- Decimal normalization ---

    #[test]
    fn test_normalize_decimal_simple() {
        assert_eq!(LiteralParser::normalize_decimal("3.14").expect("normalization should succeed"), "3.14");
    }

    #[test]
    fn test_normalize_decimal_trailing_zeros() {
        assert_eq!(
            LiteralParser::normalize_decimal("3.1400").expect("normalization should succeed"),
            "3.14"
        );
    }

    #[test]
    fn test_normalize_decimal_leading_zeros() {
        assert_eq!(
            LiteralParser::normalize_decimal("003.14").expect("normalization should succeed"),
            "3.14"
        );
    }

    #[test]
    fn test_normalize_decimal_zero() {
        assert_eq!(LiteralParser::normalize_decimal("0.0").expect("normalization should succeed"), "0.0");
    }

    #[test]
    fn test_normalize_decimal_negative() {
        assert_eq!(
            LiteralParser::normalize_decimal("-1.5").expect("normalization should succeed"),
            "-1.5"
        );
    }

    #[test]
    fn test_normalize_decimal_invalid() {
        assert!(LiteralParser::normalize_decimal("42").is_err());
        assert!(LiteralParser::normalize_decimal("abc").is_err());
    }

    // --- Double normalization ---

    #[test]
    fn test_normalize_double_simple() {
        let result = LiteralParser::normalize_double("1.5E2").expect("normalization should succeed");
        // Should be scientific notation
        assert!(result.contains('E'));
    }

    #[test]
    fn test_normalize_double_plain() {
        let result = LiteralParser::normalize_double("150.0").expect("normalization should succeed");
        assert!(result.contains('E') || result.contains('e') || result == "150");
    }

    #[test]
    fn test_normalize_double_nan() {
        assert_eq!(LiteralParser::normalize_double("NaN").expect("normalization should succeed"), "NaN");
    }

    #[test]
    fn test_normalize_double_inf() {
        assert_eq!(LiteralParser::normalize_double("inf").expect("normalization should succeed"), "INF");
    }

    #[test]
    fn test_normalize_double_negative_inf() {
        assert_eq!(LiteralParser::normalize_double("-inf").expect("normalization should succeed"), "-INF");
    }

    #[test]
    fn test_normalize_double_invalid() {
        assert!(LiteralParser::normalize_double("abc").is_err());
    }

    // --- Date validation ---

    #[test]
    fn test_normalize_date_valid() {
        assert_eq!(
            LiteralParser::normalize_date("2024-01-15").expect("normalization should succeed"),
            "2024-01-15"
        );
    }

    #[test]
    fn test_normalize_date_with_timezone_z() {
        assert_eq!(
            LiteralParser::normalize_date("2024-01-15Z").expect("normalization should succeed"),
            "2024-01-15Z"
        );
    }

    #[test]
    fn test_normalize_date_leap_day() {
        assert!(LiteralParser::normalize_date("2024-02-29").is_ok());
    }

    #[test]
    fn test_normalize_date_non_leap_year_feb29() {
        assert!(LiteralParser::normalize_date("2023-02-29").is_err());
    }

    #[test]
    fn test_normalize_date_invalid_month() {
        assert!(LiteralParser::normalize_date("2024-13-01").is_err());
    }

    #[test]
    fn test_normalize_date_invalid_day() {
        assert!(LiteralParser::normalize_date("2024-04-31").is_err());
    }

    #[test]
    fn test_normalize_date_invalid_format() {
        assert!(LiteralParser::normalize_date("24-1-5").is_err());
    }

    // --- DateTime validation ---

    #[test]
    fn test_normalize_datetime_valid() {
        let result = LiteralParser::normalize_datetime("2024-01-15T10:30:00").expect("normalization should succeed");
        assert_eq!(result, "2024-01-15T10:30:00");
    }

    #[test]
    fn test_normalize_datetime_with_z() {
        let result = LiteralParser::normalize_datetime("2024-01-15T10:30:00Z").expect("normalization should succeed");
        assert_eq!(result, "2024-01-15T10:30:00Z");
    }

    #[test]
    fn test_normalize_datetime_with_offset() {
        let result = LiteralParser::normalize_datetime("2024-01-15T10:30:00+09:00").expect("normalization should succeed");
        assert_eq!(result, "2024-01-15T10:30:00+09:00");
    }

    #[test]
    fn test_normalize_datetime_with_fractional() {
        let result = LiteralParser::normalize_datetime("2024-01-15T10:30:00.500Z").expect("normalization should succeed");
        assert_eq!(result, "2024-01-15T10:30:00.500Z");
    }

    #[test]
    fn test_normalize_datetime_invalid_hour() {
        assert!(LiteralParser::normalize_datetime("2024-01-15T25:00:00").is_err());
    }

    #[test]
    fn test_normalize_datetime_invalid_minute() {
        assert!(LiteralParser::normalize_datetime("2024-01-15T10:60:00").is_err());
    }

    #[test]
    fn test_normalize_datetime_no_t() {
        assert!(LiteralParser::normalize_datetime("2024-01-15 10:30:00").is_err());
    }

    // --- HexBinary validation ---

    #[test]
    fn test_validate_hex_binary_valid() {
        assert_eq!(
            LiteralParser::validate_hex_binary("deadbeef").expect("construction should succeed"),
            "DEADBEEF"
        );
    }

    #[test]
    fn test_validate_hex_binary_uppercase() {
        assert_eq!(
            LiteralParser::validate_hex_binary("DEADBEEF").expect("construction should succeed"),
            "DEADBEEF"
        );
    }

    #[test]
    fn test_validate_hex_binary_empty() {
        assert_eq!(LiteralParser::validate_hex_binary("").expect("construction should succeed"), "");
    }

    #[test]
    fn test_validate_hex_binary_odd_length() {
        assert!(LiteralParser::validate_hex_binary("abc").is_err());
    }

    #[test]
    fn test_validate_hex_binary_invalid_chars() {
        assert!(LiteralParser::validate_hex_binary("gggg").is_err());
    }

    // --- detect_type heuristic ---

    #[test]
    fn test_detect_type_boolean_true() {
        assert_eq!(LiteralParser::detect_type("true"), XsdType::Boolean);
    }

    #[test]
    fn test_detect_type_boolean_false() {
        assert_eq!(LiteralParser::detect_type("false"), XsdType::Boolean);
    }

    #[test]
    fn test_detect_type_boolean_one() {
        assert_eq!(LiteralParser::detect_type("1"), XsdType::Boolean);
    }

    #[test]
    fn test_detect_type_boolean_zero() {
        assert_eq!(LiteralParser::detect_type("0"), XsdType::Boolean);
    }

    #[test]
    fn test_detect_type_integer() {
        assert_eq!(LiteralParser::detect_type("42"), XsdType::Integer);
    }

    #[test]
    fn test_detect_type_negative_integer() {
        assert_eq!(LiteralParser::detect_type("-100"), XsdType::Integer);
    }

    #[test]
    fn test_detect_type_decimal() {
        assert_eq!(LiteralParser::detect_type("3.14"), XsdType::Decimal);
    }

    #[test]
    fn test_detect_type_double_scientific() {
        assert_eq!(LiteralParser::detect_type("1.5e10"), XsdType::Double);
    }

    #[test]
    fn test_detect_type_date() {
        assert_eq!(LiteralParser::detect_type("2024-01-15"), XsdType::Date);
    }

    #[test]
    fn test_detect_type_string() {
        assert_eq!(LiteralParser::detect_type("hello world"), XsdType::String);
    }

    #[test]
    fn test_detect_type_uri() {
        assert_eq!(
            LiteralParser::detect_type("http://example.org/foo"),
            XsdType::AnyUri
        );
    }

    // --- parse() with explicit datatype IRI ---

    #[test]
    fn test_parse_integer_with_iri() {
        let result = LiteralParser::parse("042", "http://www.w3.org/2001/XMLSchema#integer");
        assert!(result.is_ok());
        let parsed = result.expect("should have value");
        assert_eq!(parsed.normalized, "42");
        assert_eq!(parsed.datatype, XsdType::Integer);
    }

    #[test]
    fn test_parse_unknown_datatype() {
        let result = LiteralParser::parse("foo", "http://example.org/unknown");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LiteralError::UnknownDatatype(_)));
    }

    #[test]
    fn test_parse_boolean_with_iri() {
        let result = LiteralParser::parse("1", "http://www.w3.org/2001/XMLSchema#boolean");
        assert!(result.is_ok());
        assert_eq!(result.expect("should have value").normalized, "true");
    }

    // --- NonNegativeInteger and PositiveInteger ---

    #[test]
    fn test_normalize_nonnegative_integer_zero() {
        assert_eq!(
            LiteralParser::normalize_non_negative_integer("0").expect("normalization should succeed"),
            "0"
        );
    }

    #[test]
    fn test_normalize_nonnegative_integer_positive() {
        assert_eq!(
            LiteralParser::normalize_non_negative_integer("42").expect("normalization should succeed"),
            "42"
        );
    }

    #[test]
    fn test_normalize_nonnegative_integer_negative_fails() {
        assert!(LiteralParser::normalize_non_negative_integer("-1").is_err());
    }

    #[test]
    fn test_normalize_positive_integer_one() {
        assert_eq!(
            LiteralParser::normalize_positive_integer("1").expect("normalization should succeed"),
            "1"
        );
    }

    #[test]
    fn test_normalize_positive_integer_zero_fails() {
        assert!(LiteralParser::normalize_positive_integer("0").is_err());
    }

    #[test]
    fn test_normalize_positive_integer_negative_fails() {
        assert!(LiteralParser::normalize_positive_integer("-5").is_err());
    }

    // --- Duration ---

    #[test]
    fn test_normalize_duration_years() {
        assert_eq!(
            LiteralParser::normalize_duration("P1Y").expect("normalization should succeed"),
            "P1Y"
        );
    }

    #[test]
    fn test_normalize_duration_complex() {
        assert_eq!(
            LiteralParser::normalize_duration("P1Y2M3DT4H5M6S").expect("normalization should succeed"),
            "P1Y2M3DT4H5M6S"
        );
    }

    #[test]
    fn test_normalize_duration_negative() {
        assert_eq!(
            LiteralParser::normalize_duration("-P1Y").expect("normalization should succeed"),
            "-P1Y"
        );
    }

    #[test]
    fn test_normalize_duration_invalid() {
        assert!(LiteralParser::normalize_duration("1Y").is_err());
    }
}
