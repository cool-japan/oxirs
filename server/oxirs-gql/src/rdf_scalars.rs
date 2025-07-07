//! Custom RDF scalar types for GraphQL
//!
//! This module provides RDF-specific scalar types like IRI, Literal, DateTime, etc.

use crate::ast::Value;
use crate::types::ScalarType;
use anyhow::{anyhow, Result};
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// IRI (Internationalized Resource Identifier) scalar type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IRI {
    pub value: String,
}

impl IRI {
    pub fn new(value: String) -> Result<Self> {
        // Basic IRI validation
        if value.is_empty() {
            return Err(anyhow!("IRI cannot be empty"));
        }

        // Check for valid IRI format (basic validation)
        if !value.contains(':') {
            return Err(anyhow!("IRI must contain a scheme"));
        }

        Ok(Self { value })
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for IRI {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl FromStr for IRI {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::new(s.to_string())
    }
}

/// RDF Literal with optional datatype and language tag
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Literal {
    pub value: String,
    pub datatype: Option<IRI>,
    pub language: Option<String>,
}

impl Literal {
    pub fn new(value: String) -> Self {
        Self {
            value,
            datatype: None,
            language: None,
        }
    }

    pub fn with_datatype(mut self, datatype: IRI) -> Self {
        self.datatype = Some(datatype);
        self.language = None; // Language and datatype are mutually exclusive
        self
    }

    pub fn with_language(mut self, language: String) -> Self {
        self.language = Some(language);
        self.datatype = None; // Language and datatype are mutually exclusive
        self
    }

    pub fn is_language_tagged(&self) -> bool {
        self.language.is_some()
    }

    pub fn is_typed(&self) -> bool {
        self.datatype.is_some()
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.value)?;

        if let Some(ref lang) = self.language {
            write!(f, "@{}", lang)?;
        } else if let Some(ref datatype) = self.datatype {
            write!(f, "^^<{}>", datatype)?;
        }

        Ok(())
    }
}

/// Duration scalar type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Duration {
    pub seconds: i64,
    pub nanoseconds: u32,
}

impl Duration {
    pub fn new(seconds: i64, nanoseconds: u32) -> Self {
        Self {
            seconds,
            nanoseconds,
        }
    }

    pub fn from_seconds(seconds: i64) -> Self {
        Self {
            seconds,
            nanoseconds: 0,
        }
    }

    pub fn from_millis(millis: i64) -> Self {
        Self {
            seconds: millis / 1000,
            nanoseconds: ((millis % 1000) * 1_000_000) as u32,
        }
    }

    pub fn total_seconds(&self) -> f64 {
        self.seconds as f64 + (self.nanoseconds as f64 / 1_000_000_000.0)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PT{}S", self.total_seconds())
    }
}

/// GeoLocation scalar type for spatial data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
}

impl GeoLocation {
    pub fn new(latitude: f64, longitude: f64) -> Result<Self> {
        if latitude < -90.0 || latitude > 90.0 {
            return Err(anyhow!("Latitude must be between -90 and 90 degrees"));
        }

        if longitude < -180.0 || longitude > 180.0 {
            return Err(anyhow!("Longitude must be between -180 and 180 degrees"));
        }

        Ok(Self {
            latitude,
            longitude,
            altitude: None,
        })
    }

    pub fn with_altitude(mut self, altitude: f64) -> Self {
        self.altitude = Some(altitude);
        self
    }
}

impl fmt::Display for GeoLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(alt) = self.altitude {
            write!(f, "POINT Z({} {} {})", self.longitude, self.latitude, alt)
        } else {
            write!(f, "POINT({} {})", self.longitude, self.latitude)
        }
    }
}

/// Language-tagged string
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LangString {
    pub value: String,
    pub language: String,
}

impl LangString {
    pub fn new(value: String, language: String) -> Self {
        Self { value, language }
    }
}

impl fmt::Display for LangString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"@{}", self.value, self.language)
    }
}

/// RDF scalar types factory
pub struct RdfScalars;

impl RdfScalars {
    /// IRI scalar type
    pub fn iri() -> ScalarType {
        ScalarType::new("IRI".to_string())
            .with_description(
                "The `IRI` scalar type represents an Internationalized Resource Identifier."
                    .to_string(),
            )
            .with_serializer(|v| match v {
                Value::StringValue(s) => {
                    IRI::new(s.clone())?;
                    Ok(Value::StringValue(s.clone()))
                }
                _ => Err(anyhow!("Cannot serialize {:?} as IRI", v)),
            })
            .with_value_parser(|v| match v {
                Value::StringValue(s) => {
                    IRI::new(s.clone())?;
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse {:?} as IRI", v)),
            })
            .with_literal_parser(|v| match v {
                Value::StringValue(s) => {
                    IRI::new(s.clone())?;
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse literal {:?} as IRI", v)),
            })
    }

    /// Literal scalar type
    pub fn literal() -> ScalarType {
        ScalarType::new("Literal".to_string())
            .with_description("The `Literal` scalar type represents an RDF literal with optional datatype or language tag.".to_string())
            .with_serializer(|v| match v {
                Value::StringValue(_) => Ok(v.clone()),
                Value::ObjectValue(_) => Ok(v.clone()), // For structured literals
                _ => Err(anyhow!("Cannot serialize {:?} as Literal", v)),
            })
            .with_value_parser(|v| match v {
                Value::StringValue(_) => Ok(v.clone()),
                Value::ObjectValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse {:?} as Literal", v)),
            })
            .with_literal_parser(|v| match v {
                Value::StringValue(_) => Ok(v.clone()),
                Value::ObjectValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse literal {:?} as Literal", v)),
            })
    }

    /// DateTime scalar type
    pub fn datetime() -> ScalarType {
        ScalarType::new("DateTime".to_string())
            .with_description(
                "The `DateTime` scalar type represents date and time with timezone support."
                    .to_string(),
            )
            .with_serializer(|v| match v {
                Value::StringValue(s) => {
                    // Validate datetime format
                    DateTime::parse_from_rfc3339(s)
                        .map_err(|e| anyhow!("Invalid datetime format: {}", e))?;
                    Ok(Value::StringValue(s.clone()))
                }
                _ => Err(anyhow!("Cannot serialize {:?} as DateTime", v)),
            })
            .with_value_parser(|v| match v {
                Value::StringValue(s) => {
                    DateTime::parse_from_rfc3339(s)
                        .map_err(|e| anyhow!("Invalid datetime format: {}", e))?;
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse {:?} as DateTime", v)),
            })
            .with_literal_parser(|v| match v {
                Value::StringValue(s) => {
                    DateTime::parse_from_rfc3339(s)
                        .map_err(|e| anyhow!("Invalid datetime format: {}", e))?;
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse literal {:?} as DateTime", v)),
            })
    }

    /// Duration scalar type
    pub fn duration() -> ScalarType {
        ScalarType::new("Duration".to_string())
            .with_description("The `Duration` scalar type represents a time duration.".to_string())
            .with_serializer(|v| match v {
                Value::StringValue(s) => {
                    // Basic ISO 8601 duration validation
                    if !s.starts_with('P') {
                        return Err(anyhow!("Duration must start with 'P'"));
                    }
                    Ok(Value::StringValue(s.clone()))
                }
                Value::IntValue(i) => Ok(Value::StringValue(format!("PT{}S", i))),
                Value::FloatValue(f) => Ok(Value::StringValue(format!("PT{}S", f))),
                _ => Err(anyhow!("Cannot serialize {:?} as Duration", v)),
            })
            .with_value_parser(|v| match v {
                Value::StringValue(s) => {
                    if !s.starts_with('P') {
                        return Err(anyhow!("Duration must start with 'P'"));
                    }
                    Ok(v.clone())
                }
                Value::IntValue(_) => Ok(v.clone()),
                Value::FloatValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse {:?} as Duration", v)),
            })
            .with_literal_parser(|v| match v {
                Value::StringValue(s) => {
                    if !s.starts_with('P') {
                        return Err(anyhow!("Duration must start with 'P'"));
                    }
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse literal {:?} as Duration", v)),
            })
    }

    /// GeoLocation scalar type
    pub fn geolocation() -> ScalarType {
        ScalarType::new("GeoLocation".to_string())
            .with_description(
                "The `GeoLocation` scalar type represents geographic coordinates.".to_string(),
            )
            .with_serializer(|v| match v {
                Value::ObjectValue(obj) => {
                    // Validate required fields
                    if !obj.contains_key("latitude") || !obj.contains_key("longitude") {
                        return Err(anyhow!("GeoLocation must have latitude and longitude"));
                    }
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot serialize {:?} as GeoLocation", v)),
            })
            .with_value_parser(|v| match v {
                Value::ObjectValue(obj) => {
                    if !obj.contains_key("latitude") || !obj.contains_key("longitude") {
                        return Err(anyhow!("GeoLocation must have latitude and longitude"));
                    }
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse {:?} as GeoLocation", v)),
            })
            .with_literal_parser(|v| match v {
                Value::ObjectValue(obj) => {
                    if !obj.contains_key("latitude") || !obj.contains_key("longitude") {
                        return Err(anyhow!("GeoLocation must have latitude and longitude"));
                    }
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse literal {:?} as GeoLocation", v)),
            })
    }

    /// Language-tagged string scalar type
    pub fn lang_string() -> ScalarType {
        ScalarType::new("LangString".to_string())
            .with_description(
                "The `LangString` scalar type represents a string with a language tag.".to_string(),
            )
            .with_serializer(|v| match v {
                Value::ObjectValue(obj) => {
                    if !obj.contains_key("value") || !obj.contains_key("language") {
                        return Err(anyhow!("LangString must have value and language"));
                    }
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot serialize {:?} as LangString", v)),
            })
            .with_value_parser(|v| match v {
                Value::ObjectValue(obj) => {
                    if !obj.contains_key("value") || !obj.contains_key("language") {
                        return Err(anyhow!("LangString must have value and language"));
                    }
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse {:?} as LangString", v)),
            })
            .with_literal_parser(|v| match v {
                Value::ObjectValue(obj) => {
                    if !obj.contains_key("value") || !obj.contains_key("language") {
                        return Err(anyhow!("LangString must have value and language"));
                    }
                    Ok(v.clone())
                }
                _ => Err(anyhow!("Cannot parse literal {:?} as LangString", v)),
            })
    }
}
