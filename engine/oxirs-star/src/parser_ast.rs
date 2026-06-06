//! AST node types and format enumerations for RDF-star parsing.
//!
//! This module contains the [`StarFormat`] enumeration and its `FromStr`
//! implementation, representing the supported RDF-star serialization formats.

use std::str::FromStr;

use crate::StarError;

/// RDF-star format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StarFormat {
    /// Turtle-star format
    TurtleStar,
    /// N-Triples-star format
    NTriplesStar,
    /// TriG-star format (named graphs)
    TrigStar,
    /// N-Quads-star format
    NQuadsStar,
    /// JSON-LD-star format
    JsonLdStar,
}

impl FromStr for StarFormat {
    type Err = StarError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "turtle-star" | "ttls" => Ok(StarFormat::TurtleStar),
            "ntriples-star" | "nts" => Ok(StarFormat::NTriplesStar),
            "trig-star" | "trigs" => Ok(StarFormat::TrigStar),
            "nquads-star" | "nqs" => Ok(StarFormat::NQuadsStar),
            "json-ld-star" | "jsonld-star" | "jlds" => Ok(StarFormat::JsonLdStar),
            _ => Err(StarError::parse_error(format!("Unknown format: {s}"))),
        }
    }
}
