//! RDF/XML wrapper for integration with OxiRS types
//!
//! This module provides a temporary wrapper to bridge the gap between
//! the RDF/XML parser implementation and OxiRS native types.

use crate::model::{Triple, Quad};
use crate::OxirsError;
use crate::rdfxml::parser::RdfXmlParser;
use std::io::Read;

/// Parse RDF/XML data and convert to OxiRS quads
pub fn parse_rdfxml<R: Read>(
    reader: R,
    base_iri: Option<&str>,
    lenient: bool,
) -> Result<Vec<Quad>, OxirsError> {
    let mut parser = RdfXmlParser::new();
    
    // Configure parser
    if let Some(base) = base_iri {
        parser = parser
            .with_base_iri(base)
            .map_err(|e| OxirsError::Parse(format!("Invalid base IRI: {}", e)))?;
    }
    
    if lenient {
        parser = parser.lenient();
    }
    
    // Parse and collect quads
    let mut quads = Vec::new();
    for result in parser.for_reader(reader) {
        match result {
            Ok(triple) => {
                // The RDF/XML parser returns our native Triple type
                quads.push(Quad::from_triple(triple));
            }
            Err(e) => {
                if lenient {
                    tracing::warn!("RDF/XML parse error (ignored): {}", e);
                    continue;
                } else {
                    return Err(OxirsError::Parse(format!("RDF/XML parse error: {}", e)));
                }
            }
        }
    }
    
    Ok(quads)
}