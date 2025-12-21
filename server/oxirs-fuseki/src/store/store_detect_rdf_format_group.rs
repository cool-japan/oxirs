//! # Store - detect_rdf_format_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Detect RDF format from URL and content-type
    pub(super) fn detect_rdf_format(
        &self,
        url: &str,
        content_type: Option<&str>,
    ) -> FusekiResult<CoreRdfFormat> {
        if let Some(ct) = content_type {
            let ct_lower = ct.to_lowercase();
            if ct_lower.contains("turtle") || ct_lower.contains("text/turtle") {
                return Ok(CoreRdfFormat::Turtle);
            } else if ct_lower.contains("rdf+xml") || ct_lower.contains("application/rdf+xml") {
                return Ok(CoreRdfFormat::RdfXml);
            } else if ct_lower.contains("n-triples") || ct_lower.contains("application/n-triples") {
                return Ok(CoreRdfFormat::NTriples);
            } else if ct_lower.contains("n-quads") || ct_lower.contains("application/n-quads") {
                return Ok(CoreRdfFormat::NQuads);
            } else if ct_lower.contains("trig") || ct_lower.contains("application/trig") {
                return Ok(CoreRdfFormat::TriG);
            } else if ct_lower.contains("n3") || ct_lower.contains("text/n3") {
                return Ok(CoreRdfFormat::Turtle);
            }
        }
        let url_lower = url.to_lowercase();
        if url_lower.ends_with(".ttl") || url_lower.ends_with(".turtle") {
            Ok(CoreRdfFormat::Turtle)
        } else if url_lower.ends_with(".rdf")
            || url_lower.ends_with(".xml")
            || url_lower.ends_with(".owl")
        {
            Ok(CoreRdfFormat::RdfXml)
        } else if url_lower.ends_with(".nt") || url_lower.ends_with(".ntriples") {
            Ok(CoreRdfFormat::NTriples)
        } else if url_lower.ends_with(".nq") || url_lower.ends_with(".nquads") {
            Ok(CoreRdfFormat::NQuads)
        } else if url_lower.ends_with(".trig") {
            Ok(CoreRdfFormat::TriG)
        } else if url_lower.ends_with(".n3") {
            Ok(CoreRdfFormat::Turtle)
        } else {
            warn!(
                "Could not detect RDF format for '{}', defaulting to Turtle",
                url
            );
            Ok(CoreRdfFormat::Turtle)
        }
    }
}
