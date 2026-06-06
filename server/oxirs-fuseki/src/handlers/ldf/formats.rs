//! Response serialization with HTTP content negotiation.
//!
//! TPF servers traditionally support multiple RDF serializations. This module
//! handles `Accept` header negotiation and emits Turtle, N-Triples or JSON-LD
//! representations of a [`TpfResponse`].

use super::response::TpfResponse;
use axum::http::HeaderMap;

/// Negotiable response formats for the LDF endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdfFormat {
    Turtle,
    JsonLd,
    NTriples,
}

impl LdfFormat {
    /// Returns the IANA media type associated with the format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            LdfFormat::Turtle => "text/turtle",
            LdfFormat::JsonLd => "application/ld+json",
            LdfFormat::NTriples => "application/n-triples",
        }
    }
}

/// Choose a response format from the request's `Accept` header.
///
/// JSON-LD and N-Triples are matched explicitly; Turtle is the default both
/// for browsers (which advertise `text/html`) and unrecognised requests.
pub fn negotiate_format(headers: &HeaderMap) -> LdfFormat {
    let accept = headers
        .get("accept")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    if accept.contains("application/ld+json") {
        return LdfFormat::JsonLd;
    }
    if accept.contains("application/n-triples") {
        return LdfFormat::NTriples;
    }
    LdfFormat::Turtle
}

/// Serialize a TPF response using the requested [`LdfFormat`].
pub fn serialize_response(response: &TpfResponse, format: LdfFormat) -> String {
    match format {
        LdfFormat::Turtle => serialize_turtle(response),
        LdfFormat::NTriples => serialize_ntriples(response),
        LdfFormat::JsonLd => serialize_jsonld(response),
    }
}

fn serialize_turtle(response: &TpfResponse) -> String {
    let mut out = String::new();
    out.push_str("@prefix hydra: <http://www.w3.org/ns/hydra/core#> .\n");
    out.push_str("@prefix void: <http://rdfs.org/ns/void#> .\n\n");
    for triple in &response.triples {
        out.push_str(&format!(
            "<{}> <{}> {} .\n",
            triple.subject, triple.predicate, triple.object
        ));
    }
    out.push_str(&format!(
        "<{}> hydra:totalItems {} .\n",
        response.fragment_uri, response.metadata.total_count
    ));
    out
}

fn serialize_ntriples(response: &TpfResponse) -> String {
    let mut out = String::new();
    for triple in &response.triples {
        out.push_str(&format!(
            "<{}> <{}> {} .\n",
            triple.subject, triple.predicate, triple.object
        ));
    }
    out
}

fn serialize_jsonld(response: &TpfResponse) -> String {
    format!(
        r#"{{"@context": "http://www.w3.org/ns/hydra/context.jsonld", "@id": "{}", "totalItems": {}, "triples": []}}"#,
        response.fragment_uri, response.metadata.total_count
    )
}
