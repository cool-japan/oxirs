//! Random RDF data generation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxirs_core::format::RdfFormat;
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use scirs2_core::Rng;
use std::error::Error;

/// Generate random RDF triples
pub(super) fn generate_random_rdf<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
    let mut quads = Vec::with_capacity(count);
    let predicates = [
        "name",
        "age",
        "email",
        "address",
        "phone",
        "description",
        "value",
        "status",
        "created",
        "modified",
    ];
    for _i in 0..count {
        let subject_id = rng.random_range(0..count.max(100));
        let predicate_idx = rng.random_range(0..predicates.len());
        let object_value = rng.random_range(0..10000);
        let subject = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/resource/{}", subject_id))
                .expect("Valid IRI"),
        );
        let predicate = NamedNode::new(format!("http://example.org/{}", predicates[predicate_idx]))
            .expect("Valid IRI");
        let object = if rng.random_bool(0.5) {
            Term::Literal(Literal::new_simple_literal(format!(
                "value_{}",
                object_value
            )))
        } else {
            let xsd_integer =
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("Valid IRI");
            Term::Literal(Literal::new_typed_literal(
                object_value.to_string(),
                xsd_integer,
            ))
        };
        quads.push(Quad::new(
            subject,
            predicate,
            object,
            GraphName::DefaultGraph,
        ));
    }
    quads
}

/// Generate graph structure (nodes and edges)
pub(super) fn generate_graph_structure<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
    let mut quads = Vec::with_capacity(count);
    let node_count = (count as f64).sqrt() as usize;
    let edge_types = ["connected", "linked", "related", "parent", "child"];
    for _i in 0..count {
        let source_id = rng.random_range(0..node_count);
        let target_id = rng.random_range(0..node_count);
        let edge_type_idx = rng.random_range(0..edge_types.len());
        let subject = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/node/{}", source_id)).expect("Valid IRI"),
        );
        let predicate = NamedNode::new(format!("http://example.org/{}", edge_types[edge_type_idx]))
            .expect("Valid IRI");
        let object = Term::NamedNode(
            NamedNode::new(format!("http://example.org/node/{}", target_id)).expect("Valid IRI"),
        );
        quads.push(Quad::new(
            subject,
            predicate,
            object,
            GraphName::DefaultGraph,
        ));
    }
    quads
}

/// Parse RDF format string into RdfFormat enum
pub(super) fn parse_rdf_format(format: &str) -> Result<RdfFormat, Box<dyn Error>> {
    match format.to_lowercase().as_str() {
        "turtle" | "ttl" => Ok(RdfFormat::Turtle),
        "ntriples" | "nt" => Ok(RdfFormat::NTriples),
        "nquads" | "nq" => Ok(RdfFormat::NQuads),
        "trig" => Ok(RdfFormat::TriG),
        "rdfxml" | "rdf" | "xml" => Ok(RdfFormat::RdfXml),
        "jsonld" | "json-ld" | "json" => Ok(RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        }),
        "n3" => Ok(RdfFormat::N3),
        _ => Err(format!("Unsupported RDF format: {}", format).into()),
    }
}
