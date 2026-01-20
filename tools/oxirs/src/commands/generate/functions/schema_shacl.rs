//! SHACL schema integration for RDF generation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::*;
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use scirs2_core::Rng;
use std::error::Error;

/// Parse SHACL shapes from file (simplified implementation)
pub(super) fn parse_shacl_shapes(
    _path: &std::path::PathBuf,
    _ctx: &crate::cli::CliContext,
) -> Result<Vec<ShaclShape>, Box<dyn Error>> {
    let person_shape = ShaclShape {
        target_class: Some("http://xmlns.com/foaf/0.1/Person".to_string()),
        properties: vec![
            PropertyConstraint {
                path: "http://xmlns.com/foaf/0.1/name".to_string(),
                min_count: Some(1),
                max_count: Some(1),
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                pattern: None,
                min_length: Some(1),
                max_length: Some(100),
                min_inclusive: None,
                max_inclusive: None,
                node_kind: Some("Literal".to_string()),
                class: None,
            },
            PropertyConstraint {
                path: "http://xmlns.com/foaf/0.1/age".to_string(),
                min_count: Some(0),
                max_count: Some(1),
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                pattern: None,
                min_length: None,
                max_length: None,
                min_inclusive: Some("0".to_string()),
                max_inclusive: Some("150".to_string()),
                node_kind: Some("Literal".to_string()),
                class: None,
            },
            PropertyConstraint {
                path: "http://xmlns.com/foaf/0.1/email".to_string(),
                min_count: Some(1),
                max_count: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                pattern: Some("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$".to_string()),
                min_length: None,
                max_length: None,
                min_inclusive: None,
                max_inclusive: None,
                node_kind: Some("Literal".to_string()),
                class: None,
            },
        ],
    };
    Ok(vec![person_shape])
}

/// Generate RDF data conforming to SHACL shapes
pub(super) fn generate_from_shapes<R: Rng>(
    rng: &mut R,
    shapes: &[ShaclShape],
    count: usize,
) -> Result<Vec<Quad>, Box<dyn Error>> {
    let mut quads = Vec::new();
    for i in 0..count {
        for shape in shapes {
            let instance_uri = if let Some(target_class) = &shape.target_class {
                let class_name = target_class.split('/').next_back().unwrap_or("instance");
                Subject::NamedNode(
                    NamedNode::new(format!("http://example.org/{}/{}", class_name, i))
                        .expect("Valid IRI"),
                )
            } else {
                Subject::NamedNode(
                    NamedNode::new(format!("http://example.org/instance/{}", i))
                        .expect("Valid IRI"),
                )
            };
            if let Some(target_class) = &shape.target_class {
                quads.push(Quad::new(
                    instance_uri.clone(),
                    NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                        .expect("Valid IRI"),
                    Term::NamedNode(NamedNode::new(target_class).expect("Valid IRI")),
                    GraphName::DefaultGraph,
                ));
            }
            for prop in &shape.properties {
                let min = prop.min_count.unwrap_or(0) as usize;
                let max = prop.max_count.unwrap_or(3) as usize;
                let num_values = if max == min {
                    min
                } else {
                    rng.random_range(min..=max.min(min + 5))
                };
                for _ in 0..num_values {
                    let value = generate_property_value(rng, prop)?;
                    quads.push(Quad::new(
                        instance_uri.clone(),
                        NamedNode::new(&prop.path).expect("Valid IRI"),
                        value,
                        GraphName::DefaultGraph,
                    ));
                }
            }
        }
    }
    Ok(quads)
}

/// Generate a property value conforming to constraints
pub(super) fn generate_property_value<R: Rng>(
    rng: &mut R,
    constraint: &PropertyConstraint,
) -> Result<Term, Box<dyn Error>> {
    let node_kind = constraint.node_kind.as_deref().unwrap_or("Literal");
    match node_kind {
        "IRI" | "NamedNode" => {
            if let Some(class_iri) = &constraint.class {
                Ok(Term::NamedNode(
                    NamedNode::new(class_iri).expect("Valid IRI"),
                ))
            } else {
                let id = rng.random_range(0..1000);
                Ok(Term::NamedNode(
                    NamedNode::new(format!("http://example.org/resource/{}", id))
                        .expect("Valid IRI"),
                ))
            }
        }
        "Literal" => {
            let datatype = constraint
                .datatype
                .as_deref()
                .unwrap_or("http://www.w3.org/2001/XMLSchema#string");
            let value_str = match datatype {
                "http://www.w3.org/2001/XMLSchema#string" => generate_string_value(rng, constraint),
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int" => generate_integer_value(rng, constraint),
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double" => {
                    generate_decimal_value(rng, constraint)
                }
                "http://www.w3.org/2001/XMLSchema#boolean" => if rng.random_range(0..2) == 0 {
                    "true"
                } else {
                    "false"
                }
                .to_string(),
                "http://www.w3.org/2001/XMLSchema#date" => {
                    let year = rng.random_range(1900..=2024);
                    let month = rng.random_range(1..=12);
                    let day = rng.random_range(1..=28);
                    format!("{:04}-{:02}-{:02}", year, month, day)
                }
                "http://www.w3.org/2001/XMLSchema#dateTime" => {
                    let year = rng.random_range(2000..=2024);
                    let month = rng.random_range(1..=12);
                    let day = rng.random_range(1..=28);
                    let hour = rng.random_range(0..=23);
                    let minute = rng.random_range(0..=59);
                    let second = rng.random_range(0..=59);
                    format!(
                        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                        year, month, day, hour, minute, second
                    )
                }
                _ => generate_string_value(rng, constraint),
            };
            Ok(Term::Literal(Literal::new_typed_literal(
                value_str,
                NamedNode::new(datatype).expect("Valid IRI"),
            )))
        }
        _ => Ok(Term::Literal(Literal::new_simple_literal(
            generate_string_value(rng, constraint),
        ))),
    }
}

/// Generate string value conforming to constraints
pub(super) fn generate_string_value<R: Rng>(
    rng: &mut R,
    constraint: &PropertyConstraint,
) -> String {
    if let Some(pattern) = &constraint.pattern {
        if pattern.contains("@") {
            let names = ["alice", "bob", "carol", "dave", "emma"];
            let domains = ["example.com", "test.org", "demo.net"];
            return format!(
                "{}{}@{}",
                names[rng.random_range(0..names.len())],
                rng.random_range(1..100),
                domains[rng.random_range(0..domains.len())]
            );
        }
    }
    let min_len = constraint.min_length.unwrap_or(1) as usize;
    let max_len = constraint.max_length.unwrap_or(50) as usize;
    let target_len = rng.random_range(min_len..=max_len);
    let words = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    ];
    let mut result = String::new();
    while result.len() < target_len {
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(words[rng.random_range(0..words.len())]);
    }
    result.truncate(max_len);
    result
}

/// Generate integer value conforming to constraints
pub(super) fn generate_integer_value<R: Rng>(
    rng: &mut R,
    constraint: &PropertyConstraint,
) -> String {
    let min = constraint
        .min_inclusive
        .as_ref()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0);
    let max = constraint
        .max_inclusive
        .as_ref()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(1000);
    rng.random_range(min..=max).to_string()
}

/// Generate decimal value conforming to constraints
pub(super) fn generate_decimal_value<R: Rng>(
    rng: &mut R,
    constraint: &PropertyConstraint,
) -> String {
    let min = constraint
        .min_inclusive
        .as_ref()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let max = constraint
        .max_inclusive
        .as_ref()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1000.0);
    let value = min + (max - min) * (rng.random_range(0..10000) as f64 / 10000.0);
    format!("{:.2}", value)
}
