//! SHACL-based data generation - Generate RDF data conforming to SHACL shapes
//!
//! This module parses SHACL shapes and generates synthetic RDF data that conforms
//! to the constraints defined in the shapes.

use crate::cli::CliContext;
use oxirs_core::format::{RdfFormat, RdfParser};
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use oxirs_core::RdfTerm;
use scirs2_core::random::Random;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

/// SHACL namespace
const SHACL_NS: &str = "http://www.w3.org/ns/shacl#";

/// Common RDF namespaces
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
#[allow(dead_code)]
const RDFS_NS: &str = "http://www.w3.org/2000/01/rdf-schema#";
const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

/// SHACL shape definition
#[derive(Debug, Clone)]
pub struct ShaclShape {
    pub uri: String,
    pub target_class: Option<String>,
    pub properties: Vec<PropertyShape>,
    pub node_kind: Option<NodeKind>,
    pub closed: bool,
}

/// SHACL property shape
#[derive(Debug, Clone, Default)]
pub struct PropertyShape {
    pub path: String,
    pub datatype: Option<String>,
    pub min_count: Option<u32>,
    pub max_count: Option<u32>,
    pub pattern: Option<String>,
    pub min_length: Option<u32>,
    pub max_length: Option<u32>,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub node_kind: Option<NodeKind>,
    pub class_constraint: Option<String>,
    pub in_values: Vec<String>,
}

/// Node kind constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Iri,
    Literal,
    BlankNode,
    BlankNodeOrIri,
    BlankNodeOrLiteral,
    IriOrLiteral,
}

impl NodeKind {
    fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://www.w3.org/ns/shacl#IRI" => Some(NodeKind::Iri),
            "http://www.w3.org/ns/shacl#Literal" => Some(NodeKind::Literal),
            "http://www.w3.org/ns/shacl#BlankNode" => Some(NodeKind::BlankNode),
            "http://www.w3.org/ns/shacl#BlankNodeOrIRI" => Some(NodeKind::BlankNodeOrIri),
            "http://www.w3.org/ns/shacl#BlankNodeOrLiteral" => Some(NodeKind::BlankNodeOrLiteral),
            "http://www.w3.org/ns/shacl#IRIOrLiteral" => Some(NodeKind::IriOrLiteral),
            _ => None,
        }
    }
}

/// Parse SHACL shapes from a Turtle file
pub fn parse_shacl_shapes(
    shape_file: &Path,
    ctx: &CliContext,
) -> Result<Vec<ShaclShape>, Box<dyn std::error::Error>> {
    ctx.info(&format!(
        "Parsing SHACL shapes from {}",
        shape_file.display()
    ));

    // Read and parse the SHACL file
    let file = fs::File::open(shape_file)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(RdfFormat::Turtle);

    let mut quads = Vec::new();
    for quad_result in parser.for_reader(reader) {
        match quad_result {
            Ok(quad) => quads.push(quad),
            Err(e) => {
                ctx.warn(&format!("Warning: Failed to parse quad: {}", e));
            }
        }
    }

    ctx.info(&format!("Parsed {} RDF quads", quads.len()));

    // Extract shape definitions
    let shapes = extract_shapes(&quads, ctx)?;

    ctx.info(&format!("Found {} SHACL shapes", shapes.len()));

    Ok(shapes)
}

/// Extract SHACL shapes from parsed quads
fn extract_shapes(
    quads: &[Quad],
    ctx: &CliContext,
) -> Result<Vec<ShaclShape>, Box<dyn std::error::Error>> {
    let mut shapes = Vec::new();
    let mut shape_map: HashMap<String, ShaclShape> = HashMap::new();

    // First pass: identify all NodeShapes
    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };

        let predicate_uri = quad.predicate().as_str();

        // Check if this is a NodeShape by checking the object
        let is_node_shape = match quad.object() {
            oxirs_core::model::Object::NamedNode(nn) => {
                predicate_uri == format!("{}type", RDF_NS)
                    && (nn.as_str() == format!("{}NodeShape", SHACL_NS)
                        || nn.as_str() == format!("{}PropertyShape", SHACL_NS))
            }
            _ => false,
        };

        if is_node_shape {
            shape_map
                .entry(subject_uri.to_string())
                .or_insert(ShaclShape {
                    uri: subject_uri.to_string(),
                    target_class: None,
                    properties: Vec::new(),
                    node_kind: None,
                    closed: false,
                });
        }
    }

    ctx.info(&format!("Found {} shape candidates", shape_map.len()));

    // Second pass: extract shape properties
    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };

        if let Some(shape) = shape_map.get_mut(subject_uri) {
            let predicate_uri = quad.predicate().as_str();

            match predicate_uri {
                p if p == format!("{}targetClass", SHACL_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        shape.target_class = Some(nn.as_str().to_string());
                    }
                }
                p if p == format!("{}property", SHACL_NS) => {
                    // Extract property shape
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        let prop_shape =
                            extract_property_shape(nn.as_str(), quads).unwrap_or_default();
                        shape.properties.push(prop_shape);
                    }
                }
                p if p == format!("{}nodeKind", SHACL_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        shape.node_kind = NodeKind::from_uri(nn.as_str());
                    }
                }
                p if p == format!("{}closed", SHACL_NS) => {
                    if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                        shape.closed = lit.value() == "true";
                    }
                }
                _ => {}
            }
        }
    }

    // Collect shapes with target classes
    for shape in shape_map.values() {
        if shape.target_class.is_some() {
            shapes.push(shape.clone());
        }
    }

    Ok(shapes)
}

/// Extract property shape from quads
fn extract_property_shape(property_uri: &str, quads: &[Quad]) -> Option<PropertyShape> {
    let mut prop_shape = PropertyShape {
        path: String::new(),
        datatype: None,
        min_count: None,
        max_count: None,
        pattern: None,
        min_length: None,
        max_length: None,
        min_value: None,
        max_value: None,
        node_kind: None,
        class_constraint: None,
        in_values: Vec::new(),
    };

    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };

        if subject_uri != property_uri {
            continue;
        }

        let predicate_uri = quad.predicate().as_str();

        match predicate_uri {
            p if p == format!("{}path", SHACL_NS) => {
                if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                    prop_shape.path = nn.as_str().to_string();
                }
            }
            p if p == format!("{}datatype", SHACL_NS) => {
                if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                    prop_shape.datatype = Some(nn.as_str().to_string());
                }
            }
            p if p == format!("{}minCount", SHACL_NS) => {
                if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                    prop_shape.min_count = lit.value().parse().ok();
                }
            }
            p if p == format!("{}maxCount", SHACL_NS) => {
                if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                    prop_shape.max_count = lit.value().parse().ok();
                }
            }
            p if p == format!("{}pattern", SHACL_NS) => {
                if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                    prop_shape.pattern = Some(lit.value().to_string());
                }
            }
            p if p == format!("{}minLength", SHACL_NS) => {
                if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                    prop_shape.min_length = lit.value().parse().ok();
                }
            }
            p if p == format!("{}maxLength", SHACL_NS) => {
                if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                    prop_shape.max_length = lit.value().parse().ok();
                }
            }
            p if p == format!("{}nodeKind", SHACL_NS) => {
                if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                    prop_shape.node_kind = NodeKind::from_uri(nn.as_str());
                }
            }
            p if p == format!("{}class", SHACL_NS) => {
                if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                    prop_shape.class_constraint = Some(nn.as_str().to_string());
                }
            }
            _ => {}
        }
    }

    if !prop_shape.path.is_empty() {
        Some(prop_shape)
    } else {
        None
    }
}

/// Generate RDF data conforming to SHACL shapes
pub fn generate_from_shapes<R: scirs2_core::RngCore>(
    shapes: &[ShaclShape],
    instance_count: usize,
    rng: &mut Random<R>,
    ctx: &CliContext,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();

    ctx.info(&format!(
        "Generating {} instances from {} shapes",
        instance_count,
        shapes.len()
    ));

    for shape in shapes {
        if let Some(target_class) = &shape.target_class {
            ctx.info(&format!(
                "Generating instances for shape: {} (target class: {})",
                shape.uri, target_class
            ));

            let instances_per_shape = instance_count / shapes.len();
            for i in 0..instances_per_shape {
                let instance_uri = format!(
                    "http://example.org/instance_{}_{}",
                    shape.uri.replace("http://example.org/shapes/", ""),
                    i
                );

                // Add rdf:type triple
                quads.push(create_quad(
                    &instance_uri,
                    &format!("{}type", RDF_NS),
                    Term::NamedNode(NamedNode::new_unchecked(target_class.clone())),
                ));

                // Generate properties
                for prop in &shape.properties {
                    let prop_quads = generate_property_values(&instance_uri, prop, rng)?;
                    quads.extend(prop_quads);
                }
            }
        }
    }

    ctx.info(&format!("Generated {} RDF quads", quads.len()));

    Ok(quads)
}

/// Generate property values conforming to constraints
fn generate_property_values<R: scirs2_core::RngCore>(
    subject_uri: &str,
    prop: &PropertyShape,
    rng: &mut Random<R>,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();

    // Determine how many values to generate
    let min = prop.min_count.unwrap_or(0) as usize;
    let max = prop.max_count.unwrap_or(min.max(1) as u32) as usize;
    let count = if min == max {
        min
    } else {
        rng.gen_range(min..=max)
    };

    for _ in 0..count {
        let value = if !prop.in_values.is_empty() {
            // Pick from enumeration
            let idx = rng.gen_range(0..prop.in_values.len());
            Term::NamedNode(NamedNode::new_unchecked(prop.in_values[idx].clone()))
        } else if let Some(class_constraint) = &prop.class_constraint {
            // Generate instance of class
            let instance_id = rng.gen_range(0..1000);
            Term::NamedNode(NamedNode::new_unchecked(format!(
                "http://example.org/{}_instance_{}",
                class_constraint
                    .split('/')
                    .next_back()
                    .unwrap_or("instance"),
                instance_id
            )))
        } else if let Some(datatype) = &prop.datatype {
            // Generate typed literal
            generate_typed_literal(datatype, prop, rng)?
        } else {
            // Generate based on node kind
            match prop.node_kind {
                Some(NodeKind::Iri) => {
                    let id = rng.gen_range(0..10000);
                    Term::NamedNode(NamedNode::new_unchecked(format!(
                        "http://example.org/resource_{}",
                        id
                    )))
                }
                Some(NodeKind::Literal) | None => {
                    // Generate string literal
                    let len = prop.max_length.unwrap_or(20).min(50);
                    let string_val = generate_random_string(rng, len as usize);
                    Term::Literal(Literal::new_simple_literal(string_val))
                }
                _ => {
                    // Default to string literal
                    let string_val = generate_random_string(rng, 10);
                    Term::Literal(Literal::new_simple_literal(string_val))
                }
            }
        };

        quads.push(create_quad(subject_uri, &prop.path, value));
    }

    Ok(quads)
}

/// Generate typed literal value
fn generate_typed_literal<R: scirs2_core::RngCore>(
    datatype: &str,
    prop: &PropertyShape,
    rng: &mut Random<R>,
) -> Result<Term, Box<dyn std::error::Error>> {
    let value_str = match datatype {
        dt if dt == format!("{}string", XSD_NS) => {
            let min_len = prop.min_length.unwrap_or(1) as usize;
            let max_len = prop.max_length.unwrap_or(50) as usize;
            let len = rng.gen_range(min_len..=max_len);

            if let Some(pattern) = &prop.pattern {
                // Try to generate from pattern (simplified)
                generate_from_pattern(pattern, len, rng)
            } else {
                generate_random_string(rng, len)
            }
        }
        dt if dt == format!("{}integer", XSD_NS) => {
            let min = prop
                .min_value
                .as_ref()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);
            let max = prop
                .max_value
                .as_ref()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(1000);
            rng.gen_range(min..=max).to_string()
        }
        dt if dt == format!("{}decimal", XSD_NS) || dt == format!("{}double", XSD_NS) => {
            let val = rng.random_range(0, 10000) as f64 / 10.0;
            format!("{:.2}", val)
        }
        dt if dt == format!("{}boolean", XSD_NS) => {
            if rng.random_range(0, 2) == 0 {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        dt if dt == format!("{}date", XSD_NS) => {
            let year = rng.gen_range(2000..=2024);
            let month = rng.gen_range(1..=12);
            let day = rng.gen_range(1..=28);
            format!("{:04}-{:02}-{:02}", year, month, day)
        }
        dt if dt == format!("{}dateTime", XSD_NS) => {
            let year = rng.gen_range(2000..=2024);
            let month = rng.gen_range(1..=12);
            let day = rng.gen_range(1..=28);
            let hour = rng.gen_range(0..=23);
            let minute = rng.gen_range(0..=59);
            let second = rng.gen_range(0..=59);
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                year, month, day, hour, minute, second
            )
        }
        _ => {
            // Default to string
            generate_random_string(rng, 10)
        }
    };

    Ok(Term::Literal(Literal::new_typed_literal(
        value_str,
        NamedNode::new_unchecked(datatype.to_string()),
    )))
}

/// Generate random string
fn generate_random_string<R: scirs2_core::RngCore>(rng: &mut Random<R>, length: usize) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARS.len());
            CHARS[idx] as char
        })
        .collect()
}

/// Generate string from regex pattern (simplified)
fn generate_from_pattern<R: scirs2_core::RngCore>(
    pattern: &str,
    max_length: usize,
    rng: &mut Random<R>,
) -> String {
    // Simplified pattern matching - only handles basic patterns
    if pattern.contains("[a-z]+") {
        generate_lowercase_string(rng, max_length)
    } else if pattern.contains("[A-Z]+") {
        generate_uppercase_string(rng, max_length)
    } else if pattern.contains("[0-9]+") {
        generate_numeric_string(rng, max_length)
    } else {
        // Default to alphanumeric
        generate_random_string(rng, max_length)
    }
}

fn generate_lowercase_string<R: scirs2_core::RngCore>(
    rng: &mut Random<R>,
    length: usize,
) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARS.len());
            CHARS[idx] as char
        })
        .collect()
}

fn generate_uppercase_string<R: scirs2_core::RngCore>(
    rng: &mut Random<R>,
    length: usize,
) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARS.len());
            CHARS[idx] as char
        })
        .collect()
}

fn generate_numeric_string<R: scirs2_core::RngCore>(rng: &mut Random<R>, length: usize) -> String {
    const CHARS: &[u8] = b"0123456789";
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARS.len());
            CHARS[idx] as char
        })
        .collect()
}

/// Create a quad
fn create_quad(subject: &str, predicate: &str, object: Term) -> Quad {
    Quad::new(
        Subject::NamedNode(NamedNode::new_unchecked(subject.to_string())),
        NamedNode::new_unchecked(predicate.to_string()),
        object,
        GraphName::DefaultGraph,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_kind_from_uri() {
        assert_eq!(
            NodeKind::from_uri("http://www.w3.org/ns/shacl#IRI"),
            Some(NodeKind::Iri)
        );
        assert_eq!(
            NodeKind::from_uri("http://www.w3.org/ns/shacl#Literal"),
            Some(NodeKind::Literal)
        );
        assert_eq!(NodeKind::from_uri("invalid"), None);
    }

    #[test]
    fn test_generate_random_string() {
        let mut rng = Random::seed(42);
        let s = generate_random_string(&mut rng, 10);
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn test_generate_lowercase_string() {
        let mut rng = Random::seed(42);
        let s = generate_lowercase_string(&mut rng, 10);
        assert_eq!(s.len(), 10);
        assert!(s.chars().all(|c| c.is_ascii_lowercase()));
    }
}
