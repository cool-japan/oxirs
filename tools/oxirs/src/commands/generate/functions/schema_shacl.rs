//! SHACL schema integration for RDF generation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::*;
use super::schema_detect::detect_rdf_format;
use oxirs_core::format::RdfParser;
use oxirs_core::model::{GraphName, Literal, NamedNode, Object, Quad, Subject, Term};
use oxirs_core::RdfTerm;
use scirs2_core::RngExt;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::BufReader;

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const SH_NODE_SHAPE: &str = "http://www.w3.org/ns/shacl#NodeShape";
const SH_PROPERTY_SHAPE: &str = "http://www.w3.org/ns/shacl#PropertyShape";
const SH_TARGET_CLASS: &str = "http://www.w3.org/ns/shacl#targetClass";
const SH_PROPERTY: &str = "http://www.w3.org/ns/shacl#property";
const SH_PATH: &str = "http://www.w3.org/ns/shacl#path";
const SH_DATATYPE: &str = "http://www.w3.org/ns/shacl#datatype";
const SH_MIN_COUNT: &str = "http://www.w3.org/ns/shacl#minCount";
const SH_MAX_COUNT: &str = "http://www.w3.org/ns/shacl#maxCount";
const SH_PATTERN: &str = "http://www.w3.org/ns/shacl#pattern";
const SH_MIN_LENGTH: &str = "http://www.w3.org/ns/shacl#minLength";
const SH_MAX_LENGTH: &str = "http://www.w3.org/ns/shacl#maxLength";
const SH_MIN_INCLUSIVE: &str = "http://www.w3.org/ns/shacl#minInclusive";
const SH_MAX_INCLUSIVE: &str = "http://www.w3.org/ns/shacl#maxInclusive";
const SH_NODE_KIND: &str = "http://www.w3.org/ns/shacl#nodeKind";
const SH_CLASS: &str = "http://www.w3.org/ns/shacl#class";

/// Parse SHACL shapes from the user's schema file.
///
/// Reads and parses the file with the real RDF parser, then extracts node shapes
/// that carry a `sh:targetClass` together with their property constraints.
/// Property shapes referenced by `sh:property` are followed whether they are
/// named nodes or blank nodes. A missing, unparseable, or malformed file is
/// surfaced as an explicit error; there is no fallback to placeholder data.
pub(super) fn parse_shacl_shapes(
    path: &std::path::PathBuf,
    ctx: &crate::cli::CliContext,
) -> Result<Vec<ShaclShape>, Box<dyn Error>> {
    let format = detect_rdf_format(path)?;
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(format);

    let mut quads = Vec::new();
    for quad_result in parser.for_reader(reader) {
        quads.push(quad_result?);
    }
    ctx.info(&format!("Parsed {} RDF quads", quads.len()));

    Ok(extract_shacl_shapes(&quads))
}

/// Identifier for a subject that can anchor a shape (named node or blank node).
fn subject_key(subject: &Subject) -> Option<&str> {
    match subject {
        Subject::NamedNode(nn) => Some(nn.as_str()),
        Subject::BlankNode(bn) => Some(bn.as_str()),
        _ => None,
    }
}

/// Identifier for an object that references another node (named or blank).
fn object_key(object: &Object) -> Option<&str> {
    match object {
        Object::NamedNode(nn) => Some(nn.as_str()),
        Object::BlankNode(bn) => Some(bn.as_str()),
        _ => None,
    }
}

/// Map a `sh:nodeKind` IRI to the label the value generator understands.
fn node_kind_label(uri: &str) -> Option<String> {
    match uri {
        "http://www.w3.org/ns/shacl#IRI" => Some("IRI".to_string()),
        "http://www.w3.org/ns/shacl#Literal" => Some("Literal".to_string()),
        "http://www.w3.org/ns/shacl#BlankNode" => Some("BlankNode".to_string()),
        "http://www.w3.org/ns/shacl#BlankNodeOrIRI" => Some("BlankNodeOrIRI".to_string()),
        "http://www.w3.org/ns/shacl#BlankNodeOrLiteral" => Some("BlankNodeOrLiteral".to_string()),
        "http://www.w3.org/ns/shacl#IRIOrLiteral" => Some("IRIOrLiteral".to_string()),
        _ => None,
    }
}

/// Extract SHACL node shapes (with target classes) from parsed quads.
fn extract_shacl_shapes(quads: &[Quad]) -> Vec<ShaclShape> {
    let mut shape_map: HashMap<String, ShaclShape> = HashMap::new();

    // Pass 1: register shape subjects and their target classes.
    for quad in quads {
        let Some(subject) = subject_key(quad.subject()) else {
            continue;
        };
        match quad.predicate().as_str() {
            SH_TARGET_CLASS => {
                if let Object::NamedNode(nn) = quad.object() {
                    shape_map
                        .entry(subject.to_string())
                        .or_insert_with(new_shape)
                        .target_class = Some(nn.as_str().to_string());
                }
            }
            RDF_TYPE => {
                if let Object::NamedNode(nn) = quad.object() {
                    if matches!(nn.as_str(), SH_NODE_SHAPE | SH_PROPERTY_SHAPE) {
                        shape_map
                            .entry(subject.to_string())
                            .or_insert_with(new_shape);
                    }
                }
            }
            _ => {}
        }
    }

    // Pass 2: attach property constraints referenced via sh:property.
    for quad in quads {
        let Some(subject) = subject_key(quad.subject()) else {
            continue;
        };
        if quad.predicate().as_str() != SH_PROPERTY || !shape_map.contains_key(subject) {
            continue;
        }
        let Some(prop_ref) = object_key(quad.object()) else {
            continue;
        };
        if let Some(constraint) = extract_property_constraint(prop_ref, quads) {
            if let Some(shape) = shape_map.get_mut(subject) {
                shape.properties.push(constraint);
            }
        }
    }

    // Only shapes with a target class can seed instance generation. Sort for
    // reproducible output given a seed.
    let mut shapes: Vec<ShaclShape> = shape_map
        .into_values()
        .filter(|shape| shape.target_class.is_some())
        .collect();
    shapes.sort_by(|a, b| a.target_class.cmp(&b.target_class));
    shapes
}

/// Extract a single property constraint from the triples of a property shape.
fn extract_property_constraint(prop_ref: &str, quads: &[Quad]) -> Option<PropertyConstraint> {
    let mut constraint = PropertyConstraint {
        path: String::new(),
        min_count: None,
        max_count: None,
        datatype: None,
        pattern: None,
        min_length: None,
        max_length: None,
        min_inclusive: None,
        max_inclusive: None,
        node_kind: None,
        class: None,
    };

    for quad in quads {
        let Some(subject) = subject_key(quad.subject()) else {
            continue;
        };
        if subject != prop_ref {
            continue;
        }
        match quad.predicate().as_str() {
            SH_PATH => {
                if let Object::NamedNode(nn) = quad.object() {
                    constraint.path = nn.as_str().to_string();
                }
            }
            SH_DATATYPE => {
                if let Object::NamedNode(nn) = quad.object() {
                    constraint.datatype = Some(nn.as_str().to_string());
                }
            }
            SH_MIN_COUNT => {
                if let Object::Literal(lit) = quad.object() {
                    constraint.min_count = lit.value().parse().ok();
                }
            }
            SH_MAX_COUNT => {
                if let Object::Literal(lit) = quad.object() {
                    constraint.max_count = lit.value().parse().ok();
                }
            }
            SH_PATTERN => {
                if let Object::Literal(lit) = quad.object() {
                    constraint.pattern = Some(lit.value().to_string());
                }
            }
            SH_MIN_LENGTH => {
                if let Object::Literal(lit) = quad.object() {
                    constraint.min_length = lit.value().parse().ok();
                }
            }
            SH_MAX_LENGTH => {
                if let Object::Literal(lit) = quad.object() {
                    constraint.max_length = lit.value().parse().ok();
                }
            }
            SH_MIN_INCLUSIVE => {
                if let Object::Literal(lit) = quad.object() {
                    constraint.min_inclusive = Some(lit.value().to_string());
                }
            }
            SH_MAX_INCLUSIVE => {
                if let Object::Literal(lit) = quad.object() {
                    constraint.max_inclusive = Some(lit.value().to_string());
                }
            }
            SH_NODE_KIND => {
                if let Object::NamedNode(nn) = quad.object() {
                    constraint.node_kind = node_kind_label(nn.as_str());
                }
            }
            SH_CLASS => {
                if let Object::NamedNode(nn) = quad.object() {
                    constraint.class = Some(nn.as_str().to_string());
                }
            }
            _ => {}
        }
    }

    if constraint.path.is_empty() {
        None
    } else {
        Some(constraint)
    }
}

fn new_shape() -> ShaclShape {
    ShaclShape {
        target_class: None,
        properties: Vec::new(),
    }
}

/// Generate RDF data conforming to SHACL shapes
pub(super) fn generate_from_shapes<R: RngExt>(
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
pub(super) fn generate_property_value<R: RngExt>(
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
pub(super) fn generate_string_value<R: RngExt>(
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
pub(super) fn generate_integer_value<R: RngExt>(
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
pub(super) fn generate_decimal_value<R: RngExt>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::Random;
    use std::io::Write;

    const SHACL_TTL: &str = r#"
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://catalog.example/> .

ex:ProductShape a sh:NodeShape ;
    sh:targetClass ex:Product ;
    sh:property [
        sh:path ex:sku ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path ex:price ;
        sh:datatype xsd:integer ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:minInclusive 1 ;
        sh:maxInclusive 100 ;
    ] .
"#;

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxirs_gen_shacl_{}_{}.ttl",
            std::process::id(),
            name
        ));
        let mut file = std::fs::File::create(&path).expect("create temp shapes file");
        file.write_all(content.as_bytes())
            .expect("write temp shapes file");
        path
    }

    fn dump(quads: &[Quad]) -> String {
        quads
            .iter()
            .map(|q| format!("{:?}", q))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn parses_user_shacl_shapes_and_generates_their_iris() {
        let ctx = crate::cli::CliContext::new();
        let path = write_temp("basic", SHACL_TTL);
        let shapes = parse_shacl_shapes(&path, &ctx).expect("parse shacl shapes");
        std::fs::remove_file(&path).ok();

        assert_eq!(shapes.len(), 1);
        let shape = &shapes[0];
        assert_eq!(
            shape.target_class.as_deref(),
            Some("http://catalog.example/Product")
        );
        // Blank-node property shapes must be followed and their paths captured.
        let paths: Vec<&str> = shape.properties.iter().map(|p| p.path.as_str()).collect();
        assert!(paths.contains(&"http://catalog.example/sku"));
        assert!(paths.contains(&"http://catalog.example/price"));
        // The old placeholder shape (foaf:Person) must never leak in.
        assert!(shape.target_class.as_deref() != Some("http://xmlns.com/foaf/0.1/Person"));

        let mut rng = Random::seed(23);
        let quads = generate_from_shapes(&mut rng, &shapes, 5).expect("generate");
        assert!(!quads.is_empty());
        let text = dump(&quads);
        assert!(text.contains("http://catalog.example/Product"));
        assert!(text.contains("http://catalog.example/sku"));
        assert!(text.contains("http://catalog.example/price"));
        assert!(!text.contains("foaf/0.1/Person"));
    }

    #[test]
    fn nonexistent_shacl_file_is_error() {
        let ctx = crate::cli::CliContext::new();
        let mut path = std::env::temp_dir();
        path.push("oxirs_gen_shacl_missing_zzz_does_not_exist.ttl");
        assert!(parse_shacl_shapes(&path, &ctx).is_err());
    }

    #[test]
    fn malformed_shacl_file_is_error() {
        let ctx = crate::cli::CliContext::new();
        let path = write_temp(
            "malformed",
            "@prefix sh: <http://x/> .\nex:S a sh:NodeShape ; << broken",
        );
        let result = parse_shacl_shapes(&path, &ctx);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }
}
