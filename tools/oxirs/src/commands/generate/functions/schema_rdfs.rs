//! RDFS schema integration for RDF generation
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
const RDF_PROPERTY: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property";
const RDFS_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#Class";
const RDFS_PROPERTY: &str = "http://www.w3.org/2000/01/rdf-schema#Property";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const RDFS_COMMENT: &str = "http://www.w3.org/2000/01/rdf-schema#comment";
const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
const RDFS_SUBPROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";

/// Parse an RDFS schema from the user's schema file.
///
/// Reads and parses the file with the real RDF parser, then extracts RDFS class
/// and property definitions (labels, domains, ranges, hierarchy) from the
/// resulting quads. A missing, unparseable, or malformed file is surfaced as an
/// explicit error; there is no fallback to placeholder data.
pub(super) fn parse_rdfs_schema(
    path: &std::path::PathBuf,
    ctx: &crate::cli::CliContext,
) -> Result<RdfsSchema, Box<dyn Error>> {
    let format = detect_rdf_format(path)?;
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(format);

    let mut quads = Vec::new();
    for quad_result in parser.for_reader(reader) {
        quads.push(quad_result?);
    }
    ctx.info(&format!("Parsed {} RDF quads", quads.len()));

    Ok(extract_rdfs_schema(&quads))
}

/// Extract RDFS classes and properties from parsed quads.
fn extract_rdfs_schema(quads: &[Quad]) -> RdfsSchema {
    let mut classes: HashMap<String, RdfsClass> = HashMap::new();
    let mut properties: HashMap<String, RdfsProperty> = HashMap::new();

    // First pass: identify classes and properties by their rdf:type.
    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };
        if quad.predicate().as_str() != RDF_TYPE {
            continue;
        }
        let Object::NamedNode(obj) = quad.object() else {
            continue;
        };
        match obj.as_str() {
            RDFS_CLASS => {
                classes
                    .entry(subject_uri.to_string())
                    .or_insert_with(|| new_rdfs_class(subject_uri));
            }
            RDF_PROPERTY | RDFS_PROPERTY => {
                properties
                    .entry(subject_uri.to_string())
                    .or_insert_with(|| new_rdfs_property(subject_uri));
            }
            _ => {}
        }
    }

    // Second pass: attach class and property metadata.
    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };
        let predicate_uri = quad.predicate().as_str();

        if let Some(class_def) = classes.get_mut(subject_uri) {
            match predicate_uri {
                RDFS_LABEL => {
                    if let Object::Literal(lit) = quad.object() {
                        class_def._label = Some(lit.value().to_string());
                    }
                }
                RDFS_COMMENT => {
                    if let Object::Literal(lit) = quad.object() {
                        class_def._comment = Some(lit.value().to_string());
                    }
                }
                RDFS_SUBCLASS_OF => {
                    if let Object::NamedNode(nn) = quad.object() {
                        class_def._super_classes.push(nn.as_str().to_string());
                    }
                }
                _ => {}
            }
        }

        if let Some(prop_def) = properties.get_mut(subject_uri) {
            match predicate_uri {
                RDFS_LABEL => {
                    if let Object::Literal(lit) = quad.object() {
                        prop_def._label = Some(lit.value().to_string());
                    }
                }
                RDFS_COMMENT => {
                    if let Object::Literal(lit) = quad.object() {
                        prop_def._comment = Some(lit.value().to_string());
                    }
                }
                RDFS_DOMAIN => {
                    if let Object::NamedNode(nn) = quad.object() {
                        prop_def.domain.push(nn.as_str().to_string());
                    }
                }
                RDFS_RANGE => {
                    if let Object::NamedNode(nn) = quad.object() {
                        prop_def.range.push(nn.as_str().to_string());
                    }
                }
                RDFS_SUBPROPERTY_OF => {
                    if let Object::NamedNode(nn) = quad.object() {
                        prop_def._super_properties.push(nn.as_str().to_string());
                    }
                }
                _ => {}
            }
        }
    }

    // Sort by URI so generation is reproducible for a given seed.
    let mut classes: Vec<RdfsClass> = classes.into_values().collect();
    classes.sort_by(|a, b| a.uri.cmp(&b.uri));
    let mut properties: Vec<RdfsProperty> = properties.into_values().collect();
    properties.sort_by(|a, b| a.uri.cmp(&b.uri));

    RdfsSchema {
        classes,
        properties,
    }
}

fn new_rdfs_class(uri: &str) -> RdfsClass {
    RdfsClass {
        uri: uri.to_string(),
        _label: None,
        _comment: None,
        _super_classes: Vec::new(),
    }
}

fn new_rdfs_property(uri: &str) -> RdfsProperty {
    RdfsProperty {
        uri: uri.to_string(),
        _label: None,
        _comment: None,
        domain: Vec::new(),
        range: Vec::new(),
        _super_properties: Vec::new(),
    }
}

/// Generate RDF data conforming to RDFS schema
pub(super) fn generate_from_rdfs_schema<R: RngExt>(
    rng: &mut R,
    schema: &RdfsSchema,
    count: usize,
) -> Result<Vec<Quad>, Box<dyn Error>> {
    let mut quads = Vec::new();
    let instances_per_class = if !schema.classes.is_empty() {
        (count as f64 / schema.classes.len() as f64).ceil() as usize
    } else {
        return Ok(quads);
    };
    let mut generated_instances: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for class in &schema.classes {
        let mut class_instances = Vec::new();
        for i in 0..instances_per_class {
            let class_name = class.uri.split('/').next_back().unwrap_or("instance");
            let instance_uri_str = format!("http://example.org/{}/{}", class_name, i);
            let instance_uri =
                Subject::NamedNode(NamedNode::new(&instance_uri_str).expect("Valid IRI"));
            class_instances.push(instance_uri_str.clone());
            quads.push(Quad::new(
                instance_uri.clone(),
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .expect("Valid IRI"),
                Term::NamedNode(NamedNode::new(&class.uri).expect("Valid IRI")),
                GraphName::DefaultGraph,
            ));
            for property in &schema.properties {
                if property.domain.contains(&class.uri) {
                    let value =
                        generate_rdfs_property_value(rng, property, &generated_instances, schema)?;
                    quads.push(Quad::new(
                        instance_uri.clone(),
                        NamedNode::new(&property.uri).expect("Valid IRI"),
                        value,
                        GraphName::DefaultGraph,
                    ));
                }
            }
        }
        generated_instances.insert(class.uri.clone(), class_instances);
    }
    Ok(quads)
}

/// Generate a property value based on RDFS range constraints
pub(super) fn generate_rdfs_property_value<R: RngExt>(
    rng: &mut R,
    property: &RdfsProperty,
    generated_instances: &std::collections::HashMap<String, Vec<String>>,
    schema: &RdfsSchema,
) -> Result<Term, Box<dyn Error>> {
    let range_uri = property.range.first().ok_or("Property has no range")?;
    if schema.classes.iter().any(|c| &c.uri == range_uri) {
        if let Some(instances) = generated_instances.get(range_uri) {
            if !instances.is_empty() {
                let idx = rng.random_range(0..instances.len());
                return Ok(Term::NamedNode(
                    NamedNode::new(&instances[idx]).expect("Valid IRI"),
                ));
            }
        }
        let class_name = range_uri.split('/').next_back().unwrap_or("resource");
        return Ok(Term::NamedNode(
            NamedNode::new(format!(
                "http://example.org/{}/{}",
                class_name,
                rng.random_range(0..1000)
            ))
            .expect("Valid IRI"),
        ));
    }
    let value_str = match range_uri.as_str() {
        "http://www.w3.org/2001/XMLSchema#string" => {
            let words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"];
            let mut result = String::new();
            for _ in 0..rng.random_range(1..4) {
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(words[rng.random_range(0..words.len())]);
            }
            result
        }
        "http://www.w3.org/2001/XMLSchema#integer" | "http://www.w3.org/2001/XMLSchema#int" => {
            if property.uri.contains("age") {
                rng.random_range(18..80).to_string()
            } else if property.uri.contains("year") || property.uri.contains("Year") {
                rng.random_range(1950..2024).to_string()
            } else {
                rng.random_range(0..1000).to_string()
            }
        }
        "http://www.w3.org/2001/XMLSchema#decimal" | "http://www.w3.org/2001/XMLSchema#double" => {
            let value = rng.random_range(0..10000) as f64 / 100.0;
            format!("{:.2}", value)
        }
        "http://www.w3.org/2001/XMLSchema#boolean" => if rng.random_range(0..2) == 0 {
            "true"
        } else {
            "false"
        }
        .to_string(),
        "http://www.w3.org/2001/XMLSchema#date" => {
            let year = rng.random_range(1950..=2024);
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
        _ => format!("value_{}", rng.random_range(0..1000)),
    };
    Ok(Term::Literal(Literal::new_typed_literal(
        value_str,
        NamedNode::new(range_uri).expect("Valid IRI"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::Random;
    use std::io::Write;

    const RDFS_TTL: &str = r#"
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://library.example/> .

ex:Book a rdfs:Class ;
    rdfs:label "Book" .
ex:Author a rdfs:Class ;
    rdfs:label "Author" .
ex:title a rdf:Property ;
    rdfs:domain ex:Book ;
    rdfs:range xsd:string .
ex:writtenBy a rdf:Property ;
    rdfs:domain ex:Book ;
    rdfs:range ex:Author .
"#;

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxirs_gen_rdfs_{}_{}.ttl",
            std::process::id(),
            name
        ));
        let mut file = std::fs::File::create(&path).expect("create temp schema file");
        file.write_all(content.as_bytes())
            .expect("write temp schema file");
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
    fn parses_user_rdfs_schema_and_generates_its_iris() {
        let ctx = crate::cli::CliContext::new();
        let path = write_temp("basic", RDFS_TTL);
        let schema = parse_rdfs_schema(&path, &ctx).expect("parse rdfs schema");
        std::fs::remove_file(&path).ok();

        let class_uris: Vec<&str> = schema.classes.iter().map(|c| c.uri.as_str()).collect();
        assert!(class_uris.contains(&"http://library.example/Book"));
        assert!(class_uris.contains(&"http://library.example/Author"));
        // The old placeholder schema (ex:Person et al. at example.org) must not leak.
        assert!(!class_uris.iter().any(|u| u == &"http://example.org/Person"));

        let mut rng = Random::seed(11);
        let quads = generate_from_rdfs_schema(&mut rng, &schema, 6).expect("generate");
        assert!(!quads.is_empty());
        let text = dump(&quads);
        assert!(text.contains("http://library.example/Book"));
        assert!(text.contains("http://library.example/title"));
        assert!(!text.contains("http://example.org/Person"));
        assert!(!text.contains("http://example.org/Organization"));
    }

    #[test]
    fn nonexistent_rdfs_file_is_error() {
        let ctx = crate::cli::CliContext::new();
        let mut path = std::env::temp_dir();
        path.push("oxirs_gen_rdfs_missing_zzz_does_not_exist.ttl");
        assert!(parse_rdfs_schema(&path, &ctx).is_err());
    }

    #[test]
    fn malformed_rdfs_file_is_error() {
        let ctx = crate::cli::CliContext::new();
        let path = write_temp(
            "malformed",
            "@prefix ex: <http://x/> .\nex:A a rdfs:Class ; << broken",
        );
        let result = parse_rdfs_schema(&path, &ctx);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }
}
