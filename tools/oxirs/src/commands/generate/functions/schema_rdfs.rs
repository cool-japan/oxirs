//! RDFS schema integration for RDF generation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::*;
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use scirs2_core::Rng;
use std::error::Error;

/// Parse RDFS schema from file (simplified implementation)
pub(super) fn parse_rdfs_schema(
    _path: &std::path::PathBuf,
    _ctx: &crate::cli::CliContext,
) -> Result<RdfsSchema, Box<dyn Error>> {
    let classes = vec![
        RdfsClass {
            uri: "http://example.org/Person".to_string(),
            _label: Some("Person".to_string()),
            _comment: Some("A human being".to_string()),
            _super_classes: vec!["http://www.w3.org/2000/01/rdf-schema#Resource".to_string()],
        },
        RdfsClass {
            uri: "http://example.org/Organization".to_string(),
            _label: Some("Organization".to_string()),
            _comment: Some("An organized group of people".to_string()),
            _super_classes: vec!["http://www.w3.org/2000/01/rdf-schema#Resource".to_string()],
        },
        RdfsClass {
            uri: "http://example.org/Document".to_string(),
            _label: Some("Document".to_string()),
            _comment: Some("A written or digital document".to_string()),
            _super_classes: vec!["http://www.w3.org/2000/01/rdf-schema#Resource".to_string()],
        },
    ];
    let properties = vec![
        RdfsProperty {
            uri: "http://example.org/name".to_string(),
            _label: Some("name".to_string()),
            _comment: Some("The name of something".to_string()),
            domain: vec![
                "http://example.org/Person".to_string(),
                "http://example.org/Organization".to_string(),
            ],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec!["http://www.w3.org/2000/01/rdf-schema#label".to_string()],
        },
        RdfsProperty {
            uri: "http://example.org/age".to_string(),
            _label: Some("age".to_string()),
            _comment: Some("The age of a person".to_string()),
            domain: vec!["http://example.org/Person".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#integer".to_string()],
            _super_properties: vec![],
        },
        RdfsProperty {
            uri: "http://example.org/email".to_string(),
            _label: Some("email".to_string()),
            _comment: Some("Email address".to_string()),
            domain: vec!["http://example.org/Person".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
        },
        RdfsProperty {
            uri: "http://example.org/employedBy".to_string(),
            _label: Some("employed by".to_string()),
            _comment: Some("The organization that employs a person".to_string()),
            domain: vec!["http://example.org/Person".to_string()],
            range: vec!["http://example.org/Organization".to_string()],
            _super_properties: vec![],
        },
        RdfsProperty {
            uri: "http://example.org/foundedYear".to_string(),
            _label: Some("founded year".to_string()),
            _comment: Some("The year an organization was founded".to_string()),
            domain: vec!["http://example.org/Organization".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#integer".to_string()],
            _super_properties: vec![],
        },
        RdfsProperty {
            uri: "http://example.org/author".to_string(),
            _label: Some("author".to_string()),
            _comment: Some("The author of a document".to_string()),
            domain: vec!["http://example.org/Document".to_string()],
            range: vec!["http://example.org/Person".to_string()],
            _super_properties: vec!["http://purl.org/dc/terms/creator".to_string()],
        },
        RdfsProperty {
            uri: "http://example.org/title".to_string(),
            _label: Some("title".to_string()),
            _comment: Some("The title of a document".to_string()),
            domain: vec!["http://example.org/Document".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
        },
    ];
    Ok(RdfsSchema {
        classes,
        properties,
    })
}

/// Generate RDF data conforming to RDFS schema
pub(super) fn generate_from_rdfs_schema<R: Rng>(
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
pub(super) fn generate_rdfs_property_value<R: Rng>(
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
