//! OWL ontology integration for RDF generation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::*;
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use scirs2_core::Rng;
use std::error::Error;

/// Parse OWL ontology from file (simplified implementation)
pub(super) fn parse_owl_ontology(
    _path: &std::path::PathBuf,
    _ctx: &crate::cli::CliContext,
) -> Result<OwlOntology, Box<dyn Error>> {
    let classes = vec![
        OwlClass {
            uri: "http://example.org/University".to_string(),
            _label: Some("University".to_string()),
            _comment: Some("An institution of higher education".to_string()),
            _super_classes: vec!["http://example.org/Organization".to_string()],
            _equivalent_classes: vec![],
            _disjoint_with: vec!["http://example.org/Person".to_string()],
            restrictions: vec![
                OwlRestriction {
                    on_property: "http://example.org/hasStudent".to_string(),
                    restriction_type: OwlRestrictionType::MinCardinality(10),
                },
                OwlRestriction {
                    on_property: "http://example.org/hasFaculty".to_string(),
                    restriction_type: OwlRestrictionType::MinCardinality(5),
                },
            ],
        },
        OwlClass {
            uri: "http://example.org/Professor".to_string(),
            _label: Some("Professor".to_string()),
            _comment: Some("A university faculty member".to_string()),
            _super_classes: vec!["http://example.org/Person".to_string()],
            _equivalent_classes: vec![],
            _disjoint_with: vec!["http://example.org/Student".to_string()],
            restrictions: vec![
                OwlRestriction {
                    on_property: "http://example.org/teachesAt".to_string(),
                    restriction_type: OwlRestrictionType::ExactCardinality(1),
                },
                OwlRestriction {
                    on_property: "http://example.org/officeNumber".to_string(),
                    restriction_type: OwlRestrictionType::ExactCardinality(1),
                },
            ],
        },
        OwlClass {
            uri: "http://example.org/Student".to_string(),
            _label: Some("Student".to_string()),
            _comment: Some("A person enrolled in a university".to_string()),
            _super_classes: vec!["http://example.org/Person".to_string()],
            _equivalent_classes: vec![],
            _disjoint_with: vec!["http://example.org/Professor".to_string()],
            restrictions: vec![
                OwlRestriction {
                    on_property: "http://example.org/enrolledIn".to_string(),
                    restriction_type: OwlRestrictionType::ExactCardinality(1),
                },
                OwlRestriction {
                    on_property: "http://example.org/studentID".to_string(),
                    restriction_type: OwlRestrictionType::ExactCardinality(1),
                },
            ],
        },
        OwlClass {
            uri: "http://example.org/Course".to_string(),
            _label: Some("Course".to_string()),
            _comment: Some("An academic course".to_string()),
            _super_classes: vec![],
            _equivalent_classes: vec![],
            _disjoint_with: vec![],
            restrictions: vec![
                OwlRestriction {
                    on_property: "http://example.org/taughtBy".to_string(),
                    restriction_type: OwlRestrictionType::MinCardinality(1),
                },
                OwlRestriction {
                    on_property: "http://example.org/taughtBy".to_string(),
                    restriction_type: OwlRestrictionType::MaxCardinality(2),
                },
            ],
        },
    ];
    let properties = vec![
        OwlProperty {
            uri: "http://example.org/teachesAt".to_string(),
            _label: Some("teaches at".to_string()),
            _comment: Some("University where a professor teaches".to_string()),
            property_type: OwlPropertyType::Object,
            domain: vec!["http://example.org/Professor".to_string()],
            range: vec!["http://example.org/University".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/enrolledIn".to_string(),
            _label: Some("enrolled in".to_string()),
            _comment: Some("University where a student is enrolled".to_string()),
            property_type: OwlPropertyType::Object,
            domain: vec!["http://example.org/Student".to_string()],
            range: vec!["http://example.org/University".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/studentID".to_string(),
            _label: Some("student ID".to_string()),
            _comment: Some("Unique identifier for a student".to_string()),
            property_type: OwlPropertyType::Datatype,
            domain: vec!["http://example.org/Student".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: true,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/officeNumber".to_string(),
            _label: Some("office number".to_string()),
            _comment: Some("Office number of a professor".to_string()),
            property_type: OwlPropertyType::Datatype,
            domain: vec!["http://example.org/Professor".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/taughtBy".to_string(),
            _label: Some("taught by".to_string()),
            _comment: Some("Professor who teaches a course".to_string()),
            property_type: OwlPropertyType::Object,
            domain: vec!["http://example.org/Course".to_string()],
            range: vec!["http://example.org/Professor".to_string()],
            _super_properties: vec![],
            is_functional: false,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/hasColleague".to_string(),
            _label: Some("has colleague".to_string()),
            _comment: Some("Colleague relationship between professors".to_string()),
            property_type: OwlPropertyType::Object,
            domain: vec!["http://example.org/Professor".to_string()],
            range: vec!["http://example.org/Professor".to_string()],
            _super_properties: vec![],
            is_functional: false,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: true,
        },
        OwlProperty {
            uri: "http://example.org/name".to_string(),
            _label: Some("name".to_string()),
            _comment: Some("Name of a person or organization".to_string()),
            property_type: OwlPropertyType::Datatype,
            domain: vec![
                "http://example.org/Person".to_string(),
                "http://example.org/University".to_string(),
            ],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
    ];
    Ok(OwlOntology {
        classes,
        properties,
    })
}

/// Generate RDF data conforming to OWL ontology
pub(super) fn generate_from_owl_ontology<R: Rng>(
    rng: &mut R,
    ontology: &OwlOntology,
    count: usize,
) -> Result<Vec<Quad>, Box<dyn Error>> {
    let mut quads = Vec::new();
    let instances_per_class = if !ontology.classes.is_empty() {
        (count as f64 / ontology.classes.len() as f64).ceil() as usize
    } else {
        return Ok(quads);
    };
    let mut generated_instances: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for class in &ontology.classes {
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
            for property in &ontology.properties {
                if property.domain.contains(&class.uri) {
                    let cardinality = get_property_cardinality(class, &property.uri);
                    let num_values = if property.is_functional {
                        1
                    } else {
                        match cardinality {
                            Some((min, Some(max))) => rng.random_range(min..=max),
                            Some((min, None)) => rng.random_range(min..=(min + 3)),
                            None => rng.random_range(1..=2),
                        }
                    };
                    let mut used_values = std::collections::HashSet::new();
                    for _ in 0..num_values {
                        let value = generate_owl_property_value(
                            rng,
                            property,
                            &generated_instances,
                            ontology,
                            &mut used_values,
                        )?;
                        quads.push(Quad::new(
                            instance_uri.clone(),
                            NamedNode::new(&property.uri).expect("Valid IRI"),
                            value,
                            GraphName::DefaultGraph,
                        ));
                    }
                    if property.is_symmetric {}
                }
            }
        }
        generated_instances.insert(class.uri.clone(), class_instances);
    }
    Ok(quads)
}

/// Get cardinality constraints from OWL restrictions
pub(super) fn get_property_cardinality(
    class: &OwlClass,
    property_uri: &str,
) -> Option<(usize, Option<usize>)> {
    let mut min_card: Option<usize> = None;
    let mut max_card: Option<usize> = None;
    for restriction in &class.restrictions {
        if restriction.on_property == property_uri {
            match &restriction.restriction_type {
                OwlRestrictionType::MinCardinality(n) => min_card = Some(*n as usize),
                OwlRestrictionType::MaxCardinality(n) => max_card = Some(*n as usize),
                OwlRestrictionType::ExactCardinality(n) => {
                    min_card = Some(*n as usize);
                    max_card = Some(*n as usize);
                }
                _ => {}
            }
        }
    }
    if min_card.is_some() || max_card.is_some() {
        Some((min_card.unwrap_or(0), max_card))
    } else {
        None
    }
}

/// Generate a property value based on OWL property characteristics
pub(super) fn generate_owl_property_value<R: Rng>(
    rng: &mut R,
    property: &OwlProperty,
    generated_instances: &std::collections::HashMap<String, Vec<String>>,
    ontology: &OwlOntology,
    used_values: &mut std::collections::HashSet<String>,
) -> Result<Term, Box<dyn Error>> {
    match property.property_type {
        OwlPropertyType::Object => {
            let range_uri = property.range.first().ok_or("Property has no range")?;
            if ontology.classes.iter().any(|c| &c.uri == range_uri) {
                if let Some(instances) = generated_instances.get(range_uri) {
                    if !instances.is_empty() {
                        let idx = rng.random_range(0..instances.len());
                        return Ok(Term::NamedNode(
                            NamedNode::new(&instances[idx]).expect("Valid IRI"),
                        ));
                    }
                }
            }
            let class_name = range_uri.split('/').next_back().unwrap_or("resource");
            Ok(Term::NamedNode(
                NamedNode::new(format!(
                    "http://example.org/{}/{}",
                    class_name,
                    rng.random_range(0..1000)
                ))
                .expect("Valid IRI"),
            ))
        }
        OwlPropertyType::Datatype => {
            let range_uri = property.range.first().ok_or("Property has no range")?;
            let mut value_str = generate_datatype_value(rng, range_uri, &property.uri)?;
            if property.is_inverse_functional {
                let mut attempts = 0;
                while used_values.contains(&value_str) && attempts < 100 {
                    value_str = generate_datatype_value(rng, range_uri, &property.uri)?;
                    attempts += 1;
                }
                used_values.insert(value_str.clone());
            }
            Ok(Term::Literal(Literal::new_typed_literal(
                value_str,
                NamedNode::new(range_uri).expect("Valid IRI"),
            )))
        }
    }
}

/// Generate a datatype value based on range and property hints
pub(super) fn generate_datatype_value<R: Rng>(
    rng: &mut R,
    range_uri: &str,
    property_uri: &str,
) -> Result<String, Box<dyn Error>> {
    let value_str = match range_uri {
        "http://www.w3.org/2001/XMLSchema#string" => {
            if property_uri.contains("studentID") {
                format!("STU{:06}", rng.random_range(100000..999999))
            } else if property_uri.contains("officeNumber") {
                format!("Room-{:03}", rng.random_range(100..999))
            } else if property_uri.contains("name") {
                let first_names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"];
                let last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"];
                format!(
                    "{} {}",
                    first_names[rng.random_range(0..first_names.len())],
                    last_names[rng.random_range(0..last_names.len())]
                )
            } else {
                let words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"];
                let mut result = String::new();
                for _ in 0..rng.random_range(1..3) {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(words[rng.random_range(0..words.len())]);
                }
                result
            }
        }
        "http://www.w3.org/2001/XMLSchema#integer" | "http://www.w3.org/2001/XMLSchema#int" => {
            rng.random_range(1..1000).to_string()
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
        _ => format!("value_{}", rng.random_range(0..1000)),
    };
    Ok(value_str)
}
