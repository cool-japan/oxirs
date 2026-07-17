//! OWL ontology integration for RDF generation
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
const OWL_CLASS: &str = "http://www.w3.org/2002/07/owl#Class";
const OWL_OBJECT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#ObjectProperty";
const OWL_DATATYPE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#DatatypeProperty";
const OWL_ANNOTATION_PROPERTY: &str = "http://www.w3.org/2002/07/owl#AnnotationProperty";
const OWL_FUNCTIONAL_PROPERTY: &str = "http://www.w3.org/2002/07/owl#FunctionalProperty";
const OWL_INVERSE_FUNCTIONAL_PROPERTY: &str =
    "http://www.w3.org/2002/07/owl#InverseFunctionalProperty";
const OWL_TRANSITIVE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#TransitiveProperty";
const OWL_SYMMETRIC_PROPERTY: &str = "http://www.w3.org/2002/07/owl#SymmetricProperty";
const OWL_EQUIVALENT_CLASS: &str = "http://www.w3.org/2002/07/owl#equivalentClass";
const OWL_DISJOINT_WITH: &str = "http://www.w3.org/2002/07/owl#disjointWith";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const RDFS_COMMENT: &str = "http://www.w3.org/2000/01/rdf-schema#comment";
const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
const RDFS_SUBPROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";

/// Parse an OWL ontology from the user's schema file.
///
/// Reads and parses the file with the real RDF parser, then extracts OWL class
/// and property definitions (types, domains, ranges, characteristics) from the
/// resulting quads. A missing, unparseable, or malformed file is surfaced as an
/// explicit error; there is no fallback to placeholder data.
pub(super) fn parse_owl_ontology(
    path: &std::path::PathBuf,
    ctx: &crate::cli::CliContext,
) -> Result<OwlOntology, Box<dyn Error>> {
    let format = detect_rdf_format(path)?;
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(format);

    let mut quads = Vec::new();
    for quad_result in parser.for_reader(reader) {
        quads.push(quad_result?);
    }
    ctx.info(&format!("Parsed {} RDF quads", quads.len()));

    Ok(extract_owl_ontology(&quads))
}

/// Extract OWL classes and properties from parsed quads.
fn extract_owl_ontology(quads: &[Quad]) -> OwlOntology {
    let mut classes: HashMap<String, OwlClass> = HashMap::new();
    let mut properties: HashMap<String, OwlProperty> = HashMap::new();

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
            OWL_CLASS => {
                classes
                    .entry(subject_uri.to_string())
                    .or_insert_with(|| new_owl_class(subject_uri));
            }
            OWL_OBJECT_PROPERTY => {
                properties
                    .entry(subject_uri.to_string())
                    .or_insert_with(|| new_owl_property(subject_uri, OwlPropertyType::Object));
            }
            // The generation types model only object and datatype properties;
            // annotation properties carry literal values, so treat them as such.
            OWL_DATATYPE_PROPERTY | OWL_ANNOTATION_PROPERTY => {
                properties
                    .entry(subject_uri.to_string())
                    .or_insert_with(|| new_owl_property(subject_uri, OwlPropertyType::Datatype));
            }
            _ => {}
        }
    }

    // Second pass: attach metadata, hierarchy, and property characteristics.
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
                OWL_EQUIVALENT_CLASS => {
                    if let Object::NamedNode(nn) = quad.object() {
                        class_def._equivalent_classes.push(nn.as_str().to_string());
                    }
                }
                OWL_DISJOINT_WITH => {
                    if let Object::NamedNode(nn) = quad.object() {
                        class_def._disjoint_with.push(nn.as_str().to_string());
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
                RDF_TYPE => {
                    if let Object::NamedNode(nn) = quad.object() {
                        match nn.as_str() {
                            OWL_FUNCTIONAL_PROPERTY => prop_def.is_functional = true,
                            OWL_INVERSE_FUNCTIONAL_PROPERTY => {
                                prop_def.is_inverse_functional = true
                            }
                            OWL_TRANSITIVE_PROPERTY => prop_def._is_transitive = true,
                            OWL_SYMMETRIC_PROPERTY => prop_def.is_symmetric = true,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Sort by URI so generation is reproducible for a given seed.
    let mut classes: Vec<OwlClass> = classes.into_values().collect();
    classes.sort_by(|a, b| a.uri.cmp(&b.uri));
    let mut properties: Vec<OwlProperty> = properties.into_values().collect();
    properties.sort_by(|a, b| a.uri.cmp(&b.uri));

    OwlOntology {
        classes,
        properties,
    }
}

fn new_owl_class(uri: &str) -> OwlClass {
    OwlClass {
        uri: uri.to_string(),
        _label: None,
        _comment: None,
        _super_classes: Vec::new(),
        _equivalent_classes: Vec::new(),
        _disjoint_with: Vec::new(),
        restrictions: Vec::new(),
    }
}

fn new_owl_property(uri: &str, property_type: OwlPropertyType) -> OwlProperty {
    OwlProperty {
        uri: uri.to_string(),
        _label: None,
        _comment: None,
        property_type,
        domain: Vec::new(),
        range: Vec::new(),
        _super_properties: Vec::new(),
        is_functional: false,
        is_inverse_functional: false,
        _is_transitive: false,
        is_symmetric: false,
    }
}

/// Generate RDF data conforming to OWL ontology
pub(super) fn generate_from_owl_ontology<R: RngExt>(
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
pub(super) fn generate_owl_property_value<R: RngExt>(
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
pub(super) fn generate_datatype_value<R: RngExt>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::Random;
    use std::io::Write;

    const OWL_TTL: &str = r#"
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://widgets.example/> .

ex:Widget a owl:Class ;
    rdfs:label "Widget" .
ex:Gadget a owl:Class ;
    rdfs:label "Gadget" .
ex:serialNumber a owl:DatatypeProperty ;
    rdfs:domain ex:Widget ;
    rdfs:range xsd:string .
ex:connectsTo a owl:ObjectProperty ;
    rdfs:domain ex:Widget ;
    rdfs:range ex:Gadget .
"#;

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("oxirs_gen_owl_{}_{}.ttl", std::process::id(), name));
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
    fn parses_user_owl_schema_and_generates_its_iris() {
        let ctx = crate::cli::CliContext::new();
        let path = write_temp("basic", OWL_TTL);
        let ontology = parse_owl_ontology(&path, &ctx).expect("parse owl ontology");
        std::fs::remove_file(&path).ok();

        let class_uris: Vec<&str> = ontology.classes.iter().map(|c| c.uri.as_str()).collect();
        assert!(class_uris.contains(&"http://widgets.example/Widget"));
        assert!(class_uris.contains(&"http://widgets.example/Gadget"));
        // The old placeholder ontology must never leak in.
        assert!(!class_uris
            .iter()
            .any(|u| u.contains("example.org/University")));

        let mut rng = Random::seed(7);
        let quads = generate_from_owl_ontology(&mut rng, &ontology, 8).expect("generate");
        assert!(!quads.is_empty());
        let text = dump(&quads);
        assert!(text.contains("http://widgets.example/Widget"));
        assert!(text.contains("http://widgets.example/serialNumber"));
        assert!(!text.contains("example.org/University"));
        assert!(!text.contains("foaf/0.1/Person"));
    }

    #[test]
    fn nonexistent_owl_file_is_error() {
        let ctx = crate::cli::CliContext::new();
        let mut path = std::env::temp_dir();
        path.push("oxirs_gen_owl_missing_zzz_does_not_exist.ttl");
        assert!(parse_owl_ontology(&path, &ctx).is_err());
    }

    #[test]
    fn malformed_owl_file_is_error() {
        let ctx = crate::cli::CliContext::new();
        let path = write_temp(
            "malformed",
            "@prefix ex: <http://x/> .\nex:A a owl:Class ; << broken",
        );
        let result = parse_owl_ontology(&path, &ctx);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }
}
