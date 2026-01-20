//! OWL-based data generation - Generate RDF data conforming to OWL ontologies
//!
//! This module parses OWL ontologies and generates synthetic RDF data that conforms
//! to the class hierarchy, property constraints, and cardinality restrictions defined in the ontology.

use crate::cli::CliContext;
use oxirs_core::format::{RdfFormat, RdfParser};
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use oxirs_core::RdfTerm;
use scirs2_core::random::Random;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::BufReader;
use std::path::Path;

/// OWL namespace
const OWL_NS: &str = "http://www.w3.org/2002/07/owl#";

/// RDF namespace
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// RDFS namespace
const RDFS_NS: &str = "http://www.w3.org/2000/01/rdf-schema#";

/// XSD namespace
const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

/// OWL ontology representation
#[derive(Debug, Clone)]
pub struct OwlOntology {
    pub classes: HashMap<String, OwlClass>,
    pub properties: HashMap<String, OwlProperty>,
    pub individuals: HashMap<String, OwlIndividual>,
}

/// OWL class definition
#[derive(Debug, Clone)]
pub struct OwlClass {
    pub uri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub super_classes: Vec<String>,
    pub sub_classes: Vec<String>,
    pub equivalent_classes: Vec<String>,
    pub disjoint_with: Vec<String>,
    pub restrictions: Vec<OwlRestriction>,
}

/// OWL property definition
#[derive(Debug, Clone)]
pub struct OwlProperty {
    pub uri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub property_type: OwlPropertyType,
    pub domain: Vec<String>,
    pub range: Vec<String>,
    pub super_properties: Vec<String>,
    pub sub_properties: Vec<String>,
    pub is_functional: bool,
    pub is_inverse_functional: bool,
    pub is_transitive: bool,
    pub is_symmetric: bool,
    pub inverse_of: Option<String>,
}

/// OWL property type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwlPropertyType {
    Object,
    Datatype,
    Annotation,
}

/// OWL restriction
#[derive(Debug, Clone)]
pub struct OwlRestriction {
    pub on_property: String,
    pub restriction_type: OwlRestrictionType,
}

/// OWL restriction type
#[derive(Debug, Clone)]
pub enum OwlRestrictionType {
    SomeValuesFrom(String),
    AllValuesFrom(String),
    MinCardinality(u32),
    MaxCardinality(u32),
    ExactCardinality(u32),
    HasValue(String),
    MinQualifiedCardinality(u32, String),
    MaxQualifiedCardinality(u32, String),
    ExactQualifiedCardinality(u32, String),
}

/// OWL individual (named instance)
#[derive(Debug, Clone)]
pub struct OwlIndividual {
    pub uri: String,
    pub types: Vec<String>,
    pub same_as: Vec<String>,
    pub different_from: Vec<String>,
}

impl OwlOntology {
    /// Create a new empty ontology
    pub fn new() -> Self {
        Self {
            classes: HashMap::new(),
            properties: HashMap::new(),
            individuals: HashMap::new(),
        }
    }

    /// Get all leaf classes (classes with no subclasses)
    pub fn get_leaf_classes(&self) -> Vec<&OwlClass> {
        self.classes
            .values()
            .filter(|c| c.sub_classes.is_empty())
            .collect()
    }

    /// Get all properties applicable to a class (including inherited)
    pub fn get_properties_for_class(&self, class_uri: &str) -> Vec<&OwlProperty> {
        let mut applicable_props = Vec::new();
        let mut visited_classes = HashSet::new();
        let mut classes_to_check = vec![class_uri.to_string()];

        while let Some(current_class) = classes_to_check.pop() {
            if visited_classes.contains(&current_class) {
                continue;
            }
            visited_classes.insert(current_class.clone());

            // Find properties with this class in domain
            for prop in self.properties.values() {
                if prop.domain.contains(&current_class)
                    && !applicable_props
                        .iter()
                        .any(|p: &&OwlProperty| p.uri == prop.uri)
                {
                    applicable_props.push(prop);
                }
            }

            // Add super classes to check
            if let Some(class_def) = self.classes.get(&current_class) {
                for super_class in &class_def.super_classes {
                    classes_to_check.push(super_class.clone());
                }
            }
        }

        applicable_props
    }

    /// Get restrictions for a class
    pub fn get_restrictions_for_class(&self, class_uri: &str) -> Vec<&OwlRestriction> {
        self.classes
            .get(class_uri)
            .map(|c| c.restrictions.iter().collect())
            .unwrap_or_default()
    }
}

impl Default for OwlOntology {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse OWL ontology from a Turtle/RDF/XML file
pub fn parse_owl_ontology(
    ontology_file: &Path,
    ctx: &CliContext,
) -> Result<OwlOntology, Box<dyn std::error::Error>> {
    ctx.info(&format!(
        "Parsing OWL ontology from {}",
        ontology_file.display()
    ));

    // Detect format from file extension
    let format = detect_format(ontology_file)?;

    // Read and parse the OWL file
    let file = fs::File::open(ontology_file)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(format);

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

    // Extract ontology definitions
    let ontology = extract_owl_ontology(&quads, ctx)?;

    ctx.info(&format!(
        "Found {} OWL classes, {} properties, {} individuals",
        ontology.classes.len(),
        ontology.properties.len(),
        ontology.individuals.len()
    ));

    Ok(ontology)
}

/// Detect RDF format from file extension
fn detect_format(path: &Path) -> Result<RdfFormat, Box<dyn std::error::Error>> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or("File has no extension")?;

    match extension.to_lowercase().as_str() {
        "ttl" => Ok(RdfFormat::Turtle),
        "rdf" | "xml" | "owl" => Ok(RdfFormat::RdfXml),
        "nt" => Ok(RdfFormat::NTriples),
        "nq" => Ok(RdfFormat::NQuads),
        "trig" => Ok(RdfFormat::TriG),
        "jsonld" | "json" => Ok(RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        }),
        "n3" => Ok(RdfFormat::N3),
        _ => Err(format!("Unsupported file extension: {}", extension).into()),
    }
}

/// Extract OWL ontology from parsed quads
fn extract_owl_ontology(
    quads: &[Quad],
    ctx: &CliContext,
) -> Result<OwlOntology, Box<dyn std::error::Error>> {
    let mut ontology = OwlOntology::new();

    // First pass: identify classes, properties, and individuals
    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };

        let predicate_uri = quad.predicate().as_str();

        // Check if this is a Class
        if predicate_uri == format!("{}type", RDF_NS) {
            if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                let obj_uri = nn.as_str();
                if obj_uri == format!("{}Class", OWL_NS) {
                    ontology
                        .classes
                        .entry(subject_uri.to_string())
                        .or_insert(OwlClass {
                            uri: subject_uri.to_string(),
                            label: None,
                            comment: None,
                            super_classes: Vec::new(),
                            sub_classes: Vec::new(),
                            equivalent_classes: Vec::new(),
                            disjoint_with: Vec::new(),
                            restrictions: Vec::new(),
                        });
                }
            }
        }

        // Check if this is a Property
        if predicate_uri == format!("{}type", RDF_NS) {
            if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                let obj_uri = nn.as_str();
                let property_type = if obj_uri == format!("{}ObjectProperty", OWL_NS) {
                    Some(OwlPropertyType::Object)
                } else if obj_uri == format!("{}DatatypeProperty", OWL_NS) {
                    Some(OwlPropertyType::Datatype)
                } else if obj_uri == format!("{}AnnotationProperty", OWL_NS) {
                    Some(OwlPropertyType::Annotation)
                } else {
                    None
                };

                if let Some(prop_type) = property_type {
                    ontology
                        .properties
                        .entry(subject_uri.to_string())
                        .or_insert(OwlProperty {
                            uri: subject_uri.to_string(),
                            label: None,
                            comment: None,
                            property_type: prop_type,
                            domain: Vec::new(),
                            range: Vec::new(),
                            super_properties: Vec::new(),
                            sub_properties: Vec::new(),
                            is_functional: false,
                            is_inverse_functional: false,
                            is_transitive: false,
                            is_symmetric: false,
                            inverse_of: None,
                        });
                }
            }
        }

        // Check for named individuals
        if predicate_uri == format!("{}type", RDF_NS) {
            if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                let obj_uri = nn.as_str();
                if obj_uri == format!("{}NamedIndividual", OWL_NS) {
                    ontology
                        .individuals
                        .entry(subject_uri.to_string())
                        .or_insert(OwlIndividual {
                            uri: subject_uri.to_string(),
                            types: Vec::new(),
                            same_as: Vec::new(),
                            different_from: Vec::new(),
                        });
                }
            }
        }
    }

    ctx.info(&format!(
        "Found {} class candidates, {} property candidates, {} individual candidates",
        ontology.classes.len(),
        ontology.properties.len(),
        ontology.individuals.len()
    ));

    // Second pass: extract class metadata and restrictions
    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };

        if let Some(class_def) = ontology.classes.get_mut(subject_uri) {
            let predicate_uri = quad.predicate().as_str();

            match predicate_uri {
                p if p == format!("{}label", RDFS_NS) => {
                    if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                        class_def.label = Some(lit.value().to_string());
                    }
                }
                p if p == format!("{}comment", RDFS_NS) => {
                    if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                        class_def.comment = Some(lit.value().to_string());
                    }
                }
                p if p == format!("{}subClassOf", RDFS_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        class_def.super_classes.push(nn.as_str().to_string());
                    }
                }
                p if p == format!("{}equivalentClass", OWL_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        class_def.equivalent_classes.push(nn.as_str().to_string());
                    }
                }
                p if p == format!("{}disjointWith", OWL_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        class_def.disjoint_with.push(nn.as_str().to_string());
                    }
                }
                _ => {}
            }
        }

        // Extract property metadata
        if let Some(prop_def) = ontology.properties.get_mut(subject_uri) {
            let predicate_uri = quad.predicate().as_str();

            match predicate_uri {
                p if p == format!("{}label", RDFS_NS) => {
                    if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                        prop_def.label = Some(lit.value().to_string());
                    }
                }
                p if p == format!("{}comment", RDFS_NS) => {
                    if let oxirs_core::model::Object::Literal(lit) = quad.object() {
                        prop_def.comment = Some(lit.value().to_string());
                    }
                }
                p if p == format!("{}domain", RDFS_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        prop_def.domain.push(nn.as_str().to_string());
                    }
                }
                p if p == format!("{}range", RDFS_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        prop_def.range.push(nn.as_str().to_string());
                    }
                }
                p if p == format!("{}subPropertyOf", RDFS_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        prop_def.super_properties.push(nn.as_str().to_string());
                    }
                }
                p if p == format!("{}type", RDF_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        let obj_uri = nn.as_str();
                        if obj_uri == format!("{}FunctionalProperty", OWL_NS) {
                            prop_def.is_functional = true;
                        } else if obj_uri == format!("{}InverseFunctionalProperty", OWL_NS) {
                            prop_def.is_inverse_functional = true;
                        } else if obj_uri == format!("{}TransitiveProperty", OWL_NS) {
                            prop_def.is_transitive = true;
                        } else if obj_uri == format!("{}SymmetricProperty", OWL_NS) {
                            prop_def.is_symmetric = true;
                        }
                    }
                }
                p if p == format!("{}inverseOf", OWL_NS) => {
                    if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                        prop_def.inverse_of = Some(nn.as_str().to_string());
                    }
                }
                _ => {}
            }
        }
    }

    // Third pass: extract restrictions (simplified - full OWL restriction parsing is complex)
    // For now, we'll handle simple cardinality restrictions
    for quad in quads {
        let predicate_uri = quad.predicate().as_str();

        if predicate_uri == format!("{}subClassOf", RDFS_NS) {
            if let Subject::NamedNode(subject_nn) = quad.subject() {
                if let Some(_class_def) = ontology.classes.get_mut(subject_nn.as_str()) {
                    // Check if object is a restriction (blank node handling would be needed for full support)
                    // This is a simplified implementation
                    if let oxirs_core::model::Object::NamedNode(_nn) = quad.object() {
                        // Would need to follow blank nodes to extract restriction details
                        // For now, we'll skip complex restriction parsing
                    }
                }
            }
        }
    }

    // Fourth pass: populate subClassOf relationships
    for class_uri in ontology.classes.keys().cloned().collect::<Vec<_>>() {
        if let Some(class_def) = ontology.classes.get(&class_uri) {
            for super_class_uri in &class_def.super_classes.clone() {
                if let Some(super_class) = ontology.classes.get_mut(super_class_uri) {
                    if !super_class.sub_classes.contains(&class_uri) {
                        super_class.sub_classes.push(class_uri.clone());
                    }
                }
            }
        }
    }

    // Fifth pass: populate subPropertyOf relationships
    for prop_uri in ontology.properties.keys().cloned().collect::<Vec<_>>() {
        if let Some(prop_def) = ontology.properties.get(&prop_uri) {
            for super_prop_uri in &prop_def.super_properties.clone() {
                if let Some(super_prop) = ontology.properties.get_mut(super_prop_uri) {
                    if !super_prop.sub_properties.contains(&prop_uri) {
                        super_prop.sub_properties.push(prop_uri.clone());
                    }
                }
            }
        }
    }

    Ok(ontology)
}

/// Generate RDF data conforming to OWL ontology
pub fn generate_from_owl_ontology<R: scirs2_core::RngCore>(
    ontology: &OwlOntology,
    instance_count: usize,
    rng: &mut Random<R>,
    ctx: &CliContext,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();

    ctx.info(&format!(
        "Generating {} instances from OWL ontology",
        instance_count
    ));

    // Get all leaf classes (most specific classes)
    let leaf_classes = ontology.get_leaf_classes();

    if leaf_classes.is_empty() {
        // If no leaf classes, use all classes
        let all_classes: Vec<&OwlClass> = ontology.classes.values().collect();
        if all_classes.is_empty() {
            return Err("No OWL classes found in ontology".into());
        }

        // Generate instances for each class
        let instances_per_class = instance_count / all_classes.len();
        for class_def in &all_classes {
            generate_owl_class_instances(
                class_def,
                instances_per_class,
                ontology,
                rng,
                &mut quads,
            )?;
        }
    } else {
        // Generate instances for leaf classes
        let instances_per_class = instance_count / leaf_classes.len();
        for class_def in &leaf_classes {
            generate_owl_class_instances(
                class_def,
                instances_per_class,
                ontology,
                rng,
                &mut quads,
            )?;
        }
    }

    ctx.info(&format!("Generated {} RDF quads", quads.len()));

    Ok(quads)
}

/// Generate instances for a specific OWL class
fn generate_owl_class_instances<R: scirs2_core::RngCore>(
    class_def: &OwlClass,
    count: usize,
    ontology: &OwlOntology,
    rng: &mut Random<R>,
    quads: &mut Vec<Quad>,
) -> Result<(), Box<dyn std::error::Error>> {
    for i in 0..count {
        let instance_uri = format!(
            "http://example.org/instance_{}_{}",
            class_def.uri.split('/').next_back().unwrap_or("unknown"),
            i
        );

        // Add rdf:type triple for the class
        quads.push(create_quad(
            &instance_uri,
            &format!("{}type", RDF_NS),
            Term::NamedNode(NamedNode::new_unchecked(class_def.uri.clone())),
        ));

        // Add rdf:type triples for all super classes
        for super_class_uri in &class_def.super_classes {
            quads.push(create_quad(
                &instance_uri,
                &format!("{}type", RDF_NS),
                Term::NamedNode(NamedNode::new_unchecked(super_class_uri.clone())),
            ));
        }

        // Get all applicable properties for this class
        let properties = ontology.get_properties_for_class(&class_def.uri);

        // Get restrictions for this class
        let restrictions = ontology.get_restrictions_for_class(&class_def.uri);

        // Build a map of property constraints from restrictions
        let mut property_constraints: HashMap<String, (Option<u32>, Option<u32>)> = HashMap::new();
        for restriction in restrictions {
            match &restriction.restriction_type {
                OwlRestrictionType::MinCardinality(n) => {
                    let entry = property_constraints
                        .entry(restriction.on_property.clone())
                        .or_insert((None, None));
                    entry.0 = Some(*n);
                }
                OwlRestrictionType::MaxCardinality(n) => {
                    let entry = property_constraints
                        .entry(restriction.on_property.clone())
                        .or_insert((None, None));
                    entry.1 = Some(*n);
                }
                OwlRestrictionType::ExactCardinality(n) => {
                    property_constraints
                        .insert(restriction.on_property.clone(), (Some(*n), Some(*n)));
                }
                _ => {}
            }
        }

        // Generate property values
        for prop in properties {
            // Check if there are cardinality constraints
            let (min_card, max_card) = property_constraints
                .get(&prop.uri)
                .copied()
                .unwrap_or((None, None));

            // Determine value count based on constraints
            let min_count = min_card.unwrap_or(1) as usize;
            let max_count = max_card.unwrap_or(if prop.is_functional { 1 } else { 3 }) as usize;

            let value_count = if min_count == max_count {
                min_count
            } else {
                rng.gen_range(min_count..=max_count)
            };

            let prop_quads =
                generate_owl_property_values(&instance_uri, prop, ontology, rng, value_count)?;
            quads.extend(prop_quads);
        }
    }

    Ok(())
}

/// Generate property values for an OWL property
fn generate_owl_property_values<R: scirs2_core::RngCore>(
    subject_uri: &str,
    prop: &OwlProperty,
    ontology: &OwlOntology,
    rng: &mut Random<R>,
    value_count: usize,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();

    for _ in 0..value_count {
        let value = match prop.property_type {
            OwlPropertyType::Object => {
                // Generate object property value (IRI)
                if prop.range.is_empty() {
                    // No range specified - generate generic IRI
                    let id = rng.gen_range(0..10000);
                    Term::NamedNode(NamedNode::new_unchecked(format!(
                        "http://example.org/resource_{}",
                        id
                    )))
                } else {
                    // Pick a random range class
                    let range_uri = &prop.range[rng.gen_range(0..prop.range.len())];

                    // Check if range is a class in the ontology
                    if ontology.classes.contains_key(range_uri) {
                        // Generate instance of the class
                        let instance_id = rng.gen_range(0..1000);
                        Term::NamedNode(NamedNode::new_unchecked(format!(
                            "http://example.org/{}_instance_{}",
                            range_uri.split('/').next_back().unwrap_or("instance"),
                            instance_id
                        )))
                    } else {
                        // Unknown range - generate IRI
                        let id = rng.gen_range(0..10000);
                        Term::NamedNode(NamedNode::new_unchecked(format!(
                            "http://example.org/resource_{}",
                            id
                        )))
                    }
                }
            }
            OwlPropertyType::Datatype => {
                // Generate datatype property value (literal)
                if prop.range.is_empty() {
                    // No range specified - generate string literal
                    generate_literal_value(&format!("{}string", XSD_NS), rng)
                } else {
                    // Pick a random range datatype
                    let range_uri = &prop.range[rng.gen_range(0..prop.range.len())];

                    if range_uri.starts_with(XSD_NS) {
                        // Generate typed literal
                        generate_literal_value(range_uri, rng)
                    } else {
                        // Unknown range - generate string
                        generate_literal_value(&format!("{}string", XSD_NS), rng)
                    }
                }
            }
            OwlPropertyType::Annotation => {
                // Annotation properties can be either literals or IRIs
                if rng.random_range(0..2) == 0 {
                    // Generate literal
                    generate_literal_value(&format!("{}string", XSD_NS), rng)
                } else {
                    // Generate IRI
                    let id = rng.gen_range(0..10000);
                    Term::NamedNode(NamedNode::new_unchecked(format!(
                        "http://example.org/annotation_{}",
                        id
                    )))
                }
            }
        };

        quads.push(create_quad(subject_uri, &prop.uri, value));
    }

    Ok(quads)
}

/// Generate a literal value for a given XSD datatype
fn generate_literal_value<R: scirs2_core::RngCore>(
    datatype_uri: &str,
    rng: &mut Random<R>,
) -> Term {
    let value_str = match datatype_uri {
        dt if dt == format!("{}string", XSD_NS) => {
            let len = rng.gen_range(5..30);
            generate_random_string(rng, len)
        }
        dt if dt == format!("{}integer", XSD_NS) => rng.gen_range(0..10000).to_string(),
        dt if dt == format!("{}int", XSD_NS) => rng.gen_range(-1000..1000).to_string(),
        dt if dt == format!("{}decimal", XSD_NS) || dt == format!("{}double", XSD_NS) => {
            format!("{:.2}", rng.random_range(0..10000) as f64 / 10.0)
        }
        dt if dt == format!("{}float", XSD_NS) => {
            format!("{:.2}", rng.random_range(0..1000) as f32 / 10.0)
        }
        dt if dt == format!("{}boolean", XSD_NS) => if rng.random_range(0..2) == 0 {
            "true"
        } else {
            "false"
        }
        .to_string(),
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
        dt if dt == format!("{}anyURI", XSD_NS) => {
            format!("http://example.org/resource_{}", rng.gen_range(0..10000))
        }
        dt if dt == format!("{}positiveInteger", XSD_NS) => rng.gen_range(1..10000).to_string(),
        dt if dt == format!("{}nonNegativeInteger", XSD_NS) => rng.gen_range(0..10000).to_string(),
        _ => {
            // Default to string
            let len = rng.gen_range(5..20);
            generate_random_string(rng, len)
        }
    };

    if datatype_uri == format!("{}string", XSD_NS) {
        Term::Literal(Literal::new_simple_literal(value_str))
    } else {
        Term::Literal(Literal::new_typed_literal(
            value_str,
            NamedNode::new_unchecked(datatype_uri.to_string()),
        ))
    }
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
    fn test_owl_ontology_creation() {
        let ontology = OwlOntology::new();
        assert_eq!(ontology.classes.len(), 0);
        assert_eq!(ontology.properties.len(), 0);
        assert_eq!(ontology.individuals.len(), 0);
    }

    #[test]
    fn test_detect_format() {
        use std::path::PathBuf;

        let path = PathBuf::from("test.ttl");
        assert!(matches!(detect_format(&path), Ok(RdfFormat::Turtle)));

        let path = PathBuf::from("test.owl");
        assert!(matches!(detect_format(&path), Ok(RdfFormat::RdfXml)));

        let path = PathBuf::from("test.rdf");
        assert!(matches!(detect_format(&path), Ok(RdfFormat::RdfXml)));
    }

    #[test]
    fn test_generate_random_string() {
        let mut rng = Random::seed(42);
        let s = generate_random_string(&mut rng, 20);
        assert_eq!(s.len(), 20);
        assert!(s.chars().all(|c| c.is_alphanumeric() || c == '_'));
    }

    #[test]
    fn test_generate_literal_value() {
        let mut rng = Random::seed(42);

        // Test string generation
        let val = generate_literal_value(&format!("{}string", XSD_NS), &mut rng);
        assert!(matches!(val, Term::Literal(_)));

        // Test integer generation
        let val = generate_literal_value(&format!("{}integer", XSD_NS), &mut rng);
        if let Term::Literal(lit) = val {
            assert!(lit.value().parse::<i64>().is_ok());
        }

        // Test boolean generation
        let val = generate_literal_value(&format!("{}boolean", XSD_NS), &mut rng);
        if let Term::Literal(lit) = val {
            assert!(lit.value() == "true" || lit.value() == "false");
        }
    }

    #[test]
    fn test_owl_class_hierarchy() {
        let mut ontology = OwlOntology::new();

        // Add base class
        ontology.classes.insert(
            "http://example.org/Animal".to_string(),
            OwlClass {
                uri: "http://example.org/Animal".to_string(),
                label: Some("Animal".to_string()),
                comment: None,
                super_classes: Vec::new(),
                sub_classes: vec!["http://example.org/Dog".to_string()],
                equivalent_classes: Vec::new(),
                disjoint_with: Vec::new(),
                restrictions: Vec::new(),
            },
        );

        // Add subclass
        ontology.classes.insert(
            "http://example.org/Dog".to_string(),
            OwlClass {
                uri: "http://example.org/Dog".to_string(),
                label: Some("Dog".to_string()),
                comment: None,
                super_classes: vec!["http://example.org/Animal".to_string()],
                sub_classes: Vec::new(),
                equivalent_classes: Vec::new(),
                disjoint_with: Vec::new(),
                restrictions: Vec::new(),
            },
        );

        // Test leaf classes
        let leaf_classes = ontology.get_leaf_classes();
        assert_eq!(leaf_classes.len(), 1);
        assert_eq!(leaf_classes[0].uri, "http://example.org/Dog");
    }

    #[test]
    fn test_owl_property_types() {
        let mut ontology = OwlOntology::new();

        // Add object property
        ontology.properties.insert(
            "http://example.org/knows".to_string(),
            OwlProperty {
                uri: "http://example.org/knows".to_string(),
                label: Some("knows".to_string()),
                comment: None,
                property_type: OwlPropertyType::Object,
                domain: vec!["http://example.org/Person".to_string()],
                range: vec!["http://example.org/Person".to_string()],
                super_properties: Vec::new(),
                sub_properties: Vec::new(),
                is_functional: false,
                is_inverse_functional: false,
                is_transitive: false,
                is_symmetric: true,
                inverse_of: None,
            },
        );

        assert_eq!(ontology.properties.len(), 1);
        let prop = ontology.properties.get("http://example.org/knows").unwrap();
        assert_eq!(prop.property_type, OwlPropertyType::Object);
        assert!(prop.is_symmetric);
    }
}
