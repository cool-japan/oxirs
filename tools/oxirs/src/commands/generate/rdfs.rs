//! RDFS-based data generation - Generate RDF data conforming to RDFS schemas
//!
//! This module parses RDFS ontologies and generates synthetic RDF data that conforms
//! to the class hierarchy and property constraints defined in the schema.

use crate::cli::CliContext;
use oxirs_core::format::{RdfFormat, RdfParser};
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use oxirs_core::RdfTerm;
use scirs2_core::random::Random;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::BufReader;
use std::path::Path;

/// RDFS namespace
const RDFS_NS: &str = "http://www.w3.org/2000/01/rdf-schema#";

/// RDF namespace
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// XSD namespace
const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

/// RDFS schema definition
#[derive(Debug, Clone)]
pub struct RdfsSchema {
    pub classes: HashMap<String, RdfsClass>,
    pub properties: HashMap<String, RdfsProperty>,
}

/// RDFS class definition
#[derive(Debug, Clone)]
pub struct RdfsClass {
    pub uri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub super_classes: Vec<String>,
    pub sub_classes: Vec<String>,
}

/// RDFS property definition
#[derive(Debug, Clone)]
pub struct RdfsProperty {
    pub uri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub domain: Vec<String>,
    pub range: Vec<String>,
    pub super_properties: Vec<String>,
    pub sub_properties: Vec<String>,
}

impl RdfsSchema {
    /// Create a new empty schema
    pub fn new() -> Self {
        Self {
            classes: HashMap::new(),
            properties: HashMap::new(),
        }
    }

    /// Get all leaf classes (classes with no subclasses)
    pub fn get_leaf_classes(&self) -> Vec<&RdfsClass> {
        self.classes
            .values()
            .filter(|c| c.sub_classes.is_empty())
            .collect()
    }

    /// Get all properties applicable to a class (including inherited)
    pub fn get_properties_for_class(&self, class_uri: &str) -> Vec<&RdfsProperty> {
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
                        .any(|p: &&RdfsProperty| p.uri == prop.uri)
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

    /// Get range classes/datatypes for a property
    pub fn get_range_for_property(&self, prop_uri: &str) -> Vec<String> {
        self.properties
            .get(prop_uri)
            .map(|p| p.range.clone())
            .unwrap_or_default()
    }
}

impl Default for RdfsSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse RDFS schema from a Turtle/RDF/XML file
pub fn parse_rdfs_schema(
    schema_file: &Path,
    ctx: &CliContext,
) -> Result<RdfsSchema, Box<dyn std::error::Error>> {
    ctx.info(&format!(
        "Parsing RDFS schema from {}",
        schema_file.display()
    ));

    // Detect format from file extension
    let format = detect_format(schema_file)?;

    // Read and parse the RDFS file
    let file = fs::File::open(schema_file)?;
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

    // Extract schema definitions
    let schema = extract_rdfs_schema(&quads, ctx)?;

    ctx.info(&format!(
        "Found {} RDFS classes and {} properties",
        schema.classes.len(),
        schema.properties.len()
    ));

    Ok(schema)
}

/// Detect RDF format from file extension
fn detect_format(path: &Path) -> Result<RdfFormat, Box<dyn std::error::Error>> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or("File has no extension")?;

    match extension.to_lowercase().as_str() {
        "ttl" => Ok(RdfFormat::Turtle),
        "rdf" | "xml" => Ok(RdfFormat::RdfXml),
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

/// Extract RDFS schema from parsed quads
fn extract_rdfs_schema(
    quads: &[Quad],
    ctx: &CliContext,
) -> Result<RdfsSchema, Box<dyn std::error::Error>> {
    let mut schema = RdfsSchema::new();

    // First pass: identify classes and properties
    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };

        let predicate_uri = quad.predicate().as_str();

        // Check if this is a Class
        if predicate_uri == format!("{}type", RDF_NS) {
            if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                if nn.as_str() == format!("{}Class", RDFS_NS) {
                    schema
                        .classes
                        .entry(subject_uri.to_string())
                        .or_insert(RdfsClass {
                            uri: subject_uri.to_string(),
                            label: None,
                            comment: None,
                            super_classes: Vec::new(),
                            sub_classes: Vec::new(),
                        });
                }
            }
        }

        // Check if this is a Property
        if predicate_uri == format!("{}type", RDF_NS) {
            if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
                if nn.as_str() == format!("{}Property", RDF_NS)
                    || nn.as_str() == format!("{}Property", RDFS_NS)
                {
                    schema
                        .properties
                        .entry(subject_uri.to_string())
                        .or_insert(RdfsProperty {
                            uri: subject_uri.to_string(),
                            label: None,
                            comment: None,
                            domain: Vec::new(),
                            range: Vec::new(),
                            super_properties: Vec::new(),
                            sub_properties: Vec::new(),
                        });
                }
            }
        }
    }

    ctx.info(&format!(
        "Found {} class candidates and {} property candidates",
        schema.classes.len(),
        schema.properties.len()
    ));

    // Second pass: extract class and property metadata
    for quad in quads {
        let subject_uri = match quad.subject() {
            Subject::NamedNode(nn) => nn.as_str(),
            _ => continue,
        };

        let predicate_uri = quad.predicate().as_str();

        // Process class metadata
        if let Some(class_def) = schema.classes.get_mut(subject_uri) {
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
                _ => {}
            }
        }

        // Process property metadata
        if let Some(prop_def) = schema.properties.get_mut(subject_uri) {
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
                _ => {}
            }
        }
    }

    // Third pass: populate subClassOf relationships
    for class_uri in schema.classes.keys().cloned().collect::<Vec<_>>() {
        if let Some(class_def) = schema.classes.get(&class_uri) {
            for super_class_uri in &class_def.super_classes.clone() {
                if let Some(super_class) = schema.classes.get_mut(super_class_uri) {
                    if !super_class.sub_classes.contains(&class_uri) {
                        super_class.sub_classes.push(class_uri.clone());
                    }
                }
            }
        }
    }

    // Fourth pass: populate subPropertyOf relationships
    for prop_uri in schema.properties.keys().cloned().collect::<Vec<_>>() {
        if let Some(prop_def) = schema.properties.get(&prop_uri) {
            for super_prop_uri in &prop_def.super_properties.clone() {
                if let Some(super_prop) = schema.properties.get_mut(super_prop_uri) {
                    if !super_prop.sub_properties.contains(&prop_uri) {
                        super_prop.sub_properties.push(prop_uri.clone());
                    }
                }
            }
        }
    }

    Ok(schema)
}

/// Generate RDF data conforming to RDFS schema
pub fn generate_from_rdfs_schema<R: scirs2_core::RngCore>(
    schema: &RdfsSchema,
    instance_count: usize,
    rng: &mut Random<R>,
    ctx: &CliContext,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();

    ctx.info(&format!(
        "Generating {} instances from RDFS schema",
        instance_count
    ));

    // Get all leaf classes (most specific classes)
    let leaf_classes = schema.get_leaf_classes();

    if leaf_classes.is_empty() {
        // If no leaf classes, use all classes
        let all_classes: Vec<&RdfsClass> = schema.classes.values().collect();
        if all_classes.is_empty() {
            return Err("No RDFS classes found in schema".into());
        }

        // Generate instances for each class
        let instances_per_class = instance_count / all_classes.len();
        for class_def in &all_classes {
            generate_class_instances(class_def, instances_per_class, schema, rng, &mut quads)?;
        }
    } else {
        // Generate instances for leaf classes
        let instances_per_class = instance_count / leaf_classes.len();
        for class_def in &leaf_classes {
            generate_class_instances(class_def, instances_per_class, schema, rng, &mut quads)?;
        }
    }

    ctx.info(&format!("Generated {} RDF quads", quads.len()));

    Ok(quads)
}

/// Generate instances for a specific RDFS class
fn generate_class_instances<R: scirs2_core::RngCore>(
    class_def: &RdfsClass,
    count: usize,
    schema: &RdfsSchema,
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
        let properties = schema.get_properties_for_class(&class_def.uri);

        // Generate property values
        for prop in properties {
            let prop_quads = generate_rdfs_property_values(&instance_uri, prop, schema, rng)?;
            quads.extend(prop_quads);
        }
    }

    Ok(())
}

/// Generate property values for an RDFS property
fn generate_rdfs_property_values<R: scirs2_core::RngCore>(
    subject_uri: &str,
    prop: &RdfsProperty,
    schema: &RdfsSchema,
    rng: &mut Random<R>,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();

    // Generate 1-3 values for each property
    let value_count = rng.gen_range(1..=3);

    for _ in 0..value_count {
        let value = if prop.range.is_empty() {
            // No range specified - generate string literal
            generate_literal_value(&format!("{}string", XSD_NS), rng)
        } else {
            // Pick a random range
            let range_uri = &prop.range[rng.gen_range(0..prop.range.len())];

            // Check if range is a class or datatype
            if schema.classes.contains_key(range_uri) {
                // Generate instance of the class
                let instance_id = rng.gen_range(0..1000);
                Term::NamedNode(NamedNode::new_unchecked(format!(
                    "http://example.org/{}_instance_{}",
                    range_uri.split('/').next_back().unwrap_or("instance"),
                    instance_id
                )))
            } else if range_uri.starts_with(XSD_NS) {
                // Generate typed literal
                generate_literal_value(range_uri, rng)
            } else if range_uri.starts_with(RDFS_NS) && range_uri.ends_with("Literal") {
                // RDFS Literal - generate string
                generate_literal_value(&format!("{}string", XSD_NS), rng)
            } else {
                // Unknown range - generate IRI
                let id = rng.gen_range(0..10000);
                Term::NamedNode(NamedNode::new_unchecked(format!(
                    "http://example.org/resource_{}",
                    id
                )))
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
            format!("{:.2}", rng.random_range(0, 10000) as f64 / 10.0)
        }
        dt if dt == format!("{}float", XSD_NS) => {
            format!("{:.2}", rng.random_range(0, 1000) as f32 / 10.0)
        }
        dt if dt == format!("{}boolean", XSD_NS) => if rng.random_range(0, 2) == 0 {
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
        _ => {
            // Default to string
            generate_random_string(rng, 10)
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
    fn test_rdfs_schema_creation() {
        let schema = RdfsSchema::new();
        assert_eq!(schema.classes.len(), 0);
        assert_eq!(schema.properties.len(), 0);
    }

    #[test]
    fn test_detect_format() {
        use std::path::PathBuf;

        let path = PathBuf::from("test.ttl");
        assert!(matches!(detect_format(&path), Ok(RdfFormat::Turtle)));

        let path = PathBuf::from("test.rdf");
        assert!(matches!(detect_format(&path), Ok(RdfFormat::RdfXml)));

        let path = PathBuf::from("test.nt");
        assert!(matches!(detect_format(&path), Ok(RdfFormat::NTriples)));
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
    fn test_rdfs_class_hierarchy() {
        let mut schema = RdfsSchema::new();

        // Add base class
        schema.classes.insert(
            "http://example.org/Animal".to_string(),
            RdfsClass {
                uri: "http://example.org/Animal".to_string(),
                label: Some("Animal".to_string()),
                comment: None,
                super_classes: Vec::new(),
                sub_classes: vec!["http://example.org/Dog".to_string()],
            },
        );

        // Add subclass
        schema.classes.insert(
            "http://example.org/Dog".to_string(),
            RdfsClass {
                uri: "http://example.org/Dog".to_string(),
                label: Some("Dog".to_string()),
                comment: None,
                super_classes: vec!["http://example.org/Animal".to_string()],
                sub_classes: Vec::new(),
            },
        );

        // Test leaf classes
        let leaf_classes = schema.get_leaf_classes();
        assert_eq!(leaf_classes.len(), 1);
        assert_eq!(leaf_classes[0].uri, "http://example.org/Dog");
    }
}
