//! # Schema Inference CLI
//!
//! Analyzes RDF data and infers OWL/RDFS schema including class hierarchy,
//! property domains/ranges, and cardinality constraints.
//!
//! ## Features
//!
//! - **Class discovery**: Identify all classes used via rdf:type
//! - **Class hierarchy**: Infer rdfs:subClassOf relationships
//! - **Property discovery**: Find all properties and their usage patterns
//! - **Domain/range inference**: Infer rdfs:domain and rdfs:range for properties
//! - **Cardinality inference**: Min/max cardinality constraints per property per class
//! - **Schema export**: Generate OWL/RDFS Turtle output

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────
// Schema types
// ─────────────────────────────────────────────

/// Namespace constants.
pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
pub const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
pub const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
pub const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
pub const RDFS_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#Class";
pub const OWL_CLASS: &str = "http://www.w3.org/2002/07/owl#Class";
pub const OWL_OBJECT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#ObjectProperty";
pub const OWL_DATATYPE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#DatatypeProperty";

/// An RDF triple for schema inference.
#[derive(Debug, Clone)]
pub struct InferenceTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Whether the object is a literal (vs IRI/blank node).
    pub object_is_literal: bool,
    /// Datatype of the object (if literal).
    pub object_datatype: Option<String>,
}

impl InferenceTriple {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            object_is_literal: false,
            object_datatype: None,
        }
    }

    pub fn with_literal(mut self) -> Self {
        self.object_is_literal = true;
        self
    }

    pub fn with_datatype(mut self, dt: impl Into<String>) -> Self {
        self.object_is_literal = true;
        self.object_datatype = Some(dt.into());
        self
    }
}

/// An inferred class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredClass {
    /// Class IRI.
    pub iri: String,
    /// Number of instances.
    pub instance_count: usize,
    /// Inferred superclasses.
    pub superclasses: Vec<String>,
    /// Properties used by instances of this class.
    pub properties: Vec<String>,
}

/// An inferred property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredProperty {
    /// Property IRI.
    pub iri: String,
    /// Whether this is a datatype property (vs object property).
    pub is_datatype_property: bool,
    /// Inferred domain classes.
    pub domains: Vec<String>,
    /// Inferred range (class IRIs or datatype URIs).
    pub ranges: Vec<String>,
    /// Minimum cardinality observed.
    pub min_cardinality: usize,
    /// Maximum cardinality observed.
    pub max_cardinality: usize,
    /// Total usage count.
    pub usage_count: usize,
    /// Number of distinct subjects using this property.
    pub distinct_subjects: usize,
}

/// Complete inferred schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredSchema {
    /// All inferred classes.
    pub classes: Vec<InferredClass>,
    /// All inferred properties.
    pub properties: Vec<InferredProperty>,
    /// Total triples analyzed.
    pub triple_count: usize,
    /// Total distinct subjects.
    pub subject_count: usize,
    /// Total distinct predicates.
    pub predicate_count: usize,
}

/// Configuration for schema inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencerConfig {
    /// Minimum number of instances to include a class (default: 1).
    pub min_class_instances: usize,
    /// Minimum usage count to include a property (default: 1).
    pub min_property_usage: usize,
    /// Whether to infer subclass relationships (default: true).
    pub infer_hierarchy: bool,
    /// Whether to infer cardinality (default: true).
    pub infer_cardinality: bool,
    /// Whether to generate OWL or RDFS (default: OWL).
    pub use_owl: bool,
}

impl Default for InferencerConfig {
    fn default() -> Self {
        Self {
            min_class_instances: 1,
            min_property_usage: 1,
            infer_hierarchy: true,
            infer_cardinality: true,
            use_owl: true,
        }
    }
}

// ─────────────────────────────────────────────
// SchemaInferencer
// ─────────────────────────────────────────────

/// Infers schema from RDF data.
pub struct SchemaInferencer {
    config: InferencerConfig,
}

impl SchemaInferencer {
    /// Create a new inferencer with default configuration.
    pub fn new() -> Self {
        Self {
            config: InferencerConfig::default(),
        }
    }

    /// Create a new inferencer with the given configuration.
    pub fn with_config(config: InferencerConfig) -> Self {
        Self { config }
    }

    /// Infer schema from a set of triples.
    pub fn infer(&self, triples: &[InferenceTriple]) -> InferredSchema {
        // Step 1: Build subject-to-types mapping
        let mut subject_types: HashMap<&str, HashSet<&str>> = HashMap::new();
        for t in triples {
            if t.predicate == RDF_TYPE {
                subject_types
                    .entry(t.subject.as_str())
                    .or_default()
                    .insert(t.object.as_str());
            }
        }

        // Step 2: Discover classes
        let mut class_instances: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (subject, types) in &subject_types {
            for &class_iri in types {
                class_instances
                    .entry(class_iri)
                    .or_default()
                    .insert(subject);
            }
        }

        // Step 3: Discover properties and their usage
        let mut property_subjects: HashMap<&str, HashSet<&str>> = HashMap::new();
        let mut property_objects: HashMap<&str, HashSet<&str>> = HashMap::new();
        let mut property_is_literal: HashMap<&str, (usize, usize)> = HashMap::new(); // (literal_count, iri_count)
        let mut property_datatypes: HashMap<&str, HashSet<String>> = HashMap::new();
        let mut subject_property_count: HashMap<(&str, &str), usize> = HashMap::new();

        for t in triples {
            if t.predicate == RDF_TYPE || t.predicate == RDFS_SUBCLASS_OF {
                continue;
            }

            property_subjects
                .entry(t.predicate.as_str())
                .or_default()
                .insert(t.subject.as_str());

            if !t.object_is_literal {
                property_objects
                    .entry(t.predicate.as_str())
                    .or_default()
                    .insert(t.object.as_str());
            }

            let counts = property_is_literal
                .entry(t.predicate.as_str())
                .or_insert((0, 0));
            if t.object_is_literal {
                counts.0 += 1;
            } else {
                counts.1 += 1;
            }

            if let Some(dt) = &t.object_datatype {
                property_datatypes
                    .entry(t.predicate.as_str())
                    .or_default()
                    .insert(dt.clone());
            }

            *subject_property_count
                .entry((t.subject.as_str(), t.predicate.as_str()))
                .or_insert(0) += 1;
        }

        // Step 4: Infer domains and ranges
        let mut inferred_properties = Vec::new();
        for (&prop, subjects) in &property_subjects {
            let usage_count = subjects.len();
            if usage_count < self.config.min_property_usage {
                continue;
            }

            // Determine if datatype or object property
            let (lit_count, iri_count) = property_is_literal.get(prop).copied().unwrap_or((0, 0));
            let is_datatype = lit_count > iri_count;

            // Infer domain: classes that all subjects belong to
            let domains = self.infer_domains(prop, subjects, &subject_types);

            // Infer range
            let ranges = if is_datatype {
                // Use datatypes
                property_datatypes
                    .get(prop)
                    .map(|dts| dts.iter().cloned().collect::<Vec<_>>())
                    .unwrap_or_default()
            } else {
                // Infer from object types
                self.infer_range_classes(prop, &property_objects, &subject_types)
            };

            // Cardinality
            let (min_card, max_card) = if self.config.infer_cardinality {
                self.infer_cardinality(prop, &subject_property_count, subjects)
            } else {
                (0, 0)
            };

            inferred_properties.push(InferredProperty {
                iri: prop.to_string(),
                is_datatype_property: is_datatype,
                domains,
                ranges,
                min_cardinality: min_card,
                max_cardinality: max_card,
                usage_count: subject_property_count
                    .iter()
                    .filter(|((_, p), _)| *p == prop)
                    .map(|(_, &c)| c)
                    .sum(),
                distinct_subjects: subjects.len(),
            });
        }

        // Step 5: Build class information
        let mut inferred_classes = Vec::new();
        for (&class_iri, instances) in &class_instances {
            if instances.len() < self.config.min_class_instances {
                continue;
            }

            let properties: Vec<String> = property_subjects
                .iter()
                .filter(|(_, subs)| instances.iter().any(|i| subs.contains(i)))
                .map(|(&p, _)| p.to_string())
                .collect();

            let superclasses = if self.config.infer_hierarchy {
                self.infer_superclasses(class_iri, triples)
            } else {
                Vec::new()
            };

            inferred_classes.push(InferredClass {
                iri: class_iri.to_string(),
                instance_count: instances.len(),
                superclasses,
                properties,
            });
        }

        // Sort for deterministic output
        inferred_classes.sort_by(|a, b| a.iri.cmp(&b.iri));
        inferred_properties.sort_by(|a, b| a.iri.cmp(&b.iri));

        let subjects: HashSet<&str> = triples.iter().map(|t| t.subject.as_str()).collect();
        let predicates: HashSet<&str> = triples.iter().map(|t| t.predicate.as_str()).collect();

        InferredSchema {
            classes: inferred_classes,
            properties: inferred_properties,
            triple_count: triples.len(),
            subject_count: subjects.len(),
            predicate_count: predicates.len(),
        }
    }

    /// Export the inferred schema as Turtle.
    pub fn to_turtle(&self, schema: &InferredSchema) -> String {
        let mut ttl = String::new();

        ttl.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        ttl.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        ttl.push_str("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n");
        ttl.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

        // Classes
        for class in &schema.classes {
            let class_type = if self.config.use_owl {
                "owl:Class"
            } else {
                "rdfs:Class"
            };
            ttl.push_str(&format!("<{}> a {} .\n", class.iri, class_type));
            for sc in &class.superclasses {
                ttl.push_str(&format!("<{}> rdfs:subClassOf <{}> .\n", class.iri, sc));
            }
        }

        if !schema.classes.is_empty() {
            ttl.push('\n');
        }

        // Properties
        for prop in &schema.properties {
            let prop_type = if self.config.use_owl {
                if prop.is_datatype_property {
                    "owl:DatatypeProperty"
                } else {
                    "owl:ObjectProperty"
                }
            } else {
                "rdf:Property"
            };
            ttl.push_str(&format!("<{}> a {} .\n", prop.iri, prop_type));
            for domain in &prop.domains {
                ttl.push_str(&format!("<{}> rdfs:domain <{}> .\n", prop.iri, domain));
            }
            for range in &prop.ranges {
                ttl.push_str(&format!("<{}> rdfs:range <{}> .\n", prop.iri, range));
            }
        }

        ttl
    }

    // ─── Internal inference methods ──────────────────────

    fn infer_domains<'a>(
        &self,
        _prop: &str,
        subjects: &HashSet<&'a str>,
        subject_types: &HashMap<&'a str, HashSet<&'a str>>,
    ) -> Vec<String> {
        // Find classes that the majority of subjects belong to
        let mut class_counts: HashMap<&str, usize> = HashMap::new();
        for &subj in subjects {
            if let Some(types) = subject_types.get(subj) {
                for &t in types {
                    *class_counts.entry(t).or_insert(0) += 1;
                }
            }
        }

        let threshold = (subjects.len() as f64 * 0.8) as usize;
        let mut domains: Vec<String> = class_counts
            .iter()
            .filter(|(_, &count)| count >= threshold.max(1))
            .map(|(&class, _)| class.to_string())
            .collect();
        domains.sort();
        domains
    }

    fn infer_range_classes<'a>(
        &self,
        prop: &str,
        property_objects: &HashMap<&'a str, HashSet<&'a str>>,
        subject_types: &HashMap<&'a str, HashSet<&'a str>>,
    ) -> Vec<String> {
        let objects = match property_objects.get(prop) {
            Some(objs) => objs,
            None => return Vec::new(),
        };

        let mut class_counts: HashMap<&str, usize> = HashMap::new();
        for &obj in objects {
            if let Some(types) = subject_types.get(obj) {
                for &t in types {
                    *class_counts.entry(t).or_insert(0) += 1;
                }
            }
        }

        let threshold = (objects.len() as f64 * 0.5) as usize;
        let mut ranges: Vec<String> = class_counts
            .iter()
            .filter(|(_, &count)| count >= threshold.max(1))
            .map(|(&class, _)| class.to_string())
            .collect();
        ranges.sort();
        ranges
    }

    fn infer_cardinality<'a>(
        &self,
        prop: &str,
        subject_property_count: &HashMap<(&'a str, &'a str), usize>,
        subjects: &HashSet<&'a str>,
    ) -> (usize, usize) {
        let counts: Vec<usize> = subjects
            .iter()
            .map(|&s| subject_property_count.get(&(s, prop)).copied().unwrap_or(0))
            .filter(|&c| c > 0)
            .collect();

        if counts.is_empty() {
            return (0, 0);
        }

        let min = counts.iter().copied().min().unwrap_or(0);
        let max = counts.iter().copied().max().unwrap_or(0);
        (min, max)
    }

    fn infer_superclasses(&self, class_iri: &str, triples: &[InferenceTriple]) -> Vec<String> {
        triples
            .iter()
            .filter(|t| t.subject == class_iri && t.predicate == RDFS_SUBCLASS_OF)
            .map(|t| t.object.clone())
            .collect()
    }
}

impl Default for SchemaInferencer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
    const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";

    fn sample_triples() -> Vec<InferenceTriple> {
        vec![
            // Types
            InferenceTriple::new("ex:alice", RDF_TYPE, "ex:Person"),
            InferenceTriple::new("ex:bob", RDF_TYPE, "ex:Person"),
            InferenceTriple::new("ex:charlie", RDF_TYPE, "ex:Person"),
            InferenceTriple::new("ex:acme", RDF_TYPE, "ex:Organization"),
            // Properties
            InferenceTriple::new("ex:alice", "ex:name", "Alice").with_datatype(XSD_STRING),
            InferenceTriple::new("ex:bob", "ex:name", "Bob").with_datatype(XSD_STRING),
            InferenceTriple::new("ex:charlie", "ex:name", "Charlie").with_datatype(XSD_STRING),
            InferenceTriple::new("ex:alice", "ex:age", "30").with_datatype(XSD_INTEGER),
            InferenceTriple::new("ex:bob", "ex:age", "25").with_datatype(XSD_INTEGER),
            InferenceTriple::new("ex:alice", "ex:knows", "ex:bob"),
            InferenceTriple::new("ex:bob", "ex:knows", "ex:charlie"),
            InferenceTriple::new("ex:alice", "ex:worksFor", "ex:acme"),
        ]
    }

    // ═══ Config tests ════════════════════════════════════

    #[test]
    fn test_default_config() {
        let config = InferencerConfig::default();
        assert_eq!(config.min_class_instances, 1);
        assert!(config.infer_hierarchy);
        assert!(config.use_owl);
    }

    #[test]
    fn test_custom_config() {
        let config = InferencerConfig {
            min_class_instances: 5,
            use_owl: false,
            ..Default::default()
        };
        assert_eq!(config.min_class_instances, 5);
        assert!(!config.use_owl);
    }

    // ═══ Triple construction tests ═══════════════════════

    #[test]
    fn test_inference_triple() {
        let t = InferenceTriple::new("s", "p", "o");
        assert!(!t.object_is_literal);
        assert!(t.object_datatype.is_none());
    }

    #[test]
    fn test_inference_triple_literal() {
        let t = InferenceTriple::new("s", "p", "hello").with_literal();
        assert!(t.object_is_literal);
    }

    #[test]
    fn test_inference_triple_datatype() {
        let t = InferenceTriple::new("s", "p", "42").with_datatype(XSD_INTEGER);
        assert!(t.object_is_literal);
        assert_eq!(t.object_datatype, Some(XSD_INTEGER.to_string()));
    }

    // ═══ Class discovery tests ═══════════════════════════

    #[test]
    fn test_discover_classes() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        assert!(schema.classes.len() >= 2);
        let class_iris: Vec<&str> = schema.classes.iter().map(|c| c.iri.as_str()).collect();
        assert!(class_iris.contains(&"ex:Person"));
        assert!(class_iris.contains(&"ex:Organization"));
    }

    #[test]
    fn test_class_instance_count() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let person = schema.classes.iter().find(|c| c.iri == "ex:Person");
        assert!(person.is_some());
        assert_eq!(person.expect("person").instance_count, 3);
    }

    #[test]
    fn test_class_min_instances_filter() {
        let config = InferencerConfig {
            min_class_instances: 2,
            ..Default::default()
        };
        let inferencer = SchemaInferencer::with_config(config);
        let schema = inferencer.infer(&sample_triples());
        // Organization has 1 instance, should be filtered
        let org = schema.classes.iter().find(|c| c.iri == "ex:Organization");
        assert!(org.is_none());
    }

    // ═══ Property discovery tests ════════════════════════

    #[test]
    fn test_discover_properties() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        assert!(!schema.properties.is_empty());
        let prop_iris: Vec<&str> = schema.properties.iter().map(|p| p.iri.as_str()).collect();
        assert!(prop_iris.contains(&"ex:name"));
        assert!(prop_iris.contains(&"ex:knows"));
    }

    #[test]
    fn test_datatype_property_detection() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let name_prop = schema.properties.iter().find(|p| p.iri == "ex:name");
        assert!(name_prop.is_some());
        assert!(name_prop.expect("name").is_datatype_property);
    }

    #[test]
    fn test_object_property_detection() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let knows_prop = schema.properties.iter().find(|p| p.iri == "ex:knows");
        assert!(knows_prop.is_some());
        assert!(!knows_prop.expect("knows").is_datatype_property);
    }

    // ═══ Domain inference tests ══════════════════════════

    #[test]
    fn test_infer_domain() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let name_prop = schema
            .properties
            .iter()
            .find(|p| p.iri == "ex:name")
            .expect("name");
        assert!(!name_prop.domains.is_empty());
        assert!(name_prop.domains.contains(&"ex:Person".to_string()));
    }

    // ═══ Range inference tests ═══════════════════════════

    #[test]
    fn test_infer_datatype_range() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let name_prop = schema
            .properties
            .iter()
            .find(|p| p.iri == "ex:name")
            .expect("name");
        assert!(!name_prop.ranges.is_empty());
        assert!(name_prop.ranges.iter().any(|r| r.contains("string")));
    }

    #[test]
    fn test_infer_object_range() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let knows_prop = schema
            .properties
            .iter()
            .find(|p| p.iri == "ex:knows")
            .expect("knows");
        // Bob and Charlie are persons, so range should include Person
        assert!(knows_prop.ranges.contains(&"ex:Person".to_string()));
    }

    // ═══ Cardinality inference tests ═════════════════════

    #[test]
    fn test_cardinality_name() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let name_prop = schema
            .properties
            .iter()
            .find(|p| p.iri == "ex:name")
            .expect("name");
        // All persons have exactly 1 name
        assert_eq!(name_prop.min_cardinality, 1);
        assert_eq!(name_prop.max_cardinality, 1);
    }

    #[test]
    fn test_cardinality_knows() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let knows_prop = schema
            .properties
            .iter()
            .find(|p| p.iri == "ex:knows")
            .expect("knows");
        // alice knows bob, bob knows charlie => each has 1
        assert_eq!(knows_prop.min_cardinality, 1);
        assert_eq!(knows_prop.max_cardinality, 1);
    }

    // ═══ Schema statistics tests ═════════════════════════

    #[test]
    fn test_schema_triple_count() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        assert_eq!(schema.triple_count, 12);
    }

    #[test]
    fn test_schema_subject_count() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        // alice, bob, charlie, acme
        assert_eq!(schema.subject_count, 4);
    }

    // ═══ Turtle export tests ═════════════════════════════

    #[test]
    fn test_turtle_export_owl() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let ttl = inferencer.to_turtle(&schema);
        assert!(ttl.contains("@prefix owl:"));
        assert!(ttl.contains("owl:Class"));
        assert!(ttl.contains("owl:DatatypeProperty"));
    }

    #[test]
    fn test_turtle_export_rdfs() {
        let config = InferencerConfig {
            use_owl: false,
            ..Default::default()
        };
        let inferencer = SchemaInferencer::with_config(config);
        let schema = inferencer.infer(&sample_triples());
        let ttl = inferencer.to_turtle(&schema);
        assert!(ttl.contains("rdfs:Class"));
        assert!(ttl.contains("rdf:Property"));
    }

    #[test]
    fn test_turtle_contains_domain_range() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let ttl = inferencer.to_turtle(&schema);
        assert!(ttl.contains("rdfs:domain"));
        assert!(ttl.contains("rdfs:range"));
    }

    // ═══ Hierarchy inference tests ═══════════════════════

    #[test]
    fn test_infer_superclass() {
        let mut triples = sample_triples();
        triples.push(InferenceTriple::new(
            "ex:Person",
            RDFS_SUBCLASS_OF,
            "ex:Agent",
        ));
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&triples);
        let person = schema.classes.iter().find(|c| c.iri == "ex:Person");
        assert!(person.is_some());
        assert!(person
            .expect("person")
            .superclasses
            .contains(&"ex:Agent".to_string()));
    }

    #[test]
    fn test_no_hierarchy_config() {
        let config = InferencerConfig {
            infer_hierarchy: false,
            ..Default::default()
        };
        let mut triples = sample_triples();
        triples.push(InferenceTriple::new(
            "ex:Person",
            RDFS_SUBCLASS_OF,
            "ex:Agent",
        ));
        let inferencer = SchemaInferencer::with_config(config);
        let schema = inferencer.infer(&triples);
        let person = schema.classes.iter().find(|c| c.iri == "ex:Person");
        assert!(person.is_some());
        assert!(person.expect("person").superclasses.is_empty());
    }

    // ═══ Empty input tests ═══════════════════════════════

    #[test]
    fn test_infer_empty() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&[]);
        assert!(schema.classes.is_empty());
        assert!(schema.properties.is_empty());
        assert_eq!(schema.triple_count, 0);
    }

    // ═══ Default impl test ═══════════════════════════════

    #[test]
    fn test_default_impl() {
        let inferencer = SchemaInferencer::default();
        let schema = inferencer.infer(&[]);
        assert!(schema.classes.is_empty());
    }

    // ═══ Property usage count test ═══════════════════════

    #[test]
    fn test_property_usage_count() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let name_prop = schema
            .properties
            .iter()
            .find(|p| p.iri == "ex:name")
            .expect("name");
        assert_eq!(name_prop.usage_count, 3);
        assert_eq!(name_prop.distinct_subjects, 3);
    }

    // ═══ Class properties test ═══════════════════════════

    #[test]
    fn test_class_properties() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        let person = schema
            .classes
            .iter()
            .find(|c| c.iri == "ex:Person")
            .expect("person");
        assert!(person.properties.contains(&"ex:name".to_string()));
        assert!(person.properties.contains(&"ex:knows".to_string()));
    }

    // ═══ Sorted output test ══════════════════════════════

    #[test]
    fn test_classes_sorted() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        for w in schema.classes.windows(2) {
            assert!(w[0].iri <= w[1].iri);
        }
    }

    #[test]
    fn test_properties_sorted() {
        let inferencer = SchemaInferencer::new();
        let schema = inferencer.infer(&sample_triples());
        for w in schema.properties.windows(2) {
            assert!(w[0].iri <= w[1].iri);
        }
    }
}
