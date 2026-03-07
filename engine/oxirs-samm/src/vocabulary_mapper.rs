//! SAMM vocabulary mapper: converts SAMM model elements to RDF/OWL triples.
//!
//! Maps SAMM Aspect Model (SAMMv2) vocabulary to standard RDF/RDFS/OWL/XSD IRIs
//! and provides a minimal Turtle serialiser for the resulting triples.

// ── Constraint types ──────────────────────────────────────────────────────────

/// The specific kind of constraint applied to a SAMM characteristic.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// Numeric range with optional min/max.
    Range {
        /// Optional inclusive minimum value.
        min: Option<f64>,
        /// Optional inclusive maximum value.
        max: Option<f64>,
    },
    /// Encoding constraint (e.g. "UTF-8").
    Encoding(String),
    /// Language tag constraint (e.g. "en").
    Language(String),
    /// String/collection length with optional min/max.
    Length {
        /// Optional minimum length.
        min: Option<u64>,
        /// Optional maximum length.
        max: Option<u64>,
    },
    /// Regular expression pattern.
    Pattern(String),
}

// ── SAMM element types ────────────────────────────────────────────────────────

/// A single SAMM model element.
#[derive(Debug, Clone)]
pub enum SammElement {
    /// An Aspect with optional properties and operations.
    Aspect {
        /// Local name of the Aspect.
        name: String,
        /// Local names of the Aspect's properties.
        properties: Vec<String>,
        /// Local names of the Aspect's operations.
        operations: Vec<String>,
    },
    /// A Property with a characteristic and optional flag.
    Property {
        /// Local name of the property.
        name: String,
        /// Local name of the property's characteristic.
        characteristic: String,
        /// Whether the property is optional.
        optional: bool,
    },
    /// A Characteristic with optional base type and enumeration values.
    Characteristic {
        /// Local name of the characteristic.
        name: String,
        /// Optional XSD/RDF datatype IRI.
        base_type: Option<String>,
        /// Optional enumeration values.
        values: Option<Vec<String>>,
    },
    /// An Operation with typed input parameters and optional output.
    Operation {
        /// Local name of the operation.
        name: String,
        /// Local names of input parameters.
        input: Vec<String>,
        /// Optional local name of the output parameter.
        output: Option<String>,
    },
    /// An Entity with properties.
    Entity {
        /// Local name of the entity.
        name: String,
        /// Local names of the entity's properties.
        properties: Vec<String>,
    },
    /// A Constraint of the given type.
    Constraint {
        /// Local name of the constraint.
        name: String,
        /// The constraint variant.
        constraint_type: ConstraintType,
    },
}

// ── RDF triple ────────────────────────────────────────────────────────────────

/// A single RDF triple (subject/predicate/object as prefixed strings).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfTriple {
    /// The subject term (prefixed IRI or blank node).
    pub subject: String,
    /// The predicate term (prefixed IRI).
    pub predicate: String,
    /// The object term (prefixed IRI, blank node, or literal).
    pub object: String,
}

impl RdfTriple {
    fn new(s: &str, p: &str, o: &str) -> Self {
        RdfTriple {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
        }
    }
}

// ── Vocabulary mapper ─────────────────────────────────────────────────────────

/// Maps SAMM elements to RDF triples using SAMM/OWL/RDF vocabulary.
pub struct VocabularyMapper {
    base_iri: String,
}

// Standard namespace prefixes.
const RDF_TYPE: &str = "rdf:type";
const RDFS_LABEL: &str = "rdfs:label";
const OWL_CLASS: &str = "owl:Class";
const OWL_DATATYPE_PROPERTY: &str = "owl:DatatypeProperty";
const OWL_OBJECT_PROPERTY: &str = "owl:ObjectProperty";
const SAMM_ASPECT: &str = "samm:Aspect";
const SAMM_PROPERTY: &str = "samm:Property";
const SAMM_CHARACTERISTIC: &str = "samm:Characteristic";
const SAMM_OPERATION: &str = "samm:Operation";
const SAMM_ENTITY: &str = "samm:Entity";
const SAMM_CONSTRAINT: &str = "samm:Constraint";
const SAMM_OPTIONAL: &str = "samm:optional";
const SAMM_INPUT: &str = "samm:input";
const SAMM_OUTPUT: &str = "samm:output";
const SAMM_BASE_CHARACTERISTIC: &str = "samm:baseCharacteristic";
const SAMM_DATA_TYPE: &str = "samm:dataType";
const SAMM_VALUES: &str = "samm:values";
const SAMM_MIN_VALUE: &str = "samm:minValue";
const SAMM_MAX_VALUE: &str = "samm:maxValue";
const SAMM_MIN_LENGTH: &str = "samm:minLength";
const SAMM_MAX_LENGTH: &str = "samm:maxLength";
const SAMM_VALUE: &str = "samm:value";
const SAMM_LANGUAGE_CODE: &str = "samm:languageCode";
const SAMM_ENCODING: &str = "samm:encoding";
const SAMM_PATTERN_VALUE: &str = "samm:patternValue";
const SAMM_HAS_PROPERTY: &str = "samm:property";
const SAMM_OPERATION_PROP: &str = "samm:operation";
const RDFS_SUB_CLASS_OF: &str = "rdfs:subClassOf";
const XSD_BOOLEAN: &str = "xsd:boolean";
const XSD_DOUBLE: &str = "xsd:double";
const XSD_INTEGER: &str = "xsd:integer";
const XSD_STRING: &str = "xsd:string";

impl VocabularyMapper {
    /// Create a mapper with the given base IRI (e.g. `"https://example.org/ns#"`).
    pub fn new(base_iri: &str) -> Self {
        VocabularyMapper {
            base_iri: base_iri.to_string(),
        }
    }

    /// Dispatch to the appropriate mapping method.
    pub fn map_element(&self, element: &SammElement) -> Vec<RdfTriple> {
        match element {
            SammElement::Aspect { .. } => self.map_aspect(element),
            SammElement::Property { .. } => self.map_property(element),
            SammElement::Characteristic { .. } => self.map_characteristic(element),
            SammElement::Operation { .. } => self.map_operation(element),
            SammElement::Entity { .. } => self.map_entity(element),
            SammElement::Constraint { .. } => self.map_constraint(element),
        }
    }

    /// Map a SAMM Aspect to RDF triples.
    pub fn map_aspect(&self, element: &SammElement) -> Vec<RdfTriple> {
        let (name, properties, operations) = match element {
            SammElement::Aspect {
                name,
                properties,
                operations,
            } => (name, properties, operations),
            _ => return Vec::new(),
        };

        let subj = self.iri(name);
        let mut triples = vec![
            RdfTriple::new(&subj, RDF_TYPE, SAMM_ASPECT),
            RdfTriple::new(&subj, RDF_TYPE, OWL_CLASS),
            RdfTriple::new(&subj, RDFS_LABEL, &format!("\"{}\"", name)),
        ];

        for prop in properties {
            triples.push(RdfTriple::new(&subj, SAMM_HAS_PROPERTY, &self.iri(prop)));
        }
        for op in operations {
            triples.push(RdfTriple::new(&subj, SAMM_OPERATION_PROP, &self.iri(op)));
        }

        triples
    }

    /// Map a SAMM Property to RDF triples.
    pub fn map_property(&self, element: &SammElement) -> Vec<RdfTriple> {
        let (name, characteristic, optional) = match element {
            SammElement::Property {
                name,
                characteristic,
                optional,
            } => (name, characteristic, optional),
            _ => return Vec::new(),
        };

        let subj = self.iri(name);
        let mut triples = vec![
            RdfTriple::new(&subj, RDF_TYPE, SAMM_PROPERTY),
            RdfTriple::new(&subj, RDF_TYPE, OWL_DATATYPE_PROPERTY),
            RdfTriple::new(&subj, RDFS_LABEL, &format!("\"{}\"", name)),
            RdfTriple::new(&subj, SAMM_BASE_CHARACTERISTIC, &self.iri(characteristic)),
            RdfTriple::new(
                &subj,
                SAMM_OPTIONAL,
                &format!("\"{}\"^^{}", optional, XSD_BOOLEAN),
            ),
        ];

        if *optional {
            triples.push(RdfTriple::new(
                &subj,
                SAMM_OPTIONAL,
                &format!("\"true\"^^{}", XSD_BOOLEAN),
            ));
        }

        triples
    }

    /// Map a SAMM Characteristic to RDF triples.
    pub fn map_characteristic(&self, element: &SammElement) -> Vec<RdfTriple> {
        let (name, base_type, values) = match element {
            SammElement::Characteristic {
                name,
                base_type,
                values,
            } => (name, base_type, values),
            _ => return Vec::new(),
        };

        let subj = self.iri(name);
        let mut triples = vec![
            RdfTriple::new(&subj, RDF_TYPE, SAMM_CHARACTERISTIC),
            RdfTriple::new(&subj, RDFS_LABEL, &format!("\"{}\"", name)),
        ];

        if let Some(bt) = base_type {
            triples.push(RdfTriple::new(&subj, SAMM_DATA_TYPE, bt));
        }

        if let Some(vals) = values {
            let list_node = format!("_:list_{}", name);
            triples.push(RdfTriple::new(&subj, SAMM_VALUES, &list_node));
            for (i, v) in vals.iter().enumerate() {
                let item_node = format!("_:item_{}_{}", name, i);
                triples.push(RdfTriple::new(&list_node, "rdf:rest", &item_node));
                triples.push(RdfTriple::new(
                    &item_node,
                    SAMM_VALUE,
                    &format!("\"{}\"", v),
                ));
            }
        }

        triples
    }

    /// Map a SAMM Operation to RDF triples.
    fn map_operation(&self, element: &SammElement) -> Vec<RdfTriple> {
        let (name, input, output) = match element {
            SammElement::Operation {
                name,
                input,
                output,
            } => (name, input, output),
            _ => return Vec::new(),
        };

        let subj = self.iri(name);
        let mut triples = vec![
            RdfTriple::new(&subj, RDF_TYPE, SAMM_OPERATION),
            RdfTriple::new(&subj, RDF_TYPE, OWL_OBJECT_PROPERTY),
            RdfTriple::new(&subj, RDFS_LABEL, &format!("\"{}\"", name)),
        ];

        for param in input {
            triples.push(RdfTriple::new(&subj, SAMM_INPUT, &self.iri(param)));
        }

        if let Some(out) = output {
            triples.push(RdfTriple::new(&subj, SAMM_OUTPUT, &self.iri(out)));
        }

        triples
    }

    /// Map a SAMM Entity to RDF triples.
    fn map_entity(&self, element: &SammElement) -> Vec<RdfTriple> {
        let (name, properties) = match element {
            SammElement::Entity { name, properties } => (name, properties),
            _ => return Vec::new(),
        };

        let subj = self.iri(name);
        let mut triples = vec![
            RdfTriple::new(&subj, RDF_TYPE, SAMM_ENTITY),
            RdfTriple::new(&subj, RDF_TYPE, OWL_CLASS),
            RdfTriple::new(&subj, RDFS_LABEL, &format!("\"{}\"", name)),
        ];

        for prop in properties {
            triples.push(RdfTriple::new(&subj, SAMM_HAS_PROPERTY, &self.iri(prop)));
        }

        triples
    }

    /// Map a SAMM Constraint to RDF triples.
    pub fn map_constraint(&self, element: &SammElement) -> Vec<RdfTriple> {
        let (name, constraint_type) = match element {
            SammElement::Constraint {
                name,
                constraint_type,
            } => (name, constraint_type),
            _ => return Vec::new(),
        };

        let subj = self.iri(name);
        let mut triples = vec![
            RdfTriple::new(&subj, RDF_TYPE, SAMM_CONSTRAINT),
            RdfTriple::new(&subj, RDFS_LABEL, &format!("\"{}\"", name)),
        ];

        match constraint_type {
            ConstraintType::Range { min, max } => {
                if let Some(min_val) = min {
                    triples.push(RdfTriple::new(
                        &subj,
                        SAMM_MIN_VALUE,
                        &format!("\"{}\"^^{}", min_val, XSD_DOUBLE),
                    ));
                }
                if let Some(max_val) = max {
                    triples.push(RdfTriple::new(
                        &subj,
                        SAMM_MAX_VALUE,
                        &format!("\"{}\"^^{}", max_val, XSD_DOUBLE),
                    ));
                }
            }
            ConstraintType::Encoding(enc) => {
                triples.push(RdfTriple::new(
                    &subj,
                    SAMM_ENCODING,
                    &format!("\"{}\"^^{}", enc, XSD_STRING),
                ));
            }
            ConstraintType::Language(lang) => {
                triples.push(RdfTriple::new(
                    &subj,
                    SAMM_LANGUAGE_CODE,
                    &format!("\"{}\"^^{}", lang, XSD_STRING),
                ));
            }
            ConstraintType::Length { min, max } => {
                if let Some(min_len) = min {
                    triples.push(RdfTriple::new(
                        &subj,
                        SAMM_MIN_LENGTH,
                        &format!("\"{}\"^^{}", min_len, XSD_INTEGER),
                    ));
                }
                if let Some(max_len) = max {
                    triples.push(RdfTriple::new(
                        &subj,
                        SAMM_MAX_LENGTH,
                        &format!("\"{}\"^^{}", max_len, XSD_INTEGER),
                    ));
                }
            }
            ConstraintType::Pattern(pat) => {
                triples.push(RdfTriple::new(
                    &subj,
                    SAMM_PATTERN_VALUE,
                    &format!("\"{}\"^^{}", pat, XSD_STRING),
                ));
            }
        }

        triples
    }

    /// Serialize a slice of triples as Turtle with standard prefix declarations.
    pub fn render_turtle(&self, triples: &[RdfTriple]) -> String {
        let mut out = String::new();
        // Prefix declarations
        out.push_str(&format!("@prefix : <{}> .\n", self.base_iri));
        out.push_str("@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.0.0#> .\n");
        out.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        out.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        out.push_str("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n");
        out.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");
        out.push('\n');

        for t in triples {
            let subj = Self::turtle_term(&t.subject);
            let pred = Self::turtle_term(&t.predicate);
            let obj = Self::turtle_term(&t.object);
            out.push_str(&format!("{} {} {} .\n", subj, pred, obj));
        }

        out
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Expand a local name to a base-IRI-qualified term.
    fn iri(&self, name: &str) -> String {
        // If already prefixed or absolute, leave as-is.
        if name.contains(':') || name.starts_with('<') {
            name.to_string()
        } else {
            format!(":{}", name)
        }
    }

    /// Format a term for Turtle output.
    fn turtle_term(term: &str) -> String {
        term.to_string()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mapper() -> VocabularyMapper {
        VocabularyMapper::new("https://example.org/ns#")
    }

    fn has_triple(triples: &[RdfTriple], s: &str, p: &str, o: &str) -> bool {
        triples
            .iter()
            .any(|t| t.subject == s && t.predicate == p && t.object == o)
    }

    fn contains_predicate(triples: &[RdfTriple], p: &str) -> bool {
        triples.iter().any(|t| t.predicate == p)
    }

    // ── Aspect ────────────────────────────────────────────────────────────

    #[test]
    fn test_aspect_type_triples() {
        let m = mapper();
        let el = SammElement::Aspect {
            name: "MyAspect".to_string(),
            properties: vec![],
            operations: vec![],
        };
        let ts = m.map_aspect(&el);
        assert!(has_triple(&ts, ":MyAspect", RDF_TYPE, SAMM_ASPECT));
        assert!(has_triple(&ts, ":MyAspect", RDF_TYPE, OWL_CLASS));
    }

    #[test]
    fn test_aspect_label() {
        let m = mapper();
        let el = SammElement::Aspect {
            name: "MyAspect".to_string(),
            properties: vec![],
            operations: vec![],
        };
        let ts = m.map_aspect(&el);
        assert!(contains_predicate(&ts, RDFS_LABEL));
    }

    #[test]
    fn test_aspect_with_properties() {
        let m = mapper();
        let el = SammElement::Aspect {
            name: "A".to_string(),
            properties: vec!["p1".to_string(), "p2".to_string()],
            operations: vec![],
        };
        let ts = m.map_aspect(&el);
        assert!(contains_predicate(&ts, SAMM_HAS_PROPERTY));
        let prop_count = ts
            .iter()
            .filter(|t| t.predicate == SAMM_HAS_PROPERTY)
            .count();
        assert_eq!(prop_count, 2);
    }

    #[test]
    fn test_aspect_with_operations() {
        let m = mapper();
        let el = SammElement::Aspect {
            name: "A".to_string(),
            properties: vec![],
            operations: vec!["op1".to_string()],
        };
        let ts = m.map_aspect(&el);
        assert!(contains_predicate(&ts, SAMM_OPERATION_PROP));
    }

    // ── Property ──────────────────────────────────────────────────────────

    #[test]
    fn test_property_type_triples() {
        let m = mapper();
        let el = SammElement::Property {
            name: "myProp".to_string(),
            characteristic: "StringChar".to_string(),
            optional: false,
        };
        let ts = m.map_property(&el);
        assert!(has_triple(&ts, ":myProp", RDF_TYPE, SAMM_PROPERTY));
        assert!(has_triple(&ts, ":myProp", RDF_TYPE, OWL_DATATYPE_PROPERTY));
    }

    #[test]
    fn test_property_characteristic_link() {
        let m = mapper();
        let el = SammElement::Property {
            name: "myProp".to_string(),
            characteristic: "StringChar".to_string(),
            optional: false,
        };
        let ts = m.map_property(&el);
        assert!(ts
            .iter()
            .any(|t| t.predicate == SAMM_BASE_CHARACTERISTIC && t.object == ":StringChar"));
    }

    #[test]
    fn test_property_optional_flag() {
        let m = mapper();
        let el = SammElement::Property {
            name: "optProp".to_string(),
            characteristic: "C".to_string(),
            optional: true,
        };
        let ts = m.map_property(&el);
        assert!(contains_predicate(&ts, SAMM_OPTIONAL));
    }

    // ── Characteristic ────────────────────────────────────────────────────

    #[test]
    fn test_characteristic_type() {
        let m = mapper();
        let el = SammElement::Characteristic {
            name: "MyChar".to_string(),
            base_type: Some("xsd:string".to_string()),
            values: None,
        };
        let ts = m.map_characteristic(&el);
        assert!(has_triple(&ts, ":MyChar", RDF_TYPE, SAMM_CHARACTERISTIC));
    }

    #[test]
    fn test_characteristic_base_type() {
        let m = mapper();
        let el = SammElement::Characteristic {
            name: "MyChar".to_string(),
            base_type: Some("xsd:integer".to_string()),
            values: None,
        };
        let ts = m.map_characteristic(&el);
        assert!(has_triple(&ts, ":MyChar", SAMM_DATA_TYPE, "xsd:integer"));
    }

    #[test]
    fn test_characteristic_no_base_type() {
        let m = mapper();
        let el = SammElement::Characteristic {
            name: "MyChar".to_string(),
            base_type: None,
            values: None,
        };
        let ts = m.map_characteristic(&el);
        assert!(!contains_predicate(&ts, SAMM_DATA_TYPE));
    }

    #[test]
    fn test_characteristic_with_values() {
        let m = mapper();
        let el = SammElement::Characteristic {
            name: "Status".to_string(),
            base_type: Some("xsd:string".to_string()),
            values: Some(vec!["ACTIVE".to_string(), "INACTIVE".to_string()]),
        };
        let ts = m.map_characteristic(&el);
        assert!(contains_predicate(&ts, SAMM_VALUES));
        assert!(contains_predicate(&ts, SAMM_VALUE));
    }

    // ── Operation ─────────────────────────────────────────────────────────

    #[test]
    fn test_operation_type() {
        let m = mapper();
        let el = SammElement::Operation {
            name: "toggle".to_string(),
            input: vec![],
            output: None,
        };
        let ts = m.map_element(&el);
        assert!(has_triple(&ts, ":toggle", RDF_TYPE, SAMM_OPERATION));
    }

    #[test]
    fn test_operation_with_input() {
        let m = mapper();
        let el = SammElement::Operation {
            name: "setSpeed".to_string(),
            input: vec!["speed".to_string()],
            output: None,
        };
        let ts = m.map_element(&el);
        assert!(contains_predicate(&ts, SAMM_INPUT));
    }

    #[test]
    fn test_operation_with_output() {
        let m = mapper();
        let el = SammElement::Operation {
            name: "getTemp".to_string(),
            input: vec![],
            output: Some("temperature".to_string()),
        };
        let ts = m.map_element(&el);
        assert!(contains_predicate(&ts, SAMM_OUTPUT));
    }

    // ── Entity ────────────────────────────────────────────────────────────

    #[test]
    fn test_entity_type() {
        let m = mapper();
        let el = SammElement::Entity {
            name: "Address".to_string(),
            properties: vec!["street".to_string()],
        };
        let ts = m.map_element(&el);
        assert!(has_triple(&ts, ":Address", RDF_TYPE, SAMM_ENTITY));
        assert!(has_triple(&ts, ":Address", RDF_TYPE, OWL_CLASS));
    }

    // ── Constraints ───────────────────────────────────────────────────────

    #[test]
    fn test_constraint_range() {
        let m = mapper();
        let el = SammElement::Constraint {
            name: "SpeedRange".to_string(),
            constraint_type: ConstraintType::Range {
                min: Some(0.0),
                max: Some(200.0),
            },
        };
        let ts = m.map_constraint(&el);
        assert!(has_triple(&ts, ":SpeedRange", RDF_TYPE, SAMM_CONSTRAINT));
        assert!(contains_predicate(&ts, SAMM_MIN_VALUE));
        assert!(contains_predicate(&ts, SAMM_MAX_VALUE));
    }

    #[test]
    fn test_constraint_range_no_min() {
        let m = mapper();
        let el = SammElement::Constraint {
            name: "C".to_string(),
            constraint_type: ConstraintType::Range {
                min: None,
                max: Some(100.0),
            },
        };
        let ts = m.map_constraint(&el);
        assert!(!contains_predicate(&ts, SAMM_MIN_VALUE));
        assert!(contains_predicate(&ts, SAMM_MAX_VALUE));
    }

    #[test]
    fn test_constraint_encoding() {
        let m = mapper();
        let el = SammElement::Constraint {
            name: "EncC".to_string(),
            constraint_type: ConstraintType::Encoding("UTF-8".to_string()),
        };
        let ts = m.map_constraint(&el);
        assert!(contains_predicate(&ts, SAMM_ENCODING));
    }

    #[test]
    fn test_constraint_language() {
        let m = mapper();
        let el = SammElement::Constraint {
            name: "LangC".to_string(),
            constraint_type: ConstraintType::Language("en".to_string()),
        };
        let ts = m.map_constraint(&el);
        assert!(contains_predicate(&ts, SAMM_LANGUAGE_CODE));
    }

    #[test]
    fn test_constraint_length() {
        let m = mapper();
        let el = SammElement::Constraint {
            name: "LenC".to_string(),
            constraint_type: ConstraintType::Length {
                min: Some(1),
                max: Some(255),
            },
        };
        let ts = m.map_constraint(&el);
        assert!(contains_predicate(&ts, SAMM_MIN_LENGTH));
        assert!(contains_predicate(&ts, SAMM_MAX_LENGTH));
    }

    #[test]
    fn test_constraint_pattern() {
        let m = mapper();
        let el = SammElement::Constraint {
            name: "PatC".to_string(),
            constraint_type: ConstraintType::Pattern("[A-Z]+".to_string()),
        };
        let ts = m.map_constraint(&el);
        assert!(contains_predicate(&ts, SAMM_PATTERN_VALUE));
    }

    // ── map_element dispatch ──────────────────────────────────────────────

    #[test]
    fn test_map_element_dispatches_correctly() {
        let m = mapper();
        let aspect = SammElement::Aspect {
            name: "A".to_string(),
            properties: vec![],
            operations: vec![],
        };
        let ts = m.map_element(&aspect);
        assert!(has_triple(&ts, ":A", RDF_TYPE, SAMM_ASPECT));
    }

    // ── render_turtle ─────────────────────────────────────────────────────

    #[test]
    fn test_render_turtle_has_prefixes() {
        let m = mapper();
        let ts = vec![RdfTriple::new(":Foo", RDF_TYPE, OWL_CLASS)];
        let turtle = m.render_turtle(&ts);
        assert!(turtle.contains("@prefix samm:"));
        assert!(turtle.contains("@prefix rdf:"));
        assert!(turtle.contains("@prefix owl:"));
        assert!(turtle.contains("@prefix rdfs:"));
        assert!(turtle.contains("@prefix xsd:"));
    }

    #[test]
    fn test_render_turtle_contains_triples() {
        let m = mapper();
        let el = SammElement::Aspect {
            name: "Foo".to_string(),
            properties: vec![],
            operations: vec![],
        };
        let ts = m.map_aspect(&el);
        let turtle = m.render_turtle(&ts);
        assert!(turtle.contains(":Foo"));
        assert!(turtle.contains(SAMM_ASPECT));
    }

    #[test]
    fn test_render_turtle_empty() {
        let m = mapper();
        let turtle = m.render_turtle(&[]);
        assert!(turtle.contains("@prefix"));
    }

    // ── Non-matching variants return empty ────────────────────────────────

    #[test]
    fn test_map_aspect_with_non_aspect_element() {
        let m = mapper();
        let el = SammElement::Property {
            name: "p".to_string(),
            characteristic: "c".to_string(),
            optional: false,
        };
        let ts = m.map_aspect(&el);
        assert!(ts.is_empty());
    }

    #[test]
    fn test_map_property_with_non_property_element() {
        let m = mapper();
        let el = SammElement::Entity {
            name: "e".to_string(),
            properties: vec![],
        };
        let ts = m.map_property(&el);
        assert!(ts.is_empty());
    }

    #[test]
    fn test_map_characteristic_with_non_characteristic() {
        let m = mapper();
        let el = SammElement::Aspect {
            name: "a".to_string(),
            properties: vec![],
            operations: vec![],
        };
        let ts = m.map_characteristic(&el);
        assert!(ts.is_empty());
    }

    #[test]
    fn test_map_constraint_with_non_constraint() {
        let m = mapper();
        let el = SammElement::Entity {
            name: "e".to_string(),
            properties: vec![],
        };
        let ts = m.map_constraint(&el);
        assert!(ts.is_empty());
    }
}
