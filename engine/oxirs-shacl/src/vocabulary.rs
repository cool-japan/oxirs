//! SHACL vocabulary terms and constants
//!
//! This module defines all SHACL vocabulary terms according to the W3C SHACL specification.

use crate::{Result, ShaclError, SHACL_NS};
use oxirs_core::model::NamedNode;

/// SHACL vocabulary containing all standard SHACL terms
#[derive(Debug)]
pub struct ShaclVocabulary {
    // Core Classes
    pub node_shape: NamedNode,
    pub property_shape: NamedNode,
    pub constraint_component: NamedNode,
    pub validation_report: NamedNode,
    pub validation_result: NamedNode,

    // Target Properties
    pub target_class: NamedNode,
    pub target_node: NamedNode,
    pub target_objects_of: NamedNode,
    pub target_subjects_of: NamedNode,
    pub target: NamedNode,

    // Shape Properties
    pub path: NamedNode,
    pub deactivated: NamedNode,
    pub message: NamedNode,
    pub severity: NamedNode,
    pub label: NamedNode,
    pub description: NamedNode,
    pub group: NamedNode,
    pub order: NamedNode,

    // Core Constraints
    pub class: NamedNode,
    pub datatype: NamedNode,
    pub node_kind: NamedNode,
    pub min_count: NamedNode,
    pub max_count: NamedNode,
    pub min_exclusive: NamedNode,
    pub max_exclusive: NamedNode,
    pub min_inclusive: NamedNode,
    pub max_inclusive: NamedNode,
    pub min_length: NamedNode,
    pub max_length: NamedNode,
    pub pattern: NamedNode,
    pub flags: NamedNode,
    pub language_in: NamedNode,
    pub unique_lang: NamedNode,
    pub equals: NamedNode,
    pub disjoint: NamedNode,
    pub less_than: NamedNode,
    pub less_than_or_equals: NamedNode,
    pub in_list: NamedNode,
    pub has_value: NamedNode,

    // Logical Constraints
    pub not: NamedNode,
    pub and: NamedNode,
    pub or: NamedNode,
    pub xone: NamedNode,

    // Shape-based Constraints
    pub node: NamedNode,
    pub qualified_value_shape: NamedNode,
    pub qualified_min_count: NamedNode,
    pub qualified_max_count: NamedNode,
    pub qualified_value_shapes_disjoint: NamedNode,

    // Closed Shapes
    pub closed: NamedNode,
    pub ignored_properties: NamedNode,

    // Property Paths
    pub alternative_path: NamedNode,
    pub inverse_path: NamedNode,
    pub zero_or_more_path: NamedNode,
    pub one_or_more_path: NamedNode,
    pub zero_or_one_path: NamedNode,

    // Node Kinds
    pub iri: NamedNode,
    pub blank_node: NamedNode,
    pub literal: NamedNode,
    pub blank_node_or_iri: NamedNode,
    pub blank_node_or_literal: NamedNode,
    pub iri_or_literal: NamedNode,

    // Severity Levels
    pub violation: NamedNode,
    pub warning: NamedNode,
    pub info: NamedNode,

    // Validation Result Properties
    pub conforms: NamedNode,
    pub result: NamedNode,
    pub focus_node: NamedNode,
    pub result_path: NamedNode,
    pub value: NamedNode,
    pub source_constraint_component: NamedNode,
    pub source_shape: NamedNode,
    pub result_severity: NamedNode,
    pub result_message: NamedNode,
    pub detail: NamedNode,

    // SHACL-SPARQL
    pub sparql: NamedNode,
    pub select: NamedNode,
    pub ask: NamedNode,
    pub construct: NamedNode,
    pub update: NamedNode,
    pub prefixes: NamedNode,

    // Built-in Functions
    pub this: NamedNode,
    pub current_shape: NamedNode,
    pub shapes_graph: NamedNode,
    pub value_type: NamedNode,

    // Constraint Components
    pub class_constraint_component: NamedNode,
    pub datatype_constraint_component: NamedNode,
    pub node_kind_constraint_component: NamedNode,
    pub min_count_constraint_component: NamedNode,
    pub max_count_constraint_component: NamedNode,
    pub min_exclusive_constraint_component: NamedNode,
    pub max_exclusive_constraint_component: NamedNode,
    pub min_inclusive_constraint_component: NamedNode,
    pub max_inclusive_constraint_component: NamedNode,
    pub min_length_constraint_component: NamedNode,
    pub max_length_constraint_component: NamedNode,
    pub pattern_constraint_component: NamedNode,
    pub language_in_constraint_component: NamedNode,
    pub unique_lang_constraint_component: NamedNode,
    pub equals_constraint_component: NamedNode,
    pub disjoint_constraint_component: NamedNode,
    pub less_than_constraint_component: NamedNode,
    pub less_than_or_equals_constraint_component: NamedNode,
    pub in_constraint_component: NamedNode,
    pub has_value_constraint_component: NamedNode,
    pub not_constraint_component: NamedNode,
    pub and_constraint_component: NamedNode,
    pub or_constraint_component: NamedNode,
    pub xone_constraint_component: NamedNode,
    pub node_constraint_component: NamedNode,
    pub qualified_value_shape_constraint_component: NamedNode,
    pub closed_constraint_component: NamedNode,
    pub sparql_constraint_component: NamedNode,
}

impl Default for ShaclVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaclVocabulary {
    pub fn new() -> Self {
        Self {
            // Core Classes
            node_shape: named_node("NodeShape"),
            property_shape: named_node("PropertyShape"),
            constraint_component: named_node("ConstraintComponent"),
            validation_report: named_node("ValidationReport"),
            validation_result: named_node("ValidationResult"),

            // Target Properties
            target_class: named_node("targetClass"),
            target_node: named_node("targetNode"),
            target_objects_of: named_node("targetObjectsOf"),
            target_subjects_of: named_node("targetSubjectsOf"),
            target: named_node("target"),

            // Shape Properties
            path: named_node("path"),
            deactivated: named_node("deactivated"),
            message: named_node("message"),
            severity: named_node("severity"),
            label: named_node("label"),
            description: named_node("description"),
            group: named_node("group"),
            order: named_node("order"),

            // Core Constraints
            class: named_node("class"),
            datatype: named_node("datatype"),
            node_kind: named_node("nodeKind"),
            min_count: named_node("minCount"),
            max_count: named_node("maxCount"),
            min_exclusive: named_node("minExclusive"),
            max_exclusive: named_node("maxExclusive"),
            min_inclusive: named_node("minInclusive"),
            max_inclusive: named_node("maxInclusive"),
            min_length: named_node("minLength"),
            max_length: named_node("maxLength"),
            pattern: named_node("pattern"),
            flags: named_node("flags"),
            language_in: named_node("languageIn"),
            unique_lang: named_node("uniqueLang"),
            equals: named_node("equals"),
            disjoint: named_node("disjoint"),
            less_than: named_node("lessThan"),
            less_than_or_equals: named_node("lessThanOrEquals"),
            in_list: named_node("in"),
            has_value: named_node("hasValue"),

            // Logical Constraints
            not: named_node("not"),
            and: named_node("and"),
            or: named_node("or"),
            xone: named_node("xone"),

            // Shape-based Constraints
            node: named_node("node"),
            qualified_value_shape: named_node("qualifiedValueShape"),
            qualified_min_count: named_node("qualifiedMinCount"),
            qualified_max_count: named_node("qualifiedMaxCount"),
            qualified_value_shapes_disjoint: named_node("qualifiedValueShapesDisjoint"),

            // Closed Shapes
            closed: named_node("closed"),
            ignored_properties: named_node("ignoredProperties"),

            // Property Paths
            alternative_path: named_node("alternativePath"),
            inverse_path: named_node("inversePath"),
            zero_or_more_path: named_node("zeroOrMorePath"),
            one_or_more_path: named_node("oneOrMorePath"),
            zero_or_one_path: named_node("zeroOrOnePath"),

            // Node Kinds
            iri: named_node("IRI"),
            blank_node: named_node("BlankNode"),
            literal: named_node("Literal"),
            blank_node_or_iri: named_node("BlankNodeOrIRI"),
            blank_node_or_literal: named_node("BlankNodeOrLiteral"),
            iri_or_literal: named_node("IRIOrLiteral"),

            // Severity Levels
            violation: named_node("Violation"),
            warning: named_node("Warning"),
            info: named_node("Info"),

            // Validation Result Properties
            conforms: named_node("conforms"),
            result: named_node("result"),
            focus_node: named_node("focusNode"),
            result_path: named_node("resultPath"),
            value: named_node("value"),
            source_constraint_component: named_node("sourceConstraintComponent"),
            source_shape: named_node("sourceShape"),
            result_severity: named_node("resultSeverity"),
            result_message: named_node("resultMessage"),
            detail: named_node("detail"),

            // SHACL-SPARQL
            sparql: named_node("sparql"),
            select: named_node("select"),
            ask: named_node("ask"),
            construct: named_node("construct"),
            update: named_node("update"),
            prefixes: named_node("prefixes"),

            // Built-in Functions
            this: named_node("this"),
            current_shape: named_node("currentShape"),
            shapes_graph: named_node("shapesGraph"),
            value_type: named_node("valueType"),

            // Constraint Components
            class_constraint_component: named_node("ClassConstraintComponent"),
            datatype_constraint_component: named_node("DatatypeConstraintComponent"),
            node_kind_constraint_component: named_node("NodeKindConstraintComponent"),
            min_count_constraint_component: named_node("MinCountConstraintComponent"),
            max_count_constraint_component: named_node("MaxCountConstraintComponent"),
            min_exclusive_constraint_component: named_node("MinExclusiveConstraintComponent"),
            max_exclusive_constraint_component: named_node("MaxExclusiveConstraintComponent"),
            min_inclusive_constraint_component: named_node("MinInclusiveConstraintComponent"),
            max_inclusive_constraint_component: named_node("MaxInclusiveConstraintComponent"),
            min_length_constraint_component: named_node("MinLengthConstraintComponent"),
            max_length_constraint_component: named_node("MaxLengthConstraintComponent"),
            pattern_constraint_component: named_node("PatternConstraintComponent"),
            language_in_constraint_component: named_node("LanguageInConstraintComponent"),
            unique_lang_constraint_component: named_node("UniqueLangConstraintComponent"),
            equals_constraint_component: named_node("EqualsConstraintComponent"),
            disjoint_constraint_component: named_node("DisjointConstraintComponent"),
            less_than_constraint_component: named_node("LessThanConstraintComponent"),
            less_than_or_equals_constraint_component: named_node(
                "LessThanOrEqualsConstraintComponent",
            ),
            in_constraint_component: named_node("InConstraintComponent"),
            has_value_constraint_component: named_node("HasValueConstraintComponent"),
            not_constraint_component: named_node("NotConstraintComponent"),
            and_constraint_component: named_node("AndConstraintComponent"),
            or_constraint_component: named_node("OrConstraintComponent"),
            xone_constraint_component: named_node("XoneConstraintComponent"),
            node_constraint_component: named_node("NodeConstraintComponent"),
            qualified_value_shape_constraint_component: named_node(
                "QualifiedValueShapeConstraintComponent",
            ),
            closed_constraint_component: named_node("ClosedConstraintComponent"),
            sparql_constraint_component: named_node("SPARQLConstraintComponent"),
        }
    }

    /// Check if a named node is a SHACL term
    pub fn is_shacl_term(&self, node: &NamedNode) -> bool {
        node.as_str().starts_with(SHACL_NS)
    }

    /// Get the local name of a SHACL term
    pub fn get_local_name<'a>(&self, node: &'a NamedNode) -> Option<&'a str> {
        if self.is_shacl_term(node) {
            Some(&node.as_str()[SHACL_NS.len()..])
        } else {
            None
        }
    }

    /// Create a SHACL named node from a local name
    pub fn create_term(&self, local_name: &str) -> NamedNode {
        named_node(local_name)
    }
}

/// Helper function to create a SHACL named node
fn named_node(local_name: &str) -> NamedNode {
    NamedNode::new(format!("{SHACL_NS}{local_name}")).expect("Invalid SHACL IRI")
}

/// Standard SHACL prefixes for use in SPARQL queries
pub const SHACL_PREFIXES: &str = r#"
PREFIX sh: <http://www.w3.org/ns/shacl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"#;

/// IRI validation and prefix handling utilities
pub struct IriResolver {
    /// Base IRI for resolving relative IRIs
    base_iri: Option<String>,

    /// Namespace prefix mappings
    prefixes: std::collections::HashMap<String, String>,
}

impl IriResolver {
    /// Create a new IRI resolver
    pub fn new() -> Self {
        let mut resolver = Self {
            base_iri: None,
            prefixes: std::collections::HashMap::new(),
        };

        // Add standard prefixes
        resolver.add_standard_prefixes();
        resolver
    }

    /// Create a new IRI resolver with a base IRI
    pub fn with_base_iri(base_iri: String) -> Self {
        let mut resolver = Self::new();
        resolver.base_iri = Some(base_iri);
        resolver
    }

    /// Add standard prefixes used in SHACL
    fn add_standard_prefixes(&mut self) {
        self.prefixes
            .insert("sh".to_string(), "http://www.w3.org/ns/shacl#".to_string());
        self.prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        self.prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        self.prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        self.prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
    }

    /// Add a prefix mapping
    pub fn add_prefix(&mut self, prefix: String, namespace: String) {
        self.prefixes.insert(prefix, namespace);
    }

    /// Set the base IRI for resolving relative IRIs
    pub fn set_base_iri(&mut self, base_iri: String) {
        self.base_iri = Some(base_iri);
    }

    /// Validate that an IRI is well-formed
    pub fn validate_iri(&self, iri: &str) -> Result<()> {
        // Check if it's a valid absolute IRI
        if self.is_absolute_iri(iri) {
            return self.validate_absolute_iri(iri);
        }

        // Check if it's a prefixed name (CURIE)
        if iri.contains(':') && !iri.starts_with("http://") && !iri.starts_with("https://") {
            return self.validate_prefixed_name(iri);
        }

        // Check if it's a relative IRI
        if self.base_iri.is_some() {
            return self.validate_relative_iri(iri);
        }

        Err(ShaclError::ShapeParsing(format!(
            "Invalid IRI: '{iri}' - not absolute, not a valid prefixed name, and no base IRI set"
        )))
    }

    /// Check if an IRI is absolute
    fn is_absolute_iri(&self, iri: &str) -> bool {
        iri.starts_with("http://") || iri.starts_with("https://") || iri.starts_with("urn:")
    }

    /// Validate an absolute IRI
    fn validate_absolute_iri(&self, iri: &str) -> Result<()> {
        // Basic IRI validation - check for invalid characters
        if iri.contains(' ') || iri.contains('\t') || iri.contains('\n') || iri.contains('\r') {
            return Err(ShaclError::ShapeParsing(format!(
                "Invalid IRI '{iri}': contains whitespace characters"
            )));
        }

        // Check for other invalid characters according to RFC 3987
        let invalid_chars = ['<', '>', '"', '{', '}', '|', '^', '`', '\\'];
        for invalid_char in &invalid_chars {
            if iri.contains(*invalid_char) {
                return Err(ShaclError::ShapeParsing(format!(
                    "Invalid IRI '{iri}': contains invalid character '{invalid_char}'"
                )));
            }
        }

        Ok(())
    }

    /// Validate a prefixed name (CURIE)
    fn validate_prefixed_name(&self, curie: &str) -> Result<()> {
        if let Some(colon_pos) = curie.find(':') {
            let prefix = &curie[..colon_pos];
            let local_part = &curie[colon_pos + 1..];

            // Check if prefix is known
            if !self.prefixes.contains_key(prefix) {
                return Err(ShaclError::ShapeParsing(format!(
                    "Unknown prefix '{prefix}' in CURIE '{curie}'"
                )));
            }

            // Validate local part doesn't contain invalid characters
            if local_part.contains(' ') || local_part.contains('\t') || local_part.contains('\n') {
                return Err(ShaclError::ShapeParsing(format!(
                    "Invalid local part in CURIE '{curie}': contains whitespace"
                )));
            }

            Ok(())
        } else {
            Err(ShaclError::ShapeParsing(format!(
                "Invalid CURIE format: '{curie}'"
            )))
        }
    }

    /// Validate a relative IRI
    fn validate_relative_iri(&self, iri: &str) -> Result<()> {
        // Basic validation for relative IRIs
        if iri.contains(' ') || iri.contains('\t') || iri.contains('\n') || iri.contains('\r') {
            return Err(ShaclError::ShapeParsing(format!(
                "Invalid relative IRI '{iri}': contains whitespace characters"
            )));
        }
        Ok(())
    }

    /// Expand a prefixed IRI or validate and return absolute IRI
    pub fn expand_iri(&self, iri: &str) -> Result<String> {
        // If it's already absolute, validate and return
        if self.is_absolute_iri(iri) {
            self.validate_absolute_iri(iri)?;
            return Ok(iri.to_string());
        }

        // If it's a prefixed name (CURIE), expand it
        if iri.contains(':') && !iri.starts_with("http://") && !iri.starts_with("https://") {
            return self.expand_prefixed_name(iri);
        }

        // If it's a relative IRI, resolve against base IRI
        if let Some(base) = &self.base_iri {
            return self.resolve_relative_iri(iri, base);
        }

        Err(ShaclError::ShapeParsing(format!(
            "Cannot expand IRI '{iri}': not absolute, not a valid prefixed name, and no base IRI set"
        )))
    }

    /// Expand a prefixed name (CURIE) to full IRI
    fn expand_prefixed_name(&self, curie: &str) -> Result<String> {
        self.validate_prefixed_name(curie)?;

        if let Some(colon_pos) = curie.find(':') {
            let prefix = &curie[..colon_pos];
            let local_part = &curie[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                let expanded = format!("{namespace}{local_part}");
                return Ok(expanded);
            }
        }

        Err(ShaclError::ShapeParsing(format!(
            "Cannot expand prefixed name '{curie}'"
        )))
    }

    /// Resolve a relative IRI against a base IRI
    fn resolve_relative_iri(&self, relative_iri: &str, base_iri: &str) -> Result<String> {
        self.validate_relative_iri(relative_iri)?;
        self.validate_absolute_iri(base_iri)?;

        // Simple resolution - just concatenate for now
        // A full implementation would handle RFC 3986 resolution
        let resolved = if base_iri.ends_with('/') || base_iri.ends_with('#') {
            format!("{base_iri}{relative_iri}")
        } else {
            format!("{base_iri}/{relative_iri}")
        };

        Ok(resolved)
    }

    /// Get all known prefixes
    pub fn get_prefixes(&self) -> &std::collections::HashMap<String, String> {
        &self.prefixes
    }

    /// Get the namespace for a prefix
    pub fn get_namespace(&self, prefix: &str) -> Option<&String> {
        self.prefixes.get(prefix)
    }

    /// Create a NamedNode with IRI validation and expansion
    pub fn create_named_node(&self, iri: &str) -> Result<NamedNode> {
        let expanded_iri = self.expand_iri(iri)?;
        NamedNode::new(expanded_iri).map_err(|e| {
            ShaclError::ShapeParsing(format!("Failed to create NamedNode from IRI '{iri}': {e}"))
        })
    }
}

impl Default for IriResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shacl_vocabulary() {
        let vocab = ShaclVocabulary::new();

        // Test core classes
        assert_eq!(
            vocab.node_shape.as_str(),
            "http://www.w3.org/ns/shacl#NodeShape"
        );
        assert_eq!(
            vocab.property_shape.as_str(),
            "http://www.w3.org/ns/shacl#PropertyShape"
        );

        // Test constraint properties
        assert_eq!(vocab.class.as_str(), "http://www.w3.org/ns/shacl#class");
        assert_eq!(
            vocab.datatype.as_str(),
            "http://www.w3.org/ns/shacl#datatype"
        );

        // Test validation result properties
        assert_eq!(
            vocab.conforms.as_str(),
            "http://www.w3.org/ns/shacl#conforms"
        );
        assert_eq!(
            vocab.focus_node.as_str(),
            "http://www.w3.org/ns/shacl#focusNode"
        );
    }

    #[test]
    fn test_is_shacl_term() {
        let vocab = ShaclVocabulary::new();

        assert!(vocab.is_shacl_term(&vocab.node_shape));
        assert!(vocab.is_shacl_term(&vocab.class));

        let non_shacl = NamedNode::new("http://example.org/test").unwrap();
        assert!(!vocab.is_shacl_term(&non_shacl));
    }

    #[test]
    fn test_get_local_name() {
        let vocab = ShaclVocabulary::new();

        assert_eq!(vocab.get_local_name(&vocab.node_shape), Some("NodeShape"));
        assert_eq!(vocab.get_local_name(&vocab.class), Some("class"));

        let non_shacl = NamedNode::new("http://example.org/test").unwrap();
        assert_eq!(vocab.get_local_name(&non_shacl), None);
    }

    #[test]
    fn test_create_term() {
        let vocab = ShaclVocabulary::new();
        let custom_term = vocab.create_term("customProperty");

        assert_eq!(
            custom_term.as_str(),
            "http://www.w3.org/ns/shacl#customProperty"
        );
        assert!(vocab.is_shacl_term(&custom_term));
        assert_eq!(vocab.get_local_name(&custom_term), Some("customProperty"));
    }

    #[test]
    fn test_iri_resolver_basic() {
        let resolver = IriResolver::new();

        // Test absolute IRI validation
        assert!(resolver.validate_iri("http://example.org/test").is_ok());
        assert!(resolver.validate_iri("https://example.org/test").is_ok());
        assert!(resolver.validate_iri("urn:example:test").is_ok());

        // Test invalid IRIs
        assert!(resolver
            .validate_iri("http://example.org/test with spaces")
            .is_err());
        assert!(resolver
            .validate_iri("http://example.org/test<invalid>")
            .is_err());
    }

    #[test]
    fn test_iri_resolver_prefixes() {
        let resolver = IriResolver::new();

        // Test CURIE expansion
        assert_eq!(
            resolver.expand_iri("sh:NodeShape").unwrap(),
            "http://www.w3.org/ns/shacl#NodeShape"
        );
        assert_eq!(
            resolver.expand_iri("rdf:type").unwrap(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );

        // Test unknown prefix
        assert!(resolver.expand_iri("unknown:test").is_err());
    }

    #[test]
    fn test_iri_resolver_base_iri() {
        let resolver = IriResolver::with_base_iri("http://example.org/base/".to_string());

        // Test relative IRI resolution
        assert_eq!(
            resolver.expand_iri("test").unwrap(),
            "http://example.org/base/test"
        );
        assert_eq!(
            resolver.expand_iri("subdir/test").unwrap(),
            "http://example.org/base/subdir/test"
        );
    }

    #[test]
    fn test_iri_resolver_custom_prefix() {
        let mut resolver = IriResolver::new();
        resolver.add_prefix("ex".to_string(), "http://example.org/vocab#".to_string());

        assert_eq!(
            resolver.expand_iri("ex:test").unwrap(),
            "http://example.org/vocab#test"
        );
    }

    #[test]
    fn test_iri_resolver_named_node_creation() {
        let resolver = IriResolver::new();

        // Test creating NamedNode from CURIE
        let node = resolver.create_named_node("sh:NodeShape").unwrap();
        assert_eq!(node.as_str(), "http://www.w3.org/ns/shacl#NodeShape");

        // Test creating NamedNode from absolute IRI
        let node = resolver
            .create_named_node("http://example.org/test")
            .unwrap();
        assert_eq!(node.as_str(), "http://example.org/test");

        // Test invalid IRI
        assert!(resolver.create_named_node("invalid iri").is_err());
    }
}
