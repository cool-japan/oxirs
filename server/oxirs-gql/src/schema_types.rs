//! RDF vocabulary and schema-generation configuration types.
//!
//! These value types describe an RDF ontology (classes, properties, namespaces)
//! and the configuration that drives GraphQL schema generation from it.

use std::collections::{HashMap, HashSet};

/// RDF vocabulary information extracted from an ontology
#[derive(Debug, Clone)]
pub struct RdfVocabulary {
    pub classes: HashMap<String, RdfClass>,
    pub properties: HashMap<String, RdfProperty>,
    pub namespaces: HashMap<String, String>,
}

/// RDF class information
#[derive(Debug, Clone)]
pub struct RdfClass {
    pub uri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub super_classes: Vec<String>,
    pub properties: Vec<String>,
}

/// RDF property information
#[derive(Debug, Clone)]
pub struct RdfProperty {
    pub uri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub domain: Vec<String>,
    pub range: Vec<String>,
    pub property_type: PropertyType,
    pub functional: bool,
    pub inverse_functional: bool,
}

/// Type of RDF property
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyType {
    DataProperty,
    ObjectProperty,
    AnnotationProperty,
}

/// Configuration for schema generation
#[derive(Debug, Clone)]
pub struct SchemaGenerationConfig {
    pub include_deprecated: bool,
    pub max_depth: usize,
    pub custom_scalars: HashMap<String, String>,
    pub type_mappings: HashMap<String, String>,
    pub exclude_classes: HashSet<String>,
    pub exclude_properties: HashSet<String>,
    pub enable_introspection: bool,
    pub enable_mutations: bool,
    pub enable_subscriptions: bool,
    /// Whether to expose the raw, unauthenticated SPARQL passthrough field
    /// (`sparql(query: String!): String`) on the generated `Query` type.
    /// Disabled by default: it bypasses all GraphQL-level depth/complexity
    /// limits and lets any client run arbitrary SPARQL against the store.
    /// See `GraphQLConfig::enable_sparql_field` in the crate root, which
    /// this should generally be kept in sync with.
    pub enable_sparql_field: bool,
}

impl Default for SchemaGenerationConfig {
    fn default() -> Self {
        Self {
            include_deprecated: false,
            max_depth: 10,
            custom_scalars: HashMap::new(),
            type_mappings: Self::default_type_mappings(),
            exclude_classes: HashSet::new(),
            exclude_properties: HashSet::new(),
            enable_introspection: true,
            enable_mutations: false,
            enable_subscriptions: false,
            enable_sparql_field: false,
        }
    }
}

impl SchemaGenerationConfig {
    fn default_type_mappings() -> HashMap<String, String> {
        let mut mappings = HashMap::new();

        // XSD mappings
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#string".to_string(),
            "String".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#int".to_string(),
            "Int".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#integer".to_string(),
            "Int".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#long".to_string(),
            "Int".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#float".to_string(),
            "Float".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#double".to_string(),
            "Float".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
            "Float".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
            "Boolean".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#dateTime".to_string(),
            "DateTime".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#date".to_string(),
            "DateTime".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#time".to_string(),
            "DateTime".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#duration".to_string(),
            "Duration".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2001/XMLSchema#anyURI".to_string(),
            "IRI".to_string(),
        );

        // RDF mappings
        mappings.insert(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string(),
            "LangString".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/2000/01/rdf-schema#Literal".to_string(),
            "Literal".to_string(),
        );
        mappings.insert(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral".to_string(),
            "String".to_string(),
        );

        // RDFS mappings
        mappings.insert(
            "http://www.w3.org/2000/01/rdf-schema#Resource".to_string(),
            "IRI".to_string(),
        );

        mappings
    }
}
