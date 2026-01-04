//! SPARQL 1.1 Service Description
//!
//! This module implements the SPARQL 1.1 Service Description vocabulary,
//! allowing SPARQL endpoints to advertise their capabilities, supported features,
//! available datasets, and extension functions.
//!
//! Based on W3C SPARQL 1.1 Service Description specification and Apache Jena's
//! service description implementation.

use crate::algebra::Variable;
use crate::results::{QueryResult, ResultFormat};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// SPARQL 1.1 Service Description vocabulary namespace
pub const SD_NS: &str = "http://www.w3.org/ns/sparql-service-description#";

/// OxiRS extension vocabulary namespace
pub const OXIRS_NS: &str = "http://oxirs.io/ns/service-description#";

/// Service description for a SPARQL endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDescription {
    /// Endpoint URL
    pub endpoint: String,
    /// Default dataset description
    pub default_dataset: Option<DatasetDescription>,
    /// Named datasets
    pub named_datasets: Vec<DatasetDescription>,
    /// Supported features
    pub features: HashSet<Feature>,
    /// Supported SPARQL language features
    pub language_extensions: HashSet<LanguageExtension>,
    /// Supported result formats
    pub result_formats: Vec<ResultFormat>,
    /// Available extension functions
    pub extension_functions: Vec<ExtensionFunction>,
    /// Available stored procedures
    pub procedures: Vec<ProcedureInfo>,
    /// Available property functions
    pub property_functions: Vec<PropertyFunctionInfo>,
    /// Available custom aggregates
    pub aggregates: Vec<AggregateInfo>,
    /// Endpoint limitations
    pub limitations: ServiceLimitations,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ServiceDescription {
    /// Create new service description
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            default_dataset: None,
            named_datasets: Vec::new(),
            features: HashSet::new(),
            language_extensions: HashSet::new(),
            result_formats: vec![
                ResultFormat::Json,
                ResultFormat::Xml,
                ResultFormat::Csv,
                ResultFormat::Tsv,
            ],
            extension_functions: Vec::new(),
            procedures: Vec::new(),
            property_functions: Vec::new(),
            aggregates: Vec::new(),
            limitations: ServiceLimitations::default(),
            metadata: HashMap::new(),
        }
    }

    /// Add a feature
    pub fn add_feature(&mut self, feature: Feature) {
        self.features.insert(feature);
    }

    /// Add a language extension
    pub fn add_language_extension(&mut self, extension: LanguageExtension) {
        self.language_extensions.insert(extension);
    }

    /// Add an extension function
    pub fn add_extension_function(&mut self, function: ExtensionFunction) {
        self.extension_functions.push(function);
    }

    /// Add a stored procedure
    pub fn add_procedure(&mut self, procedure: ProcedureInfo) {
        self.procedures.push(procedure);
    }

    /// Add a property function
    pub fn add_property_function(&mut self, prop_func: PropertyFunctionInfo) {
        self.property_functions.push(prop_func);
    }

    /// Add a custom aggregate
    pub fn add_aggregate(&mut self, aggregate: AggregateInfo) {
        self.aggregates.push(aggregate);
    }

    /// Generate RDF representation (Turtle format)
    pub fn to_turtle(&self) -> Result<String> {
        let mut turtle = String::new();

        // Prefixes
        turtle.push_str("@prefix sd: <http://www.w3.org/ns/sparql-service-description#> .\n");
        turtle.push_str("@prefix oxirs: <http://oxirs.io/ns/service-description#> .\n");
        turtle.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        turtle.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

        // Service description
        turtle.push_str(&format!("<{}> a sd:Service ;\n", self.endpoint));

        // Default dataset
        if let Some(ref dataset) = self.default_dataset {
            turtle.push_str("  sd:defaultDataset [\n");
            turtle.push_str("    a sd:Dataset ;\n");
            if let Some(ref name) = dataset.name {
                turtle.push_str(&format!("    rdfs:label \"{}\" ;\n", name));
            }
            turtle.push_str(&format!("    sd:tripleCount {} ;\n", dataset.triple_count));
            turtle.push_str("  ] ;\n");
        }

        // Features
        for feature in &self.features {
            turtle.push_str(&format!("  sd:feature sd:{} ;\n", feature.as_iri()));
        }

        // Language extensions
        for ext in &self.language_extensions {
            turtle.push_str(&format!("  sd:languageExtension sd:{} ;\n", ext.as_iri()));
        }

        // Result formats
        for format in &self.result_formats {
            turtle.push_str(&format!("  sd:resultFormat <{}> ;\n", format.mime_type()));
        }

        // Extension functions
        if !self.extension_functions.is_empty() {
            turtle.push_str("  oxirs:extensionFunction\n");
            for (i, func) in self.extension_functions.iter().enumerate() {
                let comma = if i < self.extension_functions.len() - 1 {
                    ","
                } else {
                    ""
                };
                turtle.push_str(&format!("    <{}>{}\n", func.uri, comma));
            }
            turtle.push_str("  ;\n");
        }

        turtle.push_str("  .\n");

        Ok(turtle)
    }

    /// Generate JSON representation
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow!("JSON serialization failed: {}", e))
    }

    /// Generate SPARQL query result format
    pub fn to_query_result(&self) -> QueryResult {
        let variables = vec![
            Variable::new("feature").expect("hardcoded variable name should be valid"),
            Variable::new("value").expect("hardcoded variable name should be valid"),
        ];

        let mut solutions = Vec::new();

        // Add features as bindings
        for feature in &self.features {
            let mut binding = HashMap::new();
            binding.insert(
                Variable::new("feature").expect("hardcoded variable name should be valid"),
                crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: "feature".to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            binding.insert(
                Variable::new("value").expect("hardcoded variable name should be valid"),
                crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: feature.as_iri().to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            solutions.push(binding);
        }

        QueryResult::Bindings {
            variables,
            solutions,
        }
    }
}

/// Dataset description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDescription {
    /// Dataset name
    pub name: Option<String>,
    /// Default graph URIs
    pub default_graphs: Vec<String>,
    /// Named graphs
    pub named_graphs: Vec<NamedGraphDescription>,
    /// Approximate triple count
    pub triple_count: u64,
    /// Dataset metadata
    pub metadata: HashMap<String, String>,
}

impl DatasetDescription {
    /// Create new dataset description
    pub fn new() -> Self {
        Self {
            name: None,
            default_graphs: Vec::new(),
            named_graphs: Vec::new(),
            triple_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set dataset name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Set triple count
    pub fn with_triple_count(mut self, count: u64) -> Self {
        self.triple_count = count;
        self
    }

    /// Add default graph
    pub fn add_default_graph(&mut self, graph_uri: String) {
        self.default_graphs.push(graph_uri);
    }

    /// Add named graph
    pub fn add_named_graph(&mut self, graph: NamedGraphDescription) {
        self.named_graphs.push(graph);
    }
}

impl Default for DatasetDescription {
    fn default() -> Self {
        Self::new()
    }
}

/// Named graph description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedGraphDescription {
    /// Graph URI
    pub name: String,
    /// Approximate triple count
    pub triple_count: u64,
    /// Graph metadata
    pub metadata: HashMap<String, String>,
}

/// SPARQL 1.1 features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Feature {
    /// Basic federated query support
    BasicFederatedQuery,
    /// SPARQL 1.1 Query
    SPARQL11Query,
    /// SPARQL 1.1 Update
    SPARQL11Update,
    /// Aggregates
    Aggregates,
    /// Property paths
    PropertyPaths,
    /// Subqueries
    Subqueries,
    /// BIND
    Bind,
    /// VALUES
    Values,
    /// Negation (MINUS, NOT EXISTS)
    Negation,
    /// Service (federated queries)
    Service,
}

impl Feature {
    /// Convert to IRI local name
    pub fn as_iri(&self) -> &'static str {
        match self {
            Feature::BasicFederatedQuery => "BasicFederatedQuery",
            Feature::SPARQL11Query => "SPARQL11Query",
            Feature::SPARQL11Update => "SPARQL11Update",
            Feature::Aggregates => "Aggregates",
            Feature::PropertyPaths => "PropertyPaths",
            Feature::Subqueries => "Subqueries",
            Feature::Bind => "Bind",
            Feature::Values => "Values",
            Feature::Negation => "Negation",
            Feature::Service => "Service",
        }
    }
}

/// Language extensions beyond SPARQL 1.1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LanguageExtension {
    /// RDF-star (RDF 1.2) support
    RDFStar,
    /// Property functions (Jena ARQ style)
    PropertyFunctions,
    /// Stored procedures
    StoredProcedures,
    /// Custom aggregates
    CustomAggregates,
    /// Full-text search
    FullTextSearch,
    /// Geospatial queries
    GeospatialQueries,
    /// Temporal queries
    TemporalQueries,
    /// Vector similarity search
    VectorSearch,
}

impl LanguageExtension {
    /// Convert to IRI local name
    pub fn as_iri(&self) -> &'static str {
        match self {
            LanguageExtension::RDFStar => "RDFStar",
            LanguageExtension::PropertyFunctions => "PropertyFunctions",
            LanguageExtension::StoredProcedures => "StoredProcedures",
            LanguageExtension::CustomAggregates => "CustomAggregates",
            LanguageExtension::FullTextSearch => "FullTextSearch",
            LanguageExtension::GeospatialQueries => "GeospatialQueries",
            LanguageExtension::TemporalQueries => "TemporalQueries",
            LanguageExtension::VectorSearch => "VectorSearch",
        }
    }
}

/// Extension function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionFunction {
    /// Function URI
    pub uri: String,
    /// Function name (short form)
    pub name: String,
    /// Function documentation
    pub documentation: String,
    /// Parameter types
    pub parameters: Vec<ParameterInfo>,
    /// Return type
    pub return_type: String,
    /// Whether function is deterministic
    pub deterministic: bool,
}

/// Stored procedure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureInfo {
    /// Procedure URI
    pub uri: String,
    /// Procedure name
    pub name: String,
    /// Documentation
    pub documentation: String,
    /// Parameters
    pub parameters: Vec<ParameterInfo>,
    /// Whether procedure has side effects
    pub has_side_effects: bool,
    /// Whether procedure is deterministic
    pub deterministic: bool,
}

/// Property function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyFunctionInfo {
    /// Property function URI
    pub uri: String,
    /// Function name
    pub name: String,
    /// Documentation
    pub documentation: String,
    /// Subject argument type
    pub subject_type: String,
    /// Object argument type
    pub object_type: String,
}

/// Aggregate function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateInfo {
    /// Aggregate URI
    pub uri: String,
    /// Aggregate name
    pub name: String,
    /// Documentation
    pub documentation: String,
    /// Whether DISTINCT is supported
    pub supports_distinct: bool,
}

/// Parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Whether parameter is optional
    pub optional: bool,
    /// Default value
    pub default_value: Option<String>,
}

/// Service limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLimitations {
    /// Maximum query execution time (seconds)
    pub max_execution_time: Option<u64>,
    /// Maximum result size
    pub max_result_size: Option<u64>,
    /// Maximum offset
    pub max_offset: Option<u64>,
    /// Default limit if none specified
    pub default_limit: Option<u64>,
    /// Maximum limit
    pub max_limit: Option<u64>,
    /// Timeout for external SERVICE calls
    pub service_call_timeout: Option<u64>,
}

impl Default for ServiceLimitations {
    fn default() -> Self {
        Self {
            max_execution_time: Some(300), // 5 minutes
            max_result_size: Some(1_000_000),
            max_offset: Some(100_000),
            default_limit: Some(1000),
            max_limit: Some(10_000),
            service_call_timeout: Some(30),
        }
    }
}

/// Service description builder
pub struct ServiceDescriptionBuilder {
    description: ServiceDescription,
}

impl ServiceDescriptionBuilder {
    /// Create new builder
    pub fn new(endpoint: String) -> Self {
        Self {
            description: ServiceDescription::new(endpoint),
        }
    }

    /// Set default dataset
    pub fn with_default_dataset(mut self, dataset: DatasetDescription) -> Self {
        self.description.default_dataset = Some(dataset);
        self
    }

    /// Add named dataset
    pub fn add_named_dataset(mut self, dataset: DatasetDescription) -> Self {
        self.description.named_datasets.push(dataset);
        self
    }

    /// Add feature
    pub fn add_feature(mut self, feature: Feature) -> Self {
        self.description.features.insert(feature);
        self
    }

    /// Add language extension
    pub fn add_language_extension(mut self, extension: LanguageExtension) -> Self {
        self.description.language_extensions.insert(extension);
        self
    }

    /// Add result format
    pub fn add_result_format(mut self, format: ResultFormat) -> Self {
        if !self.description.result_formats.contains(&format) {
            self.description.result_formats.push(format);
        }
        self
    }

    /// Set limitations
    pub fn with_limitations(mut self, limitations: ServiceLimitations) -> Self {
        self.description.limitations = limitations;
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.description.metadata.insert(key, value);
        self
    }

    /// Build service description
    pub fn build(self) -> ServiceDescription {
        self.description
    }
}

/// Service description registry
#[derive(Clone)]
pub struct ServiceDescriptionRegistry {
    descriptions: Arc<dashmap::DashMap<String, ServiceDescription>>,
}

impl ServiceDescriptionRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            descriptions: Arc::new(dashmap::DashMap::new()),
        }
    }

    /// Register a service description
    pub fn register(&self, endpoint: String, description: ServiceDescription) {
        self.descriptions.insert(endpoint, description);
    }

    /// Get service description
    pub fn get(&self, endpoint: &str) -> Option<ServiceDescription> {
        self.descriptions.get(endpoint).map(|d| d.clone())
    }

    /// List all endpoints
    pub fn endpoints(&self) -> Vec<String> {
        self.descriptions
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Remove service description
    pub fn unregister(&self, endpoint: &str) -> Option<ServiceDescription> {
        self.descriptions.remove(endpoint).map(|(_, v)| v)
    }
}

impl Default for ServiceDescriptionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create default OxiRS service description
pub fn create_default_service_description(endpoint: String) -> ServiceDescription {
    ServiceDescriptionBuilder::new(endpoint)
        // Core SPARQL 1.1 features
        .add_feature(Feature::SPARQL11Query)
        .add_feature(Feature::Aggregates)
        .add_feature(Feature::PropertyPaths)
        .add_feature(Feature::Subqueries)
        .add_feature(Feature::Bind)
        .add_feature(Feature::Values)
        .add_feature(Feature::Negation)
        // Language extensions
        .add_language_extension(LanguageExtension::RDFStar)
        .add_language_extension(LanguageExtension::PropertyFunctions)
        .add_language_extension(LanguageExtension::StoredProcedures)
        .add_language_extension(LanguageExtension::CustomAggregates)
        // Result formats
        .add_result_format(ResultFormat::Json)
        .add_result_format(ResultFormat::Xml)
        .add_result_format(ResultFormat::Csv)
        .add_result_format(ResultFormat::Tsv)
        .add_result_format(ResultFormat::Binary)
        // Metadata
        .add_metadata("name".to_string(), "OxiRS SPARQL Endpoint".to_string())
        .add_metadata("version".to_string(), "0.1.0-rc.1".to_string())
        .add_metadata(
            "engine".to_string(),
            "OxiRS ARQ - Jena-compatible SPARQL engine".to_string(),
        )
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_description_creation() {
        let desc = ServiceDescription::new("http://example.org/sparql".to_string());
        assert_eq!(desc.endpoint, "http://example.org/sparql");
        assert!(desc.features.is_empty());
        assert!(!desc.result_formats.is_empty());
    }

    #[test]
    fn test_service_description_builder() {
        let desc = ServiceDescriptionBuilder::new("http://example.org/sparql".to_string())
            .add_feature(Feature::SPARQL11Query)
            .add_feature(Feature::Aggregates)
            .add_language_extension(LanguageExtension::RDFStar)
            .build();

        assert_eq!(desc.features.len(), 2);
        assert!(desc.features.contains(&Feature::SPARQL11Query));
        assert!(desc.features.contains(&Feature::Aggregates));
        assert_eq!(desc.language_extensions.len(), 1);
        assert!(desc
            .language_extensions
            .contains(&LanguageExtension::RDFStar));
    }

    #[test]
    fn test_dataset_description() {
        let mut dataset = DatasetDescription::new()
            .with_name("Test Dataset".to_string())
            .with_triple_count(10000);

        dataset.add_default_graph("http://example.org/graph1".to_string());
        dataset.add_named_graph(NamedGraphDescription {
            name: "http://example.org/graph2".to_string(),
            triple_count: 5000,
            metadata: HashMap::new(),
        });

        assert_eq!(dataset.name, Some("Test Dataset".to_string()));
        assert_eq!(dataset.triple_count, 10000);
        assert_eq!(dataset.default_graphs.len(), 1);
        assert_eq!(dataset.named_graphs.len(), 1);
    }

    #[test]
    fn test_feature_as_iri() {
        assert_eq!(Feature::SPARQL11Query.as_iri(), "SPARQL11Query");
        assert_eq!(Feature::PropertyPaths.as_iri(), "PropertyPaths");
        assert_eq!(Feature::Values.as_iri(), "Values");
    }

    #[test]
    fn test_language_extension_as_iri() {
        assert_eq!(LanguageExtension::RDFStar.as_iri(), "RDFStar");
        assert_eq!(
            LanguageExtension::PropertyFunctions.as_iri(),
            "PropertyFunctions"
        );
        assert_eq!(LanguageExtension::VectorSearch.as_iri(), "VectorSearch");
    }

    #[test]
    fn test_service_description_to_json() {
        let desc = ServiceDescriptionBuilder::new("http://example.org/sparql".to_string())
            .add_feature(Feature::SPARQL11Query)
            .build();

        let json = desc.to_json().unwrap();
        assert!(json.contains("http://example.org/sparql"));
        assert!(json.contains("SPARQL11Query"));
    }

    #[test]
    fn test_service_description_to_turtle() {
        let desc = ServiceDescriptionBuilder::new("http://example.org/sparql".to_string())
            .add_feature(Feature::SPARQL11Query)
            .add_feature(Feature::PropertyPaths)
            .build();

        let turtle = desc.to_turtle().unwrap();
        assert!(turtle.contains("@prefix sd:"));
        assert!(turtle.contains("sd:Service"));
        assert!(turtle.contains("sd:feature"));
        assert!(turtle.contains("SPARQL11Query"));
        assert!(turtle.contains("PropertyPaths"));
    }

    #[test]
    fn test_service_limitations() {
        let limits = ServiceLimitations::default();
        assert_eq!(limits.max_execution_time, Some(300));
        assert_eq!(limits.default_limit, Some(1000));
        assert_eq!(limits.max_limit, Some(10_000));
    }

    #[test]
    fn test_extension_function_info() {
        let func = ExtensionFunction {
            uri: "http://example.org/fn#myFunc".to_string(),
            name: "myFunc".to_string(),
            documentation: "Custom function".to_string(),
            parameters: vec![ParameterInfo {
                name: "input".to_string(),
                param_type: "xsd:string".to_string(),
                optional: false,
                default_value: None,
            }],
            return_type: "xsd:string".to_string(),
            deterministic: true,
        };

        assert_eq!(func.name, "myFunc");
        assert_eq!(func.parameters.len(), 1);
        assert!(func.deterministic);
    }

    #[test]
    fn test_service_description_registry() {
        let registry = ServiceDescriptionRegistry::new();

        let desc1 = ServiceDescription::new("http://example.org/sparql1".to_string());
        let desc2 = ServiceDescription::new("http://example.org/sparql2".to_string());

        registry.register("endpoint1".to_string(), desc1);
        registry.register("endpoint2".to_string(), desc2);

        assert_eq!(registry.endpoints().len(), 2);
        assert!(registry.get("endpoint1").is_some());
        assert!(registry.get("endpoint2").is_some());
        assert!(registry.get("endpoint3").is_none());

        registry.unregister("endpoint1");
        assert_eq!(registry.endpoints().len(), 1);
    }

    #[test]
    fn test_create_default_service_description() {
        let desc = create_default_service_description("http://example.org/sparql".to_string());

        // Check features
        assert!(desc.features.contains(&Feature::SPARQL11Query));
        assert!(desc.features.contains(&Feature::Values));
        assert!(desc.features.contains(&Feature::PropertyPaths));

        // Check extensions
        assert!(desc
            .language_extensions
            .contains(&LanguageExtension::RDFStar));
        assert!(desc
            .language_extensions
            .contains(&LanguageExtension::PropertyFunctions));

        // Check formats
        assert!(desc.result_formats.contains(&ResultFormat::Json));
        assert!(desc.result_formats.contains(&ResultFormat::Xml));
        assert!(desc.result_formats.contains(&ResultFormat::Binary));

        // Check metadata
        assert_eq!(
            desc.metadata.get("name"),
            Some(&"OxiRS SPARQL Endpoint".to_string())
        );
        assert_eq!(
            desc.metadata.get("version"),
            Some(&"0.1.0-rc.1".to_string())
        );
    }

    #[test]
    fn test_procedure_info() {
        let proc = ProcedureInfo {
            uri: "http://example.org/proc#test".to_string(),
            name: "test".to_string(),
            documentation: "Test procedure".to_string(),
            parameters: vec![],
            has_side_effects: false,
            deterministic: true,
        };

        assert_eq!(proc.name, "test");
        assert!(!proc.has_side_effects);
        assert!(proc.deterministic);
    }

    #[test]
    fn test_property_function_info() {
        let prop_func = PropertyFunctionInfo {
            uri: "http://example.org/pf#member".to_string(),
            name: "member".to_string(),
            documentation: "List membership".to_string(),
            subject_type: "Node".to_string(),
            object_type: "List".to_string(),
        };

        assert_eq!(prop_func.name, "member");
        assert_eq!(prop_func.subject_type, "Node");
        assert_eq!(prop_func.object_type, "List");
    }

    #[test]
    fn test_aggregate_info() {
        let agg = AggregateInfo {
            uri: "http://example.org/agg#median".to_string(),
            name: "MEDIAN".to_string(),
            documentation: "Median aggregate".to_string(),
            supports_distinct: true,
        };

        assert_eq!(agg.name, "MEDIAN");
        assert!(agg.supports_distinct);
    }

    #[test]
    fn test_service_description_add_functions() {
        let mut desc = ServiceDescription::new("http://example.org/sparql".to_string());

        desc.add_extension_function(ExtensionFunction {
            uri: "http://example.org/fn#test".to_string(),
            name: "test".to_string(),
            documentation: "Test".to_string(),
            parameters: vec![],
            return_type: "xsd:string".to_string(),
            deterministic: true,
        });

        desc.add_procedure(ProcedureInfo {
            uri: "http://example.org/proc#test".to_string(),
            name: "test".to_string(),
            documentation: "Test".to_string(),
            parameters: vec![],
            has_side_effects: false,
            deterministic: true,
        });

        desc.add_property_function(PropertyFunctionInfo {
            uri: "http://example.org/pf#test".to_string(),
            name: "test".to_string(),
            documentation: "Test".to_string(),
            subject_type: "Node".to_string(),
            object_type: "Node".to_string(),
        });

        desc.add_aggregate(AggregateInfo {
            uri: "http://example.org/agg#test".to_string(),
            name: "TEST".to_string(),
            documentation: "Test".to_string(),
            supports_distinct: true,
        });

        assert_eq!(desc.extension_functions.len(), 1);
        assert_eq!(desc.procedures.len(), 1);
        assert_eq!(desc.property_functions.len(), 1);
        assert_eq!(desc.aggregates.len(), 1);
    }
}
