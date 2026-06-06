//! # Validation Types
//!
//! Request/response types, query parameter structs, and shared data types
//! used by all validation endpoints.

use serde::{Deserialize, Serialize};

// ============================================================================
// Validation Request/Response Types
// ============================================================================

/// Query validation request
#[derive(Debug, Clone, Deserialize)]
pub struct QueryValidationRequest {
    /// The SPARQL query to validate
    pub query: String,
    /// Optional syntax specification (defaults to SPARQL 1.1)
    #[serde(default = "default_sparql_syntax")]
    pub syntax: String,
    /// Whether to include algebra output
    #[serde(default = "default_true")]
    pub include_algebra: bool,
    /// Whether to include optimized algebra
    #[serde(default = "default_true")]
    pub include_optimized: bool,
}

pub fn default_sparql_syntax() -> String {
    "sparql11".to_string()
}

pub fn default_true() -> bool {
    true
}

/// Query validation response
#[derive(Debug, Clone, Serialize)]
pub struct QueryValidationResponse {
    /// Whether the query is valid
    pub valid: bool,
    /// Original input query
    pub input: String,
    /// Formatted/pretty-printed query (if valid)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formatted: Option<String>,
    /// Query type (SELECT, CONSTRUCT, ASK, DESCRIBE)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_type: Option<String>,
    /// Algebra representation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algebra: Option<String>,
    /// Optimized algebra representation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algebra_optimized: Option<String>,
    /// Variables in the query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<Vec<String>>,
    /// Prefixes used in the query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefixes: Option<Vec<PrefixMapping>>,
    /// Parse errors (if invalid)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<ValidationError>,
    /// Warnings (non-fatal issues)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ValidationWarning>,
}

/// Update validation request
#[derive(Debug, Clone, Deserialize)]
pub struct UpdateValidationRequest {
    /// The SPARQL Update to validate
    pub update: String,
    /// Optional syntax specification
    #[serde(default = "default_sparql_syntax")]
    pub syntax: String,
}

/// Update validation response
#[derive(Debug, Clone, Serialize)]
pub struct UpdateValidationResponse {
    /// Whether the update is valid
    pub valid: bool,
    /// Original input update
    pub input: String,
    /// Formatted/pretty-printed update (if valid)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formatted: Option<String>,
    /// Operation types (INSERT DATA, DELETE DATA, etc.)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub operations: Vec<String>,
    /// Graphs affected by the update
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub affected_graphs: Vec<String>,
    /// Parse errors (if invalid)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<ValidationError>,
    /// Warnings
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ValidationWarning>,
}

/// IRI validation request
#[derive(Debug, Clone, Deserialize)]
pub struct IriValidationRequest {
    /// IRIs to validate (can be multiple)
    pub iris: Vec<String>,
    /// Whether to check for relative IRIs (default: true)
    #[serde(default = "default_true")]
    pub check_relative: bool,
}

/// IRI validation response
#[derive(Debug, Clone, Serialize)]
pub struct IriValidationResponse {
    /// Validation results for each IRI
    pub results: Vec<IriValidationResult>,
    /// Overall summary
    pub summary: ValidationSummary,
}

/// Individual IRI validation result
#[derive(Debug, Clone, Serialize)]
pub struct IriValidationResult {
    /// The IRI that was validated
    pub iri: String,
    /// Whether the IRI is valid
    pub valid: bool,
    /// Whether the IRI is absolute
    pub is_absolute: bool,
    /// Whether the IRI is relative
    pub is_relative: bool,
    /// IRI scheme (http, https, urn, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
    /// Errors found
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
    /// Warnings found
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// RDF data validation request
#[derive(Debug, Clone, Deserialize)]
pub struct DataValidationRequest {
    /// The RDF data to validate
    pub data: String,
    /// Content type / format (turtle, ntriples, nquads, rdfxml, jsonld)
    #[serde(default = "default_format")]
    pub format: String,
    /// Base IRI for relative IRI resolution
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
}

pub fn default_format() -> String {
    "turtle".to_string()
}

/// RDF data validation response
#[derive(Debug, Clone, Serialize)]
pub struct DataValidationResponse {
    /// Whether the data is valid
    pub valid: bool,
    /// Format detected/used
    pub format: String,
    /// Number of triples/quads parsed
    pub triple_count: usize,
    /// Number of graphs (for quads)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_count: Option<usize>,
    /// Subjects found
    pub subject_count: usize,
    /// Predicates found
    pub predicate_count: usize,
    /// Objects found
    pub object_count: usize,
    /// Blank nodes found
    pub blank_node_count: usize,
    /// Literals found
    pub literal_count: usize,
    /// Parse errors (if invalid)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<ValidationError>,
    /// Warnings
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ValidationWarning>,
    /// Sample triples (first 10)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sample_triples: Vec<String>,
}

/// Language tag validation request
#[derive(Debug, Clone, Deserialize)]
pub struct LangTagValidationRequest {
    /// Language tags to validate
    pub tags: Vec<String>,
}

/// Language tag validation response
#[derive(Debug, Clone, Serialize)]
pub struct LangTagValidationResponse {
    /// Validation results for each tag
    pub results: Vec<LangTagValidationResult>,
    /// Overall summary
    pub summary: ValidationSummary,
}

/// Individual language tag validation result
#[derive(Debug, Clone, Serialize)]
pub struct LangTagValidationResult {
    /// The tag that was validated
    pub tag: String,
    /// Whether the tag is valid (well-formed)
    pub valid: bool,
    /// Primary language subtag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Script subtag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    /// Region subtag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Variant subtags
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<String>,
    /// Extension subtags
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<String>,
    /// Private use subtag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_use: Option<String>,
    /// Errors found
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
    /// Warnings found
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Validation error details
#[derive(Debug, Clone, Serialize)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    /// Line number (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Column number (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<usize>,
    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Validation warning
#[derive(Debug, Clone, Serialize)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Line number (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Warning code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Prefix mapping
#[derive(Debug, Clone, Serialize)]
pub struct PrefixMapping {
    /// Prefix (e.g., "rdf")
    pub prefix: String,
    /// IRI (e.g., `http://www.w3.org/1999/02/22-rdf-syntax-ns#`)
    pub iri: String,
}

/// Validation summary
#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    /// Total items validated
    pub total: usize,
    /// Number of valid items
    pub valid: usize,
    /// Number of invalid items
    pub invalid: usize,
    /// Number of warnings
    pub warnings: usize,
}

// ============================================================================
// Query Parameters for GET requests
// ============================================================================

/// Query parameters for query validation
#[derive(Debug, Clone, Deserialize)]
pub struct QueryValidationParams {
    pub query: Option<String>,
    pub syntax: Option<String>,
}

/// Query parameters for update validation
#[derive(Debug, Clone, Deserialize)]
pub struct UpdateValidationParams {
    pub update: Option<String>,
    pub syntax: Option<String>,
}

/// Query parameters for IRI validation
#[derive(Debug, Clone, Deserialize)]
pub struct IriValidationParams {
    /// Single IRI or comma-separated list
    pub iri: Option<String>,
}

/// Query parameters for language tag validation
#[derive(Debug, Clone, Deserialize)]
pub struct LangTagValidationParams {
    /// Single tag or comma-separated list
    pub tag: Option<String>,
}
