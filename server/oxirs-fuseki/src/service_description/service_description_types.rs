//! SD vocabulary types: `ServiceDescription`, `SdFeature`, `SdLanguage`,
//! `SdResultFormat`, `SdInputFormat`, and `EntailmentRegime`.

// ────────────────────────────────────────────────────────────────────────────
// Namespace constants (re-exported from mod.rs)
// ────────────────────────────────────────────────────────────────────────────

/// W3C SPARQL Service Description namespace prefix
pub const SD_NS: &str = "http://www.w3.org/ns/sparql-service-description#";
/// RDF namespace
pub const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
/// RDFS namespace
pub const RDFS_NS: &str = "http://www.w3.org/2000/01/rdf-schema#";
/// VOID namespace
pub const VOID_NS: &str = "http://rdfs.org/ns/void#";
/// OWL namespace
pub const OWL_NS: &str = "http://www.w3.org/2002/07/owl#";

// ────────────────────────────────────────────────────────────────────────────
// ServiceDescription
// ────────────────────────────────────────────────────────────────────────────

/// A W3C SPARQL Service Description document
///
/// Describes the capabilities and configuration of a SPARQL endpoint,
/// including supported query languages, result formats, and extensions.
#[derive(Debug, Clone, PartialEq)]
pub struct ServiceDescription {
    /// The URL of the SPARQL endpoint
    pub endpoint_url: String,
    /// The name of the dataset this endpoint serves
    pub dataset_name: String,
    /// Features supported by this endpoint
    pub features: Vec<SdFeature>,
    /// SPARQL language variants supported
    pub supported_languages: Vec<SdLanguage>,
    /// Formats accepted for query results
    pub result_formats: Vec<SdResultFormat>,
    /// Formats accepted for RDF input (SPARQL Update, GSP)
    pub input_formats: Vec<SdInputFormat>,
    /// Extension function IRIs advertised by this endpoint
    pub extension_functions: Vec<String>,
    /// Entailment regimes supported
    pub entailment_regimes: Vec<EntailmentRegime>,
    /// Optional human-readable label for the service
    pub label: Option<String>,
    /// Optional description of the service
    pub description: Option<String>,
}

// ────────────────────────────────────────────────────────────────────────────
// SdFeature
// ────────────────────────────────────────────────────────────────────────────

/// SPARQL Service Description features (sd:Feature)
///
/// Corresponds to terms in the sd: vocabulary under <http://www.w3.org/ns/sparql-service-description#>
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SdFeature {
    /// Service can dereference IRIs to retrieve RDF descriptions
    DereferencesUris,
    /// Default graph is the union of all named graphs
    UnionDefaultGraph,
    /// Requires an explicit dataset declaration
    RequiresDataset,
    /// Supports management of empty named graphs
    EmptyGraphs,
    /// Supports basic federated query (SERVICE clause)
    BasicFederatedQuery,
    /// Supports RDF-star (RDF 1.2 quoted triples)
    RdfStar,
    /// Supports SPARQL-star extensions
    SparqlStar,
    /// Supports SPARQL 1.2 shacl-based constraints
    ConstraintValidation,
    /// Supports timeout hints from clients
    TimeoutHints,
}

impl SdFeature {
    /// Returns the full IRI for this feature in the sd: namespace
    pub fn as_iri(&self) -> &'static str {
        match self {
            SdFeature::DereferencesUris => {
                "http://www.w3.org/ns/sparql-service-description#DereferencesURIs"
            }
            SdFeature::UnionDefaultGraph => {
                "http://www.w3.org/ns/sparql-service-description#UnionDefaultGraph"
            }
            SdFeature::RequiresDataset => {
                "http://www.w3.org/ns/sparql-service-description#RequiresDataset"
            }
            SdFeature::EmptyGraphs => "http://www.w3.org/ns/sparql-service-description#EmptyGraphs",
            SdFeature::BasicFederatedQuery => {
                "http://www.w3.org/ns/sparql-service-description#BasicFederatedQuery"
            }
            SdFeature::RdfStar => "http://www.w3.org/ns/sparql-service-description#RDFStar",
            SdFeature::SparqlStar => "http://www.w3.org/ns/sparql-service-description#SPARQLStar",
            SdFeature::ConstraintValidation => {
                "http://www.w3.org/ns/sparql-service-description#ConstraintValidation"
            }
            SdFeature::TimeoutHints => {
                "http://www.w3.org/ns/sparql-service-description#TimeoutHints"
            }
        }
    }

    /// Returns a human-readable label for this feature
    pub fn label(&self) -> &'static str {
        match self {
            SdFeature::DereferencesUris => "Dereferences URIs",
            SdFeature::UnionDefaultGraph => "Union Default Graph",
            SdFeature::RequiresDataset => "Requires Dataset",
            SdFeature::EmptyGraphs => "Empty Graphs",
            SdFeature::BasicFederatedQuery => "Basic Federated Query",
            SdFeature::RdfStar => "RDF-star",
            SdFeature::SparqlStar => "SPARQL-star",
            SdFeature::ConstraintValidation => "SHACL Constraint Validation",
            SdFeature::TimeoutHints => "Timeout Hints",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SdLanguage
// ────────────────────────────────────────────────────────────────────────────

/// SPARQL language variants (sd:Language)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SdLanguage {
    /// SPARQL 1.0 Query Language
    Sparql10Query,
    /// SPARQL 1.1 Query Language
    Sparql11Query,
    /// SPARQL 1.1 Update Language
    Sparql11Update,
    /// SPARQL 1.2 Query Language (OxiRS extension)
    Sparql12Query,
    /// SPARQL 1.2 Update Language (OxiRS extension)
    Sparql12Update,
}

impl SdLanguage {
    /// Returns the full IRI for this language in the sd: namespace
    pub fn as_iri(&self) -> &'static str {
        match self {
            SdLanguage::Sparql10Query => {
                "http://www.w3.org/ns/sparql-service-description#SPARQL10Query"
            }
            SdLanguage::Sparql11Query => {
                "http://www.w3.org/ns/sparql-service-description#SPARQL11Query"
            }
            SdLanguage::Sparql11Update => {
                "http://www.w3.org/ns/sparql-service-description#SPARQL11Update"
            }
            SdLanguage::Sparql12Query => {
                "http://www.w3.org/ns/sparql-service-description#SPARQL12Query"
            }
            SdLanguage::Sparql12Update => {
                "http://www.w3.org/ns/sparql-service-description#SPARQL12Update"
            }
        }
    }

    /// Returns a human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            SdLanguage::Sparql10Query => "SPARQL 1.0 Query",
            SdLanguage::Sparql11Query => "SPARQL 1.1 Query",
            SdLanguage::Sparql11Update => "SPARQL 1.1 Update",
            SdLanguage::Sparql12Query => "SPARQL 1.2 Query",
            SdLanguage::Sparql12Update => "SPARQL 1.2 Update",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SdResultFormat
// ────────────────────────────────────────────────────────────────────────────

/// Result serialization formats (sd:ResultFormat)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SdResultFormat {
    /// SPARQL Query Results JSON Format
    SparqlResultsJson,
    /// SPARQL Query Results XML Format
    SparqlResultsXml,
    /// SPARQL Query Results CSV Format
    SparqlResultsCsv,
    /// SPARQL Query Results TSV Format
    SparqlResultsTsv,
    /// Turtle RDF serialization
    Turtle,
    /// N-Triples RDF serialization
    NTriples,
    /// N-Quads RDF serialization
    NQuads,
    /// TriG RDF serialization
    TriG,
    /// JSON-LD RDF serialization
    JsonLd,
    /// RDF/XML serialization
    RdfXml,
    /// N3 / Notation3 (legacy)
    N3,
    /// LD-Patch format
    LdPatch,
}

impl SdResultFormat {
    /// Returns the full IRI identifying this format
    pub fn as_iri(&self) -> &'static str {
        match self {
            SdResultFormat::SparqlResultsJson => "http://www.w3.org/ns/formats/SPARQL_Results_JSON",
            SdResultFormat::SparqlResultsXml => "http://www.w3.org/ns/formats/SPARQL_Results_XML",
            SdResultFormat::SparqlResultsCsv => {
                "http://www.w3.org/ns/formats/SPARQL_Results_CSV-TSV"
            }
            SdResultFormat::SparqlResultsTsv => "http://www.w3.org/ns/formats/SPARQL_Results_TSV",
            SdResultFormat::Turtle => "http://www.w3.org/ns/formats/Turtle",
            SdResultFormat::NTriples => "http://www.w3.org/ns/formats/N-Triples",
            SdResultFormat::NQuads => "http://www.w3.org/ns/formats/N-Quads",
            SdResultFormat::TriG => "http://www.w3.org/ns/formats/TriG",
            SdResultFormat::JsonLd => "http://www.w3.org/ns/formats/JSON-LD",
            SdResultFormat::RdfXml => "http://www.w3.org/ns/formats/RDF_XML",
            SdResultFormat::N3 => "http://www.w3.org/ns/formats/N3",
            SdResultFormat::LdPatch => "http://www.w3.org/ns/formats/LD_Patch",
        }
    }

    /// Returns the primary MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            SdResultFormat::SparqlResultsJson => "application/sparql-results+json",
            SdResultFormat::SparqlResultsXml => "application/sparql-results+xml",
            SdResultFormat::SparqlResultsCsv => "text/csv",
            SdResultFormat::SparqlResultsTsv => "text/tab-separated-values",
            SdResultFormat::Turtle => "text/turtle",
            SdResultFormat::NTriples => "application/n-triples",
            SdResultFormat::NQuads => "application/n-quads",
            SdResultFormat::TriG => "application/trig",
            SdResultFormat::JsonLd => "application/ld+json",
            SdResultFormat::RdfXml => "application/rdf+xml",
            SdResultFormat::N3 => "text/n3",
            SdResultFormat::LdPatch => "application/ld-patch+json",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SdInputFormat
// ────────────────────────────────────────────────────────────────────────────

/// RDF input formats accepted for GSP/SPARQL Update
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SdInputFormat {
    /// Turtle RDF serialization
    Turtle,
    /// N-Triples RDF serialization
    NTriples,
    /// N-Quads RDF serialization
    NQuads,
    /// TriG RDF serialization
    TriG,
    /// JSON-LD RDF serialization
    JsonLd,
    /// RDF/XML serialization
    RdfXml,
    /// N3 / Notation3
    N3,
}

impl SdInputFormat {
    /// Returns the full IRI identifying this format
    pub fn as_iri(&self) -> &'static str {
        match self {
            SdInputFormat::Turtle => "http://www.w3.org/ns/formats/Turtle",
            SdInputFormat::NTriples => "http://www.w3.org/ns/formats/N-Triples",
            SdInputFormat::NQuads => "http://www.w3.org/ns/formats/N-Quads",
            SdInputFormat::TriG => "http://www.w3.org/ns/formats/TriG",
            SdInputFormat::JsonLd => "http://www.w3.org/ns/formats/JSON-LD",
            SdInputFormat::RdfXml => "http://www.w3.org/ns/formats/RDF_XML",
            SdInputFormat::N3 => "http://www.w3.org/ns/formats/N3",
        }
    }

    /// Returns the primary MIME type
    pub fn mime_type(&self) -> &'static str {
        match self {
            SdInputFormat::Turtle => "text/turtle",
            SdInputFormat::NTriples => "application/n-triples",
            SdInputFormat::NQuads => "application/n-quads",
            SdInputFormat::TriG => "application/trig",
            SdInputFormat::JsonLd => "application/ld+json",
            SdInputFormat::RdfXml => "application/rdf+xml",
            SdInputFormat::N3 => "text/n3",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// EntailmentRegime
// ────────────────────────────────────────────────────────────────────────────

/// Entailment regimes supported by the endpoint
///
/// Corresponds to the OWL/RDF entailment profiles under
/// <http://www.w3.org/ns/entailment/>
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntailmentRegime {
    /// Simple entailment (RDF-graph entailment)
    Simple,
    /// RDF entailment
    Rdf,
    /// RDFS entailment
    Rdfs,
    /// OWL 2 Direct Semantics
    Owl2Direct,
    /// OWL 2 RL profile
    Owl2Rl,
    /// OWL 2 EL profile
    Owl2El,
    /// OWL 2 QL profile
    Owl2Ql,
    /// D-entailment (datatype reasoning)
    DEntailment,
}

impl EntailmentRegime {
    /// Returns the full IRI for this entailment regime
    pub fn as_iri(&self) -> &'static str {
        match self {
            EntailmentRegime::Simple => "http://www.w3.org/ns/entailment/Simple",
            EntailmentRegime::Rdf => "http://www.w3.org/ns/entailment/RDF",
            EntailmentRegime::Rdfs => "http://www.w3.org/ns/entailment/RDFS",
            EntailmentRegime::Owl2Direct => "http://www.w3.org/ns/entailment/OWL-Direct",
            EntailmentRegime::Owl2Rl => "http://www.w3.org/ns/entailment/OWL-RL",
            EntailmentRegime::Owl2El => "http://www.w3.org/ns/entailment/OWL-EL",
            EntailmentRegime::Owl2Ql => "http://www.w3.org/ns/entailment/OWL-QL",
            EntailmentRegime::DEntailment => "http://www.w3.org/ns/entailment/D",
        }
    }

    /// Returns the label for this entailment regime
    pub fn label(&self) -> &'static str {
        match self {
            EntailmentRegime::Simple => "Simple",
            EntailmentRegime::Rdf => "RDF",
            EntailmentRegime::Rdfs => "RDFS",
            EntailmentRegime::Owl2Direct => "OWL 2 Direct Semantics",
            EntailmentRegime::Owl2Rl => "OWL 2 RL",
            EntailmentRegime::Owl2El => "OWL 2 EL",
            EntailmentRegime::Owl2Ql => "OWL 2 QL",
            EntailmentRegime::DEntailment => "D-Entailment",
        }
    }
}
