//! W3C SPARQL Service Description (SD ontology) implementation
//!
//! Reference: <https://www.w3.org/TR/sparql11-service-description/>
//!
//! Allows SPARQL endpoints to self-describe their capabilities by returning
//! an RDF graph conforming to the SPARQL Service Description vocabulary.

use serde_json::json;

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

/// Builder for constructing a [`ServiceDescription`]
#[derive(Debug, Default)]
pub struct ServiceDescriptionBuilder {
    endpoint_url: String,
    dataset_name: String,
    features: Vec<SdFeature>,
    supported_languages: Vec<SdLanguage>,
    result_formats: Vec<SdResultFormat>,
    input_formats: Vec<SdInputFormat>,
    extension_functions: Vec<String>,
    entailment_regimes: Vec<EntailmentRegime>,
    label: Option<String>,
    description: Option<String>,
}

impl ServiceDescriptionBuilder {
    /// Create a new builder with required fields
    pub fn new(endpoint_url: impl Into<String>, dataset_name: impl Into<String>) -> Self {
        ServiceDescriptionBuilder {
            endpoint_url: endpoint_url.into(),
            dataset_name: dataset_name.into(),
            ..Default::default()
        }
    }

    /// Add a supported feature
    pub fn with_feature(mut self, feature: SdFeature) -> Self {
        self.features.push(feature);
        self
    }

    /// Add multiple features at once
    pub fn with_features(mut self, features: impl IntoIterator<Item = SdFeature>) -> Self {
        self.features.extend(features);
        self
    }

    /// Add a supported query/update language
    pub fn with_language(mut self, language: SdLanguage) -> Self {
        self.supported_languages.push(language);
        self
    }

    /// Add multiple supported languages
    pub fn with_languages(mut self, languages: impl IntoIterator<Item = SdLanguage>) -> Self {
        self.supported_languages.extend(languages);
        self
    }

    /// Add a supported result format
    pub fn with_result_format(mut self, format: SdResultFormat) -> Self {
        self.result_formats.push(format);
        self
    }

    /// Add multiple result formats
    pub fn with_result_formats(
        mut self,
        formats: impl IntoIterator<Item = SdResultFormat>,
    ) -> Self {
        self.result_formats.extend(formats);
        self
    }

    /// Add a supported input format
    pub fn with_input_format(mut self, format: SdInputFormat) -> Self {
        self.input_formats.push(format);
        self
    }

    /// Add multiple input formats
    pub fn with_input_formats(mut self, formats: impl IntoIterator<Item = SdInputFormat>) -> Self {
        self.input_formats.extend(formats);
        self
    }

    /// Add an extension function IRI
    pub fn with_extension_function(mut self, iri: impl Into<String>) -> Self {
        self.extension_functions.push(iri.into());
        self
    }

    /// Add multiple extension function IRIs
    pub fn with_extension_functions(mut self, iris: impl IntoIterator<Item = String>) -> Self {
        self.extension_functions.extend(iris);
        self
    }

    /// Add an entailment regime
    pub fn with_entailment_regime(mut self, regime: EntailmentRegime) -> Self {
        self.entailment_regimes.push(regime);
        self
    }

    /// Set human-readable label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set human-readable description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Build the [`ServiceDescription`]
    pub fn build(self) -> ServiceDescription {
        ServiceDescription {
            endpoint_url: self.endpoint_url,
            dataset_name: self.dataset_name,
            features: self.features,
            supported_languages: self.supported_languages,
            result_formats: self.result_formats,
            input_formats: self.input_formats,
            extension_functions: self.extension_functions,
            entailment_regimes: self.entailment_regimes,
            label: self.label,
            description: self.description,
        }
    }
}

impl ServiceDescription {
    /// Create a builder for a new `ServiceDescription`
    pub fn builder(
        endpoint_url: impl Into<String>,
        dataset_name: impl Into<String>,
    ) -> ServiceDescriptionBuilder {
        ServiceDescriptionBuilder::new(endpoint_url, dataset_name)
    }

    /// Return the default `ServiceDescription` for an OxiRS endpoint
    ///
    /// Includes all standard SPARQL 1.1 + SPARQL 1.2 capabilities.
    pub fn default_for_oxirs(endpoint_url: &str, dataset_name: &str) -> Self {
        ServiceDescriptionBuilder::new(endpoint_url, dataset_name)
            .with_label("OxiRS SPARQL Endpoint")
            .with_description(
                "OxiRS: A Rust-native, JVM-free RDF platform with SPARQL 1.1/1.2 support, \
                 GraphQL integration, and AI-augmented reasoning.",
            )
            .with_features([
                SdFeature::DereferencesUris,
                SdFeature::UnionDefaultGraph,
                SdFeature::EmptyGraphs,
                SdFeature::BasicFederatedQuery,
                SdFeature::RdfStar,
                SdFeature::SparqlStar,
            ])
            .with_languages([
                SdLanguage::Sparql10Query,
                SdLanguage::Sparql11Query,
                SdLanguage::Sparql11Update,
                SdLanguage::Sparql12Query,
                SdLanguage::Sparql12Update,
            ])
            .with_result_formats([
                SdResultFormat::SparqlResultsJson,
                SdResultFormat::SparqlResultsXml,
                SdResultFormat::SparqlResultsCsv,
                SdResultFormat::SparqlResultsTsv,
                SdResultFormat::Turtle,
                SdResultFormat::NTriples,
                SdResultFormat::NQuads,
                SdResultFormat::TriG,
                SdResultFormat::JsonLd,
                SdResultFormat::RdfXml,
            ])
            .with_input_formats([
                SdInputFormat::Turtle,
                SdInputFormat::NTriples,
                SdInputFormat::NQuads,
                SdInputFormat::TriG,
                SdInputFormat::JsonLd,
                SdInputFormat::RdfXml,
            ])
            .with_entailment_regime(EntailmentRegime::Simple)
            .with_entailment_regime(EntailmentRegime::Rdf)
            .with_entailment_regime(EntailmentRegime::Rdfs)
            .build()
    }

    /// Generate the Service Description as Turtle RDF
    ///
    /// Returns a conforming Turtle document per the W3C SD specification.
    pub fn to_turtle(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Prefixes
        out.push_str("@prefix sd: <http://www.w3.org/ns/sparql-service-description#> .\n");
        out.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        out.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        out.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");
        out.push_str("@prefix void: <http://rdfs.org/ns/void#> .\n\n");

        // Service node
        out.push_str(&format!("<{}>\n", self.endpoint_url));
        out.push_str("    a sd:Service ;\n");

        // Label
        if let Some(ref lbl) = self.label {
            out.push_str(&format!(
                "    rdfs:label \"{}\" ;\n",
                escape_turtle_string(lbl)
            ));
        }

        // Description
        if let Some(ref desc) = self.description {
            out.push_str(&format!(
                "    rdfs:comment \"{}\" ;\n",
                escape_turtle_string(desc)
            ));
        }

        // Endpoint triple
        out.push_str(&format!("    sd:endpoint <{}> ;\n", self.endpoint_url));

        // Supported languages
        for lang in &self.supported_languages {
            out.push_str(&format!("    sd:supportedLanguage <{}> ;\n", lang.as_iri()));
        }

        // Result formats
        for fmt in &self.result_formats {
            out.push_str(&format!("    sd:resultFormat <{}> ;\n", fmt.as_iri()));
        }

        // Input formats
        for fmt in &self.input_formats {
            out.push_str(&format!("    sd:inputFormat <{}> ;\n", fmt.as_iri()));
        }

        // Features
        for feat in &self.features {
            out.push_str(&format!("    sd:feature <{}> ;\n", feat.as_iri()));
        }

        // Extension functions
        for func_iri in &self.extension_functions {
            out.push_str(&format!("    sd:extensionFunction <{}> ;\n", func_iri));
        }

        // Entailment regimes (via default entailment regime)
        for regime in &self.entailment_regimes {
            out.push_str(&format!(
                "    sd:defaultEntailmentRegime <{}> ;\n",
                regime.as_iri()
            ));
        }

        // Dataset block
        out.push_str(&format!(
            "    sd:defaultDataset [\n        a sd:Dataset ;\n        sd:defaultGraph [ a sd:Graph ] ;\n        rdfs:label \"{}\"\n    ] .\n",
            escape_turtle_string(&self.dataset_name)
        ));

        out
    }

    /// Generate the Service Description as JSON-LD
    ///
    /// Returns a `serde_json::Value` conforming to the JSON-LD representation
    /// of the W3C SD vocabulary.
    pub fn to_json_ld(&self) -> serde_json::Value {
        let supported_languages: Vec<serde_json::Value> = self
            .supported_languages
            .iter()
            .map(|l| json!({ "@id": l.as_iri() }))
            .collect();

        let result_formats: Vec<serde_json::Value> = self
            .result_formats
            .iter()
            .map(|f| json!({ "@id": f.as_iri() }))
            .collect();

        let input_formats: Vec<serde_json::Value> = self
            .input_formats
            .iter()
            .map(|f| json!({ "@id": f.as_iri() }))
            .collect();

        let features: Vec<serde_json::Value> = self
            .features
            .iter()
            .map(|f| json!({ "@id": f.as_iri() }))
            .collect();

        let extension_functions: Vec<serde_json::Value> = self
            .extension_functions
            .iter()
            .map(|iri| json!({ "@id": iri }))
            .collect();

        let entailment_regimes: Vec<serde_json::Value> = self
            .entailment_regimes
            .iter()
            .map(|r| json!({ "@id": r.as_iri() }))
            .collect();

        let mut service_obj = serde_json::Map::new();
        service_obj.insert("@id".into(), json!(self.endpoint_url));
        service_obj.insert(
            "@type".into(),
            json!("http://www.w3.org/ns/sparql-service-description#Service"),
        );
        service_obj.insert(
            "http://www.w3.org/ns/sparql-service-description#endpoint".into(),
            json!([{ "@id": self.endpoint_url }]),
        );

        if let Some(ref lbl) = self.label {
            service_obj.insert(
                "http://www.w3.org/2000/01/rdf-schema#label".into(),
                json!([{ "@value": lbl }]),
            );
        }

        if let Some(ref desc) = self.description {
            service_obj.insert(
                "http://www.w3.org/2000/01/rdf-schema#comment".into(),
                json!([{ "@value": desc }]),
            );
        }

        if !supported_languages.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#supportedLanguage".into(),
                json!(supported_languages),
            );
        }

        if !result_formats.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#resultFormat".into(),
                json!(result_formats),
            );
        }

        if !input_formats.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#inputFormat".into(),
                json!(input_formats),
            );
        }

        if !features.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#feature".into(),
                json!(features),
            );
        }

        if !extension_functions.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#extensionFunction".into(),
                json!(extension_functions),
            );
        }

        if !entailment_regimes.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#defaultEntailmentRegime".into(),
                json!(entailment_regimes),
            );
        }

        // Dataset description
        let dataset = json!({
            "@type": "http://www.w3.org/ns/sparql-service-description#Dataset",
            "http://www.w3.org/ns/sparql-service-description#defaultGraph": [
                {
                    "@type": "http://www.w3.org/ns/sparql-service-description#Graph"
                }
            ],
            "http://www.w3.org/2000/01/rdf-schema#label": [
                { "@value": self.dataset_name }
            ]
        });

        service_obj.insert(
            "http://www.w3.org/ns/sparql-service-description#defaultDataset".into(),
            json!([dataset]),
        );

        json!({
            "@context": {
                "sd": "http://www.w3.org/ns/sparql-service-description#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            },
            "@graph": [serde_json::Value::Object(service_obj)]
        })
    }

    /// Generate Service Description as RDF/XML
    ///
    /// Returns a valid RDF/XML serialization suitable for returning
    /// from a SPARQL endpoint when `application/rdf+xml` is requested.
    pub fn to_rdf_xml(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        out.push('\n');
        out.push_str(r#"<rdf:RDF"#);
        out.push_str("\n    xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"");
        out.push_str("\n    xmlns:rdfs=\"http://www.w3.org/2000/01/rdf-schema#\"");
        out.push_str("\n    xmlns:sd=\"http://www.w3.org/ns/sparql-service-description#\">\n\n");

        out.push_str(&format!(
            "  <sd:Service rdf:about=\"{}\">\n",
            escape_xml_attr(&self.endpoint_url)
        ));

        if let Some(ref lbl) = self.label {
            out.push_str(&format!(
                "    <rdfs:label>{}</rdfs:label>\n",
                escape_xml_string(lbl)
            ));
        }
        if let Some(ref desc) = self.description {
            out.push_str(&format!(
                "    <rdfs:comment>{}</rdfs:comment>\n",
                escape_xml_string(desc)
            ));
        }

        out.push_str(&format!(
            "    <sd:endpoint rdf:resource=\"{}\"/>\n",
            escape_xml_attr(&self.endpoint_url)
        ));

        for lang in &self.supported_languages {
            out.push_str(&format!(
                "    <sd:supportedLanguage rdf:resource=\"{}\"/>\n",
                escape_xml_attr(lang.as_iri())
            ));
        }

        for fmt in &self.result_formats {
            out.push_str(&format!(
                "    <sd:resultFormat rdf:resource=\"{}\"/>\n",
                escape_xml_attr(fmt.as_iri())
            ));
        }

        for fmt in &self.input_formats {
            out.push_str(&format!(
                "    <sd:inputFormat rdf:resource=\"{}\"/>\n",
                escape_xml_attr(fmt.as_iri())
            ));
        }

        for feat in &self.features {
            out.push_str(&format!(
                "    <sd:feature rdf:resource=\"{}\"/>\n",
                escape_xml_attr(feat.as_iri())
            ));
        }

        for func_iri in &self.extension_functions {
            out.push_str(&format!(
                "    <sd:extensionFunction rdf:resource=\"{}\"/>\n",
                escape_xml_attr(func_iri)
            ));
        }

        for regime in &self.entailment_regimes {
            out.push_str(&format!(
                "    <sd:defaultEntailmentRegime rdf:resource=\"{}\"/>\n",
                escape_xml_attr(regime.as_iri())
            ));
        }

        out.push_str(&format!(
            "    <sd:defaultDataset>\n      <sd:Dataset>\n        <sd:defaultGraph><sd:Graph/></sd:defaultGraph>\n        <rdfs:label>{}</rdfs:label>\n      </sd:Dataset>\n    </sd:defaultDataset>\n",
            escape_xml_string(&self.dataset_name)
        ));

        out.push_str("  </sd:Service>\n</rdf:RDF>\n");
        out
    }

    /// Returns the number of result formats advertised
    pub fn result_format_count(&self) -> usize {
        self.result_formats.len()
    }

    /// Returns the number of input formats advertised
    pub fn input_format_count(&self) -> usize {
        self.input_formats.len()
    }

    /// Returns `true` if the given language IRI is supported
    pub fn supports_language(&self, language_iri: &str) -> bool {
        self.supported_languages
            .iter()
            .any(|l| l.as_iri() == language_iri)
    }

    /// Returns `true` if the given feature IRI is supported
    pub fn has_feature(&self, feature_iri: &str) -> bool {
        self.features.iter().any(|f| f.as_iri() == feature_iri)
    }

    /// Returns `true` if the given result format MIME type is supported
    pub fn supports_result_mime(&self, mime: &str) -> bool {
        self.result_formats.iter().any(|f| f.mime_type() == mime)
    }

    /// Returns `true` if the given input format MIME type is accepted
    pub fn accepts_input_mime(&self, mime: &str) -> bool {
        self.input_formats.iter().any(|f| f.mime_type() == mime)
    }

    /// Returns `true` if SPARQL 1.1 Query is supported
    pub fn supports_sparql11(&self) -> bool {
        self.supports_language(SdLanguage::Sparql11Query.as_iri())
    }

    /// Returns `true` if SPARQL 1.2 Query is supported (OxiRS extension)
    pub fn supports_sparql12(&self) -> bool {
        self.supports_language(SdLanguage::Sparql12Query.as_iri())
    }

    /// Returns `true` if SPARQL Update is supported
    pub fn supports_update(&self) -> bool {
        self.supports_language(SdLanguage::Sparql11Update.as_iri())
            || self.supports_language(SdLanguage::Sparql12Update.as_iri())
    }

    /// Returns `true` if an entailment regime is supported
    pub fn supports_entailment(&self, regime: &EntailmentRegime) -> bool {
        self.entailment_regimes.contains(regime)
    }

    /// Merge another `ServiceDescription` into this one, deduplicating entries
    ///
    /// Useful when combining descriptions from multiple sub-services.
    pub fn merge(&mut self, other: ServiceDescription) {
        for feat in other.features {
            if !self.features.contains(&feat) {
                self.features.push(feat);
            }
        }
        for lang in other.supported_languages {
            if !self.supported_languages.contains(&lang) {
                self.supported_languages.push(lang);
            }
        }
        for fmt in other.result_formats {
            if !self.result_formats.contains(&fmt) {
                self.result_formats.push(fmt);
            }
        }
        for fmt in other.input_formats {
            if !self.input_formats.contains(&fmt) {
                self.input_formats.push(fmt);
            }
        }
        for iri in other.extension_functions {
            if !self.extension_functions.contains(&iri) {
                self.extension_functions.push(iri);
            }
        }
        for regime in other.entailment_regimes {
            if !self.entailment_regimes.contains(&regime) {
                self.entailment_regimes.push(regime);
            }
        }
    }
}

/// Escape a string for use inside Turtle string literals
fn escape_turtle_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Escape a string for use inside XML attribute values
fn escape_xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escape a string for use inside XML element text content
fn escape_xml_string(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_sd() -> ServiceDescription {
        ServiceDescription::default_for_oxirs("http://localhost:3030/sparql", "default")
    }

    // ---- Construction tests -----------------------------------------------

    #[test]
    fn test_default_oxirs_endpoint_url() {
        let sd = default_sd();
        assert_eq!(sd.endpoint_url, "http://localhost:3030/sparql");
    }

    #[test]
    fn test_default_oxirs_dataset_name() {
        let sd = default_sd();
        assert_eq!(sd.dataset_name, "default");
    }

    #[test]
    fn test_default_oxirs_has_label() {
        let sd = default_sd();
        assert!(sd.label.is_some());
        assert!(sd.label.as_ref().unwrap().contains("OxiRS"));
    }

    #[test]
    fn test_default_oxirs_has_description() {
        let sd = default_sd();
        assert!(sd.description.is_some());
        assert!(sd.description.as_ref().unwrap().contains("Rust"));
    }

    #[test]
    fn test_default_has_sparql11_query() {
        let sd = default_sd();
        assert!(sd.supports_sparql11());
    }

    #[test]
    fn test_default_has_sparql12_query() {
        let sd = default_sd();
        assert!(sd.supports_sparql12());
    }

    #[test]
    fn test_default_has_sparql11_update() {
        let sd = default_sd();
        assert!(sd.supports_update());
    }

    #[test]
    fn test_default_has_sparql10_query() {
        let sd = default_sd();
        assert!(sd.supports_language(SdLanguage::Sparql10Query.as_iri()));
    }

    #[test]
    fn test_default_feature_basic_federated_query() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::BasicFederatedQuery.as_iri()));
    }

    #[test]
    fn test_default_feature_union_default_graph() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::UnionDefaultGraph.as_iri()));
    }

    #[test]
    fn test_default_feature_dereferences_uris() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::DereferencesUris.as_iri()));
    }

    #[test]
    fn test_default_feature_empty_graphs() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::EmptyGraphs.as_iri()));
    }

    #[test]
    fn test_default_feature_rdf_star() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::RdfStar.as_iri()));
    }

    #[test]
    fn test_default_feature_sparql_star() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::SparqlStar.as_iri()));
    }

    #[test]
    fn test_default_result_format_json() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/sparql-results+json"));
    }

    #[test]
    fn test_default_result_format_xml() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/sparql-results+xml"));
    }

    #[test]
    fn test_default_result_format_csv() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("text/csv"));
    }

    #[test]
    fn test_default_result_format_tsv() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("text/tab-separated-values"));
    }

    #[test]
    fn test_default_result_format_turtle() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("text/turtle"));
    }

    #[test]
    fn test_default_result_format_ntriples() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/n-triples"));
    }

    #[test]
    fn test_default_result_format_nquads() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/n-quads"));
    }

    #[test]
    fn test_default_result_format_trig() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/trig"));
    }

    #[test]
    fn test_default_result_format_jsonld() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/ld+json"));
    }

    #[test]
    fn test_default_result_format_rdfxml() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/rdf+xml"));
    }

    #[test]
    fn test_default_input_format_turtle() {
        let sd = default_sd();
        assert!(sd.accepts_input_mime("text/turtle"));
    }

    #[test]
    fn test_default_input_format_jsonld() {
        let sd = default_sd();
        assert!(sd.accepts_input_mime("application/ld+json"));
    }

    #[test]
    fn test_default_entailment_simple() {
        let sd = default_sd();
        assert!(sd.supports_entailment(&EntailmentRegime::Simple));
    }

    #[test]
    fn test_default_entailment_rdf() {
        let sd = default_sd();
        assert!(sd.supports_entailment(&EntailmentRegime::Rdf));
    }

    #[test]
    fn test_default_entailment_rdfs() {
        let sd = default_sd();
        assert!(sd.supports_entailment(&EntailmentRegime::Rdfs));
    }

    #[test]
    fn test_result_format_count_minimum() {
        let sd = default_sd();
        assert!(sd.result_format_count() >= 10);
    }

    #[test]
    fn test_input_format_count_minimum() {
        let sd = default_sd();
        assert!(sd.input_format_count() >= 6);
    }

    // ---- Turtle output tests -----------------------------------------------

    #[test]
    fn test_turtle_starts_with_prefix() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("@prefix sd:"));
    }

    #[test]
    fn test_turtle_contains_rdf_prefix() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("@prefix rdf:"));
    }

    #[test]
    fn test_turtle_contains_rdfs_prefix() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("@prefix rdfs:"));
    }

    #[test]
    fn test_turtle_contains_service_type() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("a sd:Service"));
    }

    #[test]
    fn test_turtle_contains_endpoint_url() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("http://localhost:3030/sparql"));
    }

    #[test]
    fn test_turtle_contains_sd_endpoint() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:endpoint"));
    }

    #[test]
    fn test_turtle_contains_sparql11_query_language() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:supportedLanguage"));
        assert!(turtle.contains("SPARQL11Query"));
    }

    #[test]
    fn test_turtle_contains_sparql12_query_language() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("SPARQL12Query"));
    }

    #[test]
    fn test_turtle_contains_result_format() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:resultFormat"));
    }

    #[test]
    fn test_turtle_contains_input_format() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:inputFormat"));
    }

    #[test]
    fn test_turtle_contains_feature() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:feature"));
    }

    #[test]
    fn test_turtle_contains_entailment_regime() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:defaultEntailmentRegime"));
    }

    #[test]
    fn test_turtle_contains_dataset_block() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:defaultDataset"));
        assert!(turtle.contains("sd:Dataset"));
    }

    #[test]
    fn test_turtle_contains_default_graph() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:defaultGraph"));
        assert!(turtle.contains("sd:Graph"));
    }

    #[test]
    fn test_turtle_contains_label() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("rdfs:label"));
        assert!(turtle.contains("OxiRS"));
    }

    #[test]
    fn test_turtle_contains_comment() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("rdfs:comment"));
    }

    #[test]
    fn test_turtle_dataset_name_embedded() {
        let sd =
            ServiceDescription::default_for_oxirs("http://example.org/sparql", "my-knowledge-base");
        let turtle = sd.to_turtle();
        assert!(turtle.contains("my-knowledge-base"));
    }

    #[test]
    fn test_turtle_nonempty() {
        let turtle = default_sd().to_turtle();
        assert!(!turtle.is_empty());
        assert!(turtle.len() > 200);
    }

    #[test]
    fn test_turtle_extension_functions_included() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "test")
            .with_language(SdLanguage::Sparql11Query)
            .with_extension_function("http://example.org/functions/myFunc")
            .build();
        let turtle = sd.to_turtle();
        assert!(turtle.contains("sd:extensionFunction"));
        assert!(turtle.contains("myFunc"));
    }

    // ---- JSON-LD output tests -----------------------------------------------

    #[test]
    fn test_jsonld_is_object() {
        let val = default_sd().to_json_ld();
        assert!(val.is_object());
    }

    #[test]
    fn test_jsonld_has_context() {
        let val = default_sd().to_json_ld();
        assert!(val.get("@context").is_some());
    }

    #[test]
    fn test_jsonld_context_has_sd() {
        let val = default_sd().to_json_ld();
        let ctx = val["@context"].as_object().expect("context is object");
        assert!(ctx.contains_key("sd"));
    }

    #[test]
    fn test_jsonld_context_has_rdfs() {
        let val = default_sd().to_json_ld();
        let ctx = val["@context"].as_object().expect("context is object");
        assert!(ctx.contains_key("rdfs"));
    }

    #[test]
    fn test_jsonld_has_graph() {
        let val = default_sd().to_json_ld();
        assert!(val.get("@graph").is_some());
    }

    #[test]
    fn test_jsonld_graph_nonempty() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_jsonld_service_has_id() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service = &graph[0];
        assert!(service.get("@id").is_some());
    }

    #[test]
    fn test_jsonld_service_id_is_endpoint_url() {
        let sd = ServiceDescription::default_for_oxirs("http://localhost:3030/sparql", "default");
        let val = sd.to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let id = graph[0]["@id"].as_str().expect("id is string");
        assert_eq!(id, "http://localhost:3030/sparql");
    }

    #[test]
    fn test_jsonld_service_has_type() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service_type = graph[0]["@type"].as_str().expect("@type is string");
        assert!(service_type.contains("Service"));
    }

    #[test]
    fn test_jsonld_has_supported_language() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service = &graph[0];
        let lang_key = "http://www.w3.org/ns/sparql-service-description#supportedLanguage";
        assert!(service.get(lang_key).is_some());
    }

    #[test]
    fn test_jsonld_has_result_format() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service = &graph[0];
        let fmt_key = "http://www.w3.org/ns/sparql-service-description#resultFormat";
        assert!(service.get(fmt_key).is_some());
    }

    #[test]
    fn test_jsonld_has_feature() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service = &graph[0];
        let feat_key = "http://www.w3.org/ns/sparql-service-description#feature";
        assert!(service.get(feat_key).is_some());
    }

    // ---- RDF/XML output tests -----------------------------------------------

    #[test]
    fn test_rdfxml_starts_with_xml_declaration() {
        let xml = default_sd().to_rdf_xml();
        assert!(xml.starts_with("<?xml"));
    }

    #[test]
    fn test_rdfxml_contains_rdf_rdf() {
        let xml = default_sd().to_rdf_xml();
        assert!(xml.contains("<rdf:RDF"));
    }

    #[test]
    fn test_rdfxml_contains_service_element() {
        let xml = default_sd().to_rdf_xml();
        assert!(xml.contains("<sd:Service"));
    }

    #[test]
    fn test_rdfxml_contains_endpoint_url() {
        let xml = default_sd().to_rdf_xml();
        assert!(xml.contains("http://localhost:3030/sparql"));
    }

    // ---- Builder pattern tests ----------------------------------------------

    #[test]
    fn test_builder_minimal() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds").build();
        assert_eq!(sd.endpoint_url, "http://example.org/sparql");
        assert_eq!(sd.dataset_name, "ds");
        assert!(sd.features.is_empty());
    }

    #[test]
    fn test_builder_with_single_feature() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_feature(SdFeature::EmptyGraphs)
            .build();
        assert_eq!(sd.features.len(), 1);
        assert!(sd.has_feature(SdFeature::EmptyGraphs.as_iri()));
    }

    #[test]
    fn test_builder_with_multiple_features() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_features([SdFeature::EmptyGraphs, SdFeature::BasicFederatedQuery])
            .build();
        assert_eq!(sd.features.len(), 2);
    }

    #[test]
    fn test_builder_with_extension_function() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_extension_function("http://example.org/fn/custom")
            .build();
        assert_eq!(sd.extension_functions.len(), 1);
        assert_eq!(sd.extension_functions[0], "http://example.org/fn/custom");
    }

    #[test]
    fn test_builder_with_entailment_owl_rl() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_entailment_regime(EntailmentRegime::Owl2Rl)
            .build();
        assert!(sd.supports_entailment(&EntailmentRegime::Owl2Rl));
    }

    #[test]
    fn test_builder_label_and_description() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_label("My Endpoint")
            .with_description("A test endpoint")
            .build();
        assert_eq!(sd.label.as_deref(), Some("My Endpoint"));
        assert_eq!(sd.description.as_deref(), Some("A test endpoint"));
    }

    // ---- Merge tests --------------------------------------------------------

    #[test]
    fn test_merge_adds_missing_features() {
        let mut sd1 = ServiceDescription::builder("http://a.org/sparql", "a")
            .with_feature(SdFeature::EmptyGraphs)
            .build();
        let sd2 = ServiceDescription::builder("http://b.org/sparql", "b")
            .with_feature(SdFeature::BasicFederatedQuery)
            .build();
        sd1.merge(sd2);
        assert_eq!(sd1.features.len(), 2);
    }

    #[test]
    fn test_merge_no_duplicate_features() {
        let mut sd1 = ServiceDescription::builder("http://a.org/sparql", "a")
            .with_feature(SdFeature::EmptyGraphs)
            .build();
        let sd2 = ServiceDescription::builder("http://b.org/sparql", "b")
            .with_feature(SdFeature::EmptyGraphs)
            .build();
        sd1.merge(sd2);
        assert_eq!(sd1.features.len(), 1);
    }

    #[test]
    fn test_merge_adds_missing_languages() {
        let mut sd1 = ServiceDescription::builder("http://a.org/sparql", "a")
            .with_language(SdLanguage::Sparql11Query)
            .build();
        let sd2 = ServiceDescription::builder("http://b.org/sparql", "b")
            .with_language(SdLanguage::Sparql11Update)
            .build();
        sd1.merge(sd2);
        assert_eq!(sd1.supported_languages.len(), 2);
    }

    #[test]
    fn test_merge_adds_extension_functions() {
        let mut sd1 = ServiceDescription::builder("http://a.org/sparql", "a")
            .with_extension_function("http://example.org/fn/f1")
            .build();
        let sd2 = ServiceDescription::builder("http://b.org/sparql", "b")
            .with_extension_function("http://example.org/fn/f2")
            .build();
        sd1.merge(sd2);
        assert_eq!(sd1.extension_functions.len(), 2);
    }

    // ---- IRI / MIME type tests ----------------------------------------------

    #[test]
    fn test_feature_iris_nonempty() {
        let features = [
            SdFeature::DereferencesUris,
            SdFeature::UnionDefaultGraph,
            SdFeature::RequiresDataset,
            SdFeature::EmptyGraphs,
            SdFeature::BasicFederatedQuery,
            SdFeature::RdfStar,
            SdFeature::SparqlStar,
            SdFeature::ConstraintValidation,
            SdFeature::TimeoutHints,
        ];
        for feat in features {
            assert!(
                !feat.as_iri().is_empty(),
                "Feature IRI empty for {:?}",
                feat
            );
        }
    }

    #[test]
    fn test_language_iris_nonempty() {
        let langs = [
            SdLanguage::Sparql10Query,
            SdLanguage::Sparql11Query,
            SdLanguage::Sparql11Update,
            SdLanguage::Sparql12Query,
            SdLanguage::Sparql12Update,
        ];
        for lang in langs {
            assert!(
                !lang.as_iri().is_empty(),
                "Language IRI empty for {:?}",
                lang
            );
        }
    }

    #[test]
    fn test_result_format_mime_nonempty() {
        let fmts = [
            SdResultFormat::SparqlResultsJson,
            SdResultFormat::SparqlResultsXml,
            SdResultFormat::SparqlResultsCsv,
            SdResultFormat::SparqlResultsTsv,
            SdResultFormat::Turtle,
            SdResultFormat::NTriples,
            SdResultFormat::NQuads,
            SdResultFormat::TriG,
            SdResultFormat::JsonLd,
            SdResultFormat::RdfXml,
            SdResultFormat::N3,
            SdResultFormat::LdPatch,
        ];
        for fmt in fmts {
            assert!(!fmt.mime_type().is_empty(), "MIME type empty for {:?}", fmt);
        }
    }

    #[test]
    fn test_input_format_mime_nonempty() {
        let fmts = [
            SdInputFormat::Turtle,
            SdInputFormat::NTriples,
            SdInputFormat::NQuads,
            SdInputFormat::TriG,
            SdInputFormat::JsonLd,
            SdInputFormat::RdfXml,
            SdInputFormat::N3,
        ];
        for fmt in fmts {
            assert!(!fmt.mime_type().is_empty(), "MIME type empty for {:?}", fmt);
        }
    }

    #[test]
    fn test_entailment_regime_iris_nonempty() {
        let regimes = [
            EntailmentRegime::Simple,
            EntailmentRegime::Rdf,
            EntailmentRegime::Rdfs,
            EntailmentRegime::Owl2Direct,
            EntailmentRegime::Owl2Rl,
            EntailmentRegime::Owl2El,
            EntailmentRegime::Owl2Ql,
            EntailmentRegime::DEntailment,
        ];
        for regime in regimes {
            assert!(
                !regime.as_iri().is_empty(),
                "Entailment regime IRI empty for {:?}",
                regime
            );
        }
    }

    #[test]
    fn test_escape_turtle_string_quotes() {
        let result = escape_turtle_string("say \"hello\"");
        assert!(result.contains("\\\""));
        assert!(!result.contains("say \"hello\""));
    }

    #[test]
    fn test_escape_turtle_string_newlines() {
        let result = escape_turtle_string("line1\nline2");
        assert!(result.contains("\\n"));
    }

    #[test]
    fn test_sd_ns_constant() {
        assert_eq!(SD_NS, "http://www.w3.org/ns/sparql-service-description#");
    }

    #[test]
    fn test_rdf_ns_constant() {
        assert_eq!(RDF_NS, "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
    }
}
