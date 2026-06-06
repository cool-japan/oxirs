//! Builder for constructing a [`ServiceDescription`] from a server's configuration.

use super::service_description_types::{
    EntailmentRegime, SdFeature, SdInputFormat, SdLanguage, SdResultFormat, ServiceDescription,
};

// ────────────────────────────────────────────────────────────────────────────
// Builder
// ────────────────────────────────────────────────────────────────────────────

/// Builder for constructing a [`ServiceDescription`]
#[derive(Debug, Default)]
pub struct ServiceDescriptionBuilder {
    pub(super) endpoint_url: String,
    pub(super) dataset_name: String,
    pub(super) features: Vec<SdFeature>,
    pub(super) supported_languages: Vec<SdLanguage>,
    pub(super) result_formats: Vec<SdResultFormat>,
    pub(super) input_formats: Vec<SdInputFormat>,
    pub(super) extension_functions: Vec<String>,
    pub(super) entailment_regimes: Vec<EntailmentRegime>,
    pub(super) label: Option<String>,
    pub(super) description: Option<String>,
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

// ────────────────────────────────────────────────────────────────────────────
// ServiceDescription constructor / merge helpers
// ────────────────────────────────────────────────────────────────────────────

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

    // ── query helpers ────────────────────────────────────────────────────

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
