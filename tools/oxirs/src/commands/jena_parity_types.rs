//! Core data types for the Jena parity verifier.
//!
//! Defines the taxonomy ([`ParityCategory`]), per-feature implementation status
//! ([`FeatureStatus`]), the individual feature entry ([`ParityFeature`]) and the
//! aggregate coverage statistics ([`ParitySummary`]).

/// The category a parity feature belongs to, mirroring Jena sub-project taxonomy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParityCategory {
    /// SPARQL query language features
    QueryLanguage,
    /// RDF serialization and deserialization formats
    DataFormats,
    /// OWL and RDFS entailment reasoning
    Reasoning,
    /// Authentication, DID, encryption, and security
    Security,
    /// Real-time and event stream processing
    Streaming,
    /// Machine learning and AI capabilities
    Ai,
    /// Persistence and storage backends
    Storage,
    /// Distributed query and cluster networking
    Networking,
    /// HTTP, GraphQL, and REST protocol endpoints
    Protocols,
    /// Spatial and geographic query support
    Geospatial,
    /// Shape and schema validation
    Validation,
    /// CLI tooling and developer experience
    Tooling,
    /// Industrial IoT protocol connectors
    Industrial,
}

impl ParityCategory {
    /// Returns a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            ParityCategory::QueryLanguage => "Query Language",
            ParityCategory::DataFormats => "Data Formats",
            ParityCategory::Reasoning => "Reasoning",
            ParityCategory::Security => "Security",
            ParityCategory::Streaming => "Streaming",
            ParityCategory::Ai => "AI / ML",
            ParityCategory::Storage => "Storage",
            ParityCategory::Networking => "Networking",
            ParityCategory::Protocols => "Protocols",
            ParityCategory::Geospatial => "Geospatial",
            ParityCategory::Validation => "Validation",
            ParityCategory::Tooling => "Tooling",
            ParityCategory::Industrial => "Industrial",
        }
    }
}

/// The implementation status of a feature in OxiRS relative to Jena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureStatus {
    /// Fully implemented and tested.
    Implemented,
    /// Partially implemented; percentage indicates completeness (0–100).
    PartiallyImplemented {
        /// Estimated percentage complete (0–100).
        percentage: u8,
    },
    /// Not yet implemented in OxiRS.
    NotImplemented,
    /// OxiRS-specific capability that Jena does not offer.
    BeyondJena,
}

impl FeatureStatus {
    /// Returns a short display label for the status.
    pub fn label(&self) -> &'static str {
        match self {
            FeatureStatus::Implemented => "Implemented",
            FeatureStatus::PartiallyImplemented { .. } => "Partial",
            FeatureStatus::NotImplemented => "Not Implemented",
            FeatureStatus::BeyondJena => "Beyond Jena",
        }
    }

    /// Returns a concise single-char indicator for compact tables.
    pub fn indicator(&self) -> &'static str {
        match self {
            FeatureStatus::Implemented => "[OK]",
            FeatureStatus::PartiallyImplemented { .. } => "[~]",
            FeatureStatus::NotImplemented => "[X]",
            FeatureStatus::BeyondJena => "[+]",
        }
    }

    /// Returns the effective completion percentage (0–100).
    ///
    /// - Implemented → 100
    /// - PartiallyImplemented → the stored percentage
    /// - NotImplemented → 0
    /// - BeyondJena → 100 (OxiRS has this; Jena does not)
    pub fn completion_percentage(&self) -> u8 {
        match self {
            FeatureStatus::Implemented => 100,
            FeatureStatus::PartiallyImplemented { percentage } => *percentage,
            FeatureStatus::NotImplemented => 0,
            FeatureStatus::BeyondJena => 100,
        }
    }

    /// Returns true if the feature is fully available (Implemented or BeyondJena).
    pub fn is_complete(&self) -> bool {
        matches!(self, FeatureStatus::Implemented | FeatureStatus::BeyondJena)
    }
}

/// A single feature entry in the Jena parity comparison.
#[derive(Debug, Clone)]
pub struct ParityFeature {
    /// Human-readable feature name.
    pub name: &'static str,
    /// Category for grouping in reports.
    pub category: ParityCategory,
    /// Whether Apache Jena supports this feature.
    pub jena_support: bool,
    /// Whether and how OxiRS supports this feature.
    pub oxirs_support: FeatureStatus,
    /// Optional notes on implementation gaps, extensions, or caveats.
    pub notes: Option<&'static str>,
}

impl ParityFeature {
    /// Constructs a feature that both Jena and OxiRS fully support.
    pub fn parity(
        name: &'static str,
        category: ParityCategory,
        notes: Option<&'static str>,
    ) -> Self {
        Self {
            name,
            category,
            jena_support: true,
            oxirs_support: FeatureStatus::Implemented,
            notes,
        }
    }

    /// Constructs a feature that Jena has but OxiRS only partially implements.
    pub fn partial(
        name: &'static str,
        category: ParityCategory,
        percentage: u8,
        notes: Option<&'static str>,
    ) -> Self {
        Self {
            name,
            category,
            jena_support: true,
            oxirs_support: FeatureStatus::PartiallyImplemented { percentage },
            notes,
        }
    }

    /// Constructs a feature that Jena has but OxiRS has not yet implemented.
    pub fn missing(
        name: &'static str,
        category: ParityCategory,
        notes: Option<&'static str>,
    ) -> Self {
        Self {
            name,
            category,
            jena_support: true,
            oxirs_support: FeatureStatus::NotImplemented,
            notes,
        }
    }

    /// Constructs a feature that OxiRS offers beyond what Jena provides.
    pub fn beyond_jena(
        name: &'static str,
        category: ParityCategory,
        notes: Option<&'static str>,
    ) -> Self {
        Self {
            name,
            category,
            jena_support: false,
            oxirs_support: FeatureStatus::BeyondJena,
            notes,
        }
    }

    /// Returns true if OxiRS achieves full parity or better for this feature.
    pub fn is_at_parity(&self) -> bool {
        matches!(
            self.oxirs_support,
            FeatureStatus::Implemented | FeatureStatus::BeyondJena
        )
    }

    /// Returns the OxiRS completion percentage for this feature.
    pub fn oxirs_completion(&self) -> u8 {
        self.oxirs_support.completion_percentage()
    }
}

/// Summary coverage statistics for the parity checker.
#[derive(Debug, Clone)]
pub struct ParitySummary {
    pub total_features: usize,
    pub implemented: usize,
    pub partial: usize,
    pub not_implemented: usize,
    pub beyond_jena: usize,
    pub jena_features_total: usize,
}

impl ParitySummary {
    /// Overall OxiRS coverage percentage of Jena features (0.0–100.0).
    ///
    /// Only `implemented` features (where `jena_support = true` AND `oxirs_support =
    /// Implemented`) count toward the numerator.  `beyond_jena` features have
    /// `jena_support = false` and are therefore not in the denominator, so including
    /// them in the numerator would push the ratio above 100 %.
    pub fn jena_parity_percentage(&self) -> f64 {
        if self.jena_features_total == 0 {
            return 100.0;
        }
        self.implemented as f64 / self.jena_features_total as f64 * 100.0
    }

    /// Weighted coverage accounting for partial implementations.
    pub fn weighted_coverage(&self, features: &[ParityFeature]) -> f64 {
        if features.is_empty() {
            return 0.0;
        }
        let total_points: u32 = features
            .iter()
            .filter(|f| f.jena_support)
            .map(|f| f.oxirs_completion() as u32)
            .sum();
        let max_points = self.jena_features_total as u32 * 100;
        if max_points == 0 {
            return 0.0;
        }
        total_points as f64 / max_points as f64 * 100.0
    }
}
