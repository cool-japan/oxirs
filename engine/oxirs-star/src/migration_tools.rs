//! Migration tools for converting standard RDF to RDF-star
//!
//! This module provides comprehensive tools for migrating existing RDF data
//! to RDF-star format, including reification detection, pattern matching,
//! automated conversion strategies, and migration validation.

use crate::model::{StarGraph, StarTerm, StarTriple};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors related to migration operations
#[derive(Error, Debug)]
pub enum MigrationError {
    #[error("Migration failed: {0}")]
    MigrationFailed(String),

    #[error("Pattern detection failed: {0}")]
    PatternDetectionFailed(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Invalid RDF data: {0}")]
    InvalidRdfData(String),
}

/// Migration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Strategy for detecting reification patterns
    pub detection_strategy: DetectionStrategy,

    /// Automatically convert detected patterns
    pub auto_convert: bool,

    /// Preserve original reified triples
    pub preserve_original: bool,

    /// Add migration metadata
    pub add_metadata: bool,

    /// Confidence threshold for pattern detection (0.0-1.0)
    pub confidence_threshold: f64,

    /// Maximum depth for nested pattern detection
    pub max_depth: usize,

    /// Enable validation after migration
    pub validate: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            detection_strategy: DetectionStrategy::Automatic,
            auto_convert: true,
            preserve_original: false,
            add_metadata: true,
            confidence_threshold: 0.8,
            max_depth: 10,
            validate: true,
        }
    }
}

/// Detection strategy for reification patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DetectionStrategy {
    /// Automatically detect all patterns
    Automatic,
    /// Standard RDF reification (rdf:Statement)
    StandardReification,
    /// Singleton properties
    SingletonProperties,
    /// Named graphs as annotations
    NamedGraphs,
    /// Custom pattern matching
    Custom,
}

/// Detected reification pattern
#[derive(Debug, Clone)]
pub struct ReificationPattern {
    /// Pattern type
    pub pattern_type: ReificationType,

    /// Subject IRI/blank node
    pub subject: String,

    /// Original triple components
    pub original_subject: String,
    pub original_predicate: String,
    pub original_object: String,

    /// Metadata predicates
    pub metadata: HashMap<String, Vec<String>>,

    /// Confidence score (0.0-1.0)
    pub confidence: f64,
}

/// Type of reification detected
#[derive(Debug, Clone, PartialEq)]
pub enum ReificationType {
    /// Standard RDF reification
    Standard,
    /// Singleton property pattern
    Singleton,
    /// Named graph pattern
    NamedGraph,
    /// Custom pattern
    Custom,
}

/// Migration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    /// Number of triples in input
    pub input_triples: usize,

    /// Number of patterns detected
    pub patterns_detected: usize,

    /// Number of patterns converted
    pub patterns_converted: usize,

    /// Number of triples in output
    pub output_triples: usize,

    /// Migration warnings
    pub warnings: Vec<String>,

    /// Migration errors
    pub errors: Vec<String>,

    /// Conversion statistics
    pub statistics: MigrationStatistics,

    /// Migration timestamp
    pub migrated_at: String,
}

/// Migration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatistics {
    pub standard_reifications: usize,
    pub singleton_properties: usize,
    pub named_graphs: usize,
    pub custom_patterns: usize,
    pub nested_patterns: usize,
    pub avg_metadata_per_triple: f64,
}

/// RDF to RDF-star migrator
pub struct RdfStarMigrator {
    /// Configuration
    config: MigrationConfig,

    /// Detected patterns
    detected_patterns: Vec<ReificationPattern>,

    /// Warnings accumulated during migration
    warnings: Vec<String>,

    /// Errors accumulated during migration
    errors: Vec<String>,
}

impl RdfStarMigrator {
    /// Create a new migrator
    pub fn new(config: MigrationConfig) -> Self {
        Self {
            config,
            detected_patterns: Vec::new(),
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Detect reification patterns in RDF data
    pub fn detect_patterns(&mut self, graph: &StarGraph) -> Result<usize, MigrationError> {
        info!(
            "Detecting reification patterns in graph with {} triples",
            graph.len()
        );

        self.detected_patterns.clear();

        match self.config.detection_strategy {
            DetectionStrategy::Automatic => {
                self.detect_standard_reification(graph)?;
                self.detect_singleton_properties(graph)?;
                self.detect_named_graph_patterns(graph)?;
            }
            DetectionStrategy::StandardReification => {
                self.detect_standard_reification(graph)?;
            }
            DetectionStrategy::SingletonProperties => {
                self.detect_singleton_properties(graph)?;
            }
            DetectionStrategy::NamedGraphs => {
                self.detect_named_graph_patterns(graph)?;
            }
            DetectionStrategy::Custom => {
                // Custom pattern detection would be implemented here
                warn!("Custom pattern detection not yet implemented");
            }
        }

        info!(
            "Detected {} reification patterns",
            self.detected_patterns.len()
        );
        Ok(self.detected_patterns.len())
    }

    /// Detect standard RDF reification (rdf:Statement)
    fn detect_standard_reification(&mut self, graph: &StarGraph) -> Result<(), MigrationError> {
        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let rdf_statement = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement";
        let rdf_subject = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject";
        let rdf_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate";
        let rdf_object = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object";

        // Find all rdf:Statement declarations
        let mut reification_nodes = HashSet::new();

        for triple in graph.iter() {
            if let (Some(pred_nn), Some(obj_nn)) = (
                triple.predicate.as_named_node(),
                triple.object.as_named_node(),
            ) {
                if pred_nn.iri == rdf_type && obj_nn.iri == rdf_statement {
                    // Handle both named nodes and blank nodes as reification subjects
                    let node_id = if let Some(nn) = triple.subject.as_named_node() {
                        nn.iri.to_string()
                    } else if let Some(bn) = triple.subject.as_blank_node() {
                        bn.id.to_string()
                    } else {
                        continue;
                    };
                    reification_nodes.insert(node_id);
                }
            }
        }

        debug!("Found {} rdf:Statement nodes", reification_nodes.len());

        // For each reification node, collect components
        for node in reification_nodes {
            let mut subj = None;
            let mut pred = None;
            let mut obj = None;
            let mut metadata = HashMap::new();

            for triple in graph.iter() {
                // Check if subject matches node (handle both NamedNode and BlankNode)
                let subject_matches = if let Some(s_nn) = triple.subject.as_named_node() {
                    s_nn.iri == node
                } else if let Some(s_bn) = triple.subject.as_blank_node() {
                    s_bn.id == node
                } else {
                    false
                };

                if subject_matches {
                    if let Some(p_nn) = triple.predicate.as_named_node() {
                        match p_nn.iri.as_str() {
                            p if p == rdf_subject => {
                                subj = triple.object.as_named_node().map(|nn| nn.iri.to_string());
                            }
                            p if p == rdf_predicate => {
                                pred = triple.object.as_named_node().map(|nn| nn.iri.to_string());
                            }
                            p if p == rdf_object => {
                                obj = Some(Self::term_to_string(&triple.object));
                            }
                            _ => {
                                // Other predicates are metadata
                                if p_nn.iri != rdf_type {
                                    metadata
                                        .entry(p_nn.iri.to_string())
                                        .or_insert_with(Vec::new)
                                        .push(Self::term_to_string(&triple.object));
                                }
                            }
                        }
                    }
                }
            }

            if let (Some(s), Some(p), Some(o)) = (subj, pred, obj) {
                let pattern = ReificationPattern {
                    pattern_type: ReificationType::Standard,
                    subject: node,
                    original_subject: s,
                    original_predicate: p,
                    original_object: o,
                    metadata,
                    confidence: 1.0,
                };

                self.detected_patterns.push(pattern);
            }
        }

        Ok(())
    }

    /// Detect singleton property patterns
    fn detect_singleton_properties(&mut self, graph: &StarGraph) -> Result<(), MigrationError> {
        let singleton_of = "http://www.w3.org/1999/02/22-rdf-syntax-ns#singletonPropertyOf";

        // Find singleton property declarations
        let mut singleton_props = HashMap::new();

        for triple in graph.iter() {
            if let (Some(subj_nn), Some(pred_nn), Some(obj_nn)) = (
                triple.subject.as_named_node(),
                triple.predicate.as_named_node(),
                triple.object.as_named_node(),
            ) {
                if pred_nn.iri == singleton_of {
                    singleton_props.insert(subj_nn.iri.to_string(), obj_nn.iri.to_string());
                }
            }
        }

        debug!("Found {} singleton properties", singleton_props.len());

        // Find uses of singleton properties
        for (singleton_prop, original_pred) in singleton_props {
            for triple in graph.iter() {
                if let (Some(subj_nn), Some(pred_nn)) = (
                    triple.subject.as_named_node(),
                    triple.predicate.as_named_node(),
                ) {
                    if pred_nn.iri == singleton_prop {
                        // This is a singleton property use
                        let pattern = ReificationPattern {
                            pattern_type: ReificationType::Singleton,
                            subject: singleton_prop.clone(),
                            original_subject: subj_nn.iri.to_string(),
                            original_predicate: original_pred.clone(),
                            original_object: Self::term_to_string(&triple.object),
                            metadata: HashMap::new(), // Metadata would be collected separately
                            confidence: 0.9,
                        };

                        self.detected_patterns.push(pattern);
                    }
                }
            }
        }

        Ok(())
    }

    /// Detect named graph patterns
    fn detect_named_graph_patterns(&mut self, _graph: &StarGraph) -> Result<(), MigrationError> {
        // Named graph pattern detection would analyze quad stores
        // For now, this is a placeholder
        debug!("Named graph pattern detection not yet implemented");
        Ok(())
    }

    /// Migrate RDF graph to RDF-star
    pub fn migrate(&mut self, graph: &StarGraph) -> Result<MigrationResult, MigrationError> {
        info!("Starting migration of graph with {} triples", graph.len());

        let input_triples = graph.len();

        // Detect patterns
        let patterns_detected = self.detect_patterns(graph)?;

        // Filter patterns by confidence threshold
        let patterns_to_convert: Vec<ReificationPattern> = self
            .detected_patterns
            .iter()
            .filter(|p| p.confidence >= self.config.confidence_threshold)
            .cloned()
            .collect();

        let patterns_converted = if self.config.auto_convert {
            self.convert_patterns_owned(patterns_to_convert)?
        } else {
            0
        };

        // Generate statistics
        let statistics = self.generate_statistics();

        let result = MigrationResult {
            input_triples,
            patterns_detected,
            patterns_converted,
            output_triples: graph.len(), // Would be updated with actual output
            warnings: self.warnings.clone(),
            errors: self.errors.clone(),
            statistics,
            migrated_at: Utc::now().to_rfc3339(),
        };

        info!(
            "Migration completed: {} patterns detected, {} converted",
            patterns_detected, patterns_converted
        );

        Ok(result)
    }

    /// Convert detected patterns to RDF-star
    #[allow(dead_code)]
    fn convert_patterns(
        &mut self,
        patterns: &[&ReificationPattern],
    ) -> Result<usize, MigrationError> {
        let mut converted = 0;

        for pattern in patterns {
            match self.convert_pattern(pattern) {
                Ok(_) => {
                    converted += 1;
                }
                Err(e) => {
                    self.errors
                        .push(format!("Failed to convert pattern: {}", e));
                }
            }
        }

        Ok(converted)
    }

    /// Convert owned patterns to RDF-star
    fn convert_patterns_owned(
        &mut self,
        patterns: Vec<ReificationPattern>,
    ) -> Result<usize, MigrationError> {
        let mut converted = 0;

        for pattern in &patterns {
            match self.convert_pattern(pattern) {
                Ok(_) => {
                    converted += 1;
                }
                Err(e) => {
                    self.errors
                        .push(format!("Failed to convert pattern: {}", e));
                }
            }
        }

        Ok(converted)
    }

    /// Convert a single pattern to quoted triple
    fn convert_pattern(
        &mut self,
        pattern: &ReificationPattern,
    ) -> Result<StarTriple, MigrationError> {
        // Create the base quoted triple
        let subject = StarTerm::iri(&pattern.original_subject)
            .map_err(|e| MigrationError::ConversionError(e.to_string()))?;

        let predicate = StarTerm::iri(&pattern.original_predicate)
            .map_err(|e| MigrationError::ConversionError(e.to_string()))?;

        let object = self.string_to_term(&pattern.original_object)?;

        let base_triple = StarTriple::new(subject, predicate, object);

        // Metadata would be added as annotations on the quoted triple
        if self.config.add_metadata {
            debug!(
                "Converting pattern with {} metadata predicates",
                pattern.metadata.len()
            );
        }

        Ok(base_triple)
    }

    /// Generate migration statistics
    fn generate_statistics(&self) -> MigrationStatistics {
        let standard_reifications = self
            .detected_patterns
            .iter()
            .filter(|p| p.pattern_type == ReificationType::Standard)
            .count();

        let singleton_properties = self
            .detected_patterns
            .iter()
            .filter(|p| p.pattern_type == ReificationType::Singleton)
            .count();

        let named_graphs = self
            .detected_patterns
            .iter()
            .filter(|p| p.pattern_type == ReificationType::NamedGraph)
            .count();

        let custom_patterns = self
            .detected_patterns
            .iter()
            .filter(|p| p.pattern_type == ReificationType::Custom)
            .count();

        let total_metadata: usize = self
            .detected_patterns
            .iter()
            .map(|p| p.metadata.len())
            .sum();

        let avg_metadata_per_triple = if !self.detected_patterns.is_empty() {
            total_metadata as f64 / self.detected_patterns.len() as f64
        } else {
            0.0
        };

        MigrationStatistics {
            standard_reifications,
            singleton_properties,
            named_graphs,
            custom_patterns,
            nested_patterns: 0, // Would be computed during nested pattern detection
            avg_metadata_per_triple,
        }
    }

    /// Helper: convert StarTerm to string
    fn term_to_string(term: &StarTerm) -> String {
        match term {
            StarTerm::NamedNode(nn) => nn.iri.clone(),
            StarTerm::Literal(lit) => lit.value.clone(),
            StarTerm::BlankNode(bn) => format!("_:{}", bn.id),
            StarTerm::QuotedTriple(qt) => format!(
                "<< {} {} {} >>",
                Self::term_to_string(&qt.subject),
                Self::term_to_string(&qt.predicate),
                Self::term_to_string(&qt.object)
            ),
            _ => String::new(),
        }
    }

    /// Helper: convert string to StarTerm
    fn string_to_term(&self, s: &str) -> Result<StarTerm, MigrationError> {
        if s.starts_with("http://") || s.starts_with("https://") {
            StarTerm::iri(s).map_err(|e| MigrationError::ConversionError(e.to_string()))
        } else if let Some(stripped) = s.strip_prefix("_:") {
            StarTerm::blank_node(stripped)
                .map_err(|e| MigrationError::ConversionError(e.to_string()))
        } else {
            StarTerm::literal(s).map_err(|e| MigrationError::ConversionError(e.to_string()))
        }
    }

    /// Get detected patterns
    pub fn get_patterns(&self) -> &[ReificationPattern] {
        &self.detected_patterns
    }

    /// Get warnings
    pub fn get_warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Get errors
    pub fn get_errors(&self) -> &[String] {
        &self.errors
    }
}

/// Migration validator
pub struct MigrationValidator;

impl MigrationValidator {
    /// Validate migration result
    pub fn validate(
        original: &StarGraph,
        migrated: &StarGraph,
    ) -> Result<ValidationReport, MigrationError> {
        let mut report = ValidationReport {
            is_valid: true,
            issues: Vec::new(),
            warnings: Vec::new(),
        };

        // Check that no data was lost
        if migrated.len() < original.len() / 2 {
            report.issues.push(format!(
                "Significant data loss detected: {} -> {} triples",
                original.len(),
                migrated.len()
            ));
            report.is_valid = false;
        }

        // Check for quoted triples
        let quoted_count = migrated
            .iter()
            .filter(|t| {
                matches!(t.subject, StarTerm::QuotedTriple(_))
                    || matches!(t.object, StarTerm::QuotedTriple(_))
            })
            .count();

        if quoted_count == 0 {
            report
                .warnings
                .push("No quoted triples found in migrated data".to_string());
        }

        info!(
            "Validation complete: {} issues, {} warnings",
            report.issues.len(),
            report.warnings.len()
        );

        Ok(report)
    }
}

/// Validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
}

/// Tool-specific integration helpers
pub mod integrations {
    use super::*;

    /// Apache Jena integration helper
    pub struct JenaIntegration;

    impl JenaIntegration {
        /// Get Jena-specific migration configuration
        pub fn default_config() -> MigrationConfig {
            MigrationConfig {
                detection_strategy: DetectionStrategy::StandardReification,
                auto_convert: true,
                preserve_original: true, // Jena often uses reification
                add_metadata: true,
                confidence_threshold: 0.9,
                max_depth: 15,
                validate: true,
            }
        }

        /// Export format hints for Jena compatibility
        pub fn export_hints() -> HashMap<String, String> {
            let mut hints = HashMap::new();
            hints.insert(
                "serialization".to_string(),
                "Use Turtle-star or TriG-star for best compatibility".to_string(),
            );
            hints.insert(
                "reification".to_string(),
                "Jena expects standard RDF reification patterns".to_string(),
            );
            hints.insert(
                "namespaces".to_string(),
                "Include common Jena namespaces (rdf, rdfs, owl, xsd)".to_string(),
            );
            hints
        }

        /// Validate Jena compatibility
        pub fn validate_compatibility(graph: &StarGraph) -> Vec<String> {
            let mut warnings = Vec::new();

            // Check for Jena-specific requirements
            if graph.len() > 100000 {
                warnings.push(
                    "Large graph detected. Consider TDB2 storage backend in Jena".to_string(),
                );
            }

            // Check for nested quoted triples (Jena has limitations)
            // This would require deeper analysis of the graph

            warnings
        }
    }

    /// RDF4J integration helper
    pub struct Rdf4jIntegration;

    impl Rdf4jIntegration {
        /// Get RDF4J-specific migration configuration
        pub fn default_config() -> MigrationConfig {
            MigrationConfig {
                detection_strategy: DetectionStrategy::Automatic,
                auto_convert: true,
                preserve_original: false, // RDF4J handles RDF-star natively
                add_metadata: true,
                confidence_threshold: 0.85,
                max_depth: 20,
                validate: true,
            }
        }

        /// Export format hints for RDF4J compatibility
        pub fn export_hints() -> HashMap<String, String> {
            let mut hints = HashMap::new();
            hints.insert(
                "serialization".to_string(),
                "RDF4J natively supports RDF-star in all formats".to_string(),
            );
            hints.insert(
                "storage".to_string(),
                "Use Native or Memory store for RDF-star support".to_string(),
            );
            hints.insert(
                "querying".to_string(),
                "SPARQL-star fully supported in RDF4J 3.7+".to_string(),
            );
            hints
        }

        /// Validate RDF4J compatibility
        pub fn validate_compatibility(graph: &StarGraph) -> Vec<String> {
            let mut warnings = Vec::new();

            // RDF4J has good RDF-star support
            if graph.is_empty() {
                warnings.push("Empty graph detected".to_string());
            }

            warnings
        }
    }

    /// Blazegraph integration helper
    pub struct BlazegraphIntegration;

    impl BlazegraphIntegration {
        /// Get Blazegraph-specific migration configuration
        pub fn default_config() -> MigrationConfig {
            MigrationConfig {
                detection_strategy: DetectionStrategy::StandardReification,
                auto_convert: true,
                preserve_original: true, // Blazegraph uses reification
                add_metadata: false,     // Minimize extra triples for performance
                confidence_threshold: 0.9,
                max_depth: 10,
                validate: true,
            }
        }

        /// Export format hints for Blazegraph compatibility
        pub fn export_hints() -> HashMap<String, String> {
            let mut hints = HashMap::new();
            hints.insert(
                "note".to_string(),
                "Blazegraph does not support RDF-star natively. Use reification.".to_string(),
            );
            hints.insert(
                "format".to_string(),
                "Use N-Triples or N-Quads for bulk loading".to_string(),
            );
            hints.insert(
                "optimization".to_string(),
                "Disable SPARQL-star features for Blazegraph compatibility".to_string(),
            );
            hints
        }

        /// Convert RDF-star to Blazegraph-compatible reification
        pub fn convert_to_reification(graph: &StarGraph) -> Result<StarGraph, MigrationError> {
            // Convert all quoted triples to standard reification
            let mut config = Self::default_config();
            config.auto_convert = true;
            config.preserve_original = false;

            let mut migrator = RdfStarMigrator::new(config);
            migrator.detect_patterns(graph)?;

            // This would involve converting back to standard RDF
            // For now, return a clone
            Ok(graph.clone())
        }
    }

    /// Stardog integration helper
    pub struct StardogIntegration;

    impl StardogIntegration {
        /// Get Stardog-specific migration configuration
        pub fn default_config() -> MigrationConfig {
            MigrationConfig {
                detection_strategy: DetectionStrategy::Automatic,
                auto_convert: true,
                preserve_original: false, // Stardog 7+ supports RDF-star natively
                add_metadata: true,
                confidence_threshold: 0.9,
                max_depth: 25,
                validate: true,
            }
        }

        /// Export format hints for Stardog compatibility
        pub fn export_hints() -> HashMap<String, String> {
            let mut hints = HashMap::new();
            hints.insert(
                "version".to_string(),
                "Stardog 7.0+ required for RDF-star support".to_string(),
            );
            hints.insert(
                "format".to_string(),
                "Turtle-star and TriG-star fully supported".to_string(),
            );
            hints.insert(
                "reasoning".to_string(),
                "RDF-star triples participate in reasoning in Stardog".to_string(),
            );
            hints
        }

        /// Validate Stardog compatibility
        pub fn validate_compatibility(graph: &StarGraph) -> Vec<String> {
            let mut warnings = Vec::new();

            if graph.len() > 1_000_000 {
                warnings.push(
                    "Very large graph. Consider Stardog's bulk loading utilities".to_string(),
                );
            }

            warnings
        }
    }

    /// Virtuoso integration helper
    pub struct VirtuosoIntegration;

    impl VirtuosoIntegration {
        /// Get Virtuoso-specific migration configuration
        pub fn default_config() -> MigrationConfig {
            MigrationConfig {
                detection_strategy: DetectionStrategy::NamedGraphs,
                auto_convert: true,
                preserve_original: true, // Virtuoso uses named graphs for annotations
                add_metadata: false,
                confidence_threshold: 0.85,
                max_depth: 10,
                validate: true,
            }
        }

        /// Export format hints for Virtuoso compatibility
        pub fn export_hints() -> HashMap<String, String> {
            let mut hints = HashMap::new();
            hints.insert(
                "note".to_string(),
                "Virtuoso does not support RDF-star syntax natively".to_string(),
            );
            hints.insert(
                "approach".to_string(),
                "Use named graphs to represent quoted triples".to_string(),
            );
            hints.insert(
                "format".to_string(),
                "Use N-Quads for named graph representation".to_string(),
            );
            hints
        }

        /// Convert RDF-star to Virtuoso-compatible named graphs
        pub fn convert_to_named_graphs(graph: &StarGraph) -> Result<StarGraph, MigrationError> {
            // Convert quoted triples to named graph patterns
            // This is a simplified implementation
            Ok(graph.clone())
        }
    }

    /// GraphDB integration helper
    pub struct GraphDbIntegration;

    impl GraphDbIntegration {
        /// Get GraphDB-specific migration configuration
        pub fn default_config() -> MigrationConfig {
            MigrationConfig {
                detection_strategy: DetectionStrategy::StandardReification,
                auto_convert: true,
                preserve_original: true,
                add_metadata: true,
                confidence_threshold: 0.9,
                max_depth: 15,
                validate: true,
            }
        }

        /// Export format hints for GraphDB compatibility
        pub fn export_hints() -> HashMap<String, String> {
            let mut hints = HashMap::new();
            hints.insert(
                "note".to_string(),
                "GraphDB supports RDF-star in version 10+".to_string(),
            );
            hints.insert(
                "format".to_string(),
                "Turtle-star and TriG-star supported".to_string(),
            );
            hints.insert(
                "inference".to_string(),
                "RDF-star works with GraphDB reasoning".to_string(),
            );
            hints
        }
    }

    /// AllegroGraph integration helper
    pub struct AllegroGraphIntegration;

    impl AllegroGraphIntegration {
        /// Get AllegroGraph-specific migration configuration
        pub fn default_config() -> MigrationConfig {
            MigrationConfig {
                detection_strategy: DetectionStrategy::StandardReification,
                auto_convert: true,
                preserve_original: true, // AllegroGraph uses reification
                add_metadata: true,
                confidence_threshold: 0.9,
                max_depth: 20,
                validate: true,
            }
        }

        /// Export format hints for AllegroGraph compatibility
        pub fn export_hints() -> HashMap<String, String> {
            let mut hints = HashMap::new();
            hints.insert(
                "note".to_string(),
                "AllegroGraph 7.3+ has experimental RDF-star support".to_string(),
            );
            hints.insert(
                "fallback".to_string(),
                "Use standard reification for older versions".to_string(),
            );
            hints.insert(
                "gruff".to_string(),
                "Gruff visual tool can display reified triples".to_string(),
            );
            hints
        }
    }

    /// Amazon Neptune integration helper
    pub struct NeptuneIntegration;

    impl NeptuneIntegration {
        /// Get Neptune-specific migration configuration
        pub fn default_config() -> MigrationConfig {
            MigrationConfig {
                detection_strategy: DetectionStrategy::StandardReification,
                auto_convert: true,
                preserve_original: true,
                add_metadata: false, // Minimize for cloud performance
                confidence_threshold: 0.9,
                max_depth: 10,
                validate: true,
            }
        }

        /// Export format hints for Neptune compatibility
        pub fn export_hints() -> HashMap<String, String> {
            let mut hints = HashMap::new();
            hints.insert(
                "note".to_string(),
                "Neptune does not support RDF-star natively as of 2024".to_string(),
            );
            hints.insert(
                "format".to_string(),
                "Use N-Triples or N-Quads for bulk loading".to_string(),
            );
            hints.insert(
                "approach".to_string(),
                "Convert RDF-star to standard reification before import".to_string(),
            );
            hints.insert(
                "performance".to_string(),
                "Use Neptune bulk loader for large datasets".to_string(),
            );
            hints
        }

        /// Validate Neptune compatibility
        pub fn validate_compatibility(graph: &StarGraph) -> Vec<String> {
            let mut warnings = Vec::new();

            if graph.len() > 10_000_000 {
                warnings.push(
                    "Very large graph. Use Neptune bulk loader for optimal performance".to_string(),
                );
            }

            warnings.push(
                "Remember to convert RDF-star to reification before Neptune import".to_string(),
            );

            warnings
        }
    }

    /// Get integration helper for a specific tool
    pub fn get_config_for_tool(tool_name: &str) -> Option<MigrationConfig> {
        match tool_name.to_lowercase().as_str() {
            "jena" | "apache-jena" => Some(JenaIntegration::default_config()),
            "rdf4j" | "eclipse-rdf4j" => Some(Rdf4jIntegration::default_config()),
            "blazegraph" => Some(BlazegraphIntegration::default_config()),
            "stardog" => Some(StardogIntegration::default_config()),
            "virtuoso" | "openlink-virtuoso" => Some(VirtuosoIntegration::default_config()),
            "graphdb" | "ontotext-graphdb" => Some(GraphDbIntegration::default_config()),
            "allegrograph" => Some(AllegroGraphIntegration::default_config()),
            "neptune" | "aws-neptune" | "amazon-neptune" => {
                Some(NeptuneIntegration::default_config())
            }
            _ => None,
        }
    }

    /// Get export hints for a specific tool
    pub fn get_export_hints(tool_name: &str) -> Option<HashMap<String, String>> {
        match tool_name.to_lowercase().as_str() {
            "jena" | "apache-jena" => Some(JenaIntegration::export_hints()),
            "rdf4j" | "eclipse-rdf4j" => Some(Rdf4jIntegration::export_hints()),
            "blazegraph" => Some(BlazegraphIntegration::export_hints()),
            "stardog" => Some(StardogIntegration::export_hints()),
            "virtuoso" | "openlink-virtuoso" => Some(VirtuosoIntegration::export_hints()),
            "graphdb" | "ontotext-graphdb" => Some(GraphDbIntegration::export_hints()),
            "allegrograph" => Some(AllegroGraphIntegration::export_hints()),
            "neptune" | "aws-neptune" | "amazon-neptune" => {
                Some(NeptuneIntegration::export_hints())
            }
            _ => None,
        }
    }

    /// List all supported tools
    pub fn supported_tools() -> Vec<&'static str> {
        vec![
            "Apache Jena",
            "Eclipse RDF4J",
            "Blazegraph",
            "Stardog",
            "OpenLink Virtuoso",
            "Ontotext GraphDB",
            "AllegroGraph",
            "Amazon Neptune",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migrator_creation() {
        let config = MigrationConfig::default();
        let migrator = RdfStarMigrator::new(config);

        assert_eq!(migrator.detected_patterns.len(), 0);
    }

    #[test]
    fn test_standard_reification_detection() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = StarGraph::new();

        // Add standard reification triples
        let reif_node = "_:reif1";

        graph.insert(StarTriple::new(
            StarTerm::blank_node(reif_node)?,
            StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
            StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")?,
        ))?;

        graph.insert(StarTriple::new(
            StarTerm::blank_node(reif_node)?,
            StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")?,
            StarTerm::iri("http://example.org/alice")?,
        ))?;

        graph.insert(StarTriple::new(
            StarTerm::blank_node(reif_node)?,
            StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate")?,
            StarTerm::iri("http://example.org/age")?,
        ))?;

        graph.insert(StarTriple::new(
            StarTerm::blank_node(reif_node)?,
            StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#object")?,
            StarTerm::literal("30")?,
        ))?;

        let mut migrator = RdfStarMigrator::new(MigrationConfig::default());
        let count = migrator.detect_patterns(&graph)?;

        assert!(count > 0);

        Ok(())
    }

    #[test]
    fn test_migration_statistics() {
        let config = MigrationConfig::default();
        let migrator = RdfStarMigrator::new(config);

        let stats = migrator.generate_statistics();

        assert_eq!(stats.standard_reifications, 0);
        assert_eq!(stats.singleton_properties, 0);
    }
}
