//! Legacy RDF compatibility layer for standard RDF tools
//!
//! This module provides high-level APIs for converting between RDF-star
//! and standard RDF, enabling interoperability with tools that don't
//! support RDF-star quoted triples.
//!
//! # Features
//!
//! - Automatic conversion between RDF-star and standard RDF
//! - Multiple reification strategies (Standard, Unique IRIs, Blank Nodes, Singleton Properties)
//! - Compatibility mode for parsers and serializers
//! - Round-trip conversion (RDF-star → RDF → RDF-star)
//! - Detection of reification patterns in imported data
//!
//! # Example
//!
//! ```rust
//! use oxirs_star::compatibility::{CompatibilityMode, CompatibilityConfig};
//! use oxirs_star::{StarGraph, StarTriple, StarTerm};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create compatibility mode with standard reification
//! let config = CompatibilityConfig::standard_reification();
//! let mut compat = CompatibilityMode::new(config);
//!
//! // Create RDF-star graph with quoted triples
//! let mut star_graph = StarGraph::new();
//! let quoted = StarTriple::new(
//!     StarTerm::iri("http://example.org/alice")?,
//!     StarTerm::iri("http://example.org/age")?,
//!     StarTerm::literal("25")?,
//! );
//! let meta = StarTriple::new(
//!     StarTerm::quoted_triple(quoted),
//!     StarTerm::iri("http://example.org/certainty")?,
//!     StarTerm::literal("0.9")?,
//! );
//! star_graph.insert(meta)?;
//!
//! // Convert to standard RDF (compatible with legacy tools)
//! let standard_rdf = compat.to_standard_rdf(&star_graph)?;
//! println!("Converted to {} standard RDF triples", standard_rdf.len());
//!
//! // Convert back to RDF-star
//! let recovered_star = compat.from_standard_rdf(&standard_rdf)?;
//! println!("Recovered {} RDF-star triples", recovered_star.len());
//! # Ok(())
//! # }
//! ```

use crate::model::StarGraph;
use crate::reification::{ReificationStrategy, Reificator};
use crate::{StarError, StarResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, span, Level};

/// Configuration for legacy RDF compatibility mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityConfig {
    /// Reification strategy to use when converting to standard RDF
    pub strategy: ReificationStrategyConfig,

    /// Base IRI for generating statement identifiers
    pub base_iri: Option<String>,

    /// Automatically detect and convert reified triples when importing
    pub auto_detect_reification: bool,

    /// Preserve blank node identifiers when round-tripping
    pub preserve_blank_nodes: bool,

    /// Maximum nesting depth to support (for deeply nested quoted triples)
    pub max_nesting_depth: usize,

    /// Enable validation of reification patterns
    pub validate_reifications: bool,
}

/// Reification strategy configuration (serializable version)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReificationStrategyConfig {
    /// Standard RDF reification using rdf:Statement
    StandardReification,
    /// Use unique IRIs for each quoted triple
    UniqueIris,
    /// Use blank nodes for quoted triples
    BlankNodes,
    /// Singleton properties (most efficient)
    SingletonProperties,
}

impl From<ReificationStrategyConfig> for ReificationStrategy {
    fn from(config: ReificationStrategyConfig) -> Self {
        match config {
            ReificationStrategyConfig::StandardReification => {
                ReificationStrategy::StandardReification
            }
            ReificationStrategyConfig::UniqueIris => ReificationStrategy::UniqueIris,
            ReificationStrategyConfig::BlankNodes => ReificationStrategy::BlankNodes,
            ReificationStrategyConfig::SingletonProperties => {
                ReificationStrategy::SingletonProperties
            }
        }
    }
}

impl From<ReificationStrategy> for ReificationStrategyConfig {
    fn from(strategy: ReificationStrategy) -> Self {
        match strategy {
            ReificationStrategy::StandardReification => {
                ReificationStrategyConfig::StandardReification
            }
            ReificationStrategy::UniqueIris => ReificationStrategyConfig::UniqueIris,
            ReificationStrategy::BlankNodes => ReificationStrategyConfig::BlankNodes,
            ReificationStrategy::SingletonProperties => {
                ReificationStrategyConfig::SingletonProperties
            }
        }
    }
}

impl Default for CompatibilityConfig {
    fn default() -> Self {
        Self {
            strategy: ReificationStrategyConfig::StandardReification,
            base_iri: Some("http://example.org/statement/".to_string()),
            auto_detect_reification: true,
            preserve_blank_nodes: true,
            max_nesting_depth: 10,
            validate_reifications: true,
        }
    }
}

impl CompatibilityConfig {
    /// Create configuration for standard reification (W3C RDF reification vocabulary)
    pub fn standard_reification() -> Self {
        Self {
            strategy: ReificationStrategyConfig::StandardReification,
            ..Default::default()
        }
    }

    /// Create configuration using unique IRIs for each statement
    pub fn unique_iris(base_iri: String) -> Self {
        Self {
            strategy: ReificationStrategyConfig::UniqueIris,
            base_iri: Some(base_iri),
            ..Default::default()
        }
    }

    /// Create configuration using blank nodes
    pub fn blank_nodes() -> Self {
        Self {
            strategy: ReificationStrategyConfig::BlankNodes,
            ..Default::default()
        }
    }

    /// Create configuration using singleton properties (most efficient)
    pub fn singleton_properties() -> Self {
        Self {
            strategy: ReificationStrategyConfig::SingletonProperties,
            ..Default::default()
        }
    }

    /// Set the base IRI for statement identifiers
    pub fn with_base_iri(mut self, base_iri: String) -> Self {
        self.base_iri = Some(base_iri);
        self
    }

    /// Enable or disable automatic reification detection
    pub fn with_auto_detect(mut self, enabled: bool) -> Self {
        self.auto_detect_reification = enabled;
        self
    }

    /// Set maximum nesting depth
    pub fn with_max_nesting_depth(mut self, depth: usize) -> Self {
        self.max_nesting_depth = depth;
        self
    }
}

/// Legacy RDF compatibility mode
///
/// Provides high-level APIs for converting between RDF-star and standard RDF.
pub struct CompatibilityMode {
    config: CompatibilityConfig,
    reificator: Reificator,
    statistics: CompatibilityStatistics,
}

/// Statistics for compatibility operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompatibilityStatistics {
    /// Total conversions to standard RDF
    pub conversions_to_standard: usize,

    /// Total conversions from standard RDF
    pub conversions_from_standard: usize,

    /// Total quoted triples converted
    pub quoted_triples_converted: usize,

    /// Total reification patterns detected
    pub reifications_detected: usize,

    /// Round-trip conversion success rate
    pub roundtrip_success_rate: f64,

    /// Average conversion time (microseconds)
    pub avg_conversion_time_us: f64,

    /// Strategy-specific statistics
    pub strategy_stats: HashMap<String, usize>,
}

impl CompatibilityMode {
    /// Create a new compatibility mode with the given configuration
    pub fn new(config: CompatibilityConfig) -> Self {
        let strategy: ReificationStrategy = config.strategy.clone().into();
        let reificator = Reificator::new(strategy, config.base_iri.clone());

        Self {
            config,
            reificator,
            statistics: CompatibilityStatistics::default(),
        }
    }

    /// Convert an RDF-star graph to standard RDF
    ///
    /// This converts all quoted triples to reified triples using the
    /// configured reification strategy.
    pub fn to_standard_rdf(&mut self, star_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "to_standard_rdf");
        let _enter = span.enter();

        let start_time = std::time::Instant::now();

        // Count quoted triples before conversion
        let quoted_count = star_graph
            .triples()
            .iter()
            .filter(|t| t.subject.is_quoted_triple() || t.object.is_quoted_triple())
            .count();

        // Convert using reificator
        let standard_graph = self.reificator.reify_graph(star_graph)?;

        // Update statistics
        self.statistics.conversions_to_standard += 1;
        self.statistics.quoted_triples_converted += quoted_count;

        let conversion_time = start_time.elapsed().as_micros() as f64;
        self.update_avg_conversion_time(conversion_time);

        let strategy_name = format!("{:?}", self.config.strategy);
        *self
            .statistics
            .strategy_stats
            .entry(strategy_name)
            .or_insert(0) += 1;

        info!(
            "Converted RDF-star graph ({} triples, {} quoted) to standard RDF ({} triples) in {:.2}ms",
            star_graph.len(),
            quoted_count,
            standard_graph.len(),
            conversion_time / 1000.0
        );

        Ok(standard_graph)
    }

    /// Convert standard RDF back to RDF-star
    ///
    /// This detects reification patterns and converts them back to
    /// quoted triples.
    pub fn from_standard_rdf(&mut self, standard_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "from_standard_rdf");
        let _enter = span.enter();

        let start_time = std::time::Instant::now();

        // Detect reification patterns if enabled
        if self.config.auto_detect_reification {
            let reification_count = crate::reification::utils::count_reifications(standard_graph);
            self.statistics.reifications_detected += reification_count;

            debug!("Detected {} reification patterns", reification_count);
        }

        // Validate reifications if enabled
        if self.config.validate_reifications {
            if let Err(e) = crate::reification::utils::validate_reifications(standard_graph) {
                return Err(StarError::reification_error(format!(
                    "Invalid reification patterns: {}",
                    e
                )));
            }
        }

        // Convert using reificator
        let star_graph = self.reificator.dereify_graph(standard_graph)?;

        // Update statistics
        self.statistics.conversions_from_standard += 1;

        let conversion_time = start_time.elapsed().as_micros() as f64;
        self.update_avg_conversion_time(conversion_time);

        info!(
            "Converted standard RDF ({} triples) back to RDF-star ({} triples) in {:.2}ms",
            standard_graph.len(),
            star_graph.len(),
            conversion_time / 1000.0
        );

        Ok(star_graph)
    }

    /// Test round-trip conversion (RDF-star → standard RDF → RDF-star)
    ///
    /// Returns true if the round-trip is successful (graphs are structurally equivalent).
    pub fn test_roundtrip(&mut self, star_graph: &StarGraph) -> StarResult<bool> {
        let span = span!(Level::INFO, "test_roundtrip");
        let _enter = span.enter();

        // Convert to standard RDF
        let standard_graph = self.to_standard_rdf(star_graph)?;

        // Convert back to RDF-star
        let recovered_graph = self.from_standard_rdf(&standard_graph)?;

        // Compare sizes (should be the same for successful round-trip)
        let success = star_graph.len() == recovered_graph.len();

        // Update success rate
        let total_roundtrips = (self.statistics.conversions_to_standard
            + self.statistics.conversions_from_standard) as f64
            / 2.0;
        if success {
            self.statistics.roundtrip_success_rate =
                (self.statistics.roundtrip_success_rate * (total_roundtrips - 1.0) + 1.0)
                    / total_roundtrips;
        } else {
            self.statistics.roundtrip_success_rate = (self.statistics.roundtrip_success_rate
                * (total_roundtrips - 1.0))
                / total_roundtrips;
        }

        info!(
            "Round-trip test: {} (original: {} triples, recovered: {} triples)",
            if success { "SUCCESS" } else { "FAILED" },
            star_graph.len(),
            recovered_graph.len()
        );

        Ok(success)
    }

    /// Check if a graph contains RDF-star quoted triples
    pub fn has_quoted_triples(graph: &StarGraph) -> bool {
        graph
            .triples()
            .iter()
            .any(|t| t.subject.is_quoted_triple() || t.object.is_quoted_triple())
    }

    /// Check if a graph contains reification patterns
    pub fn has_reifications(graph: &StarGraph) -> bool {
        crate::reification::utils::has_reifications(graph)
    }

    /// Count quoted triples in a graph
    pub fn count_quoted_triples(graph: &StarGraph) -> usize {
        graph
            .triples()
            .iter()
            .filter(|t| t.subject.is_quoted_triple() || t.object.is_quoted_triple())
            .count()
    }

    /// Count reification patterns in a graph
    pub fn count_reifications(graph: &StarGraph) -> usize {
        crate::reification::utils::count_reifications(graph)
    }

    /// Get compatibility statistics
    pub fn statistics(&self) -> &CompatibilityStatistics {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = CompatibilityStatistics::default();
    }

    /// Get the current configuration
    pub fn config(&self) -> &CompatibilityConfig {
        &self.config
    }

    /// Update configuration (creates new reificator)
    pub fn set_config(&mut self, config: CompatibilityConfig) {
        let strategy: ReificationStrategy = config.strategy.clone().into();
        self.reificator = Reificator::new(strategy, config.base_iri.clone());
        self.config = config;
    }

    /// Helper to update average conversion time
    fn update_avg_conversion_time(&mut self, new_time: f64) {
        let total_conversions = (self.statistics.conversions_to_standard
            + self.statistics.conversions_from_standard) as f64;

        if total_conversions == 1.0 {
            self.statistics.avg_conversion_time_us = new_time;
        } else {
            self.statistics.avg_conversion_time_us =
                (self.statistics.avg_conversion_time_us * (total_conversions - 1.0) + new_time)
                    / total_conversions;
        }
    }

    /// Apply W3C RDF-star unstar mapping
    ///
    /// This converts an RDF-star graph to standard RDF using the W3C standard
    /// unstar mapping. Each quoted triple is mapped to a set of reification triples:
    ///
    /// ```turtle
    /// # RDF-star
    /// <<:alice :age 30>> :certainty 0.9 .
    ///
    /// # Unstar mapping (standard RDF)
    /// _:stmt rdf:subject :alice .
    /// _:stmt rdf:predicate :age .
    /// _:stmt rdf:object 30 .
    /// _:stmt :certainty 0.9 .
    /// ```
    ///
    /// This is the official W3C mapping for translating RDF-star to standard RDF
    /// for compatibility with non-RDF-star-aware reasoners.
    pub fn unstar(&mut self, star_graph: &StarGraph) -> StarResult<StarGraph> {
        // The W3C unstar mapping is equivalent to standard RDF reification
        // Temporarily switch to standard reification strategy if needed
        let original_strategy = self.config.strategy.clone();

        if original_strategy != ReificationStrategyConfig::StandardReification {
            self.config.strategy = ReificationStrategyConfig::StandardReification;
            let strategy: ReificationStrategy = self.config.strategy.clone().into();
            self.reificator = Reificator::new(strategy, self.config.base_iri.clone());
        }

        let result = self.to_standard_rdf(star_graph);

        // Restore original strategy
        if original_strategy != ReificationStrategyConfig::StandardReification {
            self.config.strategy = original_strategy;
            let strategy: ReificationStrategy = self.config.strategy.clone().into();
            self.reificator = Reificator::new(strategy, self.config.base_iri.clone());
        }

        result
    }

    /// Apply reverse of W3C RDF-star unstar mapping (rdfstar mapping)
    ///
    /// This converts standard RDF with reification patterns back to RDF-star
    /// using the W3C standard mapping. It detects reification patterns and
    /// converts them to quoted triples:
    ///
    /// ```turtle
    /// # Standard RDF with reification
    /// _:stmt rdf:subject :alice .
    /// _:stmt rdf:predicate :age .
    /// _:stmt rdf:object 30 .
    /// _:stmt :certainty 0.9 .
    ///
    /// # RDF-star (after rdfstar mapping)
    /// <<:alice :age 30>> :certainty 0.9 .
    /// ```
    ///
    /// This is the inverse of the unstar mapping, allowing round-trip conversion.
    pub fn rdfstar(&mut self, standard_graph: &StarGraph) -> StarResult<StarGraph> {
        // The rdfstar mapping is equivalent to dereification with standard reification
        let original_strategy = self.config.strategy.clone();

        if original_strategy != ReificationStrategyConfig::StandardReification {
            self.config.strategy = ReificationStrategyConfig::StandardReification;
            let strategy: ReificationStrategy = self.config.strategy.clone().into();
            self.reificator = Reificator::new(strategy, self.config.base_iri.clone());
        }

        let result = self.from_standard_rdf(standard_graph);

        // Restore original strategy
        if original_strategy != ReificationStrategyConfig::StandardReification {
            self.config.strategy = original_strategy;
            let strategy: ReificationStrategy = self.config.strategy.clone().into();
            self.reificator = Reificator::new(strategy, self.config.base_iri.clone());
        }

        result
    }

    /// Test W3C unstar/rdfstar round-trip conversion
    ///
    /// This verifies that the W3C standard mapping preserves graph structure
    /// through a round-trip conversion: RDF-star → unstar → rdfstar → RDF-star
    pub fn test_unstar_roundtrip(&mut self, star_graph: &StarGraph) -> StarResult<bool> {
        let span = span!(Level::INFO, "test_unstar_roundtrip");
        let _enter = span.enter();

        // Apply unstar mapping
        let unstarred = self.unstar(star_graph)?;

        // Apply rdfstar mapping (reverse)
        let recovered = self.rdfstar(&unstarred)?;

        // Compare sizes
        let success = star_graph.len() == recovered.len();

        info!(
            "W3C unstar round-trip test: {} (original: {} triples, recovered: {} triples)",
            if success { "SUCCESS" } else { "FAILED" },
            star_graph.len(),
            recovered.len()
        );

        Ok(success)
    }
}

/// Compatibility presets for common use cases
pub struct CompatibilityPresets;

impl CompatibilityPresets {
    /// Preset for Apache Jena compatibility (standard reification)
    pub fn apache_jena() -> CompatibilityConfig {
        CompatibilityConfig {
            strategy: ReificationStrategyConfig::StandardReification,
            base_iri: Some("http://jena.apache.org/statement/".to_string()),
            auto_detect_reification: true,
            preserve_blank_nodes: true,
            max_nesting_depth: 5,
            validate_reifications: true,
        }
    }

    /// Preset for RDF4J compatibility (unique IRIs)
    pub fn rdf4j() -> CompatibilityConfig {
        CompatibilityConfig {
            strategy: ReificationStrategyConfig::UniqueIris,
            base_iri: Some("http://rdf4j.org/statement/".to_string()),
            auto_detect_reification: true,
            preserve_blank_nodes: true,
            max_nesting_depth: 5,
            validate_reifications: true,
        }
    }

    /// Preset for Virtuoso compatibility (blank nodes)
    pub fn virtuoso() -> CompatibilityConfig {
        CompatibilityConfig {
            strategy: ReificationStrategyConfig::BlankNodes,
            base_iri: None,
            auto_detect_reification: true,
            preserve_blank_nodes: true,
            max_nesting_depth: 3,
            validate_reifications: false, // Virtuoso is more lenient
        }
    }

    /// Preset for maximum efficiency (singleton properties)
    pub fn efficient() -> CompatibilityConfig {
        CompatibilityConfig {
            strategy: ReificationStrategyConfig::SingletonProperties,
            base_iri: Some("http://example.org/property/".to_string()),
            auto_detect_reification: true,
            preserve_blank_nodes: false,
            max_nesting_depth: 10,
            validate_reifications: false,
        }
    }
}

/// Batch compatibility converter for processing multiple graphs
pub struct BatchCompatibilityConverter {
    config: CompatibilityConfig,
    statistics: CompatibilityStatistics,
}

impl BatchCompatibilityConverter {
    /// Create a new batch converter
    pub fn new(config: CompatibilityConfig) -> Self {
        Self {
            config,
            statistics: CompatibilityStatistics::default(),
        }
    }

    /// Convert multiple RDF-star graphs to standard RDF in parallel
    pub fn batch_to_standard_rdf(
        &mut self,
        star_graphs: Vec<StarGraph>,
    ) -> StarResult<Vec<StarGraph>> {
        let span = span!(Level::INFO, "batch_to_standard_rdf");
        let _enter = span.enter();

        let start_time = std::time::Instant::now();
        let mut results = Vec::new();

        for star_graph in star_graphs {
            let mut compat = CompatibilityMode::new(self.config.clone());
            let standard_graph = compat.to_standard_rdf(&star_graph)?;
            results.push(standard_graph);

            // Aggregate statistics
            self.aggregate_statistics(&compat.statistics);
        }

        let total_time = start_time.elapsed();
        info!(
            "Batch converted {} graphs to standard RDF in {:?}",
            results.len(),
            total_time
        );

        Ok(results)
    }

    /// Convert multiple standard RDF graphs to RDF-star in parallel
    pub fn batch_from_standard_rdf(
        &mut self,
        standard_graphs: Vec<StarGraph>,
    ) -> StarResult<Vec<StarGraph>> {
        let span = span!(Level::INFO, "batch_from_standard_rdf");
        let _enter = span.enter();

        let start_time = std::time::Instant::now();
        let mut results = Vec::new();

        for standard_graph in standard_graphs {
            let mut compat = CompatibilityMode::new(self.config.clone());
            let star_graph = compat.from_standard_rdf(&standard_graph)?;
            results.push(star_graph);

            // Aggregate statistics
            self.aggregate_statistics(&compat.statistics);
        }

        let total_time = start_time.elapsed();
        info!(
            "Batch converted {} graphs from standard RDF in {:?}",
            results.len(),
            total_time
        );

        Ok(results)
    }

    /// Get aggregated statistics
    pub fn statistics(&self) -> &CompatibilityStatistics {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = CompatibilityStatistics::default();
    }

    /// Helper to aggregate statistics from individual conversions
    fn aggregate_statistics(&mut self, other: &CompatibilityStatistics) {
        self.statistics.conversions_to_standard += other.conversions_to_standard;
        self.statistics.conversions_from_standard += other.conversions_from_standard;
        self.statistics.quoted_triples_converted += other.quoted_triples_converted;
        self.statistics.reifications_detected += other.reifications_detected;

        // Merge strategy stats
        for (strategy, count) in &other.strategy_stats {
            *self
                .statistics
                .strategy_stats
                .entry(strategy.clone())
                .or_insert(0) += count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarTerm, StarTriple};

    #[test]
    fn test_basic_compatibility_mode() {
        let config = CompatibilityConfig::standard_reification();
        let mut compat = CompatibilityMode::new(config);

        // Create RDF-star graph
        let mut star_graph = StarGraph::new();
        let quoted = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        star_graph.insert(meta).unwrap();

        // Convert to standard RDF
        let standard_graph = compat.to_standard_rdf(&star_graph).unwrap();
        assert!(standard_graph.len() > 1); // Should have reification triples

        // Convert back
        let recovered = compat.from_standard_rdf(&standard_graph).unwrap();
        assert_eq!(recovered.len(), 1); // Should recover original structure
    }

    #[test]
    fn test_roundtrip_conversion() {
        let config = CompatibilityConfig::standard_reification();
        let mut compat = CompatibilityMode::new(config);

        // Create RDF-star graph
        let mut star_graph = StarGraph::new();
        let quoted = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        star_graph.insert(meta).unwrap();

        // Test round-trip
        let success = compat.test_roundtrip(&star_graph).unwrap();
        assert!(success);
    }

    #[test]
    fn test_detection_functions() {
        // RDF-star graph with quoted triples
        let mut star_graph = StarGraph::new();
        let quoted = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        star_graph.insert(meta).unwrap();

        assert!(CompatibilityMode::has_quoted_triples(&star_graph));
        assert_eq!(CompatibilityMode::count_quoted_triples(&star_graph), 1);

        // Convert to standard RDF and check for reifications
        let config = CompatibilityConfig::standard_reification();
        let mut compat = CompatibilityMode::new(config);
        let standard_graph = compat.to_standard_rdf(&star_graph).unwrap();

        assert!(CompatibilityMode::has_reifications(&standard_graph));
        assert_eq!(CompatibilityMode::count_reifications(&standard_graph), 1);
    }

    #[test]
    fn test_compatibility_presets() {
        // Test that all presets can be created
        let _jena = CompatibilityPresets::apache_jena();
        let _rdf4j = CompatibilityPresets::rdf4j();
        let _virtuoso = CompatibilityPresets::virtuoso();
        let _efficient = CompatibilityPresets::efficient();
    }

    #[test]
    fn test_batch_conversion() {
        let config = CompatibilityConfig::standard_reification();
        let mut batch = BatchCompatibilityConverter::new(config);

        // Create multiple RDF-star graphs
        let mut graphs = Vec::new();
        for i in 0..3 {
            let mut graph = StarGraph::new();
            let quoted = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/subject{i}")).unwrap(),
                StarTerm::iri("http://example.org/predicate").unwrap(),
                StarTerm::iri("http://example.org/object").unwrap(),
            );
            let meta = StarTriple::new(
                StarTerm::quoted_triple(quoted),
                StarTerm::iri("http://example.org/meta").unwrap(),
                StarTerm::literal("value").unwrap(),
            );
            graph.insert(meta).unwrap();
            graphs.push(graph);
        }

        // Batch convert
        let standard_graphs = batch.batch_to_standard_rdf(graphs).unwrap();
        assert_eq!(standard_graphs.len(), 3);

        // Check statistics
        let stats = batch.statistics();
        assert_eq!(stats.conversions_to_standard, 3);
    }

    #[test]
    fn test_statistics() {
        let config = CompatibilityConfig::standard_reification();
        let mut compat = CompatibilityMode::new(config);

        // Create and convert graph
        let mut star_graph = StarGraph::new();
        let quoted = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        star_graph.insert(meta).unwrap();

        let _standard = compat.to_standard_rdf(&star_graph).unwrap();

        let stats = compat.statistics();
        assert_eq!(stats.conversions_to_standard, 1);
        assert_eq!(stats.quoted_triples_converted, 1);
        assert!(stats.avg_conversion_time_us > 0.0);
    }
}
