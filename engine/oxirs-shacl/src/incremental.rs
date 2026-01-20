//! Incremental SHACL Validation
//!
//! Provides efficient incremental validation for RDF graphs that change over time.
//! Instead of re-validating the entire graph, only validates shapes affected by changes.
//!
//! Features:
//! - Delta-based validation (track additions/removals)
//! - Dependency analysis (which shapes are affected by which triples)
//! - Result caching for unchanged portions
//! - Support for batch updates
//! - Memory-efficient change tracking

use crate::{Result, ShaclError, ShapeId, ValidationReport, Validator};
use oxirs_core::{
    model::{Quad, Term},
    Store,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Represents a change to an RDF graph
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphChange {
    /// A triple was added
    Addition(Quad),
    /// A triple was removed
    Removal(Quad),
}

/// Changeset tracking additions and removals
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Changeset {
    /// Triples that were added
    pub additions: Vec<Quad>,
    /// Triples that were removed
    pub removals: Vec<Quad>,
    /// Timestamp when changeset was created
    pub timestamp: Option<u64>,
    /// Optional description
    pub description: Option<String>,
}

impl Changeset {
    /// Create a new empty changeset
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple addition
    pub fn add_addition(&mut self, quad: Quad) {
        self.additions.push(quad);
    }

    /// Add a triple removal
    pub fn add_removal(&mut self, quad: Quad) {
        self.removals.push(quad);
    }

    /// Get total number of changes
    pub fn change_count(&self) -> usize {
        self.additions.len() + self.removals.len()
    }

    /// Check if changeset is empty
    pub fn is_empty(&self) -> bool {
        self.additions.is_empty() && self.removals.is_empty()
    }

    /// Get all affected subjects
    pub fn affected_subjects(&self) -> HashSet<Term> {
        let mut subjects = HashSet::new();
        for quad in &self.additions {
            subjects.insert(quad.subject().clone().into());
        }
        for quad in &self.removals {
            subjects.insert(quad.subject().clone().into());
        }
        subjects
    }

    /// Get all affected predicates (as Terms)
    pub fn affected_predicates(&self) -> HashSet<Term> {
        let mut predicates = HashSet::new();
        for quad in &self.additions {
            predicates.insert(quad.predicate().clone().into());
        }
        for quad in &self.removals {
            predicates.insert(quad.predicate().clone().into());
        }
        predicates
    }
}

/// Configuration for incremental validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache TTL in seconds (None = infinite)
    pub cache_ttl_seconds: Option<u64>,
    /// Maximum cache size (number of validation results)
    pub max_cache_size: usize,
    /// Enable dependency analysis
    pub enable_dependency_analysis: bool,
    /// Batch size for processing changes
    pub batch_size: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl_seconds: Some(300), // 5 minutes
            max_cache_size: 10000,
            enable_dependency_analysis: true,
            batch_size: 1000,
        }
    }
}

/// Cached validation result
#[derive(Debug, Clone)]
struct CachedValidationResult {
    /// The validation report
    report: ValidationReport,
    /// When this result was cached
    cached_at: Instant,
}

/// Shape dependency information
#[derive(Debug, Clone)]
struct ShapeDependency {
    /// Predicates that affect this shape (as Terms)
    affected_by_predicates: HashSet<Term>,
    /// Target nodes for this shape
    target_nodes: HashSet<Term>,
}

/// Incremental validation engine
pub struct IncrementalValidator {
    /// Configuration
    config: IncrementalConfig,
    /// Underlying validator
    validator: Validator,
    /// Cached validation results
    cache: HashMap<String, CachedValidationResult>,
    /// Shape dependencies
    dependencies: HashMap<ShapeId, ShapeDependency>,
    /// Statistics
    stats: IncrementalStats,
}

impl IncrementalValidator {
    /// Create a new incremental validator
    pub fn new(validator: Validator, config: IncrementalConfig) -> Self {
        Self {
            config,
            validator,
            cache: HashMap::new(),
            dependencies: HashMap::new(),
            stats: IncrementalStats::default(),
        }
    }

    /// Validate a changeset incrementally
    pub fn validate_changeset(
        &mut self,
        changeset: &Changeset,
        store: &dyn Store,
    ) -> Result<ValidationReport> {
        let start = Instant::now();
        self.stats.total_changesets += 1;
        self.stats.total_changes += changeset.change_count();

        if changeset.is_empty() {
            self.stats.empty_changesets += 1;
            return Ok(ValidationReport::new());
        }

        // Analyze which shapes are affected
        let affected_shapes = if self.config.enable_dependency_analysis {
            self.analyze_affected_shapes(changeset)?
        } else {
            // Conservative: validate all shapes
            self.validator
                .shapes()
                .iter()
                .map(|(id, _)| (*id).clone())
                .collect()
        };

        self.stats.affected_shapes += affected_shapes.len();

        // Get affected focus nodes
        let affected_nodes = changeset.affected_subjects();
        self.stats.affected_nodes += affected_nodes.len();

        // Validate affected shapes and nodes
        let mut combined_report = ValidationReport::new();

        for shape_id in &affected_shapes {
            let _shape = self
                .validator
                .shapes()
                .iter()
                .find(|(id, _)| *id == shape_id)
                .map(|(_, shape)| shape.clone())
                .ok_or_else(|| {
                    ShaclError::ShapeValidation(format!("Shape not found: {}", shape_id))
                })?;

            for focus_node in &affected_nodes {
                // Check cache first
                let cache_key = self.make_cache_key(shape_id, focus_node);

                // Check cache and handle result
                let cached_report = if self.config.enable_caching {
                    self.get_cached_result(&cache_key).map(|c| c.report.clone())
                } else {
                    None
                };

                if let Some(report) = cached_report {
                    self.stats.cache_hits += 1;
                    combined_report.merge(report);
                    continue;
                } else if self.config.enable_caching {
                    self.stats.cache_misses += 1;
                }

                // Validate this node against this shape
                // TODO: Replace with actual single-node validation method
                let result = self.validator.validate_nodes(
                    store,
                    shape_id,
                    std::slice::from_ref(focus_node),
                    None,
                )?;

                // Cache the result
                if self.config.enable_caching {
                    self.cache_result(cache_key, result.clone());
                }

                combined_report.merge(result);
            }
        }

        self.stats.validation_time_ms += start.elapsed().as_millis() as u64;
        Ok(combined_report)
    }

    /// Analyze which shapes are affected by a changeset
    fn analyze_affected_shapes(&self, changeset: &Changeset) -> Result<HashSet<ShapeId>> {
        let mut affected = HashSet::new();
        let changed_predicates = changeset.affected_predicates();
        let changed_subjects = changeset.affected_subjects();

        for (shape_id, dep) in &self.dependencies {
            // Check if any changed predicate affects this shape
            let predicate_match = dep
                .affected_by_predicates
                .intersection(&changed_predicates)
                .next()
                .is_some();

            // Check if any changed subject is a target of this shape
            let subject_match = dep
                .target_nodes
                .intersection(&changed_subjects)
                .next()
                .is_some();

            if predicate_match || subject_match {
                affected.insert(shape_id.clone());
            }
        }

        // If no dependencies built yet, return all shapes (conservative)
        if affected.is_empty() && self.dependencies.is_empty() {
            for (shape_id, _) in self.validator.shapes().iter() {
                affected.insert((*shape_id).clone());
            }
        }

        Ok(affected)
    }

    /// Build dependency information for all shapes
    pub fn build_dependencies(&mut self, _store: &dyn Store) -> Result<()> {
        self.dependencies.clear();

        for (shape_id, _shape) in self.validator.shapes().iter() {
            let dep = ShapeDependency {
                affected_by_predicates: HashSet::new(),
                target_nodes: HashSet::new(),
            };

            // TODO: Extract predicates from constraints
            // In a full implementation, we would recursively analyze the constraints
            // to find all affected predicates. For now, this is a placeholder.

            // TODO: Get target nodes for this shape
            // Implement proper target extraction

            self.dependencies.insert((*shape_id).clone(), dep);
        }

        Ok(())
    }

    /// Make a cache key from shape ID and focus node
    fn make_cache_key(&self, shape_id: &ShapeId, focus_node: &Term) -> String {
        format!("{}::{:?}", shape_id, focus_node)
    }

    /// Get a cached result if available and not expired
    fn get_cached_result(&self, cache_key: &str) -> Option<&CachedValidationResult> {
        self.cache.get(cache_key).and_then(|cached| {
            if let Some(ttl_seconds) = self.config.cache_ttl_seconds {
                if cached.cached_at.elapsed() > Duration::from_secs(ttl_seconds) {
                    return None; // Expired
                }
            }
            Some(cached)
        })
    }

    /// Cache a validation result
    fn cache_result(&mut self, cache_key: String, report: ValidationReport) {
        // Evict old entries if cache is full
        if self.cache.len() >= self.config.max_cache_size {
            self.evict_oldest_cache_entries();
        }

        self.cache.insert(
            cache_key,
            CachedValidationResult {
                report,
                cached_at: Instant::now(),
            },
        );
    }

    /// Evict oldest cache entries (LRU-like)
    fn evict_oldest_cache_entries(&mut self) {
        let to_remove = self.cache.len() / 10; // Remove 10%

        // Collect entries sorted by age
        let mut entries: Vec<_> = self
            .cache
            .iter()
            .map(|(k, v)| (k.clone(), v.cached_at))
            .collect();
        entries.sort_by_key(|(_, cached_at)| *cached_at);

        // Remove oldest entries
        for (key, _) in entries.iter().take(to_remove) {
            self.cache.remove(key);
        }

        self.stats.cache_evictions += to_remove;
    }

    /// Clear the validation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &IncrementalStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = IncrementalStats::default();
    }

    /// Get underlying validator
    pub fn validator(&self) -> &Validator {
        &self.validator
    }

    /// Get underlying validator (mutable)
    pub fn validator_mut(&mut self) -> &mut Validator {
        &mut self.validator
    }
}

/// Statistics for incremental validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IncrementalStats {
    /// Total changesets processed
    pub total_changesets: usize,
    /// Total individual changes
    pub total_changes: usize,
    /// Number of empty changesets
    pub empty_changesets: usize,
    /// Total affected shapes
    pub affected_shapes: usize,
    /// Total affected nodes
    pub affected_nodes: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Cache evictions
    pub cache_evictions: usize,
    /// Total validation time (milliseconds)
    pub validation_time_ms: u64,
}

impl IncrementalStats {
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Calculate average changes per changeset
    pub fn avg_changes_per_changeset(&self) -> f64 {
        if self.total_changesets == 0 {
            0.0
        } else {
            self.total_changes as f64 / self.total_changesets as f64
        }
    }

    /// Calculate average validation time per changeset
    pub fn avg_validation_time_ms(&self) -> f64 {
        if self.total_changesets == 0 {
            0.0
        } else {
            self.validation_time_ms as f64 / self.total_changesets as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode};

    #[test]
    fn test_changeset_creation() {
        let changeset = Changeset::new();
        assert!(changeset.is_empty());
        assert_eq!(changeset.change_count(), 0);
    }

    #[test]
    fn test_changeset_additions() {
        let mut changeset = Changeset::new();

        let subject_node = NamedNode::new_unchecked("http://example.org/s");
        let predicate = NamedNode::new_unchecked("http://example.org/p");
        let object = Literal::new_simple_literal("value");

        let quad = Quad::new_default_graph(subject_node, predicate, object);
        changeset.add_addition(quad);

        assert!(!changeset.is_empty());
        assert_eq!(changeset.change_count(), 1);
        assert_eq!(changeset.affected_subjects().len(), 1);
        assert_eq!(changeset.affected_predicates().len(), 1);
    }

    #[test]
    fn test_incremental_config_default() {
        let config = IncrementalConfig::default();
        assert!(config.enable_caching);
        assert!(config.enable_dependency_analysis);
        assert_eq!(config.batch_size, 1000);
    }

    #[test]
    fn test_incremental_stats() {
        let stats = IncrementalStats {
            total_changesets: 100,
            total_changes: 500,
            cache_hits: 75,
            cache_misses: 25,
            ..Default::default()
        };

        assert_eq!(stats.cache_hit_rate(), 0.75);
        assert_eq!(stats.avg_changes_per_changeset(), 5.0);
    }

    #[test]
    fn test_cache_key_generation() {
        let validator = Validator::new();
        let config = IncrementalConfig::default();
        let incremental = IncrementalValidator::new(validator, config);

        let shape_id = ShapeId::new("test:shape");
        let focus_node = Term::NamedNode(NamedNode::new_unchecked("http://example.org/node"));

        let key = incremental.make_cache_key(&shape_id, &focus_node);
        assert!(key.contains("test:shape"));
    }
}
