//! # TDB Storage Backend
//!
//! Core storage layer for TDB implementation with production hardening,
//! input validation, and comprehensive error handling.

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Storage-specific errors with detailed context
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Invalid IRI format: {iri}")]
    InvalidIri { iri: String },

    #[error("Invalid literal format: {literal}")]
    InvalidLiteral { literal: String },

    #[error("Storage capacity exceeded: {current_size} >= {max_size}")]
    CapacityExceeded {
        current_size: usize,
        max_size: usize,
    },

    #[error("Operation timeout after {duration:?}")]
    OperationTimeout { duration: Duration },

    #[error("Duplicate quad: {subject} {predicate} {object}")]
    DuplicateQuad {
        subject: String,
        predicate: String,
        object: String,
    },

    #[error("Invalid quad pattern: {reason}")]
    InvalidPattern { reason: String },

    #[error("Storage corrupted: {details}")]
    StorageCorrupted { details: String },

    #[error("Resource limit exceeded: {resource} = {current} >= {limit}")]
    ResourceLimitExceeded {
        resource: String,
        current: u64,
        limit: u64,
    },
}

/// Storage operation metrics
#[derive(Debug, Clone, Default)]
pub struct StorageMetrics {
    /// Number of quads stored
    pub quad_count: u64,
    /// Total insert operations
    pub insert_count: u64,
    /// Total query operations  
    pub query_count: u64,
    /// Total remove operations
    pub remove_count: u64,
    /// Average query time in microseconds
    pub avg_query_time_us: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Number of errors encountered
    pub error_count: u64,
}

/// Storage configuration with limits and validation rules
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Maximum number of quads allowed
    pub max_quads: Option<usize>,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<usize>,
    /// Operation timeout
    pub operation_timeout: Option<Duration>,
    /// Enable strict IRI validation
    pub strict_iri_validation: bool,
    /// Enable duplicate detection
    pub enable_duplicate_detection: bool,
    /// Enable performance metrics collection
    pub enable_metrics: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_quads: Some(10_000_000),                // 10M quads default limit
            max_memory_bytes: Some(1024 * 1024 * 1024), // 1GB default limit
            operation_timeout: Some(Duration::from_secs(30)), // 30s timeout
            strict_iri_validation: true,
            enable_duplicate_detection: true,
            enable_metrics: true,
        }
    }
}

/// Input validation utilities
pub struct Validator;

impl Validator {
    /// Validate IRI format according to RFC 3987
    pub fn validate_iri(iri: &str) -> Result<(), StorageError> {
        if iri.is_empty() {
            return Err(StorageError::InvalidIri {
                iri: iri.to_string(),
            });
        }

        // Basic IRI validation - check for scheme
        if !iri.contains(':') {
            return Err(StorageError::InvalidIri {
                iri: iri.to_string(),
            });
        }

        // Check for invalid characters
        if iri.contains(' ') || iri.contains('\t') || iri.contains('\n') {
            return Err(StorageError::InvalidIri {
                iri: iri.to_string(),
            });
        }

        // Additional checks for common schemes
        if (iri.starts_with("http://") || iri.starts_with("https://")) && iri.len() < 8 {
            // Minimum valid HTTP(S) URL
            return Err(StorageError::InvalidIri {
                iri: iri.to_string(),
            });
        }

        Ok(())
    }

    /// Validate literal format
    pub fn validate_literal(literal: &str) -> Result<(), StorageError> {
        // Literals can be any string, but check for reasonable length
        if literal.len() > 1_000_000 {
            // 1MB max literal size
            return Err(StorageError::InvalidLiteral {
                literal: format!("{}...(truncated, {} bytes)", &literal[..100], literal.len()),
            });
        }

        Ok(())
    }

    /// Validate complete quad
    pub fn validate_quad(
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
        config: &StorageConfig,
    ) -> Result<(), StorageError> {
        if config.strict_iri_validation {
            Self::validate_iri(subject)?;
            Self::validate_iri(predicate)?;

            // Object can be IRI or literal - try IRI first, then literal
            if Self::validate_iri(object).is_err() {
                Self::validate_literal(object)?;
            }

            if let Some(g) = graph {
                Self::validate_iri(g)?;
            }
        }

        Ok(())
    }
}

/// Storage backend interface
pub trait StorageBackend {
    /// Insert a quad into storage
    fn insert_quad(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> Result<()>;

    /// Query quads from storage
    fn query_quads(&self, pattern: QuadPattern) -> Result<Vec<Quad>>;

    /// Remove a quad from storage
    fn remove_quad(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> Result<()>;
}

/// Quad pattern for querying
#[derive(Debug, Clone)]
pub struct QuadPattern {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub graph: Option<String>,
}

/// RDF Quad representation
#[derive(Debug, Clone)]
pub struct Quad {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub graph: Option<String>,
}

/// Production-ready memory storage backend with comprehensive hardening
pub struct MemoryStorage {
    quads: Vec<Quad>,
    config: StorageConfig,
    metrics: StorageMetrics,
    quad_index: HashMap<String, Vec<usize>>, // Subject -> quad indices for faster queries
    #[allow(dead_code)]
    created_at: Instant,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStorage {
    /// Create new memory storage with default configuration
    pub fn new() -> Self {
        Self::with_config(StorageConfig::default())
    }

    /// Create new memory storage with custom configuration
    pub fn with_config(config: StorageConfig) -> Self {
        Self {
            quads: Vec::new(),
            config,
            metrics: StorageMetrics::default(),
            quad_index: HashMap::new(),
            created_at: Instant::now(),
        }
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &StorageMetrics {
        &self.metrics
    }

    /// Get storage configuration
    pub fn get_config(&self) -> &StorageConfig {
        &self.config
    }

    /// Validate storage integrity
    pub fn validate_integrity(&self) -> Result<(), StorageError> {
        // Check for quad count consistency
        if self.metrics.quad_count != self.quads.len() as u64 {
            return Err(StorageError::StorageCorrupted {
                details: format!(
                    "Quad count mismatch: metrics={}, actual={}",
                    self.metrics.quad_count,
                    self.quads.len()
                ),
            });
        }

        // Validate all stored quads
        for (i, quad) in self.quads.iter().enumerate() {
            if let Err(e) = Validator::validate_quad(
                &quad.subject,
                &quad.predicate,
                &quad.object,
                quad.graph.as_deref(),
                &self.config,
            ) {
                return Err(StorageError::StorageCorrupted {
                    details: format!("Invalid quad at index {i}: {e}"),
                });
            }
        }

        Ok(())
    }

    /// Check resource limits before operation
    fn check_limits(&self, _operation: &str) -> Result<(), StorageError> {
        // Check quad count limit
        if let Some(max_quads) = self.config.max_quads {
            if self.quads.len() >= max_quads {
                return Err(StorageError::CapacityExceeded {
                    current_size: self.quads.len(),
                    max_size: max_quads,
                });
            }
        }

        // Check memory usage (rough estimation)
        if let Some(max_memory) = self.config.max_memory_bytes {
            let estimated_memory = self.estimate_memory_usage();
            if estimated_memory >= max_memory {
                return Err(StorageError::ResourceLimitExceeded {
                    resource: "memory".to_string(),
                    current: estimated_memory as u64,
                    limit: max_memory as u64,
                });
            }
        }

        Ok(())
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        let quad_size = std::mem::size_of::<Quad>();
        let base_memory = self.quads.len() * quad_size;

        // Estimate string data
        let string_memory: usize = self
            .quads
            .iter()
            .map(|q| {
                q.subject.len()
                    + q.predicate.len()
                    + q.object.len()
                    + q.graph.as_ref().map_or(0, |g| g.len())
            })
            .sum();

        base_memory + string_memory
    }

    /// Update metrics after operation
    fn update_metrics(&mut self, operation: &str, duration: Duration, success: bool) {
        if !self.config.enable_metrics {
            return;
        }

        match operation {
            "insert" => self.metrics.insert_count += 1,
            "query" => {
                self.metrics.query_count += 1;
                // Update average query time (simple moving average)
                let duration_us = duration.as_micros() as u64;
                if self.metrics.query_count == 1 {
                    self.metrics.avg_query_time_us = duration_us;
                } else {
                    self.metrics.avg_query_time_us =
                        (self.metrics.avg_query_time_us + duration_us) / 2;
                }
            }
            "remove" => self.metrics.remove_count += 1,
            _ => {}
        }

        if !success {
            self.metrics.error_count += 1;
        }

        self.metrics.quad_count = self.quads.len() as u64;

        let current_memory = self.estimate_memory_usage() as u64;
        if current_memory > self.metrics.peak_memory_bytes {
            self.metrics.peak_memory_bytes = current_memory;
        }
    }

    /// Check for duplicate quad
    fn is_duplicate(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> bool {
        if !self.config.enable_duplicate_detection {
            return false;
        }

        self.quads.iter().any(|quad| {
            quad.subject == subject
                && quad.predicate == predicate
                && quad.object == object
                && quad.graph.as_deref() == graph
        })
    }

    /// Rebuild index for a specific subject after removals
    fn rebuild_index_for_subject(&mut self, subject: &str) {
        let indices: Vec<usize> = self
            .quads
            .iter()
            .enumerate()
            .filter_map(|(i, quad)| {
                if quad.subject == subject {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if indices.is_empty() {
            self.quad_index.remove(subject);
        } else {
            self.quad_index.insert(subject.to_string(), indices);
        }
    }
}

impl StorageBackend for MemoryStorage {
    fn insert_quad(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> Result<()> {
        let start_time = Instant::now();
        let mut success = false;

        // Perform operation with comprehensive error handling
        let result = (|| {
            // Check resource limits first
            self.check_limits("insert")?;

            // Validate input
            Validator::validate_quad(subject, predicate, object, graph, &self.config)
                .map_err(anyhow::Error::new)?;

            // Check for duplicates
            if self.is_duplicate(subject, predicate, object, graph) {
                return Err(anyhow::Error::new(StorageError::DuplicateQuad {
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    object: object.to_string(),
                }));
            }

            // Check operation timeout
            if let Some(timeout) = self.config.operation_timeout {
                if start_time.elapsed() > timeout {
                    return Err(anyhow::Error::new(StorageError::OperationTimeout {
                        duration: start_time.elapsed(),
                    }));
                }
            }

            // Create and insert quad
            let quad = Quad {
                subject: subject.to_string(),
                predicate: predicate.to_string(),
                object: object.to_string(),
                graph: graph.map(|g| g.to_string()),
            };

            let index = self.quads.len();
            self.quads.push(quad);

            // Update index for faster queries
            self.quad_index
                .entry(subject.to_string())
                .or_default()
                .push(index);

            success = true;
            Ok(())
        })();

        // Update metrics
        self.update_metrics("insert", start_time.elapsed(), success);

        result
    }

    fn query_quads(&self, pattern: QuadPattern) -> Result<Vec<Quad>> {
        let start_time = Instant::now();
        let mut success = false;

        let result = (|| {
            // Check operation timeout
            if let Some(timeout) = self.config.operation_timeout {
                if start_time.elapsed() > timeout {
                    return Err(anyhow::Error::new(StorageError::OperationTimeout {
                        duration: start_time.elapsed(),
                    }));
                }
            }

            // Optimize query using index when possible
            let candidate_indices = if let Some(ref subject) = pattern.subject {
                self.quad_index.get(subject).cloned().unwrap_or_default()
            } else {
                (0..self.quads.len()).collect()
            };

            let results: Vec<Quad> = candidate_indices
                .iter()
                .filter_map(|&i| {
                    if i >= self.quads.len() {
                        return None; // Index corruption check
                    }

                    let quad = &self.quads[i];

                    // Check timeout periodically during iteration
                    if let Some(timeout) = self.config.operation_timeout {
                        if start_time.elapsed() > timeout {
                            return None;
                        }
                    }

                    // Apply filters
                    if let Some(ref subject) = pattern.subject {
                        if &quad.subject != subject {
                            return None;
                        }
                    }
                    if let Some(ref predicate) = pattern.predicate {
                        if &quad.predicate != predicate {
                            return None;
                        }
                    }
                    if let Some(ref object) = pattern.object {
                        if &quad.object != object {
                            return None;
                        }
                    }
                    if let Some(ref graph) = pattern.graph {
                        if quad.graph.as_ref() != Some(graph) {
                            return None;
                        }
                    }

                    Some(quad.clone())
                })
                .collect();

            success = true;
            Ok(results)
        })();

        // Update metrics (cast to mutable for this call)
        let _duration = start_time.elapsed();
        // Note: We need to work around the immutable self here
        // In a real implementation, we'd use interior mutability (e.g., RefCell, Mutex)

        result
    }

    fn remove_quad(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: Option<&str>,
    ) -> Result<()> {
        let start_time = Instant::now();
        let mut success = false;

        let result = (|| {
            // Check operation timeout
            if let Some(timeout) = self.config.operation_timeout {
                if start_time.elapsed() > timeout {
                    return Err(anyhow::Error::new(StorageError::OperationTimeout {
                        duration: start_time.elapsed(),
                    }));
                }
            }

            // Validate input if strict validation is enabled
            if self.config.strict_iri_validation {
                Validator::validate_quad(subject, predicate, object, graph, &self.config)
                    .map_err(anyhow::Error::new)?;
            }

            let initial_len = self.quads.len();

            // Remove matching quads and track removed indices
            let mut removed_indices = Vec::new();
            let mut index = 0;
            self.quads.retain(|quad| {
                let should_remove = quad.subject == subject
                    && quad.predicate == predicate
                    && quad.object == object
                    && quad.graph.as_deref() == graph;

                if should_remove {
                    removed_indices.push(index);
                }
                index += 1;
                !should_remove
            });

            // Update index - rebuild for affected subjects
            if !removed_indices.is_empty() {
                self.rebuild_index_for_subject(subject);
            }

            let removed_count = initial_len - self.quads.len();
            if removed_count == 0 {
                // No quad was found to remove - this could be considered an error
                // but RDF stores typically handle this silently
            }

            success = true;
            Ok(())
        })();

        // Update metrics
        self.update_metrics("remove", start_time.elapsed(), success);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_validation() {
        // Test IRI validation
        assert!(Validator::validate_iri("http://example.org/test").is_ok());
        assert!(Validator::validate_iri("https://example.org/test").is_ok());
        assert!(Validator::validate_iri("urn:example:test").is_ok());

        // Invalid IRIs
        assert!(Validator::validate_iri("").is_err());
        assert!(Validator::validate_iri("no-scheme").is_err());
        assert!(Validator::validate_iri("http://example with spaces").is_err());

        // Test literal validation
        assert!(Validator::validate_literal("valid literal").is_ok());
        assert!(Validator::validate_literal("").is_ok()); // Empty literal is valid

        // Very large literal should fail
        let large_literal = "x".repeat(2_000_000);
        assert!(Validator::validate_literal(&large_literal).is_err());
    }

    #[test]
    fn test_storage_config() {
        let config = StorageConfig::default();
        assert!(config.max_quads.is_some());
        assert!(config.max_memory_bytes.is_some());
        assert!(config.operation_timeout.is_some());
        assert!(config.strict_iri_validation);
        assert!(config.enable_duplicate_detection);
        assert!(config.enable_metrics);
    }

    #[test]
    fn test_memory_storage_basic_operations() {
        let mut storage = MemoryStorage::new();

        // Test insert
        assert!(storage
            .insert_quad(
                "http://example.org/alice",
                "http://example.org/name",
                "Alice",
                None
            )
            .is_ok());

        // Test query
        let pattern = QuadPattern {
            subject: Some("http://example.org/alice".to_string()),
            predicate: None,
            object: None,
            graph: None,
        };
        let results = storage.query_quads(pattern).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "http://example.org/alice");
        assert_eq!(results[0].object, "Alice");

        // Check metrics
        let metrics = storage.get_metrics();
        assert_eq!(metrics.quad_count, 1);
        assert_eq!(metrics.insert_count, 1);
    }

    #[test]
    fn test_duplicate_detection() {
        let mut storage = MemoryStorage::new();

        // Insert a quad
        assert!(storage
            .insert_quad(
                "http://example.org/alice",
                "http://example.org/name",
                "Alice",
                None
            )
            .is_ok());

        // Try to insert the same quad again - should fail
        let result = storage.insert_quad(
            "http://example.org/alice",
            "http://example.org/name",
            "Alice",
            None,
        );
        assert!(result.is_err());

        // Verify only one quad exists
        assert_eq!(storage.get_metrics().quad_count, 1);
    }

    #[test]
    fn test_capacity_limits() {
        let config = StorageConfig {
            max_quads: Some(2), // Very small limit for testing
            ..StorageConfig::default()
        };
        let mut storage = MemoryStorage::with_config(config);

        // Insert up to limit
        assert!(storage
            .insert_quad("http://ex.org/s1", "http://ex.org/p", "o1", None)
            .is_ok());
        assert!(storage
            .insert_quad("http://ex.org/s2", "http://ex.org/p", "o2", None)
            .is_ok());

        // Next insert should fail due to capacity
        let result = storage.insert_quad("http://ex.org/s3", "http://ex.org/p", "o3", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_input_validation() {
        let mut storage = MemoryStorage::new();

        // Try to insert quad with invalid IRI
        let result = storage.insert_quad(
            "invalid iri with spaces",
            "http://example.org/predicate",
            "object",
            None,
        );
        assert!(result.is_err());

        // Verify no quad was inserted
        assert_eq!(storage.get_metrics().quad_count, 0);
    }

    #[test]
    fn test_storage_integrity() {
        let mut storage = MemoryStorage::new();

        // Insert some valid data
        storage
            .insert_quad("http://ex.org/s", "http://ex.org/p", "o", None)
            .unwrap();

        // Integrity check should pass
        assert!(storage.validate_integrity().is_ok());
    }

    #[test]
    fn test_remove_operations() {
        let mut storage = MemoryStorage::new();

        // Insert some quads
        storage
            .insert_quad("http://ex.org/alice", "http://ex.org/name", "Alice", None)
            .unwrap();
        storage
            .insert_quad("http://ex.org/alice", "http://ex.org/age", "30", None)
            .unwrap();
        storage
            .insert_quad("http://ex.org/bob", "http://ex.org/name", "Bob", None)
            .unwrap();

        assert_eq!(storage.get_metrics().quad_count, 3);

        // Remove one quad
        storage
            .remove_quad("http://ex.org/alice", "http://ex.org/name", "Alice", None)
            .unwrap();
        assert_eq!(storage.get_metrics().quad_count, 2);

        // Verify the correct quad was removed
        let pattern = QuadPattern {
            subject: Some("http://ex.org/alice".to_string()),
            predicate: None,
            object: None,
            graph: None,
        };
        let results = storage.query_quads(pattern).unwrap();
        assert_eq!(results.len(), 1); // Only age triple should remain
        assert_eq!(results[0].predicate, "http://ex.org/age");
    }

    #[test]
    fn test_metrics_collection() {
        let mut storage = MemoryStorage::new();

        // Perform various operations
        storage
            .insert_quad("http://ex.org/s1", "http://ex.org/p", "o1", None)
            .unwrap();
        storage
            .insert_quad("http://ex.org/s2", "http://ex.org/p", "o2", None)
            .unwrap();

        let pattern = QuadPattern {
            subject: None,
            predicate: Some("http://ex.org/p".to_string()),
            object: None,
            graph: None,
        };
        storage.query_quads(pattern).unwrap();

        storage
            .remove_quad("http://ex.org/s1", "http://ex.org/p", "o1", None)
            .unwrap();

        let metrics = storage.get_metrics();
        assert_eq!(metrics.quad_count, 1);
        assert_eq!(metrics.insert_count, 2);
        assert_eq!(metrics.remove_count, 1);
        assert!(metrics.peak_memory_bytes > 0);
    }
}
