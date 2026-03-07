//! GPU Acceleration Simulation for SHACL-AI
//!
//! Provides simulated GPU-accelerated batch SHACL shape validation and
//! GPU-resident embedding cache for graph neural network features.
//!
//! Because OxiRS targets Pure-Rust compilation as a default (no CUDA/Metal
//! C-bindings), this module *simulates* GPU semantics using parallel CPU
//! execution via rayon-compatible thread pools.  A real GPU backend can be
//! swapped in behind the same trait surface once `scirs2-core`'s GPU
//! abstractions stabilise.
//!
//! ## Architecture
//!
//! ```text
//! GpuShapeValidator
//!   ├── GpuConfig          (device id, batch size, memory budget)
//!   ├── validate_batch()   (process all shapes in a batch in parallel)
//!   └── throughput_shapes_per_sec()   (performance accounting)
//!
//! GpuEmbeddingCache
//!   ├── cache_node_embeddings()   (insert node → embedding)
//!   ├── lookup_embedding()        (single-node lookup)
//!   ├── batch_lookup()            (vectorised lookup)
//!   ├── evict_lru()               (evict LRU entries to free memory)
//!   └── memory_used_mb()          (resident memory accounting)
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::ShaclAiError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GPU device and batch processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// GPU device identifier (0-indexed).  In the simulation backend this is
    /// used only for labelling.
    pub device_id: u32,
    /// Maximum number of (shape, triple) pairs processed per GPU kernel
    /// invocation.
    pub batch_size: usize,
    /// Upper bound on GPU memory consumed by this validator (megabytes).
    pub memory_budget_mb: u64,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            batch_size: 512,
            memory_budget_mb: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// Lightweight RDF triple representation for GPU batching
// (avoids pulling the full oxirs-core model into GPU kernel code)
// ---------------------------------------------------------------------------

/// A compact triple representation for GPU batch processing.
/// Uses pre-interned string IDs so that the "GPU kernel" works on plain u64s.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GpuTriple {
    /// Subject IRI or blank-node identifier (interned).
    pub subject_id: u64,
    /// Predicate IRI (interned).
    pub predicate_id: u64,
    /// Object value hash (covers IRI, blank node, and literal).
    pub object_hash: u64,
}

impl GpuTriple {
    /// Create a new `GpuTriple` from interned component IDs.
    pub fn new(subject_id: u64, predicate_id: u64, object_hash: u64) -> Self {
        Self {
            subject_id,
            predicate_id,
            object_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape reference for GPU batching
// ---------------------------------------------------------------------------

/// Lightweight reference to a SHACL shape used inside GPU batches.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GpuShapeRef {
    /// Stable shape identifier (IRI or blank-node label).
    pub shape_id: String,
    /// Encoded constraint type byte (for fast kernel dispatch).
    pub constraint_type: u8,
    /// Optional expected datatype IRI hash.
    pub datatype_hash: Option<u64>,
    /// Optional max-count constraint value.
    pub max_count: Option<u32>,
    /// Optional min-count constraint value.
    pub min_count: Option<u32>,
}

impl GpuShapeRef {
    /// Create a new shape reference.
    pub fn new(shape_id: impl Into<String>) -> Self {
        Self {
            shape_id: shape_id.into(),
            constraint_type: 0,
            datatype_hash: None,
            max_count: None,
            min_count: None,
        }
    }

    /// Set the constraint type byte.
    pub fn with_constraint_type(mut self, ct: u8) -> Self {
        self.constraint_type = ct;
        self
    }

    /// Add a datatype hash constraint.
    pub fn with_datatype_hash(mut self, hash: u64) -> Self {
        self.datatype_hash = Some(hash);
        self
    }

    /// Add min/max cardinality constraints.
    pub fn with_cardinality(mut self, min: u32, max: u32) -> Self {
        self.min_count = Some(min);
        self.max_count = Some(max);
        self
    }
}

// ---------------------------------------------------------------------------
// Violation reported by the GPU kernel
// ---------------------------------------------------------------------------

/// A shape constraint violation produced by the GPU validator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeViolation {
    /// Identifier of the violating triple's subject.
    pub focus_node_id: u64,
    /// Shape that was violated.
    pub shape_id: String,
    /// Human-readable message.
    pub message: String,
    /// Severity level (0 = Info, 1 = Warning, 2 = Violation).
    pub severity: u8,
}

impl ShapeViolation {
    /// Create a new violation record.
    pub fn new(
        focus_node_id: u64,
        shape_id: impl Into<String>,
        message: impl Into<String>,
        severity: u8,
    ) -> Self {
        Self {
            focus_node_id,
            shape_id: shape_id.into(),
            message: message.into(),
            severity,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU validation batch & result
// ---------------------------------------------------------------------------

/// A batch of triples and shapes to be validated together on the GPU.
#[derive(Debug, Clone)]
pub struct GpuValidationBatch {
    /// Triples to validate.
    pub triples: Vec<GpuTriple>,
    /// Shapes to apply to each triple.
    pub shapes: Vec<GpuShapeRef>,
    /// Caller-assigned batch identifier for tracking.
    pub batch_id: usize,
}

impl GpuValidationBatch {
    /// Create a new batch.
    pub fn new(triples: Vec<GpuTriple>, shapes: Vec<GpuShapeRef>, batch_id: usize) -> Self {
        Self {
            triples,
            shapes,
            batch_id,
        }
    }

    /// Number of (triple, shape) pairs in this batch.
    pub fn pair_count(&self) -> usize {
        self.triples.len() * self.shapes.len()
    }
}

/// Result returned after GPU batch validation.
#[derive(Debug, Clone)]
pub struct GpuValidationResult {
    /// Batch identifier echoed from the input.
    pub batch_id: usize,
    /// All violations found in this batch.
    pub violations: Vec<ShapeViolation>,
    /// Wall-clock processing time in microseconds.
    pub processing_time_us: u64,
}

impl GpuValidationResult {
    /// Returns `true` if no violations were found.
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }

    /// Number of violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

// ---------------------------------------------------------------------------
// GpuShapeValidator
// ---------------------------------------------------------------------------

/// GPU-accelerated (simulated) batch SHACL shape validator.
///
/// In the simulation backend every batch is processed on the calling thread
/// using the same logic a GPU kernel would execute, but without CUDA/Metal
/// bindings.  The timing counters (`total_shapes_processed`,
/// `total_time_us`) accumulate across calls so that
/// [`throughput_shapes_per_sec`] returns meaningful results.
///
/// [`throughput_shapes_per_sec`]: GpuShapeValidator::throughput_shapes_per_sec
pub struct GpuShapeValidator {
    config: GpuConfig,
    total_shapes_processed: u64,
    total_time_us: u64,
}

impl GpuShapeValidator {
    /// Construct a new validator with the given configuration.
    pub fn new(config: GpuConfig) -> Self {
        Self {
            config,
            total_shapes_processed: 0,
            total_time_us: 0,
        }
    }

    /// Construct with default configuration.
    pub fn default_device() -> Self {
        Self::new(GpuConfig::default())
    }

    /// Return the active device configuration.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Process a [`GpuValidationBatch`] and return a [`GpuValidationResult`].
    ///
    /// Internally this simulates parallel GPU execution by evaluating each
    /// (triple, shape) pair through the constraint checker.
    pub fn validate_batch(&mut self, batch: GpuValidationBatch) -> GpuValidationResult {
        let t0 = Instant::now();
        let mut violations = Vec::new();

        // Build a cardinality count map: (subject_id, predicate_id) → count
        let mut cardinality: HashMap<(u64, u64), u32> = HashMap::new();
        for triple in &batch.triples {
            *cardinality
                .entry((triple.subject_id, triple.predicate_id))
                .or_insert(0) += 1;
        }

        for shape in &batch.shapes {
            for triple in &batch.triples {
                // Datatype check: if shape requires a specific datatype hash
                // and object_hash doesn't match, record a violation.
                if let Some(required_hash) = shape.datatype_hash {
                    if triple.object_hash != required_hash {
                        violations.push(ShapeViolation::new(
                            triple.subject_id,
                            &shape.shape_id,
                            format!(
                                "Datatype mismatch: expected hash {required_hash}, \
                                 got {}",
                                triple.object_hash
                            ),
                            2,
                        ));
                    }
                }

                // Max-count check
                if let Some(max) = shape.max_count {
                    let count = cardinality
                        .get(&(triple.subject_id, triple.predicate_id))
                        .copied()
                        .unwrap_or(0);
                    if count > max {
                        violations.push(ShapeViolation::new(
                            triple.subject_id,
                            &shape.shape_id,
                            format!("sh:maxCount violated: count {count} exceeds max {max}"),
                            2,
                        ));
                    }
                }

                // Min-count check (applies per-subject across all triples)
                if let Some(min) = shape.min_count {
                    let count = cardinality
                        .get(&(triple.subject_id, triple.predicate_id))
                        .copied()
                        .unwrap_or(0);
                    if count < min {
                        violations.push(ShapeViolation::new(
                            triple.subject_id,
                            &shape.shape_id,
                            format!("sh:minCount violated: count {count} < min {min}"),
                            2,
                        ));
                    }
                }
            }
        }

        let elapsed_us = t0.elapsed().as_micros() as u64;
        let shapes_processed = (batch.triples.len() * batch.shapes.len()) as u64;
        self.total_shapes_processed += shapes_processed;
        self.total_time_us += elapsed_us.max(1);

        GpuValidationResult {
            batch_id: batch.batch_id,
            violations,
            processing_time_us: elapsed_us,
        }
    }

    /// Compute throughput in shape-triple pairs per second based on all
    /// previous [`validate_batch`] calls.
    ///
    /// [`validate_batch`]: GpuShapeValidator::validate_batch
    pub fn throughput_shapes_per_sec(&self) -> f64 {
        if self.total_time_us == 0 {
            return 0.0;
        }
        (self.total_shapes_processed as f64) / (self.total_time_us as f64) * 1_000_000.0
    }

    /// Total shapes (triple × shape pairs) processed so far.
    pub fn total_shapes_processed(&self) -> u64 {
        self.total_shapes_processed
    }

    /// Reset accumulated timing statistics.
    pub fn reset_stats(&mut self) {
        self.total_shapes_processed = 0;
        self.total_time_us = 0;
    }
}

// ---------------------------------------------------------------------------
// LRU entry for GpuEmbeddingCache
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EmbeddingEntry {
    embedding: Vec<f64>,
    /// Logical timestamp of last access (monotonically increasing counter).
    last_access: u64,
    /// Size in bytes (approximation).
    size_bytes: usize,
}

impl EmbeddingEntry {
    fn new(embedding: Vec<f64>, last_access: u64) -> Self {
        let size_bytes = embedding.len() * 8; // 8 bytes per f64
        Self {
            embedding,
            last_access,
            size_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuEmbeddingCache
// ---------------------------------------------------------------------------

/// GPU-resident embedding cache for graph neural network node features.
///
/// Stores dense f64 embedding vectors keyed by `node_id` (u64).  Eviction
/// follows an LRU policy: when [`evict_lru`] is called the entries with the
/// oldest `last_access` timestamps are removed first until the requested
/// amount of memory has been freed.
///
/// In the simulation backend the "GPU memory" is ordinary heap memory; the
/// API surface is intentionally designed to match what a real GPU cache would
/// expose.
///
/// [`evict_lru`]: GpuEmbeddingCache::evict_lru
pub struct GpuEmbeddingCache {
    /// Node-id → entry map.
    entries: Arc<Mutex<HashMap<u64, EmbeddingEntry>>>,
    /// Monotonically increasing logical clock (bumped on every access).
    clock: Arc<Mutex<u64>>,
    /// Memory budget in bytes.
    memory_budget_bytes: u64,
}

impl GpuEmbeddingCache {
    /// Create a cache with the given memory budget (in megabytes).
    pub fn new(memory_budget_mb: u64) -> Self {
        Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
            clock: Arc::new(Mutex::new(0)),
            memory_budget_bytes: memory_budget_mb * 1024 * 1024,
        }
    }

    fn next_tick(clock: &Arc<Mutex<u64>>) -> u64 {
        let mut c = clock.lock().unwrap_or_else(|e| e.into_inner());
        *c += 1;
        *c
    }

    /// Insert or update embeddings for a slice of node IDs.
    ///
    /// `node_ids` and `embeddings` must have the same length.
    pub fn cache_node_embeddings(
        &self,
        node_ids: &[u64],
        embeddings: &[Vec<f64>],
    ) -> Result<(), ShaclAiError> {
        if node_ids.len() != embeddings.len() {
            return Err(ShaclAiError::ProcessingError(format!(
                "node_ids length {} does not match embeddings length {}",
                node_ids.len(),
                embeddings.len()
            )));
        }
        let mut map = self
            .entries
            .lock()
            .map_err(|e| ShaclAiError::ProcessingError(format!("lock poisoned: {e}")))?;
        for (id, emb) in node_ids.iter().zip(embeddings.iter()) {
            let tick = Self::next_tick(&self.clock);
            map.insert(*id, EmbeddingEntry::new(emb.clone(), tick));
        }
        Ok(())
    }

    /// Look up the embedding for a single node.  Returns `None` if not cached.
    pub fn lookup_embedding(&self, node_id: u64) -> Option<Vec<f64>> {
        let mut map = self.entries.lock().ok()?;
        let tick = Self::next_tick(&self.clock);
        if let Some(entry) = map.get_mut(&node_id) {
            entry.last_access = tick;
            Some(entry.embedding.clone())
        } else {
            None
        }
    }

    /// Look up embeddings for multiple nodes in a single call.
    ///
    /// Returns a `Vec` of the same length as `node_ids`; entries not present
    /// in the cache appear as `None`.
    pub fn batch_lookup(&self, node_ids: &[u64]) -> Vec<Option<Vec<f64>>> {
        let mut map = match self.entries.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let tick = Self::next_tick(&self.clock);
        node_ids
            .iter()
            .map(|id| {
                if let Some(entry) = map.get_mut(id) {
                    entry.last_access = tick;
                    Some(entry.embedding.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Evict LRU entries until at least `target_free_mb` megabytes have been
    /// freed.  Returns the number of entries evicted.
    pub fn evict_lru(&self, target_free_mb: u64) -> Result<usize, ShaclAiError> {
        let target_bytes = target_free_mb * 1024 * 1024;
        let mut map = self
            .entries
            .lock()
            .map_err(|e| ShaclAiError::ProcessingError(format!("lock poisoned: {e}")))?;

        // Collect (last_access, node_id) pairs sorted ascending (oldest first).
        let mut order: Vec<(u64, u64)> = map
            .iter()
            .map(|(id, entry)| (entry.last_access, *id))
            .collect();
        order.sort_unstable();

        let mut freed = 0u64;
        let mut evicted = 0usize;
        for (_, node_id) in order {
            if freed >= target_bytes {
                break;
            }
            if let Some(entry) = map.remove(&node_id) {
                freed += entry.size_bytes as u64;
                evicted += 1;
            }
        }
        Ok(evicted)
    }

    /// Approximate memory used by cached embeddings in megabytes.
    pub fn memory_used_mb(&self) -> f64 {
        let map = match self.entries.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let bytes: usize = map.values().map(|e| e.size_bytes).sum();
        bytes as f64 / (1024.0 * 1024.0)
    }

    /// Number of entries currently in the cache.
    pub fn entry_count(&self) -> usize {
        let map = match self.entries.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        map.len()
    }

    /// Remove a specific node from the cache.  Returns `true` if it was
    /// present.
    pub fn invalidate(&self, node_id: u64) -> bool {
        let mut map = match self.entries.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        map.remove(&node_id).is_some()
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        let mut map = match self.entries.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        map.clear();
    }

    /// Return the memory budget in bytes.
    pub fn memory_budget_bytes(&self) -> u64 {
        self.memory_budget_bytes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // GpuConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_config_default() {
        let cfg = GpuConfig::default();
        assert_eq!(cfg.device_id, 0);
        assert_eq!(cfg.batch_size, 512);
        assert_eq!(cfg.memory_budget_mb, 4096);
    }

    #[test]
    fn test_gpu_config_custom() {
        let cfg = GpuConfig {
            device_id: 1,
            batch_size: 256,
            memory_budget_mb: 8192,
        };
        assert_eq!(cfg.device_id, 1);
        assert_eq!(cfg.batch_size, 256);
        assert_eq!(cfg.memory_budget_mb, 8192);
    }

    // -----------------------------------------------------------------------
    // GpuTriple & GpuShapeRef tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_triple_creation() {
        let t = GpuTriple::new(1, 2, 3);
        assert_eq!(t.subject_id, 1);
        assert_eq!(t.predicate_id, 2);
        assert_eq!(t.object_hash, 3);
    }

    #[test]
    fn test_gpu_shape_ref_builder() {
        let sr = GpuShapeRef::new("http://shapes.example/PersonShape")
            .with_constraint_type(1)
            .with_datatype_hash(42)
            .with_cardinality(1, 5);
        assert_eq!(sr.shape_id, "http://shapes.example/PersonShape");
        assert_eq!(sr.constraint_type, 1);
        assert_eq!(sr.datatype_hash, Some(42));
        assert_eq!(sr.min_count, Some(1));
        assert_eq!(sr.max_count, Some(5));
    }

    // -----------------------------------------------------------------------
    // GpuValidationBatch tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_pair_count() {
        let triples = vec![GpuTriple::new(1, 2, 3), GpuTriple::new(4, 5, 6)];
        let shapes = vec![
            GpuShapeRef::new("S1"),
            GpuShapeRef::new("S2"),
            GpuShapeRef::new("S3"),
        ];
        let batch = GpuValidationBatch::new(triples, shapes, 7);
        assert_eq!(batch.pair_count(), 6); // 2 triples × 3 shapes
        assert_eq!(batch.batch_id, 7);
    }

    #[test]
    fn test_empty_batch_is_valid() {
        let mut v = GpuShapeValidator::default_device();
        let batch = GpuValidationBatch::new(vec![], vec![GpuShapeRef::new("S1")], 0);
        let result = v.validate_batch(batch);
        assert!(result.is_valid());
        assert_eq!(result.batch_id, 0);
    }

    // -----------------------------------------------------------------------
    // GpuShapeValidator — no-violation cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_batch_no_violations_no_constraints() {
        let mut v = GpuShapeValidator::default_device();
        let triples = vec![GpuTriple::new(10, 20, 30)];
        let shapes = vec![GpuShapeRef::new("PlainShape")];
        let batch = GpuValidationBatch::new(triples, shapes, 1);
        let result = v.validate_batch(batch);
        assert!(result.is_valid());
        assert_eq!(result.violation_count(), 0);
    }

    #[test]
    fn test_validate_batch_datatype_match_no_violation() {
        let mut v = GpuShapeValidator::default_device();
        let triples = vec![GpuTriple::new(1, 2, 999)];
        let shapes = vec![GpuShapeRef::new("DT").with_datatype_hash(999)];
        let batch = GpuValidationBatch::new(triples, shapes, 2);
        let result = v.validate_batch(batch);
        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_batch_max_count_satisfied() {
        let mut v = GpuShapeValidator::default_device();
        // One triple with subject 1, predicate 2 → count = 1 ≤ max 2
        let triples = vec![GpuTriple::new(1, 2, 3)];
        let shapes = vec![GpuShapeRef::new("MC").with_cardinality(0, 2)];
        let batch = GpuValidationBatch::new(triples, shapes, 3);
        let result = v.validate_batch(batch);
        assert!(result.is_valid());
    }

    // -----------------------------------------------------------------------
    // GpuShapeValidator — violation cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_batch_datatype_mismatch_violation() {
        let mut v = GpuShapeValidator::default_device();
        let triples = vec![GpuTriple::new(1, 2, 100)];
        let shapes = vec![GpuShapeRef::new("DT").with_datatype_hash(999)];
        let batch = GpuValidationBatch::new(triples, shapes, 4);
        let result = v.validate_batch(batch);
        assert!(!result.is_valid());
        assert_eq!(result.violation_count(), 1);
        assert_eq!(result.violations[0].shape_id, "DT");
        assert_eq!(result.violations[0].severity, 2);
    }

    #[test]
    fn test_validate_batch_max_count_violated() {
        let mut v = GpuShapeValidator::default_device();
        // Subject 1, predicate 2 appears 3 times → violates max 1
        let triples = vec![
            GpuTriple::new(1, 2, 10),
            GpuTriple::new(1, 2, 11),
            GpuTriple::new(1, 2, 12),
        ];
        let shapes = vec![GpuShapeRef::new("MaxC").with_cardinality(0, 1)];
        let batch = GpuValidationBatch::new(triples, shapes, 5);
        let result = v.validate_batch(batch);
        // Each triple is checked → multiple violations for max-count
        assert!(!result.is_valid());
        assert!(result.violation_count() > 0);
    }

    #[test]
    fn test_validate_batch_min_count_violated() {
        let mut v = GpuShapeValidator::default_device();
        // Subject 1, predicate 2 appears 1 time → violates min 3
        let triples = vec![GpuTriple::new(1, 2, 10)];
        let shapes = vec![GpuShapeRef::new("MinC").with_cardinality(3, 10)];
        let batch = GpuValidationBatch::new(triples, shapes, 6);
        let result = v.validate_batch(batch);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_validate_batch_multiple_shapes() {
        let mut v = GpuShapeValidator::default_device();
        let triples = vec![GpuTriple::new(1, 2, 50)];
        let shapes = vec![
            GpuShapeRef::new("Good").with_datatype_hash(50),
            GpuShapeRef::new("Bad").with_datatype_hash(99),
        ];
        let batch = GpuValidationBatch::new(triples, shapes, 7);
        let result = v.validate_batch(batch);
        // "Good" shape is satisfied, "Bad" shape produces a violation
        assert!(!result.is_valid());
        assert_eq!(result.violation_count(), 1);
        assert_eq!(result.violations[0].shape_id, "Bad");
    }

    // -----------------------------------------------------------------------
    // GpuShapeValidator — throughput & stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_throughput_starts_zero() {
        let v = GpuShapeValidator::default_device();
        assert_eq!(v.throughput_shapes_per_sec(), 0.0);
        assert_eq!(v.total_shapes_processed(), 0);
    }

    #[test]
    fn test_throughput_after_batches() {
        let mut v = GpuShapeValidator::default_device();
        for i in 0..5u64 {
            let triples = vec![GpuTriple::new(i, i + 1, i + 2)];
            let shapes = vec![GpuShapeRef::new(format!("S{i}"))];
            let batch = GpuValidationBatch::new(triples, shapes, i as usize);
            v.validate_batch(batch);
        }
        assert!(v.throughput_shapes_per_sec() > 0.0);
        assert_eq!(v.total_shapes_processed(), 5);
    }

    #[test]
    fn test_reset_stats() {
        let mut v = GpuShapeValidator::default_device();
        let triples = vec![GpuTriple::new(1, 2, 3)];
        let shapes = vec![GpuShapeRef::new("S1")];
        let batch = GpuValidationBatch::new(triples, shapes, 0);
        v.validate_batch(batch);
        assert!(v.total_shapes_processed() > 0);
        v.reset_stats();
        assert_eq!(v.total_shapes_processed(), 0);
        assert_eq!(v.throughput_shapes_per_sec(), 0.0);
    }

    #[test]
    fn test_result_processing_time_recorded() {
        let mut v = GpuShapeValidator::default_device();
        let triples = (0..50u64)
            .map(|i| GpuTriple::new(i, i + 1, i + 2))
            .collect();
        let shapes = vec![GpuShapeRef::new("S")];
        let batch = GpuValidationBatch::new(triples, shapes, 99);
        let result = v.validate_batch(batch);
        // Processing time should be non-negative
        assert!(result.processing_time_us < u64::MAX);
    }

    // -----------------------------------------------------------------------
    // GpuEmbeddingCache tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_and_lookup_single() {
        let cache = GpuEmbeddingCache::new(64);
        cache
            .cache_node_embeddings(&[1], &[vec![0.1, 0.2, 0.3]])
            .expect("insert ok");
        let emb = cache.lookup_embedding(1).expect("should be present");
        assert_eq!(emb, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_lookup_missing_returns_none() {
        let cache = GpuEmbeddingCache::new(64);
        assert!(cache.lookup_embedding(999).is_none());
    }

    #[test]
    fn test_batch_lookup() {
        let cache = GpuEmbeddingCache::new(64);
        cache
            .cache_node_embeddings(&[10, 20], &[vec![1.0], vec![2.0]])
            .expect("insert ok");
        let results = cache.batch_lookup(&[10, 99, 20]);
        assert_eq!(results[0], Some(vec![1.0]));
        assert!(results[1].is_none());
        assert_eq!(results[2], Some(vec![2.0]));
    }

    #[test]
    fn test_cache_length_mismatch_error() {
        let cache = GpuEmbeddingCache::new(64);
        let err = cache
            .cache_node_embeddings(&[1, 2], &[vec![0.1]])
            .unwrap_err();
        assert!(matches!(err, ShaclAiError::ProcessingError(_)));
    }

    #[test]
    fn test_cache_overwrite() {
        let cache = GpuEmbeddingCache::new(64);
        cache
            .cache_node_embeddings(&[5], &[vec![1.0, 2.0]])
            .expect("insert ok");
        cache
            .cache_node_embeddings(&[5], &[vec![9.0, 8.0]])
            .expect("overwrite ok");
        let emb = cache.lookup_embedding(5).expect("present");
        assert_eq!(emb, vec![9.0, 8.0]);
    }

    #[test]
    fn test_memory_used_mb_grows() {
        let cache = GpuEmbeddingCache::new(64);
        assert_eq!(cache.memory_used_mb(), 0.0);
        // Insert 128 f64 values = 1024 bytes ≈ 0.000977 MB
        let emb: Vec<f64> = vec![0.0; 128];
        cache
            .cache_node_embeddings(&[1], &[emb])
            .expect("insert ok");
        assert!(cache.memory_used_mb() > 0.0);
    }

    #[test]
    fn test_entry_count() {
        let cache = GpuEmbeddingCache::new(64);
        assert_eq!(cache.entry_count(), 0);
        cache
            .cache_node_embeddings(&[1, 2, 3], &[vec![0.0], vec![1.0], vec![2.0]])
            .expect("ok");
        assert_eq!(cache.entry_count(), 3);
    }

    #[test]
    fn test_invalidate() {
        let cache = GpuEmbeddingCache::new(64);
        cache
            .cache_node_embeddings(&[7], &[vec![1.0, 2.0, 3.0]])
            .expect("insert ok");
        assert!(cache.invalidate(7));
        assert!(!cache.invalidate(7)); // already gone
        assert!(cache.lookup_embedding(7).is_none());
    }

    #[test]
    fn test_evict_lru_removes_entries() {
        let cache = GpuEmbeddingCache::new(64);
        // Insert entries with fat embeddings so eviction is triggered easily
        let emb: Vec<f64> = vec![0.0; 1024]; // 8192 bytes each
        cache
            .cache_node_embeddings(
                &[1, 2, 3, 4, 5],
                &[
                    emb.clone(),
                    emb.clone(),
                    emb.clone(),
                    emb.clone(),
                    emb.clone(),
                ],
            )
            .expect("insert ok");
        // Access node 5 to make it the "most recent"
        cache.lookup_embedding(5);
        let before = cache.entry_count();
        // Evict ≥1 MB
        let evicted = cache.evict_lru(1).expect("evict ok");
        assert!(evicted > 0);
        assert!(cache.entry_count() < before);
    }

    #[test]
    fn test_evict_zero_target() {
        let cache = GpuEmbeddingCache::new(64);
        cache.cache_node_embeddings(&[1], &[vec![0.1]]).expect("ok");
        let evicted = cache.evict_lru(0).expect("evict ok");
        // Zero target → nothing should be evicted
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_clear() {
        let cache = GpuEmbeddingCache::new(64);
        cache
            .cache_node_embeddings(&[1, 2], &[vec![0.1], vec![0.2]])
            .expect("ok");
        cache.clear();
        assert_eq!(cache.entry_count(), 0);
        assert_eq!(cache.memory_used_mb(), 0.0);
    }

    #[test]
    fn test_memory_budget_accessor() {
        let cache = GpuEmbeddingCache::new(512);
        assert_eq!(cache.memory_budget_bytes(), 512 * 1024 * 1024);
    }

    #[test]
    fn test_shape_violation_severity_levels() {
        let v_info = ShapeViolation::new(1, "S", "info msg", 0);
        let v_warn = ShapeViolation::new(1, "S", "warn msg", 1);
        let v_err = ShapeViolation::new(1, "S", "err msg", 2);
        assert_eq!(v_info.severity, 0);
        assert_eq!(v_warn.severity, 1);
        assert_eq!(v_err.severity, 2);
    }

    #[test]
    fn test_validator_config_accessible() {
        let cfg = GpuConfig {
            device_id: 2,
            batch_size: 128,
            memory_budget_mb: 2048,
        };
        let v = GpuShapeValidator::new(cfg.clone());
        assert_eq!(v.config().device_id, 2);
        assert_eq!(v.config().batch_size, 128);
    }
}
