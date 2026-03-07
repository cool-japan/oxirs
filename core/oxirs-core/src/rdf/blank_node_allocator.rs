//! Blank node allocation and scoping for RDF graphs.
//!
//! Provides deterministic blank node ID generation, scope-based allocation,
//! skolemization/deskolemization, and isomorphic blank node mapping.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Errors that can occur during blank node operations.
#[derive(Debug, Clone, PartialEq)]
pub enum BlankNodeError {
    /// The blank node ID is empty.
    EmptyId,
    /// The prefix contains invalid characters.
    InvalidPrefix(String),
    /// The scope does not exist.
    UnknownScope(String),
    /// Skolemization base URI is invalid.
    InvalidBaseUri(String),
    /// A mapping conflict was detected.
    MappingConflict(String),
    /// The allocator limit was exceeded.
    LimitExceeded(usize),
}

impl fmt::Display for BlankNodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlankNodeError::EmptyId => write!(f, "Blank node ID cannot be empty"),
            BlankNodeError::InvalidPrefix(p) => write!(f, "Invalid prefix: {p}"),
            BlankNodeError::UnknownScope(s) => write!(f, "Unknown scope: {s}"),
            BlankNodeError::InvalidBaseUri(u) => write!(f, "Invalid base URI: {u}"),
            BlankNodeError::MappingConflict(msg) => write!(f, "Mapping conflict: {msg}"),
            BlankNodeError::LimitExceeded(n) => write!(f, "Allocator limit exceeded: {n}"),
        }
    }
}

impl std::error::Error for BlankNodeError {}

/// A generated blank node identifier with scope information.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlankNodeId {
    /// The full blank node identifier (e.g., `"b0"`, `"doc1_b5"`).
    pub id: String,
    /// The scope this blank node belongs to, if any.
    pub scope: Option<String>,
}

impl BlankNodeId {
    /// Create a new blank node ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            scope: None,
        }
    }

    /// Create a blank node ID within a specific scope.
    pub fn with_scope(id: impl Into<String>, scope: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            scope: Some(scope.into()),
        }
    }

    /// Return the `_:id` form suitable for N-Triples/Turtle serialization.
    pub fn to_ntriples(&self) -> String {
        format!("_:{}", self.id)
    }
}

impl fmt::Display for BlankNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_:{}", self.id)
    }
}

/// Configuration for the blank node allocator.
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    /// Default prefix for generated IDs (e.g., `"b"`).
    pub default_prefix: String,
    /// Maximum number of blank nodes before an error is raised (0 = unlimited).
    pub max_allocations: usize,
    /// Base URI for skolemization (e.g., `"https://example.org/.well-known/genid/"`).
    pub skolem_base: String,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            default_prefix: "b".to_string(),
            max_allocations: 0,
            skolem_base: "https://example.org/.well-known/genid/".to_string(),
        }
    }
}

/// Tracks allocation counts per scope.
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Total blank nodes allocated across all scopes.
    pub total_allocated: u64,
    /// Blank nodes allocated per scope.
    pub per_scope: HashMap<String, u64>,
    /// Number of active scopes.
    pub active_scopes: usize,
    /// Number of skolemized nodes.
    pub skolemized: u64,
    /// Number of deskolemized nodes.
    pub deskolemized: u64,
}

/// Thread-safe blank node allocator with scope support.
///
/// Uses an atomic counter for fast, lock-free ID generation.
pub struct BlankNodeAllocator {
    /// Atomic counter for deterministic, lock-free ID generation.
    counter: Arc<AtomicU64>,
    /// Per-scope counters (scope_name -> next_id).
    scope_counters: HashMap<String, u64>,
    /// Known allocated IDs per scope for tracking.
    scope_ids: HashMap<String, Vec<String>>,
    /// Allocator configuration.
    config: AllocatorConfig,
    /// Cumulative statistics.
    stats: AllocationStats,
}

impl BlankNodeAllocator {
    /// Create a new allocator with default settings.
    pub fn new() -> Self {
        Self {
            counter: Arc::new(AtomicU64::new(0)),
            scope_counters: HashMap::new(),
            scope_ids: HashMap::new(),
            config: AllocatorConfig::default(),
            stats: AllocationStats::default(),
        }
    }

    /// Create an allocator with explicit configuration.
    pub fn with_config(config: AllocatorConfig) -> Result<Self, BlankNodeError> {
        validate_prefix(&config.default_prefix)?;
        if !config.skolem_base.is_empty() && !config.skolem_base.starts_with("http") {
            return Err(BlankNodeError::InvalidBaseUri(config.skolem_base.clone()));
        }
        Ok(Self {
            counter: Arc::new(AtomicU64::new(0)),
            scope_counters: HashMap::new(),
            scope_ids: HashMap::new(),
            config,
            stats: AllocationStats::default(),
        })
    }

    // --- Deterministic ID generation ---

    /// Allocate the next blank node using the global counter.
    pub fn next(&self) -> Result<BlankNodeId, BlankNodeError> {
        let max = self.config.max_allocations;
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        if max > 0 && (n as usize) >= max {
            // Roll back the counter
            self.counter.fetch_sub(1, Ordering::Relaxed);
            return Err(BlankNodeError::LimitExceeded(max));
        }
        Ok(BlankNodeId::new(format!(
            "{}{n}",
            self.config.default_prefix
        )))
    }

    /// Allocate a blank node with a custom prefix.
    pub fn next_with_prefix(&self, prefix: &str) -> Result<BlankNodeId, BlankNodeError> {
        validate_prefix(prefix)?;
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let max = self.config.max_allocations;
        if max > 0 && (n as usize) >= max {
            self.counter.fetch_sub(1, Ordering::Relaxed);
            return Err(BlankNodeError::LimitExceeded(max));
        }
        Ok(BlankNodeId::new(format!("{prefix}{n}")))
    }

    /// Return the current counter value (next ID that *will* be generated).
    pub fn current_counter(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Reset the global counter to zero.
    pub fn reset_counter(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    // --- Scope-based allocation ---

    /// Create a new scope for blank node allocation.
    pub fn create_scope(&mut self, name: impl Into<String>) -> Result<(), BlankNodeError> {
        let name = name.into();
        if name.is_empty() {
            return Err(BlankNodeError::EmptyId);
        }
        self.scope_counters.entry(name.clone()).or_insert(0);
        self.scope_ids.entry(name.clone()).or_default();
        self.stats.active_scopes = self.scope_counters.len();
        Ok(())
    }

    /// Allocate a blank node within a named scope.
    pub fn next_in_scope(&mut self, scope: &str) -> Result<BlankNodeId, BlankNodeError> {
        let scope_counter = self
            .scope_counters
            .get_mut(scope)
            .ok_or_else(|| BlankNodeError::UnknownScope(scope.to_string()))?;
        let n = *scope_counter;
        *scope_counter += 1;
        let id_str = format!("{}_{}{n}", scope, self.config.default_prefix);
        let blank = BlankNodeId::with_scope(id_str.clone(), scope);
        if let Some(ids) = self.scope_ids.get_mut(scope) {
            ids.push(id_str);
        }
        self.stats.total_allocated += 1;
        *self.stats.per_scope.entry(scope.to_string()).or_insert(0) += 1;
        Ok(blank)
    }

    /// List all blank node IDs allocated within a scope.
    pub fn ids_in_scope(&self, scope: &str) -> Result<&[String], BlankNodeError> {
        self.scope_ids
            .get(scope)
            .map(|v| v.as_slice())
            .ok_or_else(|| BlankNodeError::UnknownScope(scope.to_string()))
    }

    /// Remove a scope and all its tracking data.
    pub fn remove_scope(&mut self, scope: &str) -> Result<(), BlankNodeError> {
        if !self.scope_counters.contains_key(scope) {
            return Err(BlankNodeError::UnknownScope(scope.to_string()));
        }
        self.scope_counters.remove(scope);
        self.scope_ids.remove(scope);
        self.stats.active_scopes = self.scope_counters.len();
        Ok(())
    }

    /// Return the list of active scope names.
    pub fn active_scopes(&self) -> Vec<String> {
        self.scope_counters.keys().cloned().collect()
    }

    // --- Skolemization ---

    /// Skolemize a blank node ID, replacing it with a well-known IRI.
    ///
    /// E.g., `_:b0` -> `<https://example.org/.well-known/genid/b0>`
    pub fn skolemize(&mut self, blank_id: &str) -> Result<String, BlankNodeError> {
        if blank_id.is_empty() {
            return Err(BlankNodeError::EmptyId);
        }
        if self.config.skolem_base.is_empty() {
            return Err(BlankNodeError::InvalidBaseUri(
                "Skolem base URI is not configured".to_string(),
            ));
        }
        self.stats.skolemized += 1;
        Ok(format!("{}{blank_id}", self.config.skolem_base))
    }

    /// Skolemize a blank node ID with a custom base URI.
    pub fn skolemize_with_base(
        &mut self,
        blank_id: &str,
        base: &str,
    ) -> Result<String, BlankNodeError> {
        if blank_id.is_empty() {
            return Err(BlankNodeError::EmptyId);
        }
        if base.is_empty() || !base.starts_with("http") {
            return Err(BlankNodeError::InvalidBaseUri(base.to_string()));
        }
        self.stats.skolemized += 1;
        let sep = if base.ends_with('/') { "" } else { "/" };
        Ok(format!("{base}{sep}{blank_id}"))
    }

    /// Deskolemize a well-known IRI back to a blank node ID.
    ///
    /// E.g., `<https://example.org/.well-known/genid/b0>` -> `_:b0`
    pub fn deskolemize(&mut self, iri: &str) -> Result<BlankNodeId, BlankNodeError> {
        let stripped = iri
            .strip_prefix('<')
            .unwrap_or(iri)
            .strip_suffix('>')
            .unwrap_or(iri);
        if let Some(local) = stripped.strip_prefix(self.config.skolem_base.as_str()) {
            if local.is_empty() {
                return Err(BlankNodeError::EmptyId);
            }
            self.stats.deskolemized += 1;
            return Ok(BlankNodeId::new(local));
        }
        Err(BlankNodeError::InvalidBaseUri(format!(
            "IRI does not match skolem base: {iri}"
        )))
    }

    /// Deskolemize using a custom base.
    pub fn deskolemize_with_base(
        &mut self,
        iri: &str,
        base: &str,
    ) -> Result<BlankNodeId, BlankNodeError> {
        let stripped = iri
            .strip_prefix('<')
            .unwrap_or(iri)
            .strip_suffix('>')
            .unwrap_or(iri);
        if let Some(local) = stripped.strip_prefix(base) {
            let local = local.strip_prefix('/').unwrap_or(local);
            if local.is_empty() {
                return Err(BlankNodeError::EmptyId);
            }
            self.stats.deskolemized += 1;
            return Ok(BlankNodeId::new(local));
        }
        Err(BlankNodeError::InvalidBaseUri(format!(
            "IRI does not match base: {iri}"
        )))
    }

    // --- Blank node mapping ---

    /// Discover an isomorphic mapping between blank nodes of two graphs.
    ///
    /// `source` and `target` are lists of blank node IDs from each graph.
    /// Returns a bijective mapping from source IDs to target IDs if sizes match,
    /// or an error if they are incompatible.
    pub fn discover_mapping(
        source: &[&str],
        target: &[&str],
    ) -> Result<HashMap<String, String>, BlankNodeError> {
        if source.len() != target.len() {
            return Err(BlankNodeError::MappingConflict(format!(
                "Source has {} blank nodes but target has {}",
                source.len(),
                target.len()
            )));
        }
        let mut mapping = HashMap::new();
        let mut used_targets: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (i, s) in source.iter().enumerate() {
            let t = target
                .get(i)
                .ok_or_else(|| BlankNodeError::MappingConflict("Index out of bounds".into()))?;
            if used_targets.contains(*t) {
                return Err(BlankNodeError::MappingConflict(format!(
                    "Target blank node '{t}' is already mapped"
                )));
            }
            mapping.insert(s.to_string(), t.to_string());
            used_targets.insert(t.to_string());
        }
        Ok(mapping)
    }

    /// Apply a mapping to a set of blank node IDs, returning the renamed IDs.
    pub fn apply_mapping(ids: &[&str], mapping: &HashMap<String, String>) -> Vec<String> {
        ids.iter()
            .map(|id| mapping.get(*id).cloned().unwrap_or_else(|| id.to_string()))
            .collect()
    }

    /// Verify that a mapping is bijective (no duplicate targets).
    pub fn verify_mapping(mapping: &HashMap<String, String>) -> Result<(), BlankNodeError> {
        let mut seen = std::collections::HashSet::new();
        for (k, v) in mapping {
            if !seen.insert(v.clone()) {
                return Err(BlankNodeError::MappingConflict(format!(
                    "Duplicate target '{v}' for source '{k}'"
                )));
            }
        }
        Ok(())
    }

    // --- Blank node renaming ---

    /// Rename all blank nodes in a list using a new prefix.
    ///
    /// E.g., `["b0", "b1"]` with prefix `"merged_"` -> `["merged_0", "merged_1"]`.
    pub fn rename_with_prefix(
        ids: &[&str],
        new_prefix: &str,
    ) -> Result<Vec<BlankNodeId>, BlankNodeError> {
        validate_prefix(new_prefix)?;
        let mut result = Vec::with_capacity(ids.len());
        for (i, _) in ids.iter().enumerate() {
            result.push(BlankNodeId::new(format!("{new_prefix}{i}")));
        }
        Ok(result)
    }

    /// Rename blank nodes using a mapping from old to new prefix, preserving
    /// numeric suffixes.
    pub fn rename_prefix(
        ids: &[&str],
        old_prefix: &str,
        new_prefix: &str,
    ) -> Result<Vec<BlankNodeId>, BlankNodeError> {
        validate_prefix(new_prefix)?;
        let mut result = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(suffix) = id.strip_prefix(old_prefix) {
                result.push(BlankNodeId::new(format!("{new_prefix}{suffix}")));
            } else {
                result.push(BlankNodeId::new(id.to_string()));
            }
        }
        Ok(result)
    }

    /// Generate a merge-safe renaming: concatenate `scope` prefix to each ID.
    pub fn scope_rename(ids: &[&str], scope: &str) -> Result<Vec<BlankNodeId>, BlankNodeError> {
        if scope.is_empty() {
            return Err(BlankNodeError::EmptyId);
        }
        let mut result = Vec::with_capacity(ids.len());
        for id in ids {
            result.push(BlankNodeId::with_scope(format!("{scope}_{id}"), scope));
        }
        Ok(result)
    }

    // --- Statistics ---

    /// Return current allocation statistics.
    pub fn stats(&self) -> &AllocationStats {
        &self.stats
    }

    /// Return the configuration.
    pub fn config(&self) -> &AllocatorConfig {
        &self.config
    }

    /// Get an atomic clone of the counter for sharing across threads.
    pub fn shared_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.counter)
    }
}

impl Default for BlankNodeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate that a prefix contains only alphanumeric characters and underscores.
fn validate_prefix(prefix: &str) -> Result<(), BlankNodeError> {
    if prefix.is_empty() {
        return Err(BlankNodeError::InvalidPrefix(
            "prefix must not be empty".to_string(),
        ));
    }
    if !prefix
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(BlankNodeError::InvalidPrefix(prefix.to_string()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Deterministic counter-based generation --

    #[test]
    fn test_next_generates_sequential_ids() {
        let alloc = BlankNodeAllocator::new();
        let a = alloc.next().expect("first");
        let b = alloc.next().expect("second");
        assert_eq!(a.id, "b0");
        assert_eq!(b.id, "b1");
    }

    #[test]
    fn test_next_with_prefix() {
        let alloc = BlankNodeAllocator::new();
        let id = alloc.next_with_prefix("node").expect("prefix");
        assert!(id.id.starts_with("node"));
    }

    #[test]
    fn test_counter_reset() {
        let alloc = BlankNodeAllocator::new();
        let _ = alloc.next();
        let _ = alloc.next();
        assert_eq!(alloc.current_counter(), 2);
        alloc.reset_counter();
        assert_eq!(alloc.current_counter(), 0);
        let id = alloc.next().expect("after reset");
        assert_eq!(id.id, "b0");
    }

    #[test]
    fn test_to_ntriples() {
        let id = BlankNodeId::new("x42");
        assert_eq!(id.to_ntriples(), "_:x42");
    }

    #[test]
    fn test_display_trait() {
        let id = BlankNodeId::new("abc");
        assert_eq!(format!("{id}"), "_:abc");
    }

    // -- Configuration --

    #[test]
    fn test_with_config_custom_prefix() {
        let cfg = AllocatorConfig {
            default_prefix: "node".to_string(),
            ..Default::default()
        };
        let alloc = BlankNodeAllocator::with_config(cfg).expect("cfg");
        let id = alloc.next().expect("next");
        assert_eq!(id.id, "node0");
    }

    #[test]
    fn test_max_allocations_limit() {
        let cfg = AllocatorConfig {
            max_allocations: 2,
            ..Default::default()
        };
        let alloc = BlankNodeAllocator::with_config(cfg).expect("cfg");
        assert!(alloc.next().is_ok());
        assert!(alloc.next().is_ok());
        let err = alloc.next().expect_err("should exceed limit");
        assert!(matches!(err, BlankNodeError::LimitExceeded(2)));
    }

    #[test]
    fn test_invalid_prefix_rejected() {
        let cfg = AllocatorConfig {
            default_prefix: "no-dash".to_string(),
            ..Default::default()
        };
        let result = BlankNodeAllocator::with_config(cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_skolem_base_rejected() {
        let cfg = AllocatorConfig {
            skolem_base: "ftp://bad".to_string(),
            ..Default::default()
        };
        let result = BlankNodeAllocator::with_config(cfg);
        assert!(result.is_err());
    }

    // -- Scope-based allocation --

    #[test]
    fn test_create_and_use_scope() {
        let mut alloc = BlankNodeAllocator::new();
        alloc.create_scope("doc1").expect("create");
        let id = alloc.next_in_scope("doc1").expect("scoped");
        assert!(id.id.starts_with("doc1_"));
        assert_eq!(id.scope.as_deref(), Some("doc1"));
    }

    #[test]
    fn test_scope_counter_independent() {
        let mut alloc = BlankNodeAllocator::new();
        alloc.create_scope("a").expect("a");
        alloc.create_scope("b").expect("b");
        let a0 = alloc.next_in_scope("a").expect("a0");
        let b0 = alloc.next_in_scope("b").expect("b0");
        let a1 = alloc.next_in_scope("a").expect("a1");
        assert_eq!(a0.id, "a_b0");
        assert_eq!(b0.id, "b_b0");
        assert_eq!(a1.id, "a_b1");
    }

    #[test]
    fn test_unknown_scope_error() {
        let mut alloc = BlankNodeAllocator::new();
        let err = alloc.next_in_scope("missing").expect_err("unknown scope");
        assert!(matches!(err, BlankNodeError::UnknownScope(_)));
    }

    #[test]
    fn test_ids_in_scope() {
        let mut alloc = BlankNodeAllocator::new();
        alloc.create_scope("s").expect("create");
        alloc.next_in_scope("s").expect("s0");
        alloc.next_in_scope("s").expect("s1");
        let ids = alloc.ids_in_scope("s").expect("list");
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_remove_scope() {
        let mut alloc = BlankNodeAllocator::new();
        alloc.create_scope("tmp").expect("create");
        alloc.next_in_scope("tmp").expect("n");
        alloc.remove_scope("tmp").expect("remove");
        assert!(!alloc.active_scopes().contains(&"tmp".to_string()));
    }

    #[test]
    fn test_remove_unknown_scope_errors() {
        let mut alloc = BlankNodeAllocator::new();
        let err = alloc.remove_scope("nope").expect_err("remove unknown");
        assert!(matches!(err, BlankNodeError::UnknownScope(_)));
    }

    #[test]
    fn test_create_scope_empty_name_fails() {
        let mut alloc = BlankNodeAllocator::new();
        let err = alloc.create_scope("").expect_err("empty scope");
        assert!(matches!(err, BlankNodeError::EmptyId));
    }

    #[test]
    fn test_active_scopes_list() {
        let mut alloc = BlankNodeAllocator::new();
        alloc.create_scope("x").expect("x");
        alloc.create_scope("y").expect("y");
        let scopes = alloc.active_scopes();
        assert!(scopes.contains(&"x".to_string()));
        assert!(scopes.contains(&"y".to_string()));
    }

    // -- Skolemization --

    #[test]
    fn test_skolemize_default_base() {
        let mut alloc = BlankNodeAllocator::new();
        let iri = alloc.skolemize("b0").expect("skolemize");
        assert_eq!(iri, "https://example.org/.well-known/genid/b0");
    }

    #[test]
    fn test_skolemize_custom_base() {
        let mut alloc = BlankNodeAllocator::new();
        let iri = alloc
            .skolemize_with_base("n1", "https://data.org/genid/")
            .expect("custom");
        assert_eq!(iri, "https://data.org/genid/n1");
    }

    #[test]
    fn test_skolemize_custom_base_no_trailing_slash() {
        let mut alloc = BlankNodeAllocator::new();
        let iri = alloc
            .skolemize_with_base("n1", "https://data.org/genid")
            .expect("no slash");
        assert_eq!(iri, "https://data.org/genid/n1");
    }

    #[test]
    fn test_skolemize_empty_id_fails() {
        let mut alloc = BlankNodeAllocator::new();
        let err = alloc.skolemize("").expect_err("empty");
        assert!(matches!(err, BlankNodeError::EmptyId));
    }

    #[test]
    fn test_skolemize_no_base_configured() {
        let cfg = AllocatorConfig {
            skolem_base: String::new(),
            ..Default::default()
        };
        let mut alloc = BlankNodeAllocator::with_config(cfg).expect("cfg");
        let err = alloc.skolemize("b0").expect_err("no base");
        assert!(matches!(err, BlankNodeError::InvalidBaseUri(_)));
    }

    // -- Deskolemization --

    #[test]
    fn test_deskolemize_default_base() {
        let mut alloc = BlankNodeAllocator::new();
        let bn = alloc
            .deskolemize("https://example.org/.well-known/genid/b42")
            .expect("de");
        assert_eq!(bn.id, "b42");
    }

    #[test]
    fn test_deskolemize_with_angle_brackets() {
        let mut alloc = BlankNodeAllocator::new();
        let bn = alloc
            .deskolemize("<https://example.org/.well-known/genid/n7>")
            .expect("de");
        assert_eq!(bn.id, "n7");
    }

    #[test]
    fn test_deskolemize_wrong_base_fails() {
        let mut alloc = BlankNodeAllocator::new();
        let err = alloc
            .deskolemize("https://other.org/genid/b0")
            .expect_err("wrong base");
        assert!(matches!(err, BlankNodeError::InvalidBaseUri(_)));
    }

    #[test]
    fn test_deskolemize_with_custom_base() {
        let mut alloc = BlankNodeAllocator::new();
        let bn = alloc
            .deskolemize_with_base("https://data.org/genid/x1", "https://data.org/genid/")
            .expect("custom de");
        assert_eq!(bn.id, "x1");
    }

    #[test]
    fn test_roundtrip_skolemize_deskolemize() {
        let mut alloc = BlankNodeAllocator::new();
        let iri = alloc.skolemize("abc123").expect("sk");
        let bn = alloc.deskolemize(&iri).expect("de");
        assert_eq!(bn.id, "abc123");
    }

    // -- Mapping --

    #[test]
    fn test_discover_mapping_equal_sizes() {
        let source = vec!["b0", "b1", "b2"];
        let target = vec!["x0", "x1", "x2"];
        let map = BlankNodeAllocator::discover_mapping(&source, &target).expect("map");
        assert_eq!(map.get("b0"), Some(&"x0".to_string()));
        assert_eq!(map.get("b2"), Some(&"x2".to_string()));
    }

    #[test]
    fn test_discover_mapping_size_mismatch() {
        let source = vec!["b0", "b1"];
        let target = vec!["x0"];
        let err = BlankNodeAllocator::discover_mapping(&source, &target).expect_err("mismatch");
        assert!(matches!(err, BlankNodeError::MappingConflict(_)));
    }

    #[test]
    fn test_discover_mapping_duplicate_target() {
        let source = vec!["b0", "b1"];
        let target = vec!["x0", "x0"];
        let err = BlankNodeAllocator::discover_mapping(&source, &target).expect_err("dup");
        assert!(matches!(err, BlankNodeError::MappingConflict(_)));
    }

    #[test]
    fn test_apply_mapping() {
        let mut mapping = HashMap::new();
        mapping.insert("b0".to_string(), "x0".to_string());
        mapping.insert("b1".to_string(), "x1".to_string());
        let result = BlankNodeAllocator::apply_mapping(&["b0", "b1", "b2"], &mapping);
        assert_eq!(result, vec!["x0", "x1", "b2"]);
    }

    #[test]
    fn test_verify_mapping_valid() {
        let mut mapping = HashMap::new();
        mapping.insert("a".to_string(), "x".to_string());
        mapping.insert("b".to_string(), "y".to_string());
        assert!(BlankNodeAllocator::verify_mapping(&mapping).is_ok());
    }

    #[test]
    fn test_verify_mapping_duplicate_values() {
        let mut mapping = HashMap::new();
        mapping.insert("a".to_string(), "x".to_string());
        mapping.insert("b".to_string(), "x".to_string());
        let err = BlankNodeAllocator::verify_mapping(&mapping).expect_err("dup");
        assert!(matches!(err, BlankNodeError::MappingConflict(_)));
    }

    // -- Renaming --

    #[test]
    fn test_rename_with_prefix() {
        let ids = vec!["b0", "b1", "b2"];
        let result = BlankNodeAllocator::rename_with_prefix(&ids, "merged_").expect("rename");
        assert_eq!(result[0].id, "merged_0");
        assert_eq!(result[1].id, "merged_1");
        assert_eq!(result[2].id, "merged_2");
    }

    #[test]
    fn test_rename_prefix_swap() {
        let ids = vec!["old_0", "old_1"];
        let result = BlankNodeAllocator::rename_prefix(&ids, "old_", "new_").expect("swap");
        assert_eq!(result[0].id, "new_0");
        assert_eq!(result[1].id, "new_1");
    }

    #[test]
    fn test_rename_prefix_no_match_keeps_original() {
        let ids = vec!["other_0"];
        let result = BlankNodeAllocator::rename_prefix(&ids, "old_", "new_").expect("no match");
        assert_eq!(result[0].id, "other_0");
    }

    #[test]
    fn test_scope_rename() {
        let ids = vec!["b0", "b1"];
        let result = BlankNodeAllocator::scope_rename(&ids, "graph1").expect("scope rename");
        assert_eq!(result[0].id, "graph1_b0");
        assert_eq!(result[0].scope, Some("graph1".to_string()));
        assert_eq!(result[1].id, "graph1_b1");
    }

    #[test]
    fn test_scope_rename_empty_scope_fails() {
        let ids = vec!["b0"];
        let err = BlankNodeAllocator::scope_rename(&ids, "").expect_err("empty");
        assert!(matches!(err, BlankNodeError::EmptyId));
    }

    #[test]
    fn test_rename_invalid_prefix_fails() {
        let ids = vec!["b0"];
        let err = BlankNodeAllocator::rename_with_prefix(&ids, "bad-prefix").expect_err("invalid");
        assert!(matches!(err, BlankNodeError::InvalidPrefix(_)));
    }

    // -- Statistics --

    #[test]
    fn test_stats_after_scope_allocation() {
        let mut alloc = BlankNodeAllocator::new();
        alloc.create_scope("doc").expect("create");
        alloc.next_in_scope("doc").expect("n1");
        alloc.next_in_scope("doc").expect("n2");
        let stats = alloc.stats();
        assert_eq!(stats.total_allocated, 2);
        assert_eq!(stats.per_scope.get("doc"), Some(&2));
    }

    #[test]
    fn test_stats_skolemized_count() {
        let mut alloc = BlankNodeAllocator::new();
        alloc.skolemize("b0").expect("sk");
        alloc.skolemize("b1").expect("sk");
        assert_eq!(alloc.stats().skolemized, 2);
    }

    #[test]
    fn test_stats_deskolemized_count() {
        let mut alloc = BlankNodeAllocator::new();
        alloc
            .deskolemize("https://example.org/.well-known/genid/b0")
            .expect("de");
        assert_eq!(alloc.stats().deskolemized, 1);
    }

    // -- Thread safety --

    #[test]
    fn test_shared_counter_across_threads() {
        let alloc = BlankNodeAllocator::new();
        let counter = alloc.shared_counter();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = Arc::clone(&counter);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        c.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread join");
        }
        assert_eq!(counter.load(Ordering::Relaxed), 400);
    }

    #[test]
    fn test_blank_node_id_equality() {
        let a = BlankNodeId::new("x");
        let b = BlankNodeId::new("x");
        assert_eq!(a, b);
    }

    #[test]
    fn test_blank_node_id_with_scope_equality() {
        let a = BlankNodeId::with_scope("x0", "doc1");
        let b = BlankNodeId::with_scope("x0", "doc1");
        assert_eq!(a, b);
    }

    #[test]
    fn test_blank_node_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BlankNodeId::new("a"));
        set.insert(BlankNodeId::new("a"));
        set.insert(BlankNodeId::new("b"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_default_allocator() {
        let alloc = BlankNodeAllocator::default();
        let id = alloc.next().expect("default");
        assert_eq!(id.id, "b0");
    }

    #[test]
    fn test_config_accessor() {
        let alloc = BlankNodeAllocator::new();
        assert_eq!(alloc.config().default_prefix, "b");
    }

    #[test]
    fn test_empty_mapping_is_valid() {
        let map: HashMap<String, String> = HashMap::new();
        assert!(BlankNodeAllocator::verify_mapping(&map).is_ok());
    }

    #[test]
    fn test_discover_mapping_empty() {
        let source: Vec<&str> = vec![];
        let target: Vec<&str> = vec![];
        let map = BlankNodeAllocator::discover_mapping(&source, &target).expect("empty");
        assert!(map.is_empty());
    }
}
