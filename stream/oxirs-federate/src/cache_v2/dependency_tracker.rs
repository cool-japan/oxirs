//! Cache dependency graph for smart invalidation.
//!
//! Tracks which cache keys depend on which endpoints (data sources) and which
//! other cache keys (derived queries).  When an endpoint or a base key is
//! invalidated, the tracker returns the full transitive closure of keys that
//! must be evicted.
//!
//! # Design
//!
//! The dependency graph is a directed acyclic graph (DAG) where:
//! - **Nodes** are `CacheKey`s.
//! - **Edges** represent "depends on": `A → B` means key A depends on key B.
//! - Endpoints are modelled as special string identifiers (not keys).
//!
//! Invalidation is a BFS/DFS that starts from the invalidated node and
//! follows *reverse* edges (callers of the invalidated node).
//!
//! # Thread safety
//!
//! All public methods take `&mut self`.  Callers should wrap in `Mutex`.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

// Re-use CacheKey from the sibling module.
use crate::cache_v2::multi_level_cache::CacheKey;

// ─── DependencyEdge ───────────────────────────────────────────────────────────

/// The type of dependency a cache key can have.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyKind {
    /// This key directly depends on the listed endpoint's data.
    Endpoint(String),
    /// This key was derived from another cached key.
    DerivedFrom(CacheKey),
}

// ─── DependencyTracker ────────────────────────────────────────────────────────

/// Tracks cache key dependencies for smart invalidation.
pub struct DependencyTracker {
    /// Forward edges: key → set of things it depends on.
    forward: HashMap<CacheKey, HashSet<DependencyKind>>,
    /// Reverse index: endpoint_id → keys that depend on it.
    endpoint_reverse: HashMap<String, HashSet<CacheKey>>,
    /// Reverse index: parent_key → keys derived from it.
    derived_reverse: HashMap<CacheKey, HashSet<CacheKey>>,
}

impl DependencyTracker {
    /// Create an empty dependency tracker.
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            endpoint_reverse: HashMap::new(),
            derived_reverse: HashMap::new(),
        }
    }

    /// Declare that `key` depends on endpoint `endpoint_id`.
    pub fn add_endpoint_dep(&mut self, key: CacheKey, endpoint_id: impl Into<String>) {
        let ep = endpoint_id.into();
        self.forward
            .entry(key.clone())
            .or_default()
            .insert(DependencyKind::Endpoint(ep.clone()));
        self.endpoint_reverse.entry(ep).or_default().insert(key);
    }

    /// Declare that `child_key` was derived from `parent_key`.
    pub fn add_derived_dep(&mut self, child_key: CacheKey, parent_key: CacheKey) {
        self.forward
            .entry(child_key.clone())
            .or_default()
            .insert(DependencyKind::DerivedFrom(parent_key.clone()));
        self.derived_reverse
            .entry(parent_key)
            .or_default()
            .insert(child_key);
    }

    /// Remove all dependency records for a key (called when the key is evicted).
    pub fn remove_key(&mut self, key: &CacheKey) {
        if let Some(deps) = self.forward.remove(key) {
            for dep in deps {
                match dep {
                    DependencyKind::Endpoint(ep) => {
                        if let Some(set) = self.endpoint_reverse.get_mut(&ep) {
                            set.remove(key);
                        }
                    }
                    DependencyKind::DerivedFrom(parent) => {
                        if let Some(set) = self.derived_reverse.get_mut(&parent) {
                            set.remove(key);
                        }
                    }
                }
            }
        }
        // Also remove it as a parent in derived_reverse
        self.derived_reverse.remove(key);
    }

    /// Compute the set of keys that must be invalidated because `endpoint_id`
    /// changed.  Returns the **transitive closure** through derived edges.
    pub fn invalidate_endpoint(&mut self, endpoint_id: &str) -> Vec<CacheKey> {
        let seeds: Vec<CacheKey> = self
            .endpoint_reverse
            .get(endpoint_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let affected = self.transitive_derived(&seeds);

        // Clean up records for affected keys
        for key in &affected {
            self.remove_key(key);
        }

        affected
    }

    /// Compute the set of keys that must be invalidated because `base_key`
    /// changed or was evicted.  Returns the transitive closure of derived keys.
    pub fn invalidate_key(&mut self, base_key: &CacheKey) -> Vec<CacheKey> {
        let seeds = vec![base_key.clone()];
        let mut affected = self.transitive_derived(&seeds);
        // Include the base key itself
        if !affected.contains(base_key) {
            affected.push(base_key.clone());
        }
        for key in &affected {
            self.remove_key(key);
        }
        affected
    }

    /// Return all direct dependencies of a key (for inspection / debugging).
    pub fn dependencies_of(&self, key: &CacheKey) -> Vec<DependencyKind> {
        self.forward
            .get(key)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Return all keys that directly depend on `endpoint_id`.
    pub fn dependents_of_endpoint(&self, endpoint_id: &str) -> Vec<CacheKey> {
        self.endpoint_reverse
            .get(endpoint_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect()
    }

    /// Return all keys directly derived from `parent_key`.
    pub fn derived_from(&self, parent_key: &CacheKey) -> Vec<CacheKey> {
        self.derived_reverse
            .get(parent_key)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect()
    }

    /// Number of keys being tracked.
    pub fn tracked_keys(&self) -> usize {
        self.forward.len()
    }

    // ── Private ────────────────────────────────────────────────────────────────

    /// BFS to compute the transitive closure of keys reachable via derived
    /// edges, starting from `seeds`.
    fn transitive_derived(&self, seeds: &[CacheKey]) -> Vec<CacheKey> {
        let mut visited: HashSet<CacheKey> = HashSet::new();
        let mut queue: VecDeque<CacheKey> = seeds.iter().cloned().collect();

        while let Some(key) = queue.pop_front() {
            if visited.contains(&key) {
                continue;
            }
            visited.insert(key.clone());

            // Enqueue keys that were derived from this one
            if let Some(children) = self.derived_reverse.get(&key) {
                for child in children {
                    if !visited.contains(child) {
                        queue.push_back(child.clone());
                    }
                }
            }
        }

        visited.into_iter().collect()
    }
}

impl Default for DependencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn key(s: &str) -> CacheKey {
        CacheKey::from_str(s)
    }

    // ── add_endpoint_dep ──────────────────────────────────────────────────

    #[test]
    fn test_add_endpoint_dep_single() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("k1"), "ep-alpha");
        let deps = t.dependencies_of(&key("k1"));
        assert_eq!(deps.len(), 1);
        assert!(deps.contains(&DependencyKind::Endpoint("ep-alpha".to_string())));
    }

    #[test]
    fn test_add_endpoint_dep_multiple_keys_same_endpoint() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("k1"), "ep-shared");
        t.add_endpoint_dep(key("k2"), "ep-shared");
        let mut deps = t.dependents_of_endpoint("ep-shared");
        deps.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn test_add_derived_dep() {
        let mut t = DependencyTracker::new();
        t.add_derived_dep(key("child"), key("parent"));
        let deps = t.dependencies_of(&key("child"));
        assert!(deps.contains(&DependencyKind::DerivedFrom(key("parent"))));
        let derived = t.derived_from(&key("parent"));
        assert!(derived.contains(&key("child")));
    }

    // ── invalidate_endpoint ───────────────────────────────────────────────

    #[test]
    fn test_invalidate_endpoint_direct() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("k1"), "ep-x");
        t.add_endpoint_dep(key("k2"), "ep-y");

        let affected = t.invalidate_endpoint("ep-x");
        assert!(affected.contains(&key("k1")));
        assert!(!affected.contains(&key("k2")));
    }

    #[test]
    fn test_invalidate_endpoint_transitive() {
        let mut t = DependencyTracker::new();
        // k1 → ep-x
        t.add_endpoint_dep(key("k1"), "ep-x");
        // k2 derived from k1
        t.add_derived_dep(key("k2"), key("k1"));
        // k3 derived from k2
        t.add_derived_dep(key("k3"), key("k2"));

        let affected = t.invalidate_endpoint("ep-x");
        assert!(affected.contains(&key("k1")));
        assert!(affected.contains(&key("k2")));
        assert!(affected.contains(&key("k3")));
    }

    #[test]
    fn test_invalidate_endpoint_clears_records() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("k1"), "ep-z");
        let _ = t.invalidate_endpoint("ep-z");
        // After invalidation, k1 should no longer be tracked
        assert_eq!(t.tracked_keys(), 0);
    }

    #[test]
    fn test_invalidate_unknown_endpoint_returns_empty() {
        let mut t = DependencyTracker::new();
        let affected = t.invalidate_endpoint("non-existent");
        assert!(affected.is_empty());
    }

    // ── invalidate_key ────────────────────────────────────────────────────

    #[test]
    fn test_invalidate_key_includes_self() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("base"), "ep");
        let affected = t.invalidate_key(&key("base"));
        assert!(affected.contains(&key("base")));
    }

    #[test]
    fn test_invalidate_key_transitive() {
        let mut t = DependencyTracker::new();
        t.add_derived_dep(key("child1"), key("parent"));
        t.add_derived_dep(key("grandchild"), key("child1"));
        let affected = t.invalidate_key(&key("parent"));
        assert!(affected.contains(&key("parent")));
        assert!(affected.contains(&key("child1")));
        assert!(affected.contains(&key("grandchild")));
    }

    #[test]
    fn test_invalidate_key_does_not_affect_siblings() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("sib-a"), "ep");
        t.add_endpoint_dep(key("sib-b"), "ep");
        let _ = t.invalidate_key(&key("sib-a"));
        // sib-b should still be tracked
        let deps = t.dependencies_of(&key("sib-b"));
        assert!(!deps.is_empty());
    }

    // ── remove_key ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_key_cleans_up_forward() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("gone"), "ep");
        t.remove_key(&key("gone"));
        assert!(t.dependencies_of(&key("gone")).is_empty());
    }

    #[test]
    fn test_remove_key_cleans_up_reverse_index() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("gone"), "ep");
        t.remove_key(&key("gone"));
        assert!(t.dependents_of_endpoint("ep").is_empty());
    }

    // ── tracked_keys ──────────────────────────────────────────────────────

    #[test]
    fn test_tracked_keys_count() {
        let mut t = DependencyTracker::new();
        assert_eq!(t.tracked_keys(), 0);
        t.add_endpoint_dep(key("k1"), "ep");
        t.add_endpoint_dep(key("k2"), "ep");
        assert_eq!(t.tracked_keys(), 2);
    }

    // ── cycle safety ──────────────────────────────────────────────────────

    #[test]
    fn test_no_infinite_loop_on_diamond_graph() {
        // Diamond: parent → child-a → leaf, parent → child-b → leaf
        let mut t = DependencyTracker::new();
        t.add_derived_dep(key("child-a"), key("parent"));
        t.add_derived_dep(key("child-b"), key("parent"));
        t.add_derived_dep(key("leaf"), key("child-a"));
        t.add_derived_dep(key("leaf"), key("child-b"));
        // Should not panic or loop
        let affected = t.invalidate_key(&key("parent"));
        assert!(affected.contains(&key("leaf")));
    }

    #[test]
    fn test_dependencies_of_returns_all_kinds() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("k"), "ep-a");
        t.add_derived_dep(key("k"), key("parent-k"));
        let deps = t.dependencies_of(&key("k"));
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn test_derived_from_multiple_children() {
        let mut t = DependencyTracker::new();
        t.add_derived_dep(key("c1"), key("p"));
        t.add_derived_dep(key("c2"), key("p"));
        t.add_derived_dep(key("c3"), key("p"));
        let children = t.derived_from(&key("p"));
        assert_eq!(children.len(), 3);
    }

    #[test]
    fn test_invalidate_endpoint_then_reinstate_dep() {
        let mut t = DependencyTracker::new();
        t.add_endpoint_dep(key("k1"), "ep-reuse");
        let _ = t.invalidate_endpoint("ep-reuse");
        // Re-add after invalidation
        t.add_endpoint_dep(key("k1"), "ep-reuse");
        let affected = t.invalidate_endpoint("ep-reuse");
        assert!(affected.contains(&key("k1")));
    }
}
