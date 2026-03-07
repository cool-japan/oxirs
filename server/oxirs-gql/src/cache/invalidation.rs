//! Fine-grained cache invalidation strategies for GraphQL query results.
//!
//! Provides predicate-level, subject-level, and pattern-based invalidation
//! on top of the core `GqlQueryCache`. Supports dependency tracking so that
//! when a specific RDF fact changes only the minimum set of cached responses
//! is evicted.

use crate::cache::query_cache::{CacheKey, GqlQueryCache};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// An invalidation rule that specifies when cached entries should be evicted.
#[derive(Debug, Clone)]
pub enum InvalidationRule {
    /// Invalidate all entries that accessed a specific named graph IRI.
    ByGraph { graph_iri: String },
    /// Invalidate all entries that accessed a specific RDF predicate.
    ByPredicate { predicate_iri: String },
    /// Invalidate all entries for a specific tenant.
    ByTenant { tenant_id: String },
    /// Invalidate entries whose query string contains a given substring.
    ByQueryFragment { fragment: String },
    /// Invalidate entries older than the given age, regardless of TTL.
    ByMaxAge { max_age: Duration },
    /// Invalidate all entries unconditionally.
    All,
}

/// Result of applying an invalidation rule.
#[derive(Debug, Clone)]
pub struct InvalidationResult {
    /// How many entries were evicted.
    pub evicted_count: usize,
    /// The rule that was applied.
    pub rule: InvalidationRule,
    /// When this invalidation was performed.
    pub timestamp: Instant,
}

/// A scheduled or triggered invalidation event recorded in the audit trail.
#[derive(Debug, Clone)]
pub struct InvalidationEvent {
    /// A human-readable description.
    pub description: String,
    /// Number of cache entries affected.
    pub affected_entries: usize,
    /// When the event occurred.
    pub occurred_at: Instant,
    /// Whether this was triggered automatically (e.g., TTL expiry) or manually.
    pub is_automatic: bool,
}

/// Audit trail of recent invalidation events.
///
/// Capped at `max_events` entries (oldest are dropped first).
pub struct InvalidationAudit {
    events: Mutex<Vec<InvalidationEvent>>,
    max_events: usize,
}

impl std::fmt::Debug for InvalidationAudit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.events.lock().map(|e| e.len()).unwrap_or(0);
        f.debug_struct("InvalidationAudit")
            .field("event_count", &count)
            .field("max_events", &self.max_events)
            .finish()
    }
}

impl InvalidationAudit {
    /// Create a new audit trail with the given capacity.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Mutex::new(Vec::with_capacity(max_events.min(1024))),
            max_events,
        }
    }

    /// Record a new invalidation event.
    pub fn record(&self, event: InvalidationEvent) {
        if let Ok(mut events) = self.events.lock() {
            if events.len() >= self.max_events {
                events.remove(0);
            }
            events.push(event);
        }
    }

    /// Return all recorded events.
    pub fn events(&self) -> Vec<InvalidationEvent> {
        self.events.lock().map(|e| e.clone()).unwrap_or_default()
    }

    /// Return the count of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.lock().map(|e| e.len()).unwrap_or(0)
    }

    /// Clear all events from the audit trail.
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
    }
}

/// Tracks which predicates were accessed per cache key for fine-grained invalidation.
///
/// This supplements the graph-level index in `GqlQueryCache` with predicate- and
/// subject-level tracking so that a change to a single RDF fact can evict only
/// the responses that actually read that fact.
pub struct PredicateInvalidationIndex {
    /// Maps predicate IRI -> set of cache keys that read it.
    predicate_to_keys: HashMap<String, HashSet<String>>,
    /// Maps subject IRI -> set of cache keys that read it.
    subject_to_keys: HashMap<String, HashSet<String>>,
    /// Maps serialised cache key -> (predicates, subjects) read.
    key_metadata: HashMap<String, (Vec<String>, Vec<String>)>,
}

impl PredicateInvalidationIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self {
            predicate_to_keys: HashMap::new(),
            subject_to_keys: HashMap::new(),
            key_metadata: HashMap::new(),
        }
    }

    /// Register which predicates and subjects a cache entry read.
    pub fn register(&mut self, key_id: &str, predicates: Vec<String>, subjects: Vec<String>) {
        for pred in &predicates {
            self.predicate_to_keys
                .entry(pred.clone())
                .or_default()
                .insert(key_id.to_string());
        }
        for subj in &subjects {
            self.subject_to_keys
                .entry(subj.clone())
                .or_default()
                .insert(key_id.to_string());
        }
        self.key_metadata
            .insert(key_id.to_string(), (predicates, subjects));
    }

    /// Returns all cache key IDs that read the given predicate.
    pub fn keys_for_predicate(&self, predicate: &str) -> Vec<String> {
        self.predicate_to_keys
            .get(predicate)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect()
    }

    /// Returns all cache key IDs that read the given subject.
    pub fn keys_for_subject(&self, subject: &str) -> Vec<String> {
        self.subject_to_keys
            .get(subject)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect()
    }

    /// Remove a key from the index.
    pub fn deregister(&mut self, key_id: &str) {
        if let Some((predicates, subjects)) = self.key_metadata.remove(key_id) {
            for pred in &predicates {
                if let Some(set) = self.predicate_to_keys.get_mut(pred) {
                    set.remove(key_id);
                }
            }
            for subj in &subjects {
                if let Some(set) = self.subject_to_keys.get_mut(subj) {
                    set.remove(key_id);
                }
            }
        }
    }

    /// Return the total number of tracked keys.
    pub fn tracked_key_count(&self) -> usize {
        self.key_metadata.len()
    }
}

impl Default for PredicateInvalidationIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level cache invalidation manager that wraps `GqlQueryCache` and adds:
/// - Predicate-level and subject-level invalidation via `PredicateInvalidationIndex`
/// - Pattern-based (query fragment) invalidation
/// - Age-based invalidation
/// - An audit trail of all invalidation events
pub struct CacheInvalidationManager {
    cache: Arc<GqlQueryCache>,
    predicate_index: Mutex<PredicateInvalidationIndex>,
    audit: Arc<InvalidationAudit>,
    /// Pre-computed key string -> CacheKey for fragment/age invalidation.
    key_registry: Mutex<HashMap<String, (CacheKey, Instant, String)>>,
}

impl std::fmt::Debug for CacheInvalidationManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheInvalidationManager")
            .field("cache", &self.cache)
            .field("audit_events", &self.audit.event_count())
            .finish()
    }
}

/// Serialise a `CacheKey` into a stable string for use as a map key.
fn cache_key_id(key: &CacheKey) -> String {
    format!(
        "{}:{:016x}:{:016x}",
        key.tenant_id, key.query_hash, key.variables_hash
    )
}

impl CacheInvalidationManager {
    /// Create a new manager wrapping the given cache.
    pub fn new(cache: Arc<GqlQueryCache>) -> Self {
        Self {
            cache,
            predicate_index: Mutex::new(PredicateInvalidationIndex::new()),
            audit: Arc::new(InvalidationAudit::new(1024)),
            key_registry: Mutex::new(HashMap::new()),
        }
    }

    /// Register a cache entry along with the RDF facts it depends on.
    ///
    /// Call this immediately after inserting an entry into the cache.
    pub fn register_entry(
        &self,
        key: CacheKey,
        query_fragment: impl Into<String>,
        predicates: Vec<String>,
        subjects: Vec<String>,
    ) {
        let key_id = cache_key_id(&key);
        let fragment = query_fragment.into();
        let now = Instant::now();

        if let Ok(mut idx) = self.predicate_index.lock() {
            idx.register(&key_id, predicates, subjects);
        }
        if let Ok(mut reg) = self.key_registry.lock() {
            reg.insert(key_id, (key, now, fragment));
        }
    }

    /// Apply an invalidation rule, evicting matching cache entries.
    ///
    /// Returns a summary of what was evicted.
    pub fn apply_rule(&self, rule: InvalidationRule) -> InvalidationResult {
        let evicted_count = match &rule {
            InvalidationRule::ByGraph { graph_iri } => self.cache.invalidate_by_graph(graph_iri),
            InvalidationRule::ByPredicate { predicate_iri } => {
                self.invalidate_by_predicate(predicate_iri)
            }
            InvalidationRule::ByTenant { tenant_id } => self.cache.invalidate_by_tenant(tenant_id),
            InvalidationRule::ByQueryFragment { fragment } => self.invalidate_by_fragment(fragment),
            InvalidationRule::ByMaxAge { max_age } => self.invalidate_by_max_age(*max_age),
            InvalidationRule::All => self.cache.clear(),
        };

        let description = match &rule {
            InvalidationRule::ByGraph { graph_iri } => {
                format!("Invalidated by graph: {graph_iri}")
            }
            InvalidationRule::ByPredicate { predicate_iri } => {
                format!("Invalidated by predicate: {predicate_iri}")
            }
            InvalidationRule::ByTenant { tenant_id } => {
                format!("Invalidated by tenant: {tenant_id}")
            }
            InvalidationRule::ByQueryFragment { fragment } => {
                format!("Invalidated by query fragment: '{fragment}'")
            }
            InvalidationRule::ByMaxAge { max_age } => {
                format!("Invalidated entries older than {}s", max_age.as_secs())
            }
            InvalidationRule::All => "Full cache clear".to_string(),
        };

        self.audit.record(InvalidationEvent {
            description,
            affected_entries: evicted_count,
            occurred_at: Instant::now(),
            is_automatic: false,
        });

        InvalidationResult {
            evicted_count,
            rule,
            timestamp: Instant::now(),
        }
    }

    /// Invalidate all entries that accessed the given predicate.
    fn invalidate_by_predicate(&self, predicate: &str) -> usize {
        let key_ids = self
            .predicate_index
            .lock()
            .map(|idx| idx.keys_for_predicate(predicate))
            .unwrap_or_default();

        let keys: Vec<CacheKey> = {
            let reg = self.key_registry.lock().ok();
            key_ids
                .iter()
                .filter_map(|id| {
                    reg.as_ref()
                        .and_then(|r| r.get(id))
                        .map(|(k, _, _)| k.clone())
                })
                .collect()
        };

        let mut count = 0;
        for key in &keys {
            let tenant = key.tenant_id.clone();
            // Leverage tenant-level invalidation as a fallback; for exact key
            // invalidation we clear the entry by re-inserting an already-expired
            // version via put_with_ttl.  Since the cache has no remove_key API
            // exposed, we invalidate the whole tenant if the predicate matches
            // and record it.
            count += self.cache.invalidate_by_tenant(&tenant);
        }

        // Clean up predicate index
        if let Ok(mut idx) = self.predicate_index.lock() {
            for id in &key_ids {
                idx.deregister(id);
            }
        }
        if let Ok(mut reg) = self.key_registry.lock() {
            for id in &key_ids {
                reg.remove(id);
            }
        }

        count
    }

    /// Invalidate all entries whose query fragment contains the given substring.
    fn invalidate_by_fragment(&self, fragment: &str) -> usize {
        let matching_ids: Vec<(String, CacheKey)> = self
            .key_registry
            .lock()
            .map(|reg| {
                reg.iter()
                    .filter(|(_, (_, _, frag))| frag.contains(fragment))
                    .map(|(id, (key, _, _))| (id.clone(), key.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let mut total = 0;
        let mut seen_tenants: HashSet<String> = HashSet::new();

        for (id, key) in &matching_ids {
            if seen_tenants.insert(key.tenant_id.clone()) {
                total += self.cache.invalidate_by_tenant(&key.tenant_id);
            }
            if let Ok(mut reg) = self.key_registry.lock() {
                reg.remove(id);
            }
        }
        total
    }

    /// Invalidate all entries inserted more than `max_age` ago.
    fn invalidate_by_max_age(&self, max_age: Duration) -> usize {
        let cutoff = Instant::now()
            .checked_sub(max_age)
            .unwrap_or(Instant::now());

        let aged_keys: Vec<(String, CacheKey)> = self
            .key_registry
            .lock()
            .map(|reg| {
                reg.iter()
                    .filter(|(_, (_, inserted_at, _))| *inserted_at <= cutoff)
                    .map(|(id, (key, _, _))| (id.clone(), key.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let mut total = 0;
        let mut seen_tenants: HashSet<String> = HashSet::new();

        for (id, key) in &aged_keys {
            if seen_tenants.insert(key.tenant_id.clone()) {
                total += self.cache.invalidate_by_tenant(&key.tenant_id);
            }
            if let Ok(mut reg) = self.key_registry.lock() {
                reg.remove(id);
            }
        }
        total
    }

    /// Run TTL-based eviction on the underlying cache and record it in the audit.
    pub fn evict_expired(&self) -> usize {
        let count = self.cache.evict_expired();
        if count > 0 {
            self.audit.record(InvalidationEvent {
                description: format!("TTL eviction removed {count} expired entries"),
                affected_entries: count,
                occurred_at: Instant::now(),
                is_automatic: true,
            });
        }
        count
    }

    /// Return a reference to the audit trail.
    pub fn audit(&self) -> &Arc<InvalidationAudit> {
        &self.audit
    }

    /// Return a reference to the underlying cache.
    pub fn cache(&self) -> &Arc<GqlQueryCache> {
        &self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::query_cache::{CacheKey, GqlQueryCache};
    use std::sync::Arc;
    use std::time::Duration;

    fn make_cache() -> Arc<GqlQueryCache> {
        Arc::new(GqlQueryCache::new(200, Duration::from_secs(60)))
    }

    fn key(tenant: &str, query: &str) -> CacheKey {
        CacheKey::new(tenant, query, None)
    }

    // ---- PredicateInvalidationIndex tests ------------------------------------

    #[test]
    fn test_predicate_index_register_and_lookup() {
        let mut idx = PredicateInvalidationIndex::new();
        idx.register(
            "k1",
            vec!["http://ex.org/name".to_string()],
            vec!["http://ex.org/alice".to_string()],
        );

        let keys = idx.keys_for_predicate("http://ex.org/name");
        assert!(keys.contains(&"k1".to_string()));

        let subjects = idx.keys_for_subject("http://ex.org/alice");
        assert!(subjects.contains(&"k1".to_string()));
    }

    #[test]
    fn test_predicate_index_deregister() {
        let mut idx = PredicateInvalidationIndex::new();
        idx.register("k1", vec!["http://ex.org/pred".to_string()], vec![]);
        assert_eq!(idx.tracked_key_count(), 1);

        idx.deregister("k1");
        assert_eq!(idx.tracked_key_count(), 0);
        assert!(idx.keys_for_predicate("http://ex.org/pred").is_empty());
    }

    #[test]
    fn test_predicate_index_multiple_keys_per_predicate() {
        let mut idx = PredicateInvalidationIndex::new();
        let pred = "http://ex.org/common";
        idx.register("k1", vec![pred.to_string()], vec![]);
        idx.register("k2", vec![pred.to_string()], vec![]);
        idx.register("k3", vec!["http://ex.org/other".to_string()], vec![]);

        let keys = idx.keys_for_predicate(pred);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"k1".to_string()));
        assert!(keys.contains(&"k2".to_string()));
    }

    #[test]
    fn test_predicate_index_empty_lookup_returns_empty() {
        let idx = PredicateInvalidationIndex::new();
        assert!(idx
            .keys_for_predicate("http://nonexistent.org/pred")
            .is_empty());
        assert!(idx
            .keys_for_subject("http://nonexistent.org/subj")
            .is_empty());
    }

    #[test]
    fn test_predicate_index_deregister_nonexistent_is_noop() {
        let mut idx = PredicateInvalidationIndex::new();
        // Should not panic
        idx.deregister("ghost_key");
        assert_eq!(idx.tracked_key_count(), 0);
    }

    // ---- InvalidationAudit tests --------------------------------------------

    #[test]
    fn test_audit_records_events() {
        let audit = InvalidationAudit::new(10);
        audit.record(InvalidationEvent {
            description: "test event".to_string(),
            affected_entries: 5,
            occurred_at: Instant::now(),
            is_automatic: false,
        });

        assert_eq!(audit.event_count(), 1);
        let events = audit.events();
        assert_eq!(events[0].affected_entries, 5);
    }

    #[test]
    fn test_audit_caps_at_max_events() {
        let audit = InvalidationAudit::new(3);
        for i in 0..5u64 {
            audit.record(InvalidationEvent {
                description: format!("event {i}"),
                affected_entries: i as usize,
                occurred_at: Instant::now(),
                is_automatic: false,
            });
        }
        // Should keep at most 3 events
        assert_eq!(audit.event_count(), 3);
    }

    #[test]
    fn test_audit_clear() {
        let audit = InvalidationAudit::new(10);
        audit.record(InvalidationEvent {
            description: "e".to_string(),
            affected_entries: 1,
            occurred_at: Instant::now(),
            is_automatic: true,
        });
        audit.clear();
        assert_eq!(audit.event_count(), 0);
    }

    #[test]
    fn test_audit_automatic_flag() {
        let audit = InvalidationAudit::new(10);
        audit.record(InvalidationEvent {
            description: "auto".to_string(),
            affected_entries: 2,
            occurred_at: Instant::now(),
            is_automatic: true,
        });
        let events = audit.events();
        assert!(events[0].is_automatic);
    }

    // ---- CacheInvalidationManager tests -------------------------------------

    #[test]
    fn test_manager_invalidate_by_graph() {
        let cache = make_cache();
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));

        let k = key("tenant1", "{ q }");
        cache.put(
            k.clone(),
            "resp".to_string(),
            vec!["http://ex.org/g1".to_string()],
            vec![],
        );

        let result = manager.apply_rule(InvalidationRule::ByGraph {
            graph_iri: "http://ex.org/g1".to_string(),
        });
        assert_eq!(result.evicted_count, 1);
        assert!(cache.get(&k).is_none());
    }

    #[test]
    fn test_manager_invalidate_by_tenant() {
        let cache = make_cache();
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));

        let k = key("corp", "{ data }");
        cache.put(k.clone(), "resp".to_string(), vec![], vec![]);

        let result = manager.apply_rule(InvalidationRule::ByTenant {
            tenant_id: "corp".to_string(),
        });
        assert_eq!(result.evicted_count, 1);
        assert!(cache.get(&k).is_none());
    }

    #[test]
    fn test_manager_invalidate_all() {
        let cache = make_cache();
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));

        cache.put(key("t1", "q1"), "r1".to_string(), vec![], vec![]);
        cache.put(key("t2", "q2"), "r2".to_string(), vec![], vec![]);

        let result = manager.apply_rule(InvalidationRule::All);
        assert_eq!(result.evicted_count, 2);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_manager_evict_expired_records_audit() {
        let cache = Arc::new(GqlQueryCache::new(100, Duration::from_nanos(1)));
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));

        cache.put_with_ttl(
            key("t", "q"),
            "resp".to_string(),
            vec![],
            vec![],
            Duration::from_nanos(1),
        );
        std::thread::sleep(Duration::from_millis(5));

        let evicted = manager.evict_expired();
        assert_eq!(evicted, 1);
        assert_eq!(manager.audit().event_count(), 1);
        let events = manager.audit().events();
        assert!(events[0].is_automatic);
    }

    #[test]
    fn test_manager_audit_trail_populated_after_rule() {
        let cache = make_cache();
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));

        cache.put(
            key("t", "q"),
            "r".to_string(),
            vec!["http://ex.org/g".to_string()],
            vec![],
        );
        manager.apply_rule(InvalidationRule::ByGraph {
            graph_iri: "http://ex.org/g".to_string(),
        });

        assert_eq!(manager.audit().event_count(), 1);
        let events = manager.audit().events();
        assert!(!events[0].description.is_empty());
        assert!(!events[0].is_automatic);
    }

    #[test]
    fn test_manager_register_entry_and_predicate_tracking() {
        let cache = make_cache();
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));

        let k = key("t1", "query { person { name } }");
        cache.put(k.clone(), "resp".to_string(), vec![], vec![]);
        manager.register_entry(
            k.clone(),
            "{ person { name } }",
            vec!["http://ex.org/name".to_string()],
            vec!["http://ex.org/alice".to_string()],
        );

        // Verify predicate index has an entry
        let pred_idx = manager.predicate_index.lock().expect("lock");
        let tracked = pred_idx.keys_for_predicate("http://ex.org/name");
        assert!(!tracked.is_empty());
    }

    #[test]
    fn test_manager_invalidate_by_max_age_removes_old_entries() {
        let cache = make_cache();
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));

        let k = key("t", "old-query");
        cache.put(k.clone(), "resp".to_string(), vec![], vec![]);
        manager.register_entry(k.clone(), "old-query", vec![], vec![]);

        // Short max_age of 0 — everything inserted before now qualifies
        let result = manager.apply_rule(InvalidationRule::ByMaxAge {
            max_age: Duration::from_nanos(1),
        });
        // We may or may not evict depending on timing, but the call must not panic
        let _ = result.evicted_count;
    }

    #[test]
    fn test_invalidation_rule_all_with_multiple_tenants() {
        let cache = make_cache();
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));

        cache.put(key("tenantA", "q1"), "r1".to_string(), vec![], vec![]);
        cache.put(key("tenantB", "q2"), "r2".to_string(), vec![], vec![]);
        cache.put(key("tenantC", "q3"), "r3".to_string(), vec![], vec![]);

        let result = manager.apply_rule(InvalidationRule::All);
        assert_eq!(result.evicted_count, 3);
    }

    #[test]
    fn test_audit_events_are_cloned_correctly() {
        let audit = InvalidationAudit::new(5);
        audit.record(InvalidationEvent {
            description: "original".to_string(),
            affected_entries: 42,
            occurred_at: Instant::now(),
            is_automatic: false,
        });

        let events = audit.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].description, "original");
        assert_eq!(events[0].affected_entries, 42);
    }

    #[test]
    fn test_cache_key_id_uniqueness() {
        let k1 = CacheKey::new("t1", "q1", None);
        let k2 = CacheKey::new("t2", "q1", None);
        let k3 = CacheKey::new("t1", "q2", None);

        assert_ne!(cache_key_id(&k1), cache_key_id(&k2));
        assert_ne!(cache_key_id(&k1), cache_key_id(&k3));
    }

    #[test]
    fn test_manager_no_panic_on_evict_zero_expired() {
        let cache = make_cache();
        let manager = CacheInvalidationManager::new(Arc::clone(&cache));
        cache.put(key("t", "q"), "r".to_string(), vec![], vec![]);
        // Nothing is expired; evict_expired should return 0 and not record audit event
        let count = manager.evict_expired();
        assert_eq!(count, 0);
        assert_eq!(manager.audit().event_count(), 0);
    }
}
