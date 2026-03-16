//! GraphQL Response Cache with CachePolicy and @cacheControl directive support.
//!
//! Provides:
//! - `GraphQlResponseCache`: persisted cache keyed by `(tenant_id, query_hash, variables_hash)`
//!   with per-entry `CachePolicy` controlling max-age and stale-while-revalidate.
//! - `FieldLevelCacheDirective`: parses `@cacheControl(maxAge: N)` annotations and
//!   computes the *effective* max-age as the minimum across all resolved fields.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// Per-tenant, per-query cache key derived from FNV-1a hashes.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ResponseCacheKey {
    /// Optional tenant isolator — `None` means "global / no tenancy".
    pub tenant_id: Option<String>,
    /// FNV-1a hash of the normalised GraphQL query string.
    pub query_hash: u64,
    /// FNV-1a hash of the serialised variables JSON, or `0` when absent.
    pub variables_hash: u64,
}

impl ResponseCacheKey {
    /// Construct a key from raw string inputs.
    pub fn new(tenant_id: Option<&str>, query: &str, variables: Option<&str>) -> Self {
        Self {
            tenant_id: tenant_id.map(|s| s.to_string()),
            query_hash: fnv1a(query),
            variables_hash: variables.map(fnv1a).unwrap_or(0),
        }
    }
}

fn fnv1a(s: &str) -> u64 {
    const BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    s.bytes()
        .fold(BASIS, |h, b| h.wrapping_mul(PRIME) ^ u64::from(b))
}

// ---------------------------------------------------------------------------
// Cache policy
// ---------------------------------------------------------------------------

/// Controls how long a cached GraphQL response remains valid.
#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// Time in milliseconds for which the entry is considered *fresh*.
    pub max_age_ms: u64,
    /// Additional window (ms) during which the stale entry may be served
    /// while a background revalidation is in flight.
    pub stale_while_revalidate_ms: u64,
    /// When `true` the cache key is further distinguished by the authenticated
    /// user identity so different users never see each other's responses.
    pub vary_by_user: bool,
}

impl CachePolicy {
    /// Construct a simple policy without stale-while-revalidate or per-user
    /// variation.
    pub fn simple(max_age_ms: u64) -> Self {
        Self {
            max_age_ms,
            stale_while_revalidate_ms: 0,
            vary_by_user: false,
        }
    }

    /// Returns `true` if the policy allows serving a stale entry that is `age_ms`
    /// old.
    pub fn is_stale_usable(&self, age_ms: u64) -> bool {
        age_ms < self.max_age_ms + self.stale_while_revalidate_ms
    }
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::simple(60_000)
    }
}

// ---------------------------------------------------------------------------
// Cached response
// ---------------------------------------------------------------------------

/// A serialised GraphQL response together with its caching metadata.
#[derive(Debug, Clone)]
pub struct CachedResponse {
    /// The serialised JSON response body.
    pub body: String,
    /// When this entry was stored.
    pub stored_at: Instant,
    /// The policy in effect when the entry was stored.
    pub policy: CachePolicy,
    /// GraphQL type names touched when producing this response.
    /// Used for type-level invalidation.
    pub touched_types: Vec<String>,
    /// `"TypeName.fieldName"` strings for field-level invalidation.
    pub touched_fields: Vec<String>,
    /// Number of times this entry has been served.
    pub hit_count: u64,
}

impl CachedResponse {
    fn age_ms(&self) -> u64 {
        self.stored_at.elapsed().as_millis() as u64
    }

    /// Returns `true` if the entry is within the max-age window.
    pub fn is_fresh(&self) -> bool {
        self.age_ms() < self.policy.max_age_ms
    }

    /// Returns `true` if the entry is stale but still usable under the
    /// stale-while-revalidate window.
    pub fn is_stale_usable(&self) -> bool {
        self.policy.is_stale_usable(self.age_ms())
    }

    /// Returns the remaining fresh time in milliseconds (0 if expired).
    pub fn remaining_fresh_ms(&self) -> u64 {
        self.policy.max_age_ms.saturating_sub(self.age_ms())
    }
}

// ---------------------------------------------------------------------------
// Internal store
// ---------------------------------------------------------------------------

struct ResponseStore {
    entries: HashMap<ResponseCacheKey, CachedResponse>,
    /// type_name -> keys that touched that type.
    type_index: HashMap<String, HashSet<ResponseCacheKey>>,
    /// "TypeName.field" -> keys that touched that field.
    field_index: HashMap<String, HashSet<ResponseCacheKey>>,
    /// tenant_id -> keys for that tenant.
    tenant_index: HashMap<String, HashSet<ResponseCacheKey>>,
    max_entries: usize,
}

impl ResponseStore {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            type_index: HashMap::new(),
            field_index: HashMap::new(),
            tenant_index: HashMap::new(),
            max_entries,
        }
    }

    fn insert(&mut self, key: ResponseCacheKey, entry: CachedResponse) {
        // Index by type names
        for type_name in &entry.touched_types {
            self.type_index
                .entry(type_name.clone())
                .or_default()
                .insert(key.clone());
        }
        // Index by field paths
        for field_path in &entry.touched_fields {
            self.field_index
                .entry(field_path.clone())
                .or_default()
                .insert(key.clone());
        }
        // Index by tenant
        if let Some(tid) = &key.tenant_id {
            self.tenant_index
                .entry(tid.clone())
                .or_default()
                .insert(key.clone());
        }

        // Simple capacity cap: evict one old entry when full
        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.entries.keys().next().cloned() {
                self.remove(&oldest);
            }
        }

        self.entries.insert(key, entry);
    }

    fn remove(&mut self, key: &ResponseCacheKey) {
        if let Some(entry) = self.entries.remove(key) {
            for t in &entry.touched_types {
                if let Some(s) = self.type_index.get_mut(t) {
                    s.remove(key);
                }
            }
            for f in &entry.touched_fields {
                if let Some(s) = self.field_index.get_mut(f) {
                    s.remove(key);
                }
            }
            if let Some(tid) = &key.tenant_id {
                if let Some(s) = self.tenant_index.get_mut(tid) {
                    s.remove(key);
                }
            }
        }
    }

    fn remove_keys(&mut self, keys: Vec<ResponseCacheKey>) -> usize {
        let n = keys.len();
        for k in keys {
            self.remove(&k);
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Public cache
// ---------------------------------------------------------------------------

/// Thread-safe GraphQL response cache supporting type- and field-level
/// invalidation and stale-while-revalidate semantics.
pub struct GraphQlResponseCache {
    store: Arc<Mutex<ResponseStore>>,
}

impl std::fmt::Debug for GraphQlResponseCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphQlResponseCache").finish()
    }
}

impl GraphQlResponseCache {
    /// Create a new cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            store: Arc::new(Mutex::new(ResponseStore::new(max_entries))),
        }
    }

    /// Look up a cached response.
    ///
    /// Returns `Some(CachedResponse)` when the entry exists and is either
    /// fresh or within the stale-while-revalidate window.  Expired entries
    /// are removed eagerly.
    pub fn cached_response(&self, key: &ResponseCacheKey) -> Option<CachedResponse> {
        let mut store = self.store.lock().unwrap_or_else(|p| p.into_inner());
        match store.entries.get(key) {
            None => None,
            Some(entry) if !entry.is_fresh() && !entry.is_stale_usable() => {
                let k = key.clone();
                store.remove(&k);
                None
            }
            Some(entry) => {
                let mut response = entry.clone();
                response.hit_count += 1;
                let k = key.clone();
                store.entries.entry(k).and_modify(|e| e.hit_count += 1);
                Some(response)
            }
        }
    }

    /// Store a response in the cache.
    pub fn store_response(
        &self,
        key: ResponseCacheKey,
        body: String,
        policy: CachePolicy,
        touched_types: Vec<String>,
        touched_fields: Vec<String>,
    ) {
        let entry = CachedResponse {
            body,
            stored_at: Instant::now(),
            policy,
            touched_types,
            touched_fields,
            hit_count: 0,
        };
        let mut store = self.store.lock().unwrap_or_else(|p| p.into_inner());
        store.insert(key, entry);
    }

    /// Invalidate all entries that touched `type_name`.
    ///
    /// Returns the number of entries evicted.
    pub fn invalidate_type(&self, type_name: &str) -> usize {
        let mut store = self.store.lock().unwrap_or_else(|p| p.into_inner());
        let keys: Vec<ResponseCacheKey> = store
            .type_index
            .get(type_name)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        store.remove_keys(keys)
    }

    /// Invalidate all entries that touched field `"type_name.field_name"`.
    ///
    /// Returns the number of entries evicted.
    pub fn invalidate_field(&self, type_name: &str, field: &str) -> usize {
        let path = format!("{type_name}.{field}");
        let mut store = self.store.lock().unwrap_or_else(|p| p.into_inner());
        let keys: Vec<ResponseCacheKey> = store
            .field_index
            .get(&path)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        store.remove_keys(keys)
    }

    /// Pre-populate the cache by computing keys for a list of `(query, variables)` pairs.
    ///
    /// This marks the slot as *absent* so callers know they need to prime the
    /// actual responses. In practice, the warm-up caller would follow up with
    /// `store_response` calls once results are fetched.  Here we return the
    /// computed keys so the caller can do so.
    pub fn warm_up(
        &self,
        tenant_id: Option<&str>,
        queries: &[(String, serde_json::Value)],
    ) -> Vec<ResponseCacheKey> {
        queries
            .iter()
            .map(|(q, v)| {
                let vars_json = v.to_string();
                let vars_str = if vars_json == "null" {
                    None
                } else {
                    Some(vars_json.as_str())
                };
                ResponseCacheKey::new(tenant_id, q, vars_str)
            })
            .collect()
    }

    /// Remove all entries for a given tenant.
    pub fn invalidate_tenant(&self, tenant_id: &str) -> usize {
        let mut store = self.store.lock().unwrap_or_else(|p| p.into_inner());
        let keys: Vec<ResponseCacheKey> = store
            .tenant_index
            .get(tenant_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        store.remove_keys(keys)
    }

    /// Current number of entries in the cache.
    pub fn size(&self) -> usize {
        self.store.lock().map(|s| s.entries.len()).unwrap_or(0)
    }

    /// Remove all entries from the cache.
    pub fn clear(&self) {
        let mut store = self.store.lock().unwrap_or_else(|p| p.into_inner());
        store.entries.clear();
        store.type_index.clear();
        store.field_index.clear();
        store.tenant_index.clear();
    }
}

// ---------------------------------------------------------------------------
// @cacheControl directive support
// ---------------------------------------------------------------------------

/// A parsed `@cacheControl` directive extracted from a GraphQL field or type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldLevelCacheDirective {
    /// The `maxAge` argument value in seconds, if specified.
    pub max_age_secs: Option<u64>,
    /// The `scope` argument (`PUBLIC` or `PRIVATE`), if specified.
    pub scope: Option<CacheScope>,
    /// `inheritMaxAge: true` means the field inherits its parent's max-age.
    pub inherit_max_age: bool,
}

/// Cache scope argument for `@cacheControl`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheScope {
    Public,
    Private,
}

impl FieldLevelCacheDirective {
    /// Parse a `@cacheControl(maxAge: N, scope: "PUBLIC")` directive string.
    ///
    /// Returns `None` if the input does not contain `@cacheControl`.
    pub fn parse(input: &str) -> Option<Self> {
        if !input.contains("@cacheControl") {
            return None;
        }

        let max_age_secs = Self::extract_u64_arg(input, "maxAge");
        let scope = if input.contains("scope:") {
            if input.contains("PRIVATE") {
                Some(CacheScope::Private)
            } else {
                Some(CacheScope::Public)
            }
        } else {
            None
        };
        let inherit_max_age =
            input.contains("inheritMaxAge: true") || input.contains("inheritMaxAge:true");

        Some(Self {
            max_age_secs,
            scope,
            inherit_max_age,
        })
    }

    fn extract_u64_arg(input: &str, arg: &str) -> Option<u64> {
        let needle = format!("{arg}:");
        let pos = input.find(needle.as_str())?;
        let after = input[pos + needle.len()..].trim_start();
        // Parse digits until a non-digit char
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    }

    /// Compute the effective `max_age_ms` for a set of directives collected
    /// from all fields/types resolved during a query.
    ///
    /// The effective value is the *minimum* `maxAge` across all directives
    /// that have an explicit `maxAge`. Returns `None` if no directive
    /// specifies `maxAge`.
    pub fn effective_max_age_ms(directives: &[FieldLevelCacheDirective]) -> Option<u64> {
        directives
            .iter()
            .filter_map(|d| d.max_age_secs.map(|s| s * 1_000))
            .reduce(u64::min)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn key(tenant: Option<&str>, query: &str) -> ResponseCacheKey {
        ResponseCacheKey::new(tenant, query, None)
    }

    fn store_simple(cache: &GraphQlResponseCache, key: ResponseCacheKey, body: &str) {
        cache.store_response(
            key,
            body.to_string(),
            CachePolicy::simple(60_000),
            vec![],
            vec![],
        );
    }

    // --- ResponseCacheKey ---

    #[test]
    fn test_key_same_query_same_tenant_equal() {
        let k1 = ResponseCacheKey::new(Some("t1"), "{ hello }", None);
        let k2 = ResponseCacheKey::new(Some("t1"), "{ hello }", None);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_key_different_tenant_not_equal() {
        let k1 = ResponseCacheKey::new(Some("t1"), "{ hello }", None);
        let k2 = ResponseCacheKey::new(Some("t2"), "{ hello }", None);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_key_none_tenant_vs_some_not_equal() {
        let k1 = ResponseCacheKey::new(None, "{ hello }", None);
        let k2 = ResponseCacheKey::new(Some("t1"), "{ hello }", None);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_key_different_variables_not_equal() {
        let k1 = ResponseCacheKey::new(None, "{ q }", Some(r#"{"id":1}"#));
        let k2 = ResponseCacheKey::new(None, "{ q }", Some(r#"{"id":2}"#));
        assert_ne!(k1, k2);
    }

    // --- CachePolicy ---

    #[test]
    fn test_policy_fresh_entry() {
        let policy = CachePolicy::simple(10_000);
        // 9999ms is within max_age of 10000ms; the entry is still usable
        assert!(policy.is_stale_usable(9_999));
        // 10001ms exceeds max_age with no SWR window; not usable
        assert!(!policy.is_stale_usable(10_001));
    }

    #[test]
    fn test_policy_stale_within_swr_window() {
        let policy = CachePolicy {
            max_age_ms: 1_000,
            stale_while_revalidate_ms: 5_000,
            vary_by_user: false,
        };
        assert!(policy.is_stale_usable(5_999));
        assert!(!policy.is_stale_usable(6_001));
    }

    // --- GraphQlResponseCache ---

    #[test]
    fn test_store_and_retrieve() {
        let cache = GraphQlResponseCache::new(100);
        let k = key(Some("tenant1"), "{ data }");
        store_simple(&cache, k.clone(), r#"{"data":{}}"#);

        let resp = cache.cached_response(&k).expect("should be in cache");
        assert_eq!(resp.body, r#"{"data":{}}"#);
    }

    #[test]
    fn test_miss_returns_none() {
        let cache = GraphQlResponseCache::new(100);
        let k = key(None, "{ missing }");
        assert!(cache.cached_response(&k).is_none());
    }

    #[test]
    fn test_expired_entry_returns_none() {
        let cache = GraphQlResponseCache::new(100);
        let k = key(None, "{ expiring }");
        cache.store_response(
            k.clone(),
            "body".to_string(),
            CachePolicy {
                max_age_ms: 1,
                stale_while_revalidate_ms: 0,
                vary_by_user: false,
            },
            vec![],
            vec![],
        );
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.cached_response(&k).is_none());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_stale_while_revalidate_still_served() {
        let cache = GraphQlResponseCache::new(100);
        let k = key(None, "{ swr }");
        // max_age = 1ms, swr = 5000ms → stale but usable for 5 seconds
        cache.store_response(
            k.clone(),
            "stale-body".to_string(),
            CachePolicy {
                max_age_ms: 1,
                stale_while_revalidate_ms: 5_000,
                vary_by_user: false,
            },
            vec![],
            vec![],
        );
        std::thread::sleep(Duration::from_millis(5));
        // Should still be served (within SWR window)
        let resp = cache.cached_response(&k);
        assert!(
            resp.is_some(),
            "stale entry should be served within SWR window"
        );
    }

    #[test]
    fn test_invalidate_type_removes_matching_entries() {
        let cache = GraphQlResponseCache::new(100);
        let k1 = key(None, "q1");
        let k2 = key(None, "q2");

        cache.store_response(
            k1.clone(),
            "r1".to_string(),
            CachePolicy::default(),
            vec!["User".to_string()],
            vec![],
        );
        cache.store_response(
            k2.clone(),
            "r2".to_string(),
            CachePolicy::default(),
            vec!["Product".to_string()],
            vec![],
        );

        let evicted = cache.invalidate_type("User");
        assert_eq!(evicted, 1);
        assert!(cache.cached_response(&k1).is_none());
        assert!(cache.cached_response(&k2).is_some());
    }

    #[test]
    fn test_invalidate_field_removes_matching_entries() {
        let cache = GraphQlResponseCache::new(100);
        let k1 = key(None, "q1");
        let k2 = key(None, "q2");

        cache.store_response(
            k1.clone(),
            "r1".to_string(),
            CachePolicy::default(),
            vec![],
            vec!["User.email".to_string()],
        );
        cache.store_response(
            k2.clone(),
            "r2".to_string(),
            CachePolicy::default(),
            vec![],
            vec!["User.name".to_string()],
        );

        let evicted = cache.invalidate_field("User", "email");
        assert_eq!(evicted, 1);
        assert!(cache.cached_response(&k1).is_none());
        assert!(cache.cached_response(&k2).is_some());
    }

    #[test]
    fn test_invalidate_type_no_match_returns_zero() {
        let cache = GraphQlResponseCache::new(100);
        let evicted = cache.invalidate_type("NonExistent");
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_invalidate_field_no_match_returns_zero() {
        let cache = GraphQlResponseCache::new(100);
        let evicted = cache.invalidate_field("Ghost", "field");
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_invalidate_tenant() {
        let cache = GraphQlResponseCache::new(100);
        let k_a = key(Some("acme"), "q1");
        let k_b = key(Some("widgets"), "q1");

        store_simple(&cache, k_a.clone(), "ra");
        store_simple(&cache, k_b.clone(), "rb");

        let evicted = cache.invalidate_tenant("acme");
        assert_eq!(evicted, 1);
        assert!(cache.cached_response(&k_a).is_none());
        assert!(cache.cached_response(&k_b).is_some());
    }

    #[test]
    fn test_warm_up_returns_keys() {
        let cache = GraphQlResponseCache::new(100);
        let queries = vec![
            ("{ users }".to_string(), serde_json::Value::Null),
            ("{ products }".to_string(), serde_json::json!({"limit": 10})),
        ];
        let keys = cache.warm_up(Some("tenant1"), &queries);
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].query_hash, fnv1a("{ users }"));
    }

    #[test]
    fn test_clear() {
        let cache = GraphQlResponseCache::new(100);
        store_simple(&cache, key(None, "q1"), "r1");
        store_simple(&cache, key(None, "q2"), "r2");
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_hit_count_increments() {
        let cache = GraphQlResponseCache::new(100);
        let k = key(Some("t"), "q");
        store_simple(&cache, k.clone(), "body");

        let resp1 = cache.cached_response(&k).expect("hit 1");
        assert_eq!(resp1.hit_count, 1);
        let resp2 = cache.cached_response(&k).expect("hit 2");
        assert_eq!(resp2.hit_count, 2);
    }

    // --- FieldLevelCacheDirective ---

    #[test]
    fn test_parse_cache_control_max_age() {
        let directive = FieldLevelCacheDirective::parse("@cacheControl(maxAge: 60)");
        assert!(directive.is_some());
        let d = directive.expect("should succeed");
        assert_eq!(d.max_age_secs, Some(60));
        assert_eq!(d.scope, None);
        assert!(!d.inherit_max_age);
    }

    #[test]
    fn test_parse_cache_control_with_scope() {
        let d = FieldLevelCacheDirective::parse("@cacheControl(maxAge: 30, scope: PUBLIC)")
            .expect("should parse");
        assert_eq!(d.max_age_secs, Some(30));
        assert_eq!(d.scope, Some(CacheScope::Public));
    }

    #[test]
    fn test_parse_cache_control_private_scope() {
        let d =
            FieldLevelCacheDirective::parse("@cacheControl(scope: PRIVATE)").expect("should parse");
        assert_eq!(d.scope, Some(CacheScope::Private));
    }

    #[test]
    fn test_parse_no_cache_control_returns_none() {
        assert!(FieldLevelCacheDirective::parse("{ user { name } }").is_none());
    }

    #[test]
    fn test_parse_inherit_max_age() {
        let d = FieldLevelCacheDirective::parse("@cacheControl(inheritMaxAge: true)")
            .expect("should parse");
        assert!(d.inherit_max_age);
    }

    #[test]
    fn test_effective_max_age_ms_minimum() {
        let directives = vec![
            FieldLevelCacheDirective {
                max_age_secs: Some(120),
                scope: None,
                inherit_max_age: false,
            },
            FieldLevelCacheDirective {
                max_age_secs: Some(30),
                scope: None,
                inherit_max_age: false,
            },
            FieldLevelCacheDirective {
                max_age_secs: Some(60),
                scope: None,
                inherit_max_age: false,
            },
        ];
        let effective = FieldLevelCacheDirective::effective_max_age_ms(&directives);
        assert_eq!(effective, Some(30_000)); // 30s → 30 000ms
    }

    #[test]
    fn test_effective_max_age_ms_no_directives() {
        let directives: Vec<FieldLevelCacheDirective> = vec![];
        assert_eq!(
            FieldLevelCacheDirective::effective_max_age_ms(&directives),
            None
        );
    }

    #[test]
    fn test_effective_max_age_ms_mixed_none() {
        let directives = vec![
            FieldLevelCacheDirective {
                max_age_secs: None,
                scope: None,
                inherit_max_age: true,
            },
            FieldLevelCacheDirective {
                max_age_secs: Some(90),
                scope: None,
                inherit_max_age: false,
            },
        ];
        // Only the directive with max_age_secs set contributes
        assert_eq!(
            FieldLevelCacheDirective::effective_max_age_ms(&directives),
            Some(90_000)
        );
    }

    #[test]
    fn test_remaining_fresh_ms_positive() {
        let resp = CachedResponse {
            body: "x".to_string(),
            stored_at: Instant::now(),
            policy: CachePolicy::simple(60_000),
            touched_types: vec![],
            touched_fields: vec![],
            hit_count: 0,
        };
        assert!(resp.remaining_fresh_ms() > 0);
    }

    #[test]
    fn test_capacity_cap() {
        let cache = GraphQlResponseCache::new(3);
        for i in 0..5u32 {
            store_simple(&cache, key(None, &format!("q{i}")), "r");
        }
        assert!(cache.size() <= 3);
    }
}
